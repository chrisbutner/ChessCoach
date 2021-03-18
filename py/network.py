# --- Require silent TensorFlow initialization when running as UCI ---

import os
import socket

silent = bool(os.environ.get("CHESSCOACH_SILENT"))
if silent:
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Work around https://github.com/tensorflow/tensorflow/issues/45994
import sys
if not hasattr(sys, "argv") or not sys.argv:
  sys.argv = [""]

def log(*args):
  if not silent:
    print(*args)

import tensorflow as tf

# --- TPU/GPU initialization ---

class LocalTPUClusterResolver(
    tf.distribute.cluster_resolver.TPUClusterResolver):
  """LocalTPUClusterResolver."""

  def __init__(self):
    self._tpu = ""
    self.task_type = "worker"
    self.task_id = 0

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    return None

  def cluster_spec(self):
    return tf.train.ClusterSpec({})

  def get_tpu_system_metadata(self):
    return tf.tpu.experimental.TPUSystemMetadata(
        num_cores=8,
        num_hosts=1,
        num_of_cores_per_host=8,
        topology=None,
        devices=tf.config.list_logical_devices())

  def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
    return {"TPU": 8}

tpu_strategy = None
try:
  # Alpha TPU VMs
  resolver = LocalTPUClusterResolver()
  tf.tpu.experimental.initialize_tpu_system(resolver)
  tpu_strategy = tf.distribute.TPUStrategy(resolver)
except:
  try:
    # Separated TPUs
    # Passing zone-qualified cluster-injected TPU_NAME to the resolver fails, need to leave it blank.
    tpu_name = os.environ.get("TPU_NAME") or socket.gethostname()
    if "/" in tpu_name:
      tpu_name = None
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    tpu_strategy = tf.distribute.TPUStrategy(resolver)
  except:
    pass

tpus = tf.config.experimental.list_logical_devices("TPU")
gpus = tf.config.experimental.list_logical_devices("GPU")
devices = tpus + gpus
thread_ident_to_index = {}

# --- Now it's safe for further imports ---

import math
import re
import time
import threading
import numpy as np

from config import Config, PredictionStatus
from model import ModelBuilder
from training import Trainer, StudentModel
from dataset import DatasetBuilder

# --- Network ---

class PredictionModels:
  def __init__(self):
    self.full = None
    self.full_weights_path = None
    self.full_weights_last_check = None
    self.predict = None
    self.tokenizer = None
    self.commentary = None

class TrainingModels:
  def __init__(self):
    self.full = None
    self.train = None
    self.tokenizer = None
    self.commentary = None

class Network:

  def __init__(self, config, network_type, model_builder):
    self.config = config
    self.network_type = network_type
    self.model_builder = model_builder
    self.training_compiler = None
    self.initialize()

  def initialize(self):
    self.models_predict = [PredictionModels() for _ in devices]
    self.models_train = TrainingModels()
    self.tensorboard_writer_training = None
    self.tensorboard_writer_validation = None

  @property
  def info(self):
    path = self.latest_network_path()
    step_count = int(re.match(".*?([0-9]+)$", path).group(1)) if path else 0
    training_chunk_count = config.count_training_chunks()
    relative_path = config.unmake_path(path).encode("ascii") if path else b""
    return (step_count, training_chunk_count, relative_path)

  @tf.function
  def tf_predict(self, device_index, images):
    return self.models_predict[device_index].predict(images, training=False)

  @tf.function
  def tf_predict_for_training(self, images):
    return self.models_train.train(images, training=False)
  
  @tf.function
  def tf_predict_commentary(self, device_index, images):
    inputs = dict(inputs=images)
    outputs = self.models_predict[device_index].commentary(inputs)
    sequences = outputs["outputs"]
    # Can't detokenize here because of https://github.com/tensorflow/tensorflow/issues/47683
    return sequences

  def predict_batch(self, device_index, images):
    status = self.ensure_prediction(device_index)
    return (status, *self.tf_predict(device_index, images))

  def predict_commentary_batch(self, device_index, images):
    self.ensure_commentary_prediction(device_index)
    return self.tf_predict_commentary(device_index, images)

  def ensure_full(self, device_index):
    with ensure_locks[device_index]:
      # The full model may already exist.
      models = self.models_predict[device_index]
      if models.full:
        return

      models.full, models.full_weights_path = self.build_full(device_index)
      models.full_weights_last_check = time.time()
  
  def build_full(self, log_device_context, network_path=None):
    # Either load it from disk, or create a new one.
    with model_creation_lock:
      if not network_path:
        network_path = self.latest_network_path()
      if network_path:
        log_name = self.get_log_name(network_path)
        log(f"Loading model ({log_device_context}/{self.network_type}/full): {log_name}")
        model_full_path = self.model_full_path(network_path)
        full = self.model_builder()
        self.load_weights(full, model_full_path)
      else:
        log(f"Creating new model ({log_device_context}/{self.network_type}/full)")
        full = self.model_builder()
        model_full_path = None
      return full, model_full_path

  def maybe_check_update_full(self, device_index):
    interval_seconds = self.config.training["wait_milliseconds"] / 1000.0
    models = self.models_predict[device_index]
    now = time.time()
    if (now - models.full_weights_last_check) > interval_seconds:
      models.full_weights_last_check = now
      return self.check_update_full(device_index)
    return PredictionStatus.Nothing

  def check_update_full(self, device_index):
    models = self.models_predict[device_index]
    network_path = self.latest_network_path()
    if network_path:
      # Weights paths for the same network name and type will be identical up until
      # the 9-digit zero-padded step number, which we can compare lexicographically
      # with greater meaning more recent, and coalescing the empty string for no weights
      # (i.e. created from scratch).
      model_full_path = self.model_full_path(network_path)
      newer_weights_available = ((models.full_weights_path or "") < model_full_path)
      if newer_weights_available:
        log_name = self.get_log_name(network_path)
        log(f"Updating model ({device_index}/{self.network_type}/full): {log_name}")
        self.load_weights(models.full, model_full_path)
        models.full_weights_path = model_full_path
        return PredictionStatus.UpdatedNetwork
    return PredictionStatus.Nothing

  def ensure_prediction(self, device_index):
    with ensure_locks[device_index]:
      # The prediction model may already exist.
      if self.models_predict[device_index].predict:
        # Occasionally check for more recent weights to load.
        return self.maybe_check_update_full(device_index)

      # Take the prediction subset from the full model.
      self.ensure_full(device_index)
      self.models_predict[device_index].predict = ModelBuilder().subset_predict(self.models_predict[device_index].full)
      return PredictionStatus.Nothing
  
  def ensure_training(self, network_path=None):
    # The training subset may already exist.
    if self.models_train.train:
      return self.models_train.train

    # Build a full model.
    self.models_train.full, _ = self.build_full("training", network_path)

    # Take the training subset from the full model.
    self.models_train.train = ModelBuilder().subset_train(self.models_train.full)

    # Compile the new subset for training.
    self.training_compiler(self.models_train.train)

    # Set up TensorBoard.
    tensorboard_network_path = self.config.join(self.config.misc["paths"]["tensorboard"], self.config.network_name, self.network_type)
    self.tensorboard_writer_training = tf.summary.create_file_writer(self.config.join(tensorboard_network_path, "training"))
    self.tensorboard_writer_validation = tf.summary.create_file_writer(self.config.join(tensorboard_network_path, "validation"))

    return self.models_train.train

  def ensure_tokenizer(self, models):
    import tokenization

    # The tokenizer may already exist.
    if models.tokenizer:
      return models.tokenizer
    
    # Either load the SentencePiece model from disk, or train a new one.
    models.tokenizer = tokenization.ensure_tokenizer(config, ModelBuilder.transformer_vocabulary_size)
    return models.tokenizer

  def ensure_commentary_prediction(self, device_index):
    with ensure_locks[device_index]:
      models = self.models_predict[device_index]
      # The commentary model may already exist.
      # Never look for updated full model or commentary model weights for commentary prediction.
      if models.commentary:
        return models.commentary

      # The commentary model requires the tokenizer, and the full model as an encoder.
      self.ensure_tokenizer(models)
      self.ensure_full(device_index)

      # Either load from disk, or create new.
      with model_creation_lock:
        network_path = self.latest_network_path_commentary()
        if network_path:
          log_name = self.get_log_name(network_path)
          log(f"Loading model ({device_index}/{self.network_type}/commentary): {log_name}")
          models.commentary = ModelBuilder().build_commentary(self.config, models.tokenizer, models.full, strategy=None)
          model_commentary_path = self.model_commentary_path(network_path)
          self.load_weights(models.commentary, model_commentary_path)
        else:
          log(f"Creating new model ({device_index}/{self.network_type}/commentary)")
          models.commentary = ModelBuilder().build_commentary(self.config, models.tokenizer, models.full, strategy=None)

      return self.models_train.commentary

  def ensure_commentary_training(self):
    # The commentary model may already exist.
    if self.models_train.commentary:
      return self.models_train.commentary

    # The commentary model requires the tokenizer, and the full training model as an encoder.
    self.ensure_tokenizer(self.models_train)
    self.ensure_training()

    # Either load from disk, or create new.
    with model_creation_lock:
      log_device_context = "training"
      network_path = self.latest_network_path_commentary()
      if network_path:
        log_name = self.get_log_name(network_path)
        log(f"Loading model ({log_device_context}/{self.network_type}/commentary): {log_name}")
        self.models_train.commentary = ModelBuilder().build_commentary(self.config, self.models_train.tokenizer, self.models_train.full, trainer.strategy)
        model_commentary_path = self.model_commentary_path(network_path)
        self.load_weights(self.models_train.commentary, model_commentary_path)
      else:
        log(f"Creating new model ({log_device_context}/{self.network_type}/commentary)")
        self.models_train.commentary = ModelBuilder().build_commentary(self.config, self.models_train.tokenizer, self.models_train.full, trainer.strategy)

    # Compile the commentary model for training.
    self.commentary_training_compiler(self.models_train.commentary)

    return self.models_train.commentary

  # Storage may be slow, e.g. Google Cloud Storage, so retry.
  # For now, sleep for 1 second, up to 10 retry attempts (11 total).
  def load_weights(self, model, path):
    try:
      model.load_weights(path)
      return
    except:
      for _ in range(10):
        time.sleep(1.0)
        try:
          model.load_weights(path)
          return
        except:
          pass
    raise Exception(f"Failed to load weights from: {path}")

  def save(self, step):
    network_path = self.make_network_path(step)
    log_name = self.get_log_name(network_path)
    log_device_context = "training"

    # Save the full model from training.
    log(f"Saving model ({log_device_context}/{self.network_type}/full): {log_name}")
    model_full_path = self.model_full_path(network_path)
    self.models_train.full.save_weights(model_full_path, save_format="tf")

    # Save the commentary model if it exists.
    if self.models_train.commentary:
      log(f"Saving model ({log_device_context}/{self.network_type}/commentary): {log_name}")
      model_commentary_path = self.model_commentary_path(network_path)
      self.models_train.commentary.save_weights(model_commentary_path, save_format="tf")

  def get_log_name(self, network_path):
    return os.path.basename(os.path.normpath(network_path))

  def latest_network_path(self):
    return self.config.latest_network_path_for_type_and_model(self.config.network_name, self.network_type, "model")
  
  def latest_network_path_commentary(self):
    return self.config.latest_network_path_for_type_and_model(self.config.network_name, self.network_type, "commentary")

  def make_network_path(self, step):
    return make_network_path_for_network(self.config.network_name, step)

  def make_network_path_for_network(self, network_name, step):
    parent_path = self.config.misc["paths"]["networks"]
    directory_name = f"{network_name}_{str(step).zfill(9)}"
    return self.config.join(parent_path, directory_name)

  def model_full_path(self, network_path):
    return self.config.join(network_path, self.network_type, "model", "weights")

  def model_commentary_path(self, network_path):
    return self.config.join(network_path, self.network_type, "commentary", "weights")

# --- Networks ---

class Networks:

  def __init__(self, config):
    self.config = config

    # The teacher network uses the full 19*256 model.
    self.teacher = Network(config, "teacher", lambda: ModelBuilder().build(config))

    # The student network uses the smaller 8*64 model.
    self.student = Network(config, "student", lambda: ModelBuilder().build_student(config, StudentModel))

  def log(self, *args):
    log(*args)

# --- Helpers ---

thread_local = threading.local()

def choose_device_index():
  try:
    return thread_local.device_index
  except:
    pass
  thread_ident = threading.get_ident()
  with device_lock:
    next_index = len(thread_ident_to_index)
    thread_index = thread_ident_to_index.setdefault(thread_ident, next_index)
  device_index = thread_index % len(devices)
  thread_local.device_index = device_index
  return device_index

def device(device_index):
  return tf.device(devices[device_index].name)

# --- C++ API ---

def predict_batch_teacher(images):
  device_index = choose_device_index()
  with device(device_index):
    status, value, policy = networks.teacher.predict_batch(device_index, images)
    return status, np.array(memoryview(value)), np.array(memoryview(policy))

def predict_batch_student(images):
  device_index = choose_device_index()
  with device(device_index):
    status, value, policy = networks.student.predict_batch(device_index, images)
    return status, np.array(memoryview(value)), np.array(memoryview(policy))

def predict_commentary_batch(images):
  # Always use the teacher network for commentary.
  device_index = choose_device_index()
  with device(device_index):
    sequences = networks.teacher.predict_commentary_batch(device_index, images)
  # Have to detokenize here because of https://github.com/tensorflow/tensorflow/issues/47683
  comments = networks.teacher.models_predict[device_index].tokenizer.detokenize(sequences)
  return np.array(memoryview(comments)) # C++ expects bytes

def train_teacher(game_types, training_windows, step, checkpoint):
  trainer.train(networks.teacher, None, game_types, training_windows, step, checkpoint)

def train_student(game_types, training_windows, step, checkpoint):
  trainer.train(networks.student, networks.teacher, game_types, training_windows, step, checkpoint)

def train_commentary(step, checkpoint):
  # Always use the teacher network for commentary.
  trainer.train_commentary(networks.teacher, step, checkpoint)

def log_scalars_teacher(step, names, values):
  trainer.log_scalars(networks.teacher, step, names, values)

def log_scalars_student(step, names, values):
  trainer.log_scalars(networks.student, step, names, values)

def get_network_info_teacher():
  return networks.teacher.info

def get_network_info_student():
  return networks.student.info

def save_network_teacher(checkpoint):
  networks.teacher.save(checkpoint)
  
def save_network_student(checkpoint):
  networks.student.save(checkpoint)

def save_file(relative_path, data):
  config.save_file(relative_path, data)

def load_file(relative_path):
  return config.load_file(relative_path)

def file_exists(relative_path):
  return config.file_exists(relative_path)

def launch_gui(*args):
  import gui
  gui.launch(*args)

def update_gui(*args):
  import gui
  gui.update(*args)

def debug_decompress(result, image_pieces_auxiliary, policy_row_lengths, policy_indices, policy_values, decompress_positions_modulus):
  indices = tf.range(0, len(policy_row_lengths), decompress_positions_modulus, dtype=tf.int64)
  images, values, policies = datasets.decompress(result, image_pieces_auxiliary,
    policy_row_lengths, policy_indices, policy_values, indices)
  return np.array(memoryview(images)), np.array(memoryview(values)), np.array(memoryview(policies))

def optimize_parameters():
  import optimization
  optimization.Session(config).run()

# --- Initialize ---

# Build the device mapping safely.
device_lock = threading.Lock()
# Prevent fine-grained races in get-or-create logic.
ensure_locks = [threading.RLock() for _ in devices]
# Only create one TensorFlow/Keras model at a time, even if on different devices.
model_creation_lock = threading.Lock()

config = Config(bool(tpu_strategy))
networks = Networks(config)
datasets = DatasetBuilder(config)
trainer = Trainer(networks, tpu_strategy, devices, datasets)

# Start the profiler server.
profiler_server_port = 6009
tf.profiler.experimental.server.start(profiler_server_port)

# Log some configuration.
log("################################################################################")
log("Network:", config.network_name)
log("Role:", config.role)
log("Data root:", config.data_root)
log("Local data root:", config.determine_local_data_root())
log("Using TPU:", config.is_tpu)
log(f"TPU devices: {[t.name for t in tpus]}")
log(f"GPU devices: {[g.name for g in gpus]}")
log("Training devices:", trainer.device_count)
log("Per-replica batch size:", trainer.per_replica_batch_size)
log("Global batch size:", trainer.global_batch_size)
log("Per-replica batch size (commentary):", trainer.per_replica_batch_size_commentary)
log("Global batch size (commentary):", trainer.global_batch_size_commentary)
if isinstance(trainer.strategy, tf.distribute.TPUStrategy):
  log("Training strategy: TPU")
elif isinstance(trainer.strategy, tf.distribute.MirroredStrategy):
  log("Training strategy: Mirrored")
else:
  log("Training strategy: Default")
log(f"Profiler server: localhost:{profiler_server_port}")
log("################################################################################")
