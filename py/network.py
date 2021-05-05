# --- Require silent TensorFlow initialization when running as UCI ---

import os
import socket

silent = bool(os.environ.get("CHESSCOACH_SILENT"))
if silent:
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def log(*args):
  if not silent:
    # Flush is required with no tty attached; e.g. docker.
    # PYTHONUNBUFFERED/-u is an alternative, but wasn't working on latest alpha TPU VM build.
    print(*args, flush=True)

import tensorflow as tf

# --- TPU/GPU initialization ---

# Let multiple UCIs run at once in tournaments.
physical_gpus = tf.config.experimental.list_physical_devices("GPU")
if physical_gpus:
  tf.config.experimental.set_memory_growth(physical_gpus[0], True)

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
import functools

from config import Config, PredictionStatus
from model import ModelBuilder
from training import Trainer, StudentModel
from dataset import DatasetBuilder

# --- Network ---

class PredictionModels:
  def __init__(self):
    self.full = None
    self.model_path = ModelPath.NoPath
    self.model_path_last_check = None
    self.predict = None
    self.tokenizer = None
    self.commentary = None

class TrainingModels:
  def __init__(self):
    self.full = None
    self.train = None
    self.swa_full = None
    self.swa_train = None
    self.swa_network = None
    self.swa_count = 0
    self.swa_required = 0
    self.tokenizer = None
    self.commentary = None

# E.g. gs://chesscoach-eu/ChessCoach/Networks/selfplay5_000800000/teacher/model/weights
@functools.total_ordering
class ModelPath:

  def __init__(self, path):
    self.path = path

  def model_type(self):
    return config.path.basename(config.path.dirname(self.path))

  def network_type(self):
    return config.path.basename(config.path.dirname(config.path.dirname(self.path)))
  
  def log_name(self):
    return config.path.basename(config.path.dirname(config.path.dirname(config.path.dirname(self.path))))

  def step_count(self):
    return int(re.search("_([0-9]+)$", self.log_name()).group(1))

  def __str__(self):
    return self.path

  def __repr__(self):
    return self.path

  def __eq__(self, other):
    return self.path == other.path

  # Required for sort(), max()
  def __lt__(self, other):
    # Check for equality first (e.g. when looking for updated weights).
    if self == other:
      return False
    # No path is always earliest.
    if self.path is None:
      return True
    # Prefer more recent steps.
    step_count = self.step_count()
    other_step_count = other.step_count()
    if step_count < other_step_count:
      return True
    if other_step_count < step_count:
      return False
    # Prefer SWA to regular model.
    model_type = self.model_type()
    other_model_type = other.model_type()
    if model_type == "swa" and other_model_type != "swa":
      return False
    if other_model_type == "swa" and model_type != "swa":
      return True
    # Can't compare.
    raise Exception("Can't compare values")

ModelPath.NoPath = ModelPath(None)

class Network:

  def __init__(self, config, network_type, model_builder):
    self.config = config
    self.network_type = network_type
    self.model_builder = model_builder
    self.training_compiler = None
    self._network_weights = config.self_play["network_weights"]
    self.initialize()

  @property
  def network_weights(self):
    return self._network_weights

  @network_weights.setter
  def network_weights(self, value):
    self._network_weights = value

    # Clear out any loaded models, ready to lazy-load using the new weights path.
    self.initialize()

  def initialize(self):
    self.models_predict = [PredictionModels() for _ in devices]
    self.models_train = TrainingModels()
    self.tensorboard_writer_training = None
    self.tensorboard_writer_validation = None

  @property
  def info(self):
    model_path = self.latest_model_path("model")
    step_count = model_path.step_count() if model_path else 0
    swa_path = self.latest_model_path("swa")
    swa_step_count = swa_path.step_count() if swa_path else 0
    training_chunk_count = config.count_training_chunks()
    relative_path = config.unmake_path(config.path.dirname(config.path.dirname(config.path.dirname(model_path.path)))).encode("ascii") if model_path else b""
    return (step_count, swa_step_count, training_chunk_count, relative_path)

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

  def ensure_prediction_full(self, device_index):
    with ensure_locks[device_index]:
      # The full model may already exist.
      models = self.models_predict[device_index]
      if models.full:
        return

      models.full, models.model_path = self.build_full(device_index, ["model", "swa"])
      models.full_weights_last_check = time.time()
  
  def build_full(self, log_device_context, model_types):
    # Either load it from disk, or create a new one.
    with model_creation_lock:
      model_path = self.latest_model_path(model_types)
      if model_path:
        log(f"Loading model ({log_device_context}/{self.network_type}/{model_path.model_type()}): {model_path.log_name()}")
        full = self.model_builder()
        self.load_weights(full, model_path.path)
      else:
        log(f"Creating new model ({log_device_context}/{self.network_type}/model)")
        full = self.model_builder()
        model_path = ModelPath.NoPath
      return full, model_path

  def maybe_check_update_prediction_full(self, device_index):
    interval_seconds = self.config.training["wait_milliseconds"] / 1000.0
    models = self.models_predict[device_index]
    now = time.time()
    if (now - models.full_weights_last_check) > interval_seconds:
      models.full_weights_last_check = now
      return self.check_update_prediction_full(device_index)
    return PredictionStatus.Nothing

  def check_update_prediction_full(self, device_index):
    models = self.models_predict[device_index]
    model_path = self.latest_model_path(["model", "swa"])
    if model_path:
      # Compare model paths using ModelPath.__lt__ and check for a more recent one in storage.
      newer_weights_available = (models.model_path < model_path)
      if newer_weights_available:
        log(f"Updating model ({device_index}/{self.network_type}/{model_path.model_type()}): {model_path.log_name()}")
        self.load_weights(models.full, model_path.path)
        models.model_path = model_path
        return PredictionStatus.UpdatedNetwork
    return PredictionStatus.Nothing

  def ensure_prediction(self, device_index):
    with ensure_locks[device_index]:
      # The prediction model may already exist.
      if self.models_predict[device_index].predict:
        # Occasionally check for more recent weights to load.
        return self.maybe_check_update_prediction_full(device_index)

      # Take the prediction subset from the full model.
      self.ensure_prediction_full(device_index)
      self.models_predict[device_index].predict = ModelBuilder().subset_predict(self.models_predict[device_index].full)
      # Ensure that the prediction cache is clear after initial uniform predictions.
      return PredictionStatus.UpdatedNetwork

  def ensure_tensorboard(self):
    if not self.tensorboard_writer_training:
      tensorboard_network_path = self.config.join(self.config.misc["paths"]["tensorboard"], self.config.network_name, self.network_type)
      self.tensorboard_writer_training = tf.summary.create_file_writer(self.config.join(tensorboard_network_path, "training"))
      self.tensorboard_writer_validation = tf.summary.create_file_writer(self.config.join(tensorboard_network_path, "validation"))
  
  def ensure_training(self):
    # The training subset may already exist.
    if self.models_train.train:
      return self.models_train.train

    # Build a full model.
    self.models_train.full, _ = self.build_full("training", "model")

    # Take the training subset from the full model.
    self.models_train.train = ModelBuilder().subset_train(self.models_train.full)

    # Compile the new subset for training.
    self.training_compiler(self.models_train.train)

    # Set up TensorBoard.
    self.ensure_tensorboard()

    return self.models_train.train

  def ensure_tokenizer(self, models):
    import tokenization

    # The tokenizer may already exist.
    if models.tokenizer:
      return models.tokenizer
    
    # Either load the SentencePiece model from disk, or train a new one.
    models.tokenizer = tokenization.ensure_tokenizer(config, ModelBuilder.transformer_vocabulary_size, log)
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
      self.ensure_prediction_full(device_index)

      # Either load from disk, or create new.
      with model_creation_lock:
        model_path = self.latest_model_path("commentary")
        if model_path:
          log(f"Loading model ({device_index}/{self.network_type}/{model_path.model_type()}): {model_path.log_name()}")
          models.commentary = ModelBuilder().build_commentary(self.config, models.tokenizer, models.full, strategy=None)
          self.load_weights(models.commentary, model_path.path)
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
      model_path = self.latest_model_path("commentary")
      if model_path:
        log(f"Loading model ({log_device_context}/{self.network_type}/{model_path.model_type()}): {model_path.log_name()}")
        self.models_train.commentary = ModelBuilder().build_commentary(self.config, self.models_train.tokenizer, self.models_train.full, trainer.strategy)
        self.load_weights(self.models_train.commentary, model_path.path)
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

  def save(self, checkpoint):
    model_path = self.make_model_path("model", checkpoint)
    log_device_context = "training"

    # Save the full model from training.
    log(f"Saving model ({log_device_context}/{self.network_type}/{model_path.model_type()}): {model_path.log_name()}")
    self.models_train.full.save_weights(model_path.path, save_format="tf")

    # Contribute weights to the SWA model if it exists. If it just hasn't been created yet then
    # our latest weights will be loaded from disk during its creation, so no work to do here yet.
    if self.models_train.swa_full:
      log(f"Contributing weights to SWA model: ({log_device_context}/{self.network_type}/{model_path.model_type()}): {model_path.log_name()}")
      self.contribute_swa(self.models_train.full.get_weights())

    # Save the commentary model if it exists.
    if self.models_train.commentary:
      commentary_model_path = self.make_model_path("commentary", checkpoint)
      log(f"Saving model ({log_device_context}/{self.network_type}/{commentary_model_path.model_type()}): {commentary_model_path.log_name()}")
      self.models_train.commentary.save_weights(commentary_model_path.path, save_format="tf")

  def calculate_swa_networks_required(self):
    # E.g. with swa_minimum_contribution=0.01, swa_decay=0.5, contributions eventually become
    # { 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, ... }. Six of these >= 0.01.
    return int(math.log(self.config.training["swa_minimum_contribution"]) / math.log(self.config.training["swa_decay"]))

  def contribute_swa(self, weights):
    if self.models_train.swa_count == 0:
      self.models_train.swa_full.set_weights(weights)
    else:
      decay = self.config.training["swa_decay"]
      average = np.array(self.models_train.swa_full.get_weights(), dtype=object)
      average += (np.array(weights, dtype=object) - average) * decay
      self.models_train.swa_full.set_weights(average)
    self.models_train.swa_count += 1

  def save_swa(self, checkpoint, teacher_network=None):
    log_device_context = "training"

    # Set up the stochastic weight averaging (SWA) network if none exists yet.
    if not self.models_train.swa_full:
      log(f"Creating new model ({log_device_context}/{self.network_type}/swa)")
      with trainer.strategy.scope():
        self.models_train.swa_full = self.model_builder()
        self.models_train.swa_train = ModelBuilder().subset_train(self.models_train.swa_full)
        self.training_compiler(self.models_train.swa_train, learning_rate=0.0)
      self.models_train.swa_network = SwaNetwork(self.models_train.swa_train)
      self.models_train.swa_count = 0
      self.models_train.swa_required = self.calculate_swa_networks_required()
      
      # Contribute existing checkpoint weights up to the required count.
      model_paths = self.latest_model_paths("model", self.models_train.swa_required)
      scratch = self.model_builder()
      for model_path in model_paths:
        log(f"Contributing weights to SWA model: ({log_device_context}/{self.network_type}/{model_path.model_type()}): {model_path.log_name()}")
        scratch.load_weights(model_path.path)
        self.contribute_swa(scratch.get_weights())

    # Re-train batch normalization using zero learning rate after averaging weights.
    starting_step = (checkpoint - self.config.training["swa_batchnorm_steps"] + 1)
    trainer.train(self.models_train.swa_network, teacher_network, starting_step, checkpoint, log=False)

    # Save the SWA model.
    model_path = self.make_model_path("swa", checkpoint)
    log(f"Saving model ({log_device_context}/{self.network_type}/{model_path.model_type()}): {model_path.log_name()}")
    self.models_train.swa_full.save_weights(model_path.path, save_format="tf")

  def partial_sort_latest(self, x, max_count):
    count = len(x)
    partition = range(max(0, count - max_count), count)
    return np.partition(np.array(x), partition)[-max_count:]

  def latest_model_path(self, model_types):
    model_paths = self.find_filter_model_paths(model_types)
    return max(model_paths) if model_paths else None
  
  def latest_model_paths(self, model_types, max_count):
    model_paths = self.find_filter_model_paths(model_types)
    model_paths = self.partial_sort_latest(model_paths, max_count)
    return model_paths

  def find_filter_model_paths(self, model_types):
    single_model_type = isinstance(model_types, str)
    model_type_glob = model_types if single_model_type else "*"
    network_pattern = self.network_weights if self.network_weights else (self.config.network_name + "_*")
    results = self.config.latest_model_paths_for_pattern_type_model(network_pattern, self.network_type, model_type_glob)
    model_paths = [ModelPath(result) for result in results]
    if not single_model_type:
      model_paths = [m for m in model_paths if m.model_type() in model_types]
    return model_paths

  def make_model_path(self, model_type, checkpoint):
    return ModelPath(self.config.make_model_path(self.network_type, model_type, checkpoint))

class SwaNetwork:

  def __init__(self, train):
    self.train = train

  def ensure_training(self):
    return self.train

# --- Networks ---

class Networks:

  def __init__(self, config):
    self.config = config

    # The teacher network uses the full 19*256 model.
    self.teacher = Network(config, "teacher", lambda: ModelBuilder().build(config))

    # The student network uses the smaller 8*128 model.
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

def train_teacher(step, checkpoint):
  if networks.teacher.network_weights or networks.student.network_weights:
    raise ValueError("Cannot train with network_weights set")
  trainer.train(networks.teacher, None, step, checkpoint)

def train_student(step, checkpoint):
  if networks.teacher.network_weights or networks.student.network_weights:
    raise ValueError("Cannot train with network_weights set")
  trainer.train(networks.student, networks.teacher, step, checkpoint)

def train_commentary(step, checkpoint):
  # Always use the teacher network for commentary.
  trainer.train_commentary(networks.teacher, step, checkpoint)

def log_scalars_teacher(step, names, values):
  trainer.log_scalars(networks.teacher, step, names, values)

def log_scalars_student(step, names, values):
  trainer.log_scalars(networks.student, step, names, values)

def update_network_weights(network_weights):
  networks.teacher.network_weights = network_weights
  networks.student.network_weights = network_weights

def get_network_info_teacher():
  return networks.teacher.info

def get_network_info_student():
  return networks.student.info

def save_network_teacher(checkpoint):
  networks.teacher.save(checkpoint)
  
def save_network_student(checkpoint):
  networks.student.save(checkpoint)

def save_swa_network_teacher(checkpoint):
  networks.teacher.save_swa(checkpoint)
  
def save_swa_network_student(checkpoint):
  networks.student.save_swa(checkpoint, networks.teacher)

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
  optimization.optimize_parameters(config)

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
log("################################################################################")
