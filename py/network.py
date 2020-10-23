# --- Require silent TensorFlow initialization when running as UCI ---

import os
import socket

silent = bool(os.environ.get("CHESSCOACH_SILENT"))
if silent:
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def log(*args):
  if not silent:
    print(*args)

import tensorflow as tf

# --- TPU/GPU initialization ---

tpu_strategy = None
try:
  tpu_name = os.environ.get("TPU_NAME") or socket.gethostname()
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  tpu_strategy = tf.distribute.TPUStrategy(resolver)
  log("Found TPU")
except:
  log("No TPU or error resolving")

tpus = tf.config.experimental.list_logical_devices("TPU")
gpus = tf.config.experimental.list_logical_devices("GPU")
devices = tpus + gpus
thread_ident_to_index = {}
log(f"TPU devices: {[t.name for t in tpus]}")
log(f"GPU devices: {[g.name for g in gpus]}")

# --- Now it's safe for further imports ---

import math
import re
import time
import threading
import numpy as np

from config import Config
from model import ModelBuilder
import transformer
from training import Trainer
from dataset import DatasetBuilder

# --- Network ---

class Models:
  def __init__(self):
    self.full = None
    self.full_weights_path = None
    self.full_weights_last_check = None
    self.predict = None
    self.commentary_encoder = None
    self.commentary_decoder = None
    self.commentary_tokenizer = None

class Network:

  def __init__(self, config, network_type, model_builder, name):
    self.config = config
    self.network_type = network_type
    self.model_builder = model_builder
    self.training_compiler = None
    self._name = name
    self.initialize()

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, value):
    self._name = value

    # Clear out any loaded models, ready to lazy-load using the new name.
    self.initialize()

  def initialize(self):
    self.models = [Models() for _ in devices]
    self.model_train_full = None
    self.model_train = None
    self.tensorboard_writer_training = None
    self.tensorboard_writer_validation = None

  @property
  def info(self):
    path = self.latest_network_path()
    step_count = int(re.match(".*?([0-9]+)$", path).group(1)) if path else 0
    training_chunk_count = config.count_training_chunks()
    return (step_count, training_chunk_count)

  @tf.function
  def tf_predict(self, device_index, images):
      return self.models[device_index].predict(images, training=False)

  @tf.function
  def tf_predict_for_training(self, images):
      return self.model_train(images, training=False)

  def predict_batch(self, device_index, images):
    self.ensure_prediction(device_index)
    return self.tf_predict(device_index, images)

  def predict_for_training_batch(self, images):
    # Rely on caller to have ensured that the training model is set up.
    return self.tf_predict_for_training(images)

  def predict_commentary_batch(self, device_index, images):
    networks.teacher.ensure_commentary(device_index)

    encoder = self.models[device_index].commentary_encoder
    decoder = self.models[device_index].commentary_decoder
    tokenizer = self.commentary_tokenizer

    start_token = tokenizer.word_index[ModelBuilder.token_start]
    end_token = tokenizer.word_index[ModelBuilder.token_end]
    max_length = ModelBuilder.transformer_max_length

    sequences = transformer.predict_greedy(encoder, decoder,
      start_token, end_token, max_length, images)

    def trim_start_end_tokens(sequence):
      for i, token in enumerate(sequence):
        if (token == end_token):
          return sequence[1:i]
      return sequence[1:]

    sequences = [trim_start_end_tokens(s) for s in np.array(memoryview(sequences))]
    comments = tokenizer.sequences_to_texts(sequences)
    comments = np.array([c.encode("utf-8") for c in comments])
    return comments

  def ensure_full(self, device_index):
    with ensure_locks[device_index]:
      # The full model may already exist.
      models = self.models[device_index]
      if models.full:
        return

      models.full, models.full_weights_path = self.build_full(device_index)
      models.full_weights_last_check = time.time()
  
  def build_full(self, log_device_context):
    # Either load it from disk, or create a new one.
    with model_creation_lock:
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
    interval_seconds = self.config.self_play["network_update_check_interval_seconds"]
    models = self.models[device_index]
    now = time.time()
    if (now - models.full_weights_last_check) > interval_seconds:
      models.full_weights_last_check = now
      self.check_update_full(device_index)

  def check_update_full(self, device_index):
    models = self.models[device_index]
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

  def ensure_prediction(self, device_index):
    with ensure_locks[device_index]:
      # The prediction model may already exist.
      if self.models[device_index].predict:
        # Occasionally check for more recent weights to load.
        self.maybe_check_update_full(device_index)
        return

      # Take the prediction subset from the full model.
      self.ensure_full(device_index)
      self.models[device_index].predict = ModelBuilder().subset_predict(self.models[device_index].full)
  
  def ensure_training(self):
    # The training subset may already exist.
    if self.model_train:
      return self.model_train

    # Build a full model.
    self.model_train_full, _ = self.build_full("training")

    # Take the training subset from the full model.
    self.model_train = ModelBuilder().subset_train(self.model_train_full)

    # Compile the new subset for training.
    self.training_compiler(self.model_train)

    # Set up TensorBoard.
    tensorboard_network_path = self.config.join(self.config.misc["paths"]["tensorboard"], self._name, self.network_type)
    self.tensorboard_writer_training = tf.summary.create_file_writer(self.config.join(tensorboard_network_path, "training"))
    self.tensorboard_writer_validation = tf.summary.create_file_writer(self.config.join(tensorboard_network_path, "validation"))

    return self.model_train

  def ensure_commentary(self, device_index):
    with ensure_locks[device_index]:
      # The encoder, decoder and tokenizer may already exist.
      if self.models[device_index].commentary_encoder:
        # Occasionally check for more recent weights to load.
        self.maybe_check_update_full(device_index)
        return

      # Take the encoder subset from the full model.
      self.ensure_full(device_index)
      self.models[device_index].commentary_encoder = ModelBuilder().subset_commentary_encoder(self.models[device_index].full)

      # Either load decoder and tokenizer from disk, or create new.
      with model_creation_lock:
        network_path = self.latest_network_path()
        if network_path:
          log_name = self.get_log_name(network_path)
          log(f"Loading model ({device_index}/{self.network_type}/commentary): {log_name}")

          # Load the tokenizer first.
          commentary_tokenizer_path = self.commentary_tokenizer_path(network_path)
          with open(commentary_tokenizer_path, 'r') as f:
              self.commentary_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

          # Build the decoder using the tokenizer, then load weights.
          decoder, _ = ModelBuilder().build_commentary_decoder(self.config, self.commentary_tokenizer)
          self.models[device_index].commentary_decoder = decoder
          model_commentary_decoder_path = self.model_commentary_decoder_path(network_path)
          self.load_weights(self.models[device_index].commentary_decoder, model_commentary_decoder_path)
        else:
          log(f"Creating new model ({device_index}/{self.network_type}/commentary)")
          decoder, self.commentary_tokenizer = ModelBuilder().build_commentary_decoder(self.config)
          self.models[device_index].commentary_decoder = decoder

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
    raise Exception("Failed to load weights from: {path}")

  def save(self, step):
    network_path = self.make_network_path(step)
    log_name = self.get_log_name(network_path)
    log_device_context = "training"

    # Save the full model from training.
    log(f"Saving model ({log_device_context}/{self.network_type}/full): {log_name}")
    model_full_path = self.model_full_path(network_path)
    self.model_train_full.save_weights(model_full_path, save_format="tf")

    # # Save the commentary decoder and tokenizer if they exist.
    # if self.models[device_index].commentary_decoder and self.commentary_tokenizer:
    #   log(f"Saving model ({device_index}/{self.network_type}/commentary): {log_name}")

    #   # Save the commentary decoder.
    #   model_commentary_decoder_path = self.model_commentary_decoder_path(network_path)
    #   self.models[device_index].commentary_decoder.save_weights(model_commentary_decoder_path, save_format="tf")

    #   # Save the tokenizer.
    #   commentary_tokenizer_path = self.commentary_tokenizer_path(network_path)
    #   with open(commentary_tokenizer_path, 'w') as f:
    #     f.write(self.commentary_tokenizer.to_json())

  def get_log_name(self, network_path):
    return os.path.basename(os.path.normpath(network_path))

  def latest_network_path(self):
    return self.config.latest_network_path_for_type(self.name, self.network_type)

  def make_network_path(self, step):
    parent_path = self.config.misc["paths"]["networks"]
    directory_name = f"{self.name}_{str(step).zfill(9)}"
    return self.config.join(parent_path, directory_name)

  def model_full_path(self, network_path):
    return self.config.join(network_path, self.network_type, "model", "weights")

  def model_commentary_decoder_path(self, network_path):
    return self.config.join(network_path, self.network_type, "commentary_decoder", "weights")

  def commentary_tokenizer_path(self, network_path):
    return self.config.join(network_path, self.network_type, "commentary_tokenizer.json")

# --- Networks ---

class Networks:

  def __init__(self, config, name="network"):
    self.config = config

    # Set by C++ via load_network depending on use-case.
    self._name = name

    # The teacher network uses the full 19*256 model.
    self.teacher = Network(config, "teacher", lambda: ModelBuilder().build(config), self._name)

    # The student network uses the smaller 8*64 model.
    self.student = Network(config, "student", lambda: ModelBuilder().build_student(config), self._name)

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, value):
    self._name = value
    self.teacher.name = self._name
    self.student.name = self._name

  def log(self, *args):
    log(*args)

# --- Helpers ---

def choose_device_index():
  thread_ident = threading.get_ident()
  with device_lock:
    next_index = len(thread_ident_to_index)
    thread_index = thread_ident_to_index.setdefault(thread_ident, next_index)
  device_index = thread_index % len(devices)
  return device_index

def device(device_index):
  return tf.device(devices[device_index].name)

# --- C++ API ---

def predict_batch_teacher(images):
  trainer.clear_data() # Free up training memory for use in self-play or strength testing.
  device_index = choose_device_index()
  with device(device_index):
    value, policy = networks.teacher.predict_batch(device_index, images)
    return np.array(memoryview(value)), np.array(memoryview(policy))

def predict_batch_student(images):
  trainer.clear_data() # Free up training memory for use in self-play or strength testing.
  device_index = choose_device_index()
  with device(device_index):
    value, policy = networks.student.predict_batch(device_index, images)
    return np.array(memoryview(value)), np.array(memoryview(policy))

def predict_commentary_batch(images):
  device_index = choose_device_index()
  with device(device_index):
    # Always use the teacher network for commentary.
    return networks.teacher.predict_commentary_batch(device_index, images)

def train_teacher(gameTypes, trainingWindows, step, checkpoint):
  trainer.train_teacher(gameTypes, trainingWindows, step, checkpoint)

def train_student(gameTypes, trainingWindows, step, checkpoint):
  trainer.train_student(gameTypes, trainingWindows, step, checkpoint)

def train_commentary_batch(step, images, comments):
  pass # TODO
  #training.train_commentary_batch(step, images, comments)

def log_scalars_teacher(step, names, values):
  trainer.log_scalars(networks.teacher, step, names, values)

def log_scalars_student(step, names, values):
  trainer.log_scalars(networks.student, step, names, values)

def load_network(network_name):
  networks.name = network_name
  return networks.teacher.info # Assume there's a saved teacher, if anything, and return its info.

def save_network_teacher(checkpoint):
  networks.teacher.save(checkpoint)
  
def save_network_student(checkpoint):
  networks.student.save(checkpoint)

def save_file(relative_path, data):
  config.save_file(relative_path, data)

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