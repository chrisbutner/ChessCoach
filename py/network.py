from __future__ import absolute_import, division, print_function, unicode_literals

import math
import numpy
import time
import os

silent = bool(os.environ.get("CHESSCOACH_SILENT"))
if silent:
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.keras import backend as K

from config import Config
from model import ChessCoachModel
import storage

K.set_image_data_format("channels_first")

def log(*args):
  if not silent:
    print(*args)

class Network(object):

  def predict_batch(self, image):
    assert False

class KerasNetwork(Network):

  def __init__(self, model=None):
    self.model = model or ChessCoachModel().build()
    optimizer = tf.keras.optimizers.SGD(
      learning_rate=get_learning_rate(config.training_network["learning_rate_schedule"], 0),
      momentum=config.training_network["momentum"])
    losses = ["mean_squared_error", self.flat_categorical_crossentropy_from_logits]
    self.model.compile(optimizer=optimizer, loss=losses, loss_weights=[1.0, 1.0], metrics=[[], [self.flat_categorical_accuracy]])

  # This fixes an issue with categorical_crossentropy calculating incorrectly
  # over our 73*8*8 output planes - loss ends up way too small.
  def flat_categorical_crossentropy_from_logits(self, y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true=K.batch_flatten(y_true), y_pred=K.batch_flatten(y_pred), from_logits=True)

  # This fixes the same issue with categorical_accuracy. No point doing softmax on logits for argmax.
  def flat_categorical_accuracy(self, y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true=K.batch_flatten(y_true), y_pred=K.batch_flatten(y_pred))

  def predict_batch(self, image):
    assert False

class UniformNetwork(Network):

  def __init__(self):
    self.latest_values = None
    self.latest_policies = None

  def predict_batch(self, image):
    # Check both separately because of threading
    values = self.latest_values
    policies = self.latest_policies
    if (values is None) or (len(image) != len(values)):
      values = numpy.full((len(image)), 0.0, dtype=numpy.float32)
      self.latest_values = values
    if (policies is None) or (len(image) != len(policies)):
      policies = numpy.zeros((len(image), ChessCoachModel.output_planes_count, ChessCoachModel.board_side, ChessCoachModel.board_side), dtype=numpy.float32)
      self.latest_policies = policies
    return values, policies

class TensorFlowNetwork(Network):

  # NEED to pass the model in THEN extract the function, otherwise all hell breaks loose.
  def __init__(self, model):
    self.model = model
    self.function = self.model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

  def predict_batch(self, image):
    image = tf.constant(image)
    prediction = self.function(image)
    value, policy = prediction[ChessCoachModel.output_value_name], prediction[ChessCoachModel.output_policy_name]
    return numpy.array(value), numpy.array(policy)

class Networks:

  def __init__(self):
    self.network_name = "network"
    self.prediction_network = None
    self.training_network = None

def update_network_for_predictions(network_path):
  name = os.path.basename(os.path.normpath(network_path))
  log(f"Loading network (predictions): {name}...")
  while True:
    try:
      tf_model = tf.saved_model.load(network_path)
      break
    except Exception as e:
      log("Exception:", e)
      time.sleep(0.25)
  network = TensorFlowNetwork(tf_model)
  log(f"Loaded network (predictions): {name}")
  return network

def update_network_for_training(network_path):
  name = os.path.basename(os.path.normpath(network_path))
  log(f"Loading network (training): {name}...")
  while True:
    try:
      # Work around bug: AttributeError: 'CategoricalCrossentropy' object has no attribute '__name__'
      keras_model = tf.keras.models.load_model(network_path, compile=False)
      break
    except Exception as e:
      log("Exception:", e)
      time.sleep(0.25)
  network = KerasNetwork(keras_model)
  log(f"Loaded network (training): {name}")
  return network

def ensure_training():
  if not networks.training_network:
    networks.training_network = storage.load_latest_network(config, networks.network_name, update_network_for_training)
  if not networks.training_network:
    log("Creating new network (training)")
    networks.training_network = KerasNetwork()

def get_learning_rate(schedule, step):
  rate = 0.0
  for key, value in schedule:
    if step >= key:
      rate = value
    else:
      return rate

def predict_batch(image):
  return networks.prediction_network.predict_batch(image)

def train_batch(step, images, values, policies):
  ensure_training()
  learning_rate = get_learning_rate(config.training_network["learning_rate_schedule"], step)
  K.set_value(networks.training_network.model.optimizer.lr, learning_rate)

  do_log_training = ((step % config.training_network["validation_interval"]) == 0)
  if do_log_training:
    log_training_prepare(step)
  losses = networks.training_network.model.train_on_batch(images, [values, policies])
  if do_log_training:
    log_training("training", tensorboard_writer_training, step, losses)

def validate_batch(step, images, values, policies):
  ensure_training()
  log_training_prepare(step)
  losses = networks.training_network.model.test_on_batch(images, [values, policies])
  log_training("validation", tensorboard_writer_validation, step, losses)

def log_scalars(step, names, values):
  with tensorboard_writer_validation.as_default():
    tf.summary.experimental.set_step(step)
    for name, value in zip(names, values):
      tf.summary.scalar(name.decode("utf-8"), value)

def should_log_graph(step):
  return (step == 1)

def log_training_prepare(step):
  if should_log_graph(step):
    tf.summary.trace_on(graph=True, profiler=False)

def log_training(type, writer, step, losses):
  log(f"Loss: {losses[0]:.6f} (Value: {losses[1]:.6f}, Policy: {losses[2]:.6f}), Accuracy (policy argmax): {losses[3]:.6f} ({type})")
  with writer.as_default():
    tf.summary.experimental.set_step(step)
    if should_log_graph(step):
      tf.summary.trace_export("model")
    log_loss_accuracy(losses)
    log_weights()
    writer.flush()

def log_loss_accuracy(losses):
  with tf.name_scope("loss"):
    tf.summary.scalar("overall loss", losses[0])
    tf.summary.scalar("value loss", losses[1])
    tf.summary.scalar("policy loss", losses[2])
    tf.summary.scalar("L2 loss", losses[0] - losses[1] - losses[2])
  with tf.name_scope("accuracy"):
    tf.summary.scalar("policy accuracy", losses[3])

def log_weights():
  for layer in networks.training_network.model.layers:
    for weight in layer.weights:
      weight_name = weight.name.replace(':', '_')
      tf.summary.histogram(weight_name, weight)

def load_network(network_name):
  networks.network_name = network_name

  # Load latest prediction network now, but delay loading training network until necessary.
  networks.training_network = None
  networks.prediction_network = storage.load_latest_network(config, network_name, update_network_for_predictions)
  if not networks.prediction_network:
    networks.prediction_network = UniformNetwork()
    log("Loaded uniform network (predictions)")

def save_network(checkpoint):
  ensure_training()
  log(f"Saving network ({checkpoint} steps)...")
  path = storage.save_network(config, networks.network_name, checkpoint, networks.training_network)
  log(f"Saved network ({checkpoint} steps)")
  networks.prediction_network = update_network_for_predictions(path)

config = Config()
networks = Networks()
tensorboard_network_path = os.path.join(config.misc["paths"]["tensorboard"], config.training_network["name"])
tensorboard_writer_training_path = os.path.join(tensorboard_network_path, "training")
tensorboard_writer_training = tf.summary.create_file_writer(tensorboard_writer_training_path)
tensorboard_writer_validation_path = os.path.join(tensorboard_network_path, "validation")
tensorboard_writer_validation = tf.summary.create_file_writer(tensorboard_writer_validation_path)
