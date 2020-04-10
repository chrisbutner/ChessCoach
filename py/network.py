from __future__ import absolute_import, division, print_function, unicode_literals

import math
import numpy
import time
import os

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.keras import backend as K

from model import ChessCoachModel
from profiler import Profiler
import storage
import game

K.set_image_data_format("channels_first")

##########################
####### Helpers ##########

class AlphaZeroConfig(object):

  def __init__(self):
    ### Training
    self.run_name = "run1"
    self.batch_size = 2048 # OOM on GTX 1080 @ 4096
    self.chesscoach_training_factor = 4096 / self.batch_size # Increase training to compensate for lower batch size.

    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.chesscoach_slowdown_factor = self.chesscoach_training_factor
    self.learning_rate_schedule = {
        int(0 * self.chesscoach_training_factor): 2e-1 / self.chesscoach_slowdown_factor,
        int(100e3 * self.chesscoach_training_factor): 2e-2 / self.chesscoach_slowdown_factor,
        int(300e3 * self.chesscoach_training_factor): 2e-3 / self.chesscoach_slowdown_factor,
        int(500e3 * self.chesscoach_training_factor): 2e-4 / self.chesscoach_slowdown_factor
    }

class Network(object):

  def predict_batch(self, image):
    assert False

class KerasNetwork(Network):

  def __init__(self, model=None):
    self.model = model or ChessCoachModel().build()
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate_schedule[0], momentum=config.momentum)
    losses = ["mean_squared_error", self.flat_categorical_crossentropy_from_logits]
    self.model.compile(optimizer=optimizer, loss=losses, metrics=[[], ["accuracy"]])

  # This fixes an issue with categorical_crossentropy calculating incorrectly
  # over our 73*8*8 output planes - loss ends up way too small.
  def flat_categorical_crossentropy_from_logits(self, y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true=K.batch_flatten(y_true), y_pred=K.batch_flatten(y_pred), from_logits=True)

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
    if ((values is None) or (len(image) != len(values))):
      values = numpy.full((len(image)), 0.0, dtype=numpy.float32)
      self.latest_values = values
    if ((policies is None) or (len(image) != len(policies))):
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

##### End Helpers ########
##########################

def update_network_for_predictions(network_path):
  name = os.path.basename(os.path.normpath(network_path))
  print(f"Loading network (predictions): {name}...")
  while True:
    try:
      tf_model = tf.saved_model.load(network_path)
      break
    except Exception as e:
      print("Exception:", e)
      time.sleep(0.25)
  prediction_network = TensorFlowNetwork(tf_model)
  print(f"Loaded network (predictions): {name}")
  return prediction_network

def update_network_for_training(network_path):
  name = os.path.basename(os.path.normpath(network_path))
  print(f"Loading network (training): {name}...")
  while True:
    try:
      # Work around bug: AttributeError: 'CategoricalCrossentropy' object has no attribute '__name__'
      keras_model = tf.keras.models.load_model(network_path, compile=False)
      break
    except Exception as e:
      print("Exception:", e)
      time.sleep(0.25)
  training_network = KerasNetwork(keras_model)
  print(f"Loaded network (training): {name}")
  return training_network

def prepare_predictions():
  prediction_network = storage.load_latest_network(update_network_for_predictions)
  if (not prediction_network):
    prediction_network = UniformNetwork()
    print("Loaded uniform network (predictions)")
  return prediction_network

def prepare_training():
  training_network = storage.load_latest_network(update_network_for_training)
  if (not training_network):
    print("Creating new network (training)")
    training_network = KerasNetwork()
  return training_network

def predict_batch(image):
  return prediction_network.predict_batch(image)

def train_batch(step, images, values, policies):
  new_learning_rate = config.learning_rate_schedule.get(step)
  if (new_learning_rate is not None):
    K.set_value(training_network.model.optimizer.lr, new_learning_rate)

  log_training_prepare(step)
  losses = training_network.model.train_on_batch(images, [values, policies])
  log_training(step, losses)

def should_log_graph(step):
  return (step == 1)

def log_training_prepare(step):
  if (should_log_graph(step)):
    tf.summary.trace_on(graph=True, profiler=False)

def log_training(step, losses):
  print(f"Loss: {losses[0]:.6f} (Value: {losses[1]:.6f}, Policy: {losses[2]:.6f}), Accuracy (policy argmax): {losses[3]:.6f}")
  with tensorboard_writer_training.as_default():
    tf.summary.experimental.set_step(step)
    if (should_log_graph(step)):
      tf.summary.trace_export("model")
    with tf.name_scope("loss"):
      tf.summary.scalar("overall loss", losses[0])
      tf.summary.scalar("value loss", losses[1])
      tf.summary.scalar("policy loss", losses[2])
      tf.summary.scalar("L2 loss", losses[0] - losses[1] - losses[2])
    with tf.name_scope("accuracy"):
      tf.summary.scalar("policy accuracy", losses[3])
    tensorboard_writer_training.flush()

def save_network(checkpoint):
   global prediction_network
   print(f"Saving network ({checkpoint} steps)...")
   path = storage.save_network(checkpoint, training_network)
   print(f"Saved network ({checkpoint} steps)")
   prediction_network = update_network_for_predictions(path)

config = AlphaZeroConfig()
tensorboard_writer_training_path = os.path.join(storage.logs_path, config.run_name, "training")
tensorboard_writer_training = tf.summary.create_file_writer(tensorboard_writer_training_path)
prediction_network = prepare_predictions()
training_network = prepare_training()