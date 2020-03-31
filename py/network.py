from __future__ import absolute_import, division, print_function, unicode_literals

import math
import numpy
import time
import os

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.keras import backend as K

from model import ChessCoachModel, OutputValueName, OutputPolicyName
from profiler import Profiler
import storage
import game

##########################
####### Helpers ##########

class AlphaZeroConfig(object):

  def __init__(self):
    ### Self-Play
    self.num_actors = 5000

    self.num_sampling_moves = 30
    self.max_moves = 512  # for chess and shogi, 722 for Go.
    self.num_simulations = 800

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    ### Training
    self.batch_size = 2048 # OOM on GTX 1080 @ 4096
    self.chesscoach_training_factor = 4096 / self.batch_size # Increase training to compensate for lower batch size.
    self.training_steps = int(700e3 * self.chesscoach_training_factor)
    self.checkpoint_interval = 10 #int(1e3) # Currently training about 10x as slowly as AlphaZero, but self-play 100x+, so reduce accordingly.
    self.window_size = int(1e6)

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.chesscoach_slowdown_factor = 50
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
    losses = ["mean_squared_error", tf.keras.losses.CategoricalCrossentropy(from_logits=True)]
    self.model.compile(optimizer=optimizer, loss=losses)

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
      values = numpy.full((len(image)), 0.5, dtype=numpy.float32)
      self.latest_values = values
    if ((policies is None) or (len(image) != len(policies))):
      policies = numpy.zeros((len(image), 73, 8, 8), dtype=numpy.float32)
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
    value, policy = prediction[OutputValueName], prediction[OutputPolicyName]
    value = game.map_11_to_01(numpy.array(value))
    policy = numpy.array(policy)
    return value, policy

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

  losses = training_network.model.train_on_batch(images, [values, policies])
  print(f"Loss: {str(losses[0]).rjust(10)} (Value: {str(losses[1]).rjust(10)}, Policy: {str(losses[2]).rjust(10)})")

def save_network(checkpoint):
  print(f"Saving network ({checkpoint} steps)...")
  path = storage.save_network(checkpoint, training_network)
  print(f"Saved network ({checkpoint} steps)")
  update_network_for_predictions(path)

config = AlphaZeroConfig()
prediction_network = prepare_predictions()
training_network = prepare_training()