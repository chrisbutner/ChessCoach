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
    self.training_factor = 4096 / self.batch_size # Increase training to compensate for lower batch size.
    self.training_steps = int(700e3 * self.training_factor)
    self.checkpoint_interval = 100 #int(1e3) # Currently training about 100x as slowly as AlphaZero, so reduce accordingly.
    self.window_size = int(1e6)

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        int(100e3 * self.training_factor): 2e-2,
        int(300e3 * self.training_factor): 2e-3,
        int(500e3 * self.training_factor): 2e-4
    }

class Network(object):

  def predict_batch(self, image):
    assert False

class KerasNetwork(Network):

  def __init__(self):
    self.model = ChessCoachModel().build()

  def predict_batch(self, image):
    assert False

class UniformNetwork(Network):

  def __init__(self):
    self.latest_values = None
    self.latest_policies = None

  def predict_batch(self, image):
    if ((self.latest_values is None) or (len(image) != len(self.latest_values))):
      self.latest_values = numpy.full((len(image)), 0.5)
      self.latest_policies = numpy.zeros((len(image), 73, 8, 8))
    return self.latest_values, self.latest_policies

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

config = AlphaZeroConfig()
latest_network = None
training_network = None

def update_network(network_path):
  name = os.path.basename(os.path.normpath(network_path))
  print(f"Loading network: {name}...")
  global latest_network
  while True:
    try:
      tf_model = tf.saved_model.load(network_path)
      break
    except:
      time.sleep(0.001)
  latest_network = TensorFlowNetwork(tf_model)
  print(f"Loaded network: {name}")

def prepare_predictions():
  global latest_network
  if (not latest_network):
    storage.load_and_watch_networks(update_network)
  if (not latest_network):
    latest_network = UniformNetwork()
    print("Loaded uniform network")

def prepare_training():
  global training_network
  if (not training_network):
    training_network = KerasNetwork()
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate_schedule[0], momentum=config.momentum)
    losses = ["mean_squared_error", tf.keras.losses.CategoricalCrossentropy(from_logits=True)]
    training_network.model.compile(optimizer=optimizer, loss=losses)

def predict_batch(image):
  prepare_predictions()
  return latest_network.predict_batch(image)

def train_batch(step, images, values, policies):
  prepare_training()

  new_learning_rate = config.learning_rate_schedule.get(step)
  if (new_learning_rate is not None):
    K.set_value(training_network.model.optimizer.lr, new_learning_rate)

  training_network.model.train_on_batch(images, [values, policies])

def save_network(checkpoint):
  print(f"Saving network ({checkpoint} steps)...")
  storage.save_network(checkpoint, training_network)
  print(f"Saved network ({checkpoint} steps)")