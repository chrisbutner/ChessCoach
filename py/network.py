from __future__ import absolute_import, division, print_function, unicode_literals

import math
import numpy
import time
import pickle
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.keras import backend as K
from model import ChessCoachModel, OutputValueName, OutputPolicyName
from profiler import Profiler
#import chess
#import chess.pgn


##########################
####### Helpers ##########

def map_01_to_11(value):
  return 2.0 * value - 1.0

def map_11_to_01(value):
  return (value + 1.0)/2.0

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
    self.training_steps = int(700e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = 2048 # OOM on GTX 1080 @ 4096

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        100e3: 2e-2,
        300e3: 2e-3,
        500e3: 2e-4
    }

class Game(object):

  def __init__(self, terminal_value, moves, images, policies):
    self.terminal_value = terminal_value
    self.moves = moves
    self.images = images
    self.policies = policies

  def pgn(self):
    # TODO
    # return str(chess.pgn.Game.from_board(self.board))
    pass

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return self.images[state_index]

  def make_target(self, state_index: int):
    flip = ((state_index % 2) == 1)
    value = map_01_to_11((1 - self.terminal_value) if flip else self.terminal_value)
    policy = self.policies[state_index]
    return value, policy

class ReplayBuffer(object):

  def __init__(self, config: AlphaZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0) # TODO: Don't do this
    self.buffer.append(game)

  def sample_batch(self):
    # Wait until there are some games.
    while (len(self.buffer) < 1):
      time.sleep(5)

    # Sample uniformly across positions.
    move_sum = float(sum(len(g.moves) for g in self.buffer))
    games = numpy.random.choice(
        self.buffer,
        size=self.batch_size,
        p=[len(g.moves) / move_sum for g in self.buffer])
    game_pos = [(g, numpy.random.randint(len(g.moves))) for g in games]
    return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]

class Network(object):

  def __init__(self, model=None):
    self.model = model

class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      return Network()  # policy -> uniform, value -> 0.5

  def save_network(self, step: int, network: Network):
    self._networks[step] = network

##### End Helpers ########
##########################

storage = SharedStorage()
config = AlphaZeroConfig()

# TODO: I know this is temp, but really need to not serialize the config
try:
  with open("C:\\Users\\Public\\replay_buffer", "rb") as f:
    replay_buffer = pickle.load(f)
except Exception as e:
  print(e)
  replay_buffer = ReplayBuffer(config)

tf_model = tf.saved_model.load("C:\\Users\\Public\\test")
model_function = tf_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

def predict_batch(image):
  image = tf.constant(image)
  prediction = model_function(image)
  value, policy = prediction[OutputValueName], prediction[OutputPolicyName]
  value = map_11_to_01(numpy.array(value))
  policy = numpy.array(policy)

  return value, policy

def submit(terminal_value, moves, images, policies):
  replay_buffer.save_game(Game(terminal_value, moves.copy(), images.copy(), policies.copy()))
  if (len(replay_buffer.buffer) % 10 == 0):
    with open("C:\\Users\\Public\\replay_buffer", "wb") as f:
      pickle.dump(replay_buffer, f)

def train():
  #print("Sleeping")
  #time.sleep(50)
  print("Training")
  model = ChessCoachModel()
  model.build()
  optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate_schedule[0], momentum=config.momentum)
  losses = ["mean_squared_error", tf.keras.losses.CategoricalCrossentropy(from_logits=True)]
  model.model.compile(optimizer=optimizer, loss=losses, metrics=["accuracy"])
  network = Network(model)
  
  for i in range(config.training_steps):
    with Profiler("Train", threshold_time=0.0):
      if i % config.checkpoint_interval == 0:
        storage.save_network(i, network)
      batch = replay_buffer.sample_batch()

      x = tf.stack([image for image, (target_value, target_policy) in batch])
      y_value = tf.stack([target_value for image, (target_value, target_policy) in batch])
      y_policy = tf.stack([target_policy for image, (target_value, target_policy) in batch])

      new_learning_rate = config.learning_rate_schedule.get(i)
      if (new_learning_rate is not None):
        K.set_value(optimizer.lr, new_learning_rate)

      network.model.model.train_on_batch(x, [y_value, y_policy])

  storage.save_network(config.training_steps, network)
