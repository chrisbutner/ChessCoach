from __future__ import absolute_import, division, print_function, unicode_literals

import math
import numpy
import time
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.keras import backend as K
from model import ChessCoachModel, OutputValueName, OutputPolicyName
from profiler import Profiler
import storage
#import chess
#import chess.pgn

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
    self.training_steps = int(700e3)
    self.checkpoint_interval = 10 #int(1e3) # Currently training about 100x as slowly as AlphaZero, so reduce accordingly.
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

class ReplayBuffer(object):

  def __init__(self, config: AlphaZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def add_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0) # TODO: Don't do this
    self.buffer.append(game)

  def sample_batch(self):
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

##### End Helpers ########
##########################

config = AlphaZeroConfig()
replay_buffer = ReplayBuffer(config)

storage.load_and_watch_games(replay_buffer.add_game)

tf_model = tf.saved_model.load("C:\\Users\\Public\\test")
model_function = tf_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

def predict_batch(image):
  image = tf.constant(image)
  prediction = model_function(image)
  value, policy = prediction[OutputValueName], prediction[OutputPolicyName]
  value = storage.map_11_to_01(numpy.array(value))
  policy = numpy.array(policy)

  return value, policy

def train():
  if (len(replay_buffer.buffer) < 1):
    print("Sleeping until games are found")
    while (len(replay_buffer.buffer) < 1):
      time.sleep(1)
  print("Training")
  model = ChessCoachModel()
  model.build()
  optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate_schedule[0], momentum=config.momentum)
  losses = ["mean_squared_error", tf.keras.losses.CategoricalCrossentropy(from_logits=True)]
  model.model.compile(optimizer=optimizer, loss=losses, metrics=["accuracy"])
  network = Network(model)
  
  for i in range(config.training_steps):
    with Profiler("Train", threshold_time=60.0):
      if (i > 0) and (i % config.checkpoint_interval == 0):
        print(f"Saving network ({i} steps)")
        storage.save_network(i, network)
      batch = replay_buffer.sample_batch()

      x = tf.stack([image for image, (target_value, target_policy) in batch])
      y_value = tf.stack([target_value for image, (target_value, target_policy) in batch])
      y_policy = tf.stack([target_policy for image, (target_value, target_policy) in batch])

      new_learning_rate = config.learning_rate_schedule.get(i)
      if (new_learning_rate is not None):
        K.set_value(optimizer.lr, new_learning_rate)

      network.model.model.train_on_batch(x, [y_value, y_policy])

  print(f"Saving network ({config.training_steps} steps)")
  storage.save_network(config.training_steps, network)
