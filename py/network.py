import math
import numpy
from scipy import special
import tensorflow as tf
from typing import List
import time
from model import ChessCoachModel
from tensorflow.keras import backend as K
import chess
import chess.pgn
from profiler import Profiler
from operator import itemgetter

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

class Node(object):

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

# TODO: Put somewhere clean
MOVE_LOOKUP_OFFSET = 64
queen_knight_planes = [1e6 for i in range(128)] # Initialize out-of-range
# Queen moves (clockwise from up)
QUEEN_MOVES = [8, 9, 1, -7, -8, -9, -1, 7]
for i in range(len(QUEEN_MOVES)):
  for distance in range(1, 8):
    queen_knight_planes[MOVE_LOOKUP_OFFSET + QUEEN_MOVES[i] * distance] = 7 * i + (distance - 1)
# Knight moves (clockwise from up)
KNIGHT_MOVES = [17, 10, -6, -15, -17, -10, 6, 15]
for i in range(len(KNIGHT_MOVES)):
  queen_knight_planes[MOVE_LOOKUP_OFFSET + KNIGHT_MOVES[i]] = 56 + i
# Underpromotions
underpromotion_planes = {
  chess.KNIGHT: {
    MOVE_LOOKUP_OFFSET + 7: 64,
    MOVE_LOOKUP_OFFSET + 8: 65,
    MOVE_LOOKUP_OFFSET + 9: 66
  },
  chess.BISHOP: {
    MOVE_LOOKUP_OFFSET + 7: 67,
    MOVE_LOOKUP_OFFSET + 8: 68,
    MOVE_LOOKUP_OFFSET + 9: 69
  },
  chess.ROOK: {
    MOVE_LOOKUP_OFFSET + 7: 70,
    MOVE_LOOKUP_OFFSET + 8: 71,
    MOVE_LOOKUP_OFFSET + 9: 72
  }
}

def get_move_index(from_square, to_square, promotion, *, rotate):
  if (rotate):
    from_square = 63 - from_square
    to_square = 63 - to_square
  move_delta_offset = MOVE_LOOKUP_OFFSET + (to_square - from_square)
  if (promotion and (promotion != chess.QUEEN)):
    plane = underpromotion_planes[promotion][move_delta_offset]
  else:
    plane = queen_knight_planes[move_delta_offset]
  return (plane, from_square // 8, from_square % 8)

class Game(object):

  def __init__(self, history=None, board=None):
    self.history = history or []
    self.board = board or chess.Board()
    self.child_visits = []
    self.num_actions = 4672  # action space size for chess; 11259 for shogi, 362 for Go
    self.chess_terminal = None
    self.chess_legal_moves = None

  def pgn(self):
    return str(chess.pgn.Game.from_board(self.board))

  def terminal(self):
    # Game specific termination rules.
    # TODO: Lots of redundant checks inside python-chess, can optimize
    if (self.chess_terminal is None):
      self.chess_terminal = self.board.is_game_over(claim_draw=True)
    return self.chess_terminal

  def terminal_value(self, to_play):
    # Game specific value.
    # TODO: Make sure that python-chess is giving 1/2-1/2, not *, for a claimed draw
    # when it's claiming up in terminal() that the game is over. I think it's giving *
    # right now. Also, make sure that the game contains the final repetition so that
    # the network learns to actually play it in a game (in an otherwising losing position)
    # or recognizes the opponent can play it in the inverse.
    result_string = self.board.result(claim_draw=True)
    result = {
      "1-0": 1,
      "0-1": 0,
      "1/2-1/2": 0.5,
      "*": 0.5
    }[result_string]
    if (to_play != 0):
      result = 1 - result
    return result

  def legal_actions(self):
    # Game specific calculation of legal actions.
    if (self.chess_legal_moves is None):
      self.chess_legal_moves = list(self.board.legal_moves)
    return self.chess_legal_moves

  def clone(self):
    return Game(list(self.history), self.board.copy())

  def apply(self, action):
    self.history.append(action)
    self.board.push(action)
    self.invalidate()

  def invalidate(self):
    self.chess_terminal = None
    self.chess_legal_moves = None

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.values())
    self.child_visits.append({action: (child.visit_count / sum_visits) for action, child in root.children.items()})
  
  def make_image(self, state_index: int):
    # Game specific feature planes.
    
    # Roll back to the "state_index" position
    switchyard = []
    for _ in range(len(self.history) - 1 - state_index):
      switchyard.append(self.board.pop())

    # If it's black's turn, rotate the board and flip colors so that the game is viewed from black's perpsective,
    # and black's pieces still go in the "self" planes 0-5.
    # TODO: Just pieces for now (12 planes)
    image = numpy.zeros((12, 8, 8))
    pieces = self.board if (state_index % 2 == 0) else self.board.mirror().transform(chess.flip_horizontal)
    for y in range(8):
      for x in range(8):
        piece = pieces.piece_at(chess.SQUARES[8 * y + x])
        if (piece is not None):
          plane = piece.piece_type - 1 + (0 if piece.color else 6)
          image[plane][y][x] = 1

    # Roll forwards to the current position
    while (switchyard):
      self.board.push(switchyard.pop())

    return image

  def make_target(self, state_index: int):
    visits = numpy.zeros((73, 8, 8))
    rotate = (state_index % 2 == 1)
    for action, logit in self.child_visits[state_index].items():
      indices = get_move_index(action.from_square, action.to_square, action.promotion, rotate=rotate)
      visits[indices] = logit
    return (map_01_to_11(self.terminal_value(state_index % 2)), visits)

  def to_play(self):
    return len(self.history) % 2


class ReplayBuffer(object):

  def __init__(self, config: AlphaZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self):
    # Sample uniformly across positions.
    move_sum = float(sum(len(g.history) for g in self.buffer))
    games = numpy.random.choice(
        self.buffer,
        size=self.batch_size,
        p=[len(g.history) / move_sum for g in self.buffer])
    game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]
    return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]

# TODO: Put all this somewhere clean
def map_01_to_11(value):
  return 2 * value - 1

def map_11_to_01(value):
  return (value + 1)/2

class UniformPolicy:

  def __getitem__(self, key):
    return 1/(8*8*73)

class ChessPolicy:
  
  def __init__(self, planes, *, rotate):
    self.planes = planes
    self.rotate = rotate

  def __getitem__(self, key):
    indices = get_move_index(key.from_square, key.to_square, key.promotion, rotate=self.rotate)
    return self.planes[indices]

class Network(object):

  def __init__(self, model=None):
    self.model = model

  # (Value, Policy)
  def inference(self, image, *, rotate):
    if (self.model is not None):
      image = image.reshape((1, 12, 8, 8)) # TODO: Only one at a time right now, need to parallelize across MCTSs
      (value, policy) = self.model.model.predict_on_batch(image)
      policy = numpy.reshape(policy, (73, 8, 8)) # TODO: Only one at a time right now, need to parallelize across MCTSs
      return (map_11_to_01(value), ChessPolicy(policy, rotate=rotate))
    return (0.5, UniformPolicy()) # policy -> uniform, value -> 0.5

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


# # AlphaZero training is split into two independent parts: Network training and
# # self-play data generation.
# # These two parts only communicate by transferring the latest network checkpoint
# # from the training to the self-play, and the finished games from the self-play
# # to the training.
# def alphazero(config: AlphaZeroConfig):
#   storage = SharedStorage()
#   replay_buffer = ReplayBuffer(config)

#   for i in range(config.num_actors):
#     launch_job(run_selfplay, config, storage, replay_buffer)

#   train_network(config, storage, replay_buffer)

#   return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network):
  game = Game()
  while not game.terminal() and len(game.history) < config.max_moves:
    with Profiler("MCTS", threshold_time=0.0):
      action, root = run_mcts(config, game, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: Game, network: Network):
  root = Node(0)
  evaluate(root, game, network)
  add_exploration_noise(config, root)

  for _ in range(config.num_simulations):
    #with Profiler("Expand"):
    node = root
    scratch_game = game.clone()
    search_path = [node]

    while node.expanded():
      #with Profiler("Select child"):
      action, node = select_child(config, node)
      #with Profiler("Apply"):
      scratch_game.apply(action)
      search_path.append(node)

    value = evaluate(node, scratch_game, network)
    backpropagate(search_path, value, scratch_game.to_play())

  return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: Game, root: Node):
  visit_counts = [(child.visit_count, action)
                  for action, child in root.children.items()]
  if len(game.history) < config.num_sampling_moves:
    _, action = softmax_sample(visit_counts) # over the expits
  else:
    _, action = max([(visit_count, action) for (visit_count, action) in visit_counts], key=itemgetter(0))
  return action

def softmax_sample(visit_counts):
  index = numpy.random.choice(len(visit_counts), p=special.softmax([visit_count for visit_count, action in visit_counts]))
  return visit_counts[index]

# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
  _, action, child = max([(ucb_score(config, node, child), action, child)
                         for action, child in node.children.items()], key=itemgetter(0))
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
  #with Profiler("Evaluate"):
  policy_actions = game.legal_actions()
  node.to_play = game.to_play()

  # Score terminal positions mid-MCTS (TODO: probably doesn't check for "claim" draws)
  if (len(policy_actions) == 0):
    node.to_play = game.to_play()
    return game.terminal_value(game.to_play())

  # Expand the node.
  value, policy_logits = network.inference(game.make_image(-1), rotate=(node.to_play % 2 == 1))
  policy_values = special.softmax([policy_logits[a] for a in policy_actions])
  for action, p in zip(policy_actions, policy_values):
    node.children[action] = Node(p)
  return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
  actions = node.children.keys()
  noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########

def train_network(config: AlphaZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  model = ChessCoachModel()
  model.build()
  optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate_schedule[0], momentum=config.momentum)
  losses = ["mean_squared_error", tf.keras.losses.CategoricalCrossentropy(from_logits=True)]
  model.model.compile(optimizer=optimizer, loss=losses, metrics=["accuracy"])
  network = Network(model)
  
  for i in range(config.training_steps):

    # Until threading is done properly, self-play here.
    
    # Generate 5 games
    for _ in range(5):
      with Profiler("Game", threshold_time=0.0):
        play_network = storage.latest_network()
        game = play_game(config, play_network)
        replay_buffer.save_game(game)

    # Train
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


######### End Training ###########
##################################

# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero():
  config = AlphaZeroConfig()
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)

  train_network(config, storage, replay_buffer)