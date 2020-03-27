import chess
import chess.pgn

def map_01_to_11(value):
  return 2.0 * value - 1.0

def map_11_to_01(value):
  return (value + 1.0)/2.0

def map_01_to_pgn(value):
  return {
    1.0: "1-0",
    0.0: "0-1",
    0.5: "1/2-1/2"
  }[value]

class StockfishMove:

  def __init__(self, move):
    self.move = move

  def from_square(self):
    return (self.move >> 6) & 0x3F

  def to_square(self):
    return (self.move & 0x3F)

  def promotion(self):
    if ((self.move & (3 << 14)) == (1 << 14)):
      return (((self.move >> 12) & 3) + chess.KNIGHT)
    return None

  def to_python_chess(self):
    return chess.Move(self.from_square(), self.to_square(), self.promotion())

class Game(object):

  def __init__(self, terminal_value, moves, images, policies):
    self.terminal_value = self.flip_value(len(moves), terminal_value)
    self.moves = moves
    self.images = images
    self.policies = policies

  def pgn(self):
    game = chess.pgn.Game()
    #game.setup()
    node = game
    for move in self.moves:
      node = node.add_variation(StockfishMove(move).to_python_chess())
    game.headers["Result"] = map_01_to_pgn(self.terminal_value)
    return str(game)

  def make_image(self, state_index: int):
    return self.images[state_index]

  def make_target(self, state_index: int):
    value = map_01_to_11(self.flip_value(state_index, self.terminal_value))
    policy = self.policies[state_index]
    return value, policy

  def flip_value(self, state_index, value):
    return value if ((state_index % 2) == 0) else (1.0 - value)

