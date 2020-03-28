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

  def __init__(self, result, moves):
    self.result = result
    self.moves = moves

  def pgn(self):
    game = chess.pgn.Game()
    node = game
    for move in self.moves:
      node = node.add_variation(StockfishMove(move).to_python_chess())
    game.headers["Result"] = map_01_to_pgn(self.result)
    return str(game)

