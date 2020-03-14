#from model import ChessCoachModel
import chess
import numpy

class Engine:

  def __init__(self):
    #self.model = None
    self.board = None

  def initialize(self):
    # if (self.model is None):
    #   self.model = ChessCoachModel()
    #   self.model.build()
    pass

  def reset(self):
    # We need to create a new Board when setting the position anyway, so just do nothing here.
    pass

  def set_position_fen(self, fen):
    self.board = chess.Board(fen)

  def set_position_starting(self):
    self.board = chess.Board()

  def play_moves(self, moves):
    for move in moves:
      self.board.push(chess.Move.from_uci(move))
  
  def get_random_move(self):
    legal = list(self.board.legal_moves)
    choice = numpy.random.choice(legal)
    return choice.uci()

  def stop(self):
    pass