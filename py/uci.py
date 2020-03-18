import sys
import logging
from functools import partial

# if the engine or the GUI receives an unknown command or token it should just ignore it and try to parse the rest of the string in this line.
# if the engine receives a command which is not supposed to come, for example "stop" when the engine is not calculating, it should also just ignore it.

class UciManager:
  def __init__(self, engine):
    self.engine = engine
    self.command_map = {
      "uci": self.command_uci, # TODO: Eventually extract this out into "passive" stdin listener once other modes fleshed out (i.e. remove --uci arg)
      "debug": self.command_debug,
      "isready": self.command_isready,
      "setoption": self.command_setoption,
      "register": self.command_register,
      "ucinewgame": self.command_ucinewgame,
      "position": self.command_position,
      "go": self.command_go,
      "stop": self.command_stop,
      "ponderhit": self.command_ponderhit,
      "quit": self.command_quit,
    }
    self.quit = False
    self.debug = False
    self.haveReceivedUciNewGame = False

  def start(self):
    while (not self.quit):
      args = input().split()
      while (len(args) > 0):
        command = self.command_map.get(args[0])
        args = args[1:]
        if (command is not None):
          command(*args)
          break
    
  def command_uci(self, *args):
    print("id name ChessCoach")
    print("id author C. Butner, T. Li")
    # TODO: options
    print("uciok")
    sys.stdout.flush()

  def command_debug(self, *args):
    if (len(args) >= 1):
      if (args[0] == "on"):
        self.debug = True
      elif (args[0] == "off"):
        self.debug = False

  def command_isready(self, *args):
    self.engine.initialize()
    print("readyok")
    sys.stdout.flush()

  def command_setoption(self, *args):
    pass

  def command_register(self, *args):
    pass

  def command_ucinewgame(self, *args):
    self.haveReceivedNewGame = True
    self.engine.reset()

  def command_position(self, *args):
    if (not self.haveReceivedUciNewGame):
      # This GUI doesn't send "ucinewgame", so synthesize it when changing positions.
      self.engine.reset()
    
    # The FEN record is always 6 fields.
    if ((len(args) >= 7) and (args[0] == "fen")):
      self.engine.set_position_fen(" ".join(args[1:7]))
      args = args[7:]
    else:
      self.engine.set_position_starting()
      args = args[1:]

    if ((len(args) >= 2) and (args[0] == "moves")):
      self.engine.play_moves(args[1:])
    
  def command_go(self, *args):
    # TODO: Need to implement this properly, but for now just immediately play a random move.
    best = self.engine.get_random_move()
    print("bestmove ", best)

  def command_stop(self, *args):
    self.engine.stop()
    # TODO: Deal with bestmove situations

  def command_ponderhit(self, *args):
    # TODO: Need to implement ponder
    pass

  def command_quit(self, *args):
    self.engine.stop()
    self.quit = True