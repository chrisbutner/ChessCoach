import watchdog.observers
import watchdog.events
import os
import struct
import numpy

def map_01_to_11(value):
  return 2.0 * value - 1.0

def map_11_to_01(value):
  return (value + 1.0)/2.0

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
    return self.images[state_index]

  def make_target(self, state_index: int):
    flip = ((state_index % 2) == 1)
    value = map_01_to_11((1 - self.terminal_value) if flip else self.terminal_value)
    policy = self.policies[state_index]
    return value, policy

class Watcher(watchdog.events.FileSystemEventHandler):

  def __init__(self, path, handler):
    self.handler = handler
    self.observer = watchdog.observers.Observer()
    self.observer.schedule(self, path, recursive=False)
    self.observer.start()

  def on_created(self, event):
    if (not event.is_directory):
      self.handler(event.src_path)

# There's some slight dead time between loading and watching. Don't worry about it.
def load_and_watch(path, handler):
  _, _, files = next(os.walk(path))
  for file in files:
    handler(os.path.join(path, file))
  Watcher(path, handler)

def load_game(path):
  with open(path, mode="rb") as f:
    data = f.read()

  version, terminal_value, move_count = struct.unpack_from("ifi", data)
  assert version == 1
  assert move_count >= 1

  move_offset = 12
  moves = numpy.frombuffer(data, dtype=numpy.int32, count=move_count, offset=move_offset)

  images_offset = move_offset + move_count * 4
  images_count = move_count * 12 * 8 * 8
  images = numpy.frombuffer(data, dtype=numpy.float32, count=images_count, offset=images_offset)
  images.shape = (move_count, 12, 8, 8)

  policies_offset = images_offset + images_count * 4
  policies_count = move_count * 73 * 8 * 8
  policies = numpy.frombuffer(data, dtype=numpy.float32, count=policies_count, offset=policies_offset)
  policies.shape = (move_count, 73, 8, 8)

  return Game(terminal_value, moves, images, policies)

def load_and_watch_games(game_handler):
  games_path = os.path.join(os.environ["localappdata"], "ChessCoach/Training/Games")
  load_and_watch(games_path, lambda path: game_handler(load_game(path)))

def save_network(step, network):
  # TODO
  pass