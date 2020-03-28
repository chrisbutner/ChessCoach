import watchdog.observers
import watchdog.events
import os
import struct
import numpy
import time

from game import Game

# These need to be backslashes on Windows for TensorFlow's recursive creation code.
games_path = os.path.join(os.environ["localappdata"], "ChessCoach\\Training\\Games")
networks_path = os.path.join(os.environ["localappdata"], "ChessCoach\\Training\\Networks")

os.makedirs(games_path, exist_ok=True)
os.makedirs(networks_path, exist_ok=True)

class Watcher(watchdog.events.FileSystemEventHandler):

  def __init__(self, path, handler, files=True, directories=False):
    self.handler = handler
    self.files = files
    self.directories = directories
    self.observer = watchdog.observers.Observer()
    self.observer.schedule(self, path, recursive=False)
    self.observer.start()

  def on_created(self, event):
    if ((not event.is_directory and self.files) or (event.is_directory and self.directories)):
      self.handler(event.src_path)

def load_game(path):
  while True:
    try:
      with open(path, mode="rb") as f:
        data = f.read()
      break
    except:
      time.sleep(0.001)

  version, move_count, result = struct.unpack_from("HHf", data)
  assert version == 1
  assert move_count >= 1

  move_offset = 8
  moves = numpy.frombuffer(data, dtype=numpy.int16, count=move_count, offset=move_offset)

  # Ignore child_visits

  return Game(result, moves)

# There's some slight dead time between loading and watching. Don't worry about it for our use.
def load_and_watch_networks(network_path_handler):
  parent_path = networks_path
  _, directories, _ = next(os.walk(parent_path))
  for directory in reversed(directories): # Only load the latest.
    network_path_handler(os.path.join(parent_path, directory))
    break # Only load the latest.
  Watcher(parent_path, network_path_handler, files=False, directories=True) # But still load each new one.

def save_network(step, network):
  directory_name = "network_" + str(step).zfill(9)
  network.model.save(os.path.join(networks_path, directory_name), include_optimizer=False, save_format="tf")