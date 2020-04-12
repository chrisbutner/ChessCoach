import os
import struct
import numpy
import time

from game import Game

# These need to be backslashes on Windows for TensorFlow's recursive creation code.
games_path = os.path.join(os.environ["localappdata"], "ChessCoach\\Training\\Games\\Train")
networks_path = os.path.join(os.environ["localappdata"], "ChessCoach\\Training\\Networks")
logs_path = os.path.join(os.environ["localappdata"], "ChessCoach\\Training\\Logs")

os.makedirs(games_path, exist_ok=True)
os.makedirs(networks_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)

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

def load_latest_network(network_path_handler):
  parent_path = networks_path
  _, directories, _ = next(os.walk(parent_path))
  for directory in reversed(directories): # Only load the latest.
    return network_path_handler(os.path.join(parent_path, directory))
  return None

def save_network(step, network):
  directory_name = "network_" + str(step).zfill(9)
  path = os.path.join(networks_path, directory_name)
  network.model.save(path, include_optimizer=True, save_format="tf")
  return path