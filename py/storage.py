import os
import struct
import numpy
import time

# These need to be backslashes on Windows for TensorFlow's recursive creation code.
networks_path = os.path.join(os.environ["localappdata"], "ChessCoach\\Training\\Networks")
tensorboard_path = os.path.join(os.environ["localappdata"], "ChessCoach\\Training\\TensorBoard")

os.makedirs(networks_path, exist_ok=True)
os.makedirs(tensorboard_path, exist_ok=True)

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