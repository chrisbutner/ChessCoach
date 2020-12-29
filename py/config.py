import toml
import platform
import os
import posixpath
import enum
import tensorflow as tf

# Duplicated from "Config.h"
class PredictionStatus(enum.IntFlag):
  Nothing = 0
  UpdatedNetwork = (1 << 0)

class Config:

  def __init__(self, is_tpu):
    self.is_tpu = is_tpu
    self.path = posixpath if is_tpu else os.path
    self.join = self.path.join
    
    try:
      config = toml.load("config.toml")
    except:
      try:
        config = toml.load("../config.toml")
      except:
        config = toml.load("/usr/local/share/ChessCoach/config.toml")

    training_network_name = config["network"]["training_network_name"]
    assert training_network_name
    
    # Make sure that the training network named is in the list of networks, and expose the name.
    training_network_config = next(c for c in config["networks"] if c["name"] == training_network_name)
    assert(training_network_config)
    self.training_network_name = training_network_name
    
    # Promote "training" and "self_play" to attributes and merge defaults and overrides.
    training_overrides = training_network_config.get("training", {})
    training_defaults = config.get("training", {})
    self.training = { **training_defaults, **training_overrides }

    self_play_overrides = training_network_config.get("self_play", {})
    self_play_defaults = config.get("self_play", {})
    self.self_play = { **self_play_defaults, **self_play_overrides }
    
    # Make some miscellaneous config available.
    self.misc = {
      "paths": config["paths"],
      "storage": config["storage"],
    }

    # Root all paths.
    self.data_root = self.determine_data_root()
    self.training["games_path_supervised"] = self.make_dir_path(self.training["games_path_supervised"])
    self.training["games_path_training"] = self.make_dir_path(self.training["games_path_training"])
    self.training["games_path_validation"] = self.make_dir_path(self.training["games_path_validation"])
    self.training["commentary_path_supervised"] = self.make_dir_path(self.training["commentary_path_supervised"])
    self.training["commentary_path_training"] = self.make_dir_path(self.training["commentary_path_training"])
    self.training["commentary_path_validation"] = self.make_dir_path(self.training["commentary_path_validation"])
    for key, value in self.misc["paths"].items():
      if not key.startswith("gcloud") and not key.startswith("strength_test"):
        self.misc["paths"][key] = self.make_dir_path(value)

  def determine_data_root(self):
    if self.is_tpu:
      bucket = self.misc["paths"]["gcloud_bucket"]
      prefix = self.misc["paths"]["gcloud_prefix"]
      self.gcloud_preamble = f"gs://{bucket}/"
      data_root = f"{self.gcloud_preamble}{prefix}"
    elif (platform.system() == "Windows"):
      data_root = self.join(os.environ["localappdata"], "ChessCoach")
    else:
      data_home = os.environ.get("XDG_DATA_HOME") or self.join(os.environ["HOME"], ".local/share")
      data_root = self.join(data_home, "ChessCoach")
    return data_root

  def make_dir_path(self, path):
    path = self.make_path(path)

    # Create directories (unless they're on gcloud storage).
    if not self.is_tpu:
      os.makedirs(path, exist_ok=True)

    return path

  def make_path(self, path):
    # These need to be backslashes on Windows for TensorFlow's recursive creation code (tf.summary.create_file_writer).
    if not self.is_tpu and (platform.system() == "Windows"):
      path = path.replace("/", "\\")
    
    # Root any relative paths at ChessCoach's appdata directory.
    if not os.path.isabs(path):
      path = self.join(self.data_root, path)

    return path

  def unmake_path(self, path):
    return self.path.relpath(path, self.data_root)

  def latest_network_path_for_type(self, network_name, network_type):
    glob = self.join(self.misc["paths"]["networks"], network_name + "_*", network_type)
    results = tf.io.gfile.glob(glob)
    return os.path.dirname(max(results)) if results else None

  def count_training_chunks(self):
    glob = self.join(self.training["games_path_training"], "*.chunk")
    return len(tf.io.gfile.glob(glob))

  def save_file(self, relative_path, data):
    path = self.make_path(relative_path)
    tf.io.gfile.GFile(path, "wb").write(data)

  def load_file(self, relative_path):
    path = self.make_path(relative_path)
    return tf.io.gfile.GFile(path, "rb").read()

  def file_exists(self, relative_path):
    path = self.make_path(relative_path)
    return tf.io.gfile.exists(path)