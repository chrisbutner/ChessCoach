import toml
import platform
import os
import posixpath
import tensorflow as tf

class Config:

  def __init__(self, is_tpu):
    self.is_tpu = is_tpu
    self.join = posixpath.join if is_tpu else os.path.join
    
    try:
      config = toml.load("config.toml")
    except:
      try:
        config = toml.load("../config.toml")
      except:
        config = toml.load("/usr/local/share/ChessCoach/config.toml")

    training_network_name = config["network"]["training_network_name"]
    assert training_network_name
    
    # Make sure that the training network named is in the list of networks, and everything else is in place.
    training_network_config = next(c for c in config["networks"] if c["name"] == training_network_name)
    assert(training_network_config)
    overrides = training_network_config.get("training", {})
    defaults = config.get("training", {})

    # Merge the defaults and overrides. Collapse the "training" part and jam the "name" in.
    self.training_network = { **defaults, **overrides }
    self.training_network["name"] = training_network_name

    # Build the learning rate schedule dictionaries
    self.training_network["learning_rate_schedule"] = list(zip(
      self.training_network["learning_rate_schedule"]["steps"],
      self.training_network["learning_rate_schedule"]["rates"]))

    self.training_network["commentary_learning_rate_schedule"] = list(zip(
      self.training_network["commentary_learning_rate_schedule"]["steps"],
      self.training_network["commentary_learning_rate_schedule"]["rates"]))
    
    # Make some miscellaneous config available.
    self.misc = {
      "paths": config["paths"],
      "storage": config["storage"],
    }

    # Root all paths.
    self.data_root = self.determine_data_root()
    self.training_network["games_path_supervised"] = self.make_path(self.training_network["games_path_supervised"])
    self.training_network["games_path_training"] = self.make_path(self.training_network["games_path_training"])
    self.training_network["games_path_validation"] = self.make_path(self.training_network["games_path_validation"])
    self.training_network["commentary_path_supervised"] = self.make_path(self.training_network["commentary_path_supervised"])
    self.training_network["commentary_path_training"] = self.make_path(self.training_network["commentary_path_training"])
    self.training_network["commentary_path_validation"] = self.make_path(self.training_network["commentary_path_validation"])
    for key, value in self.misc["paths"].items():
      if not key.startswith("gcloud"):
        self.misc["paths"][key] = self.make_path(value)

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

  def make_path(self, path):
    # These need to be backslashes on Windows for TensorFlow's recursive creation code (tf.summary.create_file_writer).
    if not self.is_tpu and (platform.system() == "Windows"):
      path = path.replace("/", "\\")
    
    # Root any relative paths at ChessCoach's appdata directory.
    if not os.path.isabs(path):
      path = self.join(self.data_root, path)

    # Create directories (unless they're on gcloud storage).
    if not self.is_tpu:
      os.makedirs(path, exist_ok=True)

    return path

  def latest_network_path_for_type(self, network_name, network_type):
    glob = self.join(self.misc["paths"]["networks"], network_name + "*", network_type)
    results = tf.io.gfile.glob(glob)
    if results:
      return os.path.dirname(max(results))
    return None