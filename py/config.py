import toml
import platform
import os

if (platform.system() == "Windows"):
  data_root = os.path.join(os.environ["localappdata"], "ChessCoach")
else:
  data_root = os.environ.get("XDG_DATA_HOME")
  if data_root:
    data_root = os.path.join(data_root, "ChessCoach")
  else:
    data_root = os.path.join(os.environ["HOME"], ".local/share/ChessCoach")

class Config(object):

  def __init__(self):
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

    # Build the learning rate schedule dictionary.
    self.training_network["learning_rate_schedule"] = list(zip(
      self.training_network["learning_rate_schedule"]["steps"],
      self.training_network["learning_rate_schedule"]["rates"]))
    
    # Also make some miscellaneous config available.
    self.misc = {
      "paths": config["paths"]
    }

    # Root all paths.
    self.training_network["games_path_training"] = self.make_path(self.training_network["games_path_training"])
    self.training_network["games_path_validation"] = self.make_path(self.training_network["games_path_validation"])
    for key, value in self.misc["paths"].items():
      self.misc["paths"][key] = self.make_path(value)

  def make_path(self, path):
    # These need to be backslashes on Windows for TensorFlow's recursive creation code (tf.summary.create_file_writer).
    if (platform.system() == "Windows"):
      path = path.replace("/", "\\")
    
    # Root any relative paths at ChessCoach's appdata directory.
    if not os.path.isabs(path):
      path = os.path.join(data_root, path)

    # Create directories.
    os.makedirs(path, exist_ok=True)

    return path