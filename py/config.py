import toml
import platform
import os
import posixpath
import enum
import subprocess

# Duplicated from "Config.h"
class PredictionStatus(enum.IntFlag):
  Nothing = 0
  UpdatedNetwork = (1 << 0)

class Config:

  def __init__(self, is_tpu):
    self.is_tpu = is_tpu
    self.path = posixpath if is_tpu else os.path
    self.join = self.path.join
    
    py_path = os.path.dirname(os.path.abspath(__file__))
    try:
      # Slightly less risk to try this first on Windows before parent directory.
      config = toml.load(self.join(py_path, "config.toml")) # Windows, via C++
    except:
      try:
        # Running source directly in Python. Prefer this more specific path even if also installed on Linux.
        config = toml.load(self.join(py_path, "..", "config.toml"))
      except:
        config = toml.load("/usr/local/share/ChessCoach/config.toml") # Linux, via C++

    self.network_name = config["network"]["network_name"]
    assert self.network_name
    self.role = config["network"]["role"]
    assert self.role
    
    # Make sure that the network named is in the list of networks, and expose the name.
    network_config = next((c for c in config["networks"] if c["name"] == self.network_name), None)
    if not network_config:
      raise Exception(f'Network name "{self.network_name}" not found in config')
    
    # Promote "training" and "self_play" to attributes and merge defaults and overrides.
    training_overrides = network_config.get("training", {})
    training_defaults = config.get("training", {})
    self.training = { **training_defaults, **training_overrides }

    self_play_overrides = network_config.get("self_play", {})
    self_play_defaults = config.get("self_play", {})
    self.self_play = { **self_play_defaults, **self_play_overrides }
    
    # Make some miscellaneous config available.
    self.misc = {
      "paths": config["paths"],
      "storage": config["storage"],
      "optimization": config["optimization"],
    }

    # Root all paths.
    self.data_root = self.determine_data_root()
    self.training["games_path_training"] = self.make_dir_path(self.training["games_path_training"])
    self.training["games_path_validation"] = self.make_dir_path(self.training["games_path_validation"])
    self.training["commentary_path"] = self.make_dir_path(self.training["commentary_path"])
    for key, value in self.misc["paths"].items():
      if not key.startswith("tpu") and not key.startswith("strength_test"):
        self.misc["paths"][key] = self.make_dir_path(value)

    # Copy locally any files that need to be memory-mapped.
    self.replicate_locally(self.misc["paths"]["syzygy"])

  def determine_data_root(self):
    if self.is_tpu:
      # Replace tilde with the user's home directory, if present.
      return self.path.expanduser(self.misc["paths"]["tpu_data_root"])
    else:
      return self.determine_local_data_root()

  def determine_local_data_root(self):
    if (platform.system() == "Windows"):
      return self.join(os.environ["localappdata"], "ChessCoach")
    else:
      data_home = os.environ.get("XDG_DATA_HOME") or self.join(os.environ["HOME"], ".local/share")
      return self.join(data_home, "ChessCoach")

  def make_dir_path(self, path):
    path = self.make_path(path)
    # Try to create directories. This may fail for gs:// paths.
    try:
      os.makedirs(path, exist_ok=True)
    except:
      pass
    return path
  
  def make_path(self, path):
    # These need to be backslashes on Windows for TensorFlow's recursive creation code (tf.summary.create_file_writer).
    if not self.is_tpu and (platform.system() == "Windows"):
      path = path.replace("/", "\\")

    # Just in case this specific path is special-cased as absolute rather than relative, starting with tilde.
    path = self.path.expanduser(path)
    
    # Root any relative paths at the data root.
    if not self.path.isabs(path):
      path = self.join(self.data_root, path)

    return path

  def unmake_path(self, path):
    return self.path.relpath(path, self.data_root)

  def make_local_path(self, path):
    return self.join(self.determine_local_data_root(), self.unmake_path(path))

  def latest_model_paths_for_pattern_type_model(self, network_pattern, network_type, model_type):
    import tensorflow as tf
    glob = self.join(self.misc["paths"]["networks"], network_pattern, network_type, model_type)
    results = [self.join(result, "weights") for result in tf.io.gfile.glob(glob)]
    return glob, results

  def make_model_path(self, network_type, model_type, checkpoint):
    return self.join(self.misc["paths"]["networks"], f"{self.network_name}_{str(checkpoint).zfill(9)}", network_type, model_type, "weights")

  # This is brittle, but necessary and low-churn. Update code in sync with "StageTypeLookup" in Config.cpp.
  def is_swa_for_network_type(self, network_type):
    return any(stage["stage"] == "save_swa" and stage.get("target", None) == network_type for stage in self.training["stages"])

  def count_training_chunks(self):
    import tensorflow as tf
    glob = self.join(self.training["games_path_training"], "*.chunk")
    return len(tf.io.gfile.glob(glob))

  def save_file(self, relative_path, data):
    import tensorflow as tf
    path = self.make_path(relative_path)
    tf.io.gfile.GFile(path, "wb").write(data)

  def load_file(self, relative_path):
    import tensorflow as tf
    path = self.make_path(relative_path)
    return tf.io.gfile.GFile(path, "rb").read()

  def file_exists(self, relative_path):
    import tensorflow as tf
    path = self.make_path(relative_path)
    return tf.io.gfile.exists(path)

  def replicate_locally(self, path):
    assert path.startswith(self.data_root)
    local_path = self.make_local_path(path)
    if path == local_path:
      # Already local
      return
    if self.path.exists(local_path) and any(os.listdir(local_path)):
      # Already replicated
      return
    os.makedirs(local_path, exist_ok=True)
    idempotent_source = self.join(path, "*")
    process = subprocess.run(f'gsutil -m cp -r "{idempotent_source}" "{local_path}"', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    process.check_returncode()
    
