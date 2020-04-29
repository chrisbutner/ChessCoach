import toml

class Config(object):

  def __init__(self):
    try:
      config = toml.load("config.toml")
    except:
      config = toml.load("../config.toml")

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
    self.training_network["learning_rate_schedule"] = dict(zip(
      self.training_network["learning_rate_schedule"]["steps"],
      self.training_network["learning_rate_schedule"]["rates"]))

    self.log_next_train = False