import numpy as np
np.random.seed(1234)
import platform
import os
import time
import numpy as np
import re
import subprocess
from ast import literal_eval
from skopt import Optimizer
from skopt import expected_minimum
import matplotlib.pyplot as plt
from skopt.plots import plot_objective
try:
  import chesscoach
except:
  pass

class Session:

  n_initial_points = 16
  plot_color = "#36393f"

  def __init__(self, config):
    self.config = config
    self.local_output_path = config.join(config.make_local_path(config.misc["paths"]["optimization"]), time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(self.local_output_path, exist_ok=True)
    self.tournament_pgn_path = config.join(self.local_output_path, "optimization.pgn")
    self.writer = open(self.config.join(self.local_output_path, "log.txt"), "w")
    self.parameters = self.parse_parameters(config.misc["optimization"]["parameters"])
    self.optimizer = Optimizer(list(self.parameters.values()), n_initial_points=self.n_initial_points, acq_func="EI")

  def parse_parameters(self, parameters_raw):
    return dict((name, literal_eval(definition)) for name, definition in parameters_raw.items())

  def burnt_in(self, result):
    return len(result.x_iters) >= self.n_initial_points

  def log_results(self, result):
    best_point = result.x
    best_value = result.fun
    best_point_dict = dict(zip(self.parameters.keys(), best_point))
    expected_best_point, expected_best_value = expected_minimum(result)
    expected_best_point_dict = dict(zip(self.parameters.keys(), expected_best_point))
    self.log(f"Found best: {best_value} = {best_point_dict}")
    self.log(f"Expected best: {expected_best_value} = {expected_best_point_dict}")

  def plot_results(self, iteration, result):
    plt.style.use("dark_background")
    ax = plot_objective(result, dimensions=list(self.parameters.keys()))
    for i in range(len(ax)):
      for j in range(len(ax[0])):
          ax[i, j].set_facecolor(self.plot_color)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    full_plotpath = self.config.join(self.local_output_path, f"{timestr}-{iteration}.png")
    plt.gcf().patch.set_facecolor(self.plot_color)
    plt.savefig(full_plotpath, dpi=300, facecolor=self.plot_color, bbox_inches="tight")

  def log(self, data):
    print(data)
    writer = self.writer
    writer.write(data + "\n")
    writer.flush()

  def log_config(self):
    config = self.config
    self.log("################################################################################")
    self.log("Starting parameter optimization")
    self.log(f'Output path: {self.local_output_path}')
    self.log(f'EPD: {config.misc["optimization"]["epd"]}')
    self.log(f'Nodes: {config.misc["optimization"]["nodes"]}')
    self.log(f'Failure nodes: {config.misc["optimization"]["failure_nodes"]}')
    self.log(f'Position limit: {config.misc["optimization"]["position_limit"]}')
    self.log("Parameters:")
    for name, definition in self.parameters.items():
      self.log(f"{name}: {definition}")
    self.log("################################################################################")

  def evaluate(self, point_dict):
    names = list(point_dict.keys())
    values = list(point_dict.values())
    if self.config.misc["optimization"]["mode"] == "epd":
      return self.evaluate_epd(names, values)
    elif self.config.misc["optimization"]["mode"] == "tournament":
      return self.evaluate_tournament(names, values)
    else:
      raise Exception("Unknown optimization mode: expected 'epd' or 'tournament'")

  def evaluate_epd(self, names, values):
    names = [name.encode("ascii") for name in names]
    values = [float(value) for value in values]
    return chesscoach.evaluate_parameters(names, values)

  def evaluate_tournament(self, names, values):
    if platform.system() != "Windows":
      raise Exception("Tournament-based optimization currently only supported on Windows: complications with alpha TPUs")

    # Make sure that cutechess-cli doesn't append to an existing pgn.
    try:
      os.remove(self.tournament_pgn_path)
    except:
      pass

    name_optimize = "ChessCoach_Optimize"
    name_baseline = "ChessCoach_Baseline"

    tournament_games = self.config.misc["optimization"]["tournament_games"]
    seconds_per_move = self.config.misc["optimization"]["tournament_movetime_milliseconds"] / 1000.0
    
    # Run the mini-tournament and generate optimization.pgn.
    command = f"cutechess-cli.exe -engine name={name_optimize} cmd=ChessCoachUci.exe "
    for name, value in zip(names, values):
      command += f"option.{name}={value} "
    command += f"-engine name={name_baseline} cmd=ChessCoachUci.exe -each proto=uci st={seconds_per_move} timemargin=1000 dir=\"{os.getcwd()}\" "
    command += f"-games {tournament_games} -pgnout \"{self.tournament_pgn_path}\""
    subprocess.run(command, stdin=subprocess.DEVNULL, shell=True)

    # Process optimization.pgn using bayeselo to get an evaluation score.
    # NOTE: Bayeselo doesn't like quotes around paths.
    bayeselo_input = f"readpgn {self.tournament_pgn_path}\nelo\nmm\nexactdist\nratings\nx\nx\n".encode("utf-8")
    process = subprocess.run("bayeselo.exe", input=bayeselo_input, capture_output=True, shell=True)
    output = process.stdout.decode("utf-8")
    elo = int(re.search(f"{name_optimize}\\s+(-?\\d+)\\s", output).group(1))

    # Minimizing, so negate.
    return -elo

  def tell(self, iteration, point_dict, point, score):
    optimizer = self.optimizer
    self.log(f"{iteration}: {score} = {point_dict}")
    while True:
      try:
        result = optimizer.tell(point, score)
        break
      except AttributeError:
        # https://github.com/scikit-optimize/scikit-optimize/issues/981
        pass
    if self.burnt_in(result) and (iteration % self.config.misc["optimization"]["log_interval"] == 0):
      self.log_results(result)
    if self.burnt_in(result) and (optimizer.space.n_dims > 1) and (iteration % self.config.misc["optimization"]["plot_interval"] == 0):
      self.plot_results(iteration, result)

  def resume(self):
    iteration = 1
    log_path = None
    if log_path:
      self.log(f"Replaying data from {log_path}")
      with open(log_path, "r") as reader:
        for line in reader:
          if line.startswith(f"{iteration}:"):
            score, *point = [float(n) for n in re.findall(r"\d+.\d+", line)]
            point_dict = dict(zip(self.parameters.keys(), point))
            self.tell(iteration, point_dict, point, score)
            iteration += 1
    self.log("Starting")
    return iteration

  def run(self):
    optimizer = self.optimizer
    self.log_config()
    iteration = self.resume()
    while True:
      point = optimizer.ask()
      point_dict = dict(zip(self.parameters.keys(), point))
      score = self.evaluate(point_dict)
      self.tell(iteration, point_dict, point, score)
      iteration += 1