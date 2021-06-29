import platform
import os
import time
import re
import subprocess
import threading
import tempfile
from ast import literal_eval
from skopt import Optimizer
from skopt import expected_minimum
import matplotlib.pyplot as plt
from skopt.plots import plot_objective
from config import Config, ChessCoachException
try:
  import chesscoach # See PythonModule.cpp
except:
  pass

class Session:

  n_initial_points = 16
  plot_color = "#36393f"
  log_filename = "log.txt"
  engine_names = ["Optimize", "Baseline"]

  def __init__(self, config):
    self.config = config
    self.local_output_parent = config.make_local_path(config.misc["paths"]["optimization"])
    self.local_output_child = time.strftime("%Y%m%d-%H%M%S")
    self.local_output_path = config.join(self.local_output_parent, self.local_output_child)
    os.makedirs(self.local_output_path, exist_ok=True)
    self.writer = open(self.config.join(self.local_output_path, self.log_filename), "w")
    self.parameters = self.parse_parameters(config.misc["optimization"]["parameters"])
    self.optimizer = Optimizer(list(self.parameters.values()), n_initial_points=self.n_initial_points, acq_func="EI")
    self.distributed_ip_addresses = config.misc["optimization"]["distributed_ip_addresses"]
    if self.distributed_ip_addresses:
      if self.config.misc["optimization"]["mode"] != "tournament":
        raise ChessCoachException("Distributed optimization is only implemented for tournament mode")
      # This can waste hardware if factors don't line up. You could instead find a good GCD with minimal machine waste
      # and run partial instead of complete mini-tournaments in parallel.
      self.parallelism = max(1, (len(self.distributed_ip_addresses) // 2) // self.config.misc["optimization"]["tournament_games"])
    else:
      self.parallelism = 1

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
    plt.close(plt.gcf())

  def log(self, data):
    print(data)
    writer = self.writer
    writer.write(data + "\n")
    writer.flush()

  def log_config(self):
    config = self.config
    mode = self.config.misc["optimization"]["mode"]
    self.log("################################################################################")
    self.log("Starting parameter optimization")
    self.log(f'Output path: {self.local_output_path}')
    self.log(f"Mode: {mode}")
    if mode == "epd":
      self.log(f'EPD: {config.misc["optimization"]["epd"]}')
      self.log(f'Milliseconds per position: {config.misc["optimization"]["epd_movetime_milliseconds"]}')
      self.log(f'Nodes per position: {config.misc["optimization"]["epd_nodes"]}')
      self.log(f'Failure nodes: {config.misc["optimization"]["epd_failure_nodes"]}')
      self.log(f'Position limit: {config.misc["optimization"]["epd_position_limit"]}')
    elif mode == "tournament":
      self.log(f'Games per evaluation: {config.misc["optimization"]["tournament_games"]}')
      self.log(f'Time control: {config.misc["optimization"]["tournament_time_control"]}')
    self.log("Distributed: " + (f"{len(self.distributed_ip_addresses)} IPs" if self.distributed_ip_addresses else "No"))
    self.log("Parameters:")
    for name, definition in self.parameters.items():
      self.log(f"{name}: {definition}")
    self.log("################################################################################")

  def evaluate(self, point_dicts):
    if self.config.misc["optimization"]["mode"] == "epd":
      return self.evaluate_epd(point_dicts)
    elif self.config.misc["optimization"]["mode"] == "tournament":
      return self.evaluate_tournaments(point_dicts)
    else:
      raise ChessCoachException("Unexpected optimization mode: expected 'epd' or 'tournament'")

  def evaluate_epd(self, point_dicts):
    assert len(point_dicts) == 1
    names = [name.encode("ascii") for name in point_dicts[0].keys()]
    values = [float(value) for value in point_dicts[0].values()]
    return [chesscoach.evaluate_parameters(names, values)]

  def evaluate_tournaments(self, point_dicts):
    if self.distributed_ip_addresses:
      # Play UCI proxy against UCI proxy.
      threads = []
      results = []
      ips_per_point = len(self.distributed_ip_addresses) // len(point_dicts)
      for i, point_dict in enumerate(point_dicts):
        results.append(None)
        def evaluate(i):
          results[i] = self.evaluate_tournament(point_dict, self.distributed_ip_addresses[i * ips_per_point:(i + 1) * ips_per_point])
        thread = threading.Thread(target=evaluate, args=(i,))
        thread.start()
        threads.append(thread)
      for thread in threads:
        thread.join()
      return results
    else:
      # Play ChessCoach against Stockfish locally, for TPU compatibility (only one process can grab the accelerators).
      assert len(point_dicts) == 1
      return [self.evaluate_tournament(point_dicts[0], None)]

  def evaluate_tournament(self, point_dict, ip_addresses):
    tournament_games = self.config.misc["optimization"]["tournament_games"]
    if ip_addresses:
      # Play UCI proxy against UCI proxy.
      parallel_game_count = (len(ip_addresses) // 2)
      # Play more games than necessary if numbers don't divide evenly.
      partial_length = (tournament_games + parallel_game_count - 1) // parallel_game_count
      threads = []
      pgn_paths = []
      for i in range(parallel_game_count):
        pgn_paths.append(None)
        def play(i):
          reverse_sides = ((partial_length * i) % 2 == 1)
          pgn_paths[i] = self.play_partial_tournament(point_dict, ip_addresses[i * 2:(i + 1) * 2], partial_length, reverse_sides)
        thread = threading.Thread(target=play, args=(i,))
        thread.start()
        threads.append(thread)
      for thread in threads:
        thread.join()
      return self.evaluate_elo(pgn_paths)
    else:
      # Play ChessCoach against Stockfish locally, for TPU compatibility (only one process can grab the accelerators).
      pgn_path = self.play_partial_tournament(point_dict, None, tournament_games, reverse_sides=False)
      return self.evaluate_elo([pgn_path])

  def play_partial_tournament(self, point_dict, ip_addresses, game_count, reverse_sides):
    engine_optimization_options = " ".join([f"option.{name}={value}" for name, value in point_dict.items()])
    if ip_addresses:
      # Play UCI proxy against UCI proxy.
      assert len(ip_addresses) == 2
      python = "python" if platform.system() == "Windows" else "python3"
      uci_proxy_client = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uci_proxy_client.py")
      engine_commands = [python, python]
      engine_arguments = [[uci_proxy_client, ip_address] for ip_address in ip_addresses]
      engine_options = [engine_optimization_options, ""]
    else:
      # Play ChessCoach against Stockfish locally, for TPU compatibility (only one process can grab the accelerators).
      engine_commands = ["ChessCoachUci", ("stockfish_13_win_x64_bmi2" if platform.system() == "Windows" else "stockfish_13_linux_x64_bmi2")]
      engine_arguments = [[], []]
      engine_options = [engine_optimization_options, "option.Threads=1 option.Hash=512"]
    
    # Prepare to run the mini-tournament. Use a temporary file as the PGN.
    with tempfile.NamedTemporaryFile(delete=False) as pgn:
      pgn_path = pgn.name
    time_control = self.config.misc["optimization"]["tournament_time_control"]
    
    # Run the mini-tournament and generate the PGN.
    command = "cutechess-cli "
    for i in range(2):
      command += f"-engine name={self.engine_names[i]} cmd={engine_commands[i]} "
      command += "".join([f"arg={arg} " for arg in engine_arguments[i]])
      command += f"{engine_options[i]} "
    command += f"-each proto=uci tc={time_control} timemargin=5000 dir=\"{os.getcwd()}\" "
    command += f"-games {game_count} -pgnout \"{pgn_path}\" -recover "
    if reverse_sides:
      command += "-reverse "
    subprocess.run(command, stdin=subprocess.DEVNULL, shell=True)
    return pgn_path

  def evaluate_elo(self, pgn_paths):
    if len(pgn_paths) == 1:
      pgn_path = pgn_paths[0]
    else:
      with tempfile.NamedTemporaryFile(delete=False) as pgn:
        pgn_path = pgn.name
        for partial in pgn_paths:
          with open(partial, "rb") as file:
            pgn.write(file.read())

    # Process the combined PGN using bayeselo to get an evaluation score.
    # NOTE: Bayeselo doesn't like quotes around paths.
    bayeselo_input = f"readpgn {pgn_path}\nelo\nmm\nexactdist\nratings\nx\nx\n".encode("utf-8")
    process = subprocess.run("bayeselo", input=bayeselo_input, stdout=subprocess.PIPE, shell=True)
    output = process.stdout.decode("utf-8")
    elo = int(re.search(f"{self.engine_names[0]}\\s+(-?\\d+)\\s", output).group(1))

    # Minimizing, so negate.
    return -elo

  def tell(self, starting_iteration, point_dicts, points, scores):
    optimizer = self.optimizer
    iteration = starting_iteration
    for score, point_dict in zip(scores, point_dicts):
      self.log(f"{iteration}: {score} = {point_dict}")
      iteration += 1
    while True:
      try:
        result = optimizer.tell(points, scores)
        break
      except AttributeError:
        # https://github.com/scikit-optimize/scikit-optimize/issues/981
        pass
    count_before = (starting_iteration - 1)
    count_after = (iteration - 1)
    log_interval = self.config.misc["optimization"]["log_interval"]
    if self.burnt_in(result) and (count_before // log_interval < count_after // log_interval):
      self.log_results(result)
    plot_interval = self.config.misc["optimization"]["plot_interval"]
    if self.burnt_in(result) and (optimizer.space.n_dims > 1) and (count_before // plot_interval < count_after // plot_interval):
      self.plot_results(count_after, result)

  def int_or_float(self, text):
    return float(text) if "." in text else int(text)

  def resume(self):
    starting_iteration = 1
    iteration = starting_iteration
    if self.config.misc["optimization"]["resume_latest"]:
      previous = (d for d in next(os.walk(self.local_output_parent))[1] if (d != self.local_output_child))
      latest = max(previous)
      log_path = self.config.join(self.local_output_parent, latest, self.log_filename)
      self.log(f"Replaying data from {log_path}")
      point_dicts = []
      points = []
      scores = []
      with open(log_path, "r") as reader:
        for line in reader:
          if line.startswith(f"{iteration}:"):
            score, *point = [self.int_or_float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", line)[1:]]
            point_dict = dict(zip(self.parameters.keys(), point))
            point_dicts.append(point_dict)
            points.append(point)
            scores.append(score)
            iteration += 1
      self.tell(starting_iteration, point_dicts, points, scores)
    self.log("Starting")
    return iteration

  def run(self):
    optimizer = self.optimizer
    self.log_config()
    iteration = self.resume()
    while True:
      points = optimizer.ask(n_points=self.parallelism)
      point_dicts = [dict(zip(self.parameters.keys(), point)) for point in points]
      scores = self.evaluate(point_dicts)
      self.tell(iteration, point_dicts, points, scores)
      iteration += self.parallelism

def optimize_parameters(config=None):
  if not config:
    # If config is None then we're not allowed to initialize TensorFlow, so just set "is_cloud" False.
    # It shouldn't matter, since it mostly affects cloud path use, and optimization explicitly uses local paths
    # regardless of "config.is_cloud" via "make_local_path".
    config = Config(is_cloud=False)
  Session(config).run()