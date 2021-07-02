# ChessCoach, a neural network-based chess engine capable of natural-language commentary
# Copyright 2021 Chris Butner
#
# ChessCoach is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ChessCoach is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

import platform
import os
import time
import re
import subprocess
import threading
import tempfile
import socket
from ast import literal_eval
from skopt import Optimizer
from skopt import expected_minimum
import matplotlib.pyplot as plt
from skopt.plots import plot_objective
from config import Config, ChessCoachException
import uci_proxy_client
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
    self.distributed_hosts = config.misc["optimization"]["distributed_hosts"]
    if self.distributed_hosts:
      if self.config.misc["optimization"]["mode"] != "tournament":
        raise ChessCoachException("Distributed optimization is only implemented for tournament mode")
      # This can waste hardware if factors don't line up. You could instead find a good GCD with minimal machine waste
      # and run partial instead of complete mini-tournaments in parallel.
      ip_addresses = self.get_ip_addresses() # There may be more IP addresses than "hosts"; e.g., with TPU pods we can use multiple workers.
      self.ip_address_count = len(ip_addresses)
      self.parallelism = max(1, (self.ip_address_count // 2) // self.config.misc["optimization"]["tournament_games"])
    else:
      self.ip_address_count = None
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
    self.log("Distributed: " + (f"{self.ip_address_count} IPs" if self.distributed_hosts else "No"))
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

  def get_lookup(self):
    zone = self.config.misc["optimization"]["distributed_zone"]
    command = f'gcloud alpha compute tpus tpu-vm list --format="table[no-heading](name,networkEndpoints.ipAddress)" --zone {zone}'
    process = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
    output = process.stdout.decode("utf-8")
    lookup = {}
    for line in output.splitlines():
      name, ip_addresses = line.split(None, 1)
      lookup[name] = literal_eval(ip_addresses)
    return lookup

  def get_ip_addresses(self):
    ip_addresses = []
    while True:
      lookup = self.get_lookup()
      for host in self.distributed_hosts:
        try:
          ip_addresses += lookup[host]
        except:
          print(f"Failed to look up IP address(es) for host '{host}'; retrying after 5 minutes")
          time.sleep(5 * 60)
          ip_addresses = []
          break # break for, continue while
      if ip_addresses:
        threads = []
        ready = []
        for i, ip_address in enumerate(ip_addresses):
          ready.append(None)
          def check(i):
            ready[i] = self.ready_check(ip_address)
          thread = threading.Thread(target=check, args=(i,))
          thread.start()
          threads.append(thread)
        for thread in threads:
          thread.join()
        if any(not r for r in ready):
          print(f"Failed to ready-check IP address(es): {ready}; retrying after 5 minutes")
          time.sleep(5 * 60)
          ip_addresses = []
          continue # continue while
        return ip_addresses

  def ready_check(self, ip_address):
    timeout = 120
    start = time.time()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
      try:
        connection.settimeout(timeout)
        connection.connect((ip_address, uci_proxy_client.PORT))
        connection.sendall(b"stop\nisready\n")
        output = ""
        while (time.time() - start) < timeout:
          data = connection.recv(uci_proxy_client.BUFFER_SIZE)
          output += data.decode("utf-8")
          if "readyok" in output:
            return True
      except:
        pass
      return False

  def evaluate_tournaments(self, point_dicts):
    if self.distributed_hosts:
      # Play UCI proxy against UCI proxy.
      while True:
        threads = []
        results = []
        ip_addresses = self.get_ip_addresses()
        ips_per_point = len(ip_addresses) // len(point_dicts)
        for i, point_dict in enumerate(point_dicts):
          results.append(None)
          def evaluate(i):
            results[i] = self.evaluate_tournament(point_dict, ip_addresses[i * ips_per_point:(i + 1) * ips_per_point])
          thread = threading.Thread(target=evaluate, args=(i,))
          thread.start()
          threads.append(thread)
        for thread in threads:
          thread.join()
        if any(result is None for result in results):
          print(f"Parameter evaluation failed: {results}; retrying after 5 minutes")
          time.sleep(5 * 60)
          continue
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
      client = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uci_proxy_client.py")
      engine_commands = [python, python]
      engine_arguments = [[client, ip_address] for ip_address in ip_addresses]
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
    command += f"-games {game_count} -pgnout \"{pgn_path}\" -wait 1000 -recover "
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
    process = subprocess.run("bayeselo", input=bayeselo_input, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    output = process.stdout.decode("utf-8")

    # Require all games finishing properly (we may still miss some cases where cutechess-cli adjudicates a result).
    tournament_games = self.config.misc["optimization"]["tournament_games"]
    loaded_count = int(re.search(f"(\\d+) game\\(s\\) loaded", output).group(1)) # Printed to stderr
    if loaded_count < tournament_games:
      print(f"Failed to evaluate Elo: {loaded_count} games < {tournament_games} required")
      return None

    # Grab Elo from the output. Minimizing, so negate.
    elo = int(re.search(f"{self.engine_names[0]}\\s+(-?\\d+)\\s", output).group(1))
    return -elo

  def tell(self, starting_iteration, point_dicts, points, scores):
    optimizer = self.optimizer
    iteration = starting_iteration
    for score, point_dict in zip(scores, point_dicts):
      self.log(f"{iteration}: {score} = {point_dict}")
      iteration += 1
    result = optimizer.tell(points, scores)
    count_before = (starting_iteration - 1)
    count_after = (iteration - 1)
    log_interval = self.config.misc["optimization"]["log_interval"]
    if self.burnt_in(result) and (count_before // log_interval < count_after // log_interval):
      self.log_results(result)
    plot_interval = self.config.misc["optimization"]["plot_interval"]
    if self.burnt_in(result) and (optimizer.space.n_dims > 1) and (count_before // plot_interval < count_after // plot_interval):
      self.plot_results(count_after, result)

  def int_or_float(self, text):
    return float(text) if ("." in text or "e" in text) else int(text)

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
            score, *point = [self.int_or_float(n) for n in re.findall(r"-?\d[\d\-.e]*", line)[1:]]
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