# Apache Software License 2.0
#
# Copyright (c) 2020, Karlson Pfannschmidt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
##############################################################################
#
# Modifications by Chris Butner, 2021.
#
# Based on https://github.com/kiudee/chess-tuning-tools/blob/master/tune/cli.py

import time
import numpy as np
import skopt.space as skspace
from skopt.utils import create_result
from scipy.special import erfinv
import matplotlib.pyplot as plt
from bask.optimizer import Optimizer
from tune.io import parse_ranges
from tune.utils import expected_ucb
from tune.summary import confidence_intervals
from tune.plots import plot_objective
try:
  import chesscoach
except:
  pass

class Session:

  confidence_val = 0.9
  confidence_mult = erfinv(confidence_val) * np.sqrt(2)
  confidence_percent = confidence_val * 100.0

  def __init__(self, config):
    self.config = config
    self.output_path = config.join(config.misc["paths"]["optimization"], time.strftime("%Y%m%d-%H%M%S"))
    self.param_ranges = parse_ranges(config.misc["optimization"]["parameters"])
    self.optimizer = Optimizer(
      dimensions=list(self.param_ranges.values()),
      n_points=500,
      n_initial_points=16,
      gp_kwargs=dict(normalize_y=False, warp_inputs=True),
      acq_func="mes",
      acq_func_kwargs=dict(alpha="inf", n_thompson=20))
    self.X = []
    self.y = []

  def burnt_in(self):
    return self.optimizer.gp.chain_ is not None

  def log_results(self, writer):
    optimizer = self.optimizer
    result_object = create_result(Xi=self.X, yi=self.y, space=optimizer.space, models=[optimizer.gp])
    best_point, best_value = expected_ucb(result_object, alpha=0.0)
    best_point_dict = dict(zip(self.param_ranges.keys(), best_point))
    with optimizer.gp.noise_set_to_zero():
      _, best_std = optimizer.gp.predict(
          optimizer.space.transform([best_point]), return_std=True
      )
    self.log(writer, f"\nCurrent optimum: {best_point_dict}")
    self.log(writer, f"Estimated score: {best_value} (lower is better)")
    self.log(writer, f"{self.confidence_percent}% confidence interval of score: "
      f"({np.around(best_value - self.confidence_mult * best_std, 4).item()}, "
      f"{np.around(best_value + self.confidence_mult * best_std, 4).item()})")
    confidence_out = confidence_intervals(
      optimizer=optimizer,
      param_names=list(self.param_ranges.keys()),
      hdi_prob=self.confidence_val,
      opt_samples=1000,
      multimodal=False)
    self.log(writer, f"{self.confidence_percent}% confidence intervals of parameters:\n{confidence_out}")

  def plot_results(self, iteration):
    optimizer = self.optimizer
    result_object = create_result(Xi=self.X, yi=self.y, space=optimizer.space, models=[optimizer.gp])
    plt.style.use("dark_background")
    fig, ax = plt.subplots(
        nrows=optimizer.space.n_dims,
        ncols=optimizer.space.n_dims,
        figsize=(3 * optimizer.space.n_dims, 3 * optimizer.space.n_dims),
    )
    fig.patch.set_facecolor("#36393f")
    for i in range(optimizer.space.n_dims):
        for j in range(optimizer.space.n_dims):
            ax[i, j].set_facecolor("#36393f")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plot_objective(result_object, dimensions=list(self.param_ranges.keys()), fig=fig, ax=ax)
    full_plotpath = self.config.join(self.output_path, f"{timestr}-{iteration}.png")
    plt.savefig(full_plotpath, dpi=300, facecolor="#36393f")
    plt.close(fig)

  def log(self, writer, data):
    print(data)
    writer.write(data + "\n")
    writer.flush()

  def log_config(self, writer):
    config = self.config
    self.log(writer, "################################################################################")
    self.log(writer, "Starting parameter optimization")
    self.log(writer, f'EPD: {config.misc["optimization"]["epd"]}')
    self.log(writer, f'Nodes: {config.misc["optimization"]["nodes"]}')
    self.log(writer, f'Failure nodes: {config.misc["optimization"]["failure_nodes"]}')
    self.log(writer, f'Position limit: {config.misc["optimization"]["position_limit"]}')
    self.log(writer, "Parameters:")
    for name, definition in self.param_ranges.items():
      self.log(writer, f"{name}: {definition}")
    self.log(writer, "################################################################################")

  def evaluate(self, point_dict):
    names = list(n.encode("ascii") for n in point_dict.keys())
    values = list(point_dict.values())
    return chesscoach.evaluate_parameters(names, values)

  # See for documentation:
  # High-level tuning loop: https://chess-tuning-tools.readthedocs.io/
  # Underlying Bayesian optimization: https://bayes-skopt.readthedocs.io/
  def run(self):
    self.config.make_dirs(self.output_path)
    log_path = self.config.join(self.output_path, "log.txt")
    with open(log_path, "w") as writer:
      self.log_config(writer)
      optimizer = self.optimizer
      iteration = 1
      while True:
        point = optimizer.ask()
        point_dict = dict(zip(self.param_ranges.keys(), point))
        score, error = self.evaluate(point_dict)
        self.log(writer, f"{iteration}: {score} = {point_dict}")
        gp_burnin = 5 if self.burnt_in() else 100
        gp_samples = 300
        optimizer.tell(point, score, noise_vector=error, n_samples=1, gp_samples=gp_samples, gp_burnin=gp_burnin)
        if self.burnt_in() and (iteration % self.config.misc["optimization"]["log_interval"] == 0):
          self.log_results(writer)
        if self.burnt_in() and (optimizer.space.n_dims > 1) and (iteration % self.config.misc["optimization"]["plot_interval"] == 0):
          self.plot_results(iteration)
        self.X.append(point)
        self.y.append(score)
        iteration += 1