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
import os
import numpy as np
import skopt.space as skspace
from skopt.utils import create_result
from scipy.special import erfinv
import matplotlib.pyplot as plt
from bask.optimizer import Optimizer
from tune.utils import expected_ucb
from tune.summary import confidence_intervals
from tune.plots import plot_objective

class Session:

  gp_burnin = 5
  gp_samples = 300
  gp_initial_burnin = 100
  gp_initial_samples = 300

  confidence_val = 0.9
  confidence_mult = erfinv(confidence_val) * np.sqrt(2)
  confidence_percent = confidence_val * 100.0

  def __init__(self):
    self.output_path = os.path.join(r"C:\Users\Public\Optimization", time.strftime("%Y%m%d-%H%M%S"))
    self.param_ranges = dict(test=skspace.Real(0.0, 10.0), test2=skspace.Real(0.0, 10.0), test3=skspace.Real(0.0, 10.0))
    self.optimizer = Optimizer(
      dimensions=list(self.param_ranges.values()),
      n_points=500,
      n_initial_points=16,
      gp_kwargs=dict(normalize_y=False, warp_inputs=True),
      acq_func="mes",
      acq_func_kwargs=dict(alpha="inf", n_thompson=20))
    self.X = []
    self.y = []
    self.noise = []

  def flip(self, score):
    return -score

  def test(self, x1, x2, x3):
    return 50.0 - ((x1 - 4)**2 + (x2 - 5)**2)

  def burnt_in(self):
    return self.optimizer.gp.chain_ is not None

  def print_results(self, writer, result_object):
    optimizer = self.optimizer
    best_point, best_value = expected_ucb(result_object, alpha=0.0)
    best_point_dict = dict(zip(self.param_ranges.keys(), best_point))
    with optimizer.gp.noise_set_to_zero():
      _, best_std = optimizer.gp.predict(
          optimizer.space.transform([best_point]), return_std=True
      )
    self.log(writer, f"Current optimum: {best_point_dict}")
    self.log(writer, f"Estimated score: {self.flip(best_value)}")
    self.log(writer, f"{self.confidence_percent}% confidence interval of score: "
      f"({np.around(self.flip(best_value) - self.confidence_mult * best_std, 4).item()}, "
      f"{np.around(self.flip(best_value) + self.confidence_mult * best_std, 4).item()})")
    confidence_out = confidence_intervals(
      optimizer=optimizer,
      param_names=list(self.param_ranges.keys()),
      hdi_prob=self.confidence_val,
      opt_samples=1000,
      multimodal=False)
    self.log(writer, f"{self.confidence_percent}% confidence intervals of the parameters:\n{confidence_out}")

  def plot_results(self, result_object, iteration):
    optimizer = self.optimizer
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
    full_plotpath = os.path.join(self.output_path, f"{timestr}-{iteration}.png")
    plt.savefig(full_plotpath, dpi=300, facecolor="#36393f")
    plt.close(fig)

  def log(self, writer, data):
    print(data)
    writer.write(data + "\n")
    writer.flush()

  # See for documentation:
  # High-level tuning loop: https://chess-tuning-tools.readthedocs.io/
  # Underlying Bayesian optimization: https://bayes-skopt.readthedocs.io/
  def run(self):
    os.makedirs(self.output_path, exist_ok=True)
    log_path = os.path.join(self.output_path, "log.txt")
    with open(log_path, "w") as writer:
      optimizer = self.optimizer
      iteration = 1
      while True:
        point = optimizer.ask()
        point_dict = dict(zip(self.param_ranges.keys(), point))
        score = self.flip(self.test(*point))
        self.log(writer, f"{iteration}: {self.flip(score)} = {point_dict}")
        error = 1.0
        gp_burnin = 5 if self.burnt_in() else 100
        gp_samples = 300
        optimizer.tell(point, score, noise_vector=error, n_samples=1, gp_samples=gp_samples, gp_burnin=gp_burnin)
        if self.burnt_in():
          result_object = create_result(Xi=self.X, yi=self.y, space=optimizer.space, models=[optimizer.gp])
          self.print_results(writer, result_object)
          if optimizer.space.n_dims > 1:
            self.plot_results(result_object, iteration)
        self.X.append(point)
        self.y.append(score)
        self.noise.append(error)
        iteration += 1