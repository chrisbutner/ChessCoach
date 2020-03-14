from __future__ import absolute_import, division, print_function, unicode_literals

# Suppress all TensorFlow output so it doesn't interfere with UCI.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from engine import Engine
from uci import UciManager
import network

parser = argparse.ArgumentParser()
parser.add_argument("--uci", help="start in UCI mode", action="store_true")
parser.add_argument("--train", help="start in training mode", action="store_true")
args = parser.parse_args()
print("Starting ChessCoach")
#args.uci = True # ahh need this for chessbase for now
args.train = True # while testing training
if (args.uci):
  print("Starting UCI")
  engine = Engine()
  uci = UciManager(engine)
  uci.start()
if (args.train):
  print("Starting training")
  network.alphazero()
