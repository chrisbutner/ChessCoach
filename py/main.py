from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import storage
import os

parser = argparse.ArgumentParser()
args = parser.parse_args()

print("Starting Python-ChessCoach")
game = storage.load_game(os.path.join(storage.games_path, "game_000000111"))
print(game.pgn())
