from __future__ import absolute_import, division, print_function, unicode_literals

import storage
import os

def test_train():
  import network
  from model import ChessCoachModel
  import numpy
  batch_size = 2
  images = numpy.zeros((batch_size, ChessCoachModel.input_planes_count, ChessCoachModel.board_side, ChessCoachModel.board_side), dtype=numpy.float32)
  values = numpy.full((batch_size, 1), 0.123, dtype=numpy.float32)
  policies = numpy.zeros((batch_size, ChessCoachModel.output_planes_count, ChessCoachModel.board_side, ChessCoachModel.board_side), dtype=numpy.float32)
  for i in range(batch_size):
    policies[i][1][2][3] = 1.0

  network.config.training_network["validation_interval"] = 1
  network.load_network(network.config.training_network["name"])
  network.train_batch(step=1, images=images, values=values, policies=policies)

print("Starting Python-ChessCoach")
#test_train()

from attention import MultiHeadSelfAttention2D
import tensorflow as tf
import numpy

total_depth = 128
num_heads = 4
board_side = 8

# (B, D-in, H, W)
images = tf.constant(numpy.ones((7, 101, board_side, board_side), dtype=numpy.float32))
# (B, D-model, H, W)
x = tf.keras.layers.Conv2D(filters=total_depth, kernel_size=(3, 3), padding="same", data_format="channels_first")(images)
x = tf.transpose(x, [0, 2, 3, 1]) # channels first->last

test = MultiHeadSelfAttention2D(total_depth, num_heads, 1e-4, "blah")
test2 = test(x)
print(test2)