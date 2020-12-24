import os

def test_train(teacher):
  import network
  from model import ModelBuilder
  import numpy
  batch_size = network.config.training_network["batch_size"]
  images = numpy.zeros((batch_size, ModelBuilder.input_planes_count), dtype=numpy.int64)
  values = numpy.full((batch_size, 1), 0.123, dtype=numpy.float32)
  mcts_values = numpy.full((batch_size, 1), 0.1234, dtype=numpy.float32)
  policies = numpy.zeros((batch_size, ModelBuilder.output_planes_count, ModelBuilder.board_side, ModelBuilder.board_side), dtype=numpy.float32)
  for i in range(batch_size):
    policies[i][1][2][3] = 1.0

  #network.config.training_network["validation_interval"] = 1
  network.load_network(network.config.training_network["name"])
  train_batch_method = network.train_batch_teacher if teacher else network.train_batch_student
  train_batch_method(step=1, images=images, values=values, mcts_values=mcts_values, policies=policies)

def test_train_commentary():
  import network
  from model import ModelBuilder
  import numpy
  batch_size = network.config.training_network["commentary_batch_size"]
  images = numpy.zeros((batch_size, ModelBuilder.input_planes_count), dtype=numpy.int64)
  comments = [b"What a great move"] * batch_size

  #network.config.training_network["validation_interval"] = 1
  network.load_network(network.config.training_network["name"])
  network.train_commentary_batch(step=1, images=images, comments=comments)

def test_predict(teacher):
  import network
  from model import ModelBuilder
  import numpy
  batch_size = 16
  images = numpy.zeros((batch_size, ModelBuilder.input_planes_count), dtype=numpy.int64)

  network.load_network(network.config.training_network["name"])
  predict_batch_method = network.predict_batch_teacher if teacher else network.predict_batch_student
  predict_batch_method(images)

def test_predict_commentary():
  import network
  from model import ModelBuilder
  import numpy
  batch_size = 16
  images = numpy.zeros((batch_size, ModelBuilder.input_planes_count), dtype=numpy.int64)

  network.load_network(network.config.training_network["name"])
  network.predict_commentary_batch(images)

def create_save_training_network(teacher):
  import network
  from model import ModelBuilder

  network.networks.name = network.config.training_network["name"]
  save_network_method = network.save_network_teacher if teacher else network.save_network_student
  save_network_method(0)

print("Starting Python-ChessCoach")
test_train(teacher=True)
#test_train(teacher=False)
#test_train_commentary()
#test_predict(teacher=True)
#test_predict_commentary()
#create_save_training_network(teacher=True)
