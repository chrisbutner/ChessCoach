import network
from model import ChessCoachModel
import numpy
from profiler import Profiler

model = ChessCoachModel()
model.build()

def predict(image):
  with Profiler("predict", threshold_time=1.0):
    image = image.reshape((1, 12, 8, 8)) # TODO: Only one at a time right now, need to parallelize across MCTSs
    value, policy = model.model.predict_on_batch(image)

    value = network.map_11_to_01(numpy.array(value))
    policy = numpy.reshape(policy, (73, 8, 8)) # TODO: Only one at a time right now, need to parallelize across MCTSs

  return value, policy

def predict_batch(image):
  with Profiler("predict_batch", threshold_time=1.0):
    value, policy = numpy.array(model.model.predict_on_batch(image))
    value = network.map_11_to_01(numpy.array(value))

  return value, policy