import network
from model import ChessCoachModel
import numpy
from profiler import Profiler

model = ChessCoachModel()
model.build()

def predict(image):
  with Profiler("Prediction", threshold_time=1.0):
    image = image.reshape((1, 12, 8, 8)) # TODO: Only one at a time right now, need to parallelize across MCTSs
    value, policy = model.model.predict_on_batch(image)

    value = numpy.reshape(network.map_11_to_01(value), (1,1))
    policy = numpy.reshape(policy, (73, 8, 8)) # TODO: Only one at a time right now, need to parallelize across MCTSs

  return value, policy