from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
import numpy
import network
import model

tf_model = tf.saved_model.load("C:\\Users\\Public\\test")
model_function = tf_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

def predict_batch(image):
  image = tf.constant(image)
  prediction = model_function(image)
  value, policy = prediction[model.OutputValueName], prediction[model.OutputPolicyName]
  value = network.map_11_to_01(numpy.array(value))
  policy = numpy.array(policy)

  return value, policy