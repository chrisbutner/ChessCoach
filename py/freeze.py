import tensorflow as tf
import tensorflow.python.tools
from tensorflow.keras import backend as K
from model import ChessCoachModel
from profiler import Profiler
import numpy

def test():
  K.set_learning_phase(0)
  image = numpy.ones((1, 12, 8, 8), dtype="float32")

  # (1)
  # model = tf.keras.models.load_model("C:\\Users\\Public\\test", compile=False)
  # value, policy = model.predict_on_batch(image)
  # with Profiler("test", threshold_time=0.0):
  #   for i in range(1000):
  #     with Profiler("predict"):
  #       value, policy = model.predict_on_batch(image)

  # (2)
  #from tensorflow.python.saved_model import signature_constants
  #model = tf.saved_model.load("C:\\Users\\Public\\test")
  #function = model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  #image = tf.constant(image)
  #value, policy = function(image)
  #with Profiler("test", threshold_time=0.0):
  #  for i in range(1000):
  #    with Profiler("predict"):
  #      value, policy = function(image)

  # (3)
  # from tensorflow.python.framework import convert_to_constants
  # from tensorflow.python.saved_model import signature_constants
  # # model = tf.saved_model.load("C:\\Users\\Public\\test")
  # # function = model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  # # frozen_func = convert_to_constants.convert_variables_to_constants_v2(function)
  # # # print("Frozen model inputs: ")
  # # # print(frozen_func.inputs)
  # # # print("Frozen model outputs: ")
  # # # print(frozen_func.outputs)
  # # tf.io.write_graph(frozen_func.graph, "C:\\Users\\Public\\frozen", name="stillfrozen", as_text=False)
  # with tf.io.gfile.GFile("C:\\Users\\Public\\frozen\\stillfrozen", "rb") as f:
  #   graph_def = tf.compat.v1.GraphDef()
  #   loaded = graph_def.ParseFromString(f.read())
  # def wrap_frozen_graph(graph_def, inputs, outputs):
  #   def _imports_graph_def():
  #     tf.compat.v1.import_graph_def(graph_def, name="")
  #   wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
  #   import_graph = wrapped_import.graph
  #   return wrapped_import.prune(
  #       tf.nest.map_structure(import_graph.as_graph_element, inputs),
  #       tf.nest.map_structure(import_graph.as_graph_element, outputs))
  # prediction_func = wrap_frozen_graph(graph_def, ["input:0"], ["Identity:0", "Identity_1:0"])
  # image = tf.constant(image)
  # value, policy = prediction_func(image)
  # with Profiler("test", threshold_time=0.0):
  #   for i in range(1000):
  #     with Profiler("predict"):
  #       value, policy = prediction_func(image)

  # (4)
  # from tensorflow.python.framework import convert_to_constants
  # from tensorflow.python.saved_model import signature_constants
  # from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config
  # # model = tf.saved_model.load("C:\\Users\\Public\\test")
  # # function = model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  # # frozen_func = convert_to_constants.convert_variables_to_constants_v2(function)
  # # graph_def = run_graph_optimizations(
  # #   frozen_func.graph.as_graph_def(),
  # #   [t for t in frozen_func.inputs if (t.dtype != tf.resource)], # TODO, what's the issue here
  # #   frozen_func.outputs,
  # #   config=get_grappler_config([
  # #     "pruning",
  # #     "function",
  # #     "debug_stripper",
  # #     "constfold",
  # #     "shape",
  # #     "remapper",
  # #     "arithmetic",
  # #     "layout",
  # #     # skip memory, may favor space over time?
  # #     "loop",
  # #     "dependency"
  # #     ]),
  # #   graph=frozen_func.graph)
  # # # print("Frozen model inputs: ")
  # # # print(frozen_func.inputs)
  # # # print("Frozen model outputs: ")
  # # # print(frozen_func.outputs)
  # # tf.io.write_graph(frozen_func.graph, "C:\\Users\\Public\\frozen", name="grappler", as_text=False)
  # with tf.io.gfile.GFile("C:\\Users\\Public\\frozen\\grappler", "rb") as f:
  #   graph_def = tf.compat.v1.GraphDef()
  #   loaded = graph_def.ParseFromString(f.read())
  # def wrap_frozen_graph(graph_def, inputs, outputs):
  #   def _imports_graph_def():
  #     tf.compat.v1.import_graph_def(graph_def, name="")
  #   wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
  #   import_graph = wrapped_import.graph
  #   return wrapped_import.prune(
  #       tf.nest.map_structure(import_graph.as_graph_element, inputs),
  #       tf.nest.map_structure(import_graph.as_graph_element, outputs))
  # prediction_func = wrap_frozen_graph(graph_def, ["input:0"], ["Identity:0", "Identity_1:0"])
  # image = tf.constant(image)
  # value, policy = prediction_func(image)
  # with Profiler("test", threshold_time=0.0):
  #   for i in range(1000):
  #     with Profiler("predict"):
  #       value, policy = prediction_func(image)

  # (5)
  # from tensorflow.python.framework import convert_to_constants
  # from tensorflow.python.saved_model import signature_constants
  # from tensorflow.python.tools import optimize_for_inference_lib
  # # model = tf.saved_model.load("C:\\Users\\Public\\test")
  # # function = model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  # # frozen_func = convert_to_constants.convert_variables_to_constants_v2(function)
  # # # print("Frozen model inputs: ")
  # # # print(frozen_func.inputs)
  # # # print("Frozen model outputs: ")
  # # # print(frozen_func.outputs)
  # # optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(frozen_func.graph.as_graph_def(),
  # #   ["input"], ["Identity", "Identity_1"], tf.float32.as_datatype_enum)
  # # tf.io.write_graph(optimized_graph_def, "C:\\Users\\Public\\frozen", name="optimized", as_text=False)
  # with tf.io.gfile.GFile("C:\\Users\\Public\\frozen\\optimized", "rb") as f:
  #   graph_def = tf.compat.v1.GraphDef()
  #   loaded = graph_def.ParseFromString(f.read())
  # def wrap_frozen_graph(graph_def, inputs, outputs):
  #   def _imports_graph_def():
  #     tf.compat.v1.import_graph_def(graph_def, name="")
  #   wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
  #   import_graph = wrapped_import.graph
  #   return wrapped_import.prune(
  #       tf.nest.map_structure(import_graph.as_graph_element, inputs),
  #       tf.nest.map_structure(import_graph.as_graph_element, outputs))
  # prediction_func = wrap_frozen_graph(graph_def, ["input:0"], ["Identity:0", "Identity_1:0"])
  # image = tf.constant(image)
  # value, policy = prediction_func(image)
  # with Profiler("test", threshold_time=0.0):
  #   for i in range(1000):
  #     with Profiler("predict"):
  #       value, policy = prediction_func(image)

  # (6)
  # from tensorflow.python.framework import convert_to_constants
  # from tensorflow.python.saved_model import signature_constants
  # #from tensorflow.python.tools import optimize_for_inference_lib
  # from graph_transforms import TransformGraph
  # # model = tf.saved_model.load("C:\\Users\\Public\\test")
  # # function = model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  # # frozen_func = convert_to_constants.convert_variables_to_constants_v2(function)
  # # # print("Frozen model inputs: ")
  # # # print(frozen_func.inputs)
  # # # print("Frozen model outputs: ")
  # # # print(frozen_func.outputs)
  # # #optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(frozen_func.graph.as_graph_def(),
  # #   #["input"], ["Identity", "Identity_1"], tf.float32.as_datatype_enum)
  # # transforms = [
  # #   "remove_nodes(op=Identity, op=CheckNumerics)", 
  # #   "merge_duplicate_nodes",
  # #   "strip_unused_nodes",
  # #   "fold_constants(ignore_errors=true)",
  # #   "fold_batch_norms",
  # #   "fold_old_batch_norms"
  # # ]
  # # transform_graph_optimized_graph_def = TransformGraph(frozen_func.graph.as_graph_def(), [], ["Identity", "Identity_1"], transforms)
  # # tf.io.write_graph(transform_graph_optimized_graph_def, "C:\\Users\\Public\\frozen", name="transform_graph_optimized", as_text=False)
  # with tf.io.gfile.GFile("C:\\Users\\Public\\frozen\\transform_graph_optimized", "rb") as f:
  #   graph_def = tf.compat.v1.GraphDef()
  #   loaded = graph_def.ParseFromString(f.read())
  # def wrap_frozen_graph(graph_def, inputs, outputs):
  #   def _imports_graph_def():
  #     tf.compat.v1.import_graph_def(graph_def, name="")
  #   wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
  #   import_graph = wrapped_import.graph
  #   return wrapped_import.prune(
  #       tf.nest.map_structure(import_graph.as_graph_element, inputs),
  #       tf.nest.map_structure(import_graph.as_graph_element, outputs))
  # prediction_func = wrap_frozen_graph(graph_def, ["input:0"], ["Identity:0", "Identity_1:0"])
  # image = tf.constant(image)
  # value, policy = prediction_func(image)
  # with Profiler("test", threshold_time=0.0):
  #   for i in range(1000):
  #     with Profiler("predict"):
  #       value, policy = prediction_func(image)
