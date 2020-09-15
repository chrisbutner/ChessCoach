import math
import numpy
import time
import os

silent = bool(os.environ.get("CHESSCOACH_SILENT"))
if silent:
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.keras import backend as K

from config import Config
from model import ChessCoachModel
import storage
import transformer

K.set_image_data_format("channels_first")

def log(*args):
  if not silent:
    print(*args)

# This fixes an issue with categorical_crossentropy calculating incorrectly
# over our 73*8*8 output planes - loss ends up way too small.
def flat_categorical_crossentropy_from_logits(y_true, y_pred):
  return tf.keras.losses.categorical_crossentropy(y_true=K.batch_flatten(y_true), y_pred=K.batch_flatten(y_pred), from_logits=True)

# This fixes the same issue with categorical_accuracy. No point doing softmax on logits for argmax.
def flat_categorical_accuracy(y_true, y_pred):
  return tf.keras.metrics.categorical_accuracy(y_true=K.batch_flatten(y_true), y_pred=K.batch_flatten(y_pred))

class Network(object):

  def predict_batch(self, images):
    assert False

class KerasNetwork(Network):

  def __init__(self, model=None, model_commentary_decoder=None, commentary_tokenizer=None):
    self.model = model or ChessCoachModel().build(config)
    self.model_play = ChessCoachModel().subset_play(self.model)
    self.model_commentary_encoder = ChessCoachModel().subset_commentary_encoder(self.model)
    if not model_commentary_decoder or not commentary_tokenizer:
      self.model_commentary_decoder, self.commentary_tokenizer = ChessCoachModel().build_commentary_decoder(config)
    else:
      self.model_commentary_decoder = model_commentary_decoder
      self.commentary_tokenizer  = commentary_tokenizer
    
    optimizer = tf.keras.optimizers.SGD(
      learning_rate=get_learning_rate(config.training_network["learning_rate_schedule"], 0),
      momentum=config.training_network["momentum"])
    losses = ["mean_squared_error", "mean_squared_error", flat_categorical_crossentropy_from_logits, flat_categorical_crossentropy_from_logits]
    loss_weights = [config.training_network["value_loss_weight"], config.training_network["mcts_value_loss_weight"],
      config.training_network["policy_loss_weight"], config.training_network["reply_policy_loss_weight"]]
    metrics = [[], [], [flat_categorical_accuracy], [flat_categorical_accuracy]]
    self.model_play.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

  def predict_batch(self, images):
    assert False

class UniformNetwork(Network):

  def __init__(self):
    self.latest_values = None
    self.latest_policies = None

  def predict_batch(self, images):
    # Check both separately because of threading
    values = self.latest_values
    policies = self.latest_policies
    if (values is None) or (len(images) != len(values)):
      values = numpy.full((len(images)), 0.0, dtype=numpy.float32)
      self.latest_values = values
    if (policies is None) or (len(images) != len(policies)):
      policies = numpy.zeros((len(images), ChessCoachModel.output_planes_count, ChessCoachModel.board_side, ChessCoachModel.board_side), dtype=numpy.float32)
      self.latest_policies = policies
    return values, policies

class TensorFlowNetwork(Network):

  # NEED to pass the model in THEN extract the function, otherwise all hell breaks loose.
  def __init__(self, model):
    self.model = model
    self.function = self.model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

  def predict_batch(self, images):
    images = tf.constant(images)
    prediction = self.function(images)
    value, policy = prediction[ChessCoachModel.output_value_name], prediction[ChessCoachModel.output_policy_name]
    return numpy.array(value), numpy.array(policy)

class Networks:

  def __init__(self):
    self.network_name = "network"
    self.prediction_network = None
    self.training_network = None

def update_network_for_predictions(network_path):
  name = os.path.basename(os.path.normpath(network_path))
  log(f"Loading network (predictions): {name}...")
  while True:
    try:
      tf_model = tf.saved_model.load(storage.model_path(network_path))
      break
    except Exception as e:
      log("Exception:", e)
      time.sleep(0.25)
  network = TensorFlowNetwork(tf_model)
  log(f"Loaded network (predictions): {name}")
  return network

def update_network_for_training(network_path):
  name = os.path.basename(os.path.normpath(network_path))
  log(f"Loading network (training): {name}...")
  while True:
    try:
      # Don't serialize optimizer: custom loss/metrics.
      model = tf.keras.models.load_model(storage.model_path(network_path), custom_objects={
        "flat_categorical_crossentropy_from_logits": flat_categorical_crossentropy_from_logits,
        "flat_categorical_accuracy": flat_categorical_accuracy,
      }, compile=False)
      with open(storage.commentary_tokenizer_path(network_path), 'r') as f:
        commentary_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
      model_commentary_decoder, _ = ChessCoachModel().build_commentary_decoder(config, commentary_tokenizer)
      model_commentary_decoder.load_weights(storage.model_commentary_decoder_path(network_path))
      break
    except Exception as e:
      log("Exception:", e)
      time.sleep(0.25)
  network = KerasNetwork(model, model_commentary_decoder, commentary_tokenizer)
  log(f"Loaded network (training): {name}")
  return network

def ensure_training():
  if not networks.training_network:
    networks.training_network = storage.load_latest_network(config, networks.network_name, update_network_for_training)
  if not networks.training_network:
    log("Creating new network (training)")
    networks.training_network = KerasNetwork()

def get_learning_rate(schedule, step):
  rate = 0.0
  for key, value in schedule:
    if step >= key:
      rate = value
    else:
      break
  return rate

def predict_batch(images):
  return networks.prediction_network.predict_batch(images)

def predict_commentary_batch(images):
  # Use training network for now
  ensure_training()

  encoder = networks.training_network.model_commentary_encoder
  decoder = networks.training_network.model_commentary_decoder
  tokenizer = networks.training_network.commentary_tokenizer

  start_token = tokenizer.word_index[ChessCoachModel.token_start]
  end_token = tokenizer.word_index[ChessCoachModel.token_end]
  max_length = ChessCoachModel.transformer_max_length

  sequences = transformer.predict_greedy(encoder, decoder,
    start_token, end_token, max_length, images)

  def trim_start_end_tokens(sequence):
    for i, token in enumerate(sequence):
      if (token == end_token):
        return sequence[1:i]
    return sequence[1:]

  sequences = [trim_start_end_tokens(s) for s in sequences.numpy()]
  comments = tokenizer.sequences_to_texts(sequences)
  comments = numpy.array([c.encode("utf-8") for c in comments])
  return comments

def train_batch(step, images, values, mcts_values, policies, reply_policies):
  ensure_training()
  learning_rate = get_learning_rate(config.training_network["learning_rate_schedule"], step)
  K.set_value(networks.training_network.model_play.optimizer.lr, learning_rate)

  do_log_training = ((step % config.training_network["validation_interval"]) == 0)
  if do_log_training:
    log_training_prepare(step)
  losses = networks.training_network.model_play.train_on_batch(images, [values, mcts_values, policies, reply_policies])
  if do_log_training:
    log_training("training", tensorboard_writer_training, step, losses)

def validate_batch(step, images, values, mcts_values, policies, reply_policies):
  ensure_training()
  log_training_prepare(step)
  losses = networks.training_network.model_play.test_on_batch(images, [values, mcts_values, policies, reply_policies])
  log_training("validation", tensorboard_writer_validation, step, losses)

def train_commentary_batch(step, images, comments):
  ensure_training()
  learning_rate = get_learning_rate(config.training_network["learning_rate_schedule"], step)
  
  # TODO: Set LR if adam->SGD_momentum
  #K.set_value(commentary_optimizer.lr, learning_rate)

  do_log_training = ((step % config.training_network["validation_interval"]) == 0)
  comments = [f"{ChessCoachModel.token_start} {c.decode('utf-8')} {ChessCoachModel.token_end}" for c in comments]
  comments = networks.training_network.commentary_tokenizer.texts_to_sequences(comments)
  comments = tf.keras.preprocessing.sequence.pad_sequences(comments, padding="post")
  losses = transformer.train_step(
    networks.training_network.model_commentary_encoder,
    networks.training_network.model_commentary_decoder,
    images,
    comments)
  if do_log_training:
    log_training_commentary("training", tensorboard_writer_training, step, losses)

def log_scalars(step, names, values):
  with tensorboard_writer_validation.as_default():
    tf.summary.experimental.set_step(step)
    for name, value in zip(names, values):
      tf.summary.scalar(name.decode("utf-8"), value)

def should_log_graph(step):
  return (step == 1)

def log_training_prepare(step):
  if should_log_graph(step):
    tf.summary.trace_on(graph=True, profiler=False)

def log_training(type, writer, step, losses):
  log(f"Loss: {losses[0]:.4f} (V: {losses[1]:.4f}, MV: {losses[2]:.4f}, P: {losses[3]:.4f}, RP: {losses[4]:.4f}), Acc. (P): {losses[5]:.4f}, Acc. (RP): {losses[6]:.4f} ({type})")
  with writer.as_default():
    tf.summary.experimental.set_step(step)
    if should_log_graph(step):
      tf.summary.trace_export("model")
    log_loss_accuracy(losses)
    log_weights()
    writer.flush()

def log_training_commentary(type, writer, step, losses):
  log(f"Loss: {losses[0]:.4f}, Accuracy: {losses[1]:.4f} ({type})")
  # TODO: log commentary training to tensorboard
  # with writer.as_default():
  #   tf.summary.experimental.set_step(step)
  #   if should_log_graph(step):
  #     tf.summary.trace_export("model")
  #   log_loss_accuracy(losses)
  #   log_weights()
  #   writer.flush()

def log_loss_accuracy(losses):
  # Fix losses: only total includes loss weighting.
  with tf.name_scope("loss"):
    tf.summary.scalar("overall loss", losses[0])
    tf.summary.scalar("value loss", losses[1])
    tf.summary.scalar("mcts value loss", losses[2])
    tf.summary.scalar("policy loss", losses[3])
    tf.summary.scalar("reply policy loss", losses[4])
    # Equivalent to tf.math.add_n(model.losses)
    loss_weights = [config.training_network["value_loss_weight"], config.training_network["mcts_value_loss_weight"],
      config.training_network["policy_loss_weight"], config.training_network["reply_policy_loss_weight"]]
    tf.summary.scalar("L2 loss", losses[0] - (losses[1] * loss_weights[0]) - (losses[2] * loss_weights[1]) - (losses[3] * loss_weights[2]) - (losses[4] * loss_weights[3])) 
  with tf.name_scope("accuracy"):
    tf.summary.scalar("policy accuracy", losses[5])
    tf.summary.scalar("reply policy accuracy", losses[6])

def log_weights():
  for layer in networks.training_network.model.layers:
    for weight in layer.weights:
      weight_name = weight.name.replace(':', '_')
      tf.summary.histogram(weight_name, weight)

def load_network(network_name):
  networks.network_name = network_name

  # Load latest prediction network now, but delay loading training network until necessary.
  networks.training_network = None
  networks.prediction_network = storage.load_latest_network(config, network_name, update_network_for_predictions)
  if not networks.prediction_network:
    networks.prediction_network = UniformNetwork()
    log("Loaded uniform network (predictions)")

def save_network(checkpoint):
  ensure_training()
  log(f"Saving network ({checkpoint} steps)...")
  path = storage.save_network(config, networks.network_name, checkpoint, networks.training_network)
  log(f"Saved network ({checkpoint} steps)")
  networks.prediction_network = update_network_for_predictions(path)

config = Config()
networks = Networks()
tensorboard_network_path = os.path.join(config.misc["paths"]["tensorboard"], config.training_network["name"])
tensorboard_writer_training_path = os.path.join(tensorboard_network_path, "training")
tensorboard_writer_training = tf.summary.create_file_writer(tensorboard_writer_training_path)
tensorboard_writer_validation_path = os.path.join(tensorboard_network_path, "validation")
tensorboard_writer_validation = tf.summary.create_file_writer(tensorboard_writer_validation_path)
