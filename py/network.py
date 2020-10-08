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
from model import ModelBuilder
import transformer

K.set_image_data_format("channels_first")

def log(*args):
  if not silent:
    print(*args)

knowledge_distillation_temperature = 2.0
knowledge_distillation_teacher_weight = 0.7

# This fixes an issue with categorical_crossentropy calculating incorrectly
# over our 73*8*8 output planes - loss ends up way too small.
def flat_categorical_crossentropy_from_logits(y_true, y_pred):
  return tf.keras.losses.categorical_crossentropy(y_true=K.batch_flatten(y_true), y_pred=K.batch_flatten(y_pred), from_logits=True)

# This fixes the same issue with categorical_accuracy. No point doing softmax on logits for argmax.
def flat_categorical_accuracy(y_true, y_pred):
  return tf.keras.metrics.categorical_accuracy(y_true=K.batch_flatten(y_true), y_pred=K.batch_flatten(y_pred))

# Weighted sum of knowledge_distillation_temperature**2 * teacher loss + ground truth loss
def student_policy_loss(y_true, y_pred):
  teacher_logits = K.batch_flatten(y_true[0])
  true_labels = K.batch_flatten(y_true[1])
  y_pred = K.batch_flatten(y_pred)

  teacher_loss = knowledge_distillation_temperature**2 * tf.keras.losses.categorical_crossentropy(
    y_true=tf.stop_gradient(tf.nn.softmax(teacher_logits/knowledge_distillation_temperature)), y_pred=y_pred/knowledge_distillation_temperature, from_logits=True)
  true_loss = tf.keras.losses.categorical_crossentropy(y_true=true_labels, y_pred=y_pred, from_logits=True)
  return (knowledge_distillation_teacher_weight * teacher_loss) + ((1 - knowledge_distillation_teacher_weight) * true_loss)

# Accuracy against ground truth (argmax).
def student_policy_accuracy(y_true, y_pred):
  return flat_categorical_accuracy(y_true[1], y_pred)

class Model:

  def save(self, model_path):
    raise NotImplementedError

  def save_weights(self, model_path):
    raise NotImplementedError

  def load_weights(self, model_path):
    raise NotImplementedError

  def predict_batch_raw(self, images):
    raise NotImplementedError

  def predict_batch(self, images):
    raise NotImplementedError

  def train_batch(self, images, values, mcts_values, policies, reply_policies):
    raise NotImplementedError

  def validate_batch(self, images, values, mcts_values, policies, reply_policies):
    raise NotImplementedError

class UniformModel(Model):

  def __init__(self, num_outputs):
    self.num_outputs = num_outputs
    self.latest_values = None
    self.latest_policies = None

  def predict_batch_raw(self, images):
    # Check both separately because of threading
    values = self.latest_values
    policies = self.latest_policies
    if (values is None) or (len(images) != len(values)):
      values = numpy.zeros((len(images)), dtype=numpy.float32)
      self.latest_values = values
    if (policies is None) or (len(images) != len(policies)):
      policies = numpy.zeros((len(images), ModelBuilder.output_planes_count, ModelBuilder.board_side, ModelBuilder.board_side), dtype=numpy.float32)
      self.latest_policies = policies
    
    if self.num_outputs == 2:
      return values, policies
    elif self.num_outputs == 4:
      return values, policies, values, policies
    else:
      raise ValueError

  def predict_batch(self, images):
    return self.predict_batch_raw(images)

class KerasModel(Model):

  def __init__(self, model):
    if isinstance(model, KerasModel):
      raise TypeError
    self.model = model

  @classmethod
  def load(cls, model_path):
    # Don't serialize optimizer: custom loss/metrics.
    model = tf.keras.models.load_model(model_path, custom_objects={
      "flat_categorical_crossentropy_from_logits": flat_categorical_crossentropy_from_logits,
      "flat_categorical_accuracy": flat_categorical_accuracy,
    }, compile=False)
    return cls(model)

  def save(self, model_path):
    # Don't serialize optimizer: custom loss/metrics.
    self.model.save(model_path, include_optimizer=False, save_format="tf")

  def save_weights(self, model_path):
    self.model.save_weights(model_path, save_format="tf")

  def load_weights(self, model_path):
    self.model.load_weights(model_path)

  def train_batch(self, step, images, values, mcts_values, policies, reply_policies):
    learning_rate = get_learning_rate(config.training_network["learning_rate_schedule"], step)
    K.set_value(self.model.optimizer.lr, learning_rate)
    return self.model.train_on_batch(images, [values, mcts_values, policies, reply_policies], reset_metrics=False)

  def validate_batch(self, step, images, values, mcts_values, policies, reply_policies):
    return self.model.test_on_batch(images, [values, mcts_values, policies, reply_policies], reset_metrics=False)

class TensorFlowModel(Model):

  # NEED to pass the model in THEN extract the function, otherwise all hell breaks loose.
  def __init__(self, model):
    self.model = model
    self.function = self.model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    self.num_outputs = len(self.function.outputs)

  @classmethod
  def load(cls, model_path):
    model = tf.saved_model.load(model_path)
    return cls(model)

  def predict_batch_raw(self, images):
    prediction = self.function(images)
    if self.num_outputs == 2:
      values, policies = \
        prediction[ModelBuilder.output_value_name], \
        prediction[ModelBuilder.output_policy_name]
      return values, policies
    elif self.num_outputs == 4:
      values, mcts_values, policies, reply_policies = \
        prediction[ModelBuilder.output_value_name], \
        prediction[ModelBuilder.output_policy_name], \
        prediction[ModelBuilder.output_mcts_value_name], \
        prediction[ModelBuilder.output_reply_policy_name]
      return values, mcts_values, policies, reply_policies
    else:
      raise ValueError

  def predict_batch(self, images):
    images = tf.constant(images)
    prediction = self.function(images)
    if self.num_outputs == 2:
      values, policies = \
        prediction[ModelBuilder.output_value_name], \
        prediction[ModelBuilder.output_policy_name]
      return values.numpy(), policies.numpy()
    elif self.num_outputs == 4:
      values, mcts_values, policies, reply_policies = \
        prediction[ModelBuilder.output_value_name], \
        prediction[ModelBuilder.output_policy_name], \
        prediction[ModelBuilder.output_mcts_value_name], \
        prediction[ModelBuilder.output_reply_policy_name]
      return values.numpy(), mcts_values.numpy(), policies.numpy(), reply_policies.numpy()
    else:
      raise ValueError

class Network:

  def __init__(self, network_type, name):
    self.network_type = network_type
    self._name = name
    self.initialize()

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, value):
    self._name = value
    self.loading_model_predict = False

    # Clear out any loaded models, ready to lazy-load using the new name.
    self.initialize()

  def initialize(self):
    self.model_full = None
    self.model_train = None
    self.model_predict = None
    self.model_commentary_encoder = None
    self.model_commentary_decoder = None
    self.commentary_tokenizer = None

    tensorboard_network_path = os.path.join(config.misc["paths"]["tensorboard"], self._name, self.network_type)
    self.tensorboard_writer_training = tf.summary.create_file_writer(os.path.join(tensorboard_network_path, "training"))
    self.tensorboard_writer_validation = tf.summary.create_file_writer(os.path.join(tensorboard_network_path, "validation"))

  def predict_batch_raw(self, images):
    self.ensure_prediction()
    return self.model_predict.predict_batch_raw(images)

  def predict_batch(self, images):
    self.ensure_prediction()
    return self.model_predict.predict_batch(images)

  def train_batch(self, step, images, values, mcts_values, policies, reply_policies):
    self.ensure_training()
    return self.model_train.train_batch(step, images, values, mcts_values, policies, reply_policies)

  def validate_batch(self, step, images, values, mcts_values, policies, reply_policies):
    self.ensure_training()
    return self.model_train.validate_batch(step, images, values, mcts_values, policies, reply_policies)

  def predict_commentary_batch(self, images):
    # Only used in the teacher network.
    raise NotImplementedError

  def train_commentary_batch(self, images):
    # Only used in the teacher network.
    raise NotImplementedError

  def subset_prediction(self, model):
    # Prediction details vary between teacher and student, so implement in subclasses.
    raise NotImplementedError

  def compile_for_training(self):
    # Compilation details vary between teacher and student, so implement in subclasses.
    raise NotImplementedError

  def build_model_for_training(self):
    # Structure varies between teacher and student, so implement in subclasses.
    raise NotImplementedError

  def ensure_prediction(self):
    # The prediction model may already exist.
    if self.model_predict:
      return

    # Another thread may already be loading here: if so, wait for it.
    if self.loading_model_predict:
      while not self.model_predict:
        time.sleep(0.25)
      return
    self.loading_model_predict = True
    
    # Either load it from disk, or create a UniformModel (let the teacher/student implement).
    network_path = self.latest_network_path()
    if network_path:
      log_name = self.get_log_name(network_path)
      log(f"Loading model ({self.network_type}/predictions): {log_name}...")
      model_predict_path = self.model_predict_path(network_path, self.network_type)
      self.model_predict = TensorFlowModel.load(model_predict_path)
      log(f"Loaded model ({self.network_type}/predictions): {log_name}")
    else:
      log(f"Using uniform model ({self.network_type}/predictions)")
      self.model_predict = self.subset_prediction(None)

    # Finished, now model_predict is assigned.
    self.loading_model_predict = False
  
  def ensure_training(self):
    # The training subset may already exist.
    if self.model_train:
      return
    
    # If the full model doesn't exist, either load it from disk, or create a new one.
    if not self.model_full:
      network_path = self.latest_network_path()
      if network_path:
        log_name = self.get_log_name(network_path)
        log(f"Loading model ({self.network_type}/full): {log_name}...")
        model_full_path = self.model_full_path(network_path, self.network_type)
        self.model_full = self.build_model_for_training()
        self.model_full.load_weights(model_full_path)
        log(f"Loaded model ({self.network_type}/full): {log_name}")
      else:
        log(f"Creating new model ({self.network_type}/full)")
        self.model_full = self.build_model_for_training()

    # Take the training subset from the full model.
    if self.model_full:
      self.model_train = KerasModel(ModelBuilder().subset_train(self.model_full.model))

    # The training subset is definitely new, so compile for training.
    self.compile_for_training()

  def ensure_commentary(self):
    # The encoder, decoder and tokenizer may already exist.
    if self.model_commentary_encoder:
      return

    # Take the encoder subset from the full model.
    self.ensure_training()
    self.model_commentary_encoder = KerasModel(ModelBuilder().subset_commentary_encoder(self.model_full.model))

    # Either load decoder and tokenizer from disk, or create new.
    network_path = self.latest_network_path()
    if network_path:
      log_name = self.get_log_name(network_path)
      log(f"Loading model ({self.network_type}/commentary): {log_name}...")

      # Load the tokenizer first.
      commentary_tokenizer_path = self.commentary_tokenizer_path(network_path, self.network_type)
      with open(commentary_tokenizer_path, 'r') as f:
          self.commentary_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

      # Build the decoder using the tokenizer, then load weights.
      decoder, _ = ModelBuilder().build_commentary_decoder(config, self.commentary_tokenizer)
      self.model_commentary_decoder = KerasModel(decoder)
      model_commentary_decoder_path = self.model_commentary_decoder_path(network_path, self.network_type)
      self.model_commentary_decoder.load_weights(model_commentary_decoder_path)

      log(f"Loaded model ({self.network_type}/commentary): {log_name}")
    else:
      log(f"Creating new model ({self.network_type}/commentary)")
      decoder, self.commentary_tokenizer = ModelBuilder().build_commentary_decoder(config)
      self.model_commentary_decoder = KerasModel(decoder)

  def save(self, step):
    # Even if no training has happened yet, save a fresh network at least.
    self.ensure_training()
    network_path = self.make_network_path(step)
    log_name = self.get_log_name(network_path)

    # Save the full Keras model.
    log(f"Saving model ({self.network_type}/full): {log_name}...")
    model_full_path = self.model_full_path(network_path, self.network_type)
    self.model_full.save_weights(model_full_path)
    log(f"Saved model ({self.network_type}/full): {log_name}")

    # Invalidate and save the training model (appropriately subset) as the prediction model.
    # It is only ever loaded from disk as a TensorFlow model, not as a Keras model.
    self.model_predict = None
    log(f"Saving model ({self.network_type}/prediction): {log_name}...")
    model_predict_path = self.model_predict_path(network_path, self.network_type)
    model_predict_to_save = self.subset_prediction(self.model_full.model)
    model_predict_to_save.save(model_predict_path)
    log(f"Saved model ({self.network_type}/prediction): {log_name}")

  def get_log_name(self, network_path):
    return os.path.basename(os.path.normpath(network_path))

  def latest_network_path(self):
    parent_path = config.misc["paths"]["networks"]
    _, directories, _ = next(os.walk(parent_path))
    for directory in reversed(directories): # Only load the latest.
      if directory.startswith(self.name + "_") and os.path.isdir(os.path.join(parent_path, directory, self.network_type)):
        return os.path.join(parent_path, directory)
    return None

  def make_network_path(self, step):
    parent_path = config.misc["paths"]["networks"]
    directory_name = f"{self.name}_{str(step).zfill(9)}"
    return os.path.join(parent_path, directory_name)

  def model_full_path(self, network_path, network_type):
    return os.path.join(network_path, network_type, "model", "weights")

  def model_predict_path(self, network_path, network_type):
    return os.path.join(network_path, network_type, "predict")

  def model_commentary_decoder_path(self, network_path, network_type):
    return os.path.join(network_path, network_type, "commentary_decoder", "weights")

  def commentary_tokenizer_path(self, network_path, network_type):
    return os.path.join(network_path, network_type, "commentary_tokenizer.json")

class TeacherNetwork(Network):

  def __init__(self, name):
    super().__init__("teacher", name)

    # Set up the commentary optimizer in advance, for use with GradientTape in transformer.py.
    commentary_learning_rate = get_commentary_learning_rate(config.training_network["commentary_learning_rate_schedule"], 0)
    self.commentary_optimizer = tf.keras.optimizers.SGD(
      learning_rate=commentary_learning_rate,
      momentum=config.training_network["momentum"])

  # The teacher network also deals with commentary.
  def save(self, step):
    # Even if no commentary training has happened yet, save a fresh decoder, etc. at least.
    self.ensure_commentary()

    super().save(step)
    
    network_path = self.make_network_path(step)
    log_name = self.get_log_name(network_path)
    log(f"Saving model ({self.network_type}/commentary): {log_name}...")

    # Save the commentary decoder.
    model_commentary_decoder_path = self.model_commentary_decoder_path(network_path, self.network_type)
    self.model_commentary_decoder.save_weights(model_commentary_decoder_path)

    # Save the tokenizer.
    commentary_tokenizer_path = self.commentary_tokenizer_path(network_path, self.network_type)
    with open(commentary_tokenizer_path, 'w') as f:
      f.write(self.commentary_tokenizer.to_json())

    log(f"Saved model ({self.network_type}/commentary): {log_name}")

  # The teacher network needs to predict *4* outputs to feed into student training.
  def subset_prediction(self, model):
    if model:
      return KerasModel(ModelBuilder().subset_predict_teacher(model))
    else:
      return UniformModel(num_outputs=4)

  # The teacher network trains directly on supervised labels.
  def compile_for_training(self):
    optimizer = tf.keras.optimizers.SGD(
      learning_rate=get_learning_rate(config.training_network["learning_rate_schedule"], 0),
      momentum=config.training_network["momentum"])
    losses = ["mean_squared_error", "mean_squared_error", flat_categorical_crossentropy_from_logits, flat_categorical_crossentropy_from_logits]
    loss_weights = [config.training_network["value_loss_weight"], config.training_network["mcts_value_loss_weight"],
      config.training_network["policy_loss_weight"], config.training_network["reply_policy_loss_weight"]]
    metrics = [[], [], [flat_categorical_accuracy], [flat_categorical_accuracy]]
    self.model_train.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

  # The teacher network uses the full 19*256 model.
  def build_model_for_training(self):
    return KerasModel(ModelBuilder().build(config))

class StudentNetwork(Network):

  def __init__(self, name):
    super().__init__("student", name)

  # The student network needs to predict *2* outputs to feed MCTS in self-play.
  def subset_prediction(self, model):
    if model:
      return KerasModel(ModelBuilder().subset_predict_student(model))
    else:
      return UniformModel(num_outputs=2)

  # The student network trains on a combination of soft teacher labels and hard supervised labels.
  # Policy accuracy is still measured against the supervised labels.
  def compile_for_training(self):
    optimizer = tf.keras.optimizers.SGD(
      learning_rate=get_learning_rate(config.training_network["learning_rate_schedule"], 0),
      momentum=config.training_network["momentum"])
    losses = ["mean_squared_error", "mean_squared_error", student_policy_loss, student_policy_loss]
    loss_weights = [config.training_network["value_loss_weight"], config.training_network["mcts_value_loss_weight"],
      config.training_network["policy_loss_weight"], config.training_network["reply_policy_loss_weight"]]
    metrics = [[], [], [student_policy_accuracy], [student_policy_accuracy]]
    self.model_train.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

  # The student network uses the smaller 8*64 model.
  def build_model_for_training(self):
    return KerasModel(ModelBuilder().build_student(config))

class Networks:

  def __init__(self, name="network"):
    # Set by C++ in load_network depending on use-case.
    self._name = name
    self.teacher = TeacherNetwork(self._name)
    self.student = StudentNetwork(self._name)

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, value):
    self._name = value
    self.teacher.name = self._name
    self.student.name = self._name

def get_learning_rate(schedule, step):
  rate = 0.0
  for key, value in schedule:
    if step >= key:
      rate = value
    else:
      break
  return rate

def get_commentary_learning_rate(schedule, step):
  ratio = config.training_network["commentary_batch_size"] / config.training_network["batch_size"]
  return ratio * get_learning_rate(schedule, step)

def predict_batch_teacher(images):
  value, policy, _, _ = networks.teacher.predict_batch(images)
  return value, policy

def predict_batch_student(images):
  value, policy = networks.student.predict_batch(images)
  return value, policy

def predict_commentary_batch(images):
  # Predict commentary using training network for now.
  networks.teacher.ensure_commentary()

  encoder = networks.teacher.model_commentary_encoder.model
  decoder = networks.teacher.model_commentary_decoder.model
  tokenizer = networks.teacher.commentary_tokenizer

  start_token = tokenizer.word_index[ModelBuilder.token_start]
  end_token = tokenizer.word_index[ModelBuilder.token_end]
  max_length = ModelBuilder.transformer_max_length

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

def train_batch_teacher(step, images, values, mcts_values, policies, reply_policies):
  # Prepare TensorBoard logging.
  do_log_training = ((step % config.training_network["validation_interval"]) == 0)
  if do_log_training:
    log_training_prepare(step)

  # Train the teacher network.
  losses = networks.teacher.train_batch(step, images, values, mcts_values, policies, reply_policies)

  # Do Tensorboard logging (teacher).
  if do_log_training:
    log_training("training", networks.teacher.tensorboard_writer_training, step, losses, networks.teacher.model_train.model)

def train_batch_student(step, images, values, mcts_values, policies, reply_policies):
  # Prepare TensorBoard logging.
  do_log_training = ((step % config.training_network["validation_interval"]) == 0)
  if do_log_training:
    log_training_prepare(step)

  # Get the soft targets from the teacher and combine with the provided hard targets.
  images = tf.constant(images)
  _, teacher_policies, _, teacher_reply_policies = networks.teacher.predict_batch_raw(images)
  policies = tf.stack([teacher_policies, policies])
  reply_policies = tf.stack([teacher_reply_policies, reply_policies])

  # Train the student network.
  losses = networks.student.train_batch(step, images, values, mcts_values, policies, reply_policies)
  
  # Do TensorBoard logging (student).
  if do_log_training:
    log_training("training", networks.student.tensorboard_writer_training, step, losses, networks.student.model_train.model)

def validate_batch_teacher(step, images, values, mcts_values, policies, reply_policies):
  # Prepare TensorBoard logging.
  log_training_prepare(step)

  # Validate the teacher network.
  losses = networks.teacher.validate_batch(step, images, values, mcts_values, policies, reply_policies)

  # Do Tensorboard logging (teacher).
  log_training("validation", networks.teacher.tensorboard_writer_validation, step, losses, networks.teacher.model_train.model)

def validate_batch_student(step, images, values, mcts_values, policies, reply_policies):
  # Prepare TensorBoard logging.
  log_training_prepare(step)

  # Get the soft targets from the teacher and combine with the provided hard targets.
  images = tf.constant(images)
  _, teacher_policies, _, teacher_reply_policies = networks.teacher.predict_batch_raw(images)
  policies = tf.stack([teacher_policies, policies])
  reply_policies = tf.stack([teacher_reply_policies, reply_policies])

  # Validate the student network.
  losses = networks.student.validate_batch(step, images, values, mcts_values, policies, reply_policies)

  # Do Tensorboard logging (student).
  log_training("validation", networks.student.tensorboard_writer_validation, step, losses, networks.student.model_train.model)

def train_commentary_batch(step, images, comments):
  networks.teacher.ensure_commentary()

  commentary_learning_rate = get_commentary_learning_rate(config.training_network["commentary_learning_rate_schedule"], step)
  K.set_value(networks.teacher.commentary_optimizer.lr, commentary_learning_rate)

  do_log_training = ((step % config.training_network["validation_interval"]) == 0)
  comments = [f"{ModelBuilder.token_start} {c.decode('utf-8')} {ModelBuilder.token_end}" for c in comments]
  comments = networks.teacher.commentary_tokenizer.texts_to_sequences(comments)
  comments = tf.keras.preprocessing.sequence.pad_sequences(comments, padding="post")
  losses = transformer.train_step(
    networks.teacher.commentary_optimizer,
    networks.teacher.model_commentary_encoder.model,
    networks.teacher.model_commentary_decoder.model,
    images,
    comments)
  if do_log_training:
    log_training_commentary("training", networks.teacher.tensorboard_writer_training, step, losses, networks.teacher.model_commentary_decoder.model)

def log_scalars_teacher(step, names, values):
  log_scalars(networks.teacher, step, names, values)

def log_scalars_student(step, names, values):
  log_scalars(networks.student, step, names, values)

def log_scalars(network, step, names, values):
  with network.tensorboard_writer_validation.as_default():
    tf.summary.experimental.set_step(step)
    for name, value in zip(names, values):
      tf.summary.scalar(name.decode("utf-8"), value)

def should_log_graph(step):
  return (step == 1)

def log_training_prepare(step):
  if should_log_graph(step):
    tf.summary.trace_on(graph=True, profiler=False)

def log_training(type, writer, step, losses, model):
  log(f"Loss: {losses[0]:.4f} (V: {losses[1]:.4f}, MV: {losses[2]:.4f}, P: {losses[3]:.4f}, RP: {losses[4]:.4f}), Acc. (P): {losses[5]:.4f}, Acc. (RP): {losses[6]:.4f} ({type})")
  with writer.as_default():
    tf.summary.experimental.set_step(step)
    if should_log_graph(step):
      tf.summary.trace_export("model")
    log_loss_accuracy(losses)
    log_weights(model)
    writer.flush()

  # Reset metrics after "validation_interval" training batches or 1 validation batch, ready for the next "validation_interval"/1.
  model.reset_metrics()

def log_training_commentary(type, writer, step, losses, model):
  log(f"Loss: {losses[0]:.4f}, Accuracy: {losses[1]:.4f} ({type})")
  with writer.as_default():
    tf.summary.experimental.set_step(step)
    log_loss_accuracy_commentary(losses)
    log_weights(model)
    writer.flush()

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

def log_loss_accuracy_commentary(losses):
  with tf.name_scope("loss"):
    tf.summary.scalar("commentary loss", losses[0])
  with tf.name_scope("accuracy"):
    tf.summary.scalar("commentary accuracy", losses[1])

def log_weights(model):
  for layer in model.layers:
    for weight in layer.weights:
      weight_name = weight.name.replace(':', '_')
      tf.summary.histogram(weight_name, weight)

def load_network(network_name):
  networks.name = network_name

def save_network_teacher(checkpoint):
  save_network(networks.teacher, checkpoint)
  
def save_network_student(checkpoint):
  save_network(networks.student, checkpoint)

def save_network(network, checkpoint):
  network.save(checkpoint)

config = Config()
networks = Networks()