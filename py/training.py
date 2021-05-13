import time
import math
import tensorflow as tf
from tensorflow.keras import backend as K
from model import ModelBuilder

knowledge_distillation_temperature = 5.0
knowledge_distillation_teacher_weight = 0.6

transformer_label_smoothing = 0.1

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
  # See "Trainer.stack_teacher_targets" for stacking details.
  return flat_categorical_accuracy(y_true[1], y_pred)

def make_transformer_loss():
  import transformer
  vocabulary_size = ModelBuilder.transformer_vocabulary_size
  def transformer_loss(y_true, y_pred):
    return transformer.padded_cross_entropy_loss(y_pred, y_true, transformer_label_smoothing, vocabulary_size)
  return transformer_loss

class Trainer:

  def __init__(self, networks, tpu_strategy, devices, datasets):
    self.networks = networks
    self.log = networks.log
    self.networks.teacher.training_compiler = self.compile_teacher
    self.networks.student.training_compiler = self.compile_student
    self.networks.teacher.commentary_training_compiler = self.compile_commentary
    self.config = networks.config
    self.tpu_strategy = tpu_strategy
    self.device_count = len(devices)
    self.commentary_optimizer = None

    self.datasets = datasets
    self.data_glob_training = self.config.join(self.config.training["games_path_training"], "*.chunk")
    self.data_glob_validation = self.config.join(self.config.training["games_path_validation"], "*.chunk")
    self.data_glob_commentary = self.config.join(self.config.training["commentary_path_training"], "*.chunk")

    self.per_replica_batch_size = self.config.training["batch_size"]
    self.global_batch_size = self.per_replica_batch_size * self.device_count

    self.per_replica_batch_size_commentary = self.config.training["commentary_batch_size"]
    self.global_batch_size_commentary = self.per_replica_batch_size_commentary * self.device_count

    if tpu_strategy:
      self.strategy = tpu_strategy
    elif self.device_count > 1:
      self.strategy = tf.distribute.MirroredStrategy()
    else:
      self.strategy = tf.distribute.get_strategy()

  def get_learning_rate(self, schedule):
    return Schedule(schedule["steps"], schedule["rates"], self.config.training["warmup_steps"], self.device_count)

  # The teacher network trains directly on supervised labels.
  def compile_teacher(self, model, learning_rate=None):
    self.compile(model, flat_categorical_crossentropy_from_logits, flat_categorical_accuracy, learning_rate)

  # The student network trains on a combination of soft teacher labels and hard supervised labels.
  # Policy accuracy is still measured against the supervised labels.
  def compile_student(self, model, learning_rate=None):
    self.compile(model, student_policy_loss, student_policy_accuracy, learning_rate)

  def compile(self, model, policy_loss, policy_accuracy, learning_rate=None):
    if learning_rate is None:
      learning_rate = self.get_learning_rate(self.config.training["learning_rate_schedule"])
    optimizer = tf.keras.optimizers.SGD(
      learning_rate=learning_rate,
      momentum=self.config.training["momentum"])
    losses = ["mean_squared_error", "mean_squared_error", policy_loss]
    loss_weights = [self.config.training["value_loss_weight"], self.config.training["mcts_value_loss_weight"], self.config.training["policy_loss_weight"]]
    metrics = [[], [], [policy_accuracy]]
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics, steps_per_execution=self.config.training["steps_per_execution"])

  def compile_commentary(self, model):
    optimizer = tf.keras.optimizers.SGD(
      learning_rate=self.get_learning_rate(self.config.training["commentary_learning_rate_schedule"]),
      momentum=self.config.training["momentum"])
    loss = make_transformer_loss()
    model.compile(optimizer=optimizer, loss=loss, steps_per_execution=self.config.training["steps_per_execution"])

  def calculate_training_window(self, checkpoint):
    # The window will grow until reaching the desired size, then slide.
    window_max = checkpoint * self.config.training["num_games"] // self.config.training["steps"]
    window_min = max(0, window_max - self.config.training["window_size"])

    # Min is inclusive, max is exclusive, both 0-based.
    return (window_min, window_max)

  def train(self, network, teacher_network, starting_step, checkpoint, log=True):
    # Create models on the distribution strategy scope, including the teacher for knowledge distillation inference.
    with self.strategy.scope():
      model = network.ensure_training()
      if teacher_network:
        teacher_network.ensure_training()        

    # Set up data pipelines.
    training_windows = [self.calculate_training_window(checkpoint)]
    globs_training = [self.data_glob_training]
    globs_validation = [self.data_glob_validation]
    data_training = self.datasets.build_training_dataset(globs_training, training_windows, self.global_batch_size)
    data_validation = self.datasets.build_validation_dataset(globs_validation, self.global_batch_size)

    # TF forces re-iteration of validation data, so hack around it by using a subclass to maintain an iterator.
    data_validation = RollingDataset(data_validation)

    # Work out steps and intervals. Use the validation interval as an epoch to match fit()'s model.
    validation_interval = self.config.training["validation_interval"]
    checkpoint_interval = (checkpoint - starting_step + 1)
    assert checkpoint_interval % validation_interval == 0, f"Checkpoint interval ({checkpoint_interval}) must be a multiple of the validation interval ({validation_interval})"
    assert validation_interval % self.device_count == 0, f"Validation interval ({validation_interval}) must be a multiple of the device count ({self.device_count})"
    actual_validation_interval = validation_interval // self.device_count
    steps_per_execution = self.config.training["steps_per_execution"]
    assert actual_validation_interval % steps_per_execution == 0, f"Validation interval / device count ({actual_validation_interval}) must be a multiple of steps_per_execution ({steps_per_execution})"
    steps_per_epoch = actual_validation_interval
    initial_epoch = (starting_step - 1) // validation_interval
    epochs = checkpoint // validation_interval

    # If a teacher was provided, predict soft targets and combine with the provided hard targets.
    if teacher_network:
      model.init(teacher_network)

    # Train.
    callbacks = [LogCallback(self.config, self.log, network.tensorboard_writer_training, network.tensorboard_writer_validation, model, validation_interval)] if log else []
    model.fit(data_training, verbose=0, callbacks=callbacks,
      validation_data=data_validation, validation_steps=1, validation_freq=1,
      steps_per_epoch=steps_per_epoch, initial_epoch=initial_epoch, epochs=epochs)
    if log:
      self.log("Learning rate:", model.optimizer.learning_rate(model.optimizer._iterations).numpy().item())

  def train_commentary(self, network, starting_step, checkpoint):
    # Create models on the distribution strategy scope
    with self.strategy.scope():
      model = network.ensure_commentary_training()

    # Set up data pipelines.
    tokenizer = network.ensure_tokenizer(network.models_train)
    data_commentary_training = self.datasets.build_commentary_dataset(
      self.data_glob_commentary, tokenizer, self.global_batch_size_commentary, ModelBuilder.transformer_max_length)

    # Work out steps and intervals. Use the validation interval as an epoch to match fit()'s model.
    validation_interval = self.config.training["validation_interval"]
    checkpoint_interval = (checkpoint - starting_step + 1)
    assert checkpoint_interval % validation_interval == 0, f"Checkpoint interval ({checkpoint_interval}) must be a multiple of the validation interval ({validation_interval})"
    assert validation_interval % self.device_count == 0, f"Validation interval ({validation_interval}) must be a multiple of the device count ({self.device_count})"
    actual_validation_interval = validation_interval // self.device_count
    steps_per_execution = self.config.training["steps_per_execution"]
    assert actual_validation_interval % steps_per_execution == 0, f"Validation interval / device count ({actual_validation_interval}) must be a multiple of steps_per_execution ({steps_per_execution})"
    steps_per_epoch = actual_validation_interval
    initial_epoch = (starting_step - 1) // validation_interval
    epochs = checkpoint // validation_interval

    # Train.
    log_callback = CommentaryLogCallback(self.config, self.log, network.tensorboard_writer_training, network.tensorboard_writer_validation, model, validation_interval)
    model.fit(data_commentary_training, verbose=0, callbacks=[log_callback],
      steps_per_epoch=steps_per_epoch, initial_epoch=initial_epoch, epochs=epochs)
    self.log("Learning rate:", model.optimizer.learning_rate(model.optimizer._iterations).numpy().item())

  def log_scalars(self, network, step, names, values):
    network.ensure_tensorboard()
    writer = network.tensorboard_writer_validation
    with writer.as_default():
      tf.summary.experimental.set_step(step)
      for name, value in zip(names, values):
        tf.summary.scalar(name.decode("utf-8"), value)
      writer.flush()

class StudentModel(tf.keras.Model):

  def init(self, teacher_network):
    self.teacher_network = teacher_network

  def train_step(self, data):
    from tensorflow.python.eager import backprop
    from tensorflow.python.keras.engine import data_adapter
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    # Predict teacher logits and stack with provided labels to unpack in "student_policy_loss"/"student_policy_accuracy".
    y = (y[0], y[1], tf.stack([self.teacher_network.tf_predict_for_training(x)[2], y[2]], axis=0))

    with backprop.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.compiled_loss(
          y, y_pred, sample_weight, regularization_losses=self.losses)
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    from tensorflow.python.keras.engine import data_adapter
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    # Predict teacher logits and stack with provided labels to unpack in "student_policy_loss"/"student_policy_accuracy".
    y = (y[0], y[1], tf.stack([self.teacher_network.tf_predict_for_training(x)[2], y[2]], axis=0))

    y_pred = self(x, training=False)
    # Updates stateful loss metrics.
    self.compiled_loss(
        y, y_pred, sample_weight, regularization_losses=self.losses)

    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    return {m.name: m.result() for m in self.metrics}

class Schedule(tf.keras.optimizers.schedules.PiecewiseConstantDecay):

  def __init__(self, steps, rates, warmup_steps, device_count):
    # Keras sees step numbers for global batches in Model.fit() (e.g. steps_per_epoch has a divide-by-device_count),
    # not the per-replica batches our config is based on, so adjust the boundaries here correspondingly.
    boundaries = [((boundary // device_count) - 1) for boundary in steps[1:]]
    values = rates.copy()
    if not boundaries:
      boundaries.append(0)
      values.append(values[0])
    super().__init__(boundaries, values)
    self.warmup_steps = warmup_steps
    self.device_count = device_count

  def __call__(self, step):
    value = super().__call__(step)
    # Theory says scale with sqrt(global batch size); practice says scale linearly with global batch size.
    # Configuration values are tuned to the local batch size for a single device.
    value *= self.device_count
    if self.warmup_steps:
      value *= tf.cast(tf.clip_by_value(step / self.warmup_steps, 0.0, 1.0), tf.float32)
    return value

class RollingDataset(tf.data.Dataset):

  def __init__(self, wrapped):
    self.wrapped = wrapped
    self.iterator = iter(wrapped)
    # Pretend that the wrapped data isn't changing, have to pass in a variant_tensor.
    super().__init__(wrapped._variant_tensor_attr)

  def __iter__(self):
    # Return the same iterator each time to avoid resetting on new iterations.
    return self.iterator

  def _inputs(self):
    return [self.wrapped]

  @property
  def element_spec(self):
    return self.wrapped.element_spec

class LogCallback(tf.keras.callbacks.Callback):

  def __init__(self, config, log, training_writer, validation_writer, model, validation_interval):
    self.config = config
    self.log = log
    self.training_writer = training_writer
    self.validation_writer = validation_writer
    self.model = model
    self.validation_interval = validation_interval

  def on_epoch_end(self, epoch, logs=None):
    effective_step = (epoch + 1) * self.validation_interval
    training_losses = self.map_losses(logs, "")
    validation_losses = self.map_losses(logs, "val_")
    self.log_training("training", self.training_writer, effective_step, training_losses)
    self.log_training("validation", self.validation_writer, effective_step, validation_losses)

  def map_losses(self, logs, prefix):
    return [
      logs[prefix + "loss"],
      logs[prefix + ModelBuilder.output_value_name + "_loss"],
      logs[prefix + ModelBuilder.output_mcts_value_name + "_loss"],
      logs[prefix + ModelBuilder.output_policy_name + "_loss"],
      logs.get(prefix + ModelBuilder.output_policy_name + "_" + flat_categorical_accuracy.__name__,
        logs.get(prefix + ModelBuilder.output_policy_name + "_" + student_policy_accuracy.__name__))
    ]

  def log_training(self, type, writer, step, losses):
    # Fail-fast for NaNs.
    if math.isnan(losses[0]):
      raise ValueError("NaN encountered in training/validation losses")
    self.log(f"Loss: {losses[0]:.4f} (V: {losses[1]:.4f}, MV: {losses[2]:.4f}, P: {losses[3]:.4f}), Accuracy (P): {losses[4]:.4f} ({type})")
    with writer.as_default():
      tf.summary.experimental.set_step(step)
      self.log_loss_accuracy(losses)
      self.log_weights(self.model)
      writer.flush()

  def log_loss_accuracy(self, losses):
    # Fix losses: only total includes loss weighting.
    with tf.name_scope("loss"):
      tf.summary.scalar("overall loss", losses[0])
      tf.summary.scalar("value loss", losses[1])
      tf.summary.scalar("mcts value loss", losses[2])
      tf.summary.scalar("policy loss", losses[3])
      # Equivalent to tf.math.add_n(model.losses)
      loss_weights = [self.config.training["value_loss_weight"], self.config.training["mcts_value_loss_weight"],
        self.config.training["policy_loss_weight"]]
      tf.summary.scalar("L2 loss", losses[0] - (losses[1] * loss_weights[0]) - (losses[2] * loss_weights[1]) - (losses[3] * loss_weights[2])) 
    with tf.name_scope("accuracy"):
      tf.summary.scalar("policy accuracy", losses[4])

  def log_weights(self, model):
    for layer in model.layers:
      for weight in layer.weights:
        weight_name = weight.name.replace(':', '_')
        tf.summary.histogram(weight_name, weight)

class CommentaryLogCallback(tf.keras.callbacks.Callback):

  def __init__(self, config, log, training_writer, validation_writer, model, validation_interval):
    self.config = config
    self.log = log
    self.training_writer = training_writer
    self.validation_writer = validation_writer
    self.model = model
    self.validation_interval = validation_interval

  def on_epoch_end(self, epoch, logs=None):
    effective_step = (epoch + 1) * self.validation_interval
    training_losses = self.map_losses(logs, "")
    self.log_training_commentary("training", self.training_writer, effective_step, training_losses)

  def map_losses(self, logs, prefix):
    return [
      logs[prefix + "loss"],
    ]

  def log_training_commentary(self, type, writer, step, losses):
    self.log(f"Loss: {losses[0]:.4f}")
    with writer.as_default():
      tf.summary.experimental.set_step(step)
      self.log_loss_commentary(losses)
      self.log_weights(self.model)
      writer.flush()

  def log_loss_commentary(self, losses):
    with tf.name_scope("loss"):
      tf.summary.scalar("commentary loss", losses[0])

  def log_weights(self, model):
    for layer in model.layers:
      for weight in layer.weights:
        weight_name = weight.name.replace(':', '_')
        tf.summary.histogram(weight_name, weight)