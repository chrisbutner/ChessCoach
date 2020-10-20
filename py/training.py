import time
import tensorflow as tf
from tensorflow.keras import backend as K

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

class Trainer:

  def __init__(self, networks, tpu_strategy, devices, datasets):
    self.networks = networks
    self.networks.teacher.training_compiler = self.compile_teacher
    self.networks.student.training_compiler = self.compile_student
    self.config = networks.config
    self.tpu_strategy = tpu_strategy
    self.device_count = len(devices)
    self.commentary_optimizer = None
    self.datasets = datasets
    self.data_training = None
    self.data_validation = None
    self.data_training_key = None

    self.per_replica_batch_size = self.config.training_network["batch_size"]
    self.global_batch_size = self.per_replica_batch_size * self.device_count

    self.log("Devices:", self.device_count)
    self.log("Per-replica batch size:", self.per_replica_batch_size)
    self.log("Global batch size:", self.global_batch_size)

    if tpu_strategy:
      self.log("Strategy: TPU")
      self.strategy = tpu_strategy
    elif self.device_count > 1:
      self.log("Strategy: Mirrored")
      self.strategy = tf.distribute.MirroredStrategy()
    else:
      self.log("Strategy: Default")
      self.strategy = tf.distribute.get_strategy()

  def log(self, *args):
    self.networks.log(*args)

  # Would use PiecewiseConstantDecay but it hangs.
  def get_learning_rate_common(self, schedule, step, multiplier):
    rate = 0.0
    for key, value in schedule:
      if step >= key:
        rate = value
      else:
        break
    return rate * multiplier

  def get_learning_rate(self, step):
    schedule = self.config.training_network["learning_rate_schedule"]
    scale_learning_rate_with_batch_size = self.device_count
    return self.get_learning_rate_common(schedule, step, scale_learning_rate_with_batch_size)

  def get_commentary_learning_rate(self, step):
    schedule = self.config.training_network["commentary_learning_rate_schedule"]
    scale_learning_rate_with_batch_size = self.device_count
    commentary_multiplier = self.config.training_network["commentary_batch_size"] / self.config.training_network["batch_size"]
    return self.get_learning_rate_common(schedule, step, scale_learning_rate_with_batch_size * commentary_multiplier)

  # The teacher network trains directly on supervised labels.
  def compile_teacher(self, model):
    optimizer = tf.keras.optimizers.SGD(
      learning_rate=self.get_learning_rate(0),
      momentum=self.config.training_network["momentum"])
    losses = ["mean_squared_error", "mean_squared_error", flat_categorical_crossentropy_from_logits, flat_categorical_crossentropy_from_logits]
    loss_weights = [self.config.training_network["value_loss_weight"], self.config.training_network["mcts_value_loss_weight"],
      self.config.training_network["policy_loss_weight"], self.config.training_network["reply_policy_loss_weight"]]
    metrics = [[], [], [flat_categorical_accuracy], [flat_categorical_accuracy]]
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

    # Set up the commentary optimizer in advance, for use with GradientTape in transformer.py.
    self.commentary_optimizer = tf.keras.optimizers.SGD(
      learning_rate=self.get_commentary_learning_rate(0),
      momentum=self.config.training_network["momentum"])

  # The student network trains on a combination of soft teacher labels and hard supervised labels.
  # Policy accuracy is still measured against the supervised labels.
  def compile_student(self, model):
    optimizer = tf.keras.optimizers.SGD(
      learning_rate=self.get_learning_rate(0),
      momentum=self.config.training_network["momentum"])
    losses = ["mean_squared_error", "mean_squared_error", student_policy_loss, student_policy_loss]
    loss_weights = [self.config.training_network["value_loss_weight"], self.config.training_network["mcts_value_loss_weight"],
      self.config.training_network["policy_loss_weight"], self.config.training_network["reply_policy_loss_weight"]]
    metrics = [[], [], [student_policy_accuracy], [student_policy_accuracy]]
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

  # NOTE: This can be extremely slow right now (e.g. ~500 seconds after 1000 steps on 2**17 shuffle buffer)
  # but is necessary when training and self-playing on a single machine to ensure enough memory for
  # either (a) a 32-cycle_length of chunk loads plus a 2**17-shuffle buffer (~ 30 GB with overhead)
  # or (b) num_workers * prediction_batch_size games in parallel worth of Stockfish StateInfos and MCTS Nodes.
  def clear_data(self):
    if self.data_training_key:
      self.log("Clearing datasets")
    self.data_training = None
    self.data_validation = None
    self.data_training_key = None

  def train_teacher(self, gameTypes, trainingWindows, starting_step, checkpoint):
    network = self.networks.teacher
    with self.strategy.scope():
      model = network.ensure_training()

    # Set the learning rate. Checkpoints are small enough not to worry if it changes mid-way through.
    K.set_value(model.optimizer.lr, self.get_learning_rate(starting_step))

    # Set up data pipelines, or re-use existing if not cleared by self-play.
    data_training_key = (gameTypes, trainingWindows)
    if data_training_key != self.data_training_key:
      data_start = time.time()
      globs_training = ["TODO"]
      globs_validation = ["TODO"]
      # TODO: Only first type for now
      self.data_training = iter(self.datasets.build_training_dataset(globs_training[0], trainingWindows[0], self.global_batch_size))
      self.data_validation = iter(self.datasets.build_validation_dataset(globs_validation[0], self.global_batch_size))
      self.data_training_key = data_training_key
      next(self.data_training)
      next(self.data_validation)
      print(f"Datasets prepared in {(time.time() - data_start):.2f} seconds")

    # Iterate over all steps in the checkpoint, striding by the device count.
    # E.g. on a GPU, 1 device, 1-1000 steps, range(1, 1001, 1), 1000 iterations
    # E.g. on a TPU, 8 devices, 1-1000 steps, range(1, 1001, 8), 125 iterations, 125*8=1000 effective iterations
    for step in range(starting_step, checkpoint + 1, self.device_count):
      # Train.
      batch_training = next(self.data_training)
      self.train_batch(network, teacher_network=None, model=model, step=step, images=batch_training[0], targets=batch_training[1])

      # Sometimes validate.
      if self.is_validation_step(step):
        batch_validation = next(self.data_validation)
        self.validate_batch(network, teacher_network=None, model=model, step=step, images=batch_validation[0], targets=batch_validation[1])

  def train_student(self, gameTypes, trainingWindows, starting_step, checkpoint):
    # TODO: Implement
    pass

  def is_validation_step(self, step):
    # Striding steps by the device count, so validate when at or just past the validation interval.
    # E.g. on a GPU, 1 device, 100 validation interval, steps 1, 2, 3, etc., validate on step 100.
    # E.g. on a TPU, 8 devices, 100 validation interval, steps 1, 9, 17, etc., validate on step 105.
    return ((step % self.config.training_network["validation_interval"]) < self.device_count)

  def train_batch(self, network, teacher_network, model, step, images, targets):
    # Prepare TensorBoard logging.
    do_log_training = self.is_validation_step(step)
    if do_log_training:
      self.log_training_prepare(step)

    # If a teacher was provided, predict soft targets and combine with the provided hard targets.
    if teacher_network:
      _, teacher_policies, _, teacher_reply_policies = teacher_network.predict_for_training_batch(images)
      values, mcts_values, policies, reply_policies = targets
      policies = tf.stack([teacher_policies, policies])
      reply_policies = tf.stack([teacher_reply_policies, reply_policies])
      targets = (values, mcts_values, policies, reply_policies)

    # Train the model.
    losses = model.train_on_batch(images, targets, reset_metrics=False)

    # Do Tensorboard logging.
    if do_log_training:
      self.log_training("training", network.tensorboard_writer_training, step, losses, model)

  def validate_batch(self, network, teacher_network, model, step, images, targets):
    # Prepare TensorBoard logging.
    self.log_training_prepare(step)

    # If a teacher was provided, predict soft targets and combine with the provided hard targets.
    if teacher_network:
      _, teacher_policies, _, teacher_reply_policies = teacher_network.predict_for_training_batch(images)
      values, mcts_values, policies, reply_policies = targets
      policies = tf.stack([teacher_policies, policies])
      reply_policies = tf.stack([teacher_reply_policies, reply_policies])
      targets = (values, mcts_values, policies, reply_policies)

    # Train the model.
    losses = model.test_on_batch(images, targets, reset_metrics=False)

    # Do Tensorboard logging.
    self.log_training("validation", network.tensorboard_writer_validation, step, losses, model)

  # def train_commentary_batch(self, step, images, comments):
  #   networks.teacher.ensure_commentary()

  #   commentary_learning_rate = get_commentary_learning_rate(config.training_network["commentary_learning_rate_schedule"], step)
  #   K.set_value(networks.teacher.commentary_optimizer.lr, commentary_learning_rate)

  #   do_log_training = self.is_validation_step(step)
  #   comments = [f"{ModelBuilder.token_start} {c.decode('utf-8')} {ModelBuilder.token_end}" for c in comments]
  #   comments = networks.teacher.commentary_tokenizer.texts_to_sequences(comments)
  #   comments = tf.keras.preprocessing.sequence.pad_sequences(comments, padding="post")
  #   losses = transformer.train_step(
  #     networks.teacher.commentary_optimizer,
  #     networks.teacher.model_commentary_encoder.model,
  #     networks.teacher.model_commentary_decoder.model,
  #     images,
  #     comments)
  #   if do_log_training:
  #     log_training_commentary("training", networks.teacher.tensorboard_writer_training, step, losses, networks.teacher.model_commentary_decoder.model)

  def log_scalars(self, network, step, names, values):
    network.ensure_training()
    writer = network.tensorboard_writer_validation
    if writer:
      with writer.as_default():
        tf.summary.experimental.set_step(step)
        for name, value in zip(names, values):
          tf.summary.scalar(name.decode("utf-8"), value)

  def should_log_graph(self, step):
    return (step == 1)

  def log_training_prepare(self, step):
    if self.should_log_graph(step):
      tf.summary.trace_on(graph=True, profiler=False) # TODO: Dangerous without if-writer check, remove all of those

  def log_training(self, type, writer, step, losses, model):
    self.log(f"Loss: {losses[0]:.4f} (V: {losses[1]:.4f}, MV: {losses[2]:.4f}, P: {losses[3]:.4f}, RP: {losses[4]:.4f}), Acc. (P): {losses[5]:.4f}, Acc. (RP): {losses[6]:.4f} ({type})")
    if writer:
      with writer.as_default():
        tf.summary.experimental.set_step(step)
        if self.should_log_graph(step):
          tf.summary.trace_export("model")
        self.log_loss_accuracy(losses)
        self.log_weights(model)
        writer.flush()

    # Reset metrics after "validation_interval" training batches or 1 validation batch, ready for the next "validation_interval"/1.
    model.reset_metrics()

  def log_training_commentary(self, type, writer, step, losses, model):
    self.log(f"Loss: {losses[0]:.4f}, Accuracy: {losses[1]:.4f} ({type})")
    if writer:
      with writer.as_default():
        tf.summary.experimental.set_step(step)
        self.log_loss_accuracy_commentary(losses)
        self.log_weights(model)
        writer.flush()

  def log_loss_accuracy(self, losses):
    # Fix losses: only total includes loss weighting.
    with tf.name_scope("loss"):
      tf.summary.scalar("overall loss", losses[0])
      tf.summary.scalar("value loss", losses[1])
      tf.summary.scalar("mcts value loss", losses[2])
      tf.summary.scalar("policy loss", losses[3])
      tf.summary.scalar("reply policy loss", losses[4])
      # Equivalent to tf.math.add_n(model.losses)
      loss_weights = [self.config.training_network["value_loss_weight"], self.config.training_network["mcts_value_loss_weight"],
        self.config.training_network["policy_loss_weight"], self.config.training_network["reply_policy_loss_weight"]]
      tf.summary.scalar("L2 loss", losses[0] - (losses[1] * loss_weights[0]) - (losses[2] * loss_weights[1]) - (losses[3] * loss_weights[2]) - (losses[4] * loss_weights[3])) 
    with tf.name_scope("accuracy"):
      tf.summary.scalar("policy accuracy", losses[5])
      tf.summary.scalar("reply policy accuracy", losses[6])

  def log_loss_accuracy_commentary(self, losses):
    with tf.name_scope("loss"):
      tf.summary.scalar("commentary loss", losses[0])
    with tf.name_scope("accuracy"):
      tf.summary.scalar("commentary accuracy", losses[1])

  def log_weights(self, model):
    for layer in model.layers:
      for weight in layer.weights:
        weight_name = weight.name.replace(':', '_')
        tf.summary.histogram(weight_name, weight)