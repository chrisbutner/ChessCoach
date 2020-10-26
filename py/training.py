import time
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import transformer

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
  # See "Trainer.stack_teacher_targets" for stacking details.
  teacher_logits = K.batch_flatten(y_true[:, 0])
  true_labels = K.batch_flatten(y_true[:, 1])
  y_pred = K.batch_flatten(y_pred)

  teacher_loss = knowledge_distillation_temperature**2 * tf.keras.losses.categorical_crossentropy(
    y_true=tf.stop_gradient(tf.nn.softmax(teacher_logits/knowledge_distillation_temperature)), y_pred=y_pred/knowledge_distillation_temperature, from_logits=True)
  true_loss = tf.keras.losses.categorical_crossentropy(y_true=true_labels, y_pred=y_pred, from_logits=True)
  return (knowledge_distillation_teacher_weight * teacher_loss) + ((1 - knowledge_distillation_teacher_weight) * true_loss)

# Accuracy against ground truth (argmax).
def student_policy_accuracy(y_true, y_pred):
  # See "Trainer.stack_teacher_targets" for stacking details.
  return flat_categorical_accuracy(y_true[:, 1], y_pred)

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
    self.data_commentary_training = None
    self.data_globs = {
      # GameType_Supervised (Config.h)
      0: self.config.join(self.config.training["games_path_supervised"], "*.chunk"),
      # GameType_Training (Config.h)
      1: self.config.join(self.config.training["games_path_training"], "*.chunk"),
    }
    self.data_glob_validation = self.config.join(self.config.training["games_path_validation"], "*.chunk")
    # Commentary is supervised-only for now.
    self.data_glob_commentary = self.config.join(self.config.training["commentary_path_supervised"], "*.chunk")

    self.per_replica_batch_size = self.config.training["batch_size"]
    self.global_batch_size = self.per_replica_batch_size * self.device_count

    self.per_replica_batch_size_commentary = self.config.training["commentary_batch_size"]
    self.global_batch_size_commentary = self.per_replica_batch_size_commentary * self.device_count

    self.log("Devices:", self.device_count)
    self.log("Per-replica batch size:", self.per_replica_batch_size)
    self.log("Global batch size:", self.global_batch_size)
    self.log("Per-replica batch size (commentary):", self.per_replica_batch_size_commentary)
    self.log("Global batch size (commentary):", self.global_batch_size_commentary)

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
  def get_learning_rate_common(self, schedule, step):
    scale_learning_rate_with_batch_size = self.device_count
    warmup = np.interp(step, [0, self.config.training["warmup_steps"]], [0.0, 1.0]).item()
    multiplier = scale_learning_rate_with_batch_size * warmup

    rate = 0.0
    for key, value in schedule:
      if step >= key:
        rate = value
      else:
        break
    return rate * multiplier

  def get_learning_rate(self, step):
    schedule = self.config.training["learning_rate_schedule"]
    return self.get_learning_rate_common(schedule, step)

  def get_commentary_learning_rate(self, step):
    schedule = self.config.training["commentary_learning_rate_schedule"]
    return self.get_learning_rate_common(schedule, step)

  # The teacher network trains directly on supervised labels.
  def compile_teacher(self, model):
    optimizer = tf.keras.optimizers.SGD(
      learning_rate=self.get_learning_rate(0),
      momentum=self.config.training["momentum"])
    losses = ["mean_squared_error", "mean_squared_error", flat_categorical_crossentropy_from_logits, flat_categorical_crossentropy_from_logits]
    loss_weights = [self.config.training["value_loss_weight"], self.config.training["mcts_value_loss_weight"],
      self.config.training["policy_loss_weight"], self.config.training["reply_policy_loss_weight"]]
    metrics = [[], [], [flat_categorical_accuracy], [flat_categorical_accuracy]]
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

  # The student network trains on a combination of soft teacher labels and hard supervised labels.
  # Policy accuracy is still measured against the supervised labels.
  def compile_student(self, model):
    optimizer = tf.keras.optimizers.SGD(
      learning_rate=self.get_learning_rate(0),
      momentum=self.config.training["momentum"])
    losses = ["mean_squared_error", "mean_squared_error", student_policy_loss, student_policy_loss]
    loss_weights = [self.config.training["value_loss_weight"], self.config.training["mcts_value_loss_weight"],
      self.config.training["policy_loss_weight"], self.config.training["reply_policy_loss_weight"]]
    metrics = [[], [], [student_policy_accuracy], [student_policy_accuracy]]
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

  # NOTE: This can be extremely slow right now (e.g. ~500 seconds after 1000 steps on 2**17 shuffle buffer)
  # but is necessary when training and self-playing on a single machine to ensure enough memory for
  # either (a) a 32-cycle_length of chunk loads plus a 2**17-shuffle buffer (~ 30 GB with overhead)
  # or (b) num_workers * prediction_batch_size games in parallel worth of Stockfish StateInfos and MCTS Nodes.
  def clear_data(self):
    if self.data_training_key or self.data_commentary_training:
      self.log("Clearing datasets")
    self.data_training = None
    self.data_validation = None
    self.data_training_key = None
    self.data_commentary_training = None

  def train(self, network, teacher_network, game_types, training_windows, starting_step, checkpoint):
    # Create models on the distribution strategy scope, including the teacher for knowledge distillation inference.
    with self.strategy.scope():
      model = network.ensure_training()
      if teacher_network:
        teacher_network.ensure_training()

    # Set up data pipelines, or re-use existing if not cleared by self-play.
    # Numpy hijacks == and bool testing and enforces any() or all(), so convert to lists first.
    data_training_key = (game_types.tolist(), training_windows.tolist())
    if data_training_key != self.data_training_key:
      data_start = time.time()
      globs_training = [self.data_globs[t] for t in game_types]
      globs_validation = [self.data_glob_validation]
      self.data_training = iter(self.datasets.build_training_dataset(globs_training, training_windows, self.global_batch_size))
      self.data_validation = iter(self.datasets.build_validation_dataset(globs_validation, self.global_batch_size))
      self.data_training_key = data_training_key
      next(self.data_training)
      next(self.data_validation)
      print(f"Datasets prepared in {(time.time() - data_start):.2f} seconds")

    # Iterate over all steps in the checkpoint, striding by the device count.
    # E.g. on a GPU, 1 device, 1-1000 steps, range(1, 1001, 1), 1000 iterations
    # E.g. on a TPU, 8 devices, 1-1000 steps, range(1, 1001, 8), 125 iterations, 125*8=1000 effective iterations
    for step in range(starting_step, checkpoint + 1, self.device_count):
      # Set the learning rate.
      K.set_value(model.optimizer.lr, self.get_learning_rate(step))

      # Train.
      batch_training = next(self.data_training)
      self.train_batch(network, teacher_network, model, step, images=batch_training[0], targets=batch_training[1])

      # Sometimes validate.
      if self.is_validation_step(step):
        batch_validation = next(self.data_validation)
        self.validate_batch(network, teacher_network, model, step, images=batch_validation[0], targets=batch_validation[1])

  def is_validation_step(self, step):
    # Striding steps by the device count, so validate when at or just past the validation interval.
    # E.g. on a GPU, 1 device, 100 validation interval, steps 1, 2, 3, etc., validate on step 100.
    # E.g. on a TPU, 8 devices, 100 validation interval, steps 1, 9, 17, etc., validate on step 105.
    return ((step % self.config.training["validation_interval"]) < self.device_count)

  def train_batch(self, network, teacher_network, model, step, images, targets):
    # Prepare TensorBoard logging.
    do_log_training = self.is_validation_step(step)
    if do_log_training:
      self.log_training_prepare(step)

    # If a teacher was provided, predict soft targets and combine with the provided hard targets.
    if teacher_network:
      targets = self.stack_teacher_targets(teacher_network, images, targets)

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
      targets = self.stack_teacher_targets(teacher_network, images, targets)

    # Train the model.
    losses = model.test_on_batch(images, targets, reset_metrics=False)

    # Do Tensorboard logging.
    self.log_training("validation", network.tensorboard_writer_validation, step, losses, model)

  def stack_teacher_targets(self, teacher_network, images, targets):
    # Predict teacher logits.
    distributed_images = self.distribute_batch(images)
    _, _, teacher_policies, teacher_reply_policies = self.strategy.run(
      teacher_network.tf_predict_for_training, args=(distributed_images,))
    if self.device_count > 1:
      teacher_policies = self.collect_batch(teacher_policies)
      teacher_reply_policies = self.collect_batch(teacher_reply_policies)

    # Stack with provided labels in axis=1 so that axis=0 can remain the batch axis
    # and be auto-distributed to replicas by train_on_batch/test_on_batch.
    values, mcts_values, policies, reply_policies = targets
    policies = tf.stack([teacher_policies, policies], axis=1)
    reply_policies = tf.stack([teacher_reply_policies, reply_policies], axis=1)
    targets = (values, mcts_values, policies, reply_policies)
    return targets

  def distribute_batch(self, x):
    return next(iter(self.strategy.experimental_distribute_dataset(tf.data.Dataset.from_tensors(x))))

  def collect_batch(self, x):
    return tf.concat(x.values, axis=0)

  def train_commentary(self, network, starting_step, checkpoint):
    with self.strategy.scope():
      network.ensure_commentary()

      # Set up the commentary optimizer in strategy scope, for use with GradientTape in transformer.py.
      if not self.commentary_optimizer:
        self.commentary_optimizer = tf.keras.optimizers.SGD(
          learning_rate=self.get_commentary_learning_rate(0),
          momentum=self.config.training["momentum"])

      # Set up the commentary metrics in strategy scope.
      transformer.create_metrics()

    # Set up data pipelines, or re-use existing if not cleared by self-play.
    if not self.data_commentary_training:
      data_start = time.time()
      tokenizer = network.models_train.commentary_tokenizer
      self.data_commentary_training = iter(self.datasets.build_commentary_dataset(
        self.data_glob_commentary, self.global_batch_size_commentary, tokenizer, self.strategy))
      next(self.data_commentary_training)
      print(f"Datasets prepared in {(time.time() - data_start):.2f} seconds (commentary)")

    # Set up the training step tf.function.
    optimizer = self.commentary_optimizer
    encoder = network.models_train.commentary_encoder
    decoder = network.models_train.commentary_decoder

    @tf.function
    def train_step(images, comments):
      return transformer.train_step(optimizer, encoder, decoder, images, comments, num_replicas=self.device_count)

    # Iterate over all steps in the checkpoint, striding by the device count.
    # E.g. on a GPU, 1 device, 1-1000 steps, range(1, 1001, 1), 1000 iterations
    # E.g. on a TPU, 8 devices, 1-1000 steps, range(1, 1001, 8), 125 iterations, 125*8=1000 effective iterations
    for step in range(starting_step, checkpoint + 1, self.device_count):
      # Set the learning rate.
      K.set_value(optimizer.lr, self.get_commentary_learning_rate(step))

      # Train.
      batch_training = next(self.data_commentary_training)
      self.train_commentary_batch(network, decoder, train_step, step, images=batch_training[0], comments=batch_training[1])

  def train_commentary_batch(self, network, decoder, train_step, step, images, comments):
    # Prepare TensorBoard logging.
    do_log_training = self.is_validation_step(step)
    if do_log_training:
      self.log_training_prepare(step)

    losses = self.strategy.run(train_step, args=(images, comments))
    losses = (
      self.strategy.reduce(tf.distribute.ReduceOp.SUM, losses[0], axis=None),  # Sum loss over replicas.
      self.strategy.reduce(tf.distribute.ReduceOp.MEAN, losses[1], axis=None), # Average accuracy over replicas.
    )

    # Do Tensorboard logging.
    if do_log_training:
      self.log_training_commentary("training", network.tensorboard_writer_training, step, losses, decoder)

  def log_scalars(self, network, step, names, values):
    network.ensure_training()
    writer = network.tensorboard_writer_validation
    with writer.as_default():
      tf.summary.experimental.set_step(step)
      for name, value in zip(names, values):
        tf.summary.scalar(name.decode("utf-8"), value)

  def should_log_graph(self, step):
    return (step == 1)

  def log_training_prepare(self, step):
    if self.should_log_graph(step):
      tf.summary.trace_on(graph=True, profiler=False)

  def log_training(self, type, writer, step, losses, model):
    self.log(f"Loss: {losses[0]:.4f} (V: {losses[1]:.4f}, MV: {losses[2]:.4f}, P: {losses[3]:.4f}, RP: {losses[4]:.4f}), Acc. (P): {losses[5]:.4f}, Acc. (RP): {losses[6]:.4f} ({type})")
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
      loss_weights = [self.config.training["value_loss_weight"], self.config.training["mcts_value_loss_weight"],
        self.config.training["policy_loss_weight"], self.config.training["reply_policy_loss_weight"]]
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