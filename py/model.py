import tensorflow as tf
from tensorflow.keras import backend as K
K.set_image_data_format("channels_first")
from attention import MultiHeadSelfAttention2D
import transformer
from official.nlp.modeling import models
from layers import Residual
import os

class ModelBuilder:
  
  board_side = 8
  input_planes_count = 101
  output_planes_count = 73
  residual_count = 19
  filter_count = 256
  attention_heads = 8
  dense_count = 256
  weight_decay = 1e-4

  transformer_layers = 6
  transformer_filters = 512
  transformer_heads = 8
  transformer_feedforward = 2048
  transformer_dropout_rate = 0.1
  transformer_vocabulary_size = 8000
  transformer_max_length = 128

  no_progress_saturation_count = 99.0

  input_name = "Input"
  output_value_name = "OutputValue"
  output_mcts_value_name = "OutputMctsValue"
  output_policy_name = "OutputPolicy"
  output_commentary_encoder_name = "OutputCommentaryEncoder"
  
  token_start = "<start>"
  token_end = "<end>"
  token_unk = "<unk>"
  token_pad = "<pad>"

  def __init__(self):
    self.architecture_layers = {
      "conv2d": self.conv2d_layer,
      "attention": self.attention_layer,
      "augmented": self.augmented_layer,
    }

  def conv2d_layer(self, name):
    return tf.keras.layers.Conv2D(filters=self.filter_count, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), name=name)
  
  def attention_layer(self, name):
    return MultiHeadSelfAttention2D(total_depth=self.filter_count, num_heads=self.attention_heads, weight_decay=self.weight_decay, name=name)
  
  def augmented_layer(self, name):
    attention = MultiHeadSelfAttention2D(total_depth=self.filter_count // 2, num_heads=self.attention_heads, weight_decay=self.weight_decay, name=name + "/attention")
    conv2d = tf.keras.layers.Conv2D(filters=self.filter_count // 2, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), name=name + "/conv2d")
    return lambda x: tf.concat([attention(x), conv2d(x)], axis=1)

  def unpack_planes(self, x):
    shape = x.get_shape()
    x = tf.expand_dims(x, -1)
    mask = tf.bitwise.left_shift(tf.ones([], dtype=tf.int64), tf.range(64, dtype=tf.int64))
    x = tf.bitwise.bitwise_and(x, mask)
    x = tf.cast(x, tf.bool)
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, [-1, int(shape[1]), self.board_side, self.board_side])
    return x

  def stem(self, x):
    # Initial convolutional layer
    x = tf.keras.layers.Conv2D(filters=self.filter_count, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      name=f"initial/conv2d_{self.filter_count}",
      use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
    x = tf.keras.layers.BatchNormalization(axis=1, name=f"initial/batchnorm")(x)
    x = tf.keras.layers.ReLU(name=f"initial/relu")(x)
    return x

  def tower(self, x, config):
    # Residual layers
    architecture = config.training["architecture"]
    architecture_layer = self.architecture_layers[architecture]
    residual = Residual([architecture_layer, architecture_layer], [architecture, architecture], self.filter_count, self.weight_decay)
    for i in range(self.residual_count):
      x = residual.build_residual_block_v2(x, i)

    # Tower BN/ReLU
    x = tf.keras.layers.BatchNormalization(axis=1, name=f"tower/batchnorm")(x)
    x = tf.keras.layers.ReLU(name=f"tower/relu")(x)
    return x

  def value_head(self, x, name, output_name):
    value_filter_count = 1
    x = tf.keras.layers.Conv2D(filters=value_filter_count, kernel_size=(1,1), strides=1, data_format="channels_first",
      name=f"{name}/conv2d_{value_filter_count}",
      use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
    x = tf.keras.layers.BatchNormalization(axis=1, name=f"{name}/batchnorm")(x)
    x = tf.keras.layers.ReLU(name=f"{name}/relu")(x)
    x = tf.keras.layers.Flatten(name=f"{name}/flatten")(x)
    # Add bias for these layers with no more batchnorms.
    x = tf.keras.layers.Dense(self.dense_count, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), activation="relu", use_bias=True,
      name=f"{name}/dense_{self.dense_count}")(x)
    x = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), activation="tanh", use_bias=True,
      name=output_name)(x)
    return x

  def policy_head(self, x, name, output_name, config):
    architecture = config.training["architecture"]
    architecture_layer = self.architecture_layers[architecture]
    x = architecture_layer(name=f"{name}/{architecture}_{self.filter_count}")(x)
    x = tf.keras.layers.BatchNormalization(axis=1, name=f"{name}/batchnorm")(x)
    x = tf.keras.layers.ReLU(name=f"{name}/relu")(x)
    # Add bias for these layers with no more batchnorms.
    policy = tf.keras.layers.Conv2D(filters=self.output_planes_count, kernel_size=(1,1), strides=1, data_format="channels_first",
      use_bias=True, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), name=output_name)(x)
    return policy

  def commentary_encoder_head(self, x, name, output_name):
    # Just reshape to 1D-sequence, channels-last.
    x = tf.reshape(x, [-1, self.filter_count, self.board_side * self.board_side])
    encoder = tf.transpose(x, [0, 2, 1], name=output_name)
    return encoder

  def build(self, config, subclass=None, residual_count=None, filter_count=None, dense_count=None):
    subclass = subclass or tf.keras.Model
    if residual_count:
      self.residual_count = residual_count
    if filter_count:
      self.filter_count = filter_count
    if dense_count:
      self.dense_count = dense_count
    
    input = tf.keras.layers.Input(shape=(self.input_planes_count), dtype=tf.int64, name=self.input_name)

    # Unpack bit-planes of (batch, 101) int64 to (batch, 101, 8, 8) float32.
    # However, the final plane is a special case, not to be bit-unpacked, but instead interpreted as an integer
    # to be normalized via the no-progress saturation count (99).
    standard_planes = self.unpack_planes(input[:, :-1])
    special_planes = tf.cast(input[:, -1:], tf.float32)[:, :, tf.newaxis, tf.newaxis] / tf.fill([1, self.board_side, self.board_side], self.no_progress_saturation_count)
    planes = tf.concat([standard_planes, special_planes], axis=1)

    # Stem
    stem = self.stem(planes)

    # Residual tower
    tower = self.tower(stem, config)

    # Value heads
    value = self.value_head(tower, "value", self.output_value_name)
    mcts_value = self.value_head(tower, "mcts_value", self.output_mcts_value_name)

    # Policy head
    policy = self.policy_head(tower, "policy", self.output_policy_name, config)

    # Commentary encoder head
    commentary_encoder = self.commentary_encoder_head(tower, "commentary_encoder", self.output_commentary_encoder_name)

    return subclass(input, [value, mcts_value, policy, commentary_encoder])

  def build_student(self, config, subclass):
    return self.build(config, subclass, residual_count=8, filter_count=128, dense_count=128)

  def subset_train(self, model):
    return type(model)(model.input, model.outputs[:3])

  def subset_predict(self, model):
    return type(model)(model.input, model.outputs[0:4:2])

  def subset_commentary_encoder(self, model):
    return type(model)(model.input, model.outputs[3:])

  def build_commentary(self, config, tokenizer, model_full):
    eos_id = tokenizer.tokenize("").numpy().item()
    commentary_encoder = self.subset_commentary_encoder(model_full)
    decoder_layer = models.TransformerDecoder(
      num_layers=self.transformer_layers,
      num_attention_heads=self.transformer_heads,
      intermediate_size=self.transformer_feedforward,
      dropout_rate=self.transformer_dropout_rate,
      attention_dropout_rate=self.transformer_dropout_rate,
      intermediate_dropout=self.transformer_dropout_rate,
      )
    commentary_decoder = transformer.CommentaryDecoder(
      vocab_size=self.transformer_vocabulary_size,
      decoder_layer=decoder_layer,
      eos_id=eos_id,
      decode_max_length=self.transformer_max_length,
      embedding_width=self.transformer_filters,
      dropout_rate=self.transformer_dropout_rate,
    )
    commentary_model = transformer.CommentaryModel(commentary_encoder, commentary_decoder)

    # Call once to prepare the model for loading weights.
    commentary_model(dict(
      inputs=tf.ones((config.training["commentary_batch_size"], self.input_planes_count), tf.int64),
      targets=tf.ones((config.training["commentary_batch_size"], self.transformer_max_length), tf.int32)))

    return commentary_model