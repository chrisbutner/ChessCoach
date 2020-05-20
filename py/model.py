import tensorflow as tf
from tensorflow.keras import backend as K
K.set_image_data_format("channels_first")
from attention import MultiHeadSelfAttention2D
from layers import Residual

class ChessCoachModel:
  
  board_side = 8
  input_planes_count = 101
  output_planes_count = 73
  residual_count = 3 # 19 for full AlphaZero
  filter_count = 256 # 256 for full AlphaZero
  attention_heads = 16
  dense_count = 256
  weight_decay = 1e-4

  input_name = "Input"
  output_value_name = "OutputValue"
  output_policy_name = "OutputPolicy"

  fixup_scaled_initializer = lambda shape, dtype: tf.keras.initializers.he_normal()(shape, dtype) * (ChessCoachModel.residual_count ** -0.5)

  default_initializer = tf.keras.initializers.he_normal()
  residual_initializers = [fixup_scaled_initializer, tf.zeros_initializer()]
  classifier_initializer = tf.zeros_initializer()

  def __init__(self):
    self.architecture_layers = {
      "conv2d": self.conv2d_layer,
      "attention": self.attention_layer,
      "augmented": self.augmented_layer,
    }

  def conv2d_layer(self, name, initializer):
    return tf.keras.layers.Conv2D(filters=self.filter_count, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      use_bias=False, kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), name=name)
  
  def attention_layer(self, name):
    return MultiHeadSelfAttention2D(total_depth=self.filter_count, num_heads=self.attention_heads, weight_decay=self.weight_decay, name=name)
  
  def augmented_layer(self, name, initializer):
    attention = MultiHeadSelfAttention2D(total_depth=self.filter_count // 2, num_heads=self.attention_heads, weight_decay=self.weight_decay, name=name + "/attention")
    conv2d = tf.keras.layers.Conv2D(filters=self.filter_count // 2, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      use_bias=False, kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), name=name + "/conv2d")
    return lambda x: tf.concat([attention(x), conv2d(x)], axis=1)

  def stem(self, x):
    # Initial convolutional layer
    x = tf.keras.layers.Conv2D(filters=self.filter_count, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      name=f"stem/conv2d_{self.filter_count}",
      use_bias=False, kernel_initializer=self.default_initializer, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
    x = x + tf.Variable([0.0], dtype="float32", name="stem/bias0")
    x = tf.keras.layers.ReLU(name=f"initial/relu")(x)
    return x

  def tower(self, x, config):
    # Residual layers
    architecture = config.training_network["architecture"]
    architecture_layer = self.architecture_layers[architecture]
    residual = Residual([architecture_layer, architecture_layer], [architecture, architecture], self.filter_count)
    for i in range(self.residual_count):
      x = residual.build_residual_block_v2_fixup(x, i, self.residual_initializers)

    # Tower BN/ReLU
    x = x + tf.Variable([0.0], dtype="float32", name="tower/fixup_bias0")
    x = tf.keras.layers.ReLU(name=f"tower/relu")(x)
    return x

  def value_head(self, x):
    value_filter_count = 1
    x = x + tf.Variable([0.0], dtype="float32", name="value/fixup_bias0")
    x = tf.keras.layers.Conv2D(filters=value_filter_count, kernel_size=(1,1), strides=1, data_format="channels_first",
      name=f"value/conv2d_{value_filter_count}",
      use_bias=False, kernel_initializer=self.default_initializer, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
    x = x + tf.Variable([0.0], dtype="float32", name="value/fixup_bias1")
    x = tf.keras.layers.ReLU(name=f"value/relu")(x)
    x = tf.keras.layers.Flatten(name=f"value/flatten")(x)
    x = x + tf.Variable([0.0], dtype="float32", name="value/fixup_bias2")
    x = tf.keras.layers.Dense(self.dense_count, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), activation="relu", use_bias=False,
      kernel_initializer=self.default_initializer, name=f"value/dense_{self.dense_count}")(x)
    x = x + tf.Variable([0.0], dtype="float32", name="value/fixup_bias3")
    x = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), activation="tanh", use_bias=False,
      kernel_initializer=self.classifier_initializer, name=self.output_value_name)(x)
    return x

  def policy_head(self, x, config):
    architecture = config.training_network["architecture"]
    architecture_layer = self.architecture_layers[architecture]
    x = x + tf.Variable([0.0], dtype="float32", name="policy/fixup_bias0")
    x = architecture_layer(name=f"policy/{architecture}_{self.filter_count}", initializer=self.default_initializer)(x)
    x = x + tf.Variable([0.0], dtype="float32", name="policy/fixup_bias1")
    x = tf.keras.layers.ReLU(name=f"policy/relu")(x)
    x = x + tf.Variable([0.0], dtype="float32", name="policy/fixup_bias2")
    x = tf.keras.layers.Conv2D(filters=self.output_planes_count, kernel_size=(1,1), strides=1, data_format="channels_first",
      use_bias=False, kernel_initializer=self.classifier_initializer, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), name=self.output_policy_name)(x)
    return x

  def build(self, config):
    input = tf.keras.layers.Input(shape=(self.input_planes_count, self.board_side, self.board_side), dtype="float32", name=self.input_name)

    # Stem
    stem = self.stem(input)

    # Residual tower
    tower = self.tower(stem, config)

    # Value head
    value = self.value_head(tower)

    # Policy head
    policy = self.policy_head(tower, config)

    return tf.keras.Model(input, [value, policy])