import tensorflow as tf
from tensorflow.keras import backend as K

K.set_image_data_format("channels_first")

class ChessCoachModel:

  board_side = 8
  input_planes_count = 25
  output_planes_count = 73
  residual_count = 7 # 19 for full AlphaZero
  filter_count = 128 # 256 for full AlphaZero
  dense_count = 256
  se_ratio = 8 # 16 default (probably use it for 256 filter_count)
  weight_decay = 1e-4

  input_name = "Input"
  output_value_name = "OutputValue"
  output_policy_name = "OutputPolicy"

  def build_se_block(self, x, block):
    se_filter_count = self.filter_count
    se = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first", name=f"residual_{block}/se/pooling")(x)
    se = tf.keras.layers.Dense(se_filter_count // self.se_ratio, activation="relu", use_bias=False, kernel_initializer="he_normal",
      name=f"residual_{block}/se/dense_0_{se_filter_count // self.se_ratio}",
      kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(se)
    se = tf.keras.layers.Dense(se_filter_count, activation="sigmoid", use_bias=False, kernel_initializer="he_normal",
      name=f"residual_{block}/se/dense_1_{se_filter_count}",
      kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(se)
    se = tf.keras.layers.Reshape((se_filter_count, 1, 1), name=f"residual_{block}/se/reshape")(se)

    x = tf.keras.layers.Multiply(name=f"residual_{block}/se/multiply")([x, se])
    return x

  def build_residual_piece(self, x, block, piece):
    x = tf.keras.layers.Conv2D(filters=self.filter_count, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      name=f"residual_{block}/conv2d_{piece}_{self.filter_count}",
      use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
    x = tf.keras.layers.BatchNormalization(axis=1, name=f"residual_{block}/batchnorm_{piece}")(x)
    return x

  def build_residual_block(self, x, block, se=True):
    y = self.build_residual_piece(x, block, 0)
    y = tf.keras.layers.ReLU(name=f"residual_{block}/relu_0")(y)
    y = self.build_residual_piece(y, block, 1)
    if (se):
      y = self.build_se_block(y, block)
    x = tf.keras.layers.Add(name=f"residual_{block}/add")([x, y])
    x = tf.keras.layers.ReLU(name=f"residual_{block}/relu_1")(x)
    return x

  def build(self):
    input = tf.keras.layers.Input(shape=(self.input_planes_count, self.board_side, self.board_side), dtype="float32", name=self.input_name)

    # Initial convolutional layer
    x = tf.keras.layers.Conv2D(filters=self.filter_count, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      name=f"initial/conv2d_{self.filter_count}",
      use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(input)
    x = tf.keras.layers.BatchNormalization(axis=1, name=f"initial/batchnorm")(x)
    x = tf.keras.layers.ReLU(name=f"initial/relu")(x)

    # Residual layers
    for i in range(self.residual_count):
      x = self.build_residual_block(x, i)
    tower = x

    # Value head
    value_filter_count = 1
    x = tf.keras.layers.Conv2D(filters=value_filter_count, kernel_size=(1,1), strides=1, data_format="channels_first",
      name=f"value/conv2d_{value_filter_count}",
      use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(tower)
    x = tf.keras.layers.BatchNormalization(axis=1, name=f"value/batchnorm")(x)
    x = tf.keras.layers.ReLU(name=f"value/relu")(x)
    x = tf.keras.layers.Flatten(name=f"value/flatten")(x)
    x = tf.keras.layers.Dense(self.dense_count, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), activation="relu", use_bias=False,
      name=f"value/dense_{self.dense_count}")(x)
    value = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), activation="tanh", use_bias=False,
      name=self.output_value_name)(x)

    # Policy head
    x = tf.keras.layers.Conv2D(filters=self.filter_count, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      name=f"policy/conv2d_{self.filter_count}",
      use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(tower)
    x = tf.keras.layers.BatchNormalization(axis=1, name=f"policy/batchnorm")(x)
    x = tf.keras.layers.ReLU(name=f"policy/relu")(x)
    policy = tf.keras.layers.Conv2D(filters=self.output_planes_count, kernel_size=(1,1), strides=1, data_format="channels_first",
      use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), name=self.output_policy_name)(x)

    return tf.keras.Model(input, [value, policy])
