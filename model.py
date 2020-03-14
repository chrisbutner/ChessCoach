import tensorflow as tf

# TODO: lots of redundancy in regularizer, config stuff - extract out

class ChessCoachModel:

  def __init__(self):
    self.model = None

  def build_residual_piece(self, x):
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    return x

  def build_residual_block(self, x):
    y = self.build_residual_piece(x)
    y = tf.keras.layers.ReLU()(y)
    y = self.build_residual_piece(y)
    x = tf.keras.layers.Add()([x, y])
    x = tf.keras.layers.ReLU()(x)
    return x

  def build(self):
    input = tf.keras.layers.Input(shape=(12,8,8)) # Just pieces for now; dtype?

    # Zeta36 doesn't use bias, why? https://keras.io/examples/cifar10_resnet/ seems to
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input)
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.ReLU()(x)

    for _ in range(7): # 19 for full AlphaZero
      x = self.build_residual_block(x)
    tower = x

    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=1, data_format="channels_first", 
      kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(tower)
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation="relu")(x)
    x = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation="tanh")(x)
    value = x

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding="same", data_format="channels_first",
      kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(tower)
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=73, kernel_size=(1,1), strides=1, data_format="channels_first",
      kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    policy = x

    self.model = tf.keras.Model(input, [value, policy])
