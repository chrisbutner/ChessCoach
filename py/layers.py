import tensorflow as tf
from tensorflow.keras import backend as K
K.set_image_data_format("channels_first")

class Residual:

  def __init__(self, layers, layer_names, filter_count):
    self.layers = layers
    self.layer_names = layer_names
    self.filter_count = filter_count

  def build_residual_piece_v1(self, x, block, piece):
    x = self.layers[piece](name=f"residual_{block}/{self.layer_names[piece]}_{piece}_{self.filter_count}")(x)
    x = tf.keras.layers.BatchNormalization(axis=1, name=f"residual_{block}/batchnorm_{piece}")(x)
    return x

  # Requires a Conv2D/BN/ReLU before the tower.
  def build_residual_block_v1(self, x, block):
    y = self.build_residual_piece_v1(x, block, 0)
    y = tf.keras.layers.ReLU(name=f"residual_{block}/relu_0")(y)
    y = self.build_residual_piece_v1(y, block, 1)
    x = tf.keras.layers.Add(name=f"residual_{block}/add")([x, y])
    x = tf.keras.layers.ReLU(name=f"residual_{block}/relu_1")(x)
    return x

  def build_residual_piece_v2(self, x, block, piece):
    x = tf.keras.layers.BatchNormalization(axis=1, name=f"residual_{block}/batchnorm_{piece}")(x)
    x = tf.keras.layers.ReLU(name=f"residual_{block}/relu_{piece}")(x)
    x = self.layers[piece](name=f"residual_{block}/{self.layer_names[piece]}_{piece}_{self.filter_count}")(x)
    return x

  # Requires a Conv2D/BN/ReLU before the tower.
  # Requires an additional BN/ReLU after the tower.
  def build_residual_block_v2(self, x, block):
    y = self.build_residual_piece_v2(x, block, 0)
    y = self.build_residual_piece_v2(y, block, 1)
    x = tf.keras.layers.Add(name=f"residual_{block}/add")([x, y])
    return x