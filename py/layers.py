import tensorflow as tf
from tensorflow.keras import backend as K
K.set_image_data_format("channels_first")
from attention import MultiHeadSelfAttention2D

class Residual:

  def __init__(self, layers, layer_names, filter_count, weight_decay):
    self.layers = layers
    self.layer_names = layer_names
    self.filter_count = filter_count
    self.weight_decay = weight_decay

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
    if not (block == 0 and piece == 0):
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

  def build_inception_conv2d(self, x, filters, kernel_size, batch_normalize, name):
    use_bias = not batch_normalize
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same", data_format="channels_first",
      use_bias=use_bias, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
      name=f"{name}/conv2d_{filters}")(x)
    if batch_normalize:
      x = tf.keras.layers.BatchNormalization(axis=1, name=f"{name}/batchnorm")(x)
      x = tf.keras.layers.ReLU(name=f"{name}/relu")(x)
    return x

  def build_inception_residual_block_v2(self, x, block):
    #x = tf.keras.layers.BatchNormalization(axis=1, name=f"residual_{block}/batchnorm")(x)
    x = tf.keras.layers.ReLU(name=f"residual_{block}/relu")(x)

    branch_0 = self.build_inception_conv2d(x, filters=24, kernel_size=(1, 1), batch_normalize=True, name=f"residual_{block}/branch_0_0")
    #branch_0 = MultiHeadSelfAttention2D(total_depth=24, num_heads=4, weight_decay=self.weight_decay, name=f"residual_{block}/branch_0_0/attention")(x)
    #branch_0 = tf.keras.layers.BatchNormalization(axis=1, name=f"residual_{block}/branch_0_0/batchnorm")(branch_0)
    #branch_0 = tf.keras.layers.ReLU(name=f"residual_{block}/branch_0_0/relu")(branch_0)

    branch_1 = self.build_inception_conv2d(x, filters=24, kernel_size=(1, 1), batch_normalize=True, name=f"residual_{block}/branch_1_0")
    branch_1 = self.build_inception_conv2d(branch_1, filters=24, kernel_size=(3, 3), batch_normalize=True, name=f"residual_{block}/branch_1_1")

    branch_2 = self.build_inception_conv2d(x, filters=24, kernel_size=(1, 1), batch_normalize=True, name=f"residual_{block}/branch_2_0")
    branch_2 = self.build_inception_conv2d(branch_2, filters=32, kernel_size=(3, 3), batch_normalize=True, name=f"residual_{block}/branch_2_1")
    branch_2 = self.build_inception_conv2d(branch_2, filters=48, kernel_size=(3, 3), batch_normalize=True, name=f"residual_{block}/branch_2_2")

    branch_3 = self.build_inception_conv2d(x, filters=24, kernel_size=(1, 1), batch_normalize=True, name=f"residual_{block}/branch_3_0")
    branch_3 = self.build_inception_conv2d(branch_2, filters=30, kernel_size=(1, 9), batch_normalize=True, name=f"residual_{block}/branch_3_1")
    branch_3 = self.build_inception_conv2d(branch_2, filters=36, kernel_size=(9, 1), batch_normalize=True, name=f"residual_{block}/branch_3_2")

    concat = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=1)
    output = self.build_inception_conv2d(concat, filters=self.filter_count, kernel_size=(1, 1), batch_normalize=False,
      name=f"residual_{block}/output_conv2d_{self.filter_count}")
    x = tf.keras.layers.Add(name=f"residual_{block}/add")([x, output])
    # TODO: Try scale before relu in output, or try batchnorm
    return x

  def build_shufflenet_v2_residual_block_v2_tf(self, input, block):
    if block != 0:
      batch = tf.shape(input)[0]
      shape = input.shape
      input = tf.reshape(input, (batch, 2, shape[1] // 2, shape[2], shape[3]))
      input = tf.transpose(input, [0, 2, 1, 3, 4])
      input = tf.reshape(input, [batch, shape[1], shape[2], shape[3]])
    x, y = tf.split(input, 2, axis=1)

    z = tf.keras.layers.BatchNormalization(axis=1, name=f"residual_{block}/piece_0/batchnorm")(x)
    z = tf.keras.layers.ReLU(name=f"residual_{block}/piece_0/relu")(z)
    z = self.build_shufflenet_v2_pointwise(z, name=f"residual_{block}/piece_0")

    z = tf.keras.layers.BatchNormalization(axis=1, name=f"residual_{block}/piece_1/batchnorm")(z)
    z = tf.keras.layers.ReLU(name=f"residual_{block}/piece_1/relu")(z)
    z = self.build_shufflenet_v2_depthwise(z, name=f"residual_{block}/piece_1")

    z = tf.keras.layers.BatchNormalization(axis=1, name=f"residual_{block}/piece_2/batchnorm")(z)
    z = self.build_shufflenet_v2_pointwise(z, name=f"residual_{block}/piece_2")

    x = tf.keras.layers.Add(name=f"residual_{block}/add")([x, z])
    output = tf.concat([x, y], axis=1)
    return output

  def build_shufflenet_v2_pointwise(self, x, name):
    filters = self.filter_count // 2
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding="same", data_format="channels_first",
      use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
      name=f"{name}/conv2d_pw_{filters}")(x)
    return x

  def build_shufflenet_v2_depthwise(self, x, name):
    filters = self.filter_count // 2
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), depth_multiplier=1, padding="same", data_format="channels_first",
      use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
      name=f"{name}/conv2d_dw33_{filters}")(x)
    return x
