# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
##############################################################################
#
# Modifications by Chris Butner, 2020.
#
# Based on the following "tensor2tensor" files:
# - common_attention.py
# - common_layers.py
#
# Based on the following papers and included code:
# - https://arxiv.org/pdf/1906.05909.pdf
# - https://arxiv.org/pdf/1904.09925.pdf
#

import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
K.set_image_data_format("channels_first")

class MultiHeadSelfAttention2D(keras.layers.Layer):

  def __init__(self, total_depth, num_heads, weight_decay, layer_name):
    super(MultiHeadSelfAttention2D, self).__init__()

    if total_depth % num_heads != 0:
      raise ValueError("Total depth (%d) must be divisible by the number of "
                      "attention heads (%d)." % (total_depth, num_heads))

    self.total_depth = total_depth
    self.num_heads = num_heads
    self.weight_decay = weight_decay
    self.layer_name = layer_name

    self.depth_per_head = total_depth // num_heads

  def build(self, input_shape):
    print("Build:", input_shape)
    length = input_shape[2]

    # Use a relative position embedding for keys.
    self.relative_position_embeddings = self._generate_relative_positions_embeddings_2d(
        length, self.layer_name + "/relative_positions")

    # Prepare to compute qkv (initialization from tensor2tensor, compute_attention_component).
    kv_initializer_stddev = self.total_depth ** -0.5
    q_initializer_stddev = kv_initializer_stddev * (self.depth_per_head ** -0.5)
    self.query = keras.layers.Conv2D(filters=self.total_depth, kernel_size=(1, 1), data_format="channels_first",
      name=f"{self.layer_name}/query_{self.total_depth}", use_bias=False, kernel_initializer=keras.initializers.RandomNormal(stddev=q_initializer_stddev),
      kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))
    self.key = keras.layers.Conv2D(filters=self.total_depth, kernel_size=(1, 1), data_format="channels_first",
      name=f"{self.layer_name}/key_{self.total_depth}", use_bias=False, kernel_initializer=keras.initializers.RandomNormal(stddev=kv_initializer_stddev),
      kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))
    self.value = keras.layers.Conv2D(filters=self.total_depth, kernel_size=(1, 1), data_format="channels_first",
      name=f"{self.layer_name}/value_{self.total_depth}", use_bias=False, kernel_initializer=keras.initializers.RandomNormal(stddev=kv_initializer_stddev),
      kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))

    # Project heads with final 1x1 convolution.
    self.combined_output = keras.layers.Conv2D(filters=self.total_depth, kernel_size=(1, 1), data_format="channels_first",
      name=f"{self.layer_name}/output_{self.total_depth}", use_bias=False, kernel_initializer=keras.initializers.RandomNormal(stddev=q_initializer_stddev),
      kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))

    super(MultiHeadSelfAttention2D, self).build(input_shape)

  def call(self, input):
    # Expect input with shape [batch, total_depth, length, length].
    q = self.query(input)
    k = self.key(input)
    v = self.value(input)

    # After splitting, shape is [batch, heads, length, length, depth_per_head].
    q = self.split_heads_2d(q, self.num_heads)
    k = self.split_heads_2d(k, self.num_heads)
    v = self.split_heads_2d(v, self.num_heads)

    # Scale down a la Transformer to help with gradients.
    q *= self.depth_per_head**-0.5

    # Calculate actual attention, then recombine to [batch, total_depth, length, length].
    x = self.dot_product_attention_relative_2d(q, k, v)
    x = self.combine_heads_2d(x)

    # Project heads: mix across concatenated head-groups.
    x = self.combined_output(x)
    return x

  def dot_product_attention_relative_2d(self, q, k, v):
    """Calculate relative position-aware dot-product self-attention (2D).

    The attention calculation is augmented with learned representations for the
    relative position between each element in q and each element in k.

    Args:
      q: a Tensor with shape [batch, heads, length, length, depth_per_head].
      k: a Tensor with shape [batch, heads, length, length, depth_per_head].
      v: a Tensor with shape [batch, heads, length, length, depth_per_head].

    Returns:
      A Tensor.
    """
    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    q.get_shape().assert_is_compatible_with(k.get_shape())
    q.get_shape().assert_is_compatible_with(v.get_shape())

    # Compute self attention considering the relative position embeddings.
    length = self.shape_list(k)[2]
    logits = self._relative_attention_inner_2d(q, k, self.relative_position_embeddings)
    logits = tf.reshape(logits, (-1, self.num_heads, length, length, length * length))
    weights = tf.nn.softmax(logits)
    v = tf.reshape(v, (-1, self.num_heads, length * length, self.depth_per_head))
    return tf.einsum("bnijp,bnpd->bnijd", weights, v)

  def _relative_attention_inner_2d(self, x, y, z):
    """Relative position-aware dot-product attention inner calculation.

    This batches matrix multiply calculations to avoid unnecessary broadcasting.

    Args:
      x: Tensor with shape [batch_size, heads, length, length, depth_per_head].
      y: Tensor with shape [batch_size, heads, length, length, depth_per_head].
      z: Tensor with shape [length, length, length, length, depth_per_head].

    Returns:
      A Tensor with shape [batch_size, heads, length, length, length, length].
    """

    # xy_matmul is [batch_size, heads, length, length, length, length]
    xy_matmul = tf.einsum("bnijd,bnpqd->bnijpq", x, y)
    # xz_matmul is [batch_size, heads, length, length, length, length]
    xz_matmul = tf.einsum("bnijd,ijpqd->bnijpq", x, z)
    return xy_matmul + xz_matmul

  def split_heads_2d(self, x, num_heads):
    """Split channels into multiple heads.

    Args:
      x: a Tensor with shape [batch, channels, height, width]
      num_heads: an integer

    Returns:
      a Tensor with shape [batch, num_heads, height, width, channels / num_heads]
    """
    length = self.shape_list(x)[2]

    # x is [batch, num_heads, depth_per_head, length, length]
    x = tf.reshape(x, (-1, self.num_heads, self.depth_per_head, length, length))

    # x is [batch, num_heads, length, length, depth_per_head]
    x = tf.transpose(x, [0, 1, 3, 4, 2])
    return x

  def combine_heads_2d(self, x):
    """Inverse of split_heads_2d.

    Args:
      x: a Tensor with shape
        [batch, num_heads, height, width, channels / num_heads]

    Returns:
      a Tensor with shape [batch, channels, height, width]
    """
    length = self.shape_list(x)[2]

    # x is [batch, num_heads, depth_per_head, length, length]
    x = tf.transpose(x, [0, 1, 4, 2, 3])

    # x is [batch, channels, length, length]
    x = tf.reshape(x, (-1, self.total_depth, length, length))
    return x

  def shape_list(self, x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
      return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i, dim in enumerate(static):
      if dim is None:
        dim = shape[i]
      ret.append(dim)
    return ret

  def _generate_relative_positions_embeddings_2d(self, length, name):
    """Generates tensor of size [length, length, depth_per_head]."""
    vocab_size = length * 2 - 1
    dimension_depth = self.depth_per_head // 2
    # Generates embedding for each relative position of dimension depth
    # (initialization from https://arxiv.org/pdf/1904.09925.pdf).
    embeddings_initializer_stddev = self.depth_per_head ** -0.5
    embeddings_table_h = tf.get_variable(name + "/embeddings_h", [vocab_size, dimension_depth],
      trainable=True, dtype="float32", initializer=tf.random_normal_initializer(stddev=embeddings_initializer_stddev))
    embeddings_table_w = tf.get_variable(name + "/embeddings_w", [vocab_size, dimension_depth],
      trainable=True, dtype="float32", initializer=tf.random_normal_initializer(stddev=embeddings_initializer_stddev))
    relative_positions_matrix = self._generate_relative_positions_matrix(length)
    embeddings_h = tf.gather(embeddings_table_h, relative_positions_matrix)
    embeddings_h = tf.pad(embeddings_h, [[0, 0], [0, 0], [0, dimension_depth]])
    embeddings_h = embeddings_h[:, None, :, None]
    embeddings_w = tf.gather(embeddings_table_w, relative_positions_matrix)
    embeddings_w = tf.pad(embeddings_w, [[0, 0], [0, 0], [dimension_depth, 0]])
    embeddings_w = embeddings_w[None, :, None, :]
    embeddings = embeddings_h + embeddings_w
    return embeddings

  def _generate_relative_positions_matrix(self, length):
    """Generates matrix of relative positions between inputs."""
    range_vec_q = range_vec_k = tf.range(length)
    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat + length - 1
    return final_mat