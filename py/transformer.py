# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# Modifications by Chris Butner, 2021.
#
# Based on tensorflow/models/official/nlp/modeling/models/seq2seq_transformer.py
# and tensorflow/models/official/nlp/tasks/translation.py.
#
# Cuts out the transformer encoder, replaces it with a plugin encoder,
# and structures everything so that weights can be saved and loaded
# excluding the encoder plugin.
#
# Original code looked pluggable, but assumed that the raw input was already
# in the form of a sequence before encoding (e.g. in constructing cross-attention
# masks), so heavier changes were needed inside call().
#

import tensorflow as tf
from tensorflow.python.keras import backend as K
from official.modeling import tf_utils
from official.nlp import keras_nlp
from official.nlp.modeling import layers
from official.nlp.modeling.ops import sampling_module
from official.nlp.transformer import model_utils

# Don't let TensorFlow see the encoder for the purpose of minimizing loss or saving/loading weights via "AutoTrackable".
# The encoder is just a weightless head to the "full" chess-playing model, so it's saved/loaded separately.
class EncoderHider:

  def __init__(self, encoder):
    self.encoder = encoder

class CommentaryModel(tf.keras.Model):

  def __init__(self, encoder, decoder, **kwargs):
    super().__init__(**kwargs)
    self.encoder_hider = EncoderHider(encoder)
    self.decoder = decoder

  # Trade off time/space with two encodes rather than reshaping the 2x into batch
  # (we're at the limit of compiled size/memory use on TPU with current batch size).
  #
  # During training we're not backpropagating into the encoder, so always use "training=False"
  # (e.g. use moving averages for batch normalization), and stop gradients.
  #
  # Using the "EncoderHider" also prevents any warnings about missing gradients,
  # and simplifies saving and loading of weights (vs., e.g., overriding and delegating to the decoder).
  def call(self, inputs):
    # Just eat the circular import.
    from model import ModelBuilder
    # Expect (batch, 219 = 109 + 109 + 1) int64.
    images = inputs["inputs"]
    # Encode (batch, 109) int64 to (batch, 64, 256) float32 for the "before"/"previous" position.
    before = self.encoder_hider.encoder(images[:, 0:ModelBuilder.input_planes_count], training=False)
    # Encode (batch, 109) int64 to (batch, 64, 256) float32 for the "after"/"current" position.
    after = self.encoder_hider.encoder(images[:, ModelBuilder.input_planes_count:2*ModelBuilder.input_planes_count], training=False)
    # Expand (batch, 1) int64 to (batch, 1, 256) of all ones or all zeros.
    side_to_move = tf.cast(images[:, -1:], tf.bool)
    side_to_move = tf.cast(side_to_move, tf.float32)
    side_to_move = tf.tile(side_to_move[:, :, tf.newaxis], [1, 1, ModelBuilder.filter_count])
    # Concatenate encoder outputs to (batch, 129, 256) float32.
    encoder_outputs = tf.concat([before, after, side_to_move], axis=1)
    # Decode.
    inputs = {**inputs, "inputs": encoder_outputs }
    return self.decoder(inputs)

class CommentaryDecoder(tf.keras.Model):
  """Transformer model with Keras.

  Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continuous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

  def __init__(self,
               vocab_size,
               decoder_layer,
               eos_id,
               decode_max_length,
               encoder_width,
               embedding_width,
               dropout_rate,
               padded_decode=False,
               extra_decode_length=0,
               sample_temperature=1.0,
               top_p=0.9,
               dtype=tf.float32,
               **kwargs):
    """Initialize layers to build Transformer model.

    Args:
      vocab_size: Size of vocabulary.
      encoder_width: Depth of provided encoder outputs (Added)
      embedding_width: Size of hidden layer for embedding.
      dropout_rate: Dropout probability.
      padded_decode: Whether to max_sequence_length padding is used. If set
        False, max_sequence_length padding is not used.
      decode_max_length: maximum number of steps to decode a sequence.
      extra_decode_length: Beam search will run extra steps to decode.
      sample_temperature: (Added for nucleus sampling (top-p))
      top_p: (Added for nucleus sampling (top-p))
      decoder_layer: An initialized decoder layer.
      dtype: float dtype.
      eos_id: Id of end of sentence token.
      **kwargs: other keyword arguments.
    """
    super().__init__(**kwargs)
    self._vocab_size = vocab_size
    self._encoder_width = encoder_width
    self._embedding_width = embedding_width
    self._dropout_rate = dropout_rate
    self._padded_decode = padded_decode
    self._decode_max_length = decode_max_length
    self._extra_decode_length = extra_decode_length
    self._sample_temperature = sample_temperature
    self._top_p = top_p
    self._dtype = dtype
    self._eos_id = eos_id
    self.embedding_lookup = keras_nlp.layers.OnDeviceEmbedding(
        vocab_size=self._vocab_size,
        embedding_width=self._embedding_width,
        initializer=tf.random_normal_initializer(
            mean=0., stddev=self._embedding_width**-0.5),
        scale_factor=self._embedding_width**0.5)
    self.decoder_layer = decoder_layer
    self.encoder_outputs_position_encoding = layers.RelativePositionEmbedding(
        hidden_size=self._encoder_width)
    self.position_embedding = layers.RelativePositionEmbedding(
        hidden_size=self._embedding_width)
    self.decoder_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

  def _embedding_linear(self, embedding_matrix, x):
    """Uses embeddings as linear transformation weights."""
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    hidden_size = tf.shape(x)[2]
    vocab_size = tf.shape(embedding_matrix)[0]

    x = tf.reshape(x, [-1, hidden_size])
    logits = tf.matmul(
        tf.cast(x, dtype=self._dtype),
        tf.cast(embedding_matrix, self._dtype),
        transpose_b=True)

    return tf.reshape(logits, [batch_size, length, vocab_size])

  def call(self, inputs):
    """Calculate target logits or inferred target sequences.

    Args:
      inputs: a dictionary of tensors.
        Feature `inputs`: int tensor with shape [batch_size, input_length].
        Feature `targets` (optional): None or int tensor with shape
          [batch_size, target_length].

    Returns:
      If targets is defined, then return logits for each word in the target
      sequence. float tensor with shape [batch_size, target_length, vocab_size]
      If target is none, then generate output sequence one token at a time.
        returns a dictionary {
          outputs: [batch_size, decoded length]
          scores: [batch_size, float]}
      Even when float16 is used, the output tensor(s) are always float32.

    Raises:
      NotImplementedError: If try to use padded decode method on CPU/GPUs.
    """
    sources = inputs["inputs"]
    targets = inputs.get("targets", None)

    # cbutner: Encoder output is provided directly and fixed length, so no need for
    # embedding, positional encoding for embedding, attention masking or dropout at this stage.
    # The attention bias tensor is just zeros, with shape compatible with "get_padding_bias".
    encoder_outputs = sources
    encoder_outputs = tf.stop_gradient(encoder_outputs) # See "CommentaryModel.call".
    attention_bias = tf.zeros((tf.shape(encoder_outputs)[0], 1, 1, 1), dtype=self._dtype)

    # However, do apply a position encoding to the provided encoder outputs to differentiate
    # the "before" position (64 elements) from the "after" position (64 elements) and the
    # side to move (1 element).
    #
    # This will be at the main model encoder depth though, 256, rather than at the transformer
    # depth, 512. The key/query/value calculations expand the 256 to 512.
    encoder_outputs_position_encoding = self.encoder_outputs_position_encoding(encoder_outputs)
    encoder_outputs_position_encoding = tf.cast(encoder_outputs_position_encoding, encoder_outputs.dtype)
    encoder_outputs = encoder_outputs + encoder_outputs_position_encoding

    if targets is None:
      encoder_decoder_attention_bias = attention_bias
      encoder_outputs = tf.cast(encoder_outputs, self._dtype)
      if self._padded_decode:
        max_decode_length = self._decode_max_length
      else:
        max_decode_length = self._decode_max_length or (
            tf.shape(encoder_outputs)[1] + self._extra_decode_length)
      encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias,
                                               self._dtype)
      symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

      batch_size = tf.shape(encoder_outputs)[0]
      # Create initial set of IDs that will be passed to symbols_to_logits_fn.
      initial_ids = tf.zeros([batch_size], dtype=tf.int32)

      # Create cache storing decoder attention values for each layer.
      # pylint: disable=g-complex-comprehension
      init_decode_length = (max_decode_length if self._padded_decode else 0)
      num_heads = self.decoder_layer.num_attention_heads
      dim_per_head = self._embedding_width // num_heads

      cache = {
          str(layer): {
              "key":
                  tf.zeros(
                      [batch_size, init_decode_length, num_heads, dim_per_head],
                      dtype=self._dtype),
              "value":
                  tf.zeros(
                      [batch_size, init_decode_length, num_heads, dim_per_head],
                      dtype=self._dtype)
          } for layer in range(self.decoder_layer.num_layers)
      }

      # pylint: enable=g-complex-comprehension
      # Add encoder output and attention bias to the cache.
      cache["encoder_outputs"] = encoder_outputs
      cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

      # Use nucleus sampling (top-p) to sample sequences and scores.
      sampling = sampling_module.SamplingModule(
        length_normalization_fn=None,
        dtype=tf.float32,
        symbols_to_logits_fn=symbols_to_logits_fn,
        vocab_size=self._vocab_size,
        max_decode_length=max_decode_length,
        eos_id=self._eos_id,
        sample_temperature=self._sample_temperature,
        top_p=self._top_p,
        padded_decode=self._padded_decode,
        enable_greedy=False)
      decoded_ids, scores = sampling.generate(initial_ids=initial_ids, initial_cache=cache)

      top_decoded_ids = decoded_ids[:, 1:] # These used to take beam [0], just keeping names.
      top_scores = scores # These used to take beam [0], just keeping names.

      return {"outputs": top_decoded_ids, "scores": top_scores}

    decoder_inputs = self.embedding_lookup(targets)
    embedding_mask = tf.cast(
        tf.not_equal(targets, 0), self.embedding_lookup.embeddings.dtype)
    decoder_inputs = tf.cast(decoder_inputs, self._dtype)
    decoder_inputs *= tf.expand_dims(embedding_mask, -1)
    # Shift targets to the right, and remove the last element
    decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    length = tf.shape(decoder_inputs)[1]
    pos_encoding = self.position_embedding(decoder_inputs)
    pos_encoding = tf.cast(pos_encoding, self._dtype)
    decoder_inputs += pos_encoding

    decoder_inputs = self.decoder_dropout(decoder_inputs)

    decoder_shape = tf_utils.get_shape_list(decoder_inputs, expected_rank=3)
    batch_size = decoder_shape[0]
    decoder_length = decoder_shape[1]

    self_attention_mask = tf.linalg.band_part(
        tf.ones([length, length], dtype=tf.float32), -1, 0)
    self_attention_mask = tf.reshape(self_attention_mask, [1, length, length])
    self_attention_mask = tf.tile(self_attention_mask, [batch_size, 1, 1])

    # cbutner: Encoder output is calculated directly and fixed length, so "attention_mask"
    # is all ones, and needs to look at "encoder_output" shape and not "source".
    attention_mask = tf.ones((batch_size, decoder_length, tf.shape(encoder_outputs)[1]), dtype=encoder_outputs.dtype)

    outputs = self.decoder_layer(
        decoder_inputs,
        encoder_outputs,
        memory_mask=self_attention_mask,
        target_mask=attention_mask)
    logits = self._embedding_linear(self.embedding_lookup.embeddings, outputs)
    logits = tf.cast(logits, tf.float32)
    return logits

  def _get_symbols_to_logits_fn(self, max_decode_length):
    """Returns a decoding function that calculates logits of the next tokens."""
    timing_signal = self.position_embedding(
        inputs=None, length=max_decode_length + 1)
    timing_signal = tf.cast(timing_signal, self._dtype)
    decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
        max_decode_length, dtype=self._dtype)

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size, i + 1].
        i: Loop index.
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      # decoder_input = self.embedding_softmax_layer(decoder_input)
      source_decoder_input = decoder_input
      decoder_input = self.embedding_lookup(decoder_input)
      embedding_mask = tf.cast(
          tf.not_equal(source_decoder_input, 0),
          self.embedding_lookup.embeddings.dtype)
      decoder_input *= tf.expand_dims(embedding_mask, -1)
      decoder_input += timing_signal[i]
      if self._padded_decode:
        bias_shape = decoder_self_attention_bias.shape.as_list()
        self_attention_bias = tf.slice(
            decoder_self_attention_bias, [0, 0, i, 0],
            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
      else:
        self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
      decoder_shape = tf_utils.get_shape_list(decoder_input, expected_rank=3)
      batch_size = decoder_shape[0]
      decoder_length = decoder_shape[1]

      attention_bias = cache.get("encoder_decoder_attention_bias")
      attention_bias = tf.where(attention_bias < 0,
                                tf.zeros_like(attention_bias),
                                tf.ones_like(attention_bias))
      attention_bias = tf.squeeze(attention_bias, axis=[1])
      attention_mask = tf.tile(attention_bias, [1, decoder_length, 1])

      self_attention_bias = tf.where(self_attention_bias < 0,
                                     tf.zeros_like(self_attention_bias),
                                     tf.ones_like(self_attention_bias))
      self_attention_bias = tf.squeeze(self_attention_bias, axis=[1])
      self_attention_mask = tf.tile(self_attention_bias, [batch_size, 1, 1])

      decoder_outputs = self.decoder_layer(
          decoder_input,
          cache.get("encoder_outputs"),
          memory_mask=self_attention_mask,
          target_mask=attention_mask,
          cache=cache,
          decode_loop_step=i if self._padded_decode else None)

      logits = self._embedding_linear(self.embedding_lookup.embeddings,
                                      decoder_outputs)
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return symbols_to_logits_fn

def _pad_tensors_to_same_length(x, y):
  """Pad x and y so that the results have the same length (second dimension)."""
  x_length = tf.shape(x)[1]
  y_length = tf.shape(y)[1]

  max_length = tf.maximum(x_length, y_length)

  x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
  y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
  return x, y

def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
  """Calculate cross entropy loss while ignoring padding.
  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary
  Returns:
    Returns the cross entropy loss and weight tensors: float32 tensors with
      shape [batch_size, max(length_logits, length_labels)]
  """
  logits, labels = _pad_tensors_to_same_length(logits, labels)

  # Calculate smoothing cross entropy
  confidence = 1.0 - smoothing
  low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, tf.float32)
  soft_targets = tf.one_hot(
      tf.cast(labels, tf.int32),
      depth=vocab_size,
      on_value=confidence,
      off_value=low_confidence)
  xentropy = tf.nn.softmax_cross_entropy_with_logits(
      logits=logits, labels=soft_targets)

  # Calculate the best (lowest) possible value of cross entropy, and
  # subtract from the cross entropy loss.
  normalizing_constant = -(
      confidence * tf.math.log(confidence) + tf.cast(vocab_size - 1, tf.float32)
      * low_confidence * tf.math.log(low_confidence + 1e-20))
  xentropy -= normalizing_constant

  # cbutner: We now have "xentropy" as a mean logit loss per sequence.
  # The "custom training loop" way to proceed is to return (xentropy * weights)
  # and (weights), sum each of those across the *global batch* (not per-replica),
  # then divide them.
  #
  # However, we can get away with using Model.fit() and avoiding a custom loop by
  # just returning (xentropy * weights) here, letting the built-in reduction average them,
  # and for each training example using a sample weight equal to the masked sequence length
  # divided by the mean masked sequence length for the *global batch*.
  weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
  return (xentropy * weights)

# Shim "sample_top_k" to work around https://github.com/tensorflow/models/issues/10032 on TPU.
original_sample_top_k = sampling_module.sample_top_k
sampling_module.sample_top_k = lambda logits, top_k: original_sample_top_k(logits, tf.math.maximum(1, top_k))
