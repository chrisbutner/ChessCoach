import tensorflow as tf
from model import ModelBuilder

class DatasetOptions:

  def __init__(self, global_batch_size, position_shuffle_size=2**17, keep_game_proportion=0.1, keep_position_proportion=0.1, cycle_length=32):
    self.global_batch_size = global_batch_size
    self.position_shuffle_size = position_shuffle_size
    self.keep_game_proportion = keep_game_proportion
    self.keep_position_proportion = keep_position_proportion
    self.cycle_length = cycle_length

class DatasetBuilder:

  # Input and output plane details, duplicated from "Network.h"
  input_previous_position_count = 7
  input_previous_position_plus_current_count = input_previous_position_count + 1
  input_piece_planes_per_position = 12
  input_auxiliary_plane_count = 5
  input_compressed_planes_per_position = input_piece_planes_per_position + input_auxiliary_plane_count
  output_planes_shape = [73, 8, 8]
  output_planes_flat_shape = [73 * 8 * 8]

  # Chunk parameters
  compression_type = "ZLIB"
  chunk_read_buffer_size = 8 * 1024 * 1024
  positions_per_game = 135

  def __init__(self, config):
    self.config = config
    self.games_per_chunk = config.misc["storage"]["games_per_chunk"]

  feature_map = {
    "result": tf.io.FixedLenFeature([], tf.float32),
    "mcts_values": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    "image_pieces_auxiliary": tf.io.FixedLenSequenceFeature([input_compressed_planes_per_position], tf.int64, allow_missing=True),
    "policy_row_lengths": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "policy_indices": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "policy_values": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  }

  commentary_feature_map = {
    "images": tf.io.FixedLenSequenceFeature([101], tf.int64, allow_missing=True),
    "comments": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
  }

  # Values are (-1, 0, 1) in Python/TensorFlow, (0, 0.5, 1) in C++/MCTS.
  def flip_value(self, value):
    return -value

  # Result is from the first player's POV, including the starting position, and flips per player/position.
  def decompress_values(self, result, shape):
    same = tf.fill(shape, result)
    flipped = self.flip_value(same)
    values = tf.stack([same, flipped], axis=1)
    values = tf.reshape(values, [-1])
    values = values[:shape[0]]
    return values

  def decompress_images(self, image_pieces, image_auxiliary):
    image_pieces_padded = tf.pad(image_pieces, [[self.input_previous_position_count, 0], [0, 0]])
    image_piece_histories = tf.image.extract_patches(image_pieces_padded[tf.newaxis, :, :, tf.newaxis],
      [1, self.input_previous_position_plus_current_count, self.input_piece_planes_per_position, 1], [1, 1, 1, 1], [1, 1, 1, 1], "VALID")
    image_piece_histories = tf.squeeze(image_piece_histories)
    images = tf.concat([image_piece_histories, image_auxiliary], axis=1)
    return images

  def scatter_policy(self, indices_values):
    indices, values = indices_values
    indices = tf.expand_dims(indices, 1)
    policy = tf.scatter_nd(indices, values, self.output_planes_flat_shape)
    policy = tf.reshape(policy, self.output_planes_shape)
    return policy

  def decompress_policies(self, indices, values):
    return tf.map_fn(self.scatter_policy, (indices, values), fn_output_signature=tf.float32)

  def decompress_reply_policies(self, policies):
    reply_policies = policies[1:]
    reply_policies = tf.pad(reply_policies, [[0, 1], [0, 0], [0, 0], [0, 0]])
    return reply_policies

  def decompress(self, result, image_pieces_auxiliary, mcts_values, policy_row_lengths, policy_indices, policy_values):
    # Slice out piece and auxiliary planes.
    image_pieces = image_pieces_auxiliary[:, :self.input_piece_planes_per_position]
    image_auxiliary = image_pieces_auxiliary[:, self.input_piece_planes_per_position:]

    # Cast down from protobuf 64-bit.
    policy_row_lengths = tf.cast(policy_row_lengths, tf.int32)
    policy_indices = tf.cast(policy_indices, tf.int32)

    # Policy indices and values are ragged (different move possibilities per position), reconstruct.
    policy_indices = tf.RaggedTensor.from_row_lengths(policy_indices, policy_row_lengths)
    policy_values = tf.RaggedTensor.from_row_lengths(policy_values, policy_row_lengths)

    # Games are written with sparse policies and implicit history/replies,
    # so decompress to self-contained positions ready for training.
    images = self.decompress_images(image_pieces, image_auxiliary)
    values = self.decompress_values(result, tf.shape(mcts_values))
    policies = self.decompress_policies(policy_indices, policy_values)
    reply_policies = self.decompress_reply_policies(policies)

    return (images, values, policies, reply_policies)

  def parse_game(self, serialized, options):
    # Parse raw features from the tf.train.Example representing the game.
    example = tf.io.parse_single_example(serialized, self.feature_map)
    result = example["result"]
    mcts_values = example["mcts_values"]
    image_pieces_auxiliary = example["image_pieces_auxiliary"]
    policy_row_lengths = example["policy_row_lengths"]
    policy_indices = example["policy_indices"]
    policy_values = example["policy_values"]

    # Break apart and stitch together tensors, and decompress using position history.
    images, values, policies, reply_policies = self.decompress(
      result, image_pieces_auxiliary, mcts_values, policy_row_lengths, policy_indices, policy_values)
    dataset = tf.data.Dataset.from_tensor_slices((images, (values, mcts_values, policies, reply_policies)))

    # Throw away a proportion of *positions* to avoid overly correlated/periodic data. This is a time/space trade-off.
    # Throwing away more saves memory but costs CPU. Increasing shuffle buffer size saves CPU but costs memory.
    if options.keep_position_proportion < 1.0:
      dataset = dataset.filter(lambda *_: tf.random.uniform([]) < options.keep_position_proportion)
    return dataset

  def parse_chunk(self, filename, options):
    # Parse games from the chunk, stored as tf.train.Examples in TFRecords (see Storage.cpp).
    dataset = tf.data.TFRecordDataset(filename, compression_type=self.compression_type,
      buffer_size=self.chunk_read_buffer_size)

    # Throw away a proportion of *games* to avoid overly correlated/periodic data. This is a time/space trade-off.
    # Throwing away more saves memory but costs CPU. Increasing shuffle buffer size saves CPU but costs memory.
    if options.keep_game_proportion < 1.0:
      dataset = dataset.filter(lambda _: tf.random.uniform([]) < options.keep_game_proportion)

    # Parse positions from games.
    #
    # As long as the shuffle buffer is large enough to hold at least one cycle's positions, after throw-aways, there's no point
    # intentionally shuffling here.
    dataset = dataset.flat_map(lambda x: self.parse_game(x, options))
    return dataset

  def build_dataset_source(self, glob, window, options):
    # Grab chunk filenames (they need to be ordered here).
    filenames = tf.io.gfile.glob(glob)

    # Restrict to the training window.
    if window is not None:
      min_chunk_inclusive = window[0] // self.games_per_chunk
      max_chunk_exclusive = (window[1] + self.games_per_chunk - 1) // self.games_per_chunk
      filenames = filenames[min_chunk_inclusive:max_chunk_exclusive]

    # Pick chunk order randomly over the full span of the window.
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shuffle(len(filenames), reshuffle_each_iteration=True)

    # Repeat chunk loading for each source.
    dataset = dataset.repeat()

    # Parse chunks in parallel, cycling through as large a number as possible while staying under memory budget.
    dataset = dataset.interleave(lambda x: self.parse_chunk(x, options), cycle_length=options.cycle_length,
      num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    return dataset

  def build_dataset(self, sources, options):
    # Validate options.
    kept_positions_per_chunk = self.games_per_chunk * self.positions_per_game * options.keep_game_proportion * options.keep_position_proportion
    cycles_per_shuffle_buffer = options.position_shuffle_size / (options.cycle_length * kept_positions_per_chunk)
    assert (cycles_per_shuffle_buffer >= 1.0), f"cycles_per_shuffle_buffer={cycles_per_shuffle_buffer}, expect >= 1.0"
    assert (cycles_per_shuffle_buffer < 2.0), f"cycles_per_shuffle_buffer={cycles_per_shuffle_buffer}, expect < 2.0"

    # Use a deterministic interleave between dataset sources to ensure that the shuffle buffer
    # gets an equal number of positions from each, even if the sources are differently sized.
    # The block size doesn't matter too much as long as it's far smaller than the shuffle buffer size.
    dataset = tf.data.Dataset.from_tensor_slices(sources)
    dataset = dataset.interleave(lambda x: x, cycle_length=len(sources), block_length=options.global_batch_size)

    # Shuffle positions. This is still very necessary because we're exhausting each cycle of chunks before moving on to
    # the next cycle, etc., giving highly correlated positions so far. The buffer should be large enough to mix up multiple
    # cycles while still fitting on desktop and cloud hardware.
    #
    # As long as the shuffle buffer is large enough to hold at least one cycle's positions, after throw-aways, there's no point
    # adding shuffling within each interleave worker's processing, so just do it here and tune throw-aways and buffer size.
    #
    # If e.g. 2 cycles could fit in the shuffle buffer then we could have doubled the cycle length with similar results,
    # allowing for greater I/O and CPU parallelism, so aim for cycles per shuffle buffer in [1, 2).
    dataset = dataset.shuffle(options.position_shuffle_size, reshuffle_each_iteration=True)

    # Batch to the global size.
    dataset = dataset.batch(options.global_batch_size)

    # Prefetch batches.
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
  
  def build_training_dataset(self, globs, windows, global_batch_size):
    options = DatasetOptions(global_batch_size=global_batch_size)
    sources = [self.build_dataset_source(glob, window, options) for glob, window in zip(globs, windows)]
    return self.build_dataset(sources, options)

  def build_validation_dataset(self, globs, global_batch_size):
    options = DatasetOptions(
      global_batch_size=global_batch_size,
      position_shuffle_size=2**12,
      keep_game_proportion=0.003125)
    sources = [self.build_dataset_source(glob, window=None, options=options) for glob in globs]
    return self.build_dataset(sources, options)

  def parse_commentary_raw(self, serialized):
    # Parse raw features from the tf.train.Example representing the commentary chunk.
    example = tf.io.parse_single_example(serialized, self.commentary_feature_map)
    images = example["images"]
    comments = example["comments"]
    return images, comments

  def parse_sequences(self, comments, tokenizer):
    comments = [f'{ModelBuilder.token_start} {c.decode("utf-8", errors="ignore")} {ModelBuilder.token_end}' for c in comments]
    sequences = tokenizer.texts_to_sequences(comments)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=ModelBuilder.transformer_max_length, padding="post")
    return sequences

  def parse_commentary(self, dataset, tokenizer):
    for images, comments in dataset:
      comments = self.parse_sequences(comments, tokenizer)
      yield tf.data.Dataset.from_tensor_slices((images, comments))

  # We run into a combination of problems here:
  #
  # (1) Once we start using tf.data.Dataset we have to stay in graph mode, since py_function
  #     isn't allowed when working with TPUs.
  # (2) Keras and third-party tokenizers don't work in graph mode.
  # (3) TF.Text isn't supported on Windows and is allegedly planned for 2.4.
  #
  # Luckily, the entire dataset fits in memory for now, so parse, exit Dataset-land,
  # tokenize, then re-enter.
  def build_commentary_dataset(self, glob, global_batch_size, tokenizer, strategy):
    # Grab chunk filenames.
    filenames = tf.io.gfile.glob(glob)

    # Parse images/comments then read back.
    commentary_raw = tf.data.TFRecordDataset(filenames, compression_type=self.compression_type,
      buffer_size=self.chunk_read_buffer_size, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    commentary_raw = commentary_raw.map(self.parse_commentary_raw)
    commentary_raw = commentary_raw.as_numpy_iterator()

    # Tokenize comments offline.
    datasets = list(self.parse_commentary(commentary_raw, tokenizer))
    dataset = tf.data.Dataset.from_tensor_slices(datasets)
    dataset = dataset.flat_map(lambda x: x)

    # Shuffle images/comments. Roughly 500k right now, just shuffle them all.
    dataset = dataset.shuffle(2**19, reshuffle_each_iteration=True)

    # Repeat and batch to the global size.
    dataset = dataset.repeat()
    dataset = dataset.batch(global_batch_size)

    # Prefetch batches.
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Distribute for custom training.
    dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset