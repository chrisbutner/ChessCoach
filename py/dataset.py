import tensorflow as tf
from model import ModelBuilder

class DatasetOptions:

  def __init__(self, global_batch_size, position_shuffle_size=2**17, keep_game_proportion=0.2, keep_position_proportion=0.1, cycle_length=16):
    self.global_batch_size = global_batch_size
    self.position_shuffle_size = position_shuffle_size
    self.keep_game_proportion = keep_game_proportion
    self.keep_position_proportion = keep_position_proportion
    self.cycle_length = cycle_length

class CommentaryDatasetOptions:

  def __init__(self, global_batch_size, maximum_sequence_length):
    self.global_batch_size = global_batch_size
    self.maximum_sequence_length = maximum_sequence_length

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
  positions_per_game = 135 # Estimate

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
  def decompress_values(self, result, indices):
    return tf.where(indices % 2 == 0, result, self.flip_value(result))

  def decompress_images(self, image_pieces_auxiliary, indices):
    # Slice out piece and auxiliary planes.
    image_pieces = image_pieces_auxiliary[:, :self.input_piece_planes_per_position]
    image_auxiliary = image_pieces_auxiliary[:, self.input_piece_planes_per_position:]

    # Take a slice of the last N positions' piece planes and concatenate this position's auxiliary planes.
    image_pieces_padded = tf.pad(image_pieces, [[self.input_previous_position_count, 0], [0, 0]])
    gathered_pieces = tf.map_fn(lambda x: image_pieces_padded[x:x + self.input_previous_position_plus_current_count], indices)
    gathered_pieces = tf.reshape(gathered_pieces, [-1, self.input_previous_position_plus_current_count * self.input_piece_planes_per_position])
    gathered_auxiliary = tf.gather(image_auxiliary, indices)
    images = tf.concat([gathered_pieces, gathered_auxiliary], axis=1)
    return images

  def scatter_policy(self, indices_values):
    indices, values = indices_values
    indices = tf.expand_dims(indices, 1)
    policy = tf.scatter_nd(indices, values, self.output_planes_flat_shape)
    policy = tf.reshape(policy, self.output_planes_shape)
    return policy

  def decompress_policies(self, policy_row_lengths, policy_indices, policy_values, indices):
    # Cast down from protobuf 64-bit.
    policy_row_lengths = tf.cast(policy_row_lengths, tf.int32)
    policy_indices = tf.cast(policy_indices, tf.int32)
    
    # Policy indices and values are ragged (different move possibilities per position), reconstruct.
    # Validate=false improves performance and is required because policy_indices/policy_values are padded
    # by tf.io.parse_example across the batch.
    policy_indices = tf.RaggedTensor.from_row_lengths(policy_indices, policy_row_lengths, validate=False)
    policy_values = tf.RaggedTensor.from_row_lengths(policy_values, policy_row_lengths, validate=False)

    # We couldn't be selective with "keep_position_proportion" indices before the "from_row_lengths"
    # reconstruction unfortunately, but reduce down now to the selected positions.
    policy_indices = tf.gather(policy_indices, indices)
    policy_values = tf.gather(policy_values, indices)

    # Reconstruct the dense policy from sparse policy indices/values.
    return tf.map_fn(self.scatter_policy, (policy_indices, policy_values), fn_output_signature=tf.float32)

  def decompress(self, result, image_pieces_auxiliary, policy_row_lengths, policy_indices, policy_values, indices):
    # Games are written with sparse policies and implicit history,
    # so decompress to self-contained positions ready for training.
    images = self.decompress_images(image_pieces_auxiliary, indices)
    values = self.decompress_values(result, indices)
    policies = self.decompress_policies(policy_row_lengths, policy_indices, policy_values, indices)

    return (images, values, policies)

  def parse_game(self, position_count, selected, result, mcts_values, image_pieces_auxiliary, policy_row_lengths, policy_indices, policy_values, options):
    # Unpad down from the dense shape across all games in the chunk to this particular game's position count.
    mcts_values = mcts_values[:position_count]
    image_pieces_auxiliary = image_pieces_auxiliary[:position_count]
    policy_row_lengths = policy_row_lengths[:position_count]

    # Generate indices for this game's "keep_position_proportion" selection to "tf.gather" with.
    selected = selected[:position_count]
    indices = tf.reshape(tf.where(selected), [-1])

    # Break apart and stitch together tensors, and decompress using position history.
    images, values, policies = self.decompress(result, image_pieces_auxiliary, policy_row_lengths, policy_indices, policy_values, indices)
    mcts_values = tf.gather(mcts_values, indices)

    # Return the dataset mapping images to labels.
    dataset = tf.data.Dataset.from_tensor_slices((images, (values, mcts_values, policies)))
    return dataset

  def parse_games(self, batch, options):
    # Parse raw features from the tf.train.Examples representing the games.
    example = tf.io.parse_example(batch, self.feature_map)
    result = example["result"]
    mcts_values = example["mcts_values"]
    image_pieces_auxiliary = example["image_pieces_auxiliary"]
    policy_row_lengths = example["policy_row_lengths"]
    policy_indices = example["policy_indices"]
    policy_values = example["policy_values"]

    # Throw away a proportion of *positions* to avoid overly correlated/periodic data. This is a time/space trade-off.
    # Throwing away more saves memory but costs CPU. Increasing shuffle buffer size saves CPU but costs memory.
    position_count = tf.math.count_nonzero(policy_row_lengths, axis=1, dtype=tf.int32)
    selected = tf.random.uniform(tf.shape(policy_row_lengths)) < options.keep_position_proportion

    # Each game needs to be decompressed separately to avoid history leaking across games
    # and to reconstruct the ragged (across positions) and sparse (within a position) policy tensors.
    dataset = tf.data.Dataset.from_tensor_slices((position_count, selected, result, mcts_values, image_pieces_auxiliary, policy_row_lengths, policy_indices, policy_values))
    dataset = dataset.filter(lambda position_count, selected, *_: tf.math.reduce_any(selected[:position_count]))
    dataset = dataset.flat_map(lambda *x: self.parse_game(*x, options))
    return dataset

  def parse_chunk(self, filename, options):
    # Parse games from the chunk, stored as tf.train.Examples in TFRecords (see Storage.cpp).
    dataset = tf.data.TFRecordDataset(filename, compression_type=self.compression_type,
      buffer_size=self.chunk_read_buffer_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Throw away a proportion of *games* to avoid overly correlated/periodic data. This is a time/space trade-off.
    # Throwing away more saves memory but costs CPU. Increasing shuffle buffer size saves CPU but costs memory.
    if options.keep_game_proportion < 1.0:
      dataset = dataset.filter(lambda _: tf.random.uniform([]) < options.keep_game_proportion)

    # Parse positions from games.
    #
    # As long as the shuffle buffer is large enough to hold at least one cycle's positions, after throw-aways, there's no point
    # intentionally shuffling here.
    dataset = dataset.batch(self.games_per_chunk, drop_remainder=False)
    dataset = dataset.flat_map(lambda x: self.parse_games(x, options))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  def build_dataset_source(self, glob, window, options):
    # Grab chunk filenames (they need to be ordered here).
    filenames = tf.io.gfile.glob(glob)

    # Restrict to the training window.
    if window is not None:
      min_chunk_inclusive = window[0] // self.games_per_chunk
      max_chunk_exclusive = (window[1] + self.games_per_chunk - 1) // self.games_per_chunk
      filenames = filenames[min_chunk_inclusive:max_chunk_exclusive]
      chunks_expected = (max_chunk_exclusive - min_chunk_inclusive)
      if len(filenames) < chunks_expected:
        games_found = len(filenames) * self.games_per_chunk
        games_expected = chunks_expected * self.games_per_chunk
        raise Exception(f"Not enough games found - {games_found} vs. {games_expected} - add a matching 'play' stage before training")

    # Pick chunk order randomly over the full span of the window.
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shuffle(len(filenames), reshuffle_each_iteration=True)

    # Repeat chunk loading for each source.
    dataset = dataset.repeat()
    return dataset

  def build_dataset(self, sources, options):
    # Validate options.
    kept_positions_per_chunk = self.games_per_chunk * self.positions_per_game * options.keep_game_proportion * options.keep_position_proportion
    cycles_per_shuffle_buffer = options.position_shuffle_size / (options.cycle_length * kept_positions_per_chunk)
    assert (cycles_per_shuffle_buffer >= 1.0), f"cycles_per_shuffle_buffer={cycles_per_shuffle_buffer}, expect >= 1.0"
    assert (cycles_per_shuffle_buffer < 2.0), f"cycles_per_shuffle_buffer={cycles_per_shuffle_buffer}, expect < 2.0"

    # Dataset sources repeat internally, so interleaving them should give an equal number of positions
    # from each, even if the sources are differently sized.
    dataset = tf.data.Dataset.from_tensor_slices(sources)
    dataset = dataset.interleave(lambda x: x, cycle_length=len(sources))

    # Parse chunks in parallel, cycling through as large a number as possible while staying under memory budget.
    # Autotune isn't sufficient to keep the GPU/TPU loaded via disk/storage: experimentally 4/8 seem best.
    num_parallel_calls = 8 if self.config.is_tpu else 4
    dataset = dataset.interleave(lambda x: self.parse_chunk(x, options), cycle_length=options.cycle_length,
      num_parallel_calls=num_parallel_calls, deterministic=False)

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
      keep_game_proportion=0.2 / 32) # Shuffle buffer 32 times as small, 2**17/2**12
    sources = [self.build_dataset_source(glob, window=None, options=options) for glob in globs]
    return self.build_dataset(sources, options)

  def parse_commentary_record(self, record, tokenizer, options):
    # Parse raw features from the tf.train.Examples representing the games.
    example = tf.io.parse_single_example(record, self.commentary_feature_map)
    images = example["images"]
    comments = example["comments"]

    # Tokenize all comments and pad here so that we can batch across chunks.
    comments = tokenizer.tokenize(comments)
    comments = comments.to_tensor(default_value=0, shape=(comments.bounding_shape(axis=0), options.maximum_sequence_length))

    # Transformers need the targets during Model.call(), and loss needs them via fit(), so nest appropriately.
    dataset = tf.data.Dataset.from_tensor_slices((dict(inputs=images, targets=comments), comments))
    return dataset

  def parse_commentary(self, filename, tokenizer, options):
    # Parse commentary from the chunk, stored as tf.train.Examples in TFRecords (see Storage.cpp).
    dataset = tf.data.TFRecordDataset(filename, compression_type=self.compression_type,
      buffer_size=self.chunk_read_buffer_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Parse and tokenize comments. For commentary, each chunk holds 1 tf.train.Example with N images/comments.
    #
    # As long as the shuffle buffer is large enough to hold at least one cycle's positions, after throw-aways, there's no point
    # intentionally shuffling here.
    dataset = dataset.flat_map(lambda x: self.parse_commentary_record(x, tokenizer, options))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  def build_commentary_dataset(self, glob, tokenizer, global_batch_size, maximum_sequence_length):
    options = CommentaryDatasetOptions(global_batch_size, maximum_sequence_length)

    # Grab chunk filenames.
    filenames = tf.io.gfile.glob(glob)

    # Pick chunk order randomly over the full span of the window.
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shuffle(len(filenames), reshuffle_each_iteration=True)

    # Repeat chunk loading.
    dataset = dataset.repeat()

    # Parse commentary in parallel, cycling through all chunks (~40).
    # Autotune isn't sufficient to keep the GPU/TPU loaded via disk/storage: experimentally 4/8 seem best.
    cycle_length = len(filenames)
    num_parallel_calls = min(8 if self.config.is_tpu else 4, cycle_length)
    dataset = dataset.interleave(lambda x: self.parse_commentary(x, tokenizer, options), cycle_length=cycle_length,
     num_parallel_calls=num_parallel_calls, deterministic=False)

    # Shuffle images/comments. Roughly ~1m right now, just shuffle all/most.
    dataset = dataset.shuffle(2**20, reshuffle_each_iteration=True)

    # Batch to the global size.
    dataset = dataset.batch(options.global_batch_size)

    # Prefetch batches.
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset