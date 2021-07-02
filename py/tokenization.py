# ChessCoach, a neural network-based chess engine capable of natural-language commentary
# Copyright 2021 Chris Butner
#
# ChessCoach is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ChessCoach is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

import tensorflow as tf
import tensorflow_text as tf_text
import sentencepiece

english_character_coverage = 1.0

def ensure_tokenizer(config, vocabulary_size, log):
  model_path = config.join(config.training["commentary_path"], "tokenizer.model")
  try:
    return tf_text.SentencepieceTokenizer(model=tf.io.gfile.GFile(model_path, "rb").read(), add_eos=True)
  except:
    log(f"Training SentencePiece tokenizer: {model_path}")
    vocabulary_path = config.join(config.training["commentary_path"], config.training["vocabulary_filename"])
    vocabulary_iterator = iter(tf.io.gfile.GFile(vocabulary_path, "rb").readlines())
    with tf.io.gfile.GFile(model_path, "wb") as model_writer:
      sentencepiece.SentencePieceTrainer.Train(sentence_iterator=vocabulary_iterator, model_writer=model_writer, pad_id=0, unk_id=3,
        vocab_size=vocabulary_size, character_coverage=english_character_coverage)
    return tf_text.SentencepieceTokenizer(model=tf.io.gfile.GFile(model_path, "rb").read(), add_eos=True)