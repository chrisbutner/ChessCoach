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