import tensorflow as tf
import tensorflow_text as tf_text
import sentencepiece

english_character_coverage = 1.0

def ensure_tokenizer(config, vocabulary_size):
  model_prefix = config.join(config.training["commentary_path_supervised"], "tokenizer")
  model_path = model_prefix + ".model"
  try:
    return tf_text.SentencepieceTokenizer(model=tf.io.gfile.GFile(model_path, "rb").read(), add_eos=True)
  except:
    print(f"Training SentencePiece tokenizer: {model_path}")
    vocabulary_path = config.join(config.training["commentary_path_supervised"], config.training["vocabulary_filename"])
    sentencepiece.SentencePieceTrainer.Train(input=vocabulary_path, model_prefix=model_prefix, vocab_size=vocabulary_size, character_coverage=english_character_coverage)
    return tf_text.SentencepieceTokenizer(model=tf.io.gfile.GFile(model_path, "rb").read(), add_eos=True)