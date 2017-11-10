import tensorflow as tf

from inputs import vocab_lib
from inputs import data_lib
from runtime import train as train_lib
from runtime import evaluate as evaluate_lib

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'data', 'data directory.')
tf.app.flags.DEFINE_string("train_file", "train_nopos_ty=6.txt", "train data file")
tf.app.flags.DEFINE_string("test_file", "test_nopos_ty=6.txt", "test data file")

tf.app.flags.DEFINE_string("mode", "", "pretrain, train, evaluate")

def main(_):
  data_lib.maybe_gen_data()
  
  vocab_freqs = vocab_lib.get_vocab_freqs()
  model = graphs.get_model(vocab_freqs)

  if FLAGS.mode == 'pretrain':
    train_lib.pretrain(model)
  elif FLAGS.mode == 'train':
    train_lib.train(model)
  elif FLAGS.mode == 'evaluate':
    evaluate_lib.evaluate(model)

if __name__ == '__main__':
  tf.app.run()