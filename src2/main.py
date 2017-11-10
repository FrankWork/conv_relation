import os
import tensorflow as tf
import inputs

from inputs import gen_vocab as gen_vocab_lib
from inputs import gen_data as gen_data_lib

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'data', 'data directory.')
tf.app.flags.DEFINE_string("train_file", "train_nopos_ty=6.txt", "train data file")
tf.app.flags.DEFINE_string("test_file", "test_nopos_ty=6.txt", "test data file")

tf.app.flags.DEFINE_string("mode", "gen_data", "gen_vocab, gen_data")

def main(_):
  if FLAGS.mode=='gen_vocab':
    gen_vocab_lib.gen_vocab(FLAGS.train_file)
  elif FLAGS.mode=='gen_data':
    gen_data_lib.gen_data()

if __name__ == '__main__':
  tf.app.run()