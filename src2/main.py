import os
import tensorflow as tf
import inputs

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'data', 'data directory.')
tf.app.flags.DEFINE_string("train_file", "train_nopos_ty=6.txt", "train data file")
tf.app.flags.DEFINE_string("test_file", "test_nopos_ty=6.txt", "test data file")

tf.app.flags.DEFINE_string("mode", "gen_data", "generate vocab")

def main(_):
  if FLAGS.mode=='gen_vocab':
    inputs.gen_vocab(FLAGS.train_file)
  elif FLAGS.mode=='gen_data':
    inputs.gen_data()

if __name__ == '__main__':
  tf.app.run()