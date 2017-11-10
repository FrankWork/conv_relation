import tensorflow as tf

from inputs import vocab_lib
# from inputs import gen_data as gen_data_lib
# from runtime import pretrain

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'data', 'data directory.')
tf.app.flags.DEFINE_string("train_file", "train_nopos_ty=6.txt", "train data file")
tf.app.flags.DEFINE_string("test_file", "test_nopos_ty=6.txt", "test data file")

tf.app.flags.DEFINE_string("mode", "", 
                          "gen_vocab, gen_data, pretrain")

def main(_):
  if FLAGS.mode=='gen_vocab':
    vocab_lib.gen_vocab(FLAGS.train_file)
  # elif FLAGS.mode=='gen_data':
  #   vocab_ids = vocab_lib.get_vocab_ids()
  #   gen_data_lib.gen_data(vocab_ids)
  # elif FLAGS.mode == 'pretrain':
  #   vocab_freqs = vocab_lib.get_vocab_freqs()
  #   pretrain.foo()



if __name__ == '__main__':
  tf.app.run()