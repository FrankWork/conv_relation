import tensorflow as tf

from inputs import vocab_lib
from inputs import data_lib
from inputs import inputs_lib
from runtime import train as train_lib
from runtime import evaluate as evaluate_lib

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'data', 'data directory.')
tf.app.flags.DEFINE_string("train_file", "train_nopos_ty=6.txt", "train data file")
tf.app.flags.DEFINE_string("test_file", "test_nopos_ty=6.txt", "test data file")

tf.app.flags.DEFINE_string("mode", "", "pretrain, train, evaluate")

def _inputs(dataset='train', pretrain=False, bidir=False):
  return inputs_lib.inputs(
      data_dir=FLAGS.data_dir,
      phase=dataset,
      bidir=bidir,
      pretrain=pretrain,
      use_seq2seq=pretrain and FLAGS.use_seq2seq_autoencoder,
      state_size=FLAGS.rnn_cell_size,
      num_layers=FLAGS.rnn_num_layers,
      batch_size=FLAGS.batch_size,
      unroll_steps=FLAGS.num_timesteps,
      eos_id=FLAGS.vocab_size - 1)

def main(_):
  data_lib.maybe_gen_data()
  
  vocab_freqs = vocab_lib.get_vocab_freqs()
  lm_inputs = _inputs('train', pretrain=True)
  cl_inputs = _inputs('train', pretrain=False)
  eval_inputs = _inputs('test', pretrain=False)
  
  cl_inputs = _inputs('train', pretrain=False, bidir=True)
  lm_inputs = _inputs('train', pretrain=True, bidir=True)
  eval_inputs = _inputs(dataset, pretrain=False, bidir=True)
  
  model = graphs.get_model(vocab_freqs, lm_inputs, cl_inputs, eval_inputs)

  if FLAGS.mode == 'pretrain':
    train_lib.pretrain(model)
  elif FLAGS.mode == 'train':
    train_lib.train(model)
  elif FLAGS.mode == 'evaluate':
    evaluate_lib.evaluate(model)

if __name__ == '__main__':
  tf.app.run()