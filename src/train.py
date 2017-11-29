from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import time
import sys
import tensorflow as tf
from reader import base as base_reader
from models import cnn_model
from models import mtl_model



tf.app.flags.DEFINE_string("train_file", "data/train.txt", 
                             "original training file")
tf.app.flags.DEFINE_string("test_file", "data/test.txt", 
                             "original test file")
tf.app.flags.DEFINE_string("vocab_file", "data/vocab.txt", "vocab file, automantic generated")
tf.app.flags.DEFINE_string("vocab_freq_file", "data/vocab_freq.txt", "vocab freqs file, automantic generated")

tf.app.flags.DEFINE_string("train_record", "data/train.tfrecord", 
                             "training file of TFRecord format")
tf.app.flags.DEFINE_string("test_record", "data/test.tfrecord", 
                             "Test file of TFRecord format")
tf.app.flags.DEFINE_string("train_mtl_record", "data/train.tfrecord", 
                             "Multi-task learing training file of TFRecord format")
tf.app.flags.DEFINE_string("test_mtl_record", "data/test.tfrecord", 
                             "Multi-task learing test file of TFRecord format")

tf.app.flags.DEFINE_string("word_embed300_orig", 
                             "data/GoogleNews-vectors-negative300.bin", 
                             "google news word embeddding")
tf.app.flags.DEFINE_string("word_embed300_trim", 
                             "data/embed300.trim.npy", 
                             "trimmed google embedding")

tf.app.flags.DEFINE_string("word_embed50_orig", 
                             "data/embedding/senna/embeddings.txt", 
                             "senna words embeddding")
tf.app.flags.DEFINE_string("senna_words_lst", 
                             "data/embedding/senna/words.lst", 
                             "senna words list")
tf.app.flags.DEFINE_string("word_embed50_trim", 
                             "data/embed50.trim.npy", 
                             "trimmed senna embedding")

tf.app.flags.DEFINE_string("relations_file", "data/relations_new.txt", "relations file")
tf.app.flags.DEFINE_string("results_file", "data/results.txt", "predicted results file")
tf.app.flags.DEFINE_string("logdir", "saved_models/", "where to save the model")

# tf.app.flags.DEFINE_integer("freq_threshold", None, "vocab frequency threshold to keep the word")
tf.app.flags.DEFINE_integer("max_len", 97, "max length of sentences")
tf.app.flags.DEFINE_integer("num_relations", 19, "number of relations")
tf.app.flags.DEFINE_integer("word_dim", 50, "word embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 50, "number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_boolean('train_embed', False, 
                             'whether to train word embedding')

tf.app.flags.DEFINE_integer("pos_num", 123, "number of position feature")
tf.app.flags.DEFINE_integer("pos_dim", 5, "position embedding size")
tf.app.flags.DEFINE_integer("filter_size", 3, "cnn number of hidden unit")
tf.app.flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")
# tf.app.flags.DEFINE_integer("linear_size", 200, "linear layer number of hidden unit")
tf.app.flags.DEFINE_float("loss_diff_coef", 0.000001, "coefficient of Orthogonality Constraints")

tf.app.flags.DEFINE_integer("rnn_size", 100, "hidden unit of rnn")
tf.app.flags.DEFINE_integer("rnn_layers", 1, "layers of rnn")

tf.app.flags.DEFINE_integer("decay_steps", 1*80, "learning rate decay steps")
tf.app.flags.DEFINE_float("decay_rate", 0.97, "learning rate decay rate")
tf.app.flags.DEFINE_float("lrn_rate", 1e-3, "learning rate")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

tf.app.flags.DEFINE_string("model", "cnn", "cnn or mtl model")
tf.app.flags.DEFINE_boolean('test', False, 'set True to test')

FLAGS = tf.app.flags.FLAGS


def train(sess, m_train, m_valid):
  n = 1
  best = .0
  best_step = n
  start_time = time.time()
  orig_begin_time = start_time

  while True:
    try:
      epoch = n // 80
      fetches = [m_train.train_op, m_train.loss, m_train.accuracy]
      _, loss, acc = sess.run(fetches)
      if n % 80 == 0:
        now = time.time()
        duration = now - start_time
        start_time = now
        v_acc = sess.run(m_valid.accuracy)
        if best < v_acc:
          best = v_acc
          best_step = n
          m_train.save(sess, best_step)
        print("Epoch %d, loss %.2f, acc %.2f %.4f, time %.2f" % 
                                  (epoch, loss, acc, v_acc, duration))
        sys.stdout.flush()
      n += 1
    except tf.errors.OutOfRangeError:
      break

  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_step: %d, best_acc: %.4f' % (best_step, best))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()

def test(sess, m_valid):
  m_valid.restore(sess)
  fetches = [m_valid.accuracy, m_valid.prediction]
  accuracy, predictions = sess.run(fetches)
  print('accuracy: %.4f' % accuracy)
  
  base_reader.write_results(predictions, FLAGS.relations_file, FLAGS.results_file)


def main(_):
  with tf.Graph().as_default():
    train_data, test_data, word_embed = base_reader.inputs()

    # sv = tf.train.Supervisor()
    # with sv.managed_session() as sess:
    #   print('='*80)
    #   for i in range(10):
    #     arr = sess.run(test_data)
    #     print(train_data[2].shape, arr[2].shape)
    #     print(arr[2][0])
    #   exit()

    
    if FLAGS.model == 'cnn':
      m_train, m_valid = cnn_model.build_train_valid_model(word_embed, 
                                                      train_data, test_data)
    elif FLAGS.model == 'mtl':
      m_train, m_valid = mtl_model.build_train_valid_model(word_embed, 
                                                      train_data, test_data)
    
    m_train.set_saver(FLAGS.model)
    
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue
    # sv finalize the graph
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      print('='*80)
      
      if not FLAGS.test:
        # m_train.restore(sess)
        train(sess, m_train, m_valid)

      test(sess, m_valid)
  
          
if __name__ == '__main__':
  tf.app.run()
