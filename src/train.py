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



tf.app.flags.DEFINE_string("train_file", "data/train.txt", "training file")
tf.app.flags.DEFINE_string("test_file", "data/test.txt", "Test file")
tf.app.flags.DEFINE_string("vocab_file", "data/vocab.txt", "vocab file, automantic generated")
tf.app.flags.DEFINE_string("vocab_freq_file", "data/vocab_freq.txt", "vocab freqs file, automantic generated")
tf.app.flags.DEFINE_string("word_embed_orig", "data/GoogleNews-vectors-negative300.bin", "google news word embeddding")
tf.app.flags.DEFINE_string("word_embed_trim", "data/embed300.trim.npy", "trimmed google embedding")
tf.app.flags.DEFINE_string("relations_file", "data/relations_new.txt", "relations file")
tf.app.flags.DEFINE_string("results_file", "data/results.txt", "predicted results file")
tf.app.flags.DEFINE_string("logdir", "saved_models/", "where to save the model")

# tf.app.flags.DEFINE_integer("freq_threshold", None, "vocab frequency threshold to keep the word")
tf.app.flags.DEFINE_integer("max_len", 97, "if length of a sentence is large than max_len, truncate it")
tf.app.flags.DEFINE_integer("num_relations", 19, "number of relations")
tf.app.flags.DEFINE_integer("word_dim", None, "word embedding size, automatic parse from data")
tf.app.flags.DEFINE_integer("num_epochs", 50, "number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")

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


def train(sess, m_train, m_valid, train_data, get_train_feed, test_feed):
  n = 1
  best = .0
  best_step = n
  start_time = time.time()
  orig_begin_time = start_time
  for batch_data in train_data:
    epoch = n // 80
    fetches = [m_train.train_op, m_train.loss, m_train.accuracy]
    train_feed = get_train_feed(m_train, batch_data)
    _, loss, acc = sess.run(fetches, train_feed)
    if n % 80 == 0:
      now = time.time()
      duration = now - start_time
      start_time = now
      v_acc = sess.run(m_valid.accuracy, test_feed)
      if best < v_acc:
        best = v_acc
        best_step = n
        m_train.save(sess, best_step)
      print("Epoch %d, loss %.2f, acc %.2f %.4f, time %.2f" % 
                                (epoch, loss, acc, v_acc, duration))
      sys.stdout.flush()
    n += 1
  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_step: %d, best_acc: %.4f' % (best_step, best))
  print('duration: %.2f hours' % duration)

def test(sess, m_valid, feed):
  m_valid.restore(sess)
  fetches = [m_valid.accuracy, m_valid.prediction]
  accuracy, predictions = sess.run(fetches, feed)
  print('accuracy: %.4f' % accuracy)
  
  base_reader.write_results(predictions, FLAGS.relations_file, FLAGS.results_file)


def main(_):
  train_data, test_data, word_embed = base_reader.inputs(FLAGS.model=='mtl')

  with tf.Graph().as_default():
    if FLAGS.model == 'cnn':
      m_train, m_valid = cnn_model.build_train_valid_model(word_embed)
    elif FLAGS.model == 'mtl':
      m_train, m_valid = mtl_model.build_train_valid_model(word_embed)
    
    m_train.set_saver(FLAGS.model)

    test_feed = {
          m_valid.sent_id : test_data['sent_id'],
          m_valid.pos1_id : test_data['pos1_id'],
          m_valid.pos2_id : test_data['pos2_id'],
          m_valid.lexical_id : test_data['lexical_id'],
          m_valid.rid : test_data['rid'],
    }
    if FLAGS.model == 'mtl':
      test_feed[m_valid.direction] = test_data['direction']
    
    def get_train_feed(m_train, batch_data):
      batch_feed = {
          m_train.sent_id : batch_data['sent_id'],
          m_train.pos1_id : batch_data['pos1_id'],
          m_train.pos2_id : batch_data['pos2_id'],
          m_train.lexical_id : batch_data['lexical_id'],
          m_train.rid : batch_data['rid'],
      }
      if FLAGS.model == 'mtl':
        batch_feed[m_train.direction] = batch_data['direction']
      return batch_feed
    
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue
    # sv finalize the graph
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      print('='*80)
      
      if not FLAGS.test:
        # m_train.restore(sess)
        train(sess, m_train, m_valid, train_data, get_train_feed, test_feed)

      test(sess, m_valid, test_feed)
  
          
if __name__ == '__main__':
  tf.app.run()
