from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import time
import sys
import tensorflow as tf
import reader
import models



tf.app.flags.DEFINE_string("train_file", "data/train_nopos_ty=6.txt", "training file")
tf.app.flags.DEFINE_string("test_file", "data/test_nopos_ty=6.txt", "Test file")
tf.app.flags.DEFINE_string("senna_embed_file", "data/embedding/senna/embeddings.txt", "senna word embedding")
tf.app.flags.DEFINE_string("senna_words_file", "data/embedding/senna/words.lst", "senna word list")
tf.app.flags.DEFINE_string("logdir", "saved_models/", "where to save the model")

tf.app.flags.DEFINE_integer("max_len", 97, "if length of a sentence is large than max_len, truncate it")
tf.app.flags.DEFINE_integer("num_relations", 19, "number of relations")
tf.app.flags.DEFINE_integer("word_dim", 50, "word embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 50, "number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")

tf.app.flags.DEFINE_integer("pos_num", 123, "number of position feature")
tf.app.flags.DEFINE_integer("pos_dim", 5, "position embedding size")
tf.app.flags.DEFINE_integer("filter_size", 3, "cnn number of hidden unit")
tf.app.flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")
# tf.app.flags.DEFINE_integer("linear_size", 200, "linear layer number of hidden unit")

tf.app.flags.DEFINE_integer("rnn_size", 100, "hidden unit of rnn")
tf.app.flags.DEFINE_integer("rnn_layers", 1, "layers of rnn")

tf.app.flags.DEFINE_integer("decay_steps", 1*80, "learning rate decay steps")
tf.app.flags.DEFINE_float("decay_rate", 0.97, "learning rate decay rate")
tf.app.flags.DEFINE_float("lrn_rate", 1e-3, "learning rate")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

tf.app.flags.DEFINE_string("model", "cnn", "cnn model")
tf.app.flags.DEFINE_boolean('test', False, 'set True to test')

FLAGS = tf.app.flags.FLAGS

def build_cnn_model(word_embed, max_len):
  '''Relation Classification via Convolutional Deep Neural Network'''
  with tf.name_scope("Train"):
    with tf.variable_scope('CNNModel', reuse=None):
      m_train = models.CNNModel( word_embed, FLAGS.word_dim, max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    FLAGS.keep_prob, FLAGS.filter_size, FLAGS.num_filters, 
                    FLAGS.lrn_rate, FLAGS.decay_steps, FLAGS.decay_rate, is_train=True)
  with tf.name_scope('Valid'):
    with tf.variable_scope('CNNModel', reuse=True):
      m_valid = models.CNNModel( word_embed, FLAGS.word_dim, max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    1.0, FLAGS.filter_size, FLAGS.num_filters, 
                    FLAGS.lrn_rate, FLAGS.decay_steps, FLAGS.decay_rate, is_train=False)
  return m_train, m_valid

def build_rnn_model(word_embed, max_len):
  '''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''
  with tf.name_scope("Train"):
    with tf.variable_scope('RNNModel', reuse=None):
      m_train = models.RNNModel( word_embed, FLAGS.word_dim, max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, 
                    FLAGS.num_relations, FLAGS.keep_prob, 
                    FLAGS.rnn_size, FLAGS.rnn_layers,
                    FLAGS.lrn_rate, is_train=True)
  with tf.name_scope('Valid'):
    with tf.variable_scope('RNNModel', reuse=True):
      m_valid = models.RNNModel( word_embed, FLAGS.word_dim, max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, 
                    FLAGS.num_relations, 1.0, 
                    FLAGS.rnn_size, FLAGS.rnn_layers,
                    FLAGS.lrn_rate, is_train=False)
  return m_train, m_valid

def build_rcnn_model(word_embed, max_len):
  '''Bidirectional Recurrent Convolutional Neural Network for Relation Classification'''
  with tf.name_scope("Train"):
    with tf.variable_scope('RCNNModel', reuse=None):
      m_train = models.RCNNModel( word_embed, FLAGS.word_dim, max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    FLAGS.keep_prob, FLAGS.filter_size, FLAGS.num_filters, 
                    FLAGS.rnn_size, FLAGS.rnn_layers, FLAGS.lrn_rate, is_train=True)
  with tf.name_scope('Valid'):
    with tf.variable_scope('RCNNModel', reuse=True):
      m_valid = models.RCNNModel( word_embed, FLAGS.word_dim, max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    1.0, FLAGS.filter_size, FLAGS.num_filters, 
                    FLAGS.rnn_size, FLAGS.rnn_layers, FLAGS.lrn_rate, is_train=False)
  return m_train, m_valid

def build_mtl_model(word_embed, max_len):
  '''Adversarial Multi-task Learning for Text Classification'''
  with tf.name_scope("Train"):
    with tf.variable_scope('MTLModel', reuse=None):
      m_train = models.MTLModel( word_embed, FLAGS.word_dim, max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    FLAGS.keep_prob, FLAGS.filter_size, FLAGS.num_filters, 
                    FLAGS.lrn_rate, FLAGS.decay_steps, FLAGS.decay_rate, is_train=True)
  with tf.name_scope('Valid'):
    with tf.variable_scope('MTLModel', reuse=True):
      m_valid = models.MTLModel( word_embed, FLAGS.word_dim, max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    1.0, FLAGS.filter_size, FLAGS.num_filters, 
                    FLAGS.lrn_rate, FLAGS.decay_steps, FLAGS.decay_rate, is_train=False)
  return m_train, m_valid

def train(sess, m_train, m_valid, train_data, get_train_feed, test_feed, saver, save_path):
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
        saver.save(sess, save_path, best_step)
      print("Epoch %d, loss %.2f, acc %.2f %.4f, time %.2f" % 
                                (epoch, loss, acc, v_acc, duration))
      sys.stdout.flush()
    n += 1
  duration = time.time() - orig_begin_time
  print('Done training, best_step: %d, best_acc: %.4f' % (best_step, best))
  print('duration: %d' % duration)

def test(sess, acc_tensor, feed):
  accuracy = sess.run(acc_tensor, feed)
  print('accuracy: %.4f' % accuracy)

def get_saver(dir_name):
  var_list = None
  saver = tf.train.Saver(var_list)

  save_dir = os.path.join(FLAGS.logdir, dir_name)
  save_path = os.path.join(save_dir, "model.ckpt")

  return saver, save_dir, save_path


def main(_):
  raw_train_data = reader.load_raw_data(FLAGS.train_file, FLAGS.max_len)
  raw_test_data = reader.load_raw_data(FLAGS.test_file, FLAGS.max_len)

  word2id, id2word, word_embed = reader.gen_embeddings(
                      raw_train_data, raw_test_data, 
                      FLAGS.senna_embed_file, FLAGS.senna_words_file, 
                      FLAGS.word_dim)
  max_len = FLAGS.max_len + 2 # append start and end word
  train_data = reader.gen_batch_data(raw_train_data, word2id, max_len, 
                                FLAGS.num_epochs, FLAGS.batch_size, shuffle=True)
  test_data = reader.gen_batch_data(raw_test_data, word2id, max_len, 
                                  1, 2717, shuffle=False)
  test_data = test_data.__next__()

  with tf.Graph().as_default():
    if FLAGS.model == 'cnn':
      m_train, m_valid = build_cnn_model(word_embed, max_len)
      saver, save_dir, save_path = get_saver('cnn/')
    elif FLAGS.model == 'rnn':
      m_train, m_valid = build_rnn_model(word_embed, max_len)
      saver, save_dir, save_path = get_saver('rnn/')
    elif FLAGS.model == 'rcnn':
      m_train, m_valid = build_rcnn_model(word_embed, max_len)
      saver, save_dir, save_path = get_saver('rcnn/')
    elif FLAGS.model == 'mtl':
      m_train, m_valid = build_mtl_model(word_embed, max_len)
      saver, save_dir, save_path = get_saver('mtl/')


    test_feed = {
          m_valid.sent_id : test_data['sent_id'],
          m_valid.pos1_id : test_data['pos1_id'],
          m_valid.pos2_id : test_data['pos2_id'],
          m_valid.lexical_id : test_data['lexical_id'],
          m_valid.rid : test_data['rid'],
      }
    get_train_feed = lambda m_train, batch_data: {
          m_train.sent_id : batch_data['sent_id'],
          m_train.pos1_id : batch_data['pos1_id'],
          m_train.pos2_id : batch_data['pos2_id'],
          m_train.lexical_id : batch_data['lexical_id'],
          m_train.rid : batch_data['rid'],
      }
    
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue
    # sv finalize the graph
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      print('='*80)
      
      if not FLAGS.test:
        # ckpt = tf.train.get_checkpoint_state(save_dir)
        # saver.restore(sess, ckpt.model_checkpoint_path)
        train(sess, m_train, m_valid, train_data, get_train_feed, 
                                            test_feed, saver, save_path)

      ckpt = tf.train.get_checkpoint_state(save_dir)
      saver.restore(sess, ckpt.model_checkpoint_path)
      test(sess, m_valid.accuracy, test_feed)
  
          
if __name__ == '__main__':
  tf.app.run()