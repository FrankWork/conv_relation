import tensorflow as tf
from .common import *


class RNNModel(object):
  '''
  RNN with attention model
  [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification]
  (http://aclweb.org/anthology/P16-2034)
  '''
  def __init__(self, word_embed, word_dim, max_len, 
               pos_num, pos_dim, num_relations,
               keep_prob, rnn_size, rnn_layers, lrn_rate, is_train):
    # input data
    self.sent_id = tf.placeholder(tf.int32, [None, max_len])
    self.pos1_id = tf.placeholder(tf.int32, [None, max_len])
    self.pos2_id = tf.placeholder(tf.int32, [None, max_len])
    self.rid = tf.placeholder(tf.int32, [None])
    self.lexical_id = tf.placeholder(tf.int32, [None, 6])# not used

    # embedding initialization
    word_embed = tf.get_variable('word_embed', initializer=word_embed, dtype=tf.float32)
    pos_embed = tf.get_variable('pos_embed', shape=[pos_num, pos_dim])
    # word_embed = tf.get_variable('word_embed', [len(word_embed), word_dim], dtype=tf.float32)
    
    # embedding lookup
    sentence = tf.nn.embedding_lookup(word_embed, self.sent_id)   # batch_size, max_len, word_dim
    pos1 = tf.nn.embedding_lookup(pos_embed, self.pos1_id)       # batch_size, max_len, pos_dim
    pos2 = tf.nn.embedding_lookup(pos_embed, self.pos2_id)       # batch_size, max_len, pos_dim
    labels = tf.one_hot(self.rid, num_relations)       # batch_size, num_relations

    # rnn model position indicators
    input = tf.concat([sentence, pos1, pos2], axis=2)
    if is_train and keep_prob < 1:
      input = tf.nn.dropout(input, keep_prob)

    # rnn layers
    feature = rnn_forward(input, max_len, rnn_size, rnn_layers, is_train, keep_prob)
    feature_size = feature.shape.as_list()[1]
    
    if is_train and keep_prob < 1:
      feature = tf.nn.dropout(feature, keep_prob)

    # Map the features to 19 classes
    logits, _ = linear_layer('linear_rnn', feature, feature_size, num_relations)

    prediction = tf.nn.softmax(logits)
    accuracy = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    loss_ce = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    self.logits = logits
    self.prediction = prediction
    self.accuracy = accuracy
    # FIXME: L2 regularization
    self.loss = loss_ce

    if not is_train:
      return 

    global_step = tf.Variable(0, trainable=False, name='step', dtype=tf.int32)
    optimizer = tf.train.AdamOptimizer(lrn_rate)
    self.train_op = optimizer.minimize(self.loss, global_step)
    self.global_step = global_step


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
