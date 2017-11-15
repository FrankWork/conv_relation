import tensorflow as tf
from .common import *


class RCNNModel(object):
  '''
  Bidirectional Recurrent Convolutional Neural Network for Relation Classification
  http://aclweb.org/anthology/P16-1072
  '''
  def __init__(self, word_embed, word_dim, max_len, 
              pos_num, pos_dim, num_relations,
              keep_prob, filter_size, num_filters,
              rnn_size, rnn_layers, lrn_rate, is_train):
    # input data
    self.sent_id = tf.placeholder(tf.int32, [None, max_len])
    self.pos1_id = tf.placeholder(tf.int32, [None, max_len])
    self.pos2_id = tf.placeholder(tf.int32, [None, max_len])
    self.lexical_id = tf.placeholder(tf.int32, [None, 6])
    self.rid = tf.placeholder(tf.int32, [None])
    # embedding initialization
    word_embed = tf.get_variable('word_embed', initializer=word_embed, dtype=tf.float32)
    pos_embed = tf.get_variable('pos_embed', shape=[pos_num, pos_dim])
    self.labels = tf.one_hot(self.rid, num_relations)       # batch_size, num_relations

    # embedding lookup
    lexical = tf.nn.embedding_lookup(word_embed, self.lexical_id) # batch_size, 6, word_dim
    lexical = tf.reshape(lexical, [-1, 6*word_dim])

    sentence = tf.nn.embedding_lookup(word_embed, self.sent_id)   # batch_size, max_len, word_dim
    pos1 = tf.nn.embedding_lookup(pos_embed, self.pos1_id)       # batch_size, max_len, pos_dim
    pos2 = tf.nn.embedding_lookup(pos_embed, self.pos2_id)       # batch_size, max_len, pos_dim

    # rnn model
    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    rnn_input = sent_pos
    if is_train and keep_prob < 1:
      rnn_input = tf.nn.dropout(rnn_input, keep_prob)

    outputs = rnn_forward_raw(rnn_input, rnn_size, rnn_layers, is_train, keep_prob)

    rnn_fw_out = tf.concat([outputs[0], sent_pos], axis=2)
    if is_train and keep_prob < 1:
      rnn_fw_out = tf.nn.dropout(rnn_fw_out, keep_prob)
    cnn_fw_out = cnn_forward('cnn_fw', rnn_fw_out, lexical, max_len, num_filters)

    rnn_bw_out = tf.concat([outputs[1], sent_pos], axis=2)
    if is_train and keep_prob < 1:
      rnn_bw_out = tf.nn.dropout(rnn_bw_out, keep_prob)
    cnn_bw_out = cnn_forward('cnn_bw', rnn_bw_out, lexical, max_len, num_filters)

    feature = tf.concat([cnn_fw_out, cnn_bw_out], axis=2)
    feature_size = feature.shape.as_list()[1]
    
    if is_train and keep_prob < 1:
      feature = tf.nn.dropout(feature, keep_prob)

    # Map the features to 19 classes
    logits, _ = linear_layer('linear_rcnn', feature, feature_size, num_relations)

    prediction = tf.nn.softmax(logits)
    accuracy = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(self.labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    loss_ce = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))

    self.logits = logits
    self.prediction = prediction
    self.accuracy = accuracy
    self.loss = loss_ce

    if not is_train:
      return 

    self.global_step = tf.Variable(0, trainable=False, name='step', dtype=tf.int32)
    optimizer = tf.train.AdamOptimizer(lrn_rate)
    self.train_op = optimizer.minimize(self.loss, self.global_step)


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