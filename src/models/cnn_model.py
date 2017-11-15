import tensorflow as tf
from models.base_model import BaseModel
from .common import *

FLAGS = tf.app.flags.FLAGS


class CNNModel(BaseModel):
  '''
  Relation Classification via Convolutional Deep Neural Network
  http://www.aclweb.org/anthology/C14-1220
  '''

  def __init__(self, word_embed, word_dim, max_len, 
              pos_num, pos_dim, num_relations,
              keep_prob, filter_size, num_filters,
              lrn_rate, decay_steps, decay_rate, is_train):
    # input data
    self.sent_id = tf.placeholder(tf.int32, [None, max_len])
    self.pos1_id = tf.placeholder(tf.int32, [None, max_len])
    self.pos2_id = tf.placeholder(tf.int32, [None, max_len])
    self.lexical_id = tf.placeholder(tf.int32, [None, 6])
    self.rid = tf.placeholder(tf.int32, [None])
    # embedding initialization
    # xavier = tf.contrib.layers.xavier_initializer()
    word_embed = tf.get_variable('word_embed', 
                                 initializer=word_embed, 
                                 dtype=tf.float32,
                                 trainable=False)
    # word_embed = tf.get_variable('word_embed', [len(word_embed), word_dim], dtype=tf.float32)
    pos_embed = tf.get_variable('pos_embed', shape=[pos_num, pos_dim])
    relation = tf.one_hot(self.rid, num_relations)       # batch_size, num_relations

    self.labels = relation

    # # embedding lookup
    lexical = tf.nn.embedding_lookup(word_embed, self.lexical_id) # batch_size, 6, word_dim
    lexical = tf.reshape(lexical, [-1, 6*word_dim])

    sentence = tf.nn.embedding_lookup(word_embed, self.sent_id)   # batch_size, max_len, word_dim
    pos1 = tf.nn.embedding_lookup(pos_embed, self.pos1_id)       # batch_size, max_len, pos_dim
    pos2 = tf.nn.embedding_lookup(pos_embed, self.pos2_id)       # batch_size, max_len, pos_dim

    # cnn model
    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    if is_train and keep_prob < 1:
      sent_pos = tf.nn.dropout(sent_pos, keep_prob)

    feature = cnn_forward('cnn', sent_pos, lexical, max_len, num_filters)
    feature_size = feature.shape.as_list()[1]
    
    if is_train and keep_prob < 1:
      feature = tf.nn.dropout(feature, keep_prob)

    # Map the features to 19 classes
    logits, _ = linear_layer('linear_cnn', feature, feature_size, num_relations)

    prediction = tf.nn.softmax(logits)
    accuracy = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(relation, axis=1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    loss_ce = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=relation, logits=logits))

    self.logits = logits
    self.prediction = prediction
    self.accuracy = accuracy
    self.loss = loss_ce

    if not is_train:
      return 

    # global_step = tf.train.get_or_create_global_step()
    global_step = tf.Variable(0, trainable=False, name='step', dtype=tf.int32)
    optimizer = tf.train.AdamOptimizer(lrn_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):# for batch_norm
      self.train_op = optimizer.minimize(self.loss, global_step)
    self.global_step = global_step

  


def build_train_valid_model(word_embed):
  '''Relation Classification via Convolutional Deep Neural Network'''
  with tf.name_scope("Train"):
    with tf.variable_scope('CNNModel', reuse=None):
      m_train = CNNModel( word_embed, FLAGS.word_dim, FLAGS.max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    FLAGS.keep_prob, FLAGS.filter_size, FLAGS.num_filters, 
                    FLAGS.lrn_rate, FLAGS.decay_steps, FLAGS.decay_rate, is_train=True)
  with tf.name_scope('Valid'):
    with tf.variable_scope('CNNModel', reuse=True):
      m_valid = CNNModel( word_embed, FLAGS.word_dim, FLAGS.max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    1.0, FLAGS.filter_size, FLAGS.num_filters, 
                    FLAGS.lrn_rate, FLAGS.decay_steps, FLAGS.decay_rate, is_train=False)
  return m_train, m_valid