import tensorflow as tf
from models.base_model import BaseModel
from .common import *

FLAGS = tf.app.flags.FLAGS

def adversarial_loss(feature, relation, is_train, keep_prob):
  feature_size = feature.shape.as_list()[1]
  if is_train and keep_prob < 1:
      feature = tf.nn.dropout(feature, keep_prob)
  # Map the features to 10 classes
  out_size = relation.shape.as_list()[1]
  logits, loss_l2 = linear_layer('linear_adv', feature, feature_size, out_size)
  loss_adv = -tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=relation, logits=logits))
  return loss_adv, loss_l2

class MTLModel(BaseModel):
  '''
  Adversarial Multi-task Learning for Text Classification
  http://www.aclweb.org/anthology/P/P17/P17-1001.pdf
  '''
  def __init__(self, word_embed, data, word_dim, 
              pos_num, pos_dim, num_relations,
              keep_prob, num_filters,
              lrn_rate, is_train):
    # input data
    lexical, rid, direction, sentence, pos1, pos2 = data

    # embedding initialization
    w_trainable = True if FLAGS.word_dim==50 else False
    word_embed = tf.get_variable('word_embed', 
                                 initializer = word_embed, 
                                 dtype       = tf.float32,
                                 trainable   = w_trainable)
    pos1_embed = tf.get_variable('pos1_embed', shape=[pos_num, pos_dim])
    pos2_embed = tf.get_variable('pos2_embed', shape=[pos_num, pos_dim])

    # # embedding lookup
    lexical = tf.nn.embedding_lookup(word_embed, lexical) # batch_size, 6, word_dim
    lexical = tf.reshape(lexical, [-1, 6*word_dim])
    relation = tf.one_hot(rid, num_relations)

    sentence = tf.nn.embedding_lookup(word_embed, sentence)   # batch_size, max_len, word_dim
    pos1 = tf.nn.embedding_lookup(pos1_embed, pos1)       # batch_size, max_len, pos_dim
    pos2 = tf.nn.embedding_lookup(pos2_embed, pos2)       # batch_size, max_len, pos_dim

    # learn features from data
    # adversarial loss

    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    if is_train and keep_prob < 1:
      sent_pos = tf.nn.dropout(sent_pos, keep_prob)
    shared = cnn_forward('cnn-shared', sent_pos, None, num_filters, mtl=True)
    loss_adv, loss_l2 = adversarial_loss(shared, relation, is_train, keep_prob)

    # 10 classifiers for 10 tasks, task related loss
    # e.g. A-relation, B-relation and Other
    # task-A (A-relation): 3 class: (e1, e2), (e2, e1), other
    # task-B (B-relation): 3 class: (e1, e2), (e2, e1), other
    # task-O (Other)     : 2 class: true, false
    probs_buf = []
    loss_task = tf.constant(0, dtype=tf.float32)
    loss_diff = tf.constant(0, dtype=tf.float32)
    for task in range(num_relations):
      sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
      if is_train and keep_prob < 1:
        sent_pos = tf.nn.dropout(sent_pos, keep_prob)

      cnn_out = cnn_forward('cnn-%d'%task, sent_pos, None, num_filters)
      # feature 
      feature = tf.concat([cnn_out, shared, lexical], axis=1)
      # feature = tf.concat([cnn_out, lexical], axis=1)
      feature_size = feature.shape.as_list()[1]

      if is_train and keep_prob < 1:
        feature = tf.nn.dropout(feature, keep_prob)

      # task labels: 0:(e1,e2), 1:(e2,e1), 2:(other);  or 0:true, 1:false
      # self.rid:       5, 5, 7, 7, 1, O(Other)
      # self.direction: 0, 1, 0, 1, 0, 0
      # labels task==5  0, 1, 2, 2, 2, 2      3 class
      # labels task==7  2, 2, 0, 1, 2, 2      3 class
      # labels task==O  1, 1, 1, 1, 0, 0      2 class
      
      # Map the features to 3 or 2 classes
      num_class = 3
      if task == num_relations-1: # 'Other'
        num_class = 2
      logits, loss_l2_task = linear_layer('linear_%d'%task, feature, feature_size, num_class)

      probs = tf.nn.softmax(logits)
      # (batch,class) => (batch,class-1) dim 1, ignore last column
      probs = probs[:, :-1] 
      probs_buf.append(probs)
      
      # task specific loss
      other_mask = (num_class-1)*tf.ones_like(rid)
      task_labels = tf.where(tf.equal(rid, task), 
                             direction, 
                             other_mask)
      task_labels = tf.one_hot(task_labels, num_class)  # (batch, num_class)
      
      entropy = tf.reduce_mean(
                             tf.nn.softmax_cross_entropy_with_logits(
                                    labels = task_labels, 
                                    logits = logits))
      loss_task += entropy
      loss_l2 += loss_l2_task

      # Orthogonality Constraints
      # cnn_out = tf.nn.l2_normalize(cnn_out, -1)
      loss_diff += tf.reduce_sum(
                      tf.square(
                        tf.matmul(cnn_out, shared, transpose_a=True)
                      ))
    
    # get overall accuracy
    # self.rid:       5, 5, 7, 7, 1, O
    # self.direction: 0, 1, 0, 1, 0, 0
    
    # probs  task==5  0, 0, 0, 0, 0, 0      prob for 0 label
    # probs  task==5  1, 1, 1, 1, 1, 1      prob for 1 label

    # probs  task==7  0, 0, 0, 0, 0, 0      prob for 0 label
    # probs  task==7  1, 1, 1, 1, 1, 1      prob for 1 label

    # probs  task==O  0, 0, 0, 0, 0, 0      prob for 0 label
    
    # len(probs_buf) == r_10, [ p1, p2, .., pr_10]
    # p1.shape==(batch, 2), pr_10.shape==(batch, 1)
    # probs_buf => [batch, r_19]
    
    probs_buf = tf.concat(probs_buf, axis=1) # (batch, r_19)
    predicts = tf.argmax(probs_buf, axis=1, output_type=tf.int64) # (batch,)

    labels = 2 * rid + direction
    accuracy = tf.equal(predicts, labels)
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    self.logits = logits
    self.prediction = predicts
    self.accuracy = accuracy
    # self.loss = loss_task + 0.05*loss_adv + 0.01*loss_diff
    self.loss = loss_task + 0.01*loss_l2 + 0.00005*loss_adv #+ FLAGS.loss_diff_coef*loss_diff

    if not is_train:
      return 

    # global_step = tf.train.get_or_create_global_step()
    global_step = tf.Variable(0, trainable=False, name='step', dtype=tf.int32)
    optimizer = tf.train.AdamOptimizer(lrn_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):# for batch_norm
      self.train_op = optimizer.minimize(self.loss, global_step)
    self.global_step = global_step
   

def build_train_valid_model(word_embed, train_data, test_data):
  '''Adversarial Multi-task Learning for Text Classification'''
  with tf.name_scope("Train"):
    with tf.variable_scope('MTLModel', reuse=None):
      m_train = MTLModel( word_embed, train_data, FLAGS.word_dim,
                    FLAGS.pos_num, FLAGS.pos_dim, 10,
                    FLAGS.keep_prob, FLAGS.num_filters, 
                    FLAGS.lrn_rate, is_train=True)
  with tf.name_scope('Valid'):
    with tf.variable_scope('MTLModel', reuse=True):
      m_valid = MTLModel( word_embed, test_data, FLAGS.word_dim,
                    FLAGS.pos_num, FLAGS.pos_dim, 10,
                    1.0, FLAGS.num_filters, 
                    FLAGS.lrn_rate, is_train=False)
  return m_train, m_valid