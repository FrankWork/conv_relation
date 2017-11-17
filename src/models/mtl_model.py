import random
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from models.base_model import BaseModel
from .common import *


# compile:
# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
# g++ -std=c++11 -shared grl_op.cc -o grl_op.so -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2

# load op library
op_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'grl_op.so')
grl_module = tf.load_op_library(op_path)

@ops.RegisterGradient("GrlOp")
def _grl_op_grad(op, grad):
  """The gradients for `grl_op` (gradient reversal layer).

  Args:
    op: The `grl_op` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `grl_op` op.

  Returns:
    Gradients with respect to the input of `grl_op`.
  """
  return [-grad]  # List of one Tensor, since we have one input

FLAGS = tf.app.flags.FLAGS

def cnn_forward_lite(name, sent_pos, max_len, num_filters, use_grl=False):
  with tf.variable_scope(name, reuse=None):
    input = tf.expand_dims(sent_pos, axis=-1)
    if use_grl:
      input = grl_module.grl_op(input)
    input_dim = input.shape.as_list()[2]

    # convolutional layer
    # pool_outputs = []
    # filter_size = random.choice([1,2,3,4,5])
    filter_size = 3
    with tf.variable_scope('conv-%s' % filter_size):
      conv_weight = tf.get_variable('W1', 
                            [filter_size, input_dim, 1, num_filters], 
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
      conv_bias = tf.get_variable('b1', [num_filters], 
                            initializer=tf.constant_initializer(0.1))
      if use_grl:
        conv_weight = grl_module.grl_op(conv_weight)
        conv_bias = grl_module.grl_op(conv_bias)
      conv = tf.nn.conv2d(input,
                          conv_weight,
                          strides=[1, 1, input_dim, 1],
                          padding='SAME')
      # Batch normalization here
      conv = tf.nn.relu(conv + conv_bias) # batch_size, max_len, 1, num_filters
      pool = tf.nn.max_pool(conv, ksize= [1, max_len, 1, 1], 
                            strides=[1, max_len, 1, 1], padding='SAME') # batch_size, 1, 1, num_filters
      # pool_outputs.append(pool)
    # pools = tf.reshape(tf.concat(pool_outputs, 3), [-1, 3*num_filters])
    pools = tf.reshape(pool, [-1, num_filters])

    return pools

def adversarial_loss(feature, relation, is_train, keep_prob):
  feature_size = feature.shape.as_list()[1]
  if is_train and keep_prob < 1:
      feature = tf.nn.dropout(feature, keep_prob)
  # Map the features to 19 classes
  out_size = relation.shape.as_list()[1]
  logits, _ = linear_layer('linear_adv', feature, feature_size, out_size)
  loss_adv = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=relation, logits=logits))
  return loss_adv

class MTLModel(BaseModel):
  '''
  Adversarial Multi-task Learning for Text Classification
  http://www.aclweb.org/anthology/P/P17/P17-1001.pdf
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
    self.direction = tf.placeholder(tf.int32, [None])

    # embedding initialization
    word_embed = tf.get_variable('word_embed', 
                                 initializer = word_embed, 
                                 dtype       = tf.float32,
                                 trainable   = False)
    pos_embed = tf.get_variable('pos_embed', shape=[pos_num, pos_dim])

    # # embedding lookup
    lexical = tf.nn.embedding_lookup(word_embed, self.lexical_id) # batch_size, 6, word_dim
    lexical = tf.reshape(lexical, [-1, 6*word_dim])

    sentence = tf.nn.embedding_lookup(word_embed, self.sent_id)   # batch_size, max_len, word_dim
    pos1 = tf.nn.embedding_lookup(pos_embed, self.pos1_id)       # batch_size, max_len, pos_dim
    pos2 = tf.nn.embedding_lookup(pos_embed, self.pos2_id)       # batch_size, max_len, pos_dim
    relation = tf.one_hot(self.rid, num_relations)

    # learn features from data
    # adversarial loss

    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    if is_train and keep_prob < 1:
      sent_pos = tf.nn.dropout(sent_pos, keep_prob)
    shared = cnn_forward_lite('cnn-shared', sent_pos, max_len, num_filters, use_grl=True)
    loss_adv = adversarial_loss(shared, relation, is_train, keep_prob)

    # 10 classifiers for 10 tasks, task related loss
    # e.g. A-relation, B-relation and Other
    # task-A (A-relation): 3 class: (e1, e2), (e2, e1), other
    # task-B (B-relation): 3 class: (e1, e2), (e2, e1), other
    # task-O (Other)     : 2 class: true, false
    probs_buf = []
    task_features = []
    loss_task = tf.constant(0, dtype=tf.float32)
    for task in range(num_relations):
      sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
      if is_train and keep_prob < 1:
        sent_pos = tf.nn.dropout(sent_pos, keep_prob)

      cnn_out = cnn_forward_lite('cnn-%d'%task, sent_pos, max_len, num_filters)
      # feature 
      task_features.append(cnn_out)
      feature = tf.concat([cnn_out, shared, lexical], axis=1)
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
      logits, _ = linear_layer('linear_%d'%task, feature, feature_size, num_class)

      probs = tf.nn.softmax(logits)
      # (batch,class) => (batch,class-1) dim 1, ignore last column
      probs = probs[:, :-1] 
      probs_buf.append(probs)
      
      other_mask = (num_class-1)*tf.ones_like(self.rid)
      task_labels = tf.where(tf.equal(self.rid, task), 
                             self.direction, 
                             other_mask)
      task_labels = tf.one_hot(task_labels, num_class)  # (batch, num_class)
      
      entropy = tf.reduce_mean(
                             tf.nn.softmax_cross_entropy_with_logits(
                                    labels = task_labels, 
                                    logits = logits))
      loss_task += entropy
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
    # FIXME biased
    predicts = tf.argmax(probs_buf, axis=1, output_type=tf.int32) # (batch,)

    labels = 2 * self.rid + self.direction
    accuracy = tf.equal(predicts, labels)
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    # Orthogonality Constraints
    task_features = tf.stack(task_features, axis=1) # (r, batch, hsize) => (batch, r, hsize)
    shared = tf.expand_dims(shared, axis=2)# (batch, hsize, 1)
    loss_diff = tf.reduce_sum(
      tf.pow(tf.matmul(task_features, shared), 2)
    )

    self.logits = logits
    self.prediction = predicts
    self.accuracy = accuracy
    # self.loss = loss_task + 0.05*loss_adv + 0.01*loss_diff
    self.loss = loss_task + loss_adv + loss_diff

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
  '''Adversarial Multi-task Learning for Text Classification'''
  with tf.name_scope("Train"):
    with tf.variable_scope('MTLModel', reuse=None):
      m_train = MTLModel( word_embed, FLAGS.word_dim, FLAGS.max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, 10,
                    FLAGS.keep_prob, FLAGS.filter_size, FLAGS.num_filters, 
                    FLAGS.lrn_rate, FLAGS.decay_steps, FLAGS.decay_rate, is_train=True)
  with tf.name_scope('Valid'):
    with tf.variable_scope('MTLModel', reuse=True):
      m_valid = MTLModel( word_embed, FLAGS.word_dim, FLAGS.max_len,
                    FLAGS.pos_num, FLAGS.pos_dim, 10,
                    1.0, FLAGS.filter_size, FLAGS.num_filters, 
                    FLAGS.lrn_rate, FLAGS.decay_steps, FLAGS.decay_rate, is_train=False)
  return m_train, m_valid