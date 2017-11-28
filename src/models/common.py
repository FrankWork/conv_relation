import os
import tensorflow as tf
from tensorflow.python.framework import ops

FLAGS = tf.app.flags.FLAGS

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

def linear_layer(name, x, in_size, out_size, is_regularize=False):
  with tf.variable_scope(name):
    loss_l2 = tf.constant(0, dtype=tf.float32)
    w = tf.get_variable('linear_W', [in_size, out_size], 
                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('linear_b', [out_size], 
                      initializer=tf.constant_initializer(0.1))
    o = tf.nn.xw_plus_b(x, w, b) # batch_size, out_size
    if is_regularize:
      loss_l2 += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return o, loss_l2

def conv2d(name, input, filter_size, num_filters):
  input_dim = input.shape.as_list()[2]
  with tf.variable_scope(name):
    conv_weight = tf.get_variable('kernel', 
                          [filter_size, input_dim, 1, num_filters], 
                          initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv = tf.nn.conv2d(input,
                        conv_weight,
                        strides=[1, 1, input_dim, 1],
                        padding='SAME')
  return conv

def cnn_forward(name, sent_pos, lexical, num_filters, mtl=False):
  with tf.variable_scope(name):
    input = tf.expand_dims(sent_pos, axis=-1)
    if mtl:
      input = grl_module.grl_op(input)
    input_dim = input.shape.as_list()[2]

    # convolutional layer
    pool_outputs = []
    for filter_size in [3,4,5]:
      with tf.variable_scope('conv-%s' % filter_size):
        conv_weight = tf.get_variable('W1', 
                              [filter_size, input_dim, 1, num_filters], 
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv_bias = tf.get_variable('b1', [num_filters], 
                              initializer=tf.constant_initializer(0.1))
        if mtl:
          conv_weight = grl_module.grl_op(conv_weight)
          conv_bias = grl_module.grl_op(conv_bias)
        conv = tf.nn.conv2d(input,
                            conv_weight,
                            strides=[1, 1, input_dim, 1],
                            padding='SAME')
        # Batch normalization here
        if mtl:
          conv = tf.layers.batch_normalization(conv)
        conv = tf.nn.relu(conv + conv_bias) # batch_size, max_len, 1, num_filters
        # max_len = conv.shape.as_list()[1]
        max_len = FLAGS.max_len
        pool = tf.nn.max_pool(conv, ksize= [1, max_len, 1, 1], 
                              strides=[1, max_len, 1, 1], padding='SAME') # batch_size, 1, 1, num_filters
        pool_outputs.append(pool)
    pools = tf.reshape(tf.concat(pool_outputs, 3), [-1, 3*num_filters])

    # feature 
    feature = pools
    if lexical is not None:
      feature = tf.concat([lexical, feature], axis=1)
    return feature

def wide_cnn_forward(sent_pos, lexical, max_len, num_filters, 
                                        is_train, filter_sizes=[1, 2, 3, 4, 5]):
  input = tf.expand_dims(sent_pos, axis=-1)
  # convolutional layer
  pool_outputs = []
  for filter_size in filter_sizes:
    with tf.variable_scope('conv-%s' % filter_size):
      conv1 = conv2d('conv1', input, filter_size, num_filters)
      conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train)
      bias1 = tf.get_variable('bias1', [num_filters], 
                        initializer=tf.constant_initializer(0.1))
      relu1 = tf.nn.relu(conv1 + bias1) # batch_size, max_len, 1, num_filters
      relu1 = tf.reshape(relu1, [-1, max_len, num_filters])
      relu1 = tf.expand_dims(relu1, axis=-1)

      conv2 = conv2d('conv2', relu1, filter_size, num_filters)
      conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train)
      bias2 = tf.get_variable('bias2', [num_filters], 
                        initializer=tf.constant_initializer(0.1))
      relu2 = tf.nn.relu(conv2 + bias2) # batch_size, max_len, 1, num_filters

      pool = tf.nn.max_pool(relu2, ksize= [1, max_len, 1, 1], 
                            strides=[1, max_len, 1, 1], padding='SAME') # batch_size, 1, 1, num_filters
      pool_outputs.append(pool)
  pools = tf.reshape(tf.concat(pool_outputs, 3), [-1, len(filter_sizes)*num_filters])

  # feature 
  feature = tf.concat([lexical, pools], axis=1)
  return feature

def rnn_forward_raw(input, rnn_size, rnn_layers, is_train, keep_prob):
  def single_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, reuse=tf.get_variable_scope().reuse)
    if is_train and keep_prob < 1:
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell

  def rnn_cells():
    cells = [single_cell() for _ in range(rnn_layers)]
    cells = tf.contrib.rnn.MultiRNNCell(cells)
    return cells
  
  fw_cell = rnn_cells()
  bw_cell = rnn_cells()

  # FIXME: peephole connections
  outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input, 
                                                    dtype=tf.float32)
  return outputs

def rnn_forward(input, max_len, rnn_size, rnn_layers, is_train, keep_prob):
  '''
  rnn with attention
  '''
  outputs = rnn_forward_raw(input, rnn_size, rnn_layers, is_train, keep_prob)

  outputs = tf.add(outputs[0], outputs[1])# batch, len, rnn_size
  # outputs = tf.concat([outputs[0], outputs[1]], axis=2) # batch, len, 2*rnn_size

  # attention layer
  hidden_size = max_len * rnn_size
  w = tf.get_variable('attention_W', [hidden_size, max_len], 
                        initializer=tf.contrib.layers.xavier_initializer())

  M = tf.reshape(outputs, [-1, hidden_size]) # batch, hsize
  M = tf.nn.relu(M) # batch,hsize
  alpha = tf.matmul(M, w) # batch,hsize hsize,len => batch,len
  alpha = tf.nn.softmax(alpha)# batch, len
  alpha = tf.expand_dims(alpha, axis=1) # batch,1,len
  feature = tf.matmul(alpha, outputs) # batch,1,len batch,len,rnn_size  => batch,1,rnn_size
  
  # feature 
  feature = tf.reshape(feature, [-1, rnn_size])
  feature = tf.nn.relu(feature)
  return feature