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


