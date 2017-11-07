
# compile:
# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# g++ -std=c++11 -shared grl_op.cc grl_kernel.cc -o grl_op.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

import os
import tensorflow as tf
import grl_gradient

op_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'grl_op.so')
grl_module = tf.load_op_library(op_path)

x = tf.random_normal([3, 4])
w = tf.get_variable('w', [4,2], dtype=tf.float32)
b = tf.get_variable('b', [2], dtype=tf.float32)

y = tf.nn.xw_plus_b(x, w, b)
g = tf.gradients(y,[w, b])

y2 = tf.nn.xw_plus_b(x, grl_module.grl_op(w), b)
g2 = tf.gradients(y2, [w, b])

y_bool = tf.equal(y, y2)
g0_bool = tf.equal(g[0], -g2[0])
g1_bool = tf.equal(g[1], g2[1])

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  print('-'*80)
  sess.run(init_op)

  print(sess.run(y_bool))
  print(sess.run(g0_bool))
  print(sess.run(g1_bool))
  