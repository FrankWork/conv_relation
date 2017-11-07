import tensorflow as tf
import numpy as np

a = tf.random_normal([3, 2])
b = abs(a)
tb = tf.abs(a)

c = a**2
tc = tf.square(a)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  a_np, b_np, tb_np, c_np, tc_np = sess.run([a, b, tb, c, tc])
  print(a_np)

  if np.array_equal(b_np, tb_np):
    print('abs equal')
  if np.array_equal(c_np, tc_np):
    print('square equal')