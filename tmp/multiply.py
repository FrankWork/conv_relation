import random
import tensorflow as tf
import numpy as np

n = 3
a = np.random.random((n,1))
b = np.random.random((n,1))
m_np = np.multiply(a, b)
print(m_np)


m_tf = tf.multiply(tf.convert_to_tensor(a),
                   tf.convert_to_tensor(b))
x = tf.random_normal([n,1], dtype=tf.float64)
y = tf.random_normal([n,1], dtype=tf.float64)
m_tf2 = tf.multiply(x, y)
with tf.Session() as sess:
  
  print(np.equal(m_np, sess.run(m_tf)))
  
  print('-'*80)

  x_val, y_val, m_tf_val = sess.run([x, y, m_tf2]) # it must fetch value in a single run()
  m_np = np.multiply(x_val, y_val)
  
  print(np.equal(m_np, m_tf_val))
