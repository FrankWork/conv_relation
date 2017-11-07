import numpy as np
import tensorflow as tf

'''
dot product; scalar product; in Euclidean space
a = [a1, a2,…, an], b = [b1, b2,…, bn]：
a·b=a1b1+a2b2+……+anbn

inner product is a generalization of the dot product 
正交：內积为0
'''

n = 3
a = tf.random_normal([n, 1])
b = tf.random_normal([n, 1])

c = tf.reduce_sum(tf.multiply(a, b), keep_dims=True)
c2 = tf.matmul(a, b, transpose_a=True)

with tf.Session() as sess:


  a_np,b_np,c_np,c2_np = sess.run([a, b, c, c2])
  print(np.reshape(a_np,(n)))
  print(np.reshape(b_np,(n)))
  print('-'*40)

  
  print(c_np)
  print(c2_np)
  c3_np = np.dot(np.reshape(a_np, (n)), np.reshape(b_np, (n)))
  print(c3_np)