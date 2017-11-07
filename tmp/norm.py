import tensorflow as tf


a = tf.ones([3, 2])
b = tf.nn.embedding_lookup(a, [0])
c = tf.nn.l2_normalize(b, 1)
d = tf.norm(a)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  va, vb, vc, vd = sess.run([a, b, c, d])
  print(va)
  print(vb)
  print(vc)
  print(vd)