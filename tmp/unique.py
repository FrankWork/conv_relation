import tensorflow as tf

a = tf.ones([5])
b = tf.zeros([5])
c = tf.concat([a, b], axis=0)
u, _ = tf.unique(c)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  c, u = sess.run([c, u])
  print(c)
  print(u)