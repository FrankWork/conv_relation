import tensorflow as tf

a = tf.range(0, 10)
b = tf.unstack(a)
print(type(b))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  # a = sess.run(a)
  # print(a)
  b = sess.run(b)
  print(b)