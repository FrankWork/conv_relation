import tensorflow as tf

with tf.variable_scope('foo'):
  a = tf.get_variable('a', [10])
  print(tf.get_variable_scope().name)

init = tf.global_variables_initializer()

for v in tf.trainable_variables():
  print(v.name)

with tf.Session() as sess:
  sess.run(init)

  val = sess.run(a)
  print(val)
