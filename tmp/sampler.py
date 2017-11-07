import tensorflow as tf


ids = tf.range(10)
prob = tf.random_normal([10])

cond = tf.less(prob, tf.constant(0.5, shape=[10])) #tf.where())
mask = tf.where(cond, tf.constant(0, shape=[10]), ids)

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)

  # ids, negs = sess.run([ids, negs])
  # print(ids)
  # print(negs)
  ids, mask, prob = sess.run([ids, mask, prob])
  print(ids)
  print(mask)
  print(prob)