import tensorflow as tf

a = tf.random_normal([3, 2])
sum = tf.reduce_sum(a, 1, True)
loss = tf.reduce_sum(tf.maximum(sum, 0))

loss2 = tf.reduce_sum(tf.maximum(a, 0))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  sum, loss, loss2 = sess.run([sum, loss, loss2])
  print(sum)
  print(loss)
  print(loss2)
