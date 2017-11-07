import tensorflow as tf


global_step = tf.train.get_or_create_global_step()
lrn_rate = tf.train.exponential_decay(1e-2, global_step, 1, 0.97, staircase=True)
add_op = tf.assign(global_step, global_step+1)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(100):
    _, rate = sess.run([add_op, lrn_rate])
    print(rate)