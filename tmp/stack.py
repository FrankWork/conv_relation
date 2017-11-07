import tensorflow as tf

length = tf.range(1, 11)
idx = tf.stack([tf.range(10), length-1], axis=1)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  vidx = sess.run(idx)
  print(vidx)