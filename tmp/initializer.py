import tensorflow as tf
import numpy as np

vocab_size = 5
embed_size = 3

with tf.Graph().as_default(), tf.Session() as sess:
  initializer = tf.truncated_normal_initializer()
  embed = tf.get_variable('embed', shape=[vocab_size, embed_size], initializer=initializer)
  embed2 = tf.get_variable('embed2', shape=[vocab_size, embed_size], initializer=initializer)

  sess.run(tf.global_variables_initializer())

  e1, e2 = sess.run([embed, embed2])
  print(e1)
  print(e2)