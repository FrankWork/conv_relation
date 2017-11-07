import tensorflow as tf
import numpy as np

vocab_size = 5
embed_size = 3

with tf.Graph().as_default(), tf.Session() as sess:
  # unk = tf.get_variable("unk", shape=[1, embed_size],
	# 		    dtype=tf.float32, initializer=tf.ones_initializer())
  # embed = [unk]
  # embed.append(tf.convert_to_tensor(np.zeros((vocab_size, embed_size)), dtype=tf.float32))
  # embed = tf.concat(embed, axis=0, name='concat_embed')

  embed = tf.get_variable('embed', initializer=np.ones((vocab_size, embed_size)))

  val = tf.trainable_variables()

  sess.run(tf.global_variables_initializer())

  val_np = sess.run(val)
  print(val_np)