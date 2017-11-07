import tensorflow as tf

h = tf.reshape(tf.range(0, 10), [1, 10]) # 1, 10
r = tf.one_hot(tf.range(0, 3), 10, dtype=tf.int32) # 3, 10
y = h - r # 3, 10

h_batch = tf.tile(tf.reshape(tf.range(0, 10), [10, 1]), [1, 10]) # 10, 10
h_3d = tf.tile(tf.expand_dims(h_batch, axis=1), [1, 3, 1]) # 10, 3, 10
# y_batch = h_3d - tf.tile(tf.reshape(r, [1, 3, 10]), [10, 1, 1]) # (10, 10) 10, 3, 10
y_batch = h_3d - tf.reshape(r, [1, 3, 10]) # (10, 10) 10, 3, 10

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())


  vr, vh, vy = sess.run([r, h, y])
  print(vr)
  print(vh)
  print(vy)
  print('='*10)
  vh, vy = sess.run([h_3d, y_batch])
  print(vh)
  print(vy)