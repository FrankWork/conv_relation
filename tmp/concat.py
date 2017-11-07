import tensorflow as tf


a = tf.truncated_normal([2, 3, 4])
b = tf.truncated_normal([2, 3, 5])

concat1 = tf.concat([a, b], 2)
print(concat1.shape)

unstack1 = tf.unstack(concat1, axis=1)
concat2 = tf.concat(unstack1, axis=1)
print(concat2.shape)

reshape = tf.reshape(concat1, [2, 27])

bool = tf.equal(concat2, reshape)



with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  bool = sess.run(bool)
  print(bool)
  
