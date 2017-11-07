import tensorflow as tf
import numpy as np


sess = tf.Session()

rid = np.random.randint(0, 5, (10))
print(rid)
print('='*10)

prob_buf = []
for i in range(5):
  labels = tf.cast(tf.equal(rid, i), tf.int32)
  labels = tf.one_hot(labels, 2)
  prob_buf.append(labels) # (batch, 2)

  val = sess.run(labels)
  print(val)

prob_buf = tf.stack(prob_buf, axis=1) # (r, batch, 2) => (batch, r, 2)
print(prob_buf.shape)  
# print(sess.run(prob_buf))

predicts = tf.argmax(prob_buf[:,:, 1] - prob_buf[:,:,0], axis=1)
print(sess.run(predicts))
print(sess.run(tf.equal(predicts, rid)))
# predict = tf.arg_max(prob_buf[1:]-prob_buf[0], 0)