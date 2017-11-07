import tensorflow as tf

sent_id = tf.placeholder(tf.int32, [None, 3])
feed = {'sent_id': [[1,2,3],[4,5,6]]}
with tf.Session() as sess:
  s = sess.run(sent_id, {sent_id: feed['sent_id']})
  print(s)