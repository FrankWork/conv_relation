import tensorflow as tf
import os

N = 10
data_file = "data.csv"

if not os.path.exists(data_file):
  idx = 0
  with open(data_file, 'w') as f:
    while idx < N:
      data = [str(i) for i in range(idx, idx+5)]
      data = reversed(data)
      f.write(','.join(data) + '\n')
      idx += 1


filename_queue = tf.train.string_input_producer([data_file])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(N):
    # Retrieve a single instance:
    example, label = sess.run([features, col5])
    print(example, label)

  coord.request_stop()
  coord.join(threads)
