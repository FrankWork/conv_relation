from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import tensorflow as tf

N = 10
data_file = "data.tfrecords"
batch_size = 2
num_threads = 1
num_epochs = 1

# if not os.path.exists(data_file):
writer = tf.python_io.TFRecordWriter(data_file)
idx = 0
while idx < N:
  example = tf.train.Example(features=tf.train.Features(feature={
    'feature': tf.train.Feature(int64_list=tf.train.Int64List(value=[idx*10, idx*100, idx*1000, idx*10000])),
    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[idx]))
    }))
  writer.write(example.SerializeToString())
  idx += 1
writer.close()

def batch_read():
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    filename_queue = tf.train.string_input_producer([data_file], 
                                    num_epochs=num_epochs, shuffle=False)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    example = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
            'feature': tf.FixedLenFeature([4], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
          })

    feature = tf.cast(example['feature'], tf.int64)
    label = tf.cast(example['label'], tf.int64)
    feature_batch, label_batch = tf.train.shuffle_batch(
          [feature, label], batch_size=batch_size, 
          num_threads=num_threads, capacity=10 + 3 * batch_size,
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue=10)
    y = label_batch
    init_op = tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer())

    sv = tf.train.Supervisor(logdir=None)#, global_step=m_train.global_step)
    with sv.managed_session() as sess:
      sess.run(init_op)

      try:
        # while not coord.should_stop(): 
        while not sv.should_stop():
          labels_np = sess.run(y)
          print(labels_np)
          # label_np = sess.run(label)
          # print(label_np)
      except tf.errors.OutOfRangeError:
        print('Done training')

def read():
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    filename_queue = tf.train.string_input_producer([data_file], 
                                    num_epochs=num_epochs, shuffle=False)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    example = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
            'feature': tf.FixedLenFeature([4], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
          })

    feature = tf.cast(example['feature'], tf.int64)
    label = tf.cast(example['label'], tf.int64)

    # f_batch, l_batch = tf.train.batch([feature, label], 1)
    init_op = tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer())

    sv = tf.train.Supervisor(logdir=None)#, global_step=m_train.global_step)
    with sv.managed_session() as sess:
      sess.run(init_op)
      
      try:
        # while not coord.should_stop(): 
        while not sv.should_stop():
          f, l = sess.run([feature, label])
          print(f, l)
      except tf.errors.OutOfRangeError:
        print('Done training')

read()