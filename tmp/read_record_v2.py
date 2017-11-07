from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import tensorflow as tf

N = 10
data_file = "data.tfrecords"
batch_size = 1
num_threads = 1
num_epochs = 1


class Timestep(object):
  """Represents a single timestep in a SequenceWrapper."""

  def __init__(self, token, label):
    """Constructs Timestep from empty Features."""
    self._token = token
    self._label = label
    self._fill_with_defaults()

  @property
  def token(self):
    return self._token.int64_list.value[0]

  @property
  def label(self):
    return self._label.int64_list.value[0]

  def set_token(self, token):
    self._token.int64_list.value[0] = token
    return self

  def set_label(self, label):
    self._label.int64_list.value[0] = label
    return self

  def copy_from(self, timestep):
    self.set_token(timestep.token).set_label(timestep.label).set_weight(
        timestep.weight)
    return self

  def _fill_with_defaults(self):
    self._token.int64_list.value.append(0)
    self._label.int64_list.value.append(0)


class SequenceWrapper(object):
  """Wrapper around tf.SequenceExample."""

  F_TOKEN_ID = 'token_id'
  F_LABEL = 'label'

  def __init__(self):
    self._seq = tf.train.SequenceExample()
    self._flist = self._seq.feature_lists.feature_list
    self._timesteps = []

  @property
  def seq(self):
    return self._seq

  @property
  def _tokens(self):
    return self._flist[SequenceWrapper.F_TOKEN_ID].feature

  @property
  def _labels(self):
    return self._flist[SequenceWrapper.F_LABEL].feature

  def add_timestep(self):
    timestep = Timestep(
        self._tokens.add(),
        self._labels.add()
    )
    self._timesteps.append(timestep)
    return timestep

  def __iter__(self):
    for timestep in self._timesteps:
      yield timestep

  def __len__(self):
    return len(self._timesteps)

  def __getitem__(self, idx):
    return self._timesteps[idx]

def write():
  writer = tf.python_io.TFRecordWriter(data_file)
  idx = 1
  while idx <= N:
    seq = SequenceWrapper()
    for i in range(idx):
      seq.add_timestep().set_token(i).set_label(2*i)

    writer.write(seq.seq.SerializeToString())
    idx += 1
  writer.close()

def read():
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    filename_queue = tf.train.string_input_producer([data_file], 
                                    num_epochs=num_epochs, shuffle=False)
    reader = tf.TFRecordReader()
    key, serialized_record = reader.read(filename_queue)

    ctx, sequence = tf.parse_single_sequence_example(
      serialized_record,
      sequence_features={
          SequenceWrapper.F_TOKEN_ID:
              tf.FixedLenSequenceFeature([], dtype=tf.int64),
          SequenceWrapper.F_LABEL:
              tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

    feature = tf.cast(sequence[SequenceWrapper.F_TOKEN_ID], tf.int64)
    label = tf.cast(sequence[SequenceWrapper.F_LABEL], tf.int64)

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

def batch_read():
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    filename_queue = tf.train.string_input_producer([data_file], 
                                    num_epochs=num_epochs, shuffle=False)
    reader = tf.TFRecordReader()
    key, serialized_record = reader.read(filename_queue)

    ctx, sequence = tf.parse_single_sequence_example(
      serialized_record,
      sequence_features={
          SequenceWrapper.F_TOKEN_ID:
              tf.FixedLenSequenceFeature([], dtype=tf.int64),
          SequenceWrapper.F_LABEL:
              tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

    feature = tf.cast(sequence[SequenceWrapper.F_TOKEN_ID], tf.int64)
    label = tf.cast(sequence[SequenceWrapper.F_LABEL], tf.int64)

    f_batch, l_batch = tf.train.batch([feature, label], 1)

    init_op = tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer())

    sv = tf.train.Supervisor(logdir=None)#, global_step=m_train.global_step)
    with sv.managed_session() as sess:
      sess.run(init_op)
      
      try:
        # while not coord.should_stop(): 
        while not sv.should_stop():
          f, l = sess.run([f_batch, l_batch])
          print(f, l)
      except tf.errors.OutOfRangeError:
        print('Done training')

write()
read()
# batch_read()