# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for generating/preprocessing data for adversarial text models."""
# import operator
# import os
import random
# import re

import tensorflow as tf


class ShufflingTFRecordWriter(object):
  """Thin wrapper around TFRecordWriter that shuffles records."""

  def __init__(self, path):
    self._path = path
    self._records = []
    self._closed = False

  def write(self, record):
    assert not self._closed
    self._records.append(record)

  def close(self):
    assert not self._closed
    random.shuffle(self._records)
    with tf.python_io.TFRecordWriter(self._path) as f:
      for record in self._records:
        f.write(record)
    self._closed = True

  def __enter__(self):
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    self.close()


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

  def set_weight(self, weight):
    self._weight.float_list.value[0] = weight
    return self

  def copy_from(self, timestep):
    self.set_token(timestep.token).set_label(timestep.label).set_weight(
        timestep.weight)
    return self

  def _fill_with_defaults(self):
    if not self._multivalent_tokens:
      self._token.int64_list.value.append(0)
    self._label.int64_list.value.append(0)
    self._weight.float_list.value.append(0.0)

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

