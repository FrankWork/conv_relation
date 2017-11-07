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
"""Generates vocabulary and term frequency files for datasets."""


import os
import re
from collections import defaultdict
from collections import namedtuple

# Dependency imports

import tensorflow as tf

# from adversarial_text.data import data_utils
# from adversarial_text.data import document_generators

flags = tf.app.flags
FLAGS = flags.FLAGS

# Flags controlling input are in document_generators.py

flags.DEFINE_string('vocab_file', 'vocab.txt', 'Path to save vocab.txt.')
flags.DEFINE_string('vocab_freq_file', 'vocab_freq.txt', 'Path to save vocab_freq.txt.')

flags.DEFINE_integer('vocab_count_threshold', 1, 'The minimum number of '
                     'a word or bigram should occur to keep '
                     'it in the vocabulary.')

Example = namedtuple('Example', 'label entity1 entity2 sentence')
PositionPair = namedtuple('PosPair', 'first last')

def dataset(data_file):
  '''load dataset from text file
  '''
  data = []
  data_path = os.path.join(FLAGS.data_dir, data_file)
  with open(data_path) as f:
    for line in f:
      tokens = line.strip().lower().split() # space seperated
      # tokens = line.strip().split() # space seperated
      
      label = int(tokens[0])
      entity1 = PositionPair(int(tokens[1]), int(tokens[2]))
      entity2 = PositionPair(int(tokens[3]), int(tokens[4]))

      sentence = []
      for w in tokens[5:]:
        # TODO: replace digits with <digit/> token
        w = re.sub(r"[0-9]", "0", w) # replace digits with 0s
        sentence.append(w)
      example = Example(label, entity1, entity2, sentence)
      data.append(example)
  return data


def gen_vocab(train_file):
  tf.logging.set_verbosity(tf.logging.INFO)

  vocab_freqs = defaultdict(int)

  # Fill vocabulary frequencies map 
  for example in dataset(train_file):
    for token in example.sentence:
      vocab_freqs[token] += 1

  # Filter out low-occurring terms
  vocab_freqs = dict((term, freq) for term, freq in vocab_freqs.items()
                     if vocab_freqs[term] > FLAGS.vocab_count_threshold)

  # Sort by frequency
  ordered_vocab_freqs = sorted(
      vocab_freqs.items(), key= lambda item: item[1], reverse=True)

  # Limit vocab size
  if MAX_VOCAB_SIZE:
    ordered_vocab_freqs = ordered_vocab_freqs[:MAX_VOCAB_SIZE]

  # Add EOS token
  ordered_vocab_freqs.append((EOS_TOKEN, 1))

  # Write
  # tf.gfile.MakeDirs(FLAGS.output_dir)
  vocab_path = os.path.join(FLAGS.data_dir, FLAGS.vocab_file)
  vocab_freq_path = os.path.join(FLAGS.data_dir, FLAGS.vocab_freq_file)
  
  with open(vocab_path, 'w') as vocab_f:
    with open(vocab_freq_path, 'w') as freq_f:
      for word, freq in ordered_vocab_freqs:
        vocab_f.write('{}\n'.format(word))
        freq_f.write('{}\n'.format(freq))

