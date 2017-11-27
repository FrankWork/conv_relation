import os
import re
import numpy as np
import tensorflow as tf
from collections import defaultdict
from collections import namedtuple


PAD_WORD = "<pad>"
START_WORD = "<start>"
END_WORD = "<end>"
PAD_ID, START_ID, END_ID = 0,1,2

Raw_Example = namedtuple('Raw_Example', 'label entity1 entity2 sentence')
PositionPair = namedtuple('PosPair', 'first last')
MTL_Label = namedtuple('MTL_Label', 'relation direction')

FLAGS = tf.app.flags.FLAGS # load FLAGS.word_dim

def load_raw_data(filename):
  '''load raw data from text file, 
  and convert word to lower case, 
  and replace digits with 0s;

  return: a list of Raw_Example
  '''
  data = []
  with open(filename) as f:
    for line in f:
      # example = Raw_Example()
      words = line.strip().lower().split()
      
      sent = words[5:]
      sent = [re.sub("\d+", "0", w) for w in sent]

      label = int(words[0])
      if FLAGS.model=='mtl':
        # label = 0,      1,      2,     3
        # mtl_  = (0, 0)  (0, 1)  (1, 0) (1, 1)       
        label = MTL_Label(label//2, label%2)

      entity1 = PositionPair(int(words[1]), int(words[2]))
      entity2 = PositionPair(int(words[3]), int(words[4]))

      example = Raw_Example(label, entity1, entity2, sent)
      data.append(example)
  return data

def build_vocab(raw_train_data, raw_test_data, vocab_file, vocab_freq_file):
  '''collect words in sentence'''
  if not os.path.exists(vocab_file):
    # load words in data
    word_freqs = defaultdict(int)
    for example in raw_train_data:
      for w in example.sentence:
        word_freqs[w] += 1

    for example in raw_test_data:
      for w in example.sentence:
        word_freqs[w] += 1
    
    # write vocab
    with open(vocab_file, 'w') as vocab_f:
      with open(vocab_freq_file, 'w') as freq_f:
        vocab_f.write(PAD_WORD+'\n')
        vocab_f.write(START_WORD+'\n')
        vocab_f.write(END_WORD+'\n')
        freq_f.write('1\n')
        freq_f.write('1\n')
        freq_f.write('1\n')

        for w in sorted(word_freqs.keys()):
          # if word_freqs[w] > FLAGS.freq_threshold:
          vocab_f.write(w + '\n')
          freq_f.write('%d\n' % word_freqs[w])
  
  word2id = {}
  id2word = {}
  with open(vocab_file) as f:
    for i, line in enumerate(f):
      w = line.strip()
      word2id[w] = i
      id2word[id] = w
  print('%d words' % len(word2id)) 
  return word2id, id2word

def gen_google_embeddings(word2id, word_embed_orig, word_embed_trim):
  '''trim unnecessary words from original pre-trained word embedding

  Args:
    word2id: dict, {'I':0, 'you': 1, ...}
    word_embed_oirg: string, file name of the original pre-trained embedding
    word_embed_trim: string, file name of the trimmed embedding
  '''
  if not os.path.exists(word_embed_trim):
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(word_embed_orig, binary=True)
    
    shape = (len(word2id), model.vector_size)
    word_embed = np.zeros(shape, dtype=np.float32)
    for w, i in word2id.items():
      if w in model:
        word_embed[i] = model[w]
    np.save(word_embed_trim, word_embed)
  
  word_embed = np.load(word_embed_trim)
  # FLAGS.word_dim = word_embed.shape[1]
  zeros = np.zeros((FLAGS.word_dim), dtype=np.float32)
  n_zero = 0
  for i in range(word_embed.shape[0]):
    if np.array_equal(word_embed[i], zeros):
      n_zero += 1
  print("%d zero vectors." % n_zero)
  return word_embed

def gen_senna_embeddings(word2id, 
                        word_embed_orig,
                        senna_words_lst,
                        word_embed_trim):
  '''trim unnecessary words from original pre-trained word embedding

  Args:
    word2id: dict, {'I':0, 'you': 1, ...}
    word_embed_oirg: string, file name of the original pre-trained embedding
    senna_words_lst: string, file name of the senna words list w.r.t the embed
    word_embed_trim: string, file name of the trimmed embedding
  '''
  if not os.path.exists(word_embed_trim):
    pre_embed = dict()

    wl_senna = open(senna_words_lst, "r").readlines()
    em_senna = open(word_embed_orig, "r").readlines()
    for idx in range(len(wl_senna)):
      word = wl_senna[idx].strip()
      line_tokens = em_senna[idx].strip().split()
      embedding = [float(x) for x in line_tokens]
      pre_embed[word] = embedding

    shape = (len(word2id), FLAGS.word_dim)
    word_embed = np.zeros(shape, dtype=np.float32)

    for w, i in word2id.items():
      if w in pre_embed:
        word_embed[i] = pre_embed[w]
    np.save(word_embed_trim, word_embed)
  
  word_embed = np.load(word_embed_trim)
  # FLAGS.word_dim = word_embed.shape[1]
  zeros = np.zeros((FLAGS.word_dim), dtype=np.float32)
  n_zero = 0
  for i in range(word_embed.shape[0]):
    if np.array_equal(word_embed[i], zeros):
      n_zero += 1
  print("%d zero vectors." % n_zero)
  return word_embed

def map_words_to_id(raw_data, word2id):
  '''inplace convert sentence from a list of words to a list of ids
  Args:
    raw_data: a list of Raw_Example
    word2id: dict, {word: id, ...}
  '''
  for raw_example in raw_data:
    for idx, word in enumerate(raw_example.sentence):
      raw_example.sentence[idx] = word2id[word]

def _lexical_feature(raw_example):
  def _entity_context(e_idx, sent):
    ''' return [w(e-1), w(e), w(e+1)]
    '''
    context = []
    context.append(sent[e_idx])

    if e_idx == 0:
      context.append(sent[e_idx])
    else:
      context.append(sent[e_idx-1])
    
    if e_idx == len(sent)-1:
      context.append(sent[e_idx])
    else:
      context.append(sent[e_idx+1])
    
    return context

    
  e1_idx = raw_example.entity1.last
  e2_idx = raw_example.entity2.last

  context1 = _entity_context(e1_idx, raw_example.sentence)
  context2 = _entity_context(e2_idx, raw_example.sentence)

  # ignore WordNet hypernyms in paper
  lexical = context1 + context2
  return lexical

def _position_feature(raw_example):
  def distance(n):
    '''convert relative distance to positive number
    '''
    # FIXME: FLAGS.pos_num
    if n < -60:
      return 0
    if n >= -60 and n <= 60:
      return n + 61
    if n > 60:
      return 122

  e1_idx = raw_example.entity1.last
  e2_idx = raw_example.entity2.last

  position1 = []
  position2 = []
  length = len(raw_example.sentence)
  for i in range(length):
    position1.append(distance(i-e1_idx))
    position2.append(distance(i-e2_idx))
  
  return position1, position2

def build_sequence_example(raw_example):
  '''build tf.train.SequenceExample from Raw_Example
  context features : lexical, rid, direction (mtl)
  sequence features: sentence, position1, position2

  Args: 
    raw_example : type Raw_Example

  Returns:
    tf.trian.SequenceExample
  '''
  ex = tf.train.SequenceExample()

  lexical = _lexical_feature(raw_example)
  ex.context.feature['lexical'].int64_list.value.extend(lexical)

  if isinstance(raw_example.label, MTL_Label):
    rid = raw_example.label.relation
    direction = raw_example.label.direction
    ex.context.feature['rid'].int64_list.value.append(rid)
    ex.context.feature['direction'].int64_list.value.append(direction)
  else:
    rid = raw_example.label
    ex.context.feature['rid'].int64_list.value.append(rid)

  for word_id in raw_example.sentence:
    word = ex.feature_lists.feature_list['sentence'].feature.add()
    word.int64_list.value.append(word_id)
  
  position1, position2 = _position_feature(raw_example)
  for pos_val in position1:
    pos = ex.feature_lists.feature_list['position1'].feature.add()
    pos.int64_list.value.append(pos_val)
  for pos_val in position2:
    pos = ex.feature_lists.feature_list['position2'].feature.add()
    pos.int64_list.value.append(pos_val)

  return ex

def maybe_write_tfrecord(raw_data, filename):
  '''if the destination file is not exist on disk, convert the raw_data to 
  tf.trian.SequenceExample and write to file.

  Args:
    raw_data: a list of 'Raw_Example'
  '''
  if not os.path.exists(filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for raw_example in raw_data:
      example = build_sequence_example(raw_example)
      writer.write(example.SerializeToString())
    writer.close()

def _parse_tfexample(serialized_example):
  '''parse serialized tf.train.SequenceExample to tensors
  context features : lexical, rid, direction (mtl)
  sequence features: sentence, position1, position2
  '''
  context_features={
                      'lexical'   : tf.FixedLenFeature([6], tf.int64),
                      'rid'    : tf.FixedLenFeature([], tf.int64)}
  if FLAGS.model == 'mtl':
    context_features['direction'] = tf.FixedLenFeature([], tf.int64)
  sequence_features={
                      'sentence' : tf.FixedLenSequenceFeature([], tf.int64),
                      'position1'  : tf.FixedLenSequenceFeature([], tf.int64),
                      'position2'  : tf.FixedLenSequenceFeature([], tf.int64)}
  context_dict, sequence_dict = tf.parse_single_sequence_example(
                      serialized_example,
                      context_features   = context_features,
                      sequence_features  = sequence_features)

  sentence = sequence_dict['sentence']
  position1 = sequence_dict['position1']
  position2 = sequence_dict['position2']

  lexical = context_dict['lexical']
  rid = context_dict['rid']
  if FLAGS.model == 'mtl':
    direction = context_dict['direction']
    return lexical, rid, direction, sentence, position1, position2
  return lexical, rid, sentence, position1, position2

def read_tfrecord_to_batch(filename, epoch, batch_size, shuffle=True):
  '''read TFRecord file to get batch tensors for tensorflow models

  Returns:
    a tuple of batched tensors
  '''
  dataset = tf.data.TFRecordDataset([filename])
  # Parse the record into tensors
  dataset = dataset.map(_parse_tfexample) 
  if shuffle:
    dataset = dataset.shuffle(buffer_size=100)
  dataset = dataset.repeat(epoch)

  # [] for no padding, [None] for padding to maximum length
  if FLAGS.model == 'mtl':
    # lexical, rid, direction, sentence, position1, position2
    padded_shapes = ([None,], [], [], [None], [None], [None])
  else:
    # lexical, rid, sentence, position1, position2
    padded_shapes = ([None,], [], [None], [None], [None])
  dataset = dataset.padded_batch(batch_size, padded_shapes)
  
  iterator = dataset.make_one_shot_iterator()
  batch = iterator.get_next()
  return batch


def inputs():
  raw_train_data = load_raw_data(FLAGS.train_file)
  raw_test_data = load_raw_data(FLAGS.test_file)

  word2id, id2word = build_vocab(raw_train_data, 
                                 raw_test_data, 
                                 FLAGS.vocab_file, 
                                 FLAGS.vocab_freq_file)

  if FLAGS.word_dim == 50:
    word_embed = gen_senna_embeddings(word2id,
                              FLAGS.word_embed50_orig,
                              FLAGS.senna_words_lst,
                              FLAGS.word_embed50_trim)
  elif FLAGS.word_dim == 300:
    word_embed = gen_google_embeddings(word2id,
                              FLAGS.word_embed300_orig, 
                              FLAGS.word_embed300_trim)

  # map words to ids
  map_words_to_id(raw_train_data, word2id)
  map_words_to_id(raw_test_data, word2id)

  # convert raw data to TFRecord format data, and write to file
  if FLAGS.model=='mtl':
    train_record = FLAGS.train_mtl_record
    test_record = FLAGS.test_mtl_record  
  else:
    train_record = FLAGS.train_record
    test_record = FLAGS.test_record
  
  maybe_write_tfrecord(raw_train_data, train_record)
  maybe_write_tfrecord(raw_test_data, test_record)

  train_data = read_tfrecord_to_batch(train_record, 
                              FLAGS.num_epochs, FLAGS.batch_size)
  test_data = read_tfrecord_to_batch(test_record, 
                              FLAGS.num_epochs, 2717, shuffle=False)

  return train_data, test_data, word_embed

def write_results(predictions, relations_file, results_file):
  relations = []
  with open(relations_file) as f:
    for line in f:
      segment = line.strip().split()
      relations.append(segment[1])
  
  start_no = 8001
  with open(results_file, 'w') as f:
    for idx, id in enumerate(predictions):
      rel = relations[id]
      f.write('%d\t%s\n' % (start_no+idx, rel))
