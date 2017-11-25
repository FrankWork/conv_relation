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

def load_raw_data(filename, mtl_mode=False,  max_len=None):
  '''load raw data from text file, 
  and convert word to lower case, 
  and replace digits with 0s;
  if length of a sentence is large than max_len, truncate it

  return: a list of Raw_Example
  '''
  data = []
  with open(filename) as f:
    for line in f:
      # example = Raw_Example()
      words = line.strip().lower().split()
      
      if max_len:
        sent = words[5: max_len]
      else:
        sent = words[5:]
      sent = [re.sub("\d+", "0", w) for w in sent]

      label = int(words[0])
      if mtl_mode:
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
  '''
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

def write_tfrecords(raw_data, filename):
  '''convert the raw_data to tf.trian.SequenceExample and write to file
  Args:
    raw_data: a list of 'Raw_Example'
  '''
  writer = tf.python_io.TFRecordWriter(filename)
  for raw_example in raw_data:
    example = build_sequence_example(raw_example)
    writer.write(example.SerializeToString())
  writer.close()

def gen_batch_data(data, num_epoches, batch_size, shuffle=True):
  '''generate a batch iterator
  '''
  data_size = len(data)
  num_batches_per_epoch = int(np.ceil( len(data)/batch_size ))

  for _ in range(num_epoches):
    # Shuffle the data at each epoch
    if shuffle:
      indices = np.random.permutation(np.arange(data_size))
      shuffled_data = data[indices]
    else:
      shuffled_data = data

    for batch_num in range(num_batches_per_epoch):
      start = batch_num * batch_size
      end = min((batch_num + 1) * batch_size, data_size)
      batch_data = {
        'sent_id': [], 'pos1_id':[], 'pos2_id':[], 'lexical_id':[], 'rid':[]
      }
      if len(shuffled_data[0]) == 6:# mtl mode
        batch_data['direction'] = []

      for item in shuffled_data[start:end]:
        if len(item) == 6:# mtl mode
          sentence, position1, position2, lexical, rid, direction = item
          batch_data['direction'].append(direction)
        else:
          sentence, position1, position2, lexical, rid = item

        batch_data['sent_id'].append(sentence)
        batch_data['pos1_id'].append(position1)
        batch_data['pos2_id'].append(position2)
        batch_data['lexical_id'].append(lexical)
        batch_data['rid'].append(rid)
          
      yield batch_data


def inputs(mtl_mode=False):
  raw_train_data = load_raw_data(FLAGS.train_file, mtl_mode, FLAGS.max_len)
  raw_test_data = load_raw_data(FLAGS.test_file, mtl_mode, FLAGS.max_len)

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

  # TODO: finish the work below
  # filename is None
  write_tfrecords(raw_train_data, filename)
  write_tfrecords(raw_test_data, filename)

  print(raw_train_data[0].sentence)

  exit()

  FLAGS.max_len = FLAGS.max_len + 2 # append start and end word
  format_train_data = format_data(raw_train_data, word2id,  FLAGS.max_len)
  format_test_data = format_data(raw_test_data, word2id,  FLAGS.max_len)

  train_data = gen_batch_data(format_train_data,
                                FLAGS.num_epochs, 
                                FLAGS.batch_size)
  test_data = gen_batch_data(format_test_data, 1, 2717, shuffle=False)
  test_data = test_data.__next__()

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
