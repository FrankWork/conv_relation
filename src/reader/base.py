import os
import re
import numpy as np
import tensorflow as tf
from collections import defaultdict
from collections import namedtuple


PAD_WORD = "<pad>"
# START_WORD = "<start>"
# END_WORD = "<end>"
# PAD_ID, START_ID, END_ID = 0,1,2

Raw_Example = namedtuple('Raw_Example', 'label entity1 entity2 sentence')
PositionPair = namedtuple('PosPair', 'first last')
MTL_Label = namedtuple('MTL_Label', 'relation direction')

FLAGS = tf.app.flags.FLAGS # load FLAGS.word_dim

def clean_str(string):
    """
    String cleaning.
    From: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    >>> clean_str("I'll clean this (string)")
    "i 'll clean this ( string )"
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = re.sub(r"#", "", string)
    return string.strip().lower()

# FIXME clean_str changes the entity position
def load_raw_data(filename):
  '''load raw data from text file, 
  and convert word to lower case, 
  and replace digits with 0s;

  return: a list of Raw_Example
  '''
  data = []
  with open(filename) as f:
    for line in f:
      clr_line = clean_str(line)
      words = clr_line.split(' ')
      
      sent = words[5:]

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

def build_vocab(raw_train_data, raw_test_data):
  '''collect words in sentence'''
  vocab = set()
  for example in raw_train_data + raw_test_data:
    for w in example.sentence:
        vocab.add(w)
  return vocab

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
      word_embed[i] = np.random.normal(0, 0.1,(FLAGS.word_dim))
      n_zero += 1
  print("%d zero vectors." % n_zero)
  return word_embed

def gen_senna_embeddings(vocab, 
                        senna_words_embed_file,
                        senna_words_lst_file,
                        word_embed_trim_file,
                        vocab_file):
  '''trim unnecessary words from original pre-trained word embedding

  Args:
    vocab: a set of words that appear in the text
    senna_words_embed_file: file name of the original pre-trained embedding
    senna_words_lst_file: file name of the senna words list w.r.t the embed
    word_embed_trim_file: file name of the trimmed embedding
    vocab_file: filename of the saved vocab
  '''
  if not os.path.exists(word_embed_trim_file):
    senna_words=open(senna_words_lst_file,"r").readlines()
    senna_embed=open(senna_words_embed_file,"r").readlines()

    word_embed=[]
    vocab2id={}
    id2vocab = {}
    id = 0
    for senna_wid in range(len(senna_words)):
      word=senna_words[senna_wid].strip()
      if word in vocab:
        raw_embed = senna_embed[senna_wid].strip().split()
        embed = [float(x) for x in raw_embed]
        vocab2id[word] = id
        id2vocab[id] = word
        id += 1
        word_embed.append(embed)
    
    # generate missing embed
    n_miss=len(vocab)-len(vocab2id)
    miss_embed=np.random.normal(0,0.1,[n_miss, FLAGS.word_dim])
    word_embed=np.asarray(word_embed)
    word_embed=np.vstack((word_embed, miss_embed))
    words_left = vocab.difference(vocab2id.keys())
    for w in sorted(list(words_left)):
      vocab2id[w] = id
      id2vocab[id] = w
      id += 1

    # generate embed for PAD_WORD
    vocab2id[PAD_WORD]=id
    id2vocab[id] = PAD_WORD
    word_embed=np.vstack((word_embed,np.zeros([FLAGS.word_dim])))

    np.save(word_embed_trim_file, word_embed.astype(np.float32))
    with open(vocab_file, 'w') as vocab_f:
      for id in range(len(vocab2id)):
        w = id2vocab[id]
        vocab_f.write('%s\n' % w)

  word_embed = np.load(word_embed_trim_file)
  vocab2id = {}
  with open(vocab_file) as vocab_f:
    for id, line in enumerate(vocab_f):
      w = line.strip()
      vocab2id[w] = id
  return vocab2id, word_embed

def map_words_to_id(raw_data, word2id):
  '''inplace convert sentence from a list of words to a list of ids
  Args:
    raw_data: a list of Raw_Example
    word2id: dict, {word: id, ...}
  '''
  pad_id = word2id[PAD_WORD]
  for raw_example in raw_data:
    for idx, word in enumerate(raw_example.sentence):
      raw_example.sentence[idx] = word2id[word]

    # pad the sentence to FLAGS.max_len
    pad_n = FLAGS.max_len - len(raw_example.sentence)
    raw_example.sentence.extend(pad_n*[pad_id])

def _lexical_feature(raw_example):
  def _entity_context(e_idx, sent):
    ''' return [w(e-1), w(e), w(e+1)]
    '''
    context = []
    context.append(sent[e_idx])

    if e_idx >= 1:
      context.append(sent[e_idx-1])
    else:
      context.append(sent[e_idx])
    
    if e_idx < len(sent)-1:
      context.append(sent[e_idx+1])
    else:
      context.append(sent[e_idx])
    
    return context

    
  e1_idx = raw_example.entity1.first
  e2_idx = raw_example.entity2.first

  context1 = _entity_context(e1_idx, raw_example.sentence)
  context2 = _entity_context(e2_idx, raw_example.sentence)

  # ignore WordNet hypernyms in paper
  lexical = context1 + context2
  return lexical

def _position_feature(raw_example):
  def distance(n):
    '''convert relative distance to positive number
    -60), [-60, 60], (60
    '''
    # FIXME: FLAGS.pos_num
    if n < -60:
      return 0
    elif n >= -60 and n <= 60:
      return n + 61
    
    return 122

  e1_idx = raw_example.entity1.first
  e2_idx = raw_example.entity2.first

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
  with tf.device('/cpu:0'):
    dataset = tf.data.TFRecordDataset([filename])
    # Parse the record into tensors
    dataset = dataset.map(_parse_tfexample) 
    dataset = dataset.repeat(epoch)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=100)
    

    # [] for no padding, [None] for padding to maximum length
    # n = FLAGS.max_len
    # if FLAGS.model == 'mtl':
    #   # lexical, rid, direction, sentence, position1, position2
    #   padded_shapes = ([None,], [], [], [n], [n], [n])
    # else:
    #   # lexical, rid, sentence, position1, position2
    #   padded_shapes = ([None,], [], [n], [n], [n])
    # dataset = dataset.padded_batch(batch_size, padded_shapes)
    dataset = dataset.batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    return batch


def inputs():
  raw_train_data = load_raw_data(FLAGS.train_file)
  raw_test_data = load_raw_data(FLAGS.test_file)

  vocab = build_vocab(raw_train_data, raw_test_data)

  if FLAGS.word_dim == 50:
    vocab2id, word_embed = gen_senna_embeddings(vocab,
                              FLAGS.word_embed50_orig,
                              FLAGS.senna_words_lst,
                              FLAGS.word_embed50_trim,
                              FLAGS.vocab_file)
  elif FLAGS.word_dim == 300:
    word_embed = gen_google_embeddings(vocab2id,
                              FLAGS.word_embed300_orig, 
                              FLAGS.word_embed300_trim,
                              FLAGS.vocab_file)

  # map words to ids
  map_words_to_id(raw_train_data, vocab2id)
  map_words_to_id(raw_test_data, vocab2id)

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
                              FLAGS.num_epochs, FLAGS.batch_size, shuffle=True)
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
