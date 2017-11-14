import os
import re
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def load_raw_data(filename, max_len=None):
  '''load raw data from text file, 
  and convert word to lower case, 
  and replace digits with 0s;
  if length of a sentence is large than max_len, truncate it

  return: a list of list [relation_id, s_e1, e_e1, s_e2, e_e2, w1, w2, ..., wT]
  '''
  data = []
  with open(filename) as f:
    for line in f:
      example = []
      words = line.strip().lower().split()
      
      if max_len:
        sent = words[5: max_len]
      else:
        sent = words[5:]

      for w in words[:5]:
        example.append(int(w))
      for w in sent:
        w = re.sub(r"[0-9]", "0", w)
        example.append(w)
      data.append(example)
  return data

def build_vocab(raw_train_data, raw_test_data, vocab_file):
  '''collect words in sentence'''
  PAD_WORD = "<pad>"
  START_WORD = "<s>"
  END_WORD = "<e>"
  if not os.path.exists(vocab_file):
    # load words in data
    words_in_data = set()
    for example in raw_train_data:
      words_in_data.update(example[5:])

    for example in raw_test_data:
      words_in_data.update(example[5:])
    
    # write vocab
    with open(vocab_file, 'w') as f:
      f.write(PAD_WORD+'\n')
      f.write(START_WORD+'\n')
      f.write(END_WORD+'\n')
      for w in sorted(list(words_in_data)):
        f.write(w + '\n')
  
  word2id = {}
  id2word = {}
  with open(vocab_file) as f:
    for i, line in enumerate(f):
      w = line.strip()
      word2id[w] = i
      id2word[id] = w

  return word2id, id2word

def gen_embeddings(word2id, word_embed_orig, word_embed_trim):
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
  FLAGS.word_dim = word_embed.shape[1]
  zeros = np.zeros((FLAGS.word_dim), dtype=np.float32)
  n_zero = 0
  for i in range(word_embed.shape[0]):
    if np.array_equal(word_embed[i], zeros):
      n_zero += 1
  print("%d UNKs." % n_zero)
  return word_embed

def _format_data(raw_data, word2id, max_len):
  '''format data used in neural nets
  '''
  PAD_ID, START_ID, END_ID = 0,1,2
  NO_RID=18

  data = []
  for item in raw_data:
    # FIXME: entity is represented by the last word
    rid, e1_idx, e2_idx = item[0], item[2]+1, item[4]+1

    sentence = [START_ID]
    sentence.extend([word2id[w] for w in item[5:]])
    sentence.append(END_ID)
    sentence.extend([PAD_ID]*(max_len - len(sentence)))

    # ignore WordNet hypernyms in paper
    lexical = []
    if e1_idx < max_len:
      lexical.extend([sentence[e1_idx-1], sentence[e1_idx], sentence[e1_idx+1] ])
    else:
      lexical.extend([PAD_ID,PAD_ID,PAD_ID])

    if e2_idx+1 < max_len:
      lexical.extend([sentence[e2_idx-1], sentence[e2_idx], sentence[e2_idx+1] ])
    else:
      lexical.extend([PAD_ID,PAD_ID,PAD_ID])

    def _position_feature(n):
      '''
      position feature used in cnn, aka relative distance
      '''
      # FIXME: FLAGS.pos_num
      if n < -60:
        return 0
      if n >= -60 and n <= 60:
        return n + 61
      if n > 60:
        return 122
      
    position1 = []
    for i in range(max_len):
      position1.append(_position_feature(i-e1_idx))
    position2 = []
    for i in range(max_len):
      position2.append(_position_feature(i-e2_idx))

    data.append((sentence, position1, position2, lexical, rid))
  return np.array(data)

def gen_batch_data(raw_data, word2id, max_len, num_epoches, batch_size, shuffle=True):
  '''generate a batch iterator
  '''
  data = _format_data(raw_data, word2id, max_len)

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
      for item in shuffled_data[start:end]:
        sentence, position1, position2, lexical, rid = item

        batch_data['sent_id'].append(sentence)
        batch_data['pos1_id'].append(position1)
        batch_data['pos2_id'].append(position2)
        batch_data['lexical_id'].append(lexical)
        batch_data['rid'].append(rid)
      yield batch_data



