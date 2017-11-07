import re
import numpy as np


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

def gen_embeddings(raw_train_data, raw_test_data, senna_embed_file, senna_words_file, 
                  word_emb_dim):
  '''collect words in sentence and find their corresponding pre-trained word embedding
  '''
  PAD_WORD = "<pad>"
  START_WORD = "<s>"
  END_WORD = "<e>"
  
  words_in_data = set()
  for example in raw_train_data:
    words_in_data.update(example[5:])

  for example in raw_test_data:
    words_in_data.update(example[5:])

  
  word2id = {PAD_WORD:0, START_WORD: 1, END_WORD: 2}
  id2word = {0:PAD_WORD, 1: START_WORD, 2: END_WORD}
  current_id = 3
  for w in sorted(list(words_in_data)):
    word2id[w] = current_id
    id2word[current_id] = w
    current_id += 1

  senna_words = []
  with open(senna_words_file) as f:
    for line in f:
      senna_words.append(line.strip())
  
  n_words = len(word2id)
  embed = [ [] for _ in range(n_words)]
  with open(senna_embed_file) as f:
    idx = 0
    for line in f:
      senna_w = senna_words[idx]
      idx += 1
      if senna_w in word2id:
        id = word2id[senna_w]
        embed[id] = [float(val) for val in line.strip().split()]
        assert len(embed[id]) == word_emb_dim
  for id in range(n_words):
    if embed[id] == []:
      embed[id] = [.0] * word_emb_dim
  
  return word2id, id2word, embed

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
