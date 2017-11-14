
MTL_Label = namedtuple('MTL_Label', 'relation direction')


def load_mtl_label(relations_file):
  label2segment = dict() # {0:  [Component-Whole, (e2,e1)], ... }
  relation_set = set()   # (Component-Whole, ...)
  with open(relations_file) as f:
    for line in f:
      label, relation_str = line.strip().split()
      label = int(label)
      segment = relation_str.split(':')
      label2segment[label] = segment
      relation_set.add(segment[0])
  relation_set = sorted(list(relation_set))
  relation2id = dict() # {Cause-Effect: 0, ...}
  for i, rel in enumerate(relation_set):
    relation2id[rel] = i
  
  # {label: (relation, direction)}
  # label is relation with direction
  label2mtl = dict() 
  for label, segment in label2segment.items():
    relation = relation2id[segment[0]]
    if len(segment)==2 and segment[1]=='(e2,e1)':
      direction = 1
    else:
      direction = 0 # 'Other' relation has no direction
    label2mtl[label] = MTL_Label(relation, direction)

  return label2mtl

def load_raw_data(filename, label2mtl=None, max_len=None):
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
      if label2mtl:
        label = label2mtl[label]

      entity1 = PositionPair(int(words[1]), int(words[2]))
      entity2 = PositionPair(int(words[3]), int(words[4]))

      example = Raw_Example(label, entity1, entity2, sent)
      data.append(example)
  return data

def _format_data(raw_data, word2id, max_len):
  '''format data used in neural nets
  '''
  PAD_ID, START_ID, END_ID = 0,1,2
  UNK_ID = len(word2id)-1

  data = []
  for example in raw_data:
    sentence = [START_ID]
    sentence.extend([word2id[w] for w in example.sentence])
    sentence.append(END_ID)
    sentence.extend([PAD_ID]*(max_len - len(sentence)))

    # FIXME: entity is represented by the last word
    e1_idx = example.entity1.last + 1 # +1 for START_ID
    e2_idx = example.entity2.last + 1

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
  
    position1 = []
    for i in range(max_len):
      position1.append(_position_feature(i-e1_idx))
    position2 = []
    for i in range(max_len):
      position2.append(_position_feature(i-e2_idx))

    if FLAGS.mode == 'mtl':
      rid = example.label.relation
      direction = example.label.direction
      data.append((sentence, position1, position2, lexical, rid, direction))
    else:
      rid = example.label
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
      if FLAGS.mode=='mtl':
        batch_data['direction'] = []

      for item in shuffled_data[start:end]:
        if FLAGS.mode == 'mtl':
          sentence, position1, position2, lexical, rid, direction = item
        else:
          sentence, position1, position2, lexical, rid = item

        batch_data['sent_id'].append(sentence)
        batch_data['pos1_id'].append(position1)
        batch_data['pos2_id'].append(position2)
        batch_data['lexical_id'].append(lexical)
        batch_data['rid'].append(rid)
        if FLAGS.mode == 'mtl':
          batch_data['direction'].append(direction)
      yield batch_data





