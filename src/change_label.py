rel_2_id = dict()
with open('data/relations.txt') as f:
  for line in f:
    segment = line.strip().split()
    id = int(segment[0])
    rel_2_id[segment[1]] = id

sorted_rel = sorted(rel_2_id.keys())
sorted_rel.remove('Other')
sorted_rel.append('Other')
id_2_new_id = dict()

with open('data/relations_new.txt', 'w') as f:
  for new_id, rel in enumerate(sorted_rel):
    id = rel_2_id[rel]
    id_2_new_id[id] = new_id
    f.write('%d %s\n' %(new_id, rel))
  


def change_label(file, new_file):
  with open(file) as f_read:
    with open(new_file, 'w') as f_write:
      for line in f_read:
        segment = line.strip().split(' ')
        id = int(segment[0])
        new_id = id_2_new_id[id]
        f_write.write('%d %s\n' %(new_id, ' '.join(segment[1:])))

change_label('data/train_nopos_ty=6.txt', 'data/train.txt')
change_label('data/test_nopos_ty=6.txt', 'data/test.txt')
