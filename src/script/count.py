import reader
import collections

train_file = "data/train_nopos_ty=6.txt"
test_file = "data/test_nopos_ty=6.txt"

def count(file):
  '''count relation and length
  '''
  data = reader.load_raw_data(file)
  rel_counts = collections.Counter()
  len_counts = collections.Counter()
  max_len = 0

  for item in data:
    rel = item[0]
    rel_counts[rel] += 1
    n = len(item[5:])
    len_counts[n//10] += 1
    if max_len < n:
      max_len = n
  

  print("relation")
  for r in sorted(rel_counts.keys()):
    print("(%d, %d)" % (r, rel_counts[r]),end=' ')
  print()
  
  print("max_len", max_len)
  print("len/10:")
  print(len_counts.most_common())

print("train:")
count(train_file)
print("\n\ntest:")
count(test_file)

# train:
# relation
# (0, 471) (1, 1410) (2, 407) (3, 78) (4, 659) (5, 844) (6, 374) (7, 490) (8, 394) (9, 612) (10, 568) (11, 344) (12, 470) (13, 144) (14, 323) (15, 148) (16, 166) (17, 97) (18, 1) 
# len/10:
# [(1, 3936), (2, 2645), (0, 648), (3, 591), (4, 133), (5, 30), (6, 8), (7, 6), (9, 2), (8, 1)]


# test:
# relation
# (0, 150) (1, 454) (2, 134) (3, 32) (4, 194) (5, 291) (6, 153) (7, 210) (8, 123) (9, 201) (10, 211) (11, 134) (12, 162) (13, 51) (14, 108) (15, 47) (16, 39) (17, 22) (18, 1) 
# len/10:
# [(1, 1348), (2, 894), (0, 219), (3, 187), (4, 50), (5, 12), (6, 7)]
