import sys

ent = list(range(5895))
rel = list(range(19))

trip_dict = {}

for h in ent:
  trip_dict[h] = {}
  for r in rel:
    trip_dict[h][r] = ent

trip_list = [0] * len(ent)
for h in ent:
  trip_list[h] = [0] * len(rel)
  trip_list[h][r] = ent

s_dict = sys.getsizeof(trip_dict)
s_list = sys.getsizeof(trip_list)


def print_kb(bytes):
  kb = bytes / 1024
  print(kb)

print_kb(s_dict)
print_kb(s_list)

# 384.09375
# 46.1171875
