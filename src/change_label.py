relations = []
with open('data/relations.txt') as f:
  for line in f:
    segment = line.strip().split()
    relations.append(segment[1])

for i, r in enumerate(sorted(relations)):
  print(i, r)