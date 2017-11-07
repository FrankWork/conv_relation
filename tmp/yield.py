def foo():
  for i in range(10):
    yield i

iter = foo()
for i in iter:
  print(i)