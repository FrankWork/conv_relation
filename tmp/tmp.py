def foo():
  return 101

a = foo()
with open('/dev/null'):
  print(a)
