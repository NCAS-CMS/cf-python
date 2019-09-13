import cf

f = cf.read('test_file.nc', squeeze=1)[0]
g = cf.read('test_file_python2.nc', squeeze=1)[0]
f.dump()
g.dump()

print(f.properties())
print(f.equals(g, verbose=2))
