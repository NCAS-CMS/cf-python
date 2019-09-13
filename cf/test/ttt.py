import cf
b=cf.read('../../docs/_downloads/precip*nc')[0]
a=cf.read('../../docs/_downloads/air_temper*nc')[0]
lat = b.dimension_coordinate('Y')
lon = b.dimension_coordinate('X')
c = a.regrids({'latitude': lat, 'longitude': lon}, 'bilinear')
print(c)
