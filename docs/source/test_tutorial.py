
print("**Tutorial**")


print("**Sample datasets**")


print("**Import**")

import cf
cf.CF()

print("**Field construct**")


print("**Reading field constructs from datasets**")

x = cf.read('file.nc')
type(x)
len(x)
y = cf.read('*.nc')
len(y)
z = cf.read(['file.nc', 'precipitation_flux.nc'])
len(z)
y = cf.read('$PWD')
y = cf.read('$PWD', ignore_read_error=True)
len(y)

print("**Inspection**")

x
q = x[0]
t = x[1]
q
print(q)
print(t)
q.dump()
t.dump()

print("**Visualization**")


print("**Field lists**")

x = cf.read('file.nc')
y = cf.read('precipitation_flux.nc')
x
y                                       
y.extend(x)                                       
y
y[2]
y[::-1]
len(y)
len(y + y)
len(y * 4)
for f in y:
     print('field:', repr(f))

print("**Properties**")

t.properties()
t.has_property('standard_name')
t.get_property('standard_name')
t.del_property('standard_name')
t.get_property('standard_name', default='not set')
t.set_property('standard_name', value='air_temperature')
t.get_property('standard_name', default='not set')
original = t.properties()
original
t.set_properties({'foo': 'bar', 'units': 'K'})
t.properties()
t.clear_properties()
t.properties()
t.set_properties(original)
t.properties()
t.identity()
t.identities()

print("**Metadata constructs**")

t.coordinate_references
print(t.coordinate_references)
list(t.coordinate_references.keys())
for key, value in t.coordinate_references.items():
     print(key, repr(value))

print(t.dimension_coordinates)
print(t.domain_axes)
q.constructs
print(q.constructs)
t.constructs
print(t.constructs)

print("**Data**")

t.data
print(t.array)
t.dtype
t.ndim
t.shape
t.size
print(t.domain_axes)
t
t.data.shape
t.get_data_axes()
data = t.del_data()
t.has_data()
t.set_data(data)
t.data
d = cf.Data([1, 2, 3], units='days since 2004-2-28')
print(d.array)   
print(d.datetime_array)
e = cf.Data([1, 2, 3], units='days since 2004-2-28', calendar='360_day')
print(d.array)   
print(d.datetime_array)
date_time = cf.dt(2004, 2, 29)
date_time
d = cf.Data(date_time, calendar='gregorian')
print(d.array)   
d.datetime_array
date_times  = cf.dt_vector(['2004-02-29', '2004-02-30', '2004-03-01'], calendar='360_day')
print (date_times)
e = cf.Data(date_times)
print(e.array)   
print(e.datetime_array)
d = cf.Data(['2004-02-29', '2004-02-30', '2004-03-01'], calendar='360_day')
d.Units
print(d.array)
print(d.datetime_array)
e = cf.Data(['2004-02-29', '2004-03-01', '2004-03-02'], dt=True)
e.Units
print(e.datetime_array)
f = cf.Data(['2004-02-29', '2004-03-01', '2004-03-02'])
print(f.array)
f.Units
## print(f.datetime_array)
t
t2 = t.squeeze()
t2
print(t2.dimension_coordinates)
t3 = t2.insert_dimension(axis='domainaxis3', position=1)
t3
t3.transpose([2, 0, 1])
t4 = t.transpose([2, 0, 1], constructs=True)

print("**Subspacing by index**")

print(q)
new = q[::-1, 0]
print(new)
q
t[:, :, 1]
t[:, 0]
t[..., 6:3:-1, 3:6]
t[0, [2, 3, 9], [4, 8]]
t[0, :, -2]
q
q.cyclic()
q.constructs.domain_axis_identity('domainaxis1')
print(q[:, -2:3])                                           
print(q[:, 3:-2:-1])
t.data[0, [2, 3, 9], [4, 8]]

print("**Assignment by index**")

t[:, 0, 0] = -1
t[:, :, 1] = -2
t[..., 6:3:-1, 3:6] = -3
print(t.array)
import numpy
t[..., 6:3:-1, 3:6] = numpy.arange(9).reshape(3, 3)
t[0, [2, 9], [4, 8]] =  cf.Data([[-4, -5]])
t[0, [4, 7], 0] = [[-10], [-11]]
print(t.array)
print(t[0, 0, -1].array)
t[0, -1, -1] /= -10
print(t[0, 0, -1].array)
t.data[0, 0, -1] = -99
print(t[0, 0, -1].array)
t[0, :, -2] = cf.masked
print(t.array)
t[0, 4, -2] = 99
print(t[0, 4, -2].array)
t.hardmask = False
t[0, 4, -2] = 99
print(t[0, 4, -2].array)
original = t.copy()
u = t.squeeze(0)
u.transpose(inplace=True)
u.flip(inplace=True)   
t[...] = u
original.allclose(t)
t[:, :, 1:3] = u[2]
print(t[:, :, 1:3].array)
print(u[2].array)	     
t[:, :, 1:3] = u[2]
print(t[:, :, 1:3].array)

print("**Units**")

t.units
t.Units
t.units = 'degreesC'
t.units
t.Units
t.Units += 273.15
t.units
t.Units
t.data
t.Units = cf.Units('degreesC')
t.data
t.units = 'Kelvin'
t.data
t.data
t[0, 0, 0] = cf.Data(1)
t.data
t[0, 0, 0] = cf.Data(1, 'degreesC')
t.data
tas = cf.read('air_temperature.nc')[0]
time = tas.coordinate('time')
time.units
time.calendar
time.Units

print("**Filtering metadata constructs**")

print(t.constructs.filter_by_type('dimension_coordinate'))
print(t.constructs.filter_by_type('cell_method', 'field_ancillary'))
print(t.constructs.filter_by_property(
             standard_name='air_temperature standard_error'))
print(t.constructs.filter_by_property(
             standard_name='air_temperature standard_error',
             units='K'))
print(t.constructs.filter_by_property(
             'or',
	           standard_name='air_temperature standard_error',
             units='m'))
print(t.constructs.filter_by_axis('and', 'domainaxis1'))
print(t.constructs.filter_by_measure('area'))
print(t.constructs.filter_by_method('maximum'))
print(t.constructs.filter_by_type('auxiliary_coordinate').filter_by_axis('and', 'domainaxis2'))
c = t.constructs.filter_by_type('dimension_coordinate')
d = c.filter_by_property(units='degrees')
print(d)
print(t)
print(t.constructs.filter_by_identity('X'))
print(t.constructs.filter_by_identity('latitude'))
print(t.constructs.filter_by_identity('long_name=Grid latitude name'))
print(t.constructs.filter_by_identity('measure:area'))
print(t.constructs.filter_by_identity('ncvar%b'))
print(t.constructs.filter_by_identity('latitude'))
print(t.constructs('latitude'))
print(t.constructs.filter_by_key('domainancillary2'))
print(t.constructs.filter_by_key('cellmethod1'))
print(t.constructs.filter_by_key('auxiliarycoordinate2', 'cellmeasure0'))
c = t.constructs('radiation_wavelength')
c
print(c)
len(c)
c = t.constructs.filter_by_type('auxiliary_coordinate')
c
c.inverse_filter()
print(t.constructs.filter_by_type('cell_measure'))
print(t.cell_measures)

print("**Metadata construct access**")

t.construct('latitude')
t.construct('latitude', key=True)
key = t.construct_key('latitude')
t.get_construct(key)
t.constructs('latitude').value()
c = t.constructs.get(key)
t.constructs[key]
t.auxiliary_coordinate('latitude')
t.auxiliary_coordinate('latitude', key=True)
## t.construct('measure:volume')
t.construct('measure:volume', default=False)
c = t.constructs.filter_by_measure('volume')
len(c)
## c.value()
c.value(default='No construct')
## c.value(default=KeyError('My message'))
d = t.constructs('units=degrees')
len(d)
## d.value()
print(d.value(default=None))
lon = q.construct('longitude')
lon
lon.set_property('long_name', 'Longitude')
lon.properties()
area = t.constructs.filter_by_property(units='km2').value()
area
area.identity()
area.identities()
lon = q.constructs('longitude').value()
lon
lon.data
lon.data[2]
lon.data[2] = 133.33
print(lon.array)
lon.data[2] = 112.5
key = t.construct_key('latitude')
key
t.get_data_axes(key)
t.constructs.data_axes()

print("**Time**")

time = q.construct('time')
time
time.get_property('units')
time.get_property('calendar', default='standard')
print(time.array)
print(time.datetime_array)
cm = cf.TimeDuration(1, 'calendar_month', day=16, hour=12)              
cm
cf.dt(2000, 2, 1) + cm                                                  
cf.Data([1, 2, 3], 'days since 2000-02-01') + cm                       
cm.interval(cf.dt(2000, 2, 1))                                         
cm.bounds(cf.dt(2000, 2, 1))

print("**Domain**")

domain = t.domain
domain
print(domain)
description = domain.dump(display=False)
domain_latitude = t.domain.constructs('latitude').value()
field_latitude = t.constructs('latitude').value()
domain_latitude.set_property('test', 'set by domain')
print(field_latitude.get_property('test'))
field_latitude.set_property('test', 'set by field')
print(domain_latitude.get_property('test'))
domain_latitude.del_property('test')
field_latitude.has_property('test')

print("**Metadata construct types**")

print(q.domain_axes)
d = q.domain_axes.get('domainaxis1')
d
d.get_size()
print(t.coordinates)
lon = t.constructs('grid_longitude').value()
bounds = lon.bounds
bounds
bounds.data
print(bounds.array)
bounds.inherited_properties()
bounds.properties()
a = t.constructs.get('domainancillary0')
print(a.array)
bounds = a.bounds
bounds
print(bounds.array)
crs = t.constructs('standard_name:atmosphere_hybrid_height_coordinate').value()
crs
crs.dump()
crs.coordinates()
crs.datum
crs.datum.parameters()
crs.coordinate_conversion
crs.coordinate_conversion.parameters()
crs.coordinate_conversion.domain_ancillaries()
print(t.cell_methods)
t.cell_methods.ordered()
cm = t.constructs('method:mean').value()
cm
cm.get_axes()
cm.get_method()
cm.qualifiers()
cm.get_qualifier('where')
a = t.get_construct('fieldancillary0')
a
a.properties()
a.data

print("**Cyclic domain axes**")

print(q.array[0])
print(q.roll('X', shift=1).array[0])
qr = q.roll('X', shift=-3)
print(qr.array[0])
print(q.dimension_coordinate('X').array)
print(qr.dimension_coordinate('X').array)
print(q.anchor('X', -150))                         
print(q.anchor('X', -750))

print("**Subspacing by metadata**")

print(q)
print(q.construct('X').array)
q2 = q.subspace(X=112.5)
print(q2)
print(q.construct('latitude').array)
print(q.subspace(X=112.5, latitude=cf.gt(-60)))
c = cf.eq(-45) | cf.ge(20)
c
print(q.subspace(latitude=c))
print(q.subspace(X=[1, 2, 4], Y=slice(None, None, -1)))
print(q.subspace(X=cf.wi(-100, 200)))
print (q.subspace(X=slice(-2, 4)))
print(f.coordinate('T').array) #TODO
print(f.coordinate('T').datetime_array) #TODO
print(q.subspace(T=TODO (float)))
print(q.subspace(T=cf.dt('2019-01-01')))
print(TODO.subspace(T=cf.wi(cf.dt('0450-11-01', calendar='noleap'), cf.dt('0451-03-01', calendar='noleap'))))
print(q.array)
q2 = q.subspace('compress', X=[1, 2, 4, 6])
print(q2)
print(q2.array)
q2 = q.subspace('envelope', X=[1, 2, 4, 6])
print(q2)
print(q2.array)
q2 = q.subspace('full', X=[1, 2, 4, 6])
print(q2)
print(q2.array)
print(t)
print(t.construct('latitude').array)
t2 = t.subspace(latitude=cf.wi(51, 53))
print(t2.array)

print("**Filtering and sorting field lists**")

fl = cf.read('*.nc')
fl
fl.filter_by_identity('precipitation_flux')
fl.filter_by_identity(re.compile('.*potential.*'))
fl.filter_by_identity('relative_humidity')
fl('air_temperature')
fl = cf.read('file.nc')                                                                        
fl
fl.sort()                                                                                      
fl
fl.sort(key=lambda f: f.units)                                                                 
fl
print(t)
t.match_by_identity('air_temperature')
t.match_by_rank(4)
t.match_by_units('degC', exact=False)
t.match_by_construct(longitude=cf.wi(-10, 10))
t.match('specific_humidity')
t.match('specific_humidity', 'air_temperature')

print("**Encapsulating conditions**")

c = cf.Query('lt', 3)
c
c.evaluate(2)
c == 2
c != 2
c.evaluate(3)
c == cf.Data([1, 2, 3])
c == numpy.array([1, 2, 3])
ge3 = cf.Query('ge', 3)
lt5 = cf.Query('lt', 5)
c = ge3 & lt5
c 
c == 2
c != 2
c = ge3 | lt5
c
c == 2
c &= cf.Query('set', [1, 3, 5])
c
c == 2
c == 3
upper_bounds_ge_minus4 = cf.Query('ge', -4, attr='upper_bounds')
x = t.dimension_coordinate('X')
x
print(x.bounds.array)
print((upper_bounds_ge_minus4 == x).array)
cf.ge(3)
cf.ge(cf.dt('2000-3-23'))
cf.year(1999)
cf.jja()
cf.contains(4)
cf.cellsize(cf.lt(10, 'degrees'))

print("**Assignment by condition**")

print(t.array)
u = t.where(cf.lt(273.15), x=cf.masked)
print(u.array)
u = t.where(cf.lt(273.15), x=0, y=1)
print(u.array)
print(t.where(u, x=-t, y=-99).array)
print(t.where(cf.gt(0.5), x=cf.masked, construct='Y').array)

print("**Field creation**")


print("**Stage 1:** The field construct is created without metadata")


print("**Stage 2:** Metadata constructs are created independently.")


print("**Stage 3:** The metadata constructs are inserted into the field")

p = cf.Field(properties={'standard_name': 'precipitation_flux'})
p
dc = cf.DimensionCoordinate(properties={'long_name': 'Longitude'},
                               data=cf.Data([0, 1, 2.]))
dc
fa = cf.FieldAncillary(
        properties={'standard_name': 'precipitation_flux status_flag'},
        data=cf.Data(numpy.array([0, 0, 2], dtype='int8')))
fa
p = cf.Field()
p
p.set_property('standard_name', 'precipitation_flux')
p
dc = cf.DimensionCoordinate()
dc
dc.set_property('long_name', 'Longitude')
dc.set_data(cf.Data([1, 2, 3.]))
dc
fa = cf.FieldAncillary(
             data=cf.Data(numpy.array([0, 0, 2], dtype='int8')))
fa
fa.set_property('standard_name', 'precipitation_flux status_flag')
fa
longitude_axis = p.set_construct(cf.DomainAxis(3))
longitude_axis
key = p.set_construct(dc, axes=longitude_axis)
key
cm = cf.CellMethod(axes=longitude_axis, method='minimum')
p.set_construct(cm)
Q.dump()
print(tas)
import netCDF4
nc = netCDF4.Dataset('file.nc', 'r')
v = nc.variables['ta']
netcdf_array = cf.NetCDFArray(filename='file.nc', ncvar='ta',
	                               dtype=v.dtype, ndim=v.ndim,
	     		  	       shape=v.shape, size=v.size)
data_disk = cf.Data(netcdf_array)
numpy_array = v[...]
data_memory = cf.Data(numpy_array)
data_disk.equals(data_memory)
key = tas.construct_key('surface_altitude')
orog = tas.convert(key)
print(orog)
orog1 = tas.convert(key, full_domain=False) 
print(orog1)
cf.write(tas, 'tas.nc')
f = cf.read('tas.nc')
f
fields = cf.read('tas.nc', extra='domain_ancillary')
fields
orog_from_file = fields[3]
print(orog_from_file)

print("**Copying**")

u = t.copy()
u.data[0, 0, 0] = -1e30
u.data[0, 0, 0]
t.data[0, 0, 0]
u.del_construct('grid_latitude')
u.constructs('grid_latitude')
t.constructs('grid_latitude')
import copy
u = copy.deepcopy(t)
orog = t.constructs('surface_altitude').value().copy()

print("**Equality**")

t.equals(t)
t.equals(t.copy())
t.equals(t[...])
t.equals(q)
t.equals(q, verbose=True)
cf.ATOL()
cf.RTOL()
original = cf.RTOL(0.00001)
cf.RTOL()
cf.RTOL(original)
cf.RTOL()
orog = t.constructs('surface_altitude').value()
orog.equals(orog.copy())

print("**NetCDF interface**")

print(t.constructs.filter_by_ncvar('b'))
t.constructs('ncvar%x').value()
t.constructs('ncdim%x')
q.nc_get_variable()
q.nc_global_attributes()
q.nc_unlimited_dimensions()
q.nc_set_variable('humidity')
q.nc_get_variable()
q.constructs('latitude').value().nc_get_variable()

print("**Writing to disk**")

print(q)
cf.write(q, 'q_file.nc')
x
cf.write(x, 'new_file.nc')
f = cf.read('q_file.nc')[0]
q.equals(f)
f.set_property('model', 'model_A')
cf.write(f, 'f_file.nc', global_attributes='model')
f.nc_global_attributes()
f.nc_set_global_attribute('model')
f.nc_global_attributes()
cf.write(f, 'f_file.nc')
f.set_property('information', 'variable information')
f.properties()
f.nc_set_global_attribute('information', 'global information')
f.nc_global_attributes()
cf.write(f, 'f_file.nc')
cf.write(f, file_descriptors={'history': 'created in 2019'})
f_file = cf.read('f_file')[0]
f_file.nc_global_attributes()
f_file.properties()
f_file.nc_global_attributes()
f_file.set_property('Conventions', 'UGRID1.0')
cf.write(f, 'f_file.nc', Conventions='UGRID1.0')   
print(q)
key = q.construct_key('time')
axes = q.get_data_axes(key)
axes
q2 = q.insert_dimension(axis=axes[0])
q2
cf.write(q2, 'q2_file.nc')

print("**External variables**")

u = cf.read('parent.nc')[0]
print(u)
area = u.constructs('measure:area').value()
area
area.nc_get_external()
area.nc_get_variable()
area.properties()
area.has_data()
g = cf.read('parent.nc', external='external.nc')[0]
print(g)
area = u.constructs('measure:area').value()
area
area.nc_get_external()
area.nc_get_variable()
area.properties()
area.data
area.nc_set_external(True)
cf.write(g, 'new_parent.nc')
cf.write(g, 'new_parent.nc', external='new_external.nc')

print("**Statistical collapses**")

a = cf.read('timeseries.nc')[0]
print(a)
b = a.collapse('minimum')
print(b)
print(b.array)
b = a.collapse('maximum', axes='T')
b = a.collapse('T: maximum')
print(b)
print(b.array)
b = a.collapse('maximum', axes=['X', 'Y'])
b = a.collapse('X: Y: maximum')
print(b)
b = a.collapse('area: maximum')
print(b)
b = a.collapse('T: mean', weights='T')
print(b)
print (b.array)
w = a.weights(weights='T')
print(w)
print(w.array)
b = a.collapse('T: Y: mean', weights='Y')
print(b)
print (b.array)
b = a.collapse('area: mean', weights='area')
print(b)
b = a.collapse('area: mean', weights='area').collapse('T: maximum')
print(b)
print(b.array)
b = a.collapse('area: mean T: maximum', weights='area')
print(b.array)
y = cf.Y(month=12)
y
b = a.collapse('T: maximum', group=y)
print(b)
b = a.collapse('T: maximum', group=6)
print(b)
b = a.collapse('T: maximum', group=cf.djf())
print(b)
c = cf.seasons()
c
b = a.collapse('T: maximum', group=c)
print(b)
b = a.collapse('X: mean', group=cf.Data(180, 'degrees'))
print(b)
b = a.collapse('T: mean within years T: mean over years',
                within_years=cf.seasons(), weights='T')
print(b)
print(b.coordinate('T').bounds.datetime_array)
b = a.collapse('T: minimum within years T: variance over years',
                within_years=cf.seasons(), weights='T')
print(b)
print(b.coordinate('T').bounds.datetime_array)
b = a.collapse('T: mean within years T: mean over years', weights='T',
                within_years=cf.seasons(), over_years=cf.Y(5))
print(b)
print(b.coordinate('T').bounds.datetime_array)
b = a.collapse('T: mean within years T: mean over years', weights='T',
                within_years=cf.seasons(), over_years=cf.year(cf.wi(1963, 1968)))
print(b)
print(b.coordinate('T').bounds.datetime_array)
b = a.collapse('T: standard_deviation within years',
                within_years=cf.seasons(), weights='T')
print(b)
c = b.collapse('T: maximum over years')
print(c)

print("**Regridding**")

a = cf.read('air_temperature.nc')[0]
b = cf.read('precipitation_flux.nc')[0]
print(a)
print(b)
c = a.regrids(b, 'conservative')
print(c)
import numpy
lat = cf.DimensionCoordinate(data=cf.Data(numpy.arange(-90, 92.5, 2.5), 'degrees_north'))
lon = cf.DimensionCoordinate(data=cf.Data(numpy.arange(0, 360, 5.0), 'degrees_east'))
c = a.regrids({'latitude': lat, 'longitude': lon}, 'bilinear')
time = cf.DimensionCoordinate()
time.standard_name='time'
time.set_data(cf.Data(numpy.arange(0.5, 60, 1),
                       units='days since 1860-01-01', calendar='360_day'))
time
c = a.regridc({'T': time}, axes='T', method='bilinear')
c = a.regridc({'T': time}, axes='T', method='conservative')
bounds = d.create_bounds()
d.set_bounds(bounds)
c = a.regridc({'T': d}, axes='T', method='conservative')
print(c)
# c = a.regridc(TODO (field arg), axes='T', method='conservative')
print(c)
v = cf.read('vertical.nc')[0]
print(v)
z_p = v.construct('Z')
print(z_p.array)
z_ln_p = z_p.log()
print(z_ln_p.array)
_ = v.replace_construct('Z', z_ln_p)
new_z_p = cf.DimensionCoordinate(data=cf.Data([800, 705, 632, 510, 320.], 'hPa'))
new_z_ln_p = new_z_p.log()
new_v = v.regridc({'Z': new_z_ln_p}, axes='Z', method='bilinear') 
new_v.replace_construct('Z', new_z_p)
print(new_v)

print("**Mathematical operations**")

lat = q.dimension_coordinate('latitude')
lat.data
sin_lat = lat.sin()                                                                                 
sin_lat.data                                                                                        
q
q.log()
q.exp()
t   
t.log(base=10)
t.exp()
print(r)
r.iscyclic('X')
r = q.convolution_filter([0.1, 0.15, 0.5, 0.15, 0.1], axis='X')
print(r)                                                                                          
print(q.dimension_coordinate('X').bounds.array)
print(r.dimension_coordinate('X').bounds.array)
from scipy.signal import windows
exponential_weights = windows.exponential(3)                                                      
print(exponential_weights)
r = q.convolution_filter(exponential_weights, axis='Y')                                           
print(r.array)
r = q.derivative('X')
r = q.derivative('Y', one_sided_at_boundary=True)
u, v = cf.read('wind_components.nc')
zeta = cf.relative_vorticity(u, v)
print(zeta)
print(zeta.array.round(8))

print("**Aggregation**")

a = cf.read(TODO)
b = cf.read(TODO, aggregate=False)
c = cf.aggregate(b)
a.equals(c)
#WWW = cf.read(TODO, aggregate={'info': 1, 'overlap': False})
#XXX = cf.aggregate(AAA TODO, info=1, overlap=False)
#WWW.equals(XXX TODO)

print("**Compression**")

h = cf.read('contiguous.nc')[0]
print(h)
print(h.array)
h.data.get_compression_type()
print(h.data.compressed_array)
count_variable = h.data.get_count()
count_variable
print(count_variable.array)
station2 = h[1]
station2
print(station2.array)
h.data.get_compression_type()
h.data[1, 2] = -9
print(h.array)
h.data.get_compression_type()
T
print(T.array)
T.data.get_compression_type()
print(T.data.compressed_array)
count_variable = T.data.get_count()
count_variable
print(count_variable.array)
cf.write(T, 'T_contiguous.nc')
p = cf.read('gathered.nc')[0]
print(p)
print(p.array)
p.data.get_compression_type()
print(p.data.compressed_array)
list_variable = p.data.get_list()
list_variable
print(list_variable.array)
p[0]
p[1, :, 3:5]
p.data.get_compression_type()
p.data[1] = -9
p.data.get_compression_type()
P
print(P.data.rray)
P.data.get_compression_type()
print(P.data.compressed_array)
list_variable = P.data.get_list()
list_variable 
print(list_variable.array)
cf.write(P, 'P_gathered.nc')

print("**PP and UM fields files**")

#TODO read PP file
pp = cf.read(umfile.pp)
cf.write(pp, 'umfile1.nc')
type(cf.read_write.um.umread.stash2standard_name)                       
cf.read_write.um.umread.stash2standard_name[(1, 4)]                    
cf.read_write.um.umread.stash2standard_name[(1, 2)]
cf.read_write.um.umread.stash2standard_name[(1, 7)]                    
(1, 999) in cf.read_write.um.umread.stash2standard_name
with open('new_STASH.txt', 'w') as new:  
     new.write('1!999!My STASH code!1!!!ultraviolet_index!!') 
 
_ = cf.load_stash2standard_name('new_STASH.txt', merge=True)
cf.read_write.um.umread.stash2standard_name[(1, 999)]
