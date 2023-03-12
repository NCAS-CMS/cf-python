import cf
cf.CF()
x = cf.read('file.nc')
type(x)
len(x)
y = cf.read('*.nc')
len(y)
z = cf.read(['file.nc', 'precipitation_flux.nc'])
len(z)
try:
    y = cf.read('$PWD')  # Raises Exception
except Exception:
    pass
y = cf.read('$PWD', ignore_read_error=True)
len(y)
x = cf.read('file.nc')
x
q = x[0]
t = x[1]
q
print(q)
print(t)
q.dump()
t.dump()
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
q, t = cf.read('file.nc')
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
q, t = cf.read('file.nc')
t.coordinate_references
print(t.coordinate_references)
list(t.coordinate_references().keys())
for key, value in t.coordinate_references().items():
    print(key, repr(value))
print(t.dimension_coordinates)
print(t.domain_axes)
q.constructs
print(q.constructs)
t.constructs
print(t.constructs)
q, t = cf.read('file.nc')
t.data
a = t.array
type(a)
print(a)
t.dtype
t.ndim
t.shape
t.size
d = t.to_dask_array()
d
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
print(e.array)
print(e.datetime_array)
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
try:
    print(f.datetime_array)  # Raises Exception
except Exception:
    pass
q, t = cf.read('file.nc')
t
t2 = t.squeeze()
t2
print(t2.dimension_coordinates)
t3 = t2.insert_dimension(axis='domainaxis3', position=1)
t3
t3.transpose([2, 0, 1])
t4 = t.transpose(['X', 'Z', 'Y'], constructs=True)
print(q)
print(q.mask)
print(q.mask.array)
q[[0, 4], :] = cf.masked
print(q.mask.array)
q.mask.all()
q.mask.any()
cf.write(q, 'masked_q.nc')
no_mask_q = cf.read('masked_q.nc', mask=False)[0]
print(no_mask_q.array)
masked_q = no_mask_q.apply_masking()
print(masked_q.array)
q, t = cf.read('file.nc')
print(q)
new = q[::-1, 0]
print(new)
t
t[:, :, 1]
t[:, 0]
t[..., 6:3:-1, 3:6]
t[0, [2, 3, 9], [4, 8]]
t[0, :, -2]
t[..., [True, False, True, True, False, False, True, False, False]]
q
q.cyclic()
q.constructs.domain_axis_identity('domainaxis1')
print(q[:, -2:3])
print(q[:, 3:-2:-1])
t.data[0, [2, 3, 9], [4, 8]]
q, t = cf.read('file.nc')
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
t[0, 0, -1] /= -10
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
q, t = cf.read('file.nc')
t0 = t.copy()
u = t.squeeze(0)
u.transpose(inplace=True)
u.flip(inplace=True)
t[...] = u
t.allclose(t0)
print(t[:, :, 1:3].array)
print(u[2].array)
t[:, :, 1:3] = u[2]
print(t[:, :, 1:3].array)
q, t = cf.read('file.nc')
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
air_temp = cf.read('air_temperature.nc')[0]
time = air_temp.coordinate('time')
time.units
time.calendar
time.Units
q, t = cf.read('file.nc')
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
print(t.constructs.filter_by_axis('X', 'Y', axis_mode='or'))
print(t.constructs.filter_by_measure('area'))
print(t.constructs.filter_by_method('maximum'))
print(
    t.constructs.filter_by_type('auxiliary_coordinate').filter_by_axis('domainaxis2')
)
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
t.construct('latitude')
t.construct('latitude', key=True)
key = t.construct_key('latitude')
t.get_construct(key)
key, lat = t.construct_item('latitude')
t.constructs[key]
t.constructs.get(key)
t.auxiliary_coordinate('latitude')
t.auxiliary_coordinate('latitude', key=True)
t.auxiliary_coordinate('latitude', item=True)
try:
    t.construct('measure:volume')                # Raises Exception
except Exception:
    pass
t.construct('measure:volume', default=False)
try:
    t.construct('measure:volume', default=Exception("my error"))  # Raises Exception
except Exception:
    pass
c = t.constructs.filter_by_measure("volume")
len(c)
d = t.constructs("units=degrees")
len(d)
try:
    t.construct("units=degrees")  # Raises Exception
except Exception:
    pass
print(t.construct("units=degrees", default=None))
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
cf.D()
cf.Y(10, month=12)
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
print(q.domain_axes)
d = q.domain_axes().get('domainaxis1')
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
f = cf.read('geometry.nc')[0]
print(f)
lon = f.auxiliary_coordinate('X')
lon.dump()
lon.get_geometry()
print(lon.bounds.data.array)
print(lon.get_interior_ring().data.array)
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
f = cf.example_field(1)
print(f)
print(f.auxiliary_coordinate('altitude', default=None))
g = f.compute_vertical_coordinates()
g.auxiliary_coordinate('altitude').dump()
print(t.cell_methods())
t.cell_methods()
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
print(q.array[0])
print(q.roll('X', shift=1).array[0])
qr = q.roll('X', shift=-3)
print(qr.array[0])
print(q.dimension_coordinate('X').array)
print(qr.dimension_coordinate('X').array)
print(q.anchor('X', -150))
print(q.anchor('X', -750))
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
a = cf.read('timeseries.nc')[0]
print (a)
print(a.coordinate('T').array[0:9])
print(a.coordinate('T').datetime_array[0:9])
print(a.subspace(T=410.5))
print(a.subspace(T=cf.dt('1960-04-16')))
print(a.subspace(T=cf.wi(cf.dt('1962-11-01'), cf.dt('1967-03-17 07:30'))))
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
print(t2.construct('latitude').array)
print(t2.array)
q, t = cf.read('file.nc')
print(t)
indices = t.indices(grid_longitude=cf.wi(-4, -2))
indices
t[indices] = -11
print(t.array)
t[t.indices(latitude=cf.wi(51, 53))] = -99
print(t.array)
fl = cf.read('file.nc')
fl
fl.sort()
fl
fl.sort(key=lambda f: f.units)
fl
fl = cf.read('*.nc')
fl
fl.select_by_identity('precipitation_flux')
import re
fl.select_by_identity(re.compile('.*potential.*'))
fl.select_by_identity('relative_humidity')
fl('air_temperature')
fl.select('air_temperature')
print(t)
t.match_by_identity('air_temperature')
t.match_by_rank(4)
t.match_by_units('degC', exact=False)
t.match_by_construct(longitude=cf.wi(-10, 10))
t.match('specific_humidity')
t.match('specific_humidity', 'air_temperature')
c = cf.Query('lt', 3)
c
c.evaluate(2)
c == 2
c != 2
c.evaluate(3)
c == cf.Data([1, 2, 3])
print(c == numpy.array([1, 2, 3]))
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
X = t.dimension_coordinate('X')
X
print(X.bounds.array)
print((upper_bounds_ge_minus4 == X).array)
cf.contains(4)
cf.Query('lt', 4, attr='lower_bounds') &  cf.Query('ge', 4, attr='upper_bounds')
cf.ge(3)
cf.ge(cf.dt('2000-3-23'))
cf.year(1999)
cf.month(cf.wi(6, 8))
cf.jja()
cf.contains(4)
cf.cellsize(cf.lt(10, 'degrees'))
t = cf.read('file.nc')[1]
print(t.array)
u = t.where(cf.lt(273.15), x=cf.masked)
print(u.array)
u = t.where(cf.lt(273.15), x=0, y=1)
print(u.array)
print(t.where(u, x=-t, y=-99).array)
v = t.where(cf.gt(0.5), x=cf.masked, construct='grid_latitude')
print(v.array)
print(t.where(v.mask, x=cf.masked))
print(t.where(True, x=cf.masked).array)
print(t.where([0, 0, 1, 0, 1, 1, 1, 0, 0], x=cf.masked).array)
t.data.where(v.data.mask, x=cf.masked, inplace=True)
print(t.array)
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
# Start of code block
import numpy
import cf

# Initialise the field construct with properties
Q = cf.Field(properties={'project': 'research',
'standard_name': 'specific_humidity',
'units': '1'})

# Create the domain axis constructs
domain_axisT = cf.DomainAxis(1)
domain_axisY = cf.DomainAxis(5)
domain_axisX = cf.DomainAxis(8)

# Insert the domain axis constructs into the field. The
# set_construct method returns the domain axis construct key that
# will be used later to specify which domain axis corresponds to
# which dimension coordinate construct.
axisT = Q.set_construct(domain_axisT)
axisY = Q.set_construct(domain_axisY)
axisX = Q.set_construct(domain_axisX)

# Create and insert the field construct data
data = cf.Data(numpy.arange(40.).reshape(5, 8))
Q.set_data(data)

# Create the cell method constructs
cell_method1 = cf.CellMethod(axes='area', method='mean')

cell_method2 = cf.CellMethod()
cell_method2.set_axes(axisT)
cell_method2.set_method('maximum')

# Insert the cell method constructs into the field in the same
# order that their methods were applied to the data
Q.set_construct(cell_method1)
Q.set_construct(cell_method2)

# Create a "time" dimension coordinate construct, with coordinate
# bounds
dimT = cf.DimensionCoordinate(
properties={'standard_name': 'time',
'units': 'days since 2018-12-01'},
data=cf.Data([15.5]),
bounds=cf.Bounds(data=cf.Data([[0,31.]])))

# Create a "longitude" dimension coordinate construct, without
# coordinate bounds
dimX = cf.DimensionCoordinate(data=cf.Data(numpy.arange(8.)))
dimX.set_properties({'standard_name': 'longitude',
'units': 'degrees_east'})

# Create a "longitude" dimension coordinate construct
dimY = cf.DimensionCoordinate(properties={'standard_name': 'latitude',
'units'     : 'degrees_north'})
array = numpy.arange(5.)
dimY.set_data(cf.Data(array))

# Create and insert the latitude coordinate bounds
bounds_array = numpy.empty((5, 2))
bounds_array[:, 0] = array - 0.5
bounds_array[:, 1] = array + 0.5
bounds = cf.Bounds(data=cf.Data(bounds_array))
dimY.set_bounds(bounds)

# Insert the dimension coordinate constructs into the field,
# specifying to which domain axis each one corresponds
Q.set_construct(dimT)
Q.set_construct(dimY)
Q.set_construct(dimX)

# End of code block
Q.dump()
# Start of code block

import numpy
import cf

# Initialise the field construct
tas = cf.Field(
properties={'project': 'research',
'standard_name': 'air_temperature',
'units': 'K'})

# Create and set domain axis constructs
axis_T = tas.set_construct(cf.DomainAxis(1))
axis_Z = tas.set_construct(cf.DomainAxis(1))
axis_Y = tas.set_construct(cf.DomainAxis(10))
axis_X = tas.set_construct(cf.DomainAxis(9))

# Set the field construct data
tas.set_data(cf.Data(numpy.arange(90.).reshape(10, 9)))

# Create and set the cell method constructs
cell_method1 = cf.CellMethod(
axes=[axis_Y, axis_X],
method='mean',
qualifiers={'where': 'land',
'interval': [cf.Data(0.1, units='degrees')]})

cell_method2 = cf.CellMethod(axes=axis_T, method='maximum')

tas.set_construct(cell_method1)
tas.set_construct(cell_method2)

# Create and set the field ancillary constructs
field_ancillary = cf.FieldAncillary(
properties={'standard_name': 'air_temperature standard_error',
'units': 'K'},
data=cf.Data(numpy.arange(90.).reshape(10, 9)))

tas.set_construct(field_ancillary)

# Create and set the dimension coordinate constructs
dimension_coordinate_T = cf.DimensionCoordinate(
properties={'standard_name': 'time',
'units': 'days since 2018-12-01'},
data=cf.Data([15.5]),
bounds=cf.Bounds(data=cf.Data([[0., 31]])))

dimension_coordinate_Z = cf.DimensionCoordinate(
properties={'computed_standard_name': 'altitude',
'standard_name': 'atmosphere_hybrid_height_coordinate'},
data = cf.Data([1.5]),
bounds=cf.Bounds(data=cf.Data([[1.0, 2.0]])))

dimension_coordinate_Y = cf.DimensionCoordinate(
properties={'standard_name': 'grid_latitude',
'units': 'degrees'},
data=cf.Data(numpy.arange(10.)),
bounds=cf.Bounds(data=cf.Data(numpy.arange(20).reshape(10, 2))))

dimension_coordinate_X = cf.DimensionCoordinate(
properties={'standard_name': 'grid_longitude',
'units': 'degrees'},
data=cf.Data(numpy.arange(9.)),
bounds=cf.Bounds(data=cf.Data(numpy.arange(18).reshape(9, 2))))

dim_T = tas.set_construct(dimension_coordinate_T, axes=axis_T)
dim_Z = tas.set_construct(dimension_coordinate_Z, axes=axis_Z)
dim_Y = tas.set_construct(dimension_coordinate_Y)
dim_X = tas.set_construct(dimension_coordinate_X)

# Create and set the auxiliary coordinate constructs
auxiliary_coordinate_lat = cf.AuxiliaryCoordinate(
properties={'standard_name': 'latitude',
'units': 'degrees_north'},
data=cf.Data(numpy.arange(90.).reshape(10, 9)))

auxiliary_coordinate_lon = cf.AuxiliaryCoordinate(
properties={'standard_name': 'longitude',
'units': 'degrees_east'},
data=cf.Data(numpy.arange(90.).reshape(9, 10)))

array = numpy.ma.array(list('abcdefghij'))
array[0] = numpy.ma.masked
auxiliary_coordinate_name = cf.AuxiliaryCoordinate(
properties={'long_name': 'Grid latitude name'},
data=cf.Data(array))

aux_LAT  = tas.set_construct(auxiliary_coordinate_lat)
aux_LON  = tas.set_construct(auxiliary_coordinate_lon)
aux_NAME = tas.set_construct(auxiliary_coordinate_name)

# Create and set domain ancillary constructs
domain_ancillary_a = cf.DomainAncillary(
properties={'units': 'm'},
data=cf.Data([10.]),
bounds=cf.Bounds(data=cf.Data([[5., 15.]])))

domain_ancillary_b = cf.DomainAncillary(
properties={'units': '1'},
data=cf.Data([20.]),
bounds=cf.Bounds(data=cf.Data([[14, 26.]])))

domain_ancillary_orog = cf.DomainAncillary(
properties={'standard_name': 'surface_altitude',
'units': 'm'},
data=cf.Data(numpy.arange(90.).reshape(10, 9)))

domain_anc_A = tas.set_construct(domain_ancillary_a, axes=axis_Z)
domain_anc_B = tas.set_construct(domain_ancillary_b, axes=axis_Z)
domain_anc_OROG = tas.set_construct(domain_ancillary_orog)

# Create the datum for the coordinate reference constructs
datum = cf.Datum(parameters={'earth_radius': 6371007.})

# Create the coordinate conversion for the horizontal coordinate
# reference construct
coordinate_conversion_h = cf.CoordinateConversion(
parameters={'grid_mapping_name': 'rotated_latitude_longitude',
'grid_north_pole_latitude': 38.0,
'grid_north_pole_longitude': 190.0})

# Create the coordinate conversion for the vertical coordinate
# reference construct
coordinate_conversion_v = cf.CoordinateConversion(
parameters={'standard_name': 'atmosphere_hybrid_height_coordinate',
'computed_standard_name': 'altitude'},
domain_ancillaries={'a': domain_anc_A,
'b': domain_anc_B,
'orog': domain_anc_OROG})

# Create the vertical coordinate reference construct
horizontal_crs = cf.CoordinateReference(
datum=datum,
coordinate_conversion=coordinate_conversion_h,
coordinates=[dim_X,
dim_Y,
aux_LAT,
aux_LON])

# Create the vertical coordinate reference construct
vertical_crs = cf.CoordinateReference(
datum=datum,
coordinate_conversion=coordinate_conversion_v,
coordinates=[dim_Z])

# Set the coordinate reference constructs
tas.set_construct(horizontal_crs)
tas.set_construct(vertical_crs)

# Create and set the cell measure constructs
cell_measure = cf.CellMeasure(measure='area',
properties={'units': 'km2'},
data=cf.Data(numpy.arange(90.).reshape(9, 10)))

tas.set_construct(cell_measure)

# End of code block
print(tas)
q, t = cf.read('file.nc')
print(q.creation_commands())
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
t.equals(t)
t.equals(t.copy())
t.equals(t[...])
t.equals(q)
t.equals(q, verbose=2)
print(cf.atol())
print(cf.rtol())
original = cf.rtol(0.00001)
print(cf.rtol())
print(cf.rtol(original))
print(cf.rtol())
t2 = t - 0.00001
t.equals(t2)
with cf.atol(1e-5):
    print(t.equals(t2))
t.equals(t2)
orog = t.constructs('surface_altitude').value()
orog.equals(orog.copy())
print(t.constructs.filter_by_ncvar('b'))
t.constructs('ncvar%x').value()
t.constructs('ncdim%x')
q.nc_get_variable()
q.nc_global_attributes()
q.nc_set_variable('humidity')
q.nc_get_variable()
q.constructs('latitude').value().nc_get_variable()
print(q)
cf.write(q, 'q_file.nc')
x
cf.write(x, 'new_file.nc')
g = cf.example_field(2)
cf.write(g, 'append-example-file.nc')
cf.read('append-example-file.nc')
h = cf.example_field(0)
h
cf.write(h, 'append-example-file.nc', mode='a')
cf.read('append-example-file.nc')
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
cf.write(f, 'f_file.nc', file_descriptors={'history': 'created in 2019'})
f_file = cf.read('f_file.nc')[0]
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
q, t = cf.read('file.nc')
print(q)
q.set_property('comment', 'comment')
q.nc_set_group_attribute('comment', 'group comment')
q.nc_set_variable_groups(['forecast', 'model'])
q.construct('time').nc_set_variable_groups(['forecast'])
cf.write(q, 'grouped.nc')
g = cf.read('grouped.nc')[0]
print(g)
g.nc_get_variable()
g.nc_variable_groups()
g.nc_group_attributes(values=True)
g.construct('latitude').nc_get_variable()
cf.write(g, 'flat.nc', group=False)
f = cf.read('flat.nc')[0]
f.equals(g)
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
area = g.construct('measure:area')
area
area.nc_get_external()
area.nc_get_variable()
area.properties()
area.data
area.nc_set_external(True)
cf.write(g, 'new_parent.nc')
cf.write(g, 'new_parent.nc', external='new_external.nc')
a = cf.read('air_temperature.nc')[0]
a
a_parts = [a[0, : , 0:30], a[0, :, 30:96], a[1, :, 0:30], a[1, :, 30:96]]
a_parts
for i, f in enumerate(a_parts):
    cf.write(f, str(i)+'_air_temperature.nc')
x = cf.read('[0-3]_air_temperature.nc')
y = cf.read('[0-3]_air_temperature.nc', aggregate=False)
z = cf.aggregate(y)
x
z
x.equals(z)
x = cf.aggregate(a_parts)
x
a_parts[1].transpose(inplace=True)
a_parts[1].units = 'degreesC'
a_parts
z = cf.aggregate(a_parts)
z
x.equals(z)
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
# Start of code block

import numpy
import cf

# Define the array values
data = cf.Data([[280.0,-99,   -99,   -99],
[281.0, 279.0, 278.0, 279.5]])
data.where(cf.eq(-99), cf.masked, inplace=True)

# Create the field construct
T = cf.Field()
T.set_properties({'standard_name': 'air_temperature',
'units': 'K',
'featureType': 'timeSeries'})

# Create the domain axis constructs
X = T.set_construct(cf.DomainAxis(4))
Y = T.set_construct(cf.DomainAxis(2))

# Set the data for the field
T.set_data(data)

# Compress the data
T.compress('contiguous',
count_properties={'long_name': 'number of obs for this timeseries'},
inplace=True)

# End of code block
T
print(T.array)
T.data.get_compression_type()
print(T.data.compressed_array)
count_variable = T.data.get_count()
count_variable
print(count_variable.array)
cf.write(T, 'T_contiguous.nc')
# Start of code block

import numpy
import cf

# Define the ragged array values
ragged_array = cf.Data([280, 281, 279, 278, 279.5])

# Define the count array values
count_array = [1, 4]

# Create the count variable
count_variable = cf.Count(data=cf.Data(count_array))
count_variable.set_property('long_name',
'number of obs for this timeseries')

# Create the contiguous ragged array object, specifying the
# uncompressed shape
array = cf.RaggedContiguousArray(
compressed_array=ragged_array,
shape=(2, 4), size=8, ndim=2,
count_variable=count_variable)

# Create the field construct
T.set_properties({'standard_name': 'air_temperature',
'units': 'K',
'featureType': 'timeSeries'})

# Create the domain axis constructs for the uncompressed array
X = T.set_construct(cf.DomainAxis(4))
Y = T.set_construct(cf.DomainAxis(2))

# Set the data for the field
T.set_data(cf.Data(array))

# End of code block
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
# Start of code block

import numpy
import cf

# Define the gathered values
gathered_array = cf.Data([[2.0, 1, 3], [4, 0, 5]])

# Define the list array values
list_array = [1, 4, 5]

# Create the list variable
list_variable = cf.List(data=cf.Data(list_array))

# Create the gathered array object, specifying the mapping between
# compressed and uncompressed dimensions, and the uncompressed
# shape.
array = cf.GatheredArray(
compressed_array=gathered_array,
compressed_dimensions={1: [1, 2]},
shape=(2, 3, 2), size=12, ndim=3,
list_variable=list_variable
)

# Create the field construct with the domain axes and the gathered
# array
P = cf.Field(properties={'standard_name': 'precipitation_flux',
'units': 'kg m-2 s-1'})

# Create the domain axis constructs for the uncompressed array
T = P.set_construct(cf.DomainAxis(2))
Y = P.set_construct(cf.DomainAxis(3))
X = P.set_construct(cf.DomainAxis(2))

# Set the data for the field
P.set_data(cf.Data(array), axes=[T, Y, X])

# End of code block
P
print(P.data.array)
P.data.get_compression_type()
print(P.data.compressed_array)
list_variable = P.data.get_list()
list_variable
print(list_variable.array)
cf.write(P, 'P_gathered.nc')
pp = cf.read('umfile.pp')
pp
print(pp[0])
cf.write(pp, 'umfile1.nc')
stash = cf.stash2standard_name()
stash[(1, 4)]
stash[(1, 7)]
stash[(1, 2)]
stash[(1, 152)]
(1, 999) in stash
with open('new_STASH.txt', 'w') as new:
    new.write('1!999!My STASH code!1!!!ultraviolet_index!!')
cf.load_stash2standard_name('new_STASH.txt', merge=True)
new_stash = cf.stash2standard_name()
new_stash[(1, 999)]
