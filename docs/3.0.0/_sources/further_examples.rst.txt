.. currentmodule:: cf
.. default-role:: obj

Further examples
================

Reading files
-------------

The `cf.read` function will read `CF-netCDF
<http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/cf-conventions.html>`_
and `CFA-netCDF <http://www.met.reading.ac.uk/~david/cfa/0.3/>`_ files
(or URLs if DAP access is enabled) and Met Office (UK) PP files and
fields files from disk and return their contents as a field or a
:ref:`field list <fs_field_list>`, i.e. an ordered collection of
fields stored in a `cf.FieldList` object:

>>> f = cf.read('data.nc')
>>> f
[<CF Field: air_pressure(grid_latitude(30), grid_longitude(24)) Pa>,
 <CF Field: x_wind(height(19), grid_latitude(29), grid_longitude(24)) m s-1>,
 <CF Field: y_wind(height(19), grid_latitude(29), grid_longitude(24)) m s-1>,
 <CF Field: air_potential_temperature(height(19), grid_latitude(30), grid_longitude(24)) K>]
>>> f[-1]
<CF Field: air_potential_temperature(height(19), grid_latitude(30), grid_longitude(24)) K>

Multiple files may be read at once by using Unix shell wildcard
characters in file names or providing a sequence of files:

>>> f = cf.read('~/file.nc')
>>> f = cf.read('file[1-9a-c].nc')
>>> f = cf.read('dir*/*.pp')
>>> f = cf.read(['file1.nc', 'file2.nc', 'file3*.nc'])

File names may use environment variables and ``~`` expansion:

>>> f = cf.read('~/*.nc')
>>> f = cf.read('$DATA/*.nc')
>>> f = cf.read('$DATA/${MORE_DATA}/*.nc')
>>> f = cf.read('~/$DATA/${MORE_DATA}/*.nc')

For each file, the file format is inferred from the file contents, not
from the file name suffix.

Writing files
-------------

The `cf.write` function will write a field or field list to a
CF-netCDF or CFA-netCDF file on disk:

>>> cf.write(f, 'newfile.nc')

A sequence of fields and field lists may be written to the same file:

>>> cf.write([f, g], 'newfile.nc')

All of the input fields are written to the same output file, but if
metadata (such as coordinates) are identical in two or more fields
then that metadata is only written once to the output file.

Output file names are arbitrary (in particular, they do not require a
suffix).


CF properties and attributes
----------------------------

The field's CF properties are those which are intended to be essential
metadata for the data array and would be included as data variable
properties in a CF compliant netCDF file (such as
`~Field.standard_name`). The field's attributes are any other
attributes, such as the names of the files containing the data array
values.

The CF properties are returned by the field's `~Field.properties`
attribute:

>>> f.properties
{'_FillValue': 1e+20,
 'cell_methods': <CF CellMethods: time: mean>,
 'standard_name': 'air_temperature',
 'units': 'K'}

A CF property *recognised by the CF conventions* or an attribute may
be set, retrieved and deleted as standard python object attributes:

>>> f.standard_name = 'air_temperature'
>>> f.standard_name
'air_temperature'
>>> del f.standard_name
>>> getattr(f, 'standard_name')
AttributeError: Field doesn't have CF property 'long_name'
>>> getattr(f, 'standard_name', 'air_pressure')
'air_pressure'
>>> setattr(f, 'standard_name', 'air_pressure')

>>> f.ncvar = 'tas'
>>> getattr(f, 'ncvar')
'tas'
>>> del f.ncvar

Any CF property (recognised by the CF conventions or not) may be set,
retrieved and deleted with the field's `~Field.setprop`,
`~Field.getprop` and `~Field.delprop` methods:

>>> f.properties
{}
>>> f.setprop('long_name', 'temperature at 1.5m')
>>> f.getprop('long_name')
'temperature at 1.5m'
>>> f.delprop('long_name')
>>> f.getprop('long_name')
AttributeError: Field doesn't have CF property 'long_name'
>>> f.getprop('long_name', 'pressure')
'pressure'
>>> f.setprop('long_name', 'pressure')
>>> f.setprop('foo', 'bar')
>>> f.properties
{'foo': 'bar',
 'long_name': 'pressure'}

Selecting fields
----------------

Fields may be selected with the `~cf.Field.match` and
`~cf.Field.select` methods. These methods take conditions on field CF
properties, attributes and coordinates as inputs:

>>> f
[<CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: air_temperature(time(12), grid_latitude(73), grid_longitude(96)) K>]

>>> f.match('air')
[False, True]
>>> g = f.select('air_temperature', items={'longitude': 0})
>>> g
[<CF Field: air_temperature(time(12), grid_latitude(73), grid_longitude(96)) K>]

The data array
--------------

The field's data array may be retrieved as an independent numpy array
with the field's `~Field.array` attribute:

>>> print f.array
[[ 2.  4.]
 [ 5.  1.]]

A particular element of the data array may be retrieved with the
field's `~cf.Field.datum` method:

>>> f.datum(0)
2.0
>>> f.datum(-1)
1.0
>>> f.datum(0, 1)
4.0


Coordinates
-----------

A dimension or auxiliary coordinate object of the field's domain may
be returned by the field's `~cf.Field.coord` method. A coordinate
object has the same CF property and data access principles as a field:

>>> c = f.coord('time')
>>> c
<CF DimensionCoordinate: time(12)>
>>> c.properties()
{'_FillValue': None,
 'axis': 'T',
 'calendar': 'noleap',
 'long_name': 'time',
 'standard_name': 'time',
 'units': 'days since 0000-1-1'}
>>> c.Units
<CF Units: days since 0000-1-1 calendar=noleap>
>>> print c.array
[  0  30  60  90 120 150 180 210 240 270 300 330]

.. _creating-a-field:

Creating a field
---------------- 

A new field may be created by initializing a new `cf.Field`
instance. See the section on :ref:`field creation <field_creation>`
for details.
