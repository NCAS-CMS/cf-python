.. currentmodule:: cf
.. default-role:: obj

.. _field_structure:

Introduction to the `cf.Field` object
=====================================

A `cf.Field` object stores a field as defined by the `CF-netCDF
conventions <http://cfconventions.org>`_ and the `CF data model
<http://cf-trac.llnl.gov/trac/ticket/95>`_. It is a container for a
data array and metadata comprising properties to describe the physical
nature of the data and a coordinate system (called a *domain*), which
describes the positions of each element of the data array.

It is structured in exactly the same way as a filed in the CF data
model and, as in the CF data model, all components of a `cf.Field`
object are optional.

Displaying the contents
-----------------------

The structure may be exposed with three different levels of detail.

The built-in `repr` function returns a short, one-line description of
the field:

>>> f
<CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>

This gives the identity of the field (air_temperature), the identities
and sizes of its data array axes (time, latitude and longitude with
sizes 12, 64 and 128 respectively) and the units of the field's data
array (K).

The built-in `str` function returns the same information as the the
one-line output, along with short descriptions of the field's other
components:

>>> print f
Field: air_temperature (ncvar%tas)
----------------------------------
Data           : air_temperature(time(1200), latitude(64), longitude(128)) K
Cell methods   : time: mean (interval: 1.0 month)
Axes           : time(12) = [ 450-11-01 00:00:00, ...,  451-10-16 12:00:00] noleap calendar
               : latitude(64) = [-87.8638000488, ..., 87.8638000488] degrees_north
               : longitude(128) = [0.0, ..., 357.1875] degrees_east
               : height(1) = [2.0] m

This shows that the field has a cell method and four dimension
coordinates, one of which (height) is a coordinate for a size 1 axis
that is not a axis of the field's data array. The units and first and
last values of the coordinates' data arrays are given and relative
time values are translated into strings.

The field's `~cf.Field.dump` method describes each component's
properties, as well as the first and last values of the field's data
array::

   >>> f.dump()
   ----------------------------------
   Field: air_temperature (ncvar%tas)
   ----------------------------------
   experiment_id = 'pre-industrial control experiment'
   long_name = 'Surface Air Temperature'
   standard_name = 'air_temperature'
   title = 'model output prepared for IPCC AR4'

   Domain Axis: height(1)
   Domain Axis: latitude(64)
   Domain Axis: longitude(128)
   Domain Axis: time(12)
   
   Data(time(12), latitude(64), longitude(128)) = [[[236.512756348, ..., 256.93371582]]] K

   Cell Method: time: mean (interval: 1.0 month)
   
   Dimension coordinate: time
       Data(time(12)) = [ 450-11-16 00:00:00, ...,  451-10-16 12:00:00] noleap calendar
       Bounds(time(12), 2) = [[ 450-11-01 00:00:00, ...,  451-11-01 00:00:00]] noleap calendar
       axis = 'T'
       long_name = 'time'
       standard_name = 'time'
   
   Dimension coordinate: latitude
       Data(latitude(64)) = [-87.8638000488, ..., 87.8638000488] degrees_north
       Bounds(latitude(64), 2) = [[-90.0, ..., 90.0]] degrees_north
       axis = 'Y'
       long_name = 'latitude'
       standard_name = 'latitude'
   
   Dimension coordinate: longitude
       Data(longitude(128)) = [0.0, ..., 357.1875] degrees_east
       Bounds(longitude(128), 2) = [[-1.40625, ..., 358.59375]] degrees_east
       axis = 'X'
       long_name = 'longitude'
       standard_name = 'longitude'
   
   Dimension coordinate: height
       Data(height(1)) = [2.0] m
       axis = 'Z'
       long_name = 'height'
       positive = 'up'
       standard_name = 'height'

.. _fs-data-array:

Data
----

A field's data array is a `cf.Data` object and is returned by its
`~Field.data` attribute.

>>> f.data
<CF Data: [[[89.0, ..., 66.0]]] K>

The `cf.Data` object:

* Contains an N-dimensional array with many similarities to a `numpy`
  array.

* Contains the units of the array elements.

* Uses :ref:`LAMA <LAMA>` functionality to store and operate on arrays
  which are larger then the available memory.

* Supports masked arrays [#f1]_, regardless of whether or not it was
  initialized with a masked array.

Data attributes
---------------

Some of a field's reserved attributes return information on its
data. For example, to find the shape of the data and to retrieve the
data array as an actual `numpy` array:

>>> f.shape
(1, 3, 4)
>>> f.array
array([[[ 89.,  80.,  71.],
        [ 85.,  76.,  67.],
        [ 83.,  74.,  65.],
        [ 84.,  75.,  66.]]])

The data array's missing value mask may be retrieved with the
`~Field.mask` attribute. The mask is returned as a new field with a
boolean data array:

>>> m = f.mask
>>> m.data
<CF Data: [[[False, ..., True]]]>

If the field contains no missing data then a mask field with False
values is still returned.


CF properties
-------------

Standard CF data variable properties (such as
`~cf.Field.standard_name`, `~cf.Field.units`, etc.) all have reserved
attribute names. See the :ref:`list of reserved CF properties
<field_cf_properties>` for details. These properties may be set,
retrieved and deleted like normal python object attributes:

>>> f.standard_name = 'air_temperature'
>>> f.standard_name
'air_temperature'
>>> del f.standard_name

as well as with the dedicated `~Field.setprop`, `~Field.getprop` and
`~Field.delprop` field methods:

>>> f.setprop('standard_name', 'air_temperature')
>>> f.getprop('standard_name')
'air_temperature'
>>> f.delprop('standard_name')

Non-CF properties (i.e. those which are allowed by the CF conventions
but which do not have standardised meanings) *must* be accessed using
these three methods:

>>> f.setprop('project', 'CMIP7')
>>> f.getprop('project')
'CMIP7'
>>> f.delprop('project')

All of the field's CF properties may be retrieved with the field's
`~Field.properties` method:

>>> f.properties()
{'_FillValue': 1e+20,
 'project': 'CMIP7',
 'long_name': 'Surface Air Temperature',
 'standard_name': 'air_temperature',
 'units': 'K'}


Other attributes
----------------

Any other attribute may be set on directly on a field object with, in
general, no special meaning attached to it. These attributes are
distinct properties (CF and non-CF) since they are not considered as
part of the CF conventions and will not be written to files on disk.

The following attributes do, however, have particular interpretations:

=========  ==============================================================
Attribute  Description
=========  ==============================================================
`!id`      An identifier for the field in the absence of a  
           standard name. See the `~Field.identity` method for details.
`!ncvar`   The netCDF variable name of the field.           
=========  ==============================================================

All of the field's attributes may be retrieved with the field's
`~Field.attributes()` method:

>>> f.foo = 'bar'
>>> f.attributes()
{'foo': 'bar',
 'ncar': 'tas'}

Methods
-------

A field has a large range of methods which, in general, either return
information about the field or change the field in place. See the
:ref:`list of methods <field_methods>` and :ref:`manipulating fields
<manipulating-fields>` section for details.

.. _domain_structure:

Domain
------

A field's domain completely describes the location and nature of the
field's data. It comprises domain axes (which describe the field's
dimensionality), dimension coordinate, auxiliary coordinate and cell
measure objects (which themselves contain data arrays and properties
to describe them) and coordinate reference objects (which relate the
field's coordinate values to locations in a planetary reference
frame).

Each item has a unique internal identifier (is a string containing a
number), which serves to link related items.

Items
^^^^^

Domain items are stored in the following objects:

===========================  ========================
Item                         `cf` object
===========================  ========================
Dimension coordinate object  `cf.DimensionCoordinate`
Auxiliary coordinate object  `cf.AuxiliaryCoordinate`
Cell measure object          `cf.CellMeasure`
Coordinate reference object  `cf.CoordinateReference`
===========================  ========================

These items may be retrieved with a variety of methods, some specific
to each item type (such as `cf.Field.dim`) and some more generic (such
as `cf.Field.coords` and `cf.Field.item`):

===========================  ==================================================================
Item                         Field retrieval methods
===========================  ==================================================================
Dimension coordinate object  `~Field.dim`, `~Field.dims`, `~Field.coord`, `~Field.coords`
	                     `~Field.item`, `~Field.items`
Auxiliary coordinate object  `~Field.aux`, `~Field.auxs`, `~Field.coord`, `~Field.coords`
	                     `~Field.item`, `~Field.items`
Cell measure object          `~Field.measure`, `~Field.measures`, `~Field.item`, `~Field.items`
Coordinate reference object  `~Field.ref`, `~Field.refs`, `~Field.item`, `~Field.items`
===========================  ==================================================================

In each case the singular method form (such as `~Field.aux`) returns
an actual domain item whereas the plural method form (such as
`~Field.auxs`) returns a dictionary whose keys are the domain item
identifiers with corresponding values of the items themselves.

For example, to retrieve a unique dimension coordinate object with a
standard name of "time":

>>> f.dim('time')
<CF DimensionCoordinate: time(12) noleap>

To retrieve all coordinate objects and their domain identifiers:

>>> f.coords()
{'dim0': <CF DimensionCoordinate: time(12) noleap>,
 'dim1': <CF DimensionCoordinate: latitude(64) degrees_north>,
 'dim2': <CF DimensionCoordinate: longitude(128) degrees_east>,
 'dim3': <CF DimensionCoordinate: height(1) m>}

To retrieve all domain items and their domain identifiers:

>>> f.items()
{'dim0': <CF DimensionCoordinate: time(12) noleap>,
 'dim1': <CF DimensionCoordinate: latitude(64) degrees_north>,
 'dim2': <CF DimensionCoordinate: longitude(128) degrees_east>,
 'dim3': <CF DimensionCoordinate: height(1) m>}

In this example, all of the items happen to be coordinates.

Domain axes
^^^^^^^^^^^

Common axes of variation in the field's data array and the domain's
items are defined by the domain's axes.

Particular axes may be retrieved with the `~cf.Field.axes` method:

>>> f.axes()
set(['dim1', 'dim0' 'dim2' 'dim3'])
>>> f.axes(size=19)
set(['dim1'])
>>> f.axes('time')
set(['dim0'])

The axes spanned by a domain item or the field's data array may be
retrieved with the fields `~cf.Field.item_axes` or
`~cf.Field.data_axes` methods respectively:

>>> f.item_axes('time')
['dim0']
>>> f.data_axes()
['dim0', 'dim1' 'dim2' 'dim3']

Note that the field's data array may contain fewer size 1 axes than
its domain.

.. COMMENTED OUT
   .. _fs_field_list:
   
   Field list
   ----------
   
   A `cf.FieldList` object is an ordered sequence of fields.
   
   It supports the :ref:`python list-like operations
   <python:typesseq-mutable>`. For example:
   
   >>> fl
   [<CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>,
    <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>]
   >>> fl[0]
   <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>
   >>> fl[::-1]
   [<CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>,
    <CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>]
   
   >>> len(fl)
   2
   >>> f = fl.pop()
   >>> f
   <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>
   >>> len(fl)
   1
   >>> fl.append(f)
   >>> len(fl)
   2
   >>> f in fl
   True
   
   A field list, however, has :ref:`its own definitions <fs-fl-a-and-c>`
   of the arithmetic and comparison operators.
   
   
   Methods, attributes and CF properties
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
   A field list object also has all of the callable methods, reserved
   attributes and reserved CF properties that a field object has. When
   used with a field list, a callable method (such as
   `~cf.FieldList.item`) or a reserved attribute or CF property (such as
   `~cf.FieldList.Units` or `~cf.FieldList.standard_name`) is applied
   independently to each field and, unless a method (or assignment to a
   reserved attribute or CF property) carries out an in-place change to
   each field, a sequence of the results is returned.
   
   The type of sequence that may be returned will either be a new
   `cf.FieldList` object or else a new `cf.List` object. For example,
   `cf.FieldList.subspace` will return a new field list of subspaced
   fields:
   
   >>> fl
   [<CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>,
    <CF Field: air_pressure(time(12), latitude(73), longitude(96)) hPa>]
   xasxasxasxs   >>> flppppp[0, ...]
   [<CF Field: air_temperature(time(1), latitude(73), longitude(96)) K>,
    <CF Field: air_pressure(time(1), latitude(73), longitude(96)) hPa>]
   
   whereas `cf.FieldList.ndim`, `cf.FieldList.standard_name` and
   `cf.FieldList.item` return a `cf.List` of integers, strings and domain
   items respectively:
   
   >>> fl.ndim
   [3, 3]
   >>> fl.standard_name
   ['air_temperature', 'air_pressure']
   >>> fl.item('time')
   [<CF DimensionCoordinate: time(12) days since 1860-1-1>,
    <CF DimensionCoordinate: time(12) days since 1860-1-1>]
   
   A `cf.List` object is very much like a built-in list, in that it has
   all of the built-in list methods, but it also has an extra method,
   called `~cf.List.method`, which allows any callable method (with
   arguments) to be applied independently to each element of the list,
   returning the result in a new `cf.List` object:
   
   >>> fl.standard_name[::-1]
   ['air_pressure', 'air_temperature']
   >>> fl.standard_name.method('upper')
   ['AIR_TEMPERATURE', 'AIR_PRESSURE']
   >>> fl.item('time').method('getprop', 'standard_name')
   ['time', 'time']
   >>> fl.item('time').method('delrop')
   [None, None]
   >>> fl.item('time').method('setprop', 'standard_name', 'foo')
   [None, None]
   >>> fl.item('time').method('getprop', 'standard_name')
   ['foo', 'foo']
   
   The `cf.FieldList` object also has an equivalent method called
   `~cf.FieldList.method` which behaves in an analogous way, thus
   reducing the need to know which type of sequence has been returned
   from a field list method:
   
   >>> fl.getprop('standard_name') == fl.method('getprop', 'standard_name')
   True
   
   Assignment to reserved attributes and CF properties assigns the value
   to each field in turn. Similarly, deletion is carried out on each field:
   
   >>> fl.standard_name
   ['air_pressure', 'air_temperature']
   >>> fl.standard_name = 'foo'
   ['foo', 'foo']
   >>> del fl.standard_name
   >>> fl.getprop('standard_name', 'MISSING')
   ['MISSING', 'MISSING']
   
   Note that the new value is not copied prior to each field assignment,
   which may be an issue for values which are mutable objects.
   
   Changes tailored to each field in the list are easily carried out in a
   loop:
   
   >>> for f in fl:
   ...     f.long_name = 'An even longer ' + f.long_name
   
   .. _fs-fl-a-and-c:
   
   Arithmetic and comparison
   ^^^^^^^^^^^^^^^^^^^^^^^^^
   
   Any arithmetic and comparison operation is applied independently to
   each field element, so all of the :ref:`operators defined for a field
   <Arithmetic-and-comparison>` are allowed.
   
   In particular, the usual :ref:`python list-like arithmetic and
   comparison operator behaviours <python:numeric-types>` do not
   apply. For example, the ``+`` operator will concatenate two built-in
   lists, but adding ``2`` to a field list will add ``2`` to the data
   array of each of its fields.
   
   For example these commands:
   
   >>> fl + 2
   >>> 2 + fl
   >>> fl == 0
   >>> fl += 2
   
   are equivalent to:
   
   >>> cf.FieldList(f + 2 for f in fl)
   >>> cf.FieldList(2 + f for f in fl)
   >>> cf.FieldList(f == 0 for f in fl)
   >>> for f in fl:
   ...     f += 2
   
   Field versus field list
   ^^^^^^^^^^^^^^^^^^^^^^^
   
   In some contexts, whether an object is a field or a field list is not
   known. So to avoid ungainly type testing, most aspects of the
   `cf.FieldList` interface are shared by a `cf.Field` object.
   
   A field may be used in the same iterative contexts as a field list:
   
   >>> f
   <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>
   >>> f is f[0]
   True
   >>> f is f[slice(-1, None, -1)]
   True
   >>> f is f[::-1]
   True
   >>> for g in f:
   ...     print repr(g)
   ...
   <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>
   
   When it is not known whether or not you have a field or a field list,
   iterating over the output of a callable method could be complicated
   because the output of the field method will be a scalar when the
   output of the same field list method will be a sequence of
   scalars. The problem is illustrated in this example (note that
   ``f.standard_name`` is an alias for ``f.getprop('standard_name')``):
   
   >>> f = fl[0]
   >>> for x in f.standard_name:
   ...     print x+'.',
   ...
   a.i.r._.p.r.e.s.s.u.r.e.
   
   >>> for x in fl.standard_name:
   ...     print x+'.',
   ...
   air_pressure.air_temperature.
   
   To overcome this difficulty, both the field and field list have a
   method call `!iter` which has no effect on a field list, but which
   changes the output of a field's callable method (with arguments) into
   a single element sequence:
   
   >>> f = fl[0]
   >>> for x in f.iter('getprop', 'standard_name'):
   ...     print x+'.',
   ...
   air_pressure.
   
   >>> for x in fl.iter('getprop', 'standard_name'):
   ...     print x+'.',
   ...
   air_pressure.air_temperature.

----

.. rubric:: Footnotes

.. [#f1] Arrays that may have missing or invalid entries
