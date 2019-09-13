.. set tocdepth in sidebar
:tocdepth: 2

.. currentmodule:: cf
.. default-role:: obj

.. _manipulating-fields:

Manipulating `cf.Field` objects
===============================

Manipulating a field generally involves operating on its data array
and making any necessary changes to the field's domain to make it
consistent with the new array.


Data array
----------

Conversion to a numpy array
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A field's data array may be converted to either an independent numpy
array or a numpy array view (`numpy.ndarray.view`) with its
`~Field.array` and `~Field.varray` attributes respectively:

>>> a = f.array
>>> print a
[[2 -- 4 -- 6]]
>>> a[0, 0] = 999
>>> print a
[[999 -- 4 -- 6]]
>>> print f.array
[[2 -- 4 -- 6]]

Changing the numpy array view in place will also change the
field's data array in-place:

>>> v = f.varray
>>> print v
[[2 -- 4 -- 6]]
>>> v[0, 0] = 999
>>> print f.array
[[999 -- 4 -- 6]]

A field exposes the numpy array interface and so may be used as input
to any of the `numpy array creation functions
<http://docs.scipy.org/doc/numpy/reference/routines.array-creation.html#from-existing-data>`_:

>>> print f.array
[[2 -- 4 -- 6]]
>>> numpy.all(f.array)
True
>>> numpy.all(f)
True

.. note::

   The numpy array created by the `~Field.varray` or `~Field.array`
   attributes forces all of the data to be read into memory at the
   same time, which may not be possible for very large arrays.

Data mask
^^^^^^^^^

A copy of a field's missing data mask is returned by its
`~cf.Field.mask` attribute.

This mask is an independent field in its own right, and so changes to
it will not be seen by the field which generated it. See the
:ref:`assignment section <fm_assignment>` for details on how to edit
the field's mask in place.


Copying
-------

A deep copy of a field may be created with its `~Field.copy` method,
which is functionally equivalent to, but faster than, using the
:py:obj:`copy.deepcopy` function:

>>> g = f.copy()
>>> import copy
>>> g = copy.deepcopy(f)

Copying utilizes :ref:`LAMA copying functionality <LAMA_copying>`.

.. _Subspacing:

Subspacing
----------

Subspacing a field means subspacing its data array and its domain in a
consistent manner.

A field may be subspaced in "index-space" or "domain-space". In
index-space, a subspace is defined by specifying indices of the data
array, whilst in domain-space a subspace is defined as the part of the
data array corresponds to given domain item values (e.g. particular
coordinate values)


Subspacing utilizes :ref:`LAMA subspacing functionality
<LAMA_subspacing>`.


.. _indexing:

Indexing
^^^^^^^^

Subspacing by axis indices is done with the use of square brackets
([]) on a field and uses an extended Python slicing syntax which is
similar to :ref:`numpy array indexing <numpy:arrays.indexing>`:

>>> f.shape
(12, 73, 96)
>>> f[...].shape
(12, 73, 96)
>>> f[slice(0, 12), :, 10:0:-2].shape
(12, 73, 5)
>>> f[..., f.coord('longitude')<180].shape
(12, 73, 48)

There are three extensions to the numpy indexing functionality:

* Size 1 axes are never removed.

  An integer index *i* takes the *i*-th element but does not reduce
  the rank of the output array by one:

  >>> f.shape
  (12, 73, 96)
  >>> f[0].shape
  (1, 73, 96)
  >>> f[3, slice(10, 0, -2), 95:93:-1].shape
  (1, 5, 2)

* The indices for each axis work independently.

  When more than one axis’s slice is a 1-d boolean sequence or 1-d
  sequence of integers, then these indices work independently along
  each axis (similar to the way vector subscripts work in Fortran),
  rather than by their elements:

  >>> f.shape
  (12, 73, 96)
  >>> f[:, [0, 72], [5, 4, 3]].shape
  (12, 2, 3)

  Note that the indices of the last example would raise an error when
  given to a numpy array.

* Boolean indices may be any object which exposes the numpy array
  interface, such as the field's coordinate objects:

  >>> f[:, f.coord('latitude')<0].shape
  (12, 36, 96)

Alternatively, the indices may be applied by indexing
`Field.subspace`. For example:

>>> f[..., 2:34:-2, [2, 4, 5]]

is exactly equivalent to

>>> f.subspace[..., 2:34:-2, [2, 4, 5]]

.. _calling:

Domain values
^^^^^^^^^^^^^

Subspacing by values of domain items (coordinates or cell measures)
allows a subspaced field to be defined via metadata values of its
domain. The benefits of subspacing in this fashion are:

* The axes to be subspaced may identified by name.

* The position in the data array of each axis need not be known and
  the axes to be subspaced may be given in any order.

* Axes for which no subspacing is required need not be specified.

* Size 1 axes of the domain which are not spanned by the data array
  may be specified.

Coordinate values are provided as keyword arguments to a call to
`~Field.subspace`. Coordinates are identified by their
`~Coordinate.identity` or their axis's identifier in the field's
domain.

>>> f.subspace().shape
(12, 73, 96)
>>> f.subspace(latitude=0).shape
(12, 1, 96)
>>> f.subspace(latitude=cf.wi(-30, 30)).shape
(12, 25, 96)
>>> f.subspace(long=cf.ge(270, 'degrees_east'), lat=cf.set([0, 2.5, 10])).shape
(12, 3, 24)
>>> f.subspace(latitude=cf.lt(0, 'degrees_north'))
(12, 36, 96)
>>> f.subspace(latitude=[cf.lt(0, 'degrees_north'), 90])
(12, 37, 96)
>>> import math
>>> f.subspace('exact', longitude=cf.lt(math.pi, 'radian'), height=2)
(12, 73, 48)
>>> f.subspace(height=cf.gt(3))
IndexError: No indices found for 'height' values gt 3
>>> f.subspace(dim2=3.75).shape
(12, 1, 96)
>>> f.subspace(time=cf.le(cf.dt('1860-06-16 12:00:00')).shape
(6, 73, 96)
>>> f.subspace(time=cf.gt(cf.dt(1860, 7)),shape
(5, 73, 96)

Note that if a comparison function (such as `cf.wi`) does not specify
any units, then the units of the named coordinate are assumed.

.. _fm_cyclic_axes:

Cyclic axes
-----------

>>> f[..., -10, 10]
(12, 25, 96)
>>> f.subspace(longitude=cf.wi(-30, 30))
(12, 3, 24)
>>> f.subspace(long=cf.ge(270, 'degrees_east'), lat=cf.set([0, 2.5, 10])).shape
(12, 3, 24)


.. _fm_assignment:

Assignment
----------

Elements of a field's data array may be changed by assigning values
directly to an indexed subspace the field or by using the
`~cf.Field.where` method.

Assignment uses :ref:`LAMA functionality <LAMA>`, so it is possible to
assign to fields which are larger than the available memory.

Array elements may be set from a field or logically scalar object,
using the same :ref:`metadata-aware broadcasting rules <broadcasting>`
as for field arithmetic and comparison operations. In the
`~cf.Field.subspace` case, the object attribute must be broadcastable
to the defined subspace, whilst in the `~cf.Field.where` case the
object must be broadcastable to the field itself.

The treatment of missing data elements depends on the value of field's
`~cf.Field.hardmask` attribute. If it is True then masked elements
will not unmasked, otherwise masked elements may be set to any
value. In either case, unmasked elements may be set to any value
(including missing data).

Set all values to 273.15:

>>> f[...] = 273.15

or equivalently:

>>> f.where(True, 273.15, None, i=True)

Set all negative data array values to zero and leave all other
elements unchanged:

>>> g = f.where(f<0, 0)

Double the values in the northern hemisphere:

>>> index = f.indices(longitude=cf.ge(0))
>>> f[index] *= 2

See `cf.Field.where` for more examples.

Selection
---------

Field selection
^^^^^^^^^^^^^^^

Fields from field lists may be selected according to conditions on
their metadata with the `cf.FieldList.select` method (as well as the
`cf.Field.select` method). Conditions may be given on attributes and
CF properties, domain items of the field (dimension coordinate,
auxiliary coordinate, cell measure or coordinate reference objects),
the number of field domain axes and the number of field data array
axes. For example:

>>> f
[<CF Field: eastward_wind(grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>]
>>> f.select('air_temperature')
<CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>]

>>>  f.select('air_temperature', rank=2)
[]
>>>  f.select('air_temperature', items={'latitude': cf.gt(0)}, rank=cf.ge(3))
<CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>

Any of the `~FieldList.select` arguments may also be used with
`cf.read` to select fields when reading from files:

>>> f = cf.read('file*.nc', select='air_temperature')
>>> f = cf.read('file*.nc', select_options={'rank': cf.gt(2)})
>>> f = cf.read('file*.nc', select='air_temperature', select_options={'rank': cf.gt(2)})

This may be faster than reading all fields and then selecting afterwards. 

Domain item selection
^^^^^^^^^^^^^^^^^^^^^

Domain items may be retrieved with a variety of methods, some specific
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

Aggregation
-----------

Fields are aggregated into as few multidimensional fields as possible
with the `cf.aggregate` function, which implements the `CF aggregation
rules
<http://www.met.reading.ac.uk/~david/cf_aggregation_rules.html>`_.

>>> f
[<CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>,
 <CF Field: air_temperature(latitude(73), longitude(96)) K @ 273.15>]
>>> print f
Field: air_temperature (ncvar%temp)
-----------------------------------
Data           : air_temperature(time(12), latitude(73), longitude(96)) K
Cell methods   : time: mean
AXes           : time(12) = [1860-01-16 12:00:00, ..., 1860-12-16 12:00:00]
               : latitude(73) = [-90, ..., 90] degrees_north
               : longitude(96) = [0, ..., 356.25] degrees_east
               : height(1) = [2] m
Field: air_temperature (ncvar%temperature)
------------------------------------------
Data           : air_temperature(latitude(73), longitude(96)) K @ 273.15
Cell methods   : time: mean
Axes           : time(12) = [1859-12-16 12:00:00]
               : longitude(96) = [356.25, ..., 0] degrees_east
               : latitude(73) = [-90, ..., 90] degrees_north
               : height(1) = [2] m
...
>>> g = cf.aggregate(f)
>>> g
[<CF Field: air_temperature(time(13), latitude(73), longitude(96)) K>]
>>> print g
Field: air_temperature (ncvar%temperature)
------------------------------------------
Data           : air_temperature(time(13), latitude(73), longitude(96)) K
Cell methods   : time: mean
Axes           : time(13) = [1859-12-16 12:00:00, ..., 1860-12-16 12:00:00]
               : latitude(73) = [-90, ..., 90] degrees_north
               : longitude(96) = [0, ..., 356.25] degrees_east
               : height(1) = [2] m

By default, the fields returned by `cf.read` have been aggregated:

>>> f = cf.read('file*.nc')
>>> len(f)
1
>>> f = cf.read('file*.nc', aggregate=False)
>>> len(f)
12


.. _Arithmetic-and-comparison:

Arithmetic and comparison
-------------------------

Arithmetic, bitwise and comparison operations are defined on a field
as element-wise operations on its data array which yield a new
`cf.Field` object or, for augmented assignments, modify the field's
data array in-place.

A field's data array is modified in a very similar way to how a numpy
array would be modified in the same operation, i.e. :ref:`broadcasting
<broadcasting>` ensures that the operands are compatible and the data
array is modified element-wise.

Broadcasting is metadata-aware and will automatically account for
arbitrary configurations, such as axis order, but will not allow
fields with incompatible metadata to be combined, such as adding a
field of height to one of temperature.

The :ref:`resulting field's metadata <resulting_metadata>` will be
very similar to that of the operands which are also
fields. Differences arise when the existing metadata can not correctly
describe the newly created field. For example, when dividing a field
with units of *metres* by one with units of *seconds*, the resulting
field will have units of *metres per second*.

Arithmetic and comparison utilizes :ref:`LAMA functionality <LAMA>` so
data arrays larger than the available physical memory may be operated
on.

.. _broadcasting:

Broadcasting
^^^^^^^^^^^^

The term broadcasting describes how data arrays of the operands with
different shapes are treated during arithmetic, comparison and
assignment operations. Subject to certain constraints, the smaller
array is "broadcast" across the larger array so that they have
compatible shapes.

The general broadcasting rules are similar to the :mod:`broadcasting
rules implemented in numpy <numpy.doc.broadcasting>`, the only
difference occurring when both operands are fields, in which case the
fields are temporarily conformed so that:

* The fields have matching units.

* Axes are aligned according to their coordinates' metadata to ensure
  that matching axes are broadcast against each other.

* Common axes have matching axis directions.

This restructuring of the field ensures that the matching axes are
broadcast against each other.

Broadcasting is done without making needless copies of data and so is
usually very efficient.


Valid operands
^^^^^^^^^^^^^^

A field may be combined or compared with the following objects:

+----------------+----------------------------------------------------+
| Object         | Description                                        |
+================+====================================================+
|:py:obj:`int`,  | The field's data array is combined with            |
|:py:obj:`long`, | the python scalar                                  |
|:py:obj:`float` |                                                    |
+----------------+----------------------------------------------------+
|`cf.Data`       | The field's data array                             |
|with size 1     | is combined with the `cf.Data` object's scalar     |
|                | value, taking into account:                        |
|                |                                                    |
|                | * Different but equivalent units                   |
+----------------+----------------------------------------------------+
|`cf.Field`      | The two field's must satisfy the field combination |
|                | rules. The fields' data arrays and domains are     |
|                | combined taking into account:                      |
|                |                                                    |
|                | * Axis identities                                  |
|                | * Array units                                      |
|                | * Axis orders                                      |
|                | * Axis directions                                  |
|                | * Missing data values		              |
+----------------+----------------------------------------------------+


A field may appear on the left or right hand side of an operator.

.. warning::

   Combining a numpy array on the *left* with a field on the *right*
   does work, but will give generally unintended results -- namely a
   numpy array of fields.


.. _resulting_metadata:

Resulting metadata
^^^^^^^^^^^^^^^^^^

When creating a new field which has different physical properties to
the input field(s) the units will also need to be changed:

>>> f.units
'K'
>>> f += 2
>>> f.units
'K'

>>> f.units
'K'
>>> f **= 2
>>> f.units
'K2'

>>> f.units, g.units
('m', 's')
>>> h = f / g
>>> h.units
'm s-1'

When creating a new field which has a different domain to the input
fields, the new domain will in general contain the superset of the
axes of the two input fields, but may not have some of either input
field's auxiliary coordinates or size 1 dimension coordinates. Refer
to the field combination rules for details.


.. _floating_point_errors:

Floating point errors
^^^^^^^^^^^^^^^^^^^^^

It is possible to set the action to take when an arithmetic operation
produces one of the following floating-point errors:

.. tabularcolumns:: |l|l|
=================  =================================
Error              Description                      
=================  =================================
Division by zero   Infinite result obtained from    
                   finite numbers.

Overflow           Result too large to be expressed.

Invalid operation  Result is not an expressible     
                   number, typically indicates that 
                   a NaN was produced.

Underflow          Result so close to zero that some
                   precision was lost.
=================  =================================

For each type of error, one of the following actions may be chosen:

* Take no action. Allows invalid values to occur in the result data
  array.

* Print a `RuntimeWarning` (via the Python `warnings` module). Allows
  invalid values to occur in the result data array.

* Raise a `FloatingPointError` exception.

The treatment of floating-point errors is set with
`cf.Data.seterr`. Converting invalid numbers to masked values after an
arithmetic operation may be done with the `cf.Field.mask_invalid`
method. It is also possible to mask invalid numbers during arithmetic
operations (see `cf.Data.mask_fpe`).

Note that these setting apply to all data array arithmetic within the
`cf` package.

Operations on field components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Operating on a field component works in much the same was as operating
on the field itself:

>>> a = f.field_anc('air_temperature standard_error')
>>> a.data
<CF Data: [[[1.2, ..., 5.6]]] K>
>>> a += 2
>>> a.data
<CF Data: [[[3.2, ..., 7.6]]] K>

If the component has bounds, however, the bounds are operated on with
the same operand as for the variable's data array:

>>> x = f.coord('X')
>>> x.dump()
Dimension Coordinate: longitude
    standard_name = 'longitude'
    Data(128) = [0.0, ..., 357.1875] degrees_east
    Bounds(128, 2) = [[-1.40625, ..., 358.59375]] degrees_east
>>> (x + 2).dump()
Dimension Coordinate: longitude
    standard_name = 'longitude'
    Data(128) = [2.0, ..., 359.1875] degrees_east
    Bounds(128, 2) = [[0.59375, ..., 360.59375]] degrees_east
>>> (x + x).dump()
Dimension Coordinate: longitude
    axis = 'X'
    long_name = 'longitude'
    standard_name = 'longitude'
    Data(128) = [0.0, ..., 714.375] degrees_east
    Bounds(128, 2) = [[-1.40625, ..., 715.78125]] degrees_east

This means that cells do not change size when undergoing simple
relocation. For example, of a coordinate of 0.5 with cell bounds of
[-1, 1] has 2 added to it, the the coordinate becomes 2.5 with cell
bounds [1, 3].

Statistical operations
----------------------

Axes of a field may be collapsed by statistical methods with the
`cf.Field.collapse` method. Collapsing an axis involves reducing its
size with a given (typically statistical) method.

By default all axes with size greater than 1 are collapsed completely
with the given method. For example, to find the minumum of the data
array:

>>> g = f.collapse('min')

By default the calculations of means, standard deviations and
variances use a combination of volume, area and linear weights based
on the field's metadata. For example to find the mean of the data
array, weighted where possible:

>>> g = f.collapse('mean')

Specific weights may be forced with the weights parameter. For example
to find the variance of the data array, weighting the X and Y axes by
cell area, the T axis linearly and leaving all other axes unweighted:

>>> g = f.collapse('variance', weights=['area', 'T'])

A subset of the axes may be collapsed. For example, to find the mean
over the time axis:

>>> f
<CF Field: air_temperature(time(12), latitude(73), longitude(96) K>
>>> g = f.collapse('T: mean')
>>> g
<CF Field: air_temperature(time(1), latitude(73), longitude(96) K>

For example, to find the maximum over the time and height axes:

>>> g = f.collapse('T: Z: max')

or, equivalently:

>>> g = f.collapse('max', axes=['T', 'Z'])

An ordered sequence of collapses over different (or the same) subsets
of the axes may be specified. For example, to first find the mean over
the time axis and subequently the standard deviation over the latitude
and longitude axes:

>>> g = f.collapse('T: mean area: sd')

or, equivalently, in two steps:

>>> g = f.collapse('mean', axes='T').collapse('sd', axes='area')

Grouped collapses are possible, whereby groups of elements along an
axis are defined and each group is collapsed independently. The
collapsed groups are concatenated so that the collapsed axis in the
output field has a size equal to the number of groups. For example, to
find the variance along the longitude axis within each group of size
10 degrees:

>>> g = f.collapse('longitude: variance', group=cf.Data(10, 'degrees'))

Climatological statistics (a type of grouped collapse) as defined by
the CF conventions may be specified. For example, to collapse a time
axis into multiannual means of calendar monthly minima:

>>> g = f.collapse('time: minimum within years T: mean over years',
...                 within_years=cf.M())

In all collapses, missing data array elements are accounted for in the
calculation.

The following collapse methods are available, over any subset of the
axes:

=========================  =====================================================
Method                     Notes
=========================  =====================================================
Maximum                    The maximum of the values.
                           
Minimum                    The minimum of the values.
                                    
Sum                        The sum of the values.
                           
Mid-range                  The average of the maximum and the minimum of the
                           values.
                           
Range                      The absolute difference between the maximum and
                           the minimum of the values.
                           
Mean                       The unweighted mean, :math:`m`, of :math:`N`
                           values :math:`x_i` is
                           
                           .. math:: m=\frac{1}{N}\sum_{i=1}^{N} x_i
                           
                           The weighted mean, :math:`\tilde{m}`, of :math:`N`
                           values :math:`x_i` with corresponding weights
                           :math:`w_i` is
                           

                           .. math:: \tilde{m}=\frac{1}{\sum_{i=1}^{N} w_i}
                                               \sum_{i=1}^{N} w_i x_i
                           
Standard deviation         The unweighted standard deviation, :math:`s`, of
                           :math:`N` values :math:`x_i` with mean :math:`m`
                           and with :math:`N-ddof` degrees of freedom
                           (:math:`ddof\ge0`) is
                           
                           .. math:: s=\sqrt{\frac{1}{N-ddof}
                                       \sum_{i=1}^{N} (x_i - m)^2}
                           
                           The weighted standard deviation,
                           :math:`\tilde{s}_N`, of :math:`N` values
                           :math:`x_i` with corresponding weights
                           :math:`w_i`, weighted mean
                           :math:`\tilde{m}` and with :math:`N`
                           degrees of freedom is
                           
                           .. math:: \tilde{s}_N=\sqrt{\frac{1}
                                         {\sum_{i=1}^{N} w_i}
                                         \sum_{i=1}^{N} w_i(x_i -
                                         \tilde{m})^2}
                           
                           The weighted standard deviation,
                           :math:`\tilde{s}`, of :math:`N` values
                           :math:`x_i` with corresponding weights
                           :math:`w_i` and with :math:`N-ddof` degrees
                           of freedom :math:`(ddof>0)` is
                           
                           .. math:: \tilde{s}=\sqrt{ \frac{a
                                     \sum_{i=1}^{N} w_i}{a
                                     \sum_{i=1}^{N} w_i - ddof}}
                                     \tilde{s}_N
                           
                           where :math:`a` is the smallest positive
                           number whose product with each weight is an
                           integer. :math:`a \sum_{i=1}^{N} w_i` is
                           the size of a new sample created by each
                           :math:`x_i` having :math:`aw_i` repeats. In
                           practice, :math:`a` may not exist or may be
                           difficult to calculate, so :math:`a` is
                           either set to a predetermined value or an
                           approximate value is calculated (see
                           `cf.Field.collapse` for details).
                           
Variance                   The variance is the square of the standard
                           deviation.
                           
Sample size                The sample size, :math:`N`, as would be used for 
                           other statistical calculations.
                           
Sum of weights             The sum of sample weights,
                           :math:`\sum_{i=1}^{N} w_i`, as would be
                           used for other statistical calculations.

Sum of squares of weights  The sum of squares of sample weights,
                           :math:`\sum_{i=1}^{N} {w_i}^{2}`,
                           as would be used for other statistical
                           calculations.
=========================  =====================================================

Any collapse method that involves a calculation (such as calculating a
mean), as opposed to just selecting a value (such as finding a
maximum), will return a field containing double precision floating
point numbers or, if all of the input data are integers, double
precision integers. If this is not desired, then the datatype can be
reset after the collapse:

   >>> g = f.collapse('T: mean')
   >>> g.dtype = f.dtype
   >>> h = f.collapse('area: variance')
   >>> h.dtype = 'float32'
   
See `cf.Field.collapse` for more details.

Regridding operations
---------------------

A field may be regridded onto a new latitude-longitude grid:

>>> f
<CF Field: air_temperature(time(12), latitude(73), longitude(96) K>
>>> g
<CF Field: precipitation(time(24), longitude(128), latitude(64)) kg m-2 s-1>
>>> h = f.regrids(g)
>>> h
<CF Field: air_temperature(time(12), longitude(128), latitude(64) K>

By default the interpolation is first-order conservative, but bilinear
interpolation is also possible. The missing data masks of the field
and the new grid are aslo taken into account. 

See `cf.Field.regrids` for more details.

.. _units:

Units
-----

A field (as well as any other object which :ref:`inherits
<inheritance_diagrams>` from `cf.Variable`) always contains a
`cf.Units` object which gives the physical units of the values
contained in its data array.

The `cf.Units` object is stored in the field's `~Field.Units`
attribute but may also be accessed through the field's `~Field.units`
and `~Field.calendar` CF properties, which may take any value allowed
by the `CF conventions
<http://cf-pcmdi.llnl.gov/documents/cf-conventions/latest-cf-conventions-document-1>`_. In
particular, the value of the `~Field.units` CF property is a string
that can be recognized by `UNIDATA's Udunits-2 package
<http://www.unidata.ucar.edu/software/udunits/>`_, with a few
exceptions for greater consistency with CF. These are detailed by the
`cf.Units` object.

Assignment
^^^^^^^^^^

The Field's units may be assigned directly to its `cf.Units` object:

>>> f.Units.units = 'days since 1-1-1'
>>> f.Units.calendar = 'noleap'

>>> f.Units = cf.Units('metre')

But the same result is achieved by assigning to the field's
`~Field.units` and `~Field.calendar` CF properties:

>>> f.units = 'days since 1-1-1'
>>> f.calendar = 'noleap'
>>> f.Units
<CF Units: days since 1-1-1 calendar=noleap>
>>> f.units
'days since 1-1-1'
>>> f.calendar
'noleap'


Time units
^^^^^^^^^^

Time units may be given as durations of time or as an amount of time
since a reference time:

>>> f.units = 'day'
>>> f.units = 'seconds since 1992-10-8 15:15:42.5 -6:00'

.. note::

   It is recommended that the units ``'year'`` and ``'month'`` be used
   with caution, as explained in the following excerpt from the CF
   conventions: "The Udunits package defines a year to be exactly
   365.242198781 days (the interval between 2 successive passages of
   the sun through vernal equinox). It is not a calendar year. Udunits
   includes the following definitions for years: a common_year is 365
   days, a leap_year is 366 days, a Julian_year is 365.25 days, and a
   Gregorian_year is 365.2425 days. For similar reasons the unit
   ``'month'``, which is defined to be exactly year/12, should also be
   used with caution."

The date given in reference time units is always associated with one
of the calendars recognized by the CF conventions and may be set with
the *calendar* CF property (on the field or Units object).

If the calendar is not set then, as in the CF conventions, for the
purposes of calculation and comparison, it defaults to the mixed
Gregorian/Julian calendar as defined by Udunits:

>>> f.units = 'days since 2000-1-1'
>>> f.calendar
AttributeError: Can't get 'Field' attribute 'calendar'
>>> g.units = 'days since 2000-1-1'
>>> g.calendar = 'gregorian'
>>> g.Units.equals(f.Units)
True

The calendar is ignored for units other than reference time units.

Changing units
^^^^^^^^^^^^^^

Changing units to equivalent units causes the variable's data array
values to be modified in place (if required) when they are next
accessed, and not before:

>>> f.units
'metre'
>>> f.array
array([    0.,  1000.,  2000.,  3000.,  4000.])
>>> f.units = 'kilometre'
>>> f.units
'kilometre'
>>> f.array
array([ 0.,  1.,  2.,  3.,  4.])

>>> f.units
'hours since 2000-1-1'
>>> f.array
array([-1227192., -1227168., -1227144.])
>>> f.units = 'days since 1860-1-1'
>>> f.array
array([ 1.,  2.,  3.])

The `cf.Units` object may be operated on with augmented arithmetic
assignments and binary arithmetic operations:

>>> f.units
'kelvin'
>>> f.array
array([ 273.15,  274.15,  275.15,  276.15,  277.15])

>>> f.Units -= 273.15
>>> f.units
'K @ 273.15'
>>> f.array
array([ 0.,  1.,  2.,  3.,  4.])

>>> f.Units = f.Units + 273.15
>>> f.units
'K'
>>> f.array
array([ 273.15,  274.15,  275.15,  276.15,  277.15])

>>> f.units = 'K @ 237.15'
'K @ 273.15'
>>> f.array
array([ 0.,  1.,  2.,  3.,  4.])

If the field has a data array and its units are changed to
non-equivalent units then a :py:mod:`TypeError` will be raised when
the data are next accessed:

>>> f.units
'm s-1'
>>> f.units = 'K'
>>> f.array
TypeError: Units are not convertible: <CF Units: m s-1>, <CF Units: K>

Overriding units
^^^^^^^^^^^^^^^^

If the units are incorrect, either due to a data manipulation or
an incorrect encoding, it is possible to replace the existing units with
new units, which don't have to be equivalent, without altering the
data values:

>>> f.units
'mm/day'
>>> f.mean()
<CF Data: 3.3455467 mm/day>
>>> g = f.override_units('kg m-2 s-1')
>>> g.mean()
<CF Data: 3.3455467 kg m-2 s-1>
>>> g.override_units('watts m-2', i=True)
>>> g.mean()
<CF Data: 3.3455467 watts m-2>

Overriding the calendar of reference time units is done in a similar manner:

>>> f.calendar
'360_day'
>>> f.array.min()
59.0
>>> f.min()
<CF Data: 1960-02-30 00:00:00 360_day>
>>> g = f.override_calandar('gregorian')
>>> g.array.min()
59.0
>>> g.min()
<CF Data: 1960-02-29 00:00:00 gregorian>

Note that in this case the data values have remained unchanged, but
their date-time interpretation has been redefined.

See `cf.Field.override_units` and `cf.Field.override_calendar` for details.


Equality and equivalence of units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `cf.Units` object has methods for assessing whether two units are
equivalent or equal, regardless of their exact string representations.

Two units are equivalent if and only if numeric values in one unit are
convertible to numeric values in the other unit (such as
``'kilometres'`` and ``'metres'``). Two units are equal if and only if
they are equivalent and their conversion is a scale factor of 1 (such
as ``'kilometres'`` and ``'1000 metres'``). Note that equivalence and
equality are based on internally stored binary representations of the
units, rather than their string representations.

>>> f.units = 'm/s'

>>> g.units = 'm s-1'
>>> f.Units == g.Units
True
>>> f.Units.equals(g.Units)
True

>>> g.units = 'km s-1'
>>> f.Units.equivalent(g.Units)
False

>>> f.units = 'days since 1987-12-3'
>>> g.units = 'hours since 2000-12-1'	
>>> f.Units == g.Units
False
>>> f.Units.equivalent(g.Units)
True


Bounds units
^^^^^^^^^^^^

The units of variables with cell bounds (i.e. coordinates and domain
ancillaries) are always the same as the coordinate itself, and the
units of the bounds automatically change when a variable's units are
changed:

>>> c.units
'degrees'
>>> c.bounds.units
'degrees'
>>> print c.bounds.array
[  0.  90.]
>>> c.units = 'radians'
>>> c.bounds.units
'radians'
>>> print c.bounds.array
[ 0.  1.57079633]

Manipulating other variables
----------------------------

A field is a subclass of `cf.Variable`, and that class and other
subclasses of `cf.Variable` share generally similar behaviours and
methods:

========================  ===============================================
Class		          Description
========================  ===============================================
`cf.AuxiliaryCoordinate`  A CF auxiliary coordinate construct.

`cf.CellMeasure`          A CF cell measure construct containing
                          information that is needed about the size,
                          shape or location of the field's cells.
		          
`cf.Coordinate`           Base class for storing a coordinate.

`cf.DimensionCoordinate`  A CF dimension coordinate construct.
		          
`cf.Variable`             Base class for storing a data array with
                          metadata.
========================  ===============================================

In general, different axis identities, different axis orders and
different axis directions are not considered, since these objects do
not contain a coordinate system required to define these properties
(unlike a field).

Coordinates
^^^^^^^^^^^

Coordinates are a special case as they may contain a data array for
their coordinate bounds which needs to be treated consistently with
the main coordinate array. If a coordinate has bounds then all
coordinate methods also operate on the bounds in a consistent manner:

>>> c
<CF Coordinate: latitude(73, 96)>
>>> c.bounds
<CF Bounds: (73, 96, 4)>
>>> d = c[0:10]
>>> d.shape
(10, 96)
>>> d.bounds.shape
(10, 96, 4)
>>> d.transpose([1, 0])
>>> d.shape
(96, 10)
>>> d.bounds.shape
(96, 10, 4)

.. note:: 

   If the coordinate bounds are operated on independently, care should
   be taken not to break consistency with the parent coordinate.

