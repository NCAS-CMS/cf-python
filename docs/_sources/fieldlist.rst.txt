.. currentmodule:: cf
.. default-role:: obj

.. _fs_field_list:

The `cf.FieldList` object
=========================

A `cf.FieldList` is an ordered sequence of `cf.Field` objects.

List-like operators
^^^^^^^^^^^^^^^^^^^

It supports all of the :ref:`python list-like operations
<python:typesseq-mutable>`. For example:

>>> fl
[<CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>]
>>> fl[0]
<CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>
>>> fl[::-1] 
[<CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>,
 <CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>]
>>> fl[slice(1, -1, 2)]
[<CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>]

Note that an indexing by an integer returns an individual field, but
other types of index always return a field list.

>>> len(fl)
2
>>> f = fl.pop()
>>> f
<CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>
>>> len(fl)
1
>>> fl.append(f.copy())
>>> len(fl)
2
>>> f in fl
True
>>> fl
[<CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>]
>>> fl.reverse()
[<CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>,
 <CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>]
>>> fl += fl[-1].copy()
[<CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>,
 <CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>]

Selecting fields
^^^^^^^^^^^^^^^^

One or more fields from a field list may be selected with the
`~cf.FieldList.select` method that returns a new field list containing
the selected fields:

>>> fl
[<CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>]
>>> fl.select('air_temperature')
<CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>]
>>> fl.select('[air_temperature|x_wind]')
[<CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>]
>>> fl.select('NOTHING')
[]
 
Manipulating the fields
^^^^^^^^^^^^^^^^^^^^^^^

For in-place changes, a :ref:`for loop <python:for>` may be used to
process each field element. For example to reverse the data array axis
order of each field in-place:

>>> fl
[<CF Field: x_wind(grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>]
>>> for f in fl:
...     f.transpose(i=True)
...
>>> fl
[<CF Field: x_wind(grid_longitude(106), grid_latitude(110)) m s-1>,
 <CF Field: air_temperature( longitude(96), latitude(73), time(12)) K>]

In-place changes to the fields may also be done with a list
comprehension:

>>> [f.transpose(i=True) for f in fl]

For changes which result in new fields, a for loop or :ref:`list
comprehension <python:tut-listcomps>` may be used:

>>> new_fl = cf.FieldList([f.round() for f in fl])
>>> new_fl = cf.FieldList()
>>> for f in fl:
...     new_fl.append(f[0:3])
...
>>> new_fl
[<CF Field: x_wind(grid_longitude(3), grid_latitude(110)) m s-1>,
 <CF Field: air_temperature( longitude(3), latitude(73), time(12)) K>]


Sorting the fields
^^^^^^^^^^^^^^^^^^

The field list may be sorted in-place with the `~cf.FieldList.sort`
method, that works in a similar way to the `!sort` method of a
built-in list. The only difference is that by default the fields are
sorted by their identities (e.g their standard names). For example:

>>> fl
[<CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: ocean_meridional_overturning_streamfunction(time(12), region(4), depth(40), latitude(180)) m3 s-1>,
 <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>,
 <CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>]
>>> fl.sort()
>>> fl
[<CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>,
 <CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: ocean_meridional_overturning_streamfunction(time(12), region(4), depth(40), latitude(180)) m3 s-1>]
>>> fl.sort(reverse=True)
>>> fl
[<CF Field: ocean_meridional_overturning_streamfunction(time(12), region(4), depth(40), latitude(180)) m3 s-1>,
 <CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>,
 <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>]
