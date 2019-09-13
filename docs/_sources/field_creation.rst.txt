.. currentmodule:: cf
.. default-role:: obj

.. _field_creation:
 
Creating `cf.Field` objects
===========================

A new field may be created by initializing a new `cf.Field` instance
with properties and a data array:

=======================  =============================================================
Keyword                  Description
=======================  =============================================================
``attributes``           Provide the new field with non-CF attributes in a dictionary
``data``                 Provide the new field with a data array in a `cf.Data` object
``flags``                Provide the new field with self-describing flag values in a
                         `cf.Flags` object
``properties``           Provide the new field with CF properties in a dictionary
=======================  =============================================================


Other metadata items (coordinate, cell methods, etc.) are then
provided with bespoke methods:

.. autosummary::
   :nosignatures:
   :template: method.rst

   cf.Field.insert_aux
   cf.Field.insert_axis
   cf.Field.insert_cell_methods
   cf.Field.insert_domain_anc
   cf.Field.insert_data
   cf.Field.insert_dim
   cf.Field.insert_field_anc
   cf.Field.insert_measure
   cf.Field.insert_ref

For example:

>>> coord
<CF DimensionCoordinate: time(12) days since 2003-12-1>
>>> f.insert_dim(coord)

Removing field components is done with the following methods:

.. autosummary::
   :nosignatures:
   :template: method.rst

   cf.Field.remove_axis
   cf.Field.remove_axes
   cf.Field.remove_data
   cf.Field.remove_item
   cf.Field.remove_items

For example:

>>> f.remove_item('forecast_reference_time')


.. _field-creation_examples:

Examples
--------

To improve readability, it is recommended that the construction of a
field is done by first creating the components separately (data,
coordinates, properties, *etc.*), and then combining them to make the
field (as in :ref:`example 3 <fc-example3>` and :ref:`example 4
<fc-example4>`), although this may not be necessary for very simple
fields (as in :ref:`example 1 <fc-example1>` and :ref:`example 2
<fc-example2>`).

.. _fc-example1:

Example 1
~~~~~~~~~

An empty field:

>>> f = cf.Field()
>>> print f
Field: 
-------

.. _fc-example2:

Example 2
~~~~~~~~~

A field with just CF properties:

>>> f = cf.Field(properties={'standard_name': 'air_temperature',
...                          'long_name': 'temperature of air'})
...
>>> print f
Field: air_temperature
----------------------

.. _fc-example3:

Example 3
~~~~~~~~~

A field with a simple domain. Note that in this example the data and
coordinates are generated using :py:obj:`range` and `numpy.arange`
simply for the sake of having some numbers to play with. In practice
it is likely the values would have been read from a file in some
arbitrary format:

>>> import numpy
>>> data = cf.Data(numpy.arange(90.).reshape(10, 9), 'm s-1')
>>> properties = {'standard_name': 'eastward_wind'}
>>> dim0 = cf.DimensionCoordinate(data=cf.Data(range(10), 'degrees_north'),
...                               properties={'standard_name': 'latitude'})
>>> dim1 = cf.DimensionCoordinate(data=cf.Data(range(9), 'degrees_east'))
>>> dim1.standard_name = 'longitude'
>>> f = cf.Field(properties=properties)
>>> f.insert_dim(dim0)
>>> f.insert_dim(dim1)
>>> f.insert_data(data)
>>> print f
Field: eastward_wind
--------------------
Data           : eastward_wind(latitude(10), longitude(9)) m s-1
Axes           : latitude(10) = [0, ..., 9] degrees_north
               : longitude(9) = [0, ..., 8] degrees_east

Adding an auxiliary coordinate to the "latitude" axis and a cell
method may be done with the relevant methods (note that these
coordinate values are just for illustration):

>>> aux = cf.AuxiliaryCoordinate(data=cf.Data(['alpha','beta','gamma','delta','epsilon',
...                                            'zeta','eta','theta','iota','kappa']))
>>> aux.long_name = 'extra'
>>> f.insert_aux(aux, axes=['dim0'])
>>> f.insert_cell_methods('latitude: point')
>>> f.long_name = 'wind' 
>>> print f
Field: eastward_wind
--------------------
Data           : eastward_wind(latitude(10), longitude(9)) m s-1
Cell methods   : latitude: point
Axes           : latitude(10) = [0, ..., 9] degrees_north
               : longitude(9) = [0, ..., 8] degrees_east
Aux coords     : long_name:extra(latitude(10)) = [alpha, ..., kappa]

Removing the auxiliary coordinate and the cell method that were just
added is also done with a method and by simple deletion:

>>> f.remove_item({'long_name': 'extra'})
>>> del f.cell_methods
>>> print f
Field: eastward_wind
--------------------
Data           : eastward_wind(latitude(10), longitude(9)) m s-1
Axes           : latitude(10) = [0, ..., 9] degrees_north
               : longitude(9) = [0, ..., 8] degrees_east

.. _fc-example4:

Example 4
~~~~~~~~~

.. highlight:: guess

A more complicated field is created by the following script. Note that
in this example the data and coordinates are generated using
`numpy.arange` simply for the sake of having some numbers to play
with. In practice it is likely the values would have been read from a
file in some arbitrary format::

   import numpy
   import cf

  #---------------------------------------------------------------------
   # 1. CREATE the field's domain items
   #---------------------------------------------------------------------
   # Create a grid_latitude dimension coordinate
   Y = cf.DimensionCoordinate(properties={'standard_name': 'grid_latitude'},
                              data=cf.Data(numpy.arange(10.), 'degrees'))
   
   # Create a grid_longitude dimension coordinate
   X = cf.DimensionCoordinate(data=cf.Data(numpy.arange(9.), 'degrees'))
   X.standard_name = 'grid_longitude'
   
   # Create a time dimension coordinate (with bounds)
   bounds = cf.Bounds(data=cf.Data([0.5, 1.5],
                                   cf.Units('days since 2000-1-1', calendar='noleap')))
   T = cf.DimensionCoordinate(properties=dict(standard_name='time'),
                              data=cf.Data(1, cf.Units('days since 2000-1-1',
                                                       calendar='noleap')),
                              bounds=bounds)
   
   # Create a longitude auxiliary coordinate
   lat = cf.AuxiliaryCoordinate(data=cf.Data(numpy.arange(90).reshape(10, 9),
                                             'degrees_north'))
   lat.standard_name = 'latitude'
   
   # Create a latitude auxiliary coordinate
   lon = cf.AuxiliaryCoordinate(properties=dict(standard_name='longitude'),
                                data=cf.Data(numpy.arange(1, 91).reshape(9, 10),
                                             'degrees_east'))
   
   # Create a rotated_latitude_longitude grid mapping coordinate reference
   grid_mapping = cf.CoordinateReference('rotated_latitude_longitude',
                                         parameters={
                                             'grid_north_pole_latitude': 38.0,
                                             'grid_north_pole_longitude': 190.0})
   
   #---------------------------------------------------------------------
   # 3. Create the field
   #---------------------------------------------------------------------
   # Create CF properties
   properties = {'standard_name': 'eastward_wind',
                 'long_name'    : 'East Wind'}
   
   # Create the field's data array
   data = cf.Data(numpy.arange(90.).reshape(9, 10), 'm s-1')
   
   # Finally, create the field
   f = cf.Field(properties=properties)
   
   f.insert_cell_methods('latitude: point')
   
   f.insert_dim(T)
   f.insert_dim(X)
   f.insert_dim(Y)
   
   f.insert_aux(lat)
   f.insert_aux(lon)
   
   f.insert_ref(grid_mapping)
   
   f.insert_data(data)
   
   print_f = str(f)
   
   self.assertTrue(
   
   print f

.. highlight:: none

Running this script produces the following output::

   The new field:

   Field: eastward_wind
   --------------------
   Data           : eastward_wind(grid_longitude(9), grid_latitude(10)) m s-1
   Cell methods   : latitude: point
   Axes           : time(1) = [2000-01-02T00:00:00Z] noleap
                  : grid_longitude(9) = [0.0, ..., 8.0] degrees
                  : grid_latitude(10) = [0.0, ..., 9.0] degrees
   Aux coords     : latitude(grid_latitude(10), grid_longitude(9)) = [[0, ..., 89]] degrees_north
                  : longitude(grid_longitude(9), grid_latitude(10)) = [[1, ..., 90]] degrees_east
   Coord refs     : rotated_latitude_longitude

.. highlight:: guess

.. _fc-example5:

Example 5
~~~~~~~~~

:ref:`Example 4 <fc-example4>` would be slightly more complicated if
the ``grid_longitude`` and ``grid_latitude`` axes were to have the
same size. In this case the field needs be told which axes, and in
which order, are spanned by the two dimensional auxiliary coordinates
(``latitude`` and ``longitude``) and the field needs to know which
axes span the data array::
 
   import numpy
   import cf

   import cf
   import numpy
   
   #---------------------------------------------------------------------
   # 1. CREATE the field's domain items
   #---------------------------------------------------------------------
   # Create a grid_latitude dimension coordinate
   Y = cf.DimensionCoordinate(properties={'standard_name': 'grid_latitude'},
                              data=cf.Data(numpy.arange(10.), 'degrees'))
   
   # Create a grid_longitude dimension coordinate
   X = cf.DimensionCoordinate(data=cf.Data(numpy.arange(10.), 'degrees'))
   X.standard_name = 'grid_longitude'

   # Create a time dimension coordinate (with bounds)
   bounds = cf.Bounds(data=cf.Data([0.5, 1.5],
                                   cf.Units('days since 2000-1-1', calendar='noleap')))
   T = cf.DimensionCoordinate(properties=dict(standard_name='time'),
                              data=cf.Data(1, cf.Units('days since 2000-1-1',
                                                       calendar='noleap')),
                              bounds=bounds)
   
   # Create a longitude auxiliary coordinate
   lat = cf.AuxiliaryCoordinate(data=cf.Data(numpy.arange(100).reshape(10, 10),
                                             'degrees_north'))
   lat.standard_name = 'latitude'
   
   # Create a latitude auxiliary coordinate
   lon = cf.AuxiliaryCoordinate(properties=dict(standard_name='longitude'),
                                data=cf.Data(numpy.arange(1, 101).reshape(10, 10),
                                             'degrees_east'))
   
   # Create a rotated_latitude_longitude grid mapping coordinate reference
   grid_mapping = cf.CoordinateReference('rotated_latitude_longitude',
                                         parameters={
                                             'grid_north_pole_latitude': 38.0,
                                             'grid_north_pole_longitude': 190.0})

   #---------------------------------------------------------------------
   # 3. Create the field
   #---------------------------------------------------------------------
   # Create CF properties
   properties = {'standard_name': 'eastward_wind',
                 'long_name'    : 'Eastward Wind'}
   
   # Create the field's data array
   data = cf.Data(numpy.arange(100.).reshape(10, 10), 'm s-1')
   
   # Finally, create the field
   f = cf.Field(properties=properties)

   f.insert_cell_methods('latitude: point')

   f.insert_dim(T)
   f.insert_dim(X)
   f.insert_dim(Y)

   f.insert_aux(lat, axes=['Y', 'X'])
   f.insert_aux(lon, axes=['X', 'Y'])

   f.insert_ref(grid_mapping)

   f.insert_data(data, axes=['Y', 'X'])

   print f

.. highlight:: none

Running this script produces the following output::

   field: eastward_wind
   --------------------
   Data           : eastward_wind(grid_latitude(10), grid_longitude(10)) m s-1
   Cell methods   : latitude: point
   Axes           : time(1) = [2000-01-02T00:00:00Z] noleap
                  : grid_latitude(10) = [0.0, ..., 9.0] degrees
                  : grid_longitude(10) = [0.0, ..., 9.0] degrees
   Aux coords     : latitude(grid_latitude(10), grid_longitude(10)) = [[0, ..., 99]] degrees_north
                  : longitude(grid_longitude(10), grid_latitude(10)) = [[1, ..., 100]] degrees_east
   Coord refs     : rotated_latitude_longitude
   
.. highlight:: guess
