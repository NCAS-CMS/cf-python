.. currentmodule:: cf
.. default-role:: obj

.. _one_to_two_changes:
		  
Incompatible differences between versions 1.x and 2.x
=====================================================

For those familiar with the cf-python API at version 1.x, some
important, backwards incompatible changes were introduced at version
2.0.

Some of these changes could break code written at version 1.x, causing
an exception to be raised. For others, those marked with a warning,
version 1.x may work but could produce scientifically different
results.

All of the changes have been designed to make the interface more
consistent and intuitive to use and were introduced at version 2.0 to
coincide with the updated CF data model structure.

attributes
----------

.. note:: At version 2.x `cf.Field.attributes` is callable and allows
          the attributes to be updated inplace.

          >>> cf.__version__
          2.0
          >>> f.attributes()
	  {'ncvar': 'tas'}

At version 1.x it wasn't callable and returned a `!dict` copy of the
attributes.

.. _axes:

axes
----

.. note:: At version 2.x `cf.Field.axes` returns, by default, a
          :py:obj:`dict`. If the `!ordered` parameter is `True` then
          it returns an :py:obj:`~collections.OrderedDict`.

          >>> cf.__version__
          2.0
          >>> f
          <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
          >>> f.axes()
          {'dim0': <CF DomainAxis: 12>,
           'dim1': <CF DomainAxis: 64>,
           'dim2': <CF DomainAxis: 128>,
           'dim3': <CF DomainAxis: 1>}
          >>> f.axes(ordered=True)
          OrderedDict([('dim3', <CF DomainAxis: 1>),
                       ('dim0', <CF DomainAxis: 12>),
                       ('dim1', <CF DomainAxis: 64>),
                       ('dim2', <CF DomainAxis: 128>)])

At version 1.x it returned a `!set` by default or, if the `!ordered`
parameter was `True`, it returned a `!list`.

.. _axis:

axis
----

.. note:: At version 2.x `cf.Field.axis` returns, by default, a
          `cf.DomainAxis` object. A domain axis identifier (such as
          "dim2") may be returned by setting the `!key` parameter to
          `True`.

          >>> cf.__version__
          2.0
          >>> f
          <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
          >>> f.axis('X')
          <CF DomainAxis: 128>
          >>> f.axis('X', key=True)
          'dim2'

At version 1.x it always returned a domain axis identifier.

.. _bounds:

bounds
------

.. warning:: Running code written at version 1.x with the version 2.x
             library could produce scientifically different results.

.. note:: At version 2.x performing arithmetic on coordinates also
          modifies the coordinate bounds, if present.

          >>> cf.__version__
          2.0
          >>> f
          <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
          >>> t = f.coord('T')
	  >>> print t.array[0], t.bounds.array[0]
          164569.0 [ 164554.  164584.]
          >>> t -= 1000000
	  >>> print t.array[0], t.bounds.array[0]
          -835431.0 [-835446. -835416.]

	  If the bounds are not to be changed, just the coordinate
	  values, then the arithmetic may be applied to the coordinate
	  object's data array.

	  >>> t.data += 1000000
	  >>> print t.array[0], t.bounds.array[0]
          164569.0 [-835446. -835416.]

At version 1.x performing arithmetic on coordinates did not modify the
coordinate bounds - they had to be modified seperately.

.. _collapse:

collapse
--------

.. warning:: Running code written at version 1.x with the version 2.x
             library could produce scientifically different results.

.. note:: At version 2.x `cf.Field.collapse` does not, by default,
          weight the calculations, i.e. the `!weights` parameter
          defaults to `None`.

          >>> cf.__version__
          2.0
          >>> f
          <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
          >>> g = f.collapse('mean')
          >>> g.datum()
          279.1922
          >>> g = f.collapse('mean', weights=None)
          >>> g.datum()
          279.1922

          Non-equal weighting has to be specifically requested with
          the `!weights` parameter.
          
          >>> h = f.collapse('mean', axes=['X', 'Y', 'T'], weights=['T', 'area'])
          >>> h.datum()
          288.6733

At version 1.x the `!weights` parameter defaulted to ``'auto'``,
meaning that calculations were were weighted according to metadata
present in the field.

.. _dump:

dump
----

.. note:: At version 2.x the output of `cf.Field.dump` has been
          reformatted.

.. _FieldList:

FieldList
---------

.. warning:: Running code written at version 1.x with the version 2.x
             library could produce scientifically different results.

.. note:: At version 2.x `cf.FieldList` is not a subclass of
          `cf.Field`, therefore `cf.FieldList` does not inherit any
          methods from `cf.Field`.

          >>> cf.__version__
          2.0
          >>> fl
          [<CF Field: specific_humidity(latitude(73), longitude(96)) K>,
           <CF Field: air_pressure(height(17), latitude(145), longitude(196)) K>]
          >>> isinstance(fl, cf.Field)
	  False
          >>> gl = fl.collapse('mean')
          DeprecationError: collapse method has been removed from a field list. Use on individual fields.
	  >>> gl = cf.FieldList([f.collapse('mean') for f in fl])

At version 1.x `cf.FieldList` was a subclass of `cf.Field` and
inherited all of the methods that returned a `cf.Field` when used on a
field (such as `~cf.Field.collapse` and `~cf.Field.squeeze`).

.. _read:

read
----

.. note:: At version 2.x `cf.read` always returns a `cf.FieldList`.

          >>> cf.__version__
          2.0
          >>> fl = cf.read('file[12].nc')
          >>> fl
          [<CF Field: specific_humidity(latitude(73), longitude(96)) K>,
           <CF Field: air_pressure(height(17), latitude(145), longitude(196)) K>]
          >>> fl = cf.read('file3.nc')
          >>> fl
          [<CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>]

          The new function `cf.read_field` will always return a `cf.Field`
          if there is only one identified in the input file(s).
          
          >>> f = cf.read_field('file3.nc')
          >>> f
          <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>

At version 1.x `cf.read` returned a `cf.Field` if only one field was
found in the input files(s), otherwise it returned a `cf.FieldList`.

.. _regridc:

regridc
-------

.. note:: At version 2.x `cf.Field.regridc` the regridding method must
          be specified. The `!method` parameter does not have an
          ``'auto'`` option.

	  >>> cf.__version__
          2.0
          >>> f
          <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
          >>> g
          <CF Field: specific_humidity(time(360), latitude(73), longitude(96)) K>
          >>> h = f.regridc(g, 'time', 'nearest_stod')
	  >>> h
          <CF Field: air_temperature(time(360), latitude(64), longitude(128)) K>
	  
At version 1.x the regridding method defaulted to ``'auto'``, meaning
that the method was inferred according to metadata present in the
field.

.. _regrids:

regrids
-------

.. note:: At version 2.x `cf.Field.regrids` the regridding method must
          be specified. The `!method` parameter does not have an
          ``'auto'`` option.

	  >>> cf.__version__
          2.0
          >>> f
          <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
          >>> g
          <CF Field: specific_humidity(time(360), latitude(73), longitude(96)) K>
          >>> h = f.regrids(g, 'conservative')
	  >>> h
          <CF Field: air_temperature(time(12), latitude(73), longitude(96)) K>
	  
At version 1.x the regridding method defaulted to ``'auto'``, meaning
that the method was inferred according to metadata present in the
field.

properties
----------

.. note:: At version 2.x `cf.Field.properties` is callable and allows
          the CF properties to be updated inplace.

          >>> cf.__version__
          2.0
          >>> f.properties()
          {'Conventions': 'CF-1.6',
	   'experiment_id': 'pre-industrial control experiment',
	   'project_id': 'IPCC Fourth Assessment',
	   'realization': 1,
	   'standard_name': 'air_temperature',
	   'title': 'model output prepared for IPCC AR4'}

At version 1.x it wasn't callable and returned a `!dict` copy of the
CF properties.

.. _select:

select
------

.. note:: At version 2.x `cf.Field.select` has been removed. Use
          `cf.Field.match` to see if an individual field meets given
          criteria.

          >>> cf.__version__
          2.0
          >>> f
          <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
          >>> f.match('air_temperature')
          True

          `cf.FieldList.select` always returns another field list.

          >>> fl
          [<CF Field: specific_humidity(latitude(73), longitude(96)) K>,
           <CF Field: air_pressure(height(17), latitude(145), longitude(196)) K>]
          >>> fl.select('specific_humidity|air_pressure')
          [<CF Field: specific_humidity(latitude(73), longitude(96)) K>,
           <CF Field: air_pressure(height(17), latitude(145), longitude(196)) K>]
          >>> fl.select('specific_humidity')
          [<CF Field: specific_humidity(latitude(73), longitude(96)) K>]
          >>> fl.select('ocean_meridional_overturning_streamfunction')
          []

          The new method `cf.FieldList.select_field` will always return a
          `cf.Field` if there is only one identified in the field
          list.

          >>> fl.select_field('specific_humidity')
          <CF Field: specific_humidity(latitude(73), longitude(96)) K>

At version 1.x `cf.FieldList.select` returned a `cf.Field` if only one
field element was found in the field list, otherwise it returned a
`cf.FieldList`. `cf.Field.select` either returned the field itself or
an empty `cf.FieldList`.

.. COMMENTED OUT
   .. _subspace:
   
   subspace
   --------
   
   .. warning:: Running code written at version 1.x with the version 2.x
                library could produce scientifically different results.
      
   .. note:: At version 2.x indexing on a field object returns a subspace
             of the field, in much the same way that a `numpy` array is
             subspaced by indexing. This means that there are now two
             equivalent ways to subspace a field in index-space: by
             indexing the field directly or by indexing the
             `cf.Field.subspace` attribute.
   	  
             >>> cf.__version__
             2.0
             >>> f
             <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
             >>> g = f[::-2, 0, 28:]
             >>> g
             <CF Field: air_temperature(time(6), latitude(1), longitude(100)) K>
             >>> h = f.subspace[::-2, 0, 28:]
             >>> h
             <CF Field: air_temperature(time(6), latitude(1), longitude(100)) K>
             >>> g.equals(h)
             True
             
             Assignment to a subspace may be done either to a subspace
             defined by direct indexing or to one defined by indexing the
             `cf.Field.subspace` attribute.
             
             >>> g = f.copy()
             >>> g[0] = -99
             >>> h = f.copy()
             >>> h.subspace[0] = -99
             >>> g.equals(h)
             True
   
             Calling the `cf.Field.subspace` attribute to subspace the
             field in domain-space still works in the same way.
   
             >>> f.subspace(longitude=cf.gt(90), Y=0)
             <CF Field: air_temperature(time(12), latitude(1), longitude(96)) K>
   
             The `cf.FieldList.subspace` method has been removed.
   
   At version 1.x direct indexing on a field returned the itself, with no
   subspacing.

Indexing a field
----------------

.. warning:: Running code written at version 1.x with the version 2.x
             library could produce scientifically different results.

.. note:: At version 2.x indexing on a field object returns a subspace
          of the field, in exactly the way that indexing the
          `cf.Field.subspace` attribute does.
	  
          >>> cf.__version__
          2.0
          >>> f
          <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
          >>> g = f[::-2, 0, 28:]
          >>> g
          <CF Field: air_temperature(time(6), latitude(1), longitude(100)) K>
          >>> h = f.subspace[::-2, 0, 28:]
          >>> h
          <CF Field: air_temperature(time(6), latitude(1), longitude(100)) K>
          >>> g.equals(h)
          True
          >>> f.equals(f[0])          
	  False

          Assignment to a subspace of the data array may either be
          done to a subspace defined by direct indexing or to one
          defined by indexing the `cf.Field.subspace` attribute.
          
          >>> g = f.copy()
          >>> g[0] = -99
          >>> h = f.copy()
          >>> h.subspace[0] = -99
          >>> g.equals(h)
          True

At version 1.x direct indexing on a field returned the field itself,
with no subspacing.

          >>> cf.__version__
          1.5
          >>> f
          <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
          >>> f.equals(f[0])
          True
