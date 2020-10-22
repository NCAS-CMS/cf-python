import logging

from numpy import size as numpy_size

import cfdm

from . import mixin

from .constructs import Constructs
from .data import Data

from .functions import parse_indices

from .decorators import (_inplace_enabled,
                         _inplace_enabled_define_and_cleanup,
                         _manage_log_level_via_verbosity)

logger = logging.getLogger(__name__)


class Domain(mixin.FieldDomainMixin,
             mixin.Properties,
             cfdm.Domain):
    '''A domain construct of the CF data model.

    The domain represents a set of discrete "locations" in what
    generally would be a multi-dimensional space, either in the real
    world or in a model's simulated world. The data array elements of
    a field construct correspond to individual location of a domain.

    The domain construct is defined collectively by the following
    constructs of the CF data model: domain axis, dimension
    coordinate, auxiliary coordinate, cell measure, coordinate
    reference, and domain ancillary constructs; as well as properties
    to describe the domain.

    '''
    def __new__(cls, *args, **kwargs):
        '''TODO

        '''
        instance = super().__new__(cls)
        instance._Data = Data
        instance._Constructs = Constructs
        return instance

    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)

    # ----------------------------------------------------------------
    # Private attributes
    # ----------------------------------------------------------------
    @property
    def _cyclic(self):
        '''Storage for axis cyclicity

        '''
        return self._custom.get('_cyclic', set())

    @_cyclic.setter
    def _cyclic(self, value): self._custom['_cyclic'] = value

    @_cyclic.deleter
    def _cyclic(self): del self._custom['_cyclic']

    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------
    def _parse_axes(self, axes):
        '''Conform axes.

    :Parameters:

        axes: (sequence of) `str`
            TODO

    :Returns:

        `list`
            The conformed axes.

        '''
        if axes is None:
            return axes

        if isinstance(axes, str):
            axes = (axes,)

        domain_axes_keys = self.domain_axes.keys()

        out = []
        for axis in axes:
            if axis not in domain_axes_keys:
                raise ValueError("Invalid axis: {!r}".format(axis))

            out.append(axis)

        return out

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def close(self):
        '''Close all files referenced by the domain construct.

    Note that a closed file will be automatically reopened if its
    contents are subsequently required.

    :Returns:

        `None`

    **Examples:**

    >>> d.close()

        '''
        self.constructs.close()

    def cyclic(self, identity=None, iscyclic=True, period=None):
        '''Set the cyclicity of an axis.

    .. versionadded:: 3.TODO.0

    .. seealso:: `autocyclic`, `domain_axis`, `iscyclic`

    :Parameters:

        identity:
           Select the domain axis construct by one of:

              * An identity or key of a 1-d coordinate construct that
                whose data spans the domain axis construct.

              * A domain axis construct identity or key.

            The *identity* parameter selects the domain axis as
            returned by this call of the `domain_axis` method:
            ``d.domain_axis(identity)``.

        iscyclic: `bool`, optional
            If False then the axis is set to be non-cyclic. By
            default the selected axis is set to be cyclic.

        period: optional
            The period for a dimension coordinate construct which
            spans the selected axis. May be any numeric scalar object
            that can be converted to a `Data` object (which includes
            numpy array and `Data` objects). The absolute value of
            *period* is used. If *period* has units then they must be
            compatible with those of the dimension coordinates,
            otherwise it is assumed to have the same units as the
            dimension coordinates.

    :Returns:

        `set`
            The construct keys of the domain axes which were cyclic
            prior to the new setting, or the current cyclic domain
            axes if no axis was specified.

    **Examples:**

    >>> f.cyclic()
    set()
    >>> f.cyclic('X', period=360)
    set()
    >>> f.cyclic()
    {'domainaxis2'}
    >>> f.cyclic('X', iscyclic=False)
    {'domainaxis2'}
    >>> f.cyclic()
    set()

        '''
        cyclic = self._cyclic
        old = cyclic.copy()
        
        if identity is None:
            return old

        axis = self.domain_axis(identity, key=True)

        if iscyclic:
            dim = self.dimension_coordinate(axis, default=None)
            if dim is not None:
                if period is not None:
                    dim.period(period)
                elif dim.period() is None:
                    raise ValueError(
                        "A cyclic dimension coordinate must have a period")
        # --- End: if

        self._cyclic = cyclic.union((axis,))

        return old

    def domain_axis(self, identity, key=False, default=ValueError()):
        '''Return a domain axis construct, or its key.

    .. versionadded:: 3.TODO.0

    .. seealso:: `construct`, `auxiliary_coordinate`, `cell_measure`,
                 `cell_method`, `coordinate`, `coordinate_reference`,
                 `dimension_coordinate`, `domain_ancillary`,
                 `domain_axes`, `field_ancillary`

    :Parameters:

        identity:
           Select the domain axis construct by one of:

              * An identity or key of a 1-d coordinate construct that
                whose data spans the domain axis construct.

              * A domain axis construct identity or key.


            A construct identity is specified by a string
            (e.g. ``'latitude'``, ``'long_name=time'``,
            ``'ncvar%lat'``, etc.); or a compiled regular expression
            (e.g. ``re.compile('^atmosphere')``) that selects the
            relevant constructs whose identities match via
            `re.search`.

            Each construct has a number of identities, and is selected
            if any of them match any of those provided. A construct's
            identities are those returned by its `!identities`
            method. In the following example, the construct ``x`` has
            six identities:

               >>> x.identities()
               ['time', 'long_name=Time', 'foo=bar', 'standard_name=time', 'ncvar%t', 'T']

            A construct key may optionally have the ``'key%'``
            prefix. For example ``'dimensioncoordinate2'`` and
            ``'key%dimensioncoordinate2'`` are both acceptable keys.

            A position of a domain axis construct in the field
            construct's data is specified by an integer index.

            Note that in the output of a `print` call or `!dump`
            method, a construct is always described by one of its
            identities, and so this description may always be used as
            an *identity* argument.

            *Parameter example:*
              ``identity='long_name=Latitude'``

            *Parameter example:*
              ``identity='dimensioncoordinate1'``

            *Parameter example:*
              ``identity='domainaxis2'``

            *Parameter example:*
              ``identity='key%domainaxis2'``

            *Parameter example:*
              ``identity='ncdim%y'``

        key: `bool`, optional
            If True then return the selected construct key. By
            default the construct itself is returned.

        default: optional
            Return the value of the *default* parameter if a construct
            can not be found. If set to an `Exception` instance then
            it will be raised instead.

    :Returns:

        `DomainAxis` or `str`
            The selected domain axis construct, or its key.

    **Examples:**

    TODO

        '''
        domain_axes = self.domain_axes(identity)
            
        n_domain_axes = len(domain_axes)        
        if n_domain_axes == 1:
            # identity is a unique domain axis construct identity
            if key:
                return domain_axes.key()

            return domain_axes.value()

        if n_domain_axes > 1:
            return self._default(
                default,
                "No unique domain axis construct is identifiable from "
                "{!r}".format(identity)
            )
                
        # identity is not a unique domain axis construct identity
        da_key = self.domain_axis_key(identity, default=None)
        if da_key is None:
            return self._default(default, message="TODO")
            
        if key:
            return da_key
        
        return self.constructs[da_key]

    def get_data_axes(self, identity, default=ValueError()):
        '''Return the keys of the domain axis constructs spanned by the data
    of a metadata construct.

    .. versionadded:: 3.TODO.0

    .. seealso:: `del_data_axes`, `has_data_axes`, `set_data_axes`

    :Parameters:

        identity:
           Select the construct for which to return the domain axis
           constructs spanned by its data. By default the field
           construct is selected. May be:

              * The identity or key of a metadata construct.

            A construct identity is specified by a string
            (e.g. ``'latitude'``, ``'long_name=time'``,
            ``'ncvar%lat'``, etc.); or a compiled regular expression
            (e.g. ``re.compile('^atmosphere')``) that selects the
            relevant constructs whose identities match via
            `re.search`.

            Each construct has a number of identities, and is selected
            if any of them match any of those provided. A construct's
            identities are those returned by its `!identities`
            method. In the following example, the construct ``x`` has
            six identities:

               >>> x.identities()
               ['time', 'long_name=Time', 'foo=bar', 'standard_name=time', 'ncvar%t', 'T']

            A construct key may optionally have the ``'key%'``
            prefix. For example ``'dimensioncoordinate2'`` and
            ``'key%dimensioncoordinate2'`` are both acceptable keys.

            Note that in the output of a `print` call or `!dump`
            method, a construct is always described by one of its
            identities, and so this description may always be used as
            an *identity* argument.

        default: optional
            Return the value of the *default* parameter if the data
            axes have not been set.

            {{default Exception}}

    :Returns:

        `tuple`
            The keys of the domain axis constructs spanned by the data.

    **Examples:**

    TODO

        '''
        key = self.construct(identity, key=True, default=None)
        if key is None:
            return self.construct_key(identity, default=default)

        return super().get_data_axes(key=key, default=default)

    def identity(self, default='', strict=False, relaxed=False,
                 nc_only=False):
        '''Return the canonical identity.

    By default the identity is the first found of the following:

    * The "id" attribute, preceded by ``'id%'``.
    * The "cf_role" property, preceded by ``'cf_role='``.
    * The "long_name" property, preceded by ``'long_name='``.
    * The netCDF variable name, preceded by ``'ncvar%'``.
    * The value of the *default* parameter.

    .. versionadded:: 3.TODO.

    .. seealso:: `id`, `identities`

    :Parameters:

        default: optional
            If no identity can be found then return the value of the
            default parameter.

        strict: `bool`, optional
            If True then the identity is the first found of only the
            "standard_name" property or the "id" attribute.

        relaxed: `bool`, optional
            If True then the identity is the first found of only the
            "standard_name" property, the "id" attribute, the
            "long_name" property or the netCDF variable name.

        nc_only: `bool`, optional
            If True then only take the identity from the netCDF
            variable name.

    :Returns:

            The identity.

    **Examples:**

TODO
    >>> f.properties()
    {'foo': 'bar',
     'long_name': 'Air Temperature',
     'standard_name': 'air_temperature'}
    >>> f.nc_get_variable()
    'tas'
    >>> f.identity()
    'air_temperature'
    >>> f.del_property('standard_name')
    'air_temperature'
    >>> f.identity(default='no identity')
    'air_temperature'
    >>> f.identity()
    'long_name=Air Temperature'
    >>> f.del_property('long_name')
    >>> f.identity()
    'ncvar%tas'
    >>> f.nc_del_variable()
    'tas'
    >>> f.identity()
    'ncvar%tas'
    >>> f.identity()
    ''
    >>> f.identity(default='no identity')
    'no identity'

        '''
        if nc_only:
            if strict:
                raise ValueError(
                    "'strict' and 'nc_only' parameters cannot both be True")

            if relaxed:
                raise ValueError(
                    "'relaxed' and 'nc_only' parameters cannot both be True")

            n = self.nc_get_variable(None)
            if n is not None:
                return 'ncvar%{0}'.format(n)

            return default

        n = getattr(self, 'id', None)
        if n is not None:
            return 'id%{0}'.format(n)

        if relaxed:
            n = self.get_property('long_name', None)
            if n is not None:
                return 'long_name={0}'.format(n)

            n = self.nc_get_variable(None)
            if n is not None:
                return 'ncvar%{0}'.format(n)

            return default

        if strict:
            return default

        for prop in ('cf_role', 'long_name'):
            n = self.get_property(prop, None)
            if n is not None:
                return '{0}={1}'.format(prop, n)
        # --- End: for

        n = self.nc_get_variable(None)
        if n is not None:
            return 'ncvar%{0}'.format(n)

        return default

    def identities(self):
        '''Return all possible identities.

    The identities comprise:

    * The "id" attribute, preceded by ``'id%'``.
    * The ``cf_role`` property, preceeded by ``'cf_role='``.
    * The ``long_name`` property, preceeded by ``'long_name='``.
    * All other properties, preceeded by the property name and a
      equals e.g. ``'foo=bar'``.
    * The netCDF variable name, preceeded by ``'ncvar%'``.

    .. versionadded:: (cfdm) 1.9.0.0

    .. seealso:: `identity`

    :Returns:

        `list`
            The identities.

    **Examples:**

    >>> d = {{package}}.Domain()
    >>> d.set_properties({'foo': 'bar',
    ...                   'long_name': 'Domain for model'})
    >>> d.nc_set_variable('dom1')
    >>> d.identities()
    ['long_name=Domain for model', 'foo=bar', 'ncvar%dom1']

        '''
        out = super().identities()

        i = getattr(self, 'id', None)
        if i is not None:
            # Insert id attribute
            i = 'id%{0}'.format(i)
            if not out:
                out = [i]
            else:
                out.insert(0, i)
        # --- End: if

        return out

    def indices(self, *mode, **kwargs):
        '''Create indices that define a subspace of the field construct.

    The subspace is defined by identifying indices based on the
    metadata constructs.

    Metadata constructs are selected conditions are specified on their
    data. Indices for subspacing are then automatically inferred from
    where the conditions are met.

    The returned tuple of indices may be used to created a subspace by
    indexing the original field construct with them.

    Metadata constructs and the conditions on their data are defined
    by keyword parameters.

    * Any domain axes that have not been identified remain unchanged.

    * Multiple domain axes may be subspaced simultaneously, and it
      doesn't matter which order they are specified in.

    * Subspace criteria may be provided for size 1 domain axes that
      are not spanned by the field construct's data.

    * Explicit indices may also be assigned to a domain axis
      identified by a metadata construct, with either a Python `slice`
      object, or a sequence of integers or booleans.

    * For a dimension that is cyclic, a subspace defined by a slice or
      by a `Query` instance is assumed to "wrap" around the edges of
      the data.

    * Conditions may also be applied to multi-dimensional metadata
      constructs. The "compress" mode is still the default mode (see
      the positional arguments), but because the indices may not be
      acting along orthogonal dimensions, some missing data may still
      need to be inserted into the field construct's data.

    **Auxiliary masks**

    When creating an actual subspace with the indices, if the first
    element of the tuple of indices is ``'mask'`` then the extent of
    the subspace is defined only by the values of elements three and
    onwards. In this case the second element contains an "auxiliary"
    data mask that is applied to the subspace after its initial
    creation, in order to set unselected locations to missing data.

    .. seealso:: `subspace`, `where`, `__getitem__`, `__setitem__`

    :Parameters:

        mode: `str`, *optional*
            There are three modes of operation, each of which provides
            indices for a different type of subspace:

            ==============  ==========================================
            *mode*          Description
            ==============  ==========================================
            ``'compress'``  This is the default mode. Unselected
                            locations are removed to create the
                            returned subspace. Note that if a
                            multi-dimensional metadata construct is
                            being used to define the indices then some
                            missing data may still be inserted at
                            unselected locations.

            ``'envelope'``  The returned subspace is the smallest that
                            contains all of the selected
                            indices. Missing data is inserted at
                            unselected locations within the envelope.

            ``'full'``      The returned subspace has the same domain
                            as the original field construct. Missing
                            data is inserted at unselected locations.
            ==============  ==========================================

        kwargs: *optional*
            A keyword name is an identity of a metadata construct, and
            the keyword value provides a condition for inferring
            indices that apply to the dimension (or dimensions)
            spanned by the metadata construct's data. Indices are
            created that select every location for which the metadata
            construct's data satisfies the condition.

    :Returns:

        `tuple`
            The indices meeting the conditions.

    **Examples:**

    >>> q = cf.example_field(0)
    >>> print(q)
    Field: specific_humidity (ncvar%q)
    ----------------------------------
    Data            : specific_humidity(latitude(5), longitude(8)) 1
    Cell methods    : area: mean
    Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : time(1) = [2019-01-01 00:00:00]
    >>> indices = q.indices(X=112.5)
    >>> print(indices)
    (slice(0, 5, 1), slice(2, 3, 1))
    >>> q[indicies]
    <CF Field: specific_humidity(latitude(5), longitude(1)) 1>
    >>> q.indices(X=112.5, latitude=cf.gt(-60))
    (slice(1, 5, 1), slice(2, 3, 1))
    >>> q.indices(latitude=cf.eq(-45) | cf.ge(20))
    (array([1, 3, 4]), slice(0, 8, 1))
    >>> q.indices(X=[1, 2, 4], Y=slice(None, None, -1))
    (slice(4, None, -1), array([1, 2, 4]))
    >>> q.indices(X=cf.wi(-100, 200))
    (slice(0, 5, 1), slice(-2, 4, 1))
    >>> q.indices(X=slice(-2, 4))
    (slice(0, 5, 1), slice(-2, 4, 1))
    >>> q.indices('compress', X=[1, 2, 4, 6])
    (slice(0, 5, 1), array([1, 2, 4, 6]))
    >>> q.indices(Y=[True, False, True, True, False])
    (array([0, 2, 3]), slice(0, 8, 1))
    >>> q.indices('envelope', X=[1, 2, 4, 6])
    ('mask', [<CF Data(1, 6): [[False, ..., False]]>], slice(0, 5, 1), slice(1, 7, 1))
    >>> indices = q.indices('full', X=[1, 2, 4, 6])
    ('mask', [<CF Data(1, 8): [[True, ..., True]]>], slice(0, 5, 1), slice(0, 8, 1))
    >>> print(indices)
    >>> print(q)
    <CF Field: specific_humidity(latitude(5), longitude(8)) 1>

    >>> print(a)
    Field: air_potential_temperature (ncvar%air_potential_temperature)
    ------------------------------------------------------------------
    Data            : air_potential_temperature(time(120), latitude(5), longitude(8)) K
    Cell methods    : area: mean
    Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                    : latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : air_pressure(1) = [850.0] hPa
    >>> a.indices(T=410.5)
    (slice(2, 3, 1), slice(0, 5, 1), slice(0, 8, 1))
    >>> a.indices(T=cf.dt('1960-04-16'))
    (slice(4, 5, 1), slice(0, 5, 1), slice(0, 8, 1))
    >>> indices = a.indices(T=cf.wi(cf.dt('1962-11-01'),
    ...                             cf.dt('1967-03-17 07:30')))
    >>> print(indices)
    (slice(35, 88, 1), slice(0, 5, 1), slice(0, 8, 1))
    >>> a[indices]
    <CF Field: air_potential_temperature(time(53), latitude(5), longitude(8)) K>

    >>> print(t)
    Field: air_temperature (ncvar%ta)
    ---------------------------------
    Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K
    Cell methods    : grid_latitude(10): grid_longitude(9): mean where land (interval: 0.1 degrees) time(1): maximum
    Field ancils    : air_temperature standard_error(grid_latitude(10), grid_longitude(9)) = [[0.76, ..., 0.32]] K
    Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
                    : grid_latitude(10) = [2.2, ..., -1.76] degrees
                    : grid_longitude(9) = [-4.7, ..., -1.18] degrees
                    : time(1) = [2019-01-01 00:00:00]
    Auxiliary coords: latitude(grid_latitude(10), grid_longitude(9)) = [[53.941, ..., 50.225]] degrees_N
                    : longitude(grid_longitude(9), grid_latitude(10)) = [[2.004, ..., 8.156]] degrees_E
                    : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., b'kappa']
    Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
    Coord references: grid_mapping_name:rotated_latitude_longitude
                    : standard_name:atmosphere_hybrid_height_coordinate
    Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                    : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                    : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m
    >>> indices = t.indices(latitude=cf.wi(51, 53))
    >>> print(indices)
    ('mask', [<CF Data(1, 5, 9): [[[False, ..., False]]]>], slice(0, 1, 1), slice(3, 8, 1), slice(0, 9, 1))
    >>> t[indices]
    <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(5), grid_longitude(9)) K>

        '''
        if 'exact' in mode:
            _DEPRECATION_ERROR_ARG(
                self, 'indices', 'exact',
                "Keywords are now never interpreted as regular expressions."
            )  # pragma: no cover

        domain_indices = self._indices(mode, None, **kwargs)

        return domain_indices['indices']

    @_inplace_enabled(default=False)
    def roll(self, axis, shift, inplace=False):
        '''Roll the field along a cyclic axis.

    A unique axis is selected with the axes and kwargs parameters.

    .. versionadded:: 1.0

    .. seealso:: `anchor`, `axis`, `cyclic`, `iscyclic`, `period`

    :Parameters:

        axis:
            The cyclic axis to be rolled, defined by that which would
            be selected by passing the given axis description to a
            call of the field construct's `domain_axis` method. For
            example, for a value of ``'X'``, the domain axis construct
            returned by ``f.domain_axis('X')`` is selected.

        shift: `int`
            The number of places by which the selected cyclic axis is
            to be rolled.

        {{inplace: `bool`, optional}}

    :Returns:

        `Field`
            The rolled field.

    **Examples:**

    Roll the data of the "X" axis one elements to the right:

    >>> f.roll('X', 1)

    Roll the data of the "X" axis three elements to the left:

    >>> f.roll('X', -3)

        '''
        axis = self.domain_axis(
            axis, key=True,
            default=ValueError(
                "Can't roll: Bad axis specification: {!r}".format(axis)
            )
        )

        d = _inplace_enabled_define_and_cleanup(self)

        if d.domain_axes[axis].get_size() <= 1:
            return d

        # Roll the metadata constructs in-place
        d._roll_constructs(axis, shift)

        return d

    def subspace(self, **kwargs):
        '''TODO

        '''
        logger.debug(
            "{}.subspace\n"
            "    input indices  = {}".format(
                self.__class__.__name__, kwargs
            )
        )  # pragma: no cover

        domain_axes = self.domain_axes

        axes = []
        indices = []
        shape = []
        for a, b in kwargs.items():
            axes.append(a)
            shape.append(domain_axes[a].get_size())
            indices.append(b)

        indices, roll = parse_indices(tuple(shape), tuple(indices),
                                      cyclic=True)

        logger.debug(
            "    parsed indices = {}\n"
            "    roll           = {}".format(
                indices, roll
            )
        )  # pragma: no cover

        if roll:
            new = self
            cyclic_axes = self.cyclic()
            for iaxis, shift in roll.items():
                axis = axes[iaxis]
                if axis not in cyclic_axes:
                    raise IndexError(
                        "Can't take a cyclic slice from non-cyclic {!r} "
                        "axis".format(
                            self.constructs.domain_axis_identity(axis))
                    )

                new = new.roll(axis, shift)
        else:
            new = self.copy()

        # ------------------------------------------------------------
        # Set sizes of domain axes
        # ------------------------------------------------------------
        domain_axes = new.domain_axes
        for axis, index in zip(axes, indices):
            if isinstance(index, slice):
                size = abs((index.stop - index.start) / index.step)
                int_size = round(size)
                if size > int_size:
                    size = int_size + 1
                else:
                    size = int_size
            else:
                size = numpy_size(index)

            domain_axes[axis].set_size(size)

        # ------------------------------------------------------------
        # Subspace constructs with data
        # ------------------------------------------------------------
        construct_data_axes = new.constructs.data_axes()

        for key, construct in new.constructs.filter_by_data().items():
            construct_axes = construct_data_axes[key]

            dice = [indices[i] for i, axis in enumerate(axes)
                    if axis in construct_data_axes[key]]

            logger.debug(
                '    dice = {}'.format(dice))  # pragma: no cover

            # Replace existing construct with its subspace
            new.set_construct(construct[tuple(dice)], key=key,
                              axes=construct_axes, copy=False)

        return new

    @_inplace_enabled(default=False)
    def transpose(self, axes, inplace=False):
        '''Permute the axes of the data array.

TODO    By default the order of the axes is reversed, but any ordering may
    be specified by selecting the axes of the output in the required
    order.

    .. versionadded:: 3.TODO.0

    .. seealso:: `domain_axis`, `flatten`, `insert_dimension`, `flip`,
                 `squeeze`, `unsqueeze`

    :Parameters:

        axes: sequence of `str`
            Select the domain axis order, defined by the domain axes
            that would be selected by passing each given axis
            description to a call of the field construct's
            `domain_axis` method. For example, for a value of ``'X'``,
            the domain axis construct returned by
            ``f.domain_axis('X')`` is selected.

            Each domain axis of the domain construct data must be
            provided.

        constructs: `bool`, optional
            If True then metadata constructs are also transposed so
            that their axes are in the same relative order as in the
            transposed data array of the field. By default metadata
            constructs are not altered.

        {{inplace: `bool`, optional}}

    :Returns:

        `Domain` or `None`
            The domain construct with transposed constructs, or `None`
            if the operation was in-place.

    **Examples:**

    >>> f.ndim
    3
    >>> g = f.transpose()
    >>> g = f.transpose(['time', 1, 'dim2'])
    >>> f.transpose(['time', -2, 'dim2'], inplace=True)

        '''
        f = _inplace_enabled_define_and_cleanup(self)

        try:
            axes = f._parse_axes(axes)
        except ValueError as error:
            raise ValueError("Can't transpose domain: {}".format(error))

        if axes is None or len(axes) != len(f.domain_axes):
            raise ValueError(
                "Can't transpose domain: Must provide order for "
                "all domain axes. Got: {}".format(axes)
            )

        for key, construct in f.constructs.filter_by_data().items():
            construct_axes = f.get_data_axes(key)

            iaxes = [
                construct_axes.index(axis) for axis in axes
                if axis in construct_axes
            ]

            # Transpose the construct
            construct.transpose(iaxes, inplace=True)

        return f

# --- End: class
