import logging

from functools import reduce
from operator import mul as operator_mul

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


class Domain(mixin.FieldDomain,
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
        '''Storage for axis cyclicity.

        Do not change in-place.
        '''
        return self._custom.get('_cyclic', set())

    @_cyclic.setter
    def _cyclic(self, value):
        self._custom['_cyclic'] = value

    @_cyclic.deleter
    def _cyclic(self):
        self._custom['_cyclic'] = set()

    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def size(self):
        '''The number of locations in the domain.

    If there are no domain axis constructs, any domain axis construct
    has a size of 0, then a size of 0 is returned.

    :Returns:

        `int`
            The size.

        '''
        domain_axes = self.domain_axes
        if not domain_axes:
            return 0

        return reduce(
            operator_mul,
            [domain_axis.get_size(0) for domain_axis in domain_axes.values()],
            1)

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

        identity: optional
            Select the domain axis construct by one of:

            * An identity or key of a 1-d dimension or auxiliary
              coordinate construct that whose data spans the domain
              axis construct.

              {{construct selection identity}}

            * A domain axis construct identity.

              {{domain axis selection identity}}

            * The key of a domain axis construct.

            * `None`. This is the default, which selects the domain
              axis construct when there is only one of them.

            *Parameter example:*
              ``identity='time'``

            *Parameter example:*
              ``identity='domainaxis2'``

            *Parameter example:*
              ``identity='ncdim%y'``

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

    def domain_axis(self, identity=None, key=False,
                    default=ValueError()):
        '''Return a domain axis construct, or its key.

    .. versionadded:: 3.TODO.0

    .. seealso:: `construct`, `auxiliary_coordinate`, `cell_measure`,
                 `cell_method`, `coordinate`, `coordinate_reference`,
                 `dimension_coordinate`, `domain_ancillary`,
                 `domain_axes`, `field_ancillary`

    :Parameters:

        identity: optional
            Select the domain axis construct by one of:

            * An identity or key of a 1-d dimension or auxiliary
              coordinate construct that whose data spans the domain
              axis construct.

              {{construct selection identity}}

            * A domain axis construct identity.

              {{domain axis selection identity}}

            * The key of a domain axis construct.

            * `None`. This is the default, which selects the domain
              axis construct when there is only one of them.

            *Parameter example:*
              ``identity='time'``

            *Parameter example:*
              ``identity='domainaxis2'``

            *Parameter example:*
              ``identity='ncdim%y'``

        key: `bool`, optional
            If True then return the selected construct key. By
            default the construct itself is returned.

        default: optional
            Return the value of the *default* parameter if a construct
            can not be found.

            {{default Exception}}

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

    @_inplace_enabled(default=False)
    def flip(self, axes=None, inplace=False):
        '''Flip (reverse the direction of) axes of the field. TODO

    .. seealso:: `domain_axis`, `flatten`, `insert_dimension`, TODO
                 `squeeze`, `transpose`, `unsqueeze`

    :Parameters:

        axes: (sequence of) `str` , optional

            Select the domain axes to flip, defined by the domain axes
            that would be selected by passing each given axis
            description to a call of the `domain_axis` method. For
            example, for a value of ``'X'``, the domain axis construct
            returned by ``f.domain_axis('X')`` is selected.

            If no axes are provided then all axes are flipped.

        {{inplace: `bool`, optional}}

    :Returns:

        `Domain` or `None`
            The construct with flipped axes, or `None` if the
            operation was in-place.

    **Examples:**

    >>> g = f.flip()
    >>> g = f.flip('time')
    >>> g = f.flip(1)
    >>> g = f.flip(['time', 1, 'dim2'])
    >>> f.flip(['dim2'], inplace=True)

        '''
        if axes is None:
            # Flip all the axes
            axes = set(self.domain_axes)
        else:
            if isinstance(axes, str):
                axes = (axes,)

            axes = set([self.domain_axis(axis, key=True) for axis in axes])

        d = _inplace_enabled_define_and_cleanup(self)

        # Flip constructs with data
        d.constructs._flip(axes)

        return d

    def get_data_axes(self, identity, default=ValueError()):
        '''Return the keys of the domain axis constructs spanned by the data
    of a metadata construct.

    .. versionadded:: 3.TODO.0

    .. seealso:: `del_data_axes`, `has_data_axes`, `set_data_axes`

    :Parameters:

        identity: optional
            Select the construct by one of

            * A metadata construct identity.

              {{construct selection identity}}

            * The key of a metadata construct

            * `None`. This is the default, which selects the metadata
              construct when there is only one of them.

            *Parameter example:*
              ``identity='latitude'``

            *Parameter example:*
              ``identity='T'

            *Parameter example:*
              ``identity='long_name=Cell Area'``

            *Parameter example:*
              ``identity='cellmeasure1'``

            *Parameter example:*
              ``identity='measure:area'``

            *Parameter example:*
              ``identity=cf.eq('time')'``

            *Parameter example:*
              ``identity=re.compile('^lat')``

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
        '''Create indices that define a subspace of the domain construct.

    The subspace is defined by identifying indices based on the
    metadata constructs.

    Metadata constructs are selected conditions are specified on their
    data. Indices for subspacing are then automatically inferred from
    where the conditions are met.

    Metadata constructs and the conditions on their data are defined
    by keyword parameters.

    * Any domain axes that have not been identified remain unchanged.

    * Multiple domain axes may be subspaced simultaneously, and it
      doesn't matter which order they are specified in.

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

    .. versionadded:: 3.TODO.0

    .. seealso:: `subspace`, `where`, `__getitem__`, `__setitem__`

    :Parameters:

        mode: `str`, *optional*
            There are two modes of operation, each of which provides
            indices for a different type of subspace:

            ==============  ==========================================
            *mode*          Description
            ==============  ==========================================
            ``'compress'``  Return indices that identify only the
                            requested locations.

                            This is the default mode.

                            Note that if a multi-dimensional metadata
                            construct is being used to define the
                            indices then some unrequested locations
                            may also be selected.

            ``'envelope'``  The returned subspace is the smallest that
                            contains all of the requested locations.
            ==============  ==========================================

        kwargs: *optional*
            A keyword name is an identity of a metadata construct, and
            the keyword value provides a condition for inferring
            indices that apply to the dimension (or dimensions)
            spanned by the metadata construct's data. Indices are
            created that select every location for which the metadata
            construct's data satisfies the condition.

    :Returns:

        `dict`
            A dictionary of indices, keyed by the domain axis
            construct identifiers to which they apply

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
        if len(mode) > 1:
            raise ValueError(
                "Can't provide more than one positional argument. "
                "Got: {}".format(', '.join(repr(x) for x in mode))
            )

        if not mode or 'compress' in mode:
            mode = 'compress'
        elif 'envelope' in mode:
            mode = 'envelope'
        else:
            raise ValueError(
                "Invalid value for 'mode' argument: {!r}".format(mode[0])
            )

        # ------------------------------------------------------------
        # Get the indices for every domain axis in the domain, without
        # any auxiliary masks.
        # ------------------------------------------------------------
        domain_indices = self._indices(mode, None, False, **kwargs)

        # ------------------------------------------------------------
        # Return the indices
        # ------------------------------------------------------------
        return domain_indices['indices']

    def match_by_construct(self, *identities, OR=False,
                           **conditions):
        '''Whether or not there are particular metadata constructs.

    .. versionadded:: 3.TODO.0

    .. seealso:: `match`, `match_by_property`, `match_by_rank`,
                 `match_by_identity`, `match_by_ncvar`

    :Parameters:

        identities: optional
            Identify the metadata constructs by one or more of

            * A metadata construct identity.

              {{construct selection identity}}

            * The key of a metadata construct

            If a cell method construct identity is given (such as
            ``'method:mean'``) then it will only be compared with the
            most recently applied cell method operation.

            Alternatively, one or more cell method constucts may be
            identified in a single string with a CF-netCDF cell
            methods-like syntax for describing both the collapse
            dimensions, the collapse method, and any cell method
            construct qualifiers. If N cell methods are described in
            this way then they will collectively identify the N most
            recently applied cell method operations. For example,
            ``'T: maximum within years T: mean over years'`` will be
            compared with the most two most recently applied cell
            method operations.

            *Parameter example:*
              ``identity='latitude'``

            *Parameter example:*
              ``'T'

            *Parameter example:*
              ``'latitude'``

            *Parameter example:*
              ``'long_name=Cell Area'``

            *Parameter example:*
              ``'cellmeasure1'``

            *Parameter example:*
              ``'measure:area'``

            *Parameter example:*
              ``cf.eq('time')'``

            *Parameter example:*
              ``re.compile('^lat')``

            *Parameter example:*
              ``'domainancillary2', 'longitude'``

            *Parameter example:*
              ``'area: mean T: maximum'``

            *Parameter example:*
              ``'grid_latitude', 'area: mean T: maximum'``

        conditions: optional
            Identify the metadata constructs that have any of the
            given identities or construct keys, and whose data satisfy
            conditions.

            A construct identity or construct key (as defined by the
            *identities* parameter) is given as a keyword name and a
            condition on its data is given as the keyword value.

            The condition is satisfied if any of its data values
            equals the value provided.

            *Parameter example:*
              ``longitude=180.0``

            *Parameter example:*
              ``time=cf.dt('1959-12-16')``

            *Parameter example:*
              ``latitude=cf.ge(0)``

            *Parameter example:*
              ``latitude=cf.ge(0), air_pressure=500``

            *Parameter example:*
              ``**{'latitude': cf.ge(0), 'long_name=soil_level': 4}``

        OR: `bool`, optional
            If True then return `True` if at least one metadata
            construct matches at least one of the criteria given by
            the *identities* or *conditions* arguments. By default
            `True` is only returned if the field constructs matches
            each of the given criteria.

    :Returns:

        `bool`
            Whether or not the domain construct contains the specfied
            metadata constructs.

    **Examples:**

        TODO

        '''
        if identities:
            if identities[0] == 'or':
                _DEPRECATION_ERROR_ARG(
                    self, 'match_by_construct', 'or',
                    message="Use 'OR=True' instead.", version='3.1.0'
                )  # pragma: no cover

            if identities[0] == 'and':
                _DEPRECATION_ERROR_ARG(
                    self, 'match_by_construct', 'and',
                    message="Use 'OR=False' instead.", version='3.1.0'
                )  # pragma: no cover
        # --- End: if

        if not identities and not conditions:
            return True

        constructs = self.constructs

        if not constructs:
            return False

        n = 0

        for identity in identities:
            filtered = constructs(identity)
            if filtered:
                n += 1
            elif not OR:
                return False
        # --- End: for

        if conditions:
            for identity, value in conditions.items():
                if self.subspace('test', **{identity: value}):
                    n += 1
                elif not OR:
                    return False
        # --- End: if

        if OR:
            return bool(n)

        return True

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
        # TODODASK - allow multiple roll axes
        
        axis = self.domain_axis(
            axis, key=True,
            default=ValueError(
                f"Can't roll {self.__class__.__name__}. "
                f"Bad axis specification: {axis!r}"
            )
        )

        d = _inplace_enabled_define_and_cleanup(self)

        # Roll the metadata constructs in-place
        axes = d._parse_axes(axis)
        d._roll_constructs(axes, shift)

        return d

    def subspace(self, *mode, **kwargs):
        '''Create a subspace of a domain construct.

    Creation of a new domain construct that spans a subspace of the
    domain of the existing domain construct is achieved by identifying
    indices based on the metadata constructs.

    Metadata constructs and the conditions on their data are defined
    by keyword parameters.

    The subspace is defined by identifying indices based on the
    metadata constructs.

    The following beahviouts apply:

    * Any domain axes that have not been identified are indexed with
      `slice(None)`.

    * Multiple domain axes may be subspaced simultaneously, and it
      doesn't matter which order they are specified in.

    * Explicit indices may also be assigned to a domain axis
      identified by a metadata construct, with either a Python `slice`
      object, or a sequence of integers or booleans.

    * For a dimension that is cyclic, a subspace defined by a slice or
      by a `Query` instance is assumed to "wrap" around the edges of
      the data.

    * Conditions may also be applied to multi-dimensional metadata
      constructs. The "compress" mode is still the default mode (see
      the *mode* parameter). Depending on the distribution of metadata
      construct values and the subspace criteria, unselected cells may
      still occur in the new domain.

    .. seealso:: `indices`

    :Parameters:

        mode: *optional*
            There are three modes of operation, each of which provides
            a different type of subspace, plus a testing mode:

            ==============  ==========================================
            *argument*      Description
            ==============  ==========================================
            ``'compress'``  This is the default mode. Unselected
                            locations are removed to create the
                            returned subspace. Note that if a
                            multi-dimensional metadata construct is
                            being used to define the indices then some
                            unsleceted cells still occur in the new
                            domain.

            ``'envelope'``  The returned subspace is the smallest that
                            contains all of the selected
                            indices. Missing data is inserted at
                            unselected locations within the envelope.

            ``'test'``      May be used on its own or in addition to
                            one of the other positional arguments. Do
                            not create a subspace, but return `True`
                            or `False` depending on whether or not it
                            is possible to create the specified
                            subspace.
            ==============  ==========================================

        Keyword parameters: *optional*
            A keyword name is an identity of a metadata construct, and
            the keyword value provides a condition for inferring
            indices that apply to the dimension (or dimensions)
            spanned by the metadata construct's data. Indices are
            created that select every location for which the metadata
            construct's data satisfies the condition.

    :Returns:

        `Domain` or `bool`
            An independent domain construct containing the subspace of
            the original domain. If the ``'test'`` positional argument
            has been set then return `True` or `False` depending on
            whether or not it is possible to create specified
            subspace.

    **Examples:**

    There are further worked examples
    :ref:`in the tutorial <Subspacing-by-metadata>`.

    >>> g = f.subspace(X=112.5)
    >>> g = f.subspace(X=112.5, latitude=cf.gt(-60))
    >>> g = f.subspace(latitude=cf.eq(-45) | cf.ge(20))
    >>> g = f.subspace(X=[1, 2, 4], Y=slice(None, None, -1))
    >>> g = f.subspace(X=cf.wi(-100, 200))
    >>> g = f.subspace(X=slice(-2, 4))
    >>> g = f.subspace(Y=[True, False, True, True, False])
    >>> g = f.subspace(T=410.5)
    >>> g = f.subspace(T=cf.dt('1960-04-16'))
    >>> g = f.subspace(T=cf.wi(cf.dt('1962-11-01'), cf.dt('1967-03-17 07:30')))
    >>> g = f.subspace('compress', X=[1, 2, 4, 6])
    >>> g = f.subspace('envelope', X=[1, 2, 4, 6])
    >>> g = f.subspace('full', X=[1, 2, 4, 6])
    >>> g = f.subspace(latitude=cf.wi(51, 53))

        '''
        logger.debug(
            "{}.subspace\n"
            "  input kwargs = {}".format(
                self.__class__.__name__, kwargs
            )
        )  # pragma: no cover

        test = False
        if 'test' in mode:
            mode = list(mode)
            mode.remove('test')
            test = True

        if not mode and not kwargs:
            if test:
                return True

            return self.copy()

        try:
            indices = self.indices(*mode, **kwargs)
        except ValueError as error:
            if test:
                return False

            raise ValueError(error)

        if test:
            return True

        domain_axes = self.domain_axes

        axes = []
        shape = []
        indices2 = []
        for a, b in indices.items():
            axes.append(a)
            shape.append(domain_axes[a].get_size())
            indices2.append(b)

        indices, roll = parse_indices(tuple(shape), tuple(indices2),
                                      cyclic=True)

        logger.debug(
            "  axes           = {!r}\n"
            "  parsed indices = {!r}\n"
            "  roll           = {!r}".format(axes, indices, roll)
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
                old_size = domain_axes[axis].get_size()
                start, stop, step = index.indices(old_size)
                size = abs((stop - start) / step)
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

            dice = [indices[axes.index(axis)] for axis in construct_axes]

            logger.debug(
                " dice = {!r}".format(dice)
            )  # pragma: no cover

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
        d = _inplace_enabled_define_and_cleanup(self)        
        
        if axes is None:
            raise ValueError(
                f"Can't transpose {self.__class__.__name__}. "
                f"Must provide an order for all axes. Got: {axes}"
            )

        axes_in = axes
        axes = d._parse_axes(axes_in)
        
        if len(axes) != len(d.domain_axes):
            raise ValueError(
                f"Can't transpose {self.__class__.__name__}. "
                f"Must provide an order for all axes. Got: {axes_in}"
            )

        for key, construct in d.constructs.filter_by_data().items():
            construct_axes = d.get_data_axes(key)

            iaxes = [construct_axes.index(a)
                     for a in axes if a in construct_axes]

            # Transpose the construct
            construct.transpose(iaxes, inplace=True)

        return f

# --- End: class
