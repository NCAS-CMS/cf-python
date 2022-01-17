import logging
from functools import reduce
from operator import mul as operator_mul

import cfdm
import numpy as np

from . import mixin
from .constructs import Constructs
from .data import Data
from .decorators import _inplace_enabled, _inplace_enabled_define_and_cleanup
from .functions import _DEPRECATION_ERROR_ARG, parse_indices

logger = logging.getLogger(__name__)

_empty_set = set()


class Domain(mixin.FieldDomain, mixin.Properties, cfdm.Domain):
    """A domain construct of the CF data model.

    The domain represents a set of discrete "locations" in what
    generally would be a multi-dimensional space, either in the real
    world or in a model's simulated world. The data array elements of
    a field construct correspond to individual location of a domain.

    The domain construct is defined collectively by the following
    constructs of the CF data model: domain axis, dimension
    coordinate, auxiliary coordinate, cell measure, coordinate
    reference, and domain ancillary constructs; as well as properties
    to describe the domain.

    **NetCDF interface**

    {{netCDF variable}}

    {{netCDF global attributes}}

    {{netCDF group attributes}}

    {{netCDF geometry group}}

    Some components exist within multiple constructs, but when written
    to a netCDF dataset the netCDF names associated with such
    components will be arbitrarily taken from one of them. The netCDF
    variable, dimension and sample dimension names and group
    structures for such components may be set or removed consistently
    across all such components with the `nc_del_component_variable`,
    `nc_set_component_variable`, `nc_set_component_variable_groups`,
    `nc_clear_component_variable_groups`,
    `nc_del_component_dimension`, `nc_set_component_dimension`,
    `nc_set_component_dimension_groups`,
    `nc_clear_component_dimension_groups`,
    `nc_del_component_sample_dimension`,
    `nc_set_component_sample_dimension`,
    `nc_set_component_sample_dimension_groups`,
    `nc_clear_component_sample_dimension_groups` methods.

    A domain construct of the CF data model.

    The domain represents a set of discrete "locations" in what
    generally would be a multi-dimensional space, either in the real
    world or in a model's simulated world. The data array elements of
    a field construct correspond to individual location of a domain.

    The domain construct is defined collectively by the following
    constructs of the CF data model: domain axis, dimension
    coordinate, auxiliary coordinate, cell measure, coordinate
    reference, and domain ancillary constructs; as well as properties
    to describe the domain.

    """

    def __new__(cls, *args, **kwargs):
        """Creates a new Domain instance."""
        instance = super().__new__(cls)
        instance._Data = Data
        instance._Constructs = Constructs
        return instance

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)

    @property
    def _cyclic(self):
        """Storage for axis cyclicity.

        Do not change the value in-place.

        """
        return self._custom.get("_cyclic", _empty_set)

    @_cyclic.setter
    def _cyclic(self, value):
        """value must be a set.

        Do not change the value in-place.

        """
        self._custom["_cyclic"] = value

    @_cyclic.deleter
    def _cyclic(self):
        self._custom["_cyclic"] = _empty_set

    @property
    def size(self):
        """The number of locations in the domain.

        If there are no domain axis constructs, or any domain axis
        construct has a size of 0, then the size is 0.

        """
        domain_axes = self.domain_axes(todict=True)
        if not domain_axes:
            return 0

        return reduce(
            operator_mul,
            [domain_axis.get_size(0) for domain_axis in domain_axes.values()],
            1,
        )

    def close(self):
        """Close all files referenced by the domain construct.

        Note that a closed file will be automatically reopened if its
        contents are subsequently required.

        :Returns:

            `None`

        **Examples:**

        >>> d.close()

        """
        # TODODASK - is this still needed?

        self.constructs.close()

    #    def cyclic(
    #        self, *identity, iscyclic=True, period=None, config={}, **filter_kwargs
    #    ):
    #        """Set the cyclicity of an axis.
    #
    #        .. versionadded:: 3.11.0
    #
    #        .. seealso:: `autocyclic`, `domain_axis`, `iscyclic`
    #
    #        :Parameters:
    #
    #            identity, filter_kwargs: optional
    #                Select the unique domain axis construct returned by
    #                ``f.domain_axis(*identity, **filter_kwargs)``. See
    #                `domain_axis` for details.
    #
    #            iscyclic: `bool`, optional
    #                If False then the axis is set to be non-cyclic. By
    #                default the selected axis is set to be cyclic.
    #
    #            period: optional
    #                The period for a dimension coordinate construct which
    #                spans the selected axis. May be any numeric scalar
    #                object that can be converted to a `Data` object (which
    #                includes numpy array and `Data` objects). The absolute
    #                value of *period* is used. If *period* has units then
    #                they must be compatible with those of the dimension
    #                coordinates, otherwise it is assumed to have the same
    #                units as the dimension coordinates.
    #
    #            config: `dict`
    #                Additional parameters for optimizing the
    #                operation. See the code for details.
    #
    #        :Returns:
    #
    #            `set`
    #                The construct keys of the domain axes which were cyclic
    #                prior to the new setting, or the current cyclic domain
    #                axes if no axis was specified.
    #
    #        **Examples:**
    #
    #        >>> f.cyclic()
    #        set()
    #        >>> f.cyclic('X', period=360)
    #        set()
    #        >>> f.cyclic()
    #        {'domainaxis2'}
    #        >>> f.cyclic('X', iscyclic=False)
    #        {'domainaxis2'}
    #        >>> f.cyclic()
    #        set()
    #
    #        """
    #        cyclic = self._cyclic
    #        old = cyclic.copy()
    #
    #        if identity is None:
    #            return old
    #
    #        axis = self.domain_axis(identity, key=True)
    #
    #        if iscyclic:
    #            dim = self.dimension_coordinate(axis, default=None)
    #            if dim is not None:
    #                if period is not None:
    #                    dim.period(period)
    #                elif dim.period() is None:
    #                    raise ValueError(
    #                        "A cyclic dimension coordinate must have a period"
    #                    )
    #
    #        # Never change _cyclic in-place
    #        self._cyclic = cyclic.union((axis,))
    #
    #        return old
    #
    #    def domain_axis(self, identity=None, key=False, item=False,
    #                    default=ValueError()):
    #        """Return a domain axis construct, or its key.
    #
    #        .. versionadded:: 3.11.0
    #
    #        .. seealso:: `construct`, `auxiliary_coordinate`, `cell_measure`,
    #                     `cell_method`, `coordinate`, `coordinate_reference`,
    #                     `dimension_coordinate`, `domain_ancillary`,
    #                     `domain_axes`, `field_ancillary`
    #
    #        :Parameters:
    #
    #            identity: optional
    #                Select the domain axis construct.
    #
    #                {{domain axis selection}}
    #
    #                If *identity is `None` (the default) then the unique
    #                domain axis construct is selected when there is only one
    #                of them.
    #
    #                *Parameter example:*
    #                  ``identity='time'``
    #
    #                *Parameter example:*
    #                  ``identity='domainaxis2'``
    #
    #                *Parameter example:*
    #                  ``identity='ncdim%y'``
    #
    #            key: `bool`, optional
    #                If True then return the selected construct key. By
    #                default the construct itself is returned.
    #
    #            default: optional
    #                Return the value of the *default* parameter if a construct
    #                can not be found.
    #
    #                {{default Exception}}
    #
    #        :Returns:
    #
    #            `DomainAxis` or `str`
    #                The selected domain axis construct, or its key.
    #
    #        **Examples:**
    #
    #        """
    #        c = self.domain_axes(identity)
    #
    #        n = len(c)
    #        if n == 1:
    #            k, construct = c.popitem()
    #            if key:
    #                return k
    #
    #            if item:
    #                return k, construct
    #
    #            return construct
    #        elif n > 1:
    #            if default is None:
    #                return default
    #
    #            return self._default(
    #                default,
    #                f"{self.__class__.__name__}.{_method}() can't return {n} "
    #                "constructs",
    #            )
    #
    #        # identity is not a unique domain axis construct identity
    #        da_key = self.domain_axis_key(identity, default=None)
    #        if da_key is None:
    #            if default is None:
    #                return default
    #
    #            return self._default(
    #                default,
    #                message=f"No domain axis found from identity {identity!r}",
    #            )
    #
    #        if key:
    #            return da_key
    #
    #        return self.constructs[da_key]

    @_inplace_enabled(default=False)
    def flip(self, axes=None, inplace=False):
        """Flip (reverse the direction of) domain axes.

        .. seealso:: `domain_axis`, `transpose`

        :Parameters:

            axes: (sequence of) `str` , optional
                Select the domain axes to flip.

                A domain axis is identified by that which would be
                selected by passing a given axis description to a call of
                the `domain_axis` method. For example, a value of ``'X'``
                would select the domain axis construct returned by
                ``f.domain_axis('X')``.

                If no axes are provided then all axes are flipped.

            {{inplace: `bool`, optional}}

        :Returns:

            `Domain` or `None`
                The domain with flipped axes, or `None` if the operation
                was in-place.

        **Examples:**

        >>> d = cf.example_field(0).domain
        >>> print(d)
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        >>> print(d.flip('X'))
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [337.5, ..., 22.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        >>> print(d.flip(['T', 'Y']))
        Dimension coords: latitude(5) = [75.0, ..., -75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        >>> print(d.flip())
        Dimension coords: latitude(5) = [75.0, ..., -75.0] degrees_north
                        : longitude(8) = [337.5, ..., 22.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        if axes is None:
            # Flip all the axes
            axes = self.domain_axes(todict=True)
        else:
            axes = self._parse_axes(axes)

        axes = set(axes)

        # Flip constructs with data
        d.constructs._flip(axes)

        return d

    def get_data(self, default=ValueError(), _units=None, _fill_value=True):
        """Return a default value when data is requested.

        A `Domain` instance can never have data, so a default value
        must be returned if data is requested. This is useful for
        cases when it is not known in advance if a `Field` or `Domain`
        instance is in use.

        .. versionadded:: 3.11.0

        .. seealso:: `has_data`

        :Parameters:

            default: optional
                Return the value of the *default* parameter.

                {{default Exception}}

            _units: optional
                Ignored.

            _fill_value: optional
                Ignored.

        :Returns:

                The value of the *default* parameter, if an exception
                has not been raised.

        **Examples:**

        >>> d = cf.example_domain(0)
        >>> print(d.get_data(None))
        None
        >>> d.get_data()
        Traceback (most recent call last):
            ...
        ValueError: Domain has no data

        """
        if default is None:
            return

        return self._default(
            default, message=f"{self.__class__.__name__} has no data"
        )

    def get_data_axes(self, identity, default=ValueError()):
        """Return the keys of the domain axis constructs spanned by the
        data of a metadata construct.

        .. versionadded:: 3.11.0

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

        >>> d = cf.example_field(7).domain
        >>> print(d)
        Dimension coords: time(3) = [1979-05-01 12:00:00, 1979-05-02 12:00:00, 1979-05-03 12:00:00] gregorian
                        : air_pressure(1) = [850.0] hPa
                        : grid_latitude(4) = [0.44, ..., -0.88] degrees
                        : grid_longitude(5) = [-1.18, ..., 0.58] degrees
        Auxiliary coords: latitude(grid_latitude(4), grid_longitude(5)) = [[52.4243, ..., 51.1163]] degrees_north
                        : longitude(grid_latitude(4), grid_longitude(5)) = [[8.0648, ..., 10.9238]] degrees_east
        Coord references: grid_mapping_name:rotated_latitude_longitude
        >>> print(d.constructs)
        Constructs:
        {'auxiliarycoordinate0': <CF AuxiliaryCoordinate: latitude(4, 5) degrees_north>,
         'auxiliarycoordinate1': <CF AuxiliaryCoordinate: longitude(4, 5) degrees_east>,
         'coordinatereference0': <CF CoordinateReference: grid_mapping_name:rotated_latitude_longitude>,
         'dimensioncoordinate0': <CF DimensionCoordinate: time(3) days since 1979-1-1 gregorian>,
         'dimensioncoordinate1': <CF DimensionCoordinate: air_pressure(1) hPa>,
         'dimensioncoordinate2': <CF DimensionCoordinate: grid_latitude(4) degrees>,
         'dimensioncoordinate3': <CF DimensionCoordinate: grid_longitude(5) degrees>,
         'domainaxis0': <CF DomainAxis: size(3)>,
         'domainaxis1': <CF DomainAxis: size(1)>,
         'domainaxis2': <CF DomainAxis: size(4)>,
         'domainaxis3': <CF DomainAxis: size(5)>}
        >>> d.get_data_axes('grid_latitude')
        ('domainaxis2',)
        >>> d.get_data_axes('latitude')
        ('domainaxis2', 'domainaxis3')

        """
        key = self.construct(identity, key=True, default=None)
        if key is None:
            return self.construct_key(identity, default=default)

        return super().get_data_axes(key=key, default=default)

    def identity(self, default="", strict=False, relaxed=False, nc_only=False):
        """Return the canonical identity.

        By default the identity is the first found of the following:

        * The "id" attribute, preceded by ``'id%'``.
        * The "cf_role" property, preceded by ``'cf_role='``.
        * The "long_name" property, preceded by ``'long_name='``.
        * The netCDF variable name, preceded by ``'ncvar%'``.
        * The value of the *default* parameter.

        .. versionadded:: 3.11.0

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

        """
        if nc_only:
            if strict:
                raise ValueError(
                    "'strict' and 'nc_only' parameters cannot both be True"
                )

            if relaxed:
                raise ValueError(
                    "'relaxed' and 'nc_only' parameters cannot both be True"
                )

            n = self.nc_get_variable(None)
            if n is not None:
                return f"ncvar%{n}"

            return default

        n = getattr(self, "id", None)
        if n is not None:
            return f"id%{n}"

        if relaxed:
            n = self.get_property("long_name", None)
            if n is not None:
                return f"long_name={n}"

            n = self.nc_get_variable(None)
            if n is not None:
                return f"ncvar%{n}"

            return default

        if strict:
            return default

        for prop in ("cf_role", "long_name"):
            n = self.get_property(prop, None)
            if n is not None:
                return f"{prop}={n}"

        n = self.nc_get_variable(None)
        if n is not None:
            return f"ncvar%{n}"

        return default

    def identities(self):
        """Return all possible identities.

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

        """
        out = super().identities()

        i = getattr(self, "id", None)
        if i is not None:
            # Insert id attribute
            i = f"id%{i}"
            if not out:
                out = [i]
            else:
                out.insert(0, i)

        return out

    def indices(self, *mode, **kwargs):
        """Create indices that define a subspace of the domain
        construct.

        The indices returned by this method be used to create the subspace
        by passing them to the `subspace` method of the original domain
        construct.

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

        .. versionadded:: 3.11.0

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
                construct identifiers to which they apply.

        **Examples:**

        >>> d = cf.example_field(0).domain
        >>> print(d)
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> indices = d.indices(X=112.5)
        >>> indices
        {'domainaxis0': slice(0, 5, 1),
         'domainaxis1': slice(2, 3, 1),
         'domainaxis2': slice(0, 1, 1)}
        >>> print(d.subspace(**indices))
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(1) = [112.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        >>> indices = d.indices(X=112.5, Y=cf.wi(-60, 30))
        >>> indices
        {'domainaxis0': slice(1, 3, 1),
         'domainaxis1': slice(2, 3, 1),
         'domainaxis2': slice(0, 1, 1)}
        >>> print(d.subspace(**indices))
        Dimension coords: latitude(2) = [-45.0, 0.0] degrees_north
                        : longitude(1) = [112.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        >>> d.indices(X=[-1, 0], Y=slice(1, -1))
        {'domainaxis0': slice(1, 4, 1),
         'domainaxis1': slice(7, None, -7),
         'domainaxis2': slice(0, 1, 1)}
        >>> print(print(d.subspace(**indices)))
        Dimension coords: latitude(3) = [-45.0, 0.0, 45.0] degrees_north
                        : longitude(2) = [337.5, 22.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        """
        if len(mode) > 1:
            raise ValueError(
                "Can't provide more than one positional argument. "
                f"Got: {', '.join(repr(x) for x in mode)}"
            )

        if not mode or "compress" in mode:
            mode = "compress"
        elif "envelope" in mode:
            mode = "envelope"
        else:
            raise ValueError(f"Invalid value for 'mode' argument: {mode[0]!r}")

        # ------------------------------------------------------------
        # Get the indices for every domain axis in the domain, without
        # any auxiliary masks.
        # ------------------------------------------------------------
        domain_indices = self._indices(mode, None, False, **kwargs)

        # ------------------------------------------------------------
        # Return the indices
        # ------------------------------------------------------------
        return domain_indices["indices"]

    def match_by_construct(self, *identities, OR=False, **conditions):
        """Whether or not there are particular metadata constructs.

        .. versionadded:: 3.11.0

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

        >>> d = cf.example_field(0).domain
        >>> print(d)
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> d.match_by_construct("latitude")
        True
        >>> d.match_by_construct("air_pressure")
        False
        >>> d.match_by_construct("longitude", "time")
        True
        >>> d.match_by_construct(longitude=22.5)
        True
        >>> d.match_by_construct(longitude=15.5)
        False
        >>> d.match_by_construct(longitude=cf.gt(340))
        False
        >>> d.match_by_construct(longitude=cf.gt(240))
        True
        >>> d.match_by_construct(time=cf.dt("2019-01-01"))
        True
        >>> d.match_by_construct(time=cf.dt("2020-01-01"))
        False

        """
        if identities:
            if identities[0] == "or":
                _DEPRECATION_ERROR_ARG(
                    self,
                    "match_by_construct",
                    "or",
                    message="Use 'OR=True' instead.",
                    version="3.1.0",
                )  # pragma: no cover

            if identities[0] == "and":
                _DEPRECATION_ERROR_ARG(
                    self,
                    "match_by_construct",
                    "and",
                    message="Use 'OR=False' instead.",
                    version="3.1.0",
                )  # pragma: no cover

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

        if conditions:
            for identity, value in conditions.items():
                if self.subspace("test", **{identity: value}):
                    n += 1
                elif not OR:
                    return False

        if OR:
            return bool(n)

        return True

    @_inplace_enabled(default=False)
    def roll(self, axis, shift, inplace=False):
        """Roll the field along a cyclic axis.

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

        """
        # TODODASK - allow multiple roll axes

        axis = self.domain_axis(
            axis,
            key=True,
            default=ValueError(
                f"Can't roll {self.__class__.__name__}. "
                f"Bad axis specification: {axis!r}"
            ),
        )

        d = _inplace_enabled_define_and_cleanup(self)

        # Roll the metadata constructs in-place
        axes = d._parse_axes(axis)
        d._roll_constructs(axes, shift)

        return d

    def subspace(self, *mode, **kwargs):
        """Create indices that define a subspace of the domain
        construct.

        The indices returned by this method be used to create the subspace
        by passing them to the `subspace` method of the original domain
        construct.

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

        .. versionadded:: 3.11.0

        .. seealso:: `indices`

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

                ``'test'``      May be used on its own or in addition to
                                one of the other positional arguments. Do
                                not create a subspace, but return `True`
                                or `False` depending on whether or not it
                                is possible to create the specified
                                subspace.
                ==============  ==========================================

            kwargs: *optional*
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

        >>> d = cf.example_field(0).domain
        >>> print(d)
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(d.subspace(X=112.5))
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(1) = [112.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        >>> print(d.indices(X=112.5, Y=cf.wi(-60, 30)))
        Dimension coords: latitude(2) = [-45.0, 0.0] degrees_north
                        : longitude(1) = [112.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        >>> print(d.indices(X=[-1, 0], Y=slice(1, -1))
        Dimension coords: latitude(3) = [-45.0, 0.0, 45.0] degrees_north
                       : longitude(2) = [337.5, 22.5] degrees_east
                       : time(1) = [2019-01-01 00:00:00]

        """
        logger.debug(
            f"{self.__class__.__name__}.subspace\n"
            f"  input kwargs = {kwargs}"
        )  # pragma: no cover

        test = False
        if "test" in mode:
            mode = list(mode)
            mode.remove("test")
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

        domain_axes = self.domain_axes(todict=True)

        axes = []
        shape = []
        indices2 = []
        for a, b in indices.items():
            axes.append(a)
            shape.append(domain_axes[a].get_size())
            indices2.append(b)

        indices, roll = parse_indices(
            tuple(shape), tuple(indices2), cyclic=True
        )

        logger.debug(
            f"  axes           = {axes!r}\n"
            f"  parsed indices = {indices!r}\n"
            f"  roll           = {roll!r}"
        )  # pragma: no cover

        if roll:
            new = self
            cyclic_axes = self.cyclic()
            for iaxis, shift in roll.items():
                axis = axes[iaxis]
                if axis not in cyclic_axes:
                    raise IndexError(
                        "Can't take a cyclic slice from non-cyclic "
                        f"{self.constructs.domain_axis_identity(axis)!r} "
                        "axis"
                    )

                new = new.roll(axis, shift)
        else:
            new = self.copy()

        # ------------------------------------------------------------
        # Set sizes of domain axes
        # ------------------------------------------------------------
        domain_axes = new.domain_axes(todict=True)
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
                size = np.size(index)

            domain_axes[axis].set_size(size)

        # ------------------------------------------------------------
        # Subspace constructs that have data
        # ------------------------------------------------------------
        construct_data_axes = new.constructs.data_axes()

        for key, construct in new.constructs.filter_by_data().items():
            construct_axes = construct_data_axes[key]
            dice = [indices[axes.index(axis)] for axis in construct_axes]

            # Replace existing construct with its subspace
            new.set_construct(
                construct[tuple(dice)],
                key=key,
                axes=construct_axes,
                copy=False,
            )

        return new

    @_inplace_enabled(default=False)
    def transpose(self, axes, inplace=False):
        """Permute the data axes of the metadata constructs.

        Each metadata construct has its data axis order changed to the
        relative ordering defined by the *axes* parameter. For instance,
        if the given *axes* are ``['X', 'Z', 'Y']`` then a metadata
        construct whose data axis order is ('Y', 'X') will be tranposed to
        have data order ('X', 'Y').

        .. versionadded:: 3.11.0

        .. seealso:: `domain_axis`, `flip`

        :Parameters:

            axes: sequence of `str`
                Define the new domain axis order.

                A domain axis is identified by that which would be
                selected by passing a given axis description to a call of
                the `domain_axis` method. For example, a value of ``'X'``
                would select the domain axis construct returned by
                ``f.domain_axis('X')``.

                Each domain axis of the domain construct data must be
                specified.

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

        >>> d = cf.example_field(7).domain
        >>> print(d)
        Dimension coords: time(3) = [1979-05-01 12:00:00, 1979-05-02 12:00:00, 1979-05-03 12:00:00] gregorian
                        : air_pressure(1) = [850.0] hPa
                        : grid_latitude(4) = [0.44, ..., -0.88] degrees
                        : grid_longitude(5) = [-1.18, ..., 0.58] degrees
        Auxiliary coords: latitude(grid_latitude(4), grid_longitude(5)) = [[52.4243, ..., 51.1163]] degrees_north
                        : longitude(grid_latitude(4), grid_longitude(5)) = [[8.0648, ..., 10.9238]] degrees_east
        Coord references: grid_mapping_name:rotated_latitude_longitude


        >>> print(d.transpose(['X', 'T', 'Y', 'Z']))
        Dimension coords: time(3) = [1979-05-01 12:00:00, 1979-05-02 12:00:00, 1979-05-03 12:00:00] gregorian
                        : air_pressure(1) = [850.0] hPa
                        : grid_latitude(4) = [0.44, ..., -0.88] degrees
                        : grid_longitude(5) = [-1.18, ..., 0.58] degrees
        Auxiliary coords: latitude(grid_longitude(5), grid_latitude(4)) = [[52.4243, ..., 51.1163]] degrees_north
                        : longitude(grid_longitude(5), grid_latitude(4)) = [[8.0648, ..., 10.9238]] degrees_east
        Coord references: grid_mapping_name:rotated_latitude_longitude

        """
        d = _inplace_enabled_define_and_cleanup(self)

        # Parse the axes
        if axes is None:
            raise ValueError(
                f"Can't transpose {self.__class__.__name__}. "
                f"Must provide an order for all axes. Got: {axes}"
            )

        axes = d._parse_axes(axes)

        rank = self.rank
        if len(set(axes)) != rank:
            raise ValueError(
                f"Can't transpose {self.__class__.__name__}. "
                f"Must provide an unambiguous order for all "
                f"{rank} domain axes. Got: {axes}"
            )

        data_axes = d.constructs.data_axes()
        for key, construct in d.constructs.filter_by_data().items():
            construct_axes = data_axes[key]

            if len(construct_axes) < 2:
                # No need to transpose 1-d constructs
                continue

            # Transpose the construct
            iaxes = [
                construct_axes.index(a) for a in axes if a in construct_axes
            ]
            construct.transpose(iaxes, inplace=True)

            # Update the axis order
            new_axes = [construct_axes[i] for i in iaxes]
            d.set_data_axes(axes=new_axes, key=key)

        return d
