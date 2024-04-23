from math import prod
from os import sep

import cfdm
import numpy as np

from . import mixin
from .auxiliarycoordinate import AuxiliaryCoordinate
from .constructs import Constructs
from .data import Data
from .decorators import _inplace_enabled, _inplace_enabled_define_and_cleanup
from .dimensioncoordinate import DimensionCoordinate
from .domainaxis import DomainAxis
from .functions import (
    _DEPRECATION_ERROR_ARG,
    _DEPRECATION_ERROR_METHOD,
    abspath,
    indices_shape,
    parse_indices,
)

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
        instance._Constructs = Constructs
        instance._Data = Data
        instance._DomainAxis = DomainAxis
        instance._DimensionCoordinate = DimensionCoordinate
        instance._AuxiliaryCoordinate = AuxiliaryCoordinate
        return instance

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

        return prod(
            [domain_axis.get_size(0) for domain_axis in domain_axes.values()]
        )

    def add_file_location(
        self,
        location,
    ):
        """Add a new file location in-place.

        All data definitions that reference files are additionally
        referenced from the given location.

        .. versionadded:: 3.15.0

        .. seealso:: `del_file_location`, `file_locations`

        :Parameters:

            location: `str`
                The new location.

        :Returns:

            `str`
                The new location as an absolute path with no trailing
                path name component separator.

        **Examples**

        >>> f.add_file_location('/data/model/')
        '/data/model'

        """
        location = abspath(location).rstrip(sep)

        for c in self.constructs.filter_by_data(todict=True).values():
            c.add_file_location(location)

        return location

    def cfa_clear_file_substitutions(
        self,
    ):
        """Remove all of the CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Returns:

            `dict`
                {{Returns cfa_clear_file_substitutions}}

        **Examples**

        >>> d.cfa_clear_file_substitutions()
        {}

        """
        out = {}
        for c in self.constructs.filter_by_data(todict=True).values():
            out.update(c.cfa_clear_file_substitutions())

        return out

    def cfa_file_substitutions(self):
        """Return the CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Returns:

            `dict`
                {{Returns cfa_file_substitutions}}

        **Examples**

        >>> d.cfa_file_substitutions()
        {}

        """
        out = {}
        for c in self.constructs.filter_by_data(todict=True).values():
            out.update(c.cfa_file_substitutions())

        return out

    def cfa_del_file_substitution(
        self,
        base,
    ):
        """Remove a CFA-netCDF file name substitution.

        .. versionadded:: 3.15.0

        :Parameters:

            base: `str`
                {{cfa base: `str`}}

        :Returns:

            `dict`
                {{Returns cfa_del_file_substitution}}

        **Examples**

        >>> f.cfa_del_file_substitution('base')

        """
        for c in self.constructs.filter_by_data(todict=True).values():
            c.cfa_del_file_substitution(
                base,
            )

    def cfa_update_file_substitutions(
        self,
        substitutions,
    ):
        """Set CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Parameters:

            {{cfa substitutions: `dict`}}

        :Returns:

            `None`

        **Examples**

        >>> d.cfa_update_file_substitutions({'base': '/data/model'})

        """
        for c in self.constructs.filter_by_data(todict=True).values():
            c.cfa_update_file_substitutions(substitutions)

    def close(self):
        """Close all files referenced by the domain construct.

        Deprecated at version 3.14.0. All files are now
        automatically closed when not being accessed.

        Note that a closed file will be automatically reopened if its
        contents are subsequently required.

        :Returns:

            `None`

        **Examples**

        >>> d.close()

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "close",
            "All files are now automatically closed when not being accessed.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def del_file_location(
        self,
        location,
    ):
        """Remove a file location in-place.

        All data definitions that reference files will have references
        to files in the given location removed from them.

        .. versionadded:: 3.15.0

        .. seealso:: `add_file_location`, `file_locations`

        :Parameters:

            location: `str`
                 The file location to remove.

        :Returns:

            `str`
                The removed location as an absolute path with no
                trailing path name component separator.

        **Examples**

        >>> d.del_file_location('/data/model/')
        '/data/model'

        """
        location = abspath(location).rstrip(sep)

        for c in self.constructs.filter_by_data(todict=True).values():
            c.del_file_location(location)

        return location

    @classmethod
    def create_regular(cls, x_args, y_args, bounds=True):
        """
        Create a new domain with the regular longitudes and latitudes.

        .. versionadded:: 3.15.1

        .. seealso:: `cf.DimensionCoordinate.create_regular`

        :Parameters:

            x_args: sequence of numbers
                {{regular args}}

            y_args: sequence of numbers
                {{regular args}}

            bounds: `bool`, optional
                If True (default), bounds will be created
                for the coordinates, and the coordinate points will be the
                midpoints of the bounds. If False, the given ranges represent
                the coordinate points directly.

        :Returns:

            `Domain`
                The newly created domain with the specified longitude and
                latitude coordinates and bounds.

        **Examples**

        >>> import cf
        >>> domain = cf.Domain.create_regular((-180, 180, 1), (-90, 90, 1))
        >>> domain.dump()
        --------
        Domain:
        --------
        Domain Axis: latitude(180)
        Domain Axis: longitude(360)
        Dimension coordinate: longitude
            standard_name = 'longitude'
            units = 'degrees_east'
            Data(longitude(360)) = [-179.5, ..., 179.5] degrees_east
            Bounds:units = 'degrees_east'
            Bounds:Data(longitude(360), 2) = [[-180.0, ..., 180.0]] degrees_east
        Dimension coordinate: latitude
            standard_name = 'latitude'
            units = 'degrees_north'
            Data(latitude(180)) = [-89.5, ..., 89.5] degrees_north
            Bounds:units = 'degrees_north'
            Bounds:Data(latitude(180), 2) = [[-90.0, ..., 90.0]] degrees_north

        """

        x_args = np.array(x_args)

        if x_args.shape != (3,) or x_args.dtype.kind not in "fi":
            raise ValueError(
                "The args argument was incorrectly formatted. "
                f"Expected a sequence of three numbers, got {x_args}."
            )

        y_args = np.array(y_args)

        if y_args.shape != (3,) or y_args.dtype.kind not in "fi":
            raise ValueError(
                "The args argument was incorrectly formatted. "
                f"Expected a sequence of three numbers, got {y_args}."
            )

        x_range = x_args[0:2]
        y_range = y_args[0:2]

        domain = cls()

        if abs(x_range[1] - x_range[0]) > 360:
            raise ValueError(
                "The difference in x_range should not be greater than 360."
            )

        if y_range[0] < -90 or y_range[1] > 90:
            raise ValueError(
                "y_range must be within the range of -90 to 90 degrees."
            )

        longitude = domain._DimensionCoordinate.create_regular(
            x_args, "degrees_east", "longitude", bounds=bounds
        )

        latitude = domain._DimensionCoordinate.create_regular(
            y_args, "degrees_north", "latitude", bounds=bounds
        )

        domain_axis_longitude = domain.set_construct(
            domain._DomainAxis(longitude.size), copy=False
        )

        domain_axis_latitude = domain.set_construct(
            domain._DomainAxis(latitude.size), copy=False
        )

        domain.set_construct(
            longitude, axes=[domain_axis_longitude], copy=False
        )
        domain.set_construct(latitude, axes=[domain_axis_latitude], copy=False)

        return domain

    def file_locations(
        self,
    ):
        """The locations of files containing parts of the components data.

        Returns the locations of any files that may be required to
        deliver the computed data arrays of any of the component
        constructs (such as dimension coordinate constructs, cell
        measure constructs, etc.).

        .. versionadded:: 3.15.0

        .. seealso:: `add_file_location`, `del_file_location`

        :Returns:

            `set`
                The unique file locations as absolute paths with no
                trailing path name component separator.

        **Examples**

        >>> d.file_locations()
        {'/home/data1', 'file:///data2'}

        """
        out = set()
        for c in self.constructs.filter_by_data(todict=True).values():
            out.update(c.file_locations())

        return out

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

        **Examples**

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

        **Examples**

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

        **Examples**

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
            if strict:
                raise ValueError(
                    "'relaxed' and 'strict' parameters cannot both be True"
                )

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
        * The ``cf_role`` property, preceded by ``'cf_role='``.
        * The ``long_name`` property, preceded by ``'long_name='``.
        * All other properties, preceded by the property name and a
          equals e.g. ``'foo=bar'``.
        * The netCDF variable name, preceded by ``'ncvar%'``.

        .. versionadded:: (cfdm) 1.9.0.0

        .. seealso:: `identity`

        :Returns:

            `list`
                The identities.

        **Examples**

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

    def indices(self, *config, **kwargs):
        """Create indices that define a subspace of the domain
        construct.

        The indices returned by this method may be used to create the
        subspace by passing them to the `subspace` method of the
        original domain construct.

        The subspace is defined by identifying indices based on the
        metadata constructs.

        Metadata constructs are selected conditions are specified on
        their data. Indices for subspacing are then automatically
        inferred from where the conditions are met.

        Metadata constructs and the conditions on their data are
        defined by keyword parameters.

        * Any domain axes that have not been identified remain
          unchanged.

        * Multiple domain axes may be subspaced simultaneously, and it
          doesn't matter which order they are specified in.

        * Explicit indices may also be assigned to a domain axis
          identified by a metadata construct, with either a Python
          `slice` object, or a sequence of integers or booleans.

        * For a dimension that is cyclic, a subspace defined by a
          slice or by a `Query` instance is assumed to "wrap" around
          the edges of the data.

        * Conditions may also be applied to multi-dimensional metadata
          constructs. The "compress" mode is still the default mode
          (see the positional arguments), but because the indices may
          not be acting along orthogonal dimensions, some missing data
          may still need to be inserted into the field construct's
          data.

        **Halos**

        {{subspace halos}}

        .. versionadded:: 3.11.0

        .. seealso:: `subspace`, `where`, `__getitem__`,
                     `__setitem__`, `cf.Field.indices`

        :Parameters:

            {{config: optional}}

                {{subspace valid modes Domain}}

            kwargs: optional
                A keyword name is an identity of a metadata construct,
                and the keyword value provides a condition for
                inferring indices that apply to the dimension (or
                dimensions) spanned by the metadata construct's
                data. Indices are created that select every location
                for which the metadata construct's data satisfies the
                condition.

        :Returns:

            `dict`
                A dictionary of indices, keyed by the domain axis
                construct identifiers to which they apply.

        **Examples**

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
        # Get the indices for every domain axis in the domain, without
        # any auxiliary masks.
        domain_indices = self._indices(config, None, False, kwargs)

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

                Alternatively, one or more cell method constructs may
                be identified in a single string with a CF-netCDF cell
                methods-like syntax for describing both the collapse
                dimensions, the collapse method, and any cell method
                construct qualifiers. If N cell methods are described
                in this way then they will collectively identify the N
                most recently applied cell method operations. For
                example, ``'T: maximum within years T: mean over
                years'`` will be compared with the most two most
                recently applied cell method operations.

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
                Whether or not the domain construct contains the
                specified metadata constructs.

        **Examples**

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

        **Examples**

        Roll the data of the "X" axis one elements to the right:

        >>> f.roll('X', 1)

        Roll the data of the "X" axis three elements to the left:

        >>> f.roll('X', -3)

        """
        # TODODASK: Consider allowing multiple roll axes, now that
        #           Data supports them.

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

    def subspace(self, *config, **kwargs):
        """Create a subspace of the field construct.

        Creation of a new domain construct which spans a subspace of
        the domain of an existing domain construct is achieved by
        identifying indices based on the metadata constructs
        (subspacing by metadata). The new domain construct is created
        with the same properties as the original domain construct.

        **Subspacing by metadata**

        Subspacing by metadata selects metadata constructs and
        specifies conditions on their data. Indices for subspacing are
        then automatically inferred from where the conditions are met.

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

        **Halos**

        {{subspace halos}}

        .. versionadded:: 3.11.0

        .. seealso:: `indices`, `cf.Field.subspace`

        :Parameters:

            {{config: optional}}

                {{subspace valid modes Domain}}

            kwargs: optional
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

        **Examples**

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
        test = False
        if "test" in config:
            config = list(config)
            config.remove("test")
            test = True

        if not config and not kwargs:
            if test:
                return True

            return self.copy()

        try:
            indices = self.indices(*config, **kwargs)
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
        for axis, size in zip(axes, indices_shape(indices, shape)):
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
        relative ordering defined by the *axes* parameter. For
        instance, if the given *axes* are ``['X', 'Z', 'Y']`` then a
        metadata construct whose data axis order is ``('Y', 'X')``
        will be transposed to have data order ``('X', 'Y')``.

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

        **Examples**

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
