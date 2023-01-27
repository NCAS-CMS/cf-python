# from itertools import chain

from ..data.data import Data
from ..decorators import (
    _deprecated_kwarg_check,
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
)
from ..units import Units

_units_degrees = Units("degrees")


class Coordinate:
    """Mixin class for dimension or auxiliary coordinate constructs.

    .. versionadded:: 3.2.0

    """

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def ctype(self):
        """The CF coordinate type.

        One of ``'T'``, ``'X'``, ``'Y'`` or ``'Z'`` if the coordinate
        construct is for the respective CF axis type, otherwise
        `None`.

        .. seealso:: `T`, `X`, `~cf.Coordinate.Y`, `Z`

        **Examples**

        >>> c.X
        True
        >>> c.ctype
        'X'

        >>> c.T
        True
        >>> c.ctype
        'T'

        """
        if self.X:
            return "X"

        if self.T:
            return "T"

        if self.Y:
            return "Y"

        if self.Z:
            return "Z"

    @property
    def T(self):
        """True if and only if the data are coordinates for a CF 'T'
        axis.

        CF 'T' axis coordinates are defined by having one or more of
        the following:

          * The `axis` property has the value ``'T'``
          * Units of latitude

        .. seealso:: `ctype`, `X`, `~cf.Coordinate.Y`, `Z`

        **Examples**

        >>> c = cf.{{class}}()
        >>> c.Units = cf.Units('seconds since 1992-10-08')
        >>> c.T
        True

        """
        if self.Units.isreftime:
            return True

        axis = self.get_property("axis", None)
        if axis is not None:
            return axis == "T"

        # Still here? Then check the bounds.
        if self.has_bounds():
            bounds = self.get_bounds(None)
            if bounds is not None:
                return bounds.T

        return False

    @property
    def X(self):
        """True if and only if the data are coordinates for a CF 'X'
        axis.

        CF 'X' axis coordinates are defined by having one or more of
        the following:

          * The `axis` property has the value ``'X'``
          * Units of longitude
          * The `standard_name` property is one of ``'longitude'``,
            ``'projection_x_coordinate'`` or ``'grid_longitude'``

        .. seealso:: `ctype`, `T`, `~cf.Coordinate.Y`, `Z`

        **Examples**

        >>> f = cf.example_field(0)
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> c = f.coordinate('longitude')
        >>> c.dump()
        Dimension coordinate: longitude
            standard_name = 'longitude'
            units = 'degrees_east'
            Data(8) = [22.5, ..., 337.5] degrees_east
            Bounds:units = 'degrees_east'
            Bounds:Data(8, 2) = [[0.0, ..., 360.0]] degrees_east

        >>> c.X
        True
        >>> c.Y
        False

        """
        standard_name = self.get_property("standard_name", None)
        if standard_name is not None and standard_name in (
            "longitude",
            "projection_x_coordinate",
            "grid_longitude",
        ):
            return True

        if self.Units.islongitude:
            return True

        axis = self.get_property("axis", None)
        if axis is not None:
            return axis == "X"

        # Still here? Then check the bounds.
        if self.has_bounds():
            bounds = self.get_bounds(None)
            if bounds is not None:
                return bounds.X

        return False

    @property
    def Y(self):
        """True if and only if the data are coordinates for a CF 'Y'
        axis.

        CF 'Y' axis coordinates are defined by having one or more of
        the following:

          * The `axis` property has the value ``'Y'``
          * Units of latitude
          * The `standard_name` property is one of ``'latitude'``,
            ``'projection_y_coordinate'`` or ``'grid_latitude'``

        .. seealso:: `ctype`, `T`, `X`, `Z`

        **Examples**

        >>> c.Units
        <CF Units: degree_north>
        >>> c.Y
        True

        >>> c.standard_name == 'latitude'
        >>> c.Y
        True

        """
        standard_name = self.get_property("standard_name", None)
        if standard_name is not None and standard_name in (
            "latitude",
            "projection_y_coordinate",
            "grid_latitude",
        ):
            return True

        if self.Units.islatitude:
            return True

        axis = self.get_property("axis", None)
        if axis is not None:
            return axis == "Y"

        # Still here? Then check the bounds.
        if self.has_bounds():
            bounds = self.get_bounds(None)
            if bounds is not None:
                return bounds.Y

        return False

    @property
    def Z(self):
        """True if and only if the data are coordinates for a CF 'Z'
        axis.

        CF 'Z' axis coordinates are defined by having one or more of
        the following:

          * The `axis` property has the value ``'Z'``
          * Units of pressure, level, layer or sigma_level
          * The `positive` property has the value ``'up'`` or ``'down'``
            (case insensitive)
          * The `standard_name` property is one of
            ``'atmosphere_ln_pressure_coordinate'``,
            ``'atmosphere_sigma_coordinate'``,
            ``'atmosphere_hybrid_sigma_pressure_coordinate'``,
            ``'atmosphere_hybrid_height_coordinate'``,
            ``'atmosphere_sleve_coordinate``', ``'ocean_sigma_coordinate'``,
            ``'ocean_s_coordinate'``, ``'ocean_s_coordinate_g1'``,
            ``'ocean_s_coordinate_g2'``, ``'ocean_sigma_z_coordinate'`` or
            ``'ocean_double_sigma_coordinate'``

        .. seealso:: `ctype`, `T`, `X`, `~cf.Coordinate.Y`

        **Examples**

        >>> c.Units
        <CF Units: Pa>
        >>> c.Z
        True

        >>> c.Units.equivalent(cf.Units('K')) and c.positive == 'up'
        True
        >>> c.Z
        True

        >>> c.axis == 'Z' and c.Z
        True

        >>> c.Units
        <CF Units: sigma_level>
        >>> c.Z
        True

        >>> c.standard_name
        'ocean_sigma_coordinate'
        >>> c.Z
        True

        """
        standard_name = self.get_property("standard_name", None)
        if standard_name is not None and standard_name in (
            "atmosphere_ln_pressure_coordinate",
            "atmosphere_sigma_coordinate",
            "atmosphere_hybrid_sigma_pressure_coordinate",
            "atmosphere_hybrid_height_coordinate",
            "atmosphere_sleve_coordinate",
            "ocean_sigma_coordinate",
            "ocean_s_coordinate",
            "ocean_s_coordinate_g1",
            "ocean_s_coordinate_g2",
            "ocean_sigma_z_coordinate",
            "ocean_double_sigma_coordinate",
        ):
            return True

        units = self.Units
        if units.ispressure:
            return True

        positive = self.get_property("positive", None)
        if positive is not None:
            return str(positive).lower() in ("up", "down")

        axis = self.get_property("axis", None)
        if axis is not None:
            return axis == "Z"

        if units and units.units in ("level", "layer" "sigma_level"):
            return True

        # Still here? Then check the bounds.
        if self.has_bounds():
            bounds = self.get_bounds(None)
            if bounds is not None:
                return bounds.Z

        return False

    # ----------------------------------------------------------------
    # CF properties
    # ----------------------------------------------------------------
    @property
    def axis(self):
        """The axis CF property.

        The `axis` property may be used to specify the type of
        coordinates. It may take one of the values `'X'`, `'Y'`, `'Z'`
        or `'T'` which stand for a longitude, latitude, vertical, or
        time axis respectively. A value of `'X'`, `'Y'` or `'Z'` may
        also also used to identify generic spatial coordinates (the
        values `'X'` and `'Y'` being used to identify horizontal
        coordinates).

        **Examples**

        >>> c.axis = 'Y'
        >>> c.axis
        'Y'
        >>> del c.axis

        >>> c.set_property('axis', 'T')
        >>> c.get_property('axis')
        'T'
        >>> c.del_property('axis')

        """
        return self.get_property("axis", default=AttributeError())

    @axis.setter
    def axis(self, value):
        self.set_property("axis", value, copy=False)

    @axis.deleter
    def axis(self):
        self.del_property("axis")

    @property
    def positive(self):
        """The positive CF property.

        The direction of positive (i.e., the direction in which the
        coordinate values are increasing), whether up or down, cannot
        in all cases be inferred from the `units`. The direction of
        positive is useful for applications displaying the data. The
        `positive` attribute may have the value ``'up'`` or ``'down'``
        (case insensitive).

        For example, if ocean depth coordinates encode the depth of
        the surface as 0 and the depth of 1000 meters as 1000 then the
        `postive` property will have the value `'down'`.

        **Examples**

        >>> c.positive = 'up'
        >>> c.positive
        'up'
        >>> del c.positive

        >>> c.set_property('positive', 'down')
        >>> c.get_property('positive')
        'down'
        >>> c.del_property('positive')

        """
        return self.get_property("positive", default=AttributeError())

    @positive.setter
    def positive(self, value):
        self.set_property("positive", value, copy=False)
        self._direction = None

    @positive.deleter
    def positive(self):
        self.del_property("positive")
        self._direction = None

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    @_inplace_enabled(default=False)
    def autoperiod(self, inplace=False, config={}):
        """Set the period of cyclicity where it can be determined.

        TODO A dimension is set to be cyclic if it has a unique
        longitude (or grid longitude) dimension coordinate construct
        with bounds and the first and last bounds values differ by 360
        degrees (or an equivalent amount in other units).

        .. versionadded:: 3.5.0

        .. seealso:: `isperiodic`, `period`

        :Parameters:

            TODO

            config: `dict`
                Additional parameters for optimising the
                operation. See the code for details.

                .. versionadded:: 3.9.0

        :Returns:

            TODO

        **Examples**

        TODO

        """
        c = _inplace_enabled_define_and_cleanup(self)

        if "cyclic" in config and not config["cyclic"]:
            return c

        if "X" in config:
            X = config["X"]
            if not X:
                return c
        else:
            X = None

        if c.period() is not None:
            return c

        if X is None and not (
            c.Units.islongitude
            or c.get_property("standard_name", None) == "grid_longitude"
        ):
            return c

        period = config.get("period")
        if period is None:
            units = c.Units
            if units.islongitude:
                period = Data(360.0, units="degrees_east")
            else:
                period = Data(360.0, units="degrees")

            period.Units = units

        c.period(period=period)

        return c

    @_deprecated_kwarg_check(
        "relaxed_identity", version="3.0.0", removed_at="4.0.0"
    )
    def identity(
        self,
        default="",
        strict=False,
        relaxed=False,
        nc_only=False,
        relaxed_identity=None,
        _ctype=True,
    ):
        """Return the canonical identity.

        By default the identity is the first found of the following:

        * The "standard_name" property.
        * The "id" attribute, preceded by ``'id%'``.
        * The "cf_role" property, preceded by ``'cf_role='``.
        * The "axis" property, preceded by ``'axis='``.
        * The "long_name" property, preceded by ``'long_name='``.
        * The netCDF variable name, preceded by ``'ncvar%'``.
        * The coordinate type (``'X'``, ``'Y'``, ``'Z'`` or ``'T'``).
        * The value of the *default* parameter.

        .. versionadded:: 3.0.0

        .. seealso:: `id`, `identities`

        :Parameters:

            default: optional
                If no identity can be found then return the value of
                the default parameter.

            strict: `bool`, optional
                If True then the identity is the first found of only
                the "standard_name" property or the "id" attribute.

            relaxed: `bool`, optional
                If True then the identity is the first found of only
                the "standard_name" property, the "id" attribute, the
                "long_name" property or the netCDF variable name.

            nc_only: `bool`, optional
                If True then only take the identity from the netCDF
                variable name.

            relaxed_identity: deprecated at version 3.0.0

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
        out = super().identity(
            strict=strict, relaxed=relaxed, nc_only=nc_only, default=None
        )
        if out is not None:
            return out

        ctype = self.ctype
        if ctype is not None:
            return ctype

        return default

    def identities(self, generator=False, ctypes=None, **kwargs):
        """Return all possible identities.

        The identities comprise:

        * The "standard_name" property.
        * The "id" attribute, preceded by ``'id%'``.
        * The "cf_role" property, preceded by ``'cf_role='``.
        * The "axis" property, preceded by ``'axis='``.
        * The "long_name" property, preceded by ``'long_name='``.
        * All other properties (including "standard_name"), preceded by
          the property name and an ``'='``.
        * The coordinate type (``'X'``, ``'Y'``, ``'Z'`` or ``'T'``).
        * The netCDF variable name, preceded by ``'ncvar%'``.

        .. versionadded:: 3.0.0

        .. seealso:: `id`, `identity`

        :Parameters:

            {{generator: `bool`, optional}}

            ctypes: (sequence of) `str`
                If set then return the coordinate type (if any) as the
                first identity and restrict the possible coordinate
                types to be any of these characters. By default, a
                coordinate type is the last identity. Setting to a
                subset of ``'XTYZ'`` can give performance
                improvements, as it will reduce the number of
                coordinate types that are checked in circumstances
                when particular coordinate types have been ruled out a
                priori. If a coordinate type is omitted then it will
                not be in the returned identities even if the
                coordinate construct is of that type. Coordinate types
                are checked in the order given.

                *Parameter example:*
                  ``ctypes='Y'``

                *Parameter example:*
                  ``ctypes='XY'``

                *Parameter example:*
                  ``ctypes=('T', 'X')``

        :Returns:

            `list`
                The identities.

        **Examples**

        >>> f.properties()
        {'foo': 'bar',
         'long_name': 'Air Temperature',
         'standard_name': 'air_temperature'}
        >>> f.nc_get_variable()
        'tas'
        >>> f.identities()
        ['air_temperature',
         'long_name=Air Temperature',
         'foo=bar',
         'standard_name=air_temperature',
         'ncvar%tas']

        """
        if ctypes:
            pre = (self._ctypes_iter(ctypes),)
            pre0 = kwargs.pop("pre", None)
            if pre0:
                pre = tuple(pre0) + pre

            kwargs["pre"] = pre
        else:
            post = (self._ctypes_iter("XTYZ"),)
            post0 = kwargs.pop("post", None)
            if post0:
                post += tuple(post0)

            kwargs["post"] = post

        return super().identities(generator=generator, **kwargs)

    def _ctypes_iter(self, ctypes):
        """Generator for returning the coordinate type letter."""
        for c in ctypes:
            if getattr(self, c):
                # This coordinate construct is of this type
                yield c
                return
