from ..decorators import (_inplace_enabled,
                          _inplace_enabled_define_and_cleanup,)

from ..data.data import Data


class Coordinate():
    '''Mixin class for dimension or auxiliary coordinate constructs.

    .. versionadded:: 3.2.0

    '''
    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def ctype(self):
        '''The CF coordinate type.

    One of ``'T'``, ``'X'``, ``'Y'`` or ``'Z'`` if the coordinate
    construct is for the respective CF axis type, otherwise `None`.

    .. seealso:: `T`, `X`, `~cf.Coordinate.Y`, `Z`

    **Examples:**

    >>> c.X
    True
    >>> c.ctype
    'X'

    >>> c.T
    True
    >>> c.ctype
    'T'

        '''
        for t in ('T', 'X', 'Y', 'Z'):
            if getattr(self, t):
                return t

    @property
    def T(self):
        '''True if and only if the data are coordinates for a CF 'T' axis.

    CF 'T' axis coordinates are defined by having one or more of the
    following:

      * The `axis` property has the value ``'T'``
      * Units of latitude

    .. seealso:: `ctype`, `X`, `~cf.Coordinate.Y`, `Z`

    **Examples:**

    >>> c.Units
    <CF Units: seconds since 1992-10-8>
    >>> c.T
    True

        '''
        out = (self.Units.isreftime or
               self.get_property('axis', None) == 'T')

        if out:
            return True

        # Still here? Then check the bounds.
        if self.has_bounds():
            bounds = self.get_bounds(None)
            if bounds is not None:
                return bounds.T
        # --- End: if

        return False

    @property
    def X(self):
        '''True if and only if the data are coordinates for a CF 'X' axis.

    CF 'X' axis coordinates are defined by having one or more of the
    following:

      * The `axis` property has the value ``'X'``
      * Units of longitude
      * The `standard_name` property is one of ``'longitude'``,
        ``'projection_x_coordinate'`` or ``'grid_longitude'``

    .. seealso:: `ctype`, `T`, `~cf.Coordinate.Y`, `Z`

    **Examples:**

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

        '''
        standard_names = ('longitude',
                          'projection_x_coordinate',
                          'grid_longitude')
        units = self.Units
        out = (units.islongitude or
               self.get_property('axis', None) == 'X' or
               self.get_property('standard_name', None) in standard_names)

        if out:
            return True

        # Still here? Then check the bounds.
        if self.has_bounds():
            bounds = self.get_bounds(None)
            if bounds is not None:
                return bounds.X
        # --- End: if

        return False

    @property
    def Y(self):
        '''True if and only if the data are coordinates for a CF 'Y' axis.

    CF 'Y' axis coordinates are defined by having one or more of the
    following:

      * The `axis` property has the value ``'Y'``
      * Units of latitude
      * The `standard_name` property is one of ``'latitude'``,
        ``'projection_y_coordinate'`` or ``'grid_latitude'``

    .. seealso:: `ctype`, `T`, `X`, `Z`

    **Examples:**

    >>> c.Units
    <CF Units: degree_north>
    >>> c.Y
    True

    >>> c.standard_name == 'latitude'
    >>> c.Y
    True

        '''
        standard_names = ('latitude',
                          'projection_y_coordinate',
                          'grid_latitude')

        units = self.Units
        out = (units.islatitude or
               self.get_property('axis', None) == 'Y' or
               self.get_property('standard_name', None) in standard_names)

        if out:
            return True

        # Still here? Then check the bounds.
        if self.has_bounds():
            bounds = self.get_bounds(None)
            if bounds is not None:
                return bounds.Y
        # --- End: if

        return False

    @property
    def Z(self):
        '''True if and only if the data are coordinates for a CF 'Z' axis.

    CF 'Z' axis coordinates are defined by having one or more of the
    following:

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

    **Examples:**

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

        '''
        standard_names = ('atmosphere_ln_pressure_coordinate',
                          'atmosphere_sigma_coordinate',
                          'atmosphere_hybrid_sigma_pressure_coordinate',
                          'atmosphere_hybrid_height_coordinate',
                          'atmosphere_sleve_coordinate',
                          'ocean_sigma_coordinate',
                          'ocean_s_coordinate',
                          'ocean_s_coordinate_g1',
                          'ocean_s_coordinate_g2',
                          'ocean_sigma_z_coordinate',
                          'ocean_double_sigma_coordinate')

        units = self.Units
        out = (
            units.ispressure or
            (str(self.get_property('positive', 'Z')).lower()
             in ('up', 'down')) or
            self.get_property('axis', None) == 'Z' or
            (units and units.units in ('level', 'layer' 'sigma_level')) or
            self.get_property('standard_name', None) in standard_names
        )

        if out:
            return True

        # Still here? Then check the bounds.
        if self.has_bounds():
            bounds = self.get_bounds(None)
            if bounds is not None:
                return bounds.Z
        # --- End: if

        return False

    # ----------------------------------------------------------------
    # CF properties
    # ----------------------------------------------------------------
    @property
    def axis(self):
        '''The axis CF property.

    The `axis` property may be used to specify the type of
    coordinates. It may take one of the values `'X'`, `'Y'`, `'Z'` or
    `'T'` which stand for a longitude, latitude, vertical, or time
    axis respectively. A value of `'X'`, `'Y'` or `'Z'` may also also
    used to identify generic spatial coordinates (the values `'X'` and
    `'Y'` being used to identify horizontal coordinates).

    **Examples:**

    >>> c.axis = 'Y'
    >>> c.axis
    'Y'
    >>> del c.axis

    >>> c.set_property('axis', 'T')
    >>> c.get_property('axis')
    'T'
    >>> c.del_property('axis')

        '''
        return self.get_property('axis', default=AttributeError())

    @axis.setter
    def axis(self, value):
        self.set_property('axis', value)

    @axis.deleter
    def axis(self):
        self.del_property('axis')

    @property
    def positive(self):
        '''The positive CF property.

    The direction of positive (i.e., the direction in which the
    coordinate values are increasing), whether up or down, cannot in
    all cases be inferred from the `units`. The direction of positive
    is useful for applications displaying the data. The `positive`
    attribute may have the value ``'up'`` or ``'down'`` (case
    insensitive).

    For example, if ocean depth coordinates encode the depth of the
    surface as 0 and the depth of 1000 meters as 1000 then the
    `postive` property will have the value `'down'`.

    **Examples:**

    >>> c.positive = 'up'
    >>> c.positive
    'up'
    >>> del c.positive

    >>> c.set_property('positive', 'down')
    >>> c.get_property('positive')
    'down'
    >>> c.del_property('positive')

        '''
        return self.get_property('positive', default=AttributeError())

    @positive.setter
    def positive(self, value):
        self.set_property('positive', value)
        self._direction = None

    @positive.deleter
    def positive(self):
        self.del_property('positive')
        self._direction = None

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    @_inplace_enabled
    def autoperiod(self, inplace=False):
        '''TODO Set dimensions to be cyclic.

    TODO A dimension is set to be cyclic if it has a unique longitude (or
    grid longitude) dimension coordinate construct with bounds and the
    first and last bounds values differ by 360 degrees (or an
    equivalent amount in other units).

    .. versionadded:: 3.5.0

    .. seealso:: `isperiodic`, `period`

    :Parameters:

        TODO

    :Returns:

        TODO

    **Examples:**

    TODO

        '''
        c = _inplace_enabled_define_and_cleanup(self)

        if c.period() is not None:
            return c

        if not (c.Units.islongitude
                or c.get_property('standard_name', None) == 'grid_longitude'):
            return c

        c.period(Data(360.0, units=c.Units))

        return c

# --- End: class
