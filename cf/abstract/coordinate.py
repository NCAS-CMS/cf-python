from .. import mixin

from ..data.data import Data


class Coordinate(mixin.PropertiesDataBounds):
    '''Mixin class for dimension or auxiliary coordinate constructs.

.. versionadded:: 3.0.0

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
        
CF 'T' axis coordinates are defined by having units of reference time.

.. seealso:: `ctype`, `X`, `~cf.Coordinate.Y`, `Z`

**Examples:**

>>> c.Units
<CF Units: seconds since 1992-10-8>
>>> c.T
True

        '''
        return self.Units.isreftime
    #--- End: def

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

>>> c.Units
<CF Units: degreeE>
>>> c.X
True
 
>>> c.standard_name
'longitude'
>>> c.X
True

>>> c.axis == 'X' and c.X
True

        '''
#        data  = self.get_data(None)
#        if data is not None and data.ndim > 1:
#            return self.get_property('axis', None) == 'X'
            
        return (self.Units.islongitude or
                self.get_property('axis', None) == 'X' or
                self.get_property('standard_name', None) in ('longitude',
                                                             'projection_x_coordinate',
                                                             'grid_longitude'))
    #--- End: def

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
#        if self.ndim > 1:
#            return self.get_property('axis', None) == 'Y'

        return (self.Units.islatitude or 
                self.get_property('axis', None) == 'Y' or 
                self.get_property('standard_name', 'Y') in ('latitude',
                                                            'projection_y_coordinate',
                                                            'grid_latitude'))
    #--- End: def

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
        if self.ndim > 1:
            return self.get_property('axis', None) == 'Z'
        
        units = self.Units
        if (units.ispressure or
            str(self.get_property('positive', 'Z')).lower() in ('up', 'down') or
            self.get_property('axis', None) == 'Z' or
            (units and units.units in ('level', 'layer' 'sigma_level')) or
            self.get_property('standard_name', None) in
            ('atmosphere_ln_pressure_coordinate',
             'atmosphere_sigma_coordinate',
             'atmosphere_hybrid_sigma_pressure_coordinate',
             'atmosphere_hybrid_height_coordinate',
             'atmosphere_sleve_coordinate',
             'ocean_sigma_coordinate',
             'ocean_s_coordinate',
             'ocean_s_coordinate_g1',
             'ocean_s_coordinate_g2',
             'ocean_sigma_z_coordinate',
             'ocean_double_sigma_coordinate')):
            return True
        else:
            return False
    #--- End: def

    # ----------------------------------------------------------------
    # CF properties
    # ----------------------------------------------------------------
    @property
    def axis(self):
        '''The axis CF property.

The `axis` property may be used to specify the type of coordinates. It
may take one of the values `'X'`, `'Y'`, `'Z'` or `'T'` which stand
for a longitude, latitude, vertical, or time axis respectively. A
value of `'X'`, `'Y'` or `'Z'` may also also used to identify generic
spatial coordinates (the values `'X'` and `'Y'` being used to identify
horizontal coordinates).

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
    #--- End: def
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
#    def period(self, *value):
#        '''Set the period for cyclic coordinates.
#
#:Parameters:
#
#    value: data-like or `None`, optional
#        The period. The absolute value is used.
#
#        {+data-like-scalar}
#
#:Returns:
#
#    out: `cf.Data` or `None`
#        The period prior to the change, or the current period if no
#        *value* was specified. In either case, None is returned if the
#        period had not been set previously.
#
#**Examples:**
#
#>>> print c.period()
#None
#>>> c.Units
#<CF Units: degrees_east>
#>>> print c.period(360)
#None
#>>> c.period()
#<CF Data: 360.0 'degrees_east'>
#>>> import math
#>>> c.period(cf.Data(2*math.pi, 'radians'))
#<CF Data: 360.0 degrees_east>
#>>> c.period()
#<CF Data: 6.28318530718 radians>
#>>> c.period(None)
#<CF Data: 6.28318530718 radians>
#>>> print c.period()
#None
#>>> print c.period(-360)
#None
#>>> c.period()
#<CF Data: 360.0 degrees_east>
#
#        '''     
#        old = self._period
#        if old is not None:
#            old = old.copy()
#
#        if not value:
#            return old
#
#        value = value[0]
#
#        if value is not None:
#            value = Data.asdata(value)
#            units = value.Units
#            if not units:
#                value = value.override_units(self.Units)
#            elif units != self.Units:
#                if units.equivalent(self.Units):
#                    value.Units = self.Units
#                else:
#                    raise ValueError(
#"Period units {!r} are not equivalent to coordinate units {!r}".format(
#    units, self.Units))
#            #--- End: if
#
#            value = abs(value)
#            value.dtype = float
#            
#            if self.isdimension:
#                # Faster than `range`
#                array = self.array
#                r =  abs(array[-1] - array[0])
#            else:
#                r = self.data.range().datum(0)
#
#            if r >= value.datum(0):
#                raise ValueError(
#"The coordinate range {!r} is not less than the period {!r}".format(
#    range, value))
#        #--- End: if
#
#        self._period = value
#
#        return old
#    #--- End: def

#--- End: class
