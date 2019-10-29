from functools import partial as functools_partial
#from netCDF4   import default_fillvals

from numpy import array       as numpy_array
from numpy import result_type as numpy_result_type
from numpy import vectorize   as numpy_vectorize


from ..cfdatetime   import dt
from ..functions    import equivalent as cf_equivalent
from ..functions    import inspect    as cf_inspect
from ..functions    import default_netCDF_fillvals
from ..query        import Query
from ..timeduration import TimeDuration
from ..units        import Units

from ..data.data import Data

from . import Properties

from ..functions import (_DEPRECATION_ERROR_KWARGS,
                         _DEPRECATION_ERROR_METHOD,
                        )


_units_None = Units()

_month_units = ('month', 'months')
_year_units  = ('year', 'years', 'yr')

_relational_methods = ('__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__')


class PropertiesData(Properties):
    '''Mixin class for a data array with metadata.

    '''
    _special_properties = ('units',
                           'calendar')

#    def __init__(self, properties=None, data=None, source=None,
#                 copy=True, _use_data=True):
#        '''**Initialization**
#
#    :Parameters:
#    
#        properties: `dict`, optional
#            Set descriptive properties. The dictionary keys are
#            property names, with corresponding values. Ignored if the
#            *source* parameter is set.
#    
#            *Parameter example:*
#              ``properties={'standard_name': 'altitude'}``
#            
#            Properties may also be set after initialisation with the
#            `set_properties` and `set_property` methods.
#    
#        data: `Data`, optional
#            Set the data. Ignored if the *source* parameter is set.
#            
#            The data also may be set after initialisation with the
#            `set_data` method.
#            
#        source: optional
#            Initialize the properties and data from those of *source*.
#    
#        copy: `bool`, optional
#
#            If False then do not deep copy input parameters prior to
#            initialization. By default arguments are deep copied.
#
#        '''
#        if _use_data and data is not None and properties:
#            if not data.Units:
#                units = properties.get('units')
#                if units is not None:
#                    data = data.override_units(Units(units, properties.get('calendar')))
#        #--- End: if
#        
#        super().__init__(properties=properties, data=data, source=source,
#                         copy=copy, _use_data=_use_data)

        
    def __array__(self, *dtype):
        '''Returns a numpy array representation of the data.

        '''
        data = self.get_data(None)
        if data is not None:
            return data.__array__(*dtype)

        raise ValueError("{} has no data".format(self.__class__.__name__))


    def __contains__(self, value):
        '''Called to implement membership test operators.

    x.__contains__(y) <==> y in x

        '''
        data = self.get_data(None)
        if data is None:
            return False
        
        return value in data


    def __data__(self):
        '''Returns a new reference to the data.

    Allows the construct to initialize a `Data` object.

    :Returns:

        `Data`

    **Examples:**

    >>> f.data
    <CF Data(12): [14, ..., 56] km)
    >>> cf.Data(f)
    <CF Data(12): [14, ..., 56] km)
    >>> cf.Data.asdata(f)
    <CF Data(12): [14, ..., 56] km)

        '''
        data =self.get_data(None)
        if data is not None:
            return data

        raise ValueError("{} has no data".format(self.__class__.__name__))

    
    def __setitem__(self, indices, value):
        '''Called to implement assignment to x[indices]

    x.__setitem__(indices, value) <==> x[indices]

        '''
        data = self.get_data(None)
        if data is None:
            raise ValueError("Can't set elements when there is no data")
        
        if isinstance(value, self.__class__):        
            value = value.data
        
        data[indices] = value


    def __add__(self, y):
        '''The binary arithmetic operation ``+``

    x.__add__(y) <==> x+y

        '''        
        return self._binary_operation(y, '__add__')


    def __iadd__(self, y):
        '''The augmented arithmetic assignment ``+=``

    x.__iadd__(y) <==> x+=y

        '''
        return self._binary_operation(y, '__iadd__')


    def __radd__(self, y):
        '''The binary arithmetic operation ``+`` with reflected operands

    x.__radd__(y) <==> y+x

        '''
        return self._binary_operation(y, '__radd__')


    def __sub__(self, y):
        '''The binary arithmetic operation ``-``

    x.__sub__(y) <==> x-y

        '''
        return self._binary_operation(y, '__sub__')


    def __isub__(self, y):
        '''The augmented arithmetic assignment ``-=``

    x.__isub__(y) <==> x-=y

        '''
        return self._binary_operation(y, '__isub__')


    def __rsub__(self, y):
        '''The binary arithmetic operation ``-`` with reflected operands

    x.__rsub__(y) <==> y-x

        '''    
        return self._binary_operation(y, '__rsub__')


    def __mul__(self, y):
        '''The binary arithmetic operation ``*``

    x.__mul__(y) <==> x*y

        '''
        return self._binary_operation(y, '__mul__')


    def __imul__(self, y):
        '''The augmented arithmetic assignment ``*=``

    x.__imul__(y) <==> x*=y

        '''
        return self._binary_operation(y, '__imul__')


    def __rmul__(self, y):
        '''The binary arithmetic operation ``*`` with reflected operands

    x.__rmul__(y) <==> y*x

        '''       
        return self._binary_operation(y, '__rmul__')


    def __div__(self, y):
        '''The binary arithmetic operation ``/``

    x.__div__(y) <==> x/y

        '''
        return self._binary_operation(y, '__div__')


    def __idiv__(self, y):
        '''The augmented arithmetic assignment ``/=``

    x.__idiv__(y) <==> x/=y

        '''
        return self._binary_operation(y, '__idiv__')

    
    def __rdiv__(self, y):
        '''The binary arithmetic operation ``/`` with reflected operands

    x.__rdiv__(y) <==> y/x

        '''
        return self._binary_operation(y, '__rdiv__')


    def __floordiv__(self, y):
        '''The binary arithmetic operation ``//``

    x.__floordiv__(y) <==> x//y

        '''     
        return self._binary_operation(y, '__floordiv__')


    def __ifloordiv__(self, y):
        '''The augmented arithmetic assignment ``//=``

    x.__ifloordiv__(y) <==> x//=y

        '''
        return self._binary_operation(y, '__ifloordiv__')


    def __rfloordiv__(self, y):
        '''The binary arithmetic operation ``//`` with reflected operands

    x.__rfloordiv__(y) <==> y//x

        '''
        return self._binary_operation(y, '__rfloordiv__')


    def __truediv__(self, y):
        '''The binary arithmetic operation ``/`` (true division)

    x.__truediv__(y) <==> x/y

        '''
        return self._binary_operation(y, '__truediv__')


    def __itruediv__(self, y):
        '''The augmented arithmetic assignment ``/=`` (true division)

    x.__itruediv__(y) <==> x/=y

        '''
        return self._binary_operation(y, '__itruediv__')


    def __rtruediv__(self, y):
        '''The binary arithmetic operation ``/`` (true division) with
    reflected operands

    x.__rtruediv__(y) <==> y/x

        '''    
        return self._binary_operation(y, '__rtruediv__')


    def __pow__(self, y, modulo=None):
        '''The binary arithmetic operations ``**`` and ``pow``

    x.__pow__(y) <==> x**y

        '''  
        if modulo is not None:
            raise NotImplementedError("3-argument power not supported for %r" %
                                      self.__class__.__name__)

        return self._binary_operation(y, '__pow__')


    def __ipow__(self, y, modulo=None):
        '''The augmented arithmetic assignment ``**=``

    x.__ipow__(y) <==> x**=y

        '''     
        if modulo is not None:
            raise NotImplementedError("3-argument power not supported for %r" %
                                      self.__class__.__name__)

        return self._binary_operation(y, '__ipow__')


    def __rpow__(self, y, modulo=None):
        '''The binary arithmetic operations ``**`` and ``pow`` with reflected
    operands

    x.__rpow__(y) <==> y**x

        '''       
        if modulo is not None:
            raise NotImplementedError("3-argument power not supported for %r" %
                                      self.__class__.__name__)

        return self._binary_operation(y, '__rpow__')


    def __mod__(self, y):
        '''The binary arithmetic operation ``%``

    x.__mod__(y) <==> x % y

    .. versionadded:: 1.0

        '''
        return self._binary_operation(y, '__mod__')


    def __imod__(self, y):
        '''The binary arithmetic operation ``%=``

    x.__imod__(y) <==> x %= y

    .. versionadded:: 1.0
        
        '''
        return self._binary_operation(y, '__imod__')


    def __rmod__(self, y):
        '''The binary arithmetic operation ``%`` with reflected operands

    x.__rmod__(y) <==> y % x

    .. versionadded:: 1.0

        '''
        return self._binary_operation(y, '__rmod__')


    def __eq__(self, y):
        '''The rich comparison operator ``==``

    x.__eq__(y) <==> x==y

        '''
        return self._binary_operation(y, '__eq__')


    def __ne__(self, y):
        '''The rich comparison operator ``!=``

    x.__ne__(y) <==> x!=y

        '''
        return self._binary_operation(y, '__ne__')

    
    def __ge__(self, y):
        '''The rich comparison operator ``>=``

    x.__ge__(y) <==> x>=y

        '''
        return self._binary_operation(y, '__ge__')


    def __gt__(self, y):
        '''The rich comparison operator ``>``

    x.__gt__(y) <==> x>y

        '''
        return self._binary_operation(y, '__gt__')


    def __le__(self, y):
        '''The rich comparison operator ``<=``

    x.__le__(y) <==> x<=y

        '''
        return self._binary_operation(y, '__le__')


    def __lt__(self, y):
        '''The rich comparison operator ``<``

    x.__lt__(y) <==> x<y

        '''
        return self._binary_operation(y, '__lt__')

    
    def __and__(self, y):
        '''The binary bitwise operation ``&``

    x.__and__(y) <==> x&y

        '''
        return self._binary_operation(y, '__and__')


    def __iand__(self, y):
        '''The augmented bitwise assignment ``&=``

    x.__iand__(y) <==> x&=y

        '''
        return self._binary_operation(y, '__iand__')


    def __rand__(self, y):
        '''The binary bitwise operation ``&`` with reflected operands

    x.__rand__(y) <==> y&x

        '''
        return self._binary_operation(y, '__rand__')


    def __or__(self, y):
        '''The binary bitwise operation ``|``

    x.__or__(y) <==> x|y

        '''
        return self._binary_operation(y, '__or__')


    def __ior__(self, y):
        '''The augmented bitwise assignment ``|=``

    x.__ior__(y) <==> x|=y

        '''
        return self._binary_operation(y, '__ior__')


    def __ror__(self, y):
        '''The binary bitwise operation ``|`` with reflected operands

    x.__ror__(y) <==> y|x

        '''
        return self._binary_operation(y, '__ror__')


    def __xor__(self, y):
        '''The binary bitwise operation ``^``

    x.__xor__(y) <==> x^y

        '''
        return self._binary_operation(y, '__xor__')


    def __ixor__(self, y):
        '''The augmented bitwise assignment ``^=``

    x.__ixor__(y) <==> x^=y

        '''
        return self._binary_operation(y, '__ixor__')


    def __rxor__(self, y):
        '''The binary bitwise operation ``^`` with reflected operands

    x.__rxor__(y) <==> y^x

        '''
        return self._binary_operation(y, '__rxor__')


    def __lshift__(self, y):
        '''The binary bitwise operation ``<<``

    x.__lshift__(y) <==> x<<y

        '''
        return self._binary_operation(y, '__lshift__')


    def __ilshift__(self, y):
        '''The augmented bitwise assignment ``<<=``

    x.__ilshift__(y) <==> x<<=y

        '''
        return self._binary_operation(y, '__ilshift__')


    def __rlshift__(self, y):
        '''The binary bitwise operation ``<<`` with reflected operands

    x.__rlshift__(y) <==> y<<x

        '''
        return self._binary_operation(y, '__rlshift__')


    def __rshift__(self, y):
        '''The binary bitwise operation ``>>``

    x.__lshift__(y) <==> x>>y
        
        '''
        return self._binary_operation(y, '__rshift__')


    def __irshift__(self, y):
        '''The augmented bitwise assignment ``>>=``

    x.__irshift__(y) <==> x>>=y

        '''
        return self._binary_operation(y, '__irshift__')


    def __rrshift__(self, y):
        '''The binary bitwise operation ``>>`` with reflected operands

    x.__rrshift__(y) <==> y>>x

        '''
        return self._binary_operation(y, '__rrshift__')


    def __abs__(self):
        '''The unary arithmetic operation ``abs``

    x.__abs__() <==> abs(x)

        '''       
        return self._unary_operation('__abs__')


    def __neg__(self):
        '''The unary arithmetic operation ``-``

    x.__neg__() <==> -x

        '''
        return self._unary_operation('__neg__')


    def __invert__(self):
        '''The unary bitwise operation ``~``

    x.__invert__() <==> ~x

        '''
        return self._unary_operation('__invert__')


    def __pos__(self):
        '''The unary arithmetic operation ``+``

    x.__pos__() <==> +x

        '''
        return self._unary_operation('__pos__')


    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------
    def _binary_operation(self, y, method):
        '''Implement binary arithmetic and comparison operations.

    The operations act on the constructs data with the numpy
    broadcasting rules.
    
    It is intended to be called by the binary arithmetic and
    comparison methods, such as `!__sub__` and `!__lt__`.
    
    :Parameters:
    
        operation: `str`
            The binary arithmetic or comparison method name (such as
            ``'__imul__'`` or ``'__ge__'``).
    
    :Returns:
    
            A new construct, or the same construct if the operation
            was in-place.
    
    **Examples:**
    
    >>> w = u._binary_operation(u, '__add__')
    >>> w = u._binary_operation(v, '__lt__')
    >>> u._binary_operation(2, '__imul__')
    >>> u._binary_operation(v, '__idiv__')

        '''
        data = self.get_data(None)
        if data is None:
            raise ValueError( 
                "Can't apply {} to a {} object with no data: {!r}".format(
                    method, self.__class__.__name__, self))

        inplace = method[2] == 'i'

        units = self.Units
        sn = self.get_property('standard_name', None)
        ln = self.get_property('long_name', None)

        try:
            other_sn = y.get_property('standard_name', None)
            other_ln = y.get_property('long_name', None)
        except AttributeError:
            other_sn = None
            other_ln = None
            
        if isinstance(y, self.__class__):
            y = y.data
        
        if not inplace:
            new = self.copy() #data=False) TODO
            new_data = data._binary_operation(y, method)
            new.set_data(new_data, copy=False)
        else:
            new = self
            new.data._binary_operation(y, method)

        if method in _relational_methods:
            # Booleans have no units
            new.override_units(Units(), inplace=True)
            
        # ------------------------------------------------------------
        # Remove misleading identities
        # ------------------------------------------------------------
        if sn != other_sn:
            if sn is not None and other_sn is not None:
                new.del_property('standard_name', None)
                new.del_property('long_name', None)
            elif other_sn is not None:
                new.set_property('standard_name', other_sn)
                if other_ln is None:
                    new.del_property('long_name', None)
                else:
                    new.set_property('long_name', other_ln)
        elif ln is None and other_ln is not None:
            new.set_property('long_name', other_ln)
        
        new_units = new.Units
        if (method in _relational_methods or
            not units.equivalent(new_units) and
            not (units.isreftime and new_units.isreftime)):
            new.del_property('standard_name', None)
            new.del_property('long_name', None)

        return new


#    def _ooo(self):
#        '''
#        '''
#        units = self.Units
#        sn = self.get_property('standard_name', None)
#        ln = self.get_property('long_name', None)
#
#        try:
#            other_sn = y.get_property('standard_name', None)
#            other_ln = y.get_property('long_name', None)
#        except AttributeError:
#            other_sn = None
#            other_ln = None
#            
#        if isinstance(y, self.__class__):
#            y = y.data
#        
#        if not inplace:
#            new = self.copy() #data=False) TODO
#            new_data = data._binary_operation(y, method)
#            new.set_data(new_data, copy=False)
#        else:
#            new = self
#            new.data._binary_operation(y, method)
#
#
#        if sn != other_sn:
#            if sn is not None and other_sn is not None:
#                new.del_property('standard_name', None)
#                new.del_property('long_name', None)
#            elif other_sn is not None:
#                new.set_property('standard_name', other_sn)
#                if other_ln is None:
#                    new.del_property('long_name', None)
#                else:
#                    new.set_property('long_name', other_ln)
#        elif ln is None and other_ln is not None:
#            new.set_property('long_name', other_ln)
#        
#        new_units = new.Units
#        if (not units.equivalent(new_units) and
#            not (units.isreftime and new_units.isreftime)):
#            new.del_property('standard_name', None)
#            new.del_property('long_name', None)   
        
#    def _change_axis_names(self, dim_name_map):
#        '''Change the axis names of the Data object.
#
#:Parameters:
#
#    dim_name_map: `dict`
#
#:Returns:
#
#    `None`
#
#**Examples:**
#
#>>> f._change_axis_names({'0': 'dim1', '1': 'dim2'})
#
#        '''
#        data = self.get_data(None)
#        if data is not None:
#            data.change_axis_names(dim_name_map)


    def _conform_for_assignment(self, other):
        '''TODO

        '''    
        return other


    def _equivalent_data(self, other, atol=None, rtol=None,
                         verbose=False):
        '''TODO

    Two real numbers ``x`` and ``y`` are considered equal if
    ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on absolute
    differences) and ``rtol`` (the tolerance on relative differences) are
    positive, typically very small numbers. See the *atol* and *rtol*
    parameters.
    
    :Parameters:
    
        transpose: `dict`, optional
    
        atol: `float`, optional
            The tolerance on absolute differences between real
            numbers. The default value is set by the `ATOL` function.
    
        rtol: `float`, optional
            The tolerance on relative differences between real
            numbers. The default value is set by the `RTOL` function.
    
    :Returns:
    
        `bool`
            Whether or not the two variables have equivalent data arrays.

        '''
        if self.has_data() != other.has_data():
            if verbose:
                print("{}: Only one construct has data: {!r}, {!r}".format(
                    self.__class__.__name__, self, other))
            return False

        if not self.has_data():
            return True

        data0 = self.get_data()
        data1 = other.get_data()

        if data0.shape != data1.shape:
            if verbose:
                print("{}: Data have different shapes: {}, {}".format(
                    self.__class__.__name__, data0.shape, data1.shape))
            return False              
 
        if not data0.Units.equivalent(data1.Units):
            if verbose:
                print("{}: Data have non-equivalent units: {!r}, {!r}".format(
                    self.__class__.__name__, data0.Units, data1.Units))
            return  False

#        if atol is None:
#            atol = ATOL()        
#        if rtol is None:
#            rtol = RTOL()
            
        if not data0.allclose(data1, rtol=rtol, atol=atol):
            if verbose:
                print("{}: Data have non-equivalent values: {!r}, {!r}".format(
                    self.__class__.__name__, data0, data1))
            return False

        return True


    def _parse_axes(self, axes):
        '''TODO

        '''
        if axes is None:
            return axes

        if isinstance(axes, int):
            axes = (axes,)
        
        ndim = self.ndim
        return [(i + ndim if i < 0 else i) for i in axes]

    
    def _parse_match(self, match):
        '''Called by `match`

    :Parameters:
    
        match: 
            As for the *match* parameter of `match` method.
    
    :Returns:
    
        `list`
        '''        
        if not match:
            return ()

        if isinstance(match, (str, dict, Query)):
            match = (match,)

        matches = []
        for m in match:            
            if isinstance(m, str):
                if '=' in m:
                    # CF property (string-valued)
                    m = m.split('=')
                    matches.append({m[0]: '='.join(m[1:])})
                else:
                    # Identity (string-valued) or python attribute
                    # (string-valued) or axis type
                    matches.append({None: m})

            elif isinstance(m, dict):
                # Dictionary
                matches.append(m)

            else:
                # Identity (not string-valued, e.g. cf.Query).
                matches.append({None: m})
        #--- End: for

        return matches


    def __query_set__(self, values):
        '''TODO

        '''
        new = self.copy()
        new.set_data(self.data.__query_set__(values), copy=False)
        return new


#    def _query_contain(self, value):
#        '''TODO#
#
#        '''
#        new = self.copy()
#        new.set_data(self.data._query_contain(value), copy=False)
#        return new


#    def _query_contains(self, value):
#        '''TODO
#
#        '''
#        new = self.copy()
#        new.set_data(self.data._query_contains(value), copy=False)
#        return new


    def __query_wi__(self, value):
        '''TODO

        '''
        new = self.copy()
        new.set_data(self.data.__query_wi__(value), copy=False)
        return new


    def __query_wo__(self, value):
        '''TODO
1
        '''
        new = self.copy()
        new.set_data(self.data.__query_wo__(value), copy=False)
        return new


    def _unary_operation(self, method):
        '''Implement unary arithmetic operations on the data array.

    :Parameters:
    
        method: `str`
            The unary arithmetic method name (such as "__abs__").
    
    :Returns:
    
            TODO
    
    **Examples:**
    
    >>> print(v.array)
    [1 2 -3 -4 -5]
    
    >>> w = v._unary_operation('__abs__')
    >>> print(w.array)
    [1 2 3 4 5]
    
    >>> w = v.__abs__()
    >>> print(w.array)
    [1 2 3 4 5]
    
    >>> w = abs(v)
    >>> print(w.array)
    [1 2 3 4 5]

        '''
        data = self.get_data(None)
        if data is None:
            raise ValueError("Can't apply {} to a {} with no data".format(
                method, self.__class__.__name__))

        new = self.copy()
        
        new_data = data._unary_operation(method)
        new.set_data(new_data, copy=False)
        
        return new


    def _YMDhms(self, attr):
        '''TODO
        '''
        data = self.get_data(None)
        if data is None:
            raise ValueError(
                "ERROR: Can't get {}s when there is no data array".format(attr))        
                
        out = self.copy() # data=False)

        out.set_data(getattr(data, attr), copy=False)
        
        out.del_property('standard_name', None)
        out.set_property('long_name', attr)

        out.override_units(Units(), inplace=True)
        
        return out


#    def _hmmm(self, method):
#        data = self.get_data(None)
#        if data is not None:
#            out = self.copy() #data=False)
#            out.set_data(getattr(data, method)(), copy=False)
#            out.del_property('standard_name', None)
#            out.set_property('long_name', method)
#            return out
#        #--- End: if
#
#        raise ValueError(
#            "ERROR: Can't get {0} when there is no data array".format(method))        


    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def T(self):
        '''`True` if and only if the data are coordinates for a CF 'T' axis.
        
    CF 'T' axis coordinates are defined by having units of reference
    time
    
    .. seealso:: `X`, `Y`, `Z`
    
    **Examples:**
    
    >>> c.T
    False

        '''              
        return False


    @property
    def X(self):
        '''Always False.

    .. seealso:: `T`, `Y`, `Z`
    
    **Examples:**
    
    >>> print(f.X)
    False

        '''              
        return False


    @property
    def Y(self):
        '''Always False.

    .. seealso:: `T`, `X`, `Z`
    
    **Examples:**
    
    >>> print(f.Y)
    False

        '''              
        return False


    @property
    def Z(self):
        '''Always False.

    .. seealso:: `T`, `X`, `Y`
    
    **Examples:**
    
    >>> print(f.Z)
    False

        '''              
        return False


    @property
    def binary_mask(self):
        '''A binary (0 and 1) missing data mask of the data array.

    The binary mask's data array comprises dimensionless 32-bit
    integers and has 0 where the data array has missing data and 1
    otherwise.
    
    **Examples:**
    
    >>> print(f.mask.array)
    [[ True  False  True False]]
    >>> b = f.binary_mask()
    >>> print(b.array)
    [[0 1 0 1]]

        '''
        out = type(self)()
        out.set_propoerty('long_name', 'binary_mask')
        out.set_data(self.data.binary_mask(), copy=False)
        return out

    
    @property
    def data(self):
        '''The `Data` object containing the data array.

    * ``f.data = x`` is equivalent to ``f.set_data(x, copy=False)``
    
    * ``x = f.data`` is equivalent to ``x = f.get_data()``
    
    * ``del f.data`` is equivalent to ``f.del_data()``
    
    * ``hasattr(f, 'data')`` is equivalent to ``f.has_data()``
    
    .. seealso:: `del_data`, `get_data`, `has_data`, `set_data`

        '''
        return self.get_data()
    @data.setter
    def data(self, value):
        self.set_data(value, set_axes=False, copy=False)
    @data.deleter
    def data(self):
        return self.del_data()


    @property
    def reference_datetime(self):
        '''The reference date-time of units of elapsed time.

    **Examples**

    >>> f.units
    'days since 2000-1-1'
    >>> f.reference_datetime
    cftime.DatetimeNoLeap(2000-01-01 00:00:00)

        '''
        units = self.Units
        if not units.isreftime:
            raise AttributeError(
                "{0} doesn't have attribute 'reference_datetime'".format(
                    self.__class__.__name__))
        return dt(units.reftime, calendar=units._calendar)

    @reference_datetime.setter
    def reference_datetime(self, value):
        units = self.Units
        if not units.isreftime:
            raise AttributeError(
                "Can't set 'reference_datetime' for non reference date-time units".format(
                    self.__class__.__name__))

        units = units.units.split(' since ')
        try:
            self.units = "{0} since {1}".format(units[0], value)
        except (ValueError, TypeError):
            raise ValueError(
                "Can't override reference date-time {0!r} with {1!r}".format(
                    units[1], value))


    @property
    def Units(self):
        '''The `cf.Units` object containing the units of the data array.

    Stores the units and calendar CF properties in an internally
    consistent manner. These are mirrored by the `units` and
    `calendar` CF properties respectively.
    
    **Examples:**
    
    >>> f.Units
    <Units: K>
    
    >>> f.Units
    <Units: days since 2014-1-1 calendar=noleap>

        '''
        data = self.get_data(None)
        if data is not None:
            return data.Units
        
        try:
            return self._custom['Units']
        except KeyError:
            self._custom['Units'] = _units_None
            return _units_None

    @Units.setter
    def Units(self, value):
        data = self.get_data(None)
        if data is not None:
            data.Units = value
        else:
            self._custom['Units'] = value

#        units = getattr(value, 'units', None)
#        if units is not None:
#            self.set_property('units', units)
#    
#        calendar = getattr(value, 'calendar', None)
#        if calendar is not None:
#            self.set_property('calendar', calendar)

    @Units.deleter
    def Units(self):
        raise AttributeError(
            "Can't delete {} attribute 'Units'. Use the override_units method.".format(
                self.__class__.__name__))


    @property
    def year(self):
        '''The year of each date-time data array element.

    Only applicable to data arrays with reference time units.
    
    .. seealso:: `month`, `day`, `hour`, `minute`, second`
    
    **Examples:**
    
    >>> print(f.datetime_array)
    [0450-11-15 00:00:00  0450-12-16 12:30:00  0451-01-16 12:00:45]
    >>> print(f.year.array)
    [450  450  451]

        ''' 
        return self._YMDhms('year')

    
    @property
    def month(self):
        '''The month of each date-time data array element.

    Only applicable to data arrays with reference time units.
    
    .. seealso:: `year`, `day`, `hour`, `minute`, second`
    
    **Examples:**
    
    >>> print(f.datetime_array)
    [0450-11-15 00:00:00  0450-12-16 12:30:00  0451-01-16 12:00:45]
    >>> print(f.month.array)
    [11  12  1]

        '''
        return self._YMDhms('month')


    @property
    def day(self):
        '''The day of each date-time data array element.

    Only applicable to data arrays with reference time units.
    
    .. seealso:: `year`, `month`, `hour`, `minute`, second`
    
    **Examples:**
    
    >>> print(f.datetime_array)
    [0450-11-15 00:00:00  0450-12-16 12:30:00  0451-01-16 12:00:45]
    >>> print(f.day.array)
    [15  16  16]

        '''
        return self._YMDhms('day')


    @property
    def hour(self):
        '''The hour of each date-time data array element.
    
    Only applicable to data arrays with reference time units.
    
    .. seealso:: `year`, `month`, `day`, `minute`, second`
    
    **Examples:**
    
    >>> print(f.datetime_array)
    [0450-11-15 00:00:00  0450-12-16 12:30:00  0451-01-16 12:00:45]
    >>> print(f.hour.array)
    [ 0  12  12]

        '''
        return self._YMDhms('hour')

    
    @property
    def minute(self):
        '''The minute of each date-time data array element.

    Only applicable to data arrays with reference time units.
    
    .. seealso:: `year`, `month`, `day`, `hour`, second`
    
    **Examples:**
    
    >>> print(f.datetime_array)
    [0450-11-15 00:00:00  0450-12-16 12:30:00  0451-01-16 12:00:45]
    >>> print(f.minute.array)
    [ 0 30  0]

        '''
        return self._YMDhms('minute')


    @property
    def second(self):
        '''The second of each date-time data array element.

    Only applicable to data arrays with reference time units.
    
    .. seealso:: `year`, `month`, `day`, `hour`, `minute`
    
    **Examples:**
    
    >>> print(f.datetime_array)
    [0450-11-15 00:00:00  0450-12-16 12:30:00  0451-01-16 12:00:45]
    >>> print(f.second.array)
    [ 0  0 45]

        '''
        return self._YMDhms('second')


    @property
    def mask(self):
        '''The mask of the data array.

    Values of True indicate masked elements.
    
    .. seealso:: `binary_mask`
    
    **Examples:**
    
    >>> f.shape
    (12, 73, 96)
    >>> m = f.mask
    >>> m.long_name
    'mask'
    >>> m.shape
    (12, 73, 96)
    >>> m.dtype
    dtype('bool')
    >>> m.data
    <CF Data(12, 73, 96): [[[True, ..., False]]] >

        '''
        if not self.has_data():
            raise ValueError(
                "ERROR: Can't get mask when there is no data array")

        out = self.copy()

        out.set_data(self.data.mask, copy=False)

        out.override_units(Units(), inplace=True)
        
        out.clear_properties()
        out.set_property('long_name', 'mask')

        out.nc_del_variable(default=None)

        return out

    
    # ----------------------------------------------------------------
    # CF properties
    # ----------------------------------------------------------------
    @property
    def add_offset(self):
        '''The add_offset CF property.

    If present then this number is *subtracted* from the data prior to
    it being written to a file. If both `scale_factor` and
    `add_offset` properties are present, the offset is subtracted
    before the data are scaled. See
    http://cfconventions.org/latest.html for details.
    
    **Examples:**
    
    >>> f.add_offset = -4.0
    >>> f.add_offset
    -4.0
    >>> del f.add_offset
    
    >>> f.set_property('add_offset', 10.5)
    >>> f.get_property('add_offset')
    10.5
    >>> f.del_property('add_offset')
    10.5
    >>> f.has_property('add_offset')
    False

        '''
        return self.get_property('add_offset', default=AttributeError())

    @add_offset.setter
    def add_offset(self, value):
        self.set_property('add_offset', value)
        self.dtype = numpy_result_type(self.dtype, numpy_array(value).dtype)

    @add_offset.deleter
    def add_offset(self):
        self.delprop('add_offset', default=AttributeError())
        if not self.has_property('scale_factor'):
            del self.dtype


    @property
    def calendar(self):
        '''The calendar CF property.

    The calendar used for encoding time data. See
    http://cfconventions.org/latest.html for details.
    
    **Examples:**
    
    >>> f.calendar = 'noleap'
    >>> f.calendar
    'noleap'
    >>> del f.calendar
    
    >>> f.set_property('calendar', 'proleptic_gregorian')
    >>> f.get_property('calendar')
    'proleptic_gregorian'
    >>> f.del_property('calendar')
    'proleptic_gregorian'
    >>> f.has_property('calendar')
    False

        '''
        value = getattr(self.Units, 'calendar', None)
        if value is None:
            raise AttributeError(
                "{} doesn't have CF property 'calendar'".format(
                    self.__class__.__name__))
        return value


    @calendar.setter
    def calendar(self, value):
        self.Units = Units(getattr(self, 'units', None), value)


    @calendar.deleter
    def calendar(self):
        if getattr(self, 'calendar', None) is None:
            raise AttributeError(
                "Can't delete non-existent {} CF property 'calendar'".format(
                    self.__class__.__name__))
        
        self.Units = Units(getattr(self, 'units', None))
    
    @property
    def _FillValue(self):
        '''The _FillValue CF property.

    A value used to represent missing or undefined data.
    
    Note that this property is primarily for writing data to disk and
    is independent of the missing data mask. It may, however, get used
    when unmasking data array elements. See
    http://cfconventions.org/latest.html for details.
    
    The recommended way of retrieving the missing data value is with
    the `fill_value` method.
    
    .. seealso:: `fill_value`, `missing_value`,
                 `cf.default_netCDF_fillvals`
    
    **Examples:**
    
    >>> f._FillValue = -1.0e30
    >>> f._FillValue
    -1e+30
    >>> del f._FillValue

    >>> f.set_property('_FillValue', -1.0e30)
    >>> f.get_property('_FillValue')
    -1e+30
    >>> f.del_property('_FillValue')
    -1e30
    >>> f.del_property('_FillValue', None)
    None

        '''
        return self.get_property('_FillValue', default=AttributeError())
    @_FillValue.setter
    def _FillValue(self, value):
        self.set_property('_FillValue', value)
    @_FillValue.deleter
    def _FillValue(self):
        self.del_property('_FillValue', default=AttributeError())


    @property
    def missing_value(self):
        '''The missing_value CF property.

    A value used to represent missing or undefined data (deprecated by
    the netCDF user guide). See http://cfconventions.org/latest.html
    for details.
    
    Note that this attribute is used primarily for writing data to
    disk and is independent of the missing data mask. It may, however,
    be used when unmasking data array elements.
    
    The recommended way of retrieving the missing data value is with
    the `fill_value` method.
    
    .. seealso:: `_FillValue`, `fill_value`,
                 `cf.default_netCDF_fillvals`
    
    **Examples:**
    
    >>> f.missing_value = 1.0e30
    >>> f.missing_value
    1e+30
    >>> del f.missing_value
        
    >>> f.set_property('missing_value', -1.0e30)
    >>> f.get_property('missing_value')
    -1e+30              
    >>> f.del_property('missing_value')
    -1e30               
    >>> f.del_property('missing_value', None)
    None

        '''
        return self.get_property('missing_value', default=AttributeError())
    @missing_value.setter
    def missing_value(self, value):
        self.set_property('missing_value', value)
    @missing_value.deleter
    def missing_value(self):
        self.del_property('missing_value', default=AttributeError())


    @property
    def scale_factor(self):
        '''The scale_factor CF property.

    If present then the data are *divided* by this factor prior to it
    being written to a file. If both `scale_factor` and `add_offset`
    properties are present, the offset is subtracted before the data
    are scaled. See http://cfconventions.org/latest.html for details.
    
    **Examples:**
    
    >>> f.scale_factor = 10.0
    >>> f.scale_factor
    10.0
    >>> del f.scale_factor
    
    >>> f.set_property('scale_factor', 10.0)
    >>> f.get_property('scale_factor')
    10.0
    >>> f.del_property('scale_factor')
    10
    >>> f.has_property('scale_factor')
    False
        
        '''
        return self.get_property('scale_factor', default=AttributeError())
    @scale_factor.setter
    def scale_factor(self, value): self.set_property('scale_factor', value)
    @scale_factor.deleter
    def scale_factor(self):        self.del_property('scale_factor', default=AttributeError())

    
    @property
    def units(self):
        '''The units CF property.

    The units of the data. The value of the `units` property is a
    string that can be recognized by UNIDATA's Udunits package
    (http://www.unidata.ucar.edu/software/udunits). See
    http://cfconventions.org/latest.html for details.
    
    **Examples:**
    
    >>> f.units = 'K'
    >>> f.units
    'K'
    >>> del f.units
    
    >>> f.set_property('units', 'm.s-1')
    >>> f.get_property('units')
    'm.s-1'
    >>> f.has_property('units')
    True

        '''
        value = getattr(self.Units, 'units', None)
        if value is None:
            raise AttributeError("{} doesn't have CF property 'units'".format(
                self.__class__.__name__))

        return value
    
    @units.setter
    def units(self, value):
        self.Units = Units(value, getattr(self, 'calendar', None))
    
    @units.deleter
    def units(self):
        if getattr(self, 'units', None) is None:
            raise AttributeError(
                "Can't delete non-existent {} CF property 'units'".format(
                    self.__class__.__name__))
        
        self.Units = Units(None, getattr(self, 'calendar', None))
    

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def mask_invalid(self, inplace=False, i=False):
        '''Mask the array where invalid values occur.

    Note that:
    
    * Invalid values are Nan or inf
    
    * Invalid values in the results of arithmetic operations only
      occur if the raising of `FloatingPointError` exceptions has been
      suppressed by `cf.data.seterr`.
    
    * If the raising of `FloatingPointError` exceptions has been
      allowed then invalid values in the results of arithmetic
      operations it is possible for them to be automatically converted
      to masked values, depending on the setting of
      `cf.data.mask_fpe`. In this case, such automatic conversion
      might be faster than calling `mask_invalid`.
    
    .. seealso:: `cf.data.mask_fpe`, `cf.data.seterr`
    
    :Parameters:
    
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
        TODO
    
    **Examples:**
    
    >>> print(f.array)
    [ 0.  1.]
    >>> print(g.array)
    [ 1.  2.]
    
    >>> old = cf.data.seterr('ignore')
    >>> h = g/f
    >>> print(h.array)
    [ inf   2.]
    >>> h.mask_invalid(inplace=True)
    >>> print(h.array)
    [--  2.]
    
    >>> h = g**12345
    >>> print(h.array)
    [ 1.  inf]
    >>> h.mask_invalid(inplace=True)
    >>> print(h.array)
    [1.  --]
    
    >>> old = cf.data.seterr('raise')
    >>> old = cf.data.mask_fpe(True)
    >>> print((g/f).array)
    [ --  2]
    >>> print((g**12345).array)
    [1.  -- ]

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'mask_invalid', i=True) # pragma: no cover
            
        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.mask_invalid(inplace=True)

        if inplace:
            v = None
        return v
    
    
    def max(self):
        '''The maximum of the data array.
g
    .. seealso:: `mean`, `mid_range`, `min`, `range`, `sample_size`,
                 `sd`, `sum`, `var`
    
    :Returns: 
    
        `Data`    
            The maximum of the data array.
    
    **Examples:**
    
    >>> f.data
    <CF Data(12, 64, 128): [[[236.512756, ..., 256.93371]]] K>
    >>> f.max()
    <CF Data(): 311.343780 K>

        '''
        data = self.get_data(None)
        if data is not None:
            return data.max(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the maximum when there is no data array")       
    

    def mean(self):
        '''The unweighted mean the data array.

    .. seealso:: `max`, `mid_range`, `min`, `range`, `sample_size`, `sd`,
                 `sum`, `var`
    
    :Returns: 
    
        `Data`
            The unweighted mean the data array.
    
    **Examples:**
    
    >>> f.data
    <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
    >>> f.mean()
    <CF Data(): 280.192227593 K>

        '''
        data = self.get_data(None)
        if data is not None:
            return data.mean(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the mean when there is no data array")       
    

    def mid_range(self):
        '''The unweighted average of the maximum and minimum of the data
    array.
    
    .. seealso:: `max`, `mean`, `min`, `range`, `sample_size`, `sd`,
                 `sum`, `var`
    
    :Returns: 
    
        `Data`
            The unweighted average of the maximum and minimum of the
            data array.
    
    **Examples:**
    
    >>> f.data
    <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
    >>> f.mid_range()
    <CF Data(): 255.08618927 K>

        '''
        data = self.get_data(None)
        if data is not None:
            return data.mid_range(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the mid-range when there is no data array")       
    

    def min(self):
        '''The minimum of the data array.

    .. seealso:: `max`, `mean`, `mid_range`, `range`, `sample_size`,
                 `sd`, `sum`, `var`
    
    :Returns: 
    
        `Data`
            The minimum of the data array.
    
    **Examples:**
    
    >>> f.data
    <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
    >>> f.min()
    <CF Data(): 198.828598022 K>

        '''
        data = self.get_data(None)
        if data is not None:
            return data.min(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the minimum when there is no data array")       
    

    def range(self):
        '''The absolute difference between the maximum and minimum of the data
    array.
    
    .. seealso:: `max`, `mean`, `mid_range`, `min`, `sample_size`,
                 `sd`, `sum`, `var`
    
    :Returns: 
    
        `Data`
            The absolute difference between the maximum and minimum of
            the data array.
    
    **Examples:**
    
    >>> f.data
    <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
    >>> f.range()
    <CF Data(): 112.515182495 K>

        '''
        data = self.get_data(None)
        if data is not None:
            return data.range(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the range when there is no data array")       


    def sample_size(self):
        '''The number of non-missing data elements in the data array.

    .. seealso:: `count`, `max`, `mean`, `mid_range`, `min`, `range`,
                 `sd`, `sum`, `var`
    
    :Returns: 
    
        `Data`
            The number of non-missing data elements in the data array.
    
    **Examples:**
    
    >>> f.data
    <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
    >>> f.sample_size()
    <CF Data(): 98304.0>

        '''
        data = self.get_data(None)
        if data is not None:
            return data.sample_size(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the sample size when there is no data array")
    

    def sd(self):
        '''The unweighted sample standard deviation of the data array.

    .. seealso:: `max`, `mean`, `mid_range`, `min`, `range`,
                 `sample_size`, `sum`, `var`
    
    :Returns: 
    
        `Data`
            The unweighted standard deviation of the data array.
    
    **Examples:**
    
    >>> f.data
    <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
    >>> f.sd()
    <CF Data(): 22.685052535 K>

        '''
        data = self.get_data(None)
        if data is not None:
            return data.sd(squeeze=True, ddof=0)
          
        raise ValueError(
            "ERROR: Can't get the standard deviation when there is no data array")
    

    def sum(self):
       	'''The sum of the data array.

    .. seealso:: `max`, `mean`, `mid_range`, `min`, `range`,
                 `sample_size`, `sd`, `var`
    
    :Returns: 
    
        `Data`
            The sum of the data array.
    
    **Examples:**
    
    >>> f.data
    <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
    >>> f.sum()
    <CF Data(): 27544016.7413 K>

        '''
        data = self.get_data(None)
        if data is not None:
            return data.sum(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the sum when there is no data array")       
    

    def swapaxes(self, axis0, axis1, inplace=False):
        '''Interchange two axes of an array.

    .. seealso:: `flatten`, `flip`, `insert_dimension`, `squeeze`,
                 `transpose`
    
    :Parameters:
    
        axis0, axis1: `int`, `int`
            Select the axes to swap. Each axis is identified by its
            original integer position.
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
    :Returns:
    
            The construct with data with swapped axis positions. If
            the operation was in-place then `None` is returned.

    **Examples:**
    
    >>> f.shape
    (1, 2, 3)
    >>> f.swapaxes(1, 0).shape
    (2, 1, 3)
    >>> f.swapaxes(0, -1).shape
    (3, 2, 1)
    >>> f.swapaxes(1, 1).shape
    (1, 2, 3)
    >>> f.swapaxes(-1, -1).shape
    (1, 2, 3)

        '''
        if inplace:
            v = self
        else:
            v = self.copy()
            
        data = v.get_data(None)
        if data is not None:
            data.swapaxes(axis0, axis1, inplace=True)

        if inplace:            
            v = None
        return v

    
    def var(self):
        '''The unweighted sample variance of the data array.
        
    .. seealso:: `max`, `mean`, `mid_range`, `min`, `range`,
                 `sample_size`, `sd`, `sum`
    
    :Returns: 
    
        `Data`
            The unweighted variance of the data array.
    
    **Examples:**
    
    >>> f.data
    <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
    >>> f.var()
    <CF Data(): 514.611608515 K2>

        '''
        data = self.get_data(None)
        if data is None:
            raise ValueError(
                "ERROR: Can't get the variance when there is no data array")
                
        return data.var(squeeze=True, ddof=0)          
    

    @property
    def subspace(self):
        '''Return a new variable whose data is subspaced.

    This attribute may be indexed to select a subspace from dimension
    index values.
    
    **Subspacing by indexing**
    
    Subspacing by dimension indices uses an extended Python slicing
    syntax, which is similar numpy array indexing. There are two
    extensions to the numpy indexing functionality:
    
TODO


    * Size 1 dimensions are never removed.
    
      An integer index i takes the i-th element but does not reduce
      the rank of the output array by one.
    
    * When advanced indexing is used on more than one dimension, the
      advanced indices work independently.
    
      When more than one dimension's slice is a 1-d boolean array or
      1-d sequence of integers, then these indices work independently
      along each dimension (similar to the way vector subscripts work
      in Fortran), rather than by their elements.
    
    **Examples:**

        '''
        return Subspace(self)
    

    @property
    def shape(self):
        '''A tuple of the data array's dimension sizes.

    .. seealso:: `data`, `hasdata`, `ndim`, `size`
    
    **Examples:**
    
    >>> f.shape
    (73, 96)
    >>> f.ndim
    2
    
    >>> f.ndim
    0
    >>> f.shape
    ()
    
    >>> f.hasdata
    True
    >>> len(f.shape) == f.dnim
    True
    >>> reduce(lambda x, y: x*y, f.shape, 1) == f.size
    True

        '''
        return self.data.shape
    

    @property
    def ndim(self):
        '''The number of dimensions in the data array.

    .. seealso:: `data`, `hasdata`, `isscalar`, `shape`
    
    **Examples:**
    
    >>> f.hasdata
    True
    >>> f.shape
    (73, 96)
    >>> f.ndim
    2
    
    >>> f.shape
    ()
    >>> f.ndim
    0

        '''
        return self.data.ndim
    

    @property
    def size(self):
        '''The number of elements in the data array.

    .. seealso:: `data`, `hasdata`, `ndim`, `shape`
    
    **Examples:**
    
    >>> f.shape
    (73, 96)
    >>> f.size
    7008
    
    >>> f.shape
    ()
    >>> f.ndim
    0
    >>> f.size
    1
    
    >>> f.shape
    (1, 1, 1)
    >>> f.ndim
    3
    >>> f.size
    1
    
    >>> f.hasdata
    True
    >>> f.size == reduce(lambda x, y: x*y, f.shape, 1)
    True

        '''
        return self.data.size
    

    @property
    def datetime_array(self):
        '''An independent numpy array of date-time objects.

    Only applicable for reference time units.
    
    If the calendar has not been set then the CF default calendar will
    be used and the units will be updated accordingly.
    
    The data type of the data array is unchanged.
    
    .. seealso:: `array`, `asdatetime`, `asreftime`, `datetime_array`,
                 `dtvarray`, `varray`
    
    **Examples:**

        '''
        data = self.get_data(None)
        if data is None:
            raise AttributeError(
                "{} has no data array".format(self.__class__.__name__))
        
        return data.datetime_array
    

    @property
    def dtype(self):
        '''The `numpy` data type of the data array.

    By default this is the data type with the smallest size and
    smallest scalar kind to which all sub-arrays of the master data
    array may be safely cast without loss of information. For example,
    if the sub-arrays have data types 'int64' and 'float32' then the
    master data array's data type will be 'float64'; or if the
    sub-arrays have data types 'int64' and 'int32' then the master
    data array's data type will be 'int64'.
    
    Setting the data type to a `numpy.dtype` object, or any object
    convertible to a `numpy.dtype` object, will cause the master data
    array elements to be recast to the specified type at the time that
    they are next accessed, and not before. This does not immediately
    change the master data array elements, so, for example,
    reinstating the original data type prior to data access results in
    no loss of information.
    
    Deleting the data type forces the default behaviour. Note that if
    the data type of any sub-arrays has changed after `dtype` has been
    set (which could occur if the data array is accessed) then the
    reinstated default data type may be different to the data type
    prior to `dtype` being set.
    
    **Examples:**
    
    >>> f.dtype
    dtype('float64')
    >>> type(f.dtype)
    <type 'numpy.dtype'>
    
    >>> print(f.array)
    [0.5 1.5 2.5]
    >>> import numpy
    >>> f.dtype = numpy.dtype(int)
    >>> print(f.array)
    [0 1 2]
    >>> f.dtype = bool
    >>> print(f.array)
    [False  True  True]
    >>> f.dtype = 'float64'
    >>> print(f.array)
    [ 0.  1.  1.]
    
    >>> print(f.array)
    [0.5 1.5 2.5]
    >>> f.dtype = int
    >>> f.dtype = bool
    >>> f.dtype = float
    >>> print(f.array)
    [ 0.5  1.5  2.5]

        '''
        data = self.get_data(None)
        if data is None:
            raise AttributeError("{} doesn't have attribute 'dtype'".format(
                self.__class__.__name__))
        
        return data.dtype
    
    @dtype.setter
    def dtype(self, value):
# DCH - allow dtype to be set before data c.f.  Units
        data = self.get_data(None)
        if data is not None:
            data.dtype = value
    
    @dtype.deleter
    def dtype(self):
        data = self.get_data(None)
        if data is not None:
            del data.dtype
    

    @property
    def hardmask(self):
        '''Whether the mask is hard (True) or soft (False).

    When the mask is hard, masked elements of the data array can not
    be unmasked by assignment, but unmasked elements may be still be
    masked.
    
    When the mask is soft, masked entries of the data array may be
    unmasked by assignment and unmasked entries may be masked.
    
    By default, the mask is hard.
    
    .. seealso:: `where`, `subspace`, `__setitem__`
    
    **Examples:**
    
    >>> f.hardmask = False
    >>> f.hardmask
    False

        '''
        data = self.get_data(None)
        if data is None:
            raise AttributeError(
                "{} doesn't have attribute 'hardmask'".format(self.__class__.__name__))
        
        return data.hardmask
    
    
    @hardmask.setter
    def hardmask(self, value):
        data = self.get_data(None)
        if data is None:
            raise AttributeError(
                "{} doesn't have attribute 'hardmask'".format(self.__class__.__name__))
                    
        data.hardmask = value
    
    @hardmask.deleter
    def hardmask(self):
        raise AttributeError(
            "Won't delete {} attribute 'hardmask'".format(self.__class__.__name__))
    

    @property
    def array(self):
        '''A numpy array deep copy of the data array.

    Changing the returned numpy array does not change the data array.
    
    .. seealso:: `data`, `datetime_array`, `varray`
    
    **Examples:**
    
    >>> f.data
    <CF Data(5): [0, ... 4] kg m-1 s-2>
    >>> a = f.array
    >>> type(a)
    <type 'numpy.ndarray'>
    >>> print(a)
    [0 1 2 3 4]
    >>> a[0] = 999
    >>> print(a)
    [999 1 2 3 4]
    >>> print(f.array)
    [0 1 2 3 4]
    >>> f.data
    <CF Data(5): [0, ... 4] kg m-1 s-2>

        '''
        data = self.get_data(None)
        if data is None:
            raise AttributeError("{} has no data array".format(self.__class__.__name__))

        return data.array
    

    @property
    def varray(self):
        '''A numpy array view of the data array.

    Changing the elements of the returned view changes the data array.
    
    .. seealso:: `array`, `data`, `datetime_array`
    
    **Examples:**
    
    >>> f.data
    <CF Data(5): [0, ... 4] kg m-1 s-2>
    >>> a = f.array
    >>> type(a)
    <type 'numpy.ndarray'>
    >>> print(a)
    [0 1 2 3 4]
    >>> a[0] = 999
    >>> print(a)
    [999 1 2 3 4]
    >>> print(f.array)
    [999 1 2 3 4]
    >>> f.data
    <CF Data(5): [999, ... 4] kg m-1 s-2>

        '''
        data = self.get_data(None)
        if data is None:
            raise AttributeError("{} has no data array".format(self.__class__.__name__))

        return data.varray
    

    @property
    def isauxiliary(self): 
        '''True if the variable is an auxiliary coordinate object.

    .. seealso:: `isdimension`, `isdomainancillary`,
                 `isfieldancillary`, `ismeasure`
    
    **Examples:** 
    
    >>> f.isauxiliary
    False

        '''
        return False
    

    @property
    def isdimension(self): 
        '''True if the variable is a dimension coordinate object.

    .. seealso:: `isauxiliary`, `isdomainancillary`,
                 `isfieldancillary`, `ismeasure`
    
    **Examples:** 
    
    >>> f.isdimension
    False

        '''
        return False
    

    @property
    def isdomainancillary(self): 
        '''True if the variable is a domain ancillary object.

    .. seealso:: `isauxiliary`, `isdimension`, `isfieldancillary`,
                 `ismeasure`
    
    **Examples:** 
    
    >>> f.isdomainancillary
    False

        '''
        return False
    

    @property
    def isfieldancillary(self): 
        '''True if the variable is a field ancillary object.

    .. seealso:: `isauxiliary`, `isdimension`, `isdomainancillary`,
                 `ismeasure`
    
    **Examples:** 
    
    >>> f.isfieldancillary
    False

        '''
        return False
    

    @property
    def ismeasure(self): 
        '''True if the variable is a cell measure object.

    .. seealso:: `isauxiliary`, `isdimension`, `isdomainancillary`,
                 `isfieldancillary`
    
    **Examples:** 
    
    >>> f.ismeasure
    False

        '''
        return False
    

    @property
    def isscalar(self):
        '''True if the data array is scalar.

    .. seealso:: `has_data`, `ndim`
    
    **Examples:**
    
    >>> f.ndim
    0
    >>> f.isscalar
    True
    
    >>> f.ndim >= 1
    True
    >>> f.isscalar
    False
    
    >>> f.hasdata
    False
    >>> f.isscalar
    False

        '''
        data = self.get_data(None)
        if data is None:
            return False

        return data.isscalar
    

    def ceil(self, inplace=False, i=False):
        '''The ceiling of the data, element-wise.

    The ceiling of ``x`` is the smallest integer ``n``, such that
    ``n>=x``.
    
    .. versionadded:: 1.0
    
    .. seealso:: `floor`, `rint`, `trunc`
    
    :Parameters:
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
            The construct with ceilinged of data. If the operation was
            in-place then `None` is returned.
    
    **Examples:**
    
    >>> print(f.array)
    [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
    >>> print(f.ceil().array)
    [-1. -1. -1. -1.  0.  1.  2.  2.  2.]
    >>> f.ceil(inplace=True)
    >>> print(f.array)
    [-1. -1. -1. -1.  0.  1.  2.  2.  2.]

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'ceil', i=True) # pragma: no cover

        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.ceil(inplace=True)

        if inplace:
            v = None
        return v
    

    def chunk(self, chunksize=None):
        '''Partition the data array.

    :Parameters:
    
        chunksize: `int`
    
    :Returns:
     
        `None`

        '''
        data = self.get_data(None)
        if data is not None:
            data.chunk(chunksize)
    

    def clip(self, a_min, a_max, units=None, inplace=False, i=False):
        '''Limit the values in the data.

    Given an interval, values outside the interval are clipped to the
    interval edges. For example, if an interval of ``[0, 1]`` is
    specified, values smaller than 0 become 0, and values larger than
    1 become 1.
    
    :Parameters:
     
        a_min:
            Minimum value. If `None`, clipping is not performed on
            lower interval edge. Not more than one of `a_min` and
            `a_max` may be `None`.
    
        a_max:
            Maximum value. If `None`, clipping is not performed on
            upper interval edge. Not more than one of `a_min` and
            `a_max` may be `None`.
    
        units: `str` or `Units`
            Specify the units of *a_min* and *a_max*. By default the
            same units as the data are assumed.
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns: 
    
            The construct with clipped data. If the operation was
            in-place then `None` is returned.
    
    **Examples:**
    
    >>> g = f.clip(-90, 90)
    >>> g = f.clip(-90, 90, 'degrees_north')

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'clip', i=True) # pragma: no cover

        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.clip(a_min, a_max, units=units, inplace=True)

        if inplace:
            v = None
        return v
    

    def close(self):
        '''Close all files referenced by the construct.

    Note that a closed file will be automatically reopened if its
    contents are subsequently required.
    
    .. seealso:: `files`
    
    :Returns:
    
        `None`
    
    **Examples:**
    
    >>> f.close()

        '''
        data = self.get_data(None)
        if data is not None:
            data.close()
    

    @classmethod
    def concatenate(cls, variables, axis=0, _preserve=True):
        '''Join a sequence of variables together.

    :Parameters:
    
        variables: sequence of constructs.
    
        axis: `int`, optional
    
    :Returns:
    
    TODO

        '''
        variable0 = variables[0]

        if len(variables) == 1:
            return variable0.copy()

        out = variable0.copy() #data=False)
        
        data = Data.concatenate([v.get_data() for v in variables],
                                axis=axis,
                                _preserve=_preserve)
        out.set_data(data, copy=False)
        
        return out
    

    def cos(self, bounds=True, inplace=False, i=False):
        '''Take the trigonometric cosine of the data, element-wise.

    Units are accounted for in the calculation, so that the the cosine
    of 90 degrees_east is 0.0, as is the cosine of 1.57079632
    radians. If the units are not equivalent to radians (such as
    Kelvin) then they are treated as if they were radians.
    
    The output units are '1' (nondimensionsal).

    The "standard_name" and "long_name" properties are removed from
    the result.
    
    .. seealso:: `sin`, `tan`
    
    :Parameters:
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
            The construct with the cosine of data values. If the
            operation was in-place then `None` is returned.
    
    **Examples:**
    
    >>> f.Units
    <Units: degrees_east>
    >>> print(f.array)
    [[-90 0 90 --]]
    >>> f.cos()
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[0.0 1.0 0.0 --]]
    
    >>> f.Units
    <Units: m s-1>
    >>> print(f.array)
    [[1 2 3 --]]
    >>> f.cos()
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[0.540302305868 -0.416146836547 -0.9899924966 --]]

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'cos', i=True) # pragma: no cover

        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.cos(inplace=True)

        # Remove misleading identities
        v.del_property('standard_name', None)
        v.del_property('long_name', None)
        
        if inplace:
            v = None
        return v
    

    def count(self):
        '''Count the non-masked elements of the data.

    :Returns:
    
        `int`
            The number of non-masked elements.
    
    **Examples:**
    
    >>> n = f.count()

        '''
        data = self.get_data(None)
        if data is None:
            raise AttributeError(
                "Can't count when there are data")
            
        return data.count()
    

    def count_masked(self):
        '''Count the masked elements of the data.

    :Returns:
    
        `int`
            The number of masked elements.
    
    **Examples:**
    
    >>> n = f.count_masked()
        
        '''
        data = self.get_data(None)
        if data is None:
            raise AttributeError(
                "Can't count masked when there are data")
            
        return data.count_masked()
    

    def cyclic(self, axes=None, iscyclic=True):
        '''Set the cyclicity of an axis.

    .. seealso:: `iscyclic`
    
    :Parameters:
    
        axes: (sequence of) `int`
            The axes to be set. Each axis is identified by its integer
            position. By default no axes are set.
            
        iscyclic: `bool`, optional
            If False then the axis is set to be non-cyclic. By default
            the axis is set to be cyclic.
    
    :Returns:
    
        `!set`
    
    **Examples:**
    
    >>> f.cyclic()
    set()
    >>> f.cyclic(1)
    set()
    >>> f.cyclic()
    {1} TODO

        '''
        data = self.get_data(None)
        if data is None:
            return set()
       
        return data.cyclic(axes, iscyclic)
    
            
    def datum(self, *index):
        '''Return an element of the data array as a standard Python scalar.

    The first and last elements are always returned with
    ``f.datum(0)`` and ``f.datum(-1)`` respectively, even if the data
    array is a scalar array or has two or more dimensions.
    
    :Parameters:
    
        index: optional
            Specify which element to return. When no positional
            arguments are provided, the method only works for data
            arrays with one element (but any number of dimensions),
            and the single element is returned. If positional
            arguments are given then they must be one of the
            following:
    
              * An integer. This argument is interpreted as a flat
                index into the array, specifying which element to copy
                and return.
             
                *Parameter example:*
                  If the data aray shape is ``(2, 3, 6)`` then:
                    * ``f.datum(0)``  is equivalent to ``f.datum(0, 0, 0)``.
                    * ``f.datum(-1)`` is equivalent to ``f.datum(1, 2, 5)``.
                    * ``f.datum(16)`` is equivalent to ``f.datum(0, 2, 4)``.
    
                If *index* is ``0`` or ``-1`` then the first or last
                data array element respecitively will be returned,
                even if the data array is a scalar array or has two or
                more dimensions.  ..
             
              * Two or more integers. These arguments are interpreted
                as a multidimensionsal index to the array. There must
                be the same number of integers as data array
                dimensions.  ..
             
              * A tuple of integers. This argument is interpreted as a
                multidimensionsal index to the array. There must be
                the same number of integers as data array dimensions.
             
                *Example:*    
                  ``f.datum((0, 2, 4))`` is equivalent to ``f.datum(0,
                  2, 4)``; and ``f.datum(())`` is equivalent to
                  ``f.datum()``.
    
    :Returns:
    
            A copy of the specified element of the array as a suitable
            Python scalar.
    
    **Examples:**
    
    >>> print(f.array)
    2
    >>> f.datum()
    2
    >>> 2 == f.datum(0) == f.datum(-1) == f.datum(())
    True
    
    >>> print(f.array)
    [[2]]
    >>> 2 == f.datum() == f.datum(0) == f.datum(-1)
    True
    >>> 2 == f.datum(0, 0) == f.datum((-1, -1)) == f.datum(-1, 0)
    True
    
    >>> print(f.array)
    [[4 -- 6]
     [1 2 3]]
    >>> f.datum(0)
    4     
    >>> f.datum(-1)
    3     
    >>> f.datum(1)
    masked
    >>> f.datum(4)
    2     
    >>> f.datum(-2)
    2     
    >>> f.datum(0, 0)
    4     
    >>> f.datum(-2, -1)
    6     
    >>> f.datum(1, 2)
    3     
    >>> f.datum((0, 2))
    6

        '''
        data = self.get_data(None)
        if data is None:
            raise ValueError(
                "ERROR: Can't return an element when there is no data array")
        
        return data.datum(*index)
    

    def equals(self, other, rtol=None, atol=None, verbose=False,
               ignore_data_type=False, ignore_fill_value=False,
               ignore_properties=(), ignore_compression=False,
               ignore_type=False):
        '''Whether two instances are the same.

    Equality is strict by default. This means that:
    
    * the same descriptive properties must be present, with the same
      values and data types, and vector-valued properties must also
      have same the size and be element-wise equal (see the
      *ignore_properties* and *ignore_data_type* parameters), and
    
    ..
    
    * if there are data arrays then they must have same shape and data
      type, the same missing data mask, and be element-wise equal (see
      the *ignore_data_type* parameter).
    
    Two real numbers ``x`` and ``y`` are considered equal if
    ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on absolute
    differences) and ``rtol`` (the tolerance on relative differences)
    are positive, typically very small numbers. See the *atol* and
    *rtol* parameters.
    
    If data arrays are compressed then the compression type and the
    underlying compressed arrays must be the same, as well as the
    arrays in their uncompressed forms. See the *ignore_compression*
    parameter.
    
    Any type of object may be tested but, in general, equality is only
    possible with another object of the same type, or a subclass of
    one. See the *ignore_type* parameter.
    
    NetCDF elements, such as netCDF variable and dimension names, do
    not constitute part of the CF data model and so are not checked.
    
    .. versionadded:: 1.7.0
    
    :Parameters:
    
        other: 
            The object to compare for equality.
    
        atol: float, optional
            The tolerance on absolute differences between real
            numbers. The default value is set by the `cf.ATOL`
            function.
            
        rtol: float, optional
            The tolerance on relative differences between real
            numbers. The default value is set by the `cf.RTOL`
            function.
    
        ignore_fill_value: `bool`, optional
            If True then the "_FillValue" and "missing_value"
            properties are omitted from the comparison.
    
        verbose: `bool`, optional
            If True then print information about differences that lead
            to inequality.
    
        ignore_properties: sequence of `str`, optional
            The names of properties to omit from the comparison.
    
        ignore_data_type: `bool`, optional
            If True then ignore the data types in all numerical
            comparisons. By default different numerical data types
            imply inequality, regardless of whether the elements are
            within the tolerance for equality.
    
        ignore_compression: `bool`, optional
            If True then any compression applied to the underlying
            arrays is ignored and only the uncompressed arrays are
            tested for equality. By default the compression type and,
            if appliciable, the underlying compressed arrays must be
            the same, as well as the arrays in their uncompressed
            forms.
    
        ignore_type: `bool`, optional
            Any type of object may be tested but, in general, equality
            is only possible with another object of the same type, or
            a subclass of one. If *ignore_type* is True then equality
            is possible for any object with a compatible API.
    
    :Returns: 
      
        `bool`
            Whether the two instances are equal.
    
    **Examples:**
    
    >>> f.equals(f)
    True
    >>> f.equals(f.copy())
    True
    >>> f.equals('a string')
    False
    >>> f.equals(f - 1)
    False

        '''
        # Check that each instance has the same Units
        try:
            if not self.Units.equals(other.Units):
                if verbose:
                    print("{0}: Different Units: {1!r} != {2!r}".format(
                        self.__class__.__name__, self.Units, other.Units))
                    return False
        except AttributeError:
            pass
        
        ignore_properties = tuple(ignore_properties) + self._special_properties

        return super().equals(other, rtol=rtol, atol=atol,
                              verbose=verbose, ignore_data_type=ignore_data_type,
                              ignore_fill_value=ignore_fill_value,
                              ignore_properties=ignore_properties,
                              ignore_type=ignore_type)

    
    def equivalent(self, other, rtol=None, atol=None, traceback=False):
        '''True if two constructs are equal, False otherwise.

    Two real numbers ``x`` and ``y`` are considered equal if
    ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on absolute
    differences) and ``rtol`` (the tolerance on relative differences)
    are positive, typically very small numbers. See the *atol* and
    *rtol* parameters.
    
    :Parameters:
    
        other: 
            The object to compare for equality.
    
        atol: `float`, optional
            The tolerance on absolute differences between real
            numbers. The default value is set by the `ATOL` function.
    
        rtol: `float`, optional
            The tolerance on relative differences between real
            numbers. The default value is set by the `RTOL` function.

        '''     
        if self is other:
            return True

        # Check that each instance is the same type
        if type(self) != type(other):
            print("{}: Different types: {}, {}".format(
                self.__class__.__name__,
                self.__class__.__name__,
                other.__class__.__name__))
            return False
       
        identity0 = self.identity()
        identity1 = other.identity()

        if identity0 is None or identity1 is None or identity0 != identity1:
            # add traceback
            return False
                  
        # ------------------------------------------------------------
        # Check the special attributes
        # ------------------------------------------------------------
        self_special  = self._private['special_attributes']
        other_special = other._private['special_attributes']
        if set(self_special) != set(other_special):
            if traceback:
                print("%s: Different attributes: %s" %
                      (self.__class__.__name__,
                       set(self_special).symmetric_difference(other_special)))
            return False

        for attr, x in self_special.iteritems():
            y = other_special[attr]

            result = cf_equivalent(x, y, rtol=rtol, atol=atol,
                                   traceback=traceback)
               
            if not result:
                if traceback:
                    print("{}: Different {} attributes: {!r}, {!r}".format(
                        self.__class__.__name__, attr, x, y))
                return False
        #--- End: for

        # ------------------------------------------------------------
        # Check the data
        # ------------------------------------------------------------
        if not self._equivalent_data(other, rtol=rtol, atol=atol,
                                     traceback=traceback):
            # add traceback
            return False
            
        return True
    

    def convert_reference_time(self, units=None,
                               calendar_months=False,
                               calendar_years=False, inplace=False,
                               i=False):
        '''Convert reference time data values to have new units.

    Conversion is done by decoding the reference times to date-time
    objects and then re-encoding them for the new units.
    
    Any conversions are possible, but this method is primarily for
    conversions which require a change in the date-times originally
    encoded. For example, use this method to reinterpret data values
    in units of "months" since a reference time to data values in
    "calendar months" since a reference time. This is often necessary
    when units of "calendar months" were intended but encoded as
    "months", which have special definition. See the note and examples
    below for more details.
    
    For conversions which do not require a change in the date-times
    implied by the data values, this method will be considerably
    slower than a simple reassignment of the units. For example, if
    the original units are ``'days since 2000-12-1'`` then ``c.Units =
    cf.Units('days since 1901-1-1')`` will give the same result and be
    considerably faster than ``c.convert_reference_time(cf.Units('days
    since 1901-1-1'))``.
    
    .. note:: It is recommended that the units "year" and "month" be
              used with caution, as explained in the following excerpt
              from the CF conventions: "The Udunits package defines a
              year to be exactly 365.242198781 days (the interval
              between 2 successive passages of the sun through vernal
              equinox). It is not a calendar year. Udunits includes
              the following definitions for years: a common_year is
              365 days, a leap_year is 366 days, a Julian_year is
              365.25 days, and a Gregorian_year is 365.2425 days. For
              similar reasons the unit ``month``, which is defined to
              be exactly year/12, should also be used with caution.
    
    :Parameters:
    
        units: `Units`, optional
            The reference time units to convert to. By default the
            units days since the original reference time in the the
            original calendar.
    
            *Parameter example:*
              If the original units are ``'months since 2000-1-1'`` in
              the Gregorian calendar then the default units to convert
              to are ``'days since 2000-1-1'`` in the Gregorian
              calendar.
    
        calendar_months: `bool`, optional 
            If True then treat units of ``'months'`` as if they were
            calendar months (in whichever calendar is originally
            specified), rather than a 12th of the interval between 2
            successive passages of the sun through vernal equinox
            (i.e. 365.242198781/12 days).
    
        calendar_years: `bool`, optional
            If True then treat units of ``'years'`` as if they were
            calendar years (in whichever calendar is originally
            specified), rather than the interval between 2 successive
            passages of the sun through vernal equinox
            (i.e. 365.242198781 days).
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns: 
     
            The construct with converted reference time data values.
    
    **Examples:**
    
    >>> print(f.array)
    [1  2  3  4]
    >>> f.Units
    <Units: months since 2000-1-1>
    >>> print(f.datetime_array)
    [datetime.datetime(2000, 1, 31, 10, 29, 3, 831197) TODO
     datetime.datetime(2000, 3, 1, 20, 58, 7, 662441)
     datetime.datetime(2000, 4, 1, 7, 27, 11, 493645)
     datetime.datetime(2000, 5, 1, 17, 56, 15, 324889)]
    >>> f.convert_reference_time(calendar_months=True, inplace=True)
    >>> print(f.datetime_array)
    [datetime.datetime(2000, 2, 1, 0, 0) TODOx
     datetime.datetime(2000, 3, 1, 0, 0)
     datetime.datetime(2000, 4, 1, 0, 0)
     datetime.datetime(2000, 5, 1, 0, 0)]
    >>> print(f.array)
    [  31.   60.   91.  121.]
    >>> f.Units
    <Units: days since 2000-1-1>

        '''
        def _convert_reftime_units(value, units, reftime): #, calendar):
            '''sads

    :Parameters:

        value: number

        units: `Units`

    :Returns:

        `datetime.datetime` or `cf.Datetime`

            '''
            t = TimeDuration(value, units=units)
            if value > 0:
                return t.interval(reftime, end=False)[1]
            else:
                return t.interval(reftime, end=True)[0]
        #--- End: def

        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'convert_reference_time', i=True) # pragma: no cover
            
        if not self.Units.isreftime:
            raise ValueError(
                "{} must have reference time units, not {!r}".format(
                    self.__class__.__name__, self.Units))

        if inplace:
            v = self
        else:
            v = self.copy()

        units0 = self.Units
        
        if units is None:
            # By default, set the target units to "days since
            # <reference time of self.Units>,
            # calendar=<self.calendar>"
            units = Units('days since '+units0.units.split(' since ')[1],
                          calendar=units0._calendar)
        elif not getattr(units, 'isreftime', False):
            raise ValueError(
                "New units must be reference time units, not {0!r}".format(units))
           
        if units0._units_since_reftime in _month_units:
            if calendar_months:
                units0 = Units('calendar_'+units0.units, calendar=units0._calendar)
            else:
                units0 = Units('days since '+units0.units.split(' since ')[1],
                                calendar=units0._calendar)
                v.Units = units0
        elif units0._units_since_reftime in _year_units:
            if calendar_years:
                units0 = Units('calendar_'+units0.units, calendar=units0._calendar)
            else:
                units0 = Units('days since '+units0.units.split(' since ')[1],
                                calendar=units0._calendar)
                v.Units = units0
        

        # Not LAMAed!
        v.set_data(Data(            
            numpy_vectorize(
                functools_partial(_convert_reftime_units,
                                  units=units0._units_since_reftime,
                                  reftime=dt(units0.reftime, calendar=units0._calendar),
                                  ),
                otypes=[object])(v),
            units=units))

        if inplace:
            v = None

        return v    


    def flatten(self, axes=None, inplace=False):
        '''Flatten axes of the data

    Any subset of the axes may be flattened.

    The shape of the data may change, but the size will not.

    The flattening is executed in row-major (C-style) order. For
    example, the array ``[[1, 2], [3, 4]]`` would be flattened across
    both dimensions to ``[1 2 3 4]``.

    .. versionaddedd:: 3.0.2

    .. seealso:: `insert_dimension`, `flip`, `swapaxes`, `transpose`

    :Parameters:
   
        axes: (sequence of) int or str, optional
            Select the axes.  By default all axes are flattened. The
            *axes* argument may be one, or a sequence, of:
    
              * An internal axis identifier. Selects this axis.
            ..
    
              * An integer. Selects the axis coresponding to the given
                position in the list of axes of the data array.
    
            No axes are flattened if *axes* is an empty sequence.
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
    :Returns:

            The construct with flattened data, or `None` if the
            operation was in-place.

    **Examples**

    >>> f.shape
    (1, 2, 3, 4)
    >>> f.flatten().shape
    (24,)
    >>> f.flatten([1, 3]).shape
    (1, 8, 3)
    >>> f.flatten([0, -1], inplace=True)
    >>> f.shape
    (4, 2, 3)

        '''
        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.flatten(axes, inplace=True)

        if inplace:
            v = None
        return v

        
    def floor(self, inplace=False, i=False):
        '''Floor the data array, element-wise.

    The floor of ``x`` is the largest integer ``n``, such that
    ``n<=x``.
    
    .. versionadded:: 1.0
    
    .. seealso:: `ceil`, `rint`, `trunc`
    
    :Parameters:
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
            
            The construct with floored data. If the operation was
            in-place then `None` is returned.
    
    **Examples:**
    
    >>> print(f.array)
    [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
    >>> print(f.floor().array)
    [-2. -2. -2. -1.  0.  1.  1.  1.  1.]
    >>> f.floor(inplace=True)
    >>> print(f.array)
    [-2. -2. -2. -1.  0.  1.  1.  1.  1.]

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'floor', i=True) # pragma: no cover
            
        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.floor(inplace=True)

        if inplace:
            v = None
        return v
    

    def match_by_naxes(self, *naxes):
        '''Whether or not the data has a given dimensionality.

    .. versionadded:: 3.0.0
    
    .. seealso:: `match`, `match_by_identity`, `match_by_property`,
                 `match_by_units`
    
    :Parameters:
    
        naxes: optional
            Dimensionalities to be compared.
    
            A dimensionality given by an `int` or a `Query` object.
    
            If no numbers are provided then there is always a match.
         
    :Returns:
    
        `bool`
            Whether or not there is a match.
    
    **Examples:**
    
    >>> f.ndim
    3
    >>> f.match_by_naxes(3)
    True
    >>> f.match_by_naxes(cf.ge(1))
    True
    >>> f.match_by_naxes(1, 2, 3)
    True
    >>> f.match_by_naxes(2)
    False
    >>> f.match_by_naxes(cf.gt(3))
    False

        '''
        if not naxes:
            return True

        data = self.get_data(None)
        if data is None:
            return False
        
        self_ndim = data.ndim        
        for ndim in naxes:
            ok = (ndim == self_ndim)
            if ok:
                return True
        #--- End: for

        return False
    

    def match_by_units(self, *units, exact=True):
        '''Whether or not the construct has given units.

    .. versionadded:: 3.0.0
    
    .. seealso:: `match`, `match_by_identity`, `match_by_property`,
                 `match_by_naxes`
    
    :Parameters:
        
        units: optional
            Units to be compared.
    
            Units are specified by a string or compiled regular
            expression (e.g. ``'km'``, ``'m s-1'``,
            ``re.compile('^kilo')``, etc.) or a `Units` object
            (e.g. ``Units('km')``, ``Units('m s-1')``, etc.).
                    
            If no units are provided then there is always a match.
         
        exact: `bool`, optional
            If False then a match occurs if the construct's units
            are equivalent to any of those given by *units*. For
            example, metres and are equivelent to kilometres. By
            default, a match only occurs if the construct's units are
            exactly one of those given by *units*. Note that the
            format of the units is not important, i.e. 'm' is exactly
            the same as 'metres' for this purpose.
    
    :Returns:
    
        `bool`
            Whether or not there is a match.
    
    **Examples:**
    
    >>> f.units
    'metres'
    >>> f.match_by_units('metres')
    True
    >>> f.match_by_units('m')
    True
    >>> f.match_by_units(Units('m'))
    True
    >>> f.match_by_units('m', 'kilogram')
    True
    >>> f.match_by_units('km', exact=False)
    True
    >>> f.match_by_units(cf.Units('km'), exact=False)
    True
    >>> f.match_by_units(re.compile('^met'))
    True
    >>> f.match_by_units(cf.Units('km'))
    False
    >>> f.match_by_units(cf.Units('kg m-2'))
    False

        '''
        if not units:
            return True

        self_units = self.Units
        
        ok = False
        for value0 in units:
            try:
                # re.compile object                
                ok = value0.search(self_units.units)
            except (AttributeError, TypeError):
                if exact:
                    ok = Units(value0).equals(self_units)
                else:            
                    ok = Units(value0).equivalent(self_units)
            #--- End: if

            if ok:
                break
        #--- End: for

        return ok
    

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def all(self):
        '''Test whether all data elements evaluate to True.

    Performs a logical "and" over the data array and returns the
    result. Masked values are considered as True during computation.
    
    .. seealso:: `allclose`, `any`
    
    :Returns:
    
        `bool`
            Whether ot not all data elements evaluate to True.
    
    **Examples:**
    
    >>> print(f.array)
    [[0  3  0]]
    >>> f.all()
    False
    
    >>> print(f.array)
    [[1  3  --]]
    >>> f.all()
    True

        '''
        data = self.get_data(None)
        if data is not None:
            return data.all()

        return False
    

    def allclose(self, y, atol=None, rtol=None):
        '''Test whether all data are element-wise equal to other,
    broadcastable data.
    
    Two real numbers ``x`` and ``y`` are considered equal if
    ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on absolute
    differences) and ``rtol`` (the tolerance on relative differences)
    are positive, typically very small numbers. See the *atol* and
    *rtol* parameters.
    
    .. seealso:: `all`, `any`, `isclose`
    
    :Parameters:
    
        y:
            The object to be compared with the data array. *y* must be
            broadcastable to the data array and if *y* has units then
            they must be compatible. May be any object that can be
            converted to a `Data` object (which includes numpy array
            and `Data` objects).
    
        atol: `float`, optional
            The tolerance on absolute differences between real
            numbers. The default value is set by the `ATOL` function.
    
        rtol: `float`, optional
            The tolerance on relative differences between real
            numbers. The default value is set by the `RTOL` function.
    
    :Returns:
    
        `bool` 
            Returns `True` if the data are equal within the given
            tolerance; `False` otherwise.
    
    **Examples:**
            
    >>> x = f.allclose(g)

        '''
        data = self.get_data(None)
        if data is None:
            return False

        if isinstance(y, self.__class__):
            y_data = y.get_data(None)
            if y_data is None:
                return False

            y = self._conform_for_assignment(y)
            y_data = y.get_data()
        else:
            try:
                y_data = y.get_data(None)
            except AttributeError:
                y_data = y
            else:
                if y_data is None:
                    y_data = y
        #--- End: if
        
        return data.allclose(y_data, rtol=rtol, atol=atol)
    

    def any(self):
        '''Test whether any data elements evaluate to True.

    Performs a logical "or" over the data array and returns the
    result. Masked values are considered as False during computation.
    
    .. seealso:: `all`, `allclose`
    
    :Returns:
    
        `bool`
            Whether ot not any data elements evaluate to `True`.
    
    **Examples:**
    
    >>> print(f.array)
    [[0  0  0]]
    >>> f.any()
    False
    
    >>> print(f.array)
    [[--  0  0]]
    >>> f.any()
    False
    
    >>> print(f.array)
    [[--  3  0]]
    >>> f.any()
    True

        '''
        data = self.get_data(None)
        if data is not None:
            return data.any()

        return False


    def files(self):
        '''Return the names of any files containing parts of the data array.

    .. seealso:: `close`
    
    :Returns:
    
        `!set`
            The file names in normalized, absolute form.
    
    **Examples:**
    
    >>> f = cf.read_field('../file[123].nc')
    >>> f.files()
    {'/data/user/file1.nc',
     '/data/user/file2.nc',
     '/data/user/file3.nc'}
    >>> a = f.array
    >>> f.files()
    set()

        '''
        data = self.get_data(None)
        if data is None:
            out = set()
        else:
            out = data.files()

        return out
    

    def fill_value(self, default=None):
        '''Return the data array missing data value.

    This is the value of the `missing_value` CF property, or if that
    is not set, the value of the `_FillValue` CF property, else if
    that is not set, ``None``. In the last case the default `numpy`
    missing data value for the array's data type is assumed if a
    missing data value is required.
    
    .. seealso:: `cf.default_netCDF_fillvals`, `_FillValue`,
                 `missing_value`

    :Parameters:
    
        default: optional
            If the missing value is unset then return this value. By
            default, *default* is `None`. If *default* is the special
            value ``'netCDF'`` then return the netCDF default value
            appropriate to the data array's data type is used. These
            may be found with the `cf.default_netCDF_fillvals`
            function. For example:
    
            >>> cf.default_netCDF_fillvals()
            {'S1': '\x00',
             'i1': -127,
             'u1': 255,
             'i2': -32767,
             'u2': 65535,
             'i4': -2147483647,
             'u4': 4294967295,
             'i8': -9223372036854775806,
             'u8': 18446744073709551614,
             'f4': 9.969209968386869e+36,
             'f8': 9.969209968386869e+36}

    :Returns:
    
            The missing deata value or, if one has not been set, the
            value specified by *default*
    
    **Examples:**
    
    >>> f.fill_value()
    None
    >>> f._FillValue = -1e30
    >>> f.fill_value()
    -1e30
    >>> f.missing_value = 1073741824
    >>> f.fill_value()
    1073741824
    >>> del f.missing_value
    >>> f.fill_value()
    -1e30
    >>> del f._FillValue
    >>> f.fill_value()
    None
    >>> f.dtype
    dtype('float64')
    >>> f.fill_value(default='netCDF')
    9.969209968386869e+36
    >>> f._FillValue = -999
    >>> f.fill_value(default='netCDF')
    -999

        '''
        fillval = self.get_property('missing_value', None)
        if fillval is None:
            fillval = self.get_property('_FillValue', None)

        if fillval is None:
            if default == 'netCDF':
                d = self.dtype
                fillval = default_netCDF_fillvals()[d.kind + str(d.itemsize)]
            else:
                fillval = default 
        #--- End: if

        return fillval
    

    def flip(self, axes=None, inplace=False, i=False):
        '''Flip (reverse the direction of) data dimensions.

    .. seealso:: `flatten`, `insert_dimension`, `squeeze`,
                 `transpose`, `unsqueeze`
    
    :Parameters:
    
        axes: optional
           Select the domain axes to flip. One, or a sequence, of:
    
              * The position of the dimension in the data.
    
            If no axes are specified then all axes are flipped.
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use the *inplace* parameter instead.
    
    :Returns:
    
            The construct with flipped axes, or `None` if the
            operation was in-place.
    
    **Examples:**
    
    >>> f.flip()
    >>> f.flip(1)
    >>> f.flip([0, 1])
    
    >>> g = f[::-1, :, ::-1]
    >>> f.flip([2, 0]).equals(g)
    True

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'flip', i=True) # pragma: no cover
            
        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.flip(axes, inplace=True)

        if inplace:
            v = None            
        return v
    

    def exp(self, inplace=False, i=False):
        '''The exponential of the data, element-wise.

    The "standard_name" and "long_name" properties are removed from
    the result.

    .. seealso:: `log`
    
    :Parameters:
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
            The construct with the exponential of data values. If the
            operation was in-place then `None` is returned.
    
    **Examples:**
    
    >>> f.data
    <CF Data(1, 2): [[1, 2]]>
    >>> f.exp().data            
    <CF Data(1, 2): [[2.71828182846, 7.38905609893]]>
    
    >>> f.data
    <CF Data(1, 2): [[1, 2]] 2>
    >>> f.exp().data            
    <CF Data(1, 2): [[7.38905609893, 54.5981500331]]>
    
    >>> f.data
    <CF Data(1, 2): [[1, 2]] kg m-1 s-2>
    >>> f.exp()          
    ValueError: Can't take exponential of dimensional quantities: <Units: kg m-1 s-2>

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'exp', i=True) # pragma: no cover
            
        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.exp(inplace=True)

        # Remove misleading identities
        v.del_property('standard_name', None)
        v.del_property('long_name', None)       

        if inplace:
            v = None
        return v
    

    def sin(self, inplace=False, i=False):
        '''The trigonometric sine of the data, element-wise.

    Units are accounted for in the calculation. For example, the the
    sine of 90 degrees_east is 1.0, as is the sine of 1.57079632
    radians. If the units are not equivalent to radians (such as
    Kelvin) then they are treated as if they were radians.
    
    The Units are changed to '1' (nondimensionsal).

    The "standard_name" and "long_name" properties are removed from
    the result.
    
    .. seealso:: `cos`, `tan`
    
    :Parameters:
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
            The construct with the sine of data values. If the
            operation was in-place then `None` is returned.
    
    **Examples:**
    
    >>> f.Units
    <Units: degrees_north>
    >>> print(f.array)
    [[-90 0 90 --]]
    >>> f.sin()
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[-1.0 0.0 1.0 --]]
    
    >>> f.Units
    <Units: m s-1>
    >>> print(f.array)
    [[1 2 3 --]]
    >>> f.sin()
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[0.841470984808 0.909297426826 0.14112000806 --]]

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'sin', i=True) # pragma: no cover
            
        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.sin(inplace=True)

        # Remove misleading identities
        v.del_property('standard_name', None)
        v.del_property('long_name', None)
        
        if inplace:
            v = None
        return v
    

    def tan(self, inplace=False, i=False):
        '''The trigonometric tangent of the data, element-wise.

    Units are accounted for in the calculation, so that the the
    tangent of 180 degrees_east is 0.0, as is the sine of
    3.141592653589793 radians. If the units are not equivalent to
    radians (such as Kelvin) then they are treated as if they were
    radians.
    
    The Units are changed to '1' (nondimensionsal).

    The "standard_name" and "long_name" properties are removed from
    the result.
    
    .. seealso:: `cos`, `sin`
    
    :Parameters:
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
            The construct with the tangent of data values. If the
            operation was in-place then `None` is returned.
    
    **Examples:**
    
    >>> f.Units
    <Units: degrees_north>
    >>> print(f.array)
    [[-45 0 45 --]]
    >>> f.tan()
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[-1.0 0.0 1.0 --]]
    
    >>> f.Units
    <Units: m s-1>
    >>> print(f.array)
    [[1 2 3 --]]
    >>> f.tan()
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[1.55740772465 -2.18503986326 -0.142546543074 --]]

        '''     
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'tan', i=True) # pragma: no cover
            
        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.tan(inplace=True)

        # Remove misleading identities
        v.del_property('standard_name', None)
        v.del_property('long_name', None)        

        if inplace:
            v = None
        return v


    def log(self, base=None, inplace=False, i=False):
        '''The logarithm of the data array.

    By default the natural logarithm is taken, but any base may be
    specified.

    The "standard_name" and "long_name" properties are removed from
    the result.
    
    .. seealso:: `exp`
    
    :Parameters:
    
        base: number, optional
            The base of the logiarthm. By default a natural logiarithm
            is taken.    
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
            The construct with the logarithm of data values.
    
    **Examples:**
    
    >>> f.data
    <CF Data(1, 2): [[1, 2]]>
    >>> f.log().data
    <CF Data(1, 2): [[0.0, 0.69314718056]] ln(re 1)>
    
    >>> f.data
    <CF Data(1, 2): [[1, 2]] 2>
    >>> f.log().data
    <CF Data(1, 2): [[0.0, 0.69314718056]] ln(re 2 1)>
    
    >>> f.data
    <CF Data(1, 2): [[1, 2]] kg s-1 m-2>
    >>> f.log().data
    <CF Data(1, 2): [[0.0, 0.69314718056]] ln(re 1 m-2.kg.s-1)>
    
    >>> f.log(inplace=True)
    >>> f.Units
    <Units: >
    >>> f.log()
    ValueError: Can't take the logarithm to the base 2.718281828459045 of <Units: >

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'log', i=True) # pragma: no cover
            
        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.log(base, inplace=True)

        # Remove misleading identities
        v.del_property('standard_name', None)
        v.del_property('long_name', None)
        
        if inplace:
            v = None
        return v
    

    def trunc(self, inplace=False, i=False):
        '''Truncate the data, element-wise.

    The truncated value of the scalar ``x``, is the nearest integer
    ``i`` which is closer to zero than ``x`` is. I.e. the fractional
    part of the signed number ``x`` is discarded.
    
    .. versionadded:: 1.0
    
    .. seealso:: `ceil`, `floor`, `rint`
    
    :Parameters:
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
            The construct with truncated data. If the operation was
            in-place then `None` is returned.
    
    **Examples:**
    
    >>> print(f.array)
    [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
    >>> print(f.trunc().array)
    [-1. -1. -1. -1.  0.  1.  1.  1.  1.]
    >>> f.trunc(inplace=True)
    >>> print(f.array)
    [-1. -1. -1. -1.  0.  1.  1.  1.  1.]

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'trunc', i=True) # pragma: no cover

        if inplace:
            v = self
        else:
            v = self.copy()
            
        data = v.get_data(None)
        if data is not None:
            data.trunc(inplace=True)

        if inplace:
            v = None
        return v


    def unique(self):
        '''The unique elements of the data.

    :Returns:
    
        `Data`
            The unique data array values in a one dimensional `Data`
            object.
    
    **Examples:**
    
    >>> print(f.array)
    [[4 2 1]
     [1 2 3]]
    >>> print(f.unique().array)
    [1 2 3 4]
    >>> f[1, -1] = cf.masked
    >>> print(f.array)
    [[4 2 1]
     [1 2 --]]
    >>> print(f.unique().array)
    [1 2 4]

        '''
        data = self.get_data(None)
        if data is not None:
            return data.unique()

        raise ValueError(
            "ERROR: Can't get unique values when there is no data array")


    def identity(self, default='', strict=False, relaxed=False,
                 nc_only=False, relaxed_identity=None):
        '''Return the canonical identity.

    By default the identity is the first found of the following:
    
    * The "standard_name" property.
    * The "id" attribute, preceeded by ``'id%'``.
    * The "cf_role" property, preceeded by ``'cf_role='``.
    * The "axis" property, preceeded by ``'axis='``.
    * The "long_name" property, preceeded by ``'long_name='``.
    * The netCDF variable name, preceeded by ``'ncvar%'``.
    * The coordinate type (``'X'``, ``'Y'``, ``'Z'`` or ``'T'``).
    * The value of the *default* parameter.
    
    .. versionadded:: 3.0.0
    
    .. seealso:: `id`, `identities`
    
    :Parameters:
    
        default: optional
            If no identity can be found then return the value of the
            default parameter.
    
        strict: `bool`, optional 
            If True then only take the identity from the
            "standard_name" property or the "id" attribute, in that
            order.

        relaxed: `bool`, optional
            If True then only take the identity from the
            "standard_name" property, the "id" attribute, the
            "long_name" property or netCDF variable name, in that
            order.

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

        '''
        if relaxed_identity:
            _DEPRECATION_ERROR_KWARGS(self, 'identity',
                                      relaxed_identity=True) # pragma: no cover

        if nc_only:
            if strict:
                raise ValueError("'strict' and 'nc_only' parameters cannot both be True")
            
            if relaxed:
                raise ValueError("'relaxed' and 'nc_only' parameters cannot both be True")
            
            n = self.nc_get_variable(None)
            if n is not None:
                return 'ncvar%{0}'.format(n)
            
            return default
            
        n = self.get_property('standard_name', None)
        if n is not None:
            return '{0}'.format(n)

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
        
        for prop in  ('cf_role', 'axis', 'long_name'):
            n = self.get_property(prop, None)
            if n is not None:
                return '{0}={1}'.format(prop, n)
        #--- End: for

        n = self.nc_get_variable(None)
        if n is not None:
            return 'ncvar%{0}'.format(n)
        
        for ctype in ('X', 'Y', 'Z', 'T'):
            if getattr(self, ctype, False):
                return ctype
        #--- End: for

        return default


    def identities(self):
        '''Return all possible identities.

    The identities comprise:
    
    * The "standard_name" property.
    * The "id" attribute, preceeded by ``'id%'``.
    * The "cf_role" property, preceeded by ``'cf_role='``.
    * The "axis" property, preceeded by ``'axis='``.
    * The "long_name" property, preceeded by ``'long_name='``.
    * All other properties (including "standard_name"), preceeded by
      the property name and an ``'='``.
    * The coordinate type (``'X'``, ``'Y'``, ``'Z'`` or ``'T'``).
    * The netCDF variable name, preceeded by ``'ncvar%'``.
    
    .. versionadded:: 3.0.0
    
    .. seealso:: `id`, `identity`
    
    :Returns:
    
        `list`
            The identities.
    
    **Examples:**
    
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

        '''
        out = super().identities()

        i = getattr(self, 'id', None)
        if i is not None:
            # Insert id attribute
            i = 'id%{0}'.format(i)
            if not out:
                out = [i]
            else:
                out0 = out[0]
                if '=' in out0 or '%' in out0 or True in [a == out0 for a in 'XYZT']:
                    out.insert(0, i)
                else:
                    out.insert(1, i)
        #--- End: if

        for ctype in ('X', 'Y', 'Z', 'T'):
            if getattr(self, ctype, False):
                out.append(ctype)
        #--- End: for
        
        return out


    def inspect(self):
        '''Inspect the object for debugging.

    .. seealso:: `cf.inspect`
    
    :Returns: 
    
        `None`

        '''
        print(cf_inspect(self)) # pragma: no cover


    def get_data(self, default=ValueError()):
        '''Return the data.

    Note that a `Data` instance is returned. Use its `array` attribute
    to return the data as an independent `numpy` array.
    
    The units, calendar and fill value properties are, if set,
    inserted into the data.
    
    .. versionadded:: 1.7.0
    
    .. seealso:: `array`, `data`, `del_data`, `has_data`, `set_data`
    
    :Parameters:
    
        default: optional
            Return the value of the *default* parameter if data have
            not been set. If set to an `Exception` instance then it
            will be raised instead.
    
    :Returns:
    
            The data.
    
    **Examples:**
    
    >>> d = cf.Data(range(10))
    >>> f.set_data(d)
    >>> f.has_data()
    True
    >>> f.get_data()
    <CF Data(10): [0, ..., 9]>
    >>> f.del_data()
    <CF Data(10): [0, ..., 9]>
    >>> f.has_data()
    False
    >>> print(f.get_data(None))
    None
    >>> print(f.del_data(None))
    None

        '''
        return super().get_data(default=default, _units=False)


    def override_calendar(self, calendar, inplace=False,  i=False):
        '''Override the calendar of date-time units.

    The new calendar **need not** be equivalent to the original one
    and the data array elements will not be changed to reflect the new
    units. Therefore, this method should only be used when it is known
    that the data array values are correct but the calendar has been
    incorrectly encoded.
    
    Not to be confused with setting the `calendar` or `Units`
    attributes to a calendar which is equivalent to the original
    calendar
    
    .. seealso:: `calendar`, `override_units`, `units`, `Units`
    
    :Parameters:
    
        calendar: `str`
            The new calendar.
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
    TODO
    
    **Examples:**
    
    TODO
    
    >>> g = f.override_calendar('noleap')

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'override_calendar', i=True) # pragma: no cover

        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.override_calendar(calendar, inplace=True)
            v._custom['Units'] = data.Units
        else:
            if not v.Units.isreftime:
                raise ValueError(
                    "Can't override the calender of non-reference-time units: {0!r}".format(
                        self.Units))
                
            v.Units = Units(getattr(v.Units, 'units', None), calendar=calendar)

        if inplace:
            v = None
        return v


    def override_units(self, units, inplace=False, i=False):
        '''Override the units.

    The new units **need not** be equivalent to the original ones and
    the data array elements will not be changed to reflect the new
    units. Therefore, this method should only be used when it is known
    that the data array values are correct but the units have
    incorrectly encoded.
    
    Not to be confused with setting `units` or `Units` attributes to
    units which are equivalent to the original units.
    
    .. seealso:: `calendar`, `override_calendar`, `units`, `Units`
    
    :Parameters:
    
        units: `str` or `Units`
            The new units for the data array.
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
    TODO
    
    **Examples:**
    
    >>> f.Units
    <Units: hPa>
    >>> f.datum(0)
    100000.0
    >>> f.override_units('km')
    >>> f.Units
    <Units: km>
    >>> f.datum(0)
    100000.0
    >>> f.override_units(Units('watts'))
    >>> f.Units
    <Units: watts>
    >>> f.datum(0)
    100000.0

        '''        
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'override_units', i=True) # pragma: no cover

        if inplace:
            v = self
        else:
            v = self.copy()

        units = Units(units)
        
        data = v.get_data(None)
        if data is not None:
            data.override_units(units, inplace=True)
            v._custom['Units'] = units
        else:
            v.Units = units


        if inplace:
            v = None
        return v


    def rint(self, inplace=False, i=False):
        '''Round the data to the nearest integer, element-wise.

    .. versionadded:: 1.0
    
    .. seealso:: `ceil`, `floor`, `trunc`
    
    :Parameters:
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
            
            The construct with rounded data. If the operation was
            in-place then `None` is returned.
    
    **Examples:**
    
    >>> print(f.array)
    [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
    >>> print(f.rint().array)
    [-2. -2. -1. -1.  0.  1.  1.  2.  2.]
    >>> f.rint(inplace=True)
    >>> print(f.array)
    [-2. -2. -1. -1.  0.  1.  1.  2.  2.]

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'rint', i=True) # pragma: no cover
            
        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.rint(inplace=True)
            
        if inplace:
            v = None
        return v


    def round(self, decimals=0, inplace=False, i=False):
        '''Round the data to the given number of decimals.

    Values exactly halfway between rounded decimal values are rounded
    to the nearest even value. Thus 1.5 and 2.5 round to 2.0, -0.5 and
    0.5 round to 0.0, etc. Results may also be surprising due to the
    inexact representation of decimal fractions in the IEEE floating
    point standard and errors introduced when scaling by powers of
    ten.
     
    .. versionadded:: 1.1.4
    
    .. seealso:: `ceil`, `floor`, `rint`, `trunc`
    
    :Parameters:
    	
        decimals: `int`, optional
            Number of decimal places to round to (0 by default). If
            decimals is negative, it specifies the number of positions
            to the left of the decimal point.
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
            The construct with rounded data. If the operation was
            in-place then `None` is returned.
    
    **Examples:**
    
    >>> print(f.array)
    [-1.81, -1.41, -1.01, -0.91,  0.09,  1.09,  1.19,  1.59,  1.99])
    >>> print(f.round().array)
    [-2., -1., -1., -1.,  0.,  1.,  1.,  2.,  2.]
    >>> print(f.round(1).array)
    [-1.8, -1.4, -1. , -0.9,  0.1,  1.1,  1.2,  1.6,  2. ]
    >>> print(f.round(-1).array)
    [-0., -0., -0., -0.,  0.,  0.,  0.,  0.,  0.]

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'round', i=True) # pragma: no cover
            
        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.round(decimals=decimals, inplace=True)

        if inplace:
            v = self
        return v


    def roll(self, iaxis, shift, inplace=False, i=False):
        '''Roll the data along an axis.

    .. seealso:: `flatten`, `insert_dimension`, `flip`, `squeeze`,
                 `transpose`
    
    :Parameters:
    
        iaxis: `int`
            TODO
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    :Returns:
    
        TODO
    
    **Examples:**
    
    TODO

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'roll', i=True) # pragma: no cover

        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is not None:
            data.roll(iaxis, shift, inplace=True)

        if inplace:
            v = None
        return v


    def set_data(self, data, copy=True):
        '''Set the data.

    The units, calendar and fill value of the incoming `Data` instance
    are removed prior to insertion.
    
    .. versionadded:: 3.0.0
    
    .. seealso:: `data`, `del_data`, `get_data`, `has_data`
    
    :Parameters:
    
        data: `Data`
            The data to be inserted.
    
        copy: `bool`, optional
            If False then do not copy the data prior to insertion. By
            default the data are copied.
    
    :Returns:
    
        `None`
    
    **Examples:**
    
    >>> d = Data(range(10))
    >>> f.set_data(d)
    >>> f.has_data()
    True
    >>> f.get_data()
    <Data(10): [0, ..., 9]>
    >>> f.del_data()
    <Data(10): [0, ..., 9]>
    >>> f.has_data()
    False
    >>> print(f.get_data(None))
    None
    >>> print(f.del_data(None))
    None

        '''
        if not data.Units:
            units = getattr(self, 'Units', None)
            if units is not None:
                if copy:
                    copy = False
                    data = data.override_units(units, inplace=False)
                else:
                    data.override_units(units, inplace=True)
        #--- End: if

        if copy:
            data = data.copy()
   
        self._set_component('data', data, copy=False)


    def where(self, condition, x=None, y=None, inplace=False, i=False,
              _debug=False):
        '''Set data array elements depending on a condition.

    .. seealso:: `cf.masked`, `hardmask`, `subspace`
    
    :Parameters:
    
        TODO

    :Returns:
    
        TODO

    **Examples:**

        TODO

        '''
        if i:
            _DEPRECATION_ERROR_KWARGS(self, 'where', i=True) # pragma: no cover

        if inplace:
            v = self
        else:
            v = self.copy()

        data = v.get_data(None)
        if data is None:
            raise ValueError(
                "ERROR: Can't set data in nonexistent data array")

        try:
            condition_data = condition.get_data(None)
        except AttributeError:
            pass
        else:
            if condition_data is None:
                raise ValueError(
                    "ERROR: Can't set data from {} with no data array".format(
                        condition.__class__.__name__))

            condition = condition_data
            
        try:
            x_data = x.get_data(None)
        except AttributeError:
            pass
        else:
            if x_data is None:
                raise ValueError(
                    "ERROR: Can't set data from {} with no data array".format(
                        x.__class__.__name__))

            x = x_data
            
        try:
            y_data = y.get_data(None)
        except AttributeError:
            pass
        else:
            if y_data is None:
                raise ValueError(
                    "ERROR: Can't set data from {} with no data array".format(
                        y.__class__.__name__))

            y = y_data
            
        data.where(condition, x, y, inplace=True, _debug=_debug)

        if inplace:
            v = None
        return v

    
    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    @property
    def dtarray(self):
        '''Alias for `datetime_array`.

        '''
        return self.datetime_array

    
    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    @property
    def attributes(self):
        '''A dictionary of the attributes which are not CF properties.

        Deprecated at version 3.0.0.

        '''
        _DEPRECATION_ERROR_ATTRIBUTE(self, 'attributes')
    

    @property
    def Data(self):
        '''The `Data` object containing the data array.
        
        Deprecated at version 3.0.0. Use 'data' attribute or
        'get_data' method instead.

        '''
        _DEPRECATATION_ERROR_ATTRIBUTE(
            self, 'Data',
            "Use 'data' attribute or 'get_data' method instead.") # pragma: no cover
    @data.setter
    def Data(self, value):
        _DEPRECATATION_ERROR_ATTRIBUTE(
            self, 'Data',
            "Use 'data' attribute or 'set_data' method instead.") # pragma: no cover
    @data.deleter
    def Data(self):
        _DEPRECATATION_ERROR_ATTRIBUTE(
            self, 'Data',
            "Use 'data' attribute or 'del_data' method instead.") # pragma: no cover


    @property
    def dtvarray(self):
        '''A numpy array view the data array converted to date-time objects.

        Deprecated at version 3.0.0.

        '''
        _DEPRECATION_ERROR_ATTRIBUTE(self, 'dtvarray') # pragma: no cover

        
    @property
    def hasbounds(self):
        '''`True` if there are cell bounds.

    Deprecated at version 3.0.0. Use 'has_bounds' method instead.

    If present, cell bounds are stored in the `!bounds` attribute.
    
    **Examples:**
    
    >>> if c.hasbounds:
    ...     b = c.bounds

        '''
        _DEPRECATION_ERROR_ATTRIBUTE(self, 'hasbounds', "Use 'has_bounds' method instead")


    @property
    def hasdata(self):
        '''True if there is a data array.

    Deprecated at version 3.0.0. Use 'has_data' method instead.
    
    If present, the data array is stored in the `data` attribute.
    
    .. seealso:: `data`, `hasbounds`
    
    **Examples:**
    
    >>> if f.hasdata:
    ...     print(f.data)

        '''
        _DEPRECATION_ERROR_ATTRIBUTE(self, 'hasdata', "Use 'has_data' method instead")

        
    @property
    def unsafe_array(self):
        '''A numpy array of the data.

        Deprecated at version 3.0.0. Use 'array' attribute instead.

        '''      
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, 'unsafe_array',
            "Use 'array' attribute instead.") # pragma: no cover


    def asdatetime(self, i=False):
        '''Convert the internal representation of data array elements to
    date-time objects.
    
    Only applicable to construct with reference time units.
    
    If the calendar has not been set then the CF default calendar will be
    used and the units will be updated accordingly.
    
    .. seealso:: `asreftime`
    
    :Parameters:
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
    
    **Examples:**
    
    >>> t.asdatetime().dtype
    dtype('float64')
    >>> t.asdatetime().dtype
    dtype('O')

        '''
        _DEPRECATION_ERROR_METHOD(self, 'asdatetime') # pragma: no cover

        
    def asreftime(self, i=False):
        '''Convert the internal representation of data array elements
    to numeric reference times.
    
    Only applicable to constructs with reference time units.
    
    If the calendar has not been set then the CF default calendar will be
    used and the units will be updated accordingly.
    
    .. seealso:: `asdatetime`
    
    :Parameters:
    
        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.
    
        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.
        
    **Examples:**
    
    >>> t.asdatetime().dtype
    dtype('O')
    >>> t.asreftime().dtype
    dtype('float64')

        '''
        _DEPRECATION_ERROR_METHOD(self, 'asreftime') # pragma: no cover
    

    def expand_dims(self, position=0, i=False):
        '''Insert a size 1 axis into the data array.

        Deprecated at version 3.0.0. Use method 'insert_dimension'
        instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'expand_dims',
            "Use method 'insert_dimension' instead.") # pragma: no cover


    def insert_data(self, data, copy=True):
        '''Deprecated at version 3.0.0. Use method 'set_data' instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'insert_data',
            "Use method 'set_data' instead.") # pragma: no cover


    def name(self, default=None, identity=False, ncvar=False,
             relaxed_identity=None):
        '''Return a name for construct.

        Deprecated at version 3.0.0. Use method 'identity' instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'name',
            "Use method 'identity' instead") # pragma: no cover


    def remove_data(self):
        '''Remove and return the data array.

        Deprecated at version 3.0.0. Use method 'del_data' instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'remove_data',
            "Use method 'del_data' instead.") # pragma: no cover


    def select(self, *args, **kwargs):
        '''Deprecated at version 3.0.0.

        '''
        _DEPRECATION_ERROR_METHOD(self, 'select') # pragma: no cover


#--- End: class


class Subspace:
    '''TODO
    '''
    __slots__ = ('variable',)

    def __init__(self, variable):
        '''Set the contained variable.

        '''
        self.variable = variable


    def __getitem__(self, indices):
        '''Called to implement evaluation of x[indices].

    x.__getitem__(indices) <==> x[indices]

        '''
        return self.variable[indices]


    def __setitem__(self, indices, value):
        '''Called to implement assignment to x[indices]

    x.__setitem__(indices, value) <==> x[indices]

        '''
        if isinstance(value, self.__class__):
            value = value.data

        self.variable[indices] = value


#--- End: class
