from functools import reduce
from operator  import mul

import logging

from numpy import size as numpy_size

from . import PropertiesData

from ..functions    import parse_indices
from ..functions    import equivalent as cf_equivalent
from ..functions    import inspect    as cf_inspect
from ..functions    import (_DEPRECATION_ERROR_METHOD,
                            _DEPRECATION_ERROR_ATTRIBUTE)

from ..decorators import (_inplace_enabled,
                          _inplace_enabled_define_and_cleanup,
                          _deprecated_kwarg_check,
                          _manage_log_level_via_verbosity)

from ..query        import Query
from ..units        import Units

from ..data.data import Data


_units_None = Units()

_month_units = ('month', 'months')
_year_units = ('year', 'years', 'yr')

logger = logging.getLogger(__name__)


class PropertiesDataBounds(PropertiesData):
    '''Mixin class for a data array with descriptive properties and cell
    bounds.

    '''
    def __getitem__(self, indices):
        '''Return a subspace of the field construct defined by indices.

    x.__getitem__(indices) <==> x[indices]

        '''

        if indices is Ellipsis:
            return self.copy()

        # Parse the index
        if not isinstance(indices, tuple):
            indices = (indices,)

        arg0 = indices[0]
        if isinstance(arg0, str) and arg0 == 'mask':
            auxiliary_mask = indices[:2]
            indices2 = indices[2:]
        else:
            auxiliary_mask = None
            indices2 = indices

        indices, roll = parse_indices(self.shape, indices2, cyclic=True)

        if roll:
            new = self
            data = self.data
            axes = data._axes
            cyclic_axes = data._cyclic
            for iaxis, shift in roll.items():
                if axes[iaxis] not in cyclic_axes:
                    raise IndexError(
                        "Can't do a cyclic slice on a non-cyclic axis")

                new = new.roll(iaxis, shift)
        else:
            new = self.copy()  # data=False)

#       data = self.data

        if auxiliary_mask:
            findices = tuple(auxiliary_mask) + tuple(indices)
        else:
            findices = tuple(indices)

        cname = self.__class__.__name__
        logger.debug(
            '{}.__getitem__: shape    = {}'.format(cname, self.shape)
        )  # pragma: no cover
        logger.debug(
            '{}.__getitem__: indices2 = {}'.format(cname, indices2)
        )  # pragma: no cover
        logger.debug(
            '{}.__getitem__: indices  = {}'.format(cname, indices)
        )  # pragma: no cover
        logger.debug(
            '{}.__getitem__: findices = {}'.format(cname, findices)
        )  # pragma: no cover

        data = self.get_data(None)
        if data is not None:
            new.set_data(data[findices], copy=False)

        # Subspace the interior ring array, if there is one.
        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            new.set_interior_ring(interior_ring[tuple(indices)], copy=False)

        # Subspace the bounds, if there are any
        bounds = self.get_bounds(None)
        if bounds is not None:
            bounds_data = bounds.get_data(None)
            if bounds_data is not None:
                findices = list(findices)
#                if data.ndim <= 1 and not self.has_geometry():
                if bounds.ndim <= 2:
                    index = indices[0]
                    if isinstance(index, slice):
                        if index.step and index.step < 0:
                            # This scalar or 1-d variable has been
                            # reversed so reverse its bounds (as per
                            # 7.1 of the conventions)
                            findices.append(slice(None, None, -1))
                    elif data.size > 1 and index[-1] < index[0]:
                        # This 1-d variable has been reversed so
                        # reverse its bounds (as per 7.1 of the
                        # conventions)
                        findices.append(slice(None, None, -1))
                # --- End: if

                if auxiliary_mask:
                    findices[1] = [mask.insert_dimension(-1) for mask in
                                   findices[1]]

                logger.debug(
                    '{}.__getitem__: findices for bounds = {}'.format(
                        self.__class__.__name__, findices)
                )  # pragma: no cover

                new.bounds.set_data(bounds_data[tuple(findices)], copy=False)
        # --- End: if

        # Remove the direction, as it may now be wrong
        new._custom.pop('direction', None)

        # Return the new bounded variable
        return new

    def __eq__(self, y):
        '''The rich comparison operator ``==``

    x.__eq__(y) <==> x==y

        '''
        return self._binary_operation(y, '__eq__', False)

    def __ne__(self, y):
        '''The rich comparison operator ``!=``

    x.__ne__(y) <==> x!=y

        '''
        return self._binary_operation(y, '__ne__', False)

    def __ge__(self, y):
        '''The rich comparison operator ``>=``

    x.__ge__(y) <==> x>=y

        '''
        return self._binary_operation(y, '__ge__', False)

    def __gt__(self, y):
        '''The rich comparison operator ``>``

    x.__gt__(y) <==> x>y

        '''
        return self._binary_operation(y, '__gt__', False)

    def __le__(self, y):
        '''The rich comparison operator ``<=``

    x.__le__(y) <==> x<=y

        '''
        return self._binary_operation(y, '__le__', False)

    def __lt__(self, y):
        '''The rich comparison operator ``<``

    x.__lt__(y) <==> x<y

        '''
        return self._binary_operation(y, '__lt__', False)

    def __and__(self, other):
        '''The binary bitwise operation ``&``

    x.__and__(y) <==> x&y

        '''
        return self._binary_operation(other, '__and__', False)

    def __iand__(self, other):
        '''The augmented bitwise assignment ``&=``

    x.__iand__(y) <==> x&=y

        '''
        return self._binary_operation(other, '__iand__', False)

    def __rand__(self, other):
        '''The binary bitwise operation ``&`` with reflected operands

    x.__rand__(y) <==> y&x

        '''
        return self._binary_operation(other, '__rand__', False)

    def __or__(self, other):
        '''The binary bitwise operation ``|``

    x.__or__(y) <==> x|y

        '''
        return self._binary_operation(other, '__or__', False)

    def __ior__(self, other):
        '''The augmented bitwise assignment ``|=``

    x.__ior__(y) <==> x|=y

        '''
        return self._binary_operation(other, '__ior__', False)

    def __ror__(self, other):
        '''The binary bitwise operation ``|`` with reflected operands

    x.__ror__(y) <==> y|x

        '''
        return self._binary_operation(other, '__ror__', False)

    def __xor__(self, other):
        '''The binary bitwise operation ``^``

    x.__xor__(y) <==> x^y

        '''
        return self._binary_operation(other, '__xor__', False)

    def __ixor__(self, other):
        '''The augmented bitwise assignment ``^=``

    x.__ixor__(y) <==> x^=y

        '''
        return self._binary_operation(other, '__ixor__', False)

    def __rxor__(self, other):
        '''The binary bitwise operation ``^`` with reflected operands

    x.__rxor__(y) <==> y^x

        '''
        return self._binary_operation(other, '__rxor__', False)

    def __lshift__(self, y):
        '''The binary bitwise operation ``<<``

    x.__lshift__(y) <==> x<<y

        '''
        return self._binary_operation(y, '__lshift__', False)

    def __ilshift__(self, y):
        '''The augmented bitwise assignment ``<<=``

    x.__ilshift__(y) <==> x<<=y

        '''
        return self._binary_operation(y, '__ilshift__', False)

    def __rlshift__(self, y):
        '''The binary bitwise operation ``<<`` with reflected operands

    x.__rlshift__(y) <==> y<<x

        '''
        return self._binary_operation(y, '__rlshift__', False)

    def __rshift__(self, y):
        '''The binary bitwise operation ``>>``

    x.__lshift__(y) <==> x>>y

        '''
        return self._binary_operation(y, '__rshift__', False)

    def __irshift__(self, y):
        '''The augmented bitwise assignment ``>>=``

    x.__irshift__(y) <==> x>>=y

        '''
        return self._binary_operation(y, '__irshift__', False)

    def __rrshift__(self, y):
        '''The binary bitwise operation ``>>`` with reflected operands

    x.__rrshift__(y) <==> y>>x

        '''
        return self._binary_operation(y, '__rrshift__', False)

    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------
    def _binary_operation(self, other, method, bounds=True):
        '''Implement binary arithmetic and comparison operations.

    The operations act on the construct's data array with the numpy
    broadcasting rules.

    If the construct has bounds then they are operated on with the
    same data as the construct's data.

    It is intended to be called by the binary arithmetic and comparison
    methods, such as `!__sub__` and `!__lt__`.

    :Parameters:

        other:

        method: `str`
            The binary arithmetic or comparison method name (such as
            ``'__imul__'`` or ``'__ge__'``).

        bounds: `bool`, optional
            If False then ignore the bounds and remove them from the
            result. By default the bounds are operated on as well.

    :Returns:

            A new construct, or the same construct if the operation
            was in-place.

        '''
        inplace = method[2] == 'i'

        has_bounds = bounds and self.has_bounds()

        if has_bounds and inplace and other is self:
            other = other.copy()

        new = super()._binary_operation(other, method)

        if has_bounds:
            # try:
            #     other_has_bounds = other.has_bounds()
            # except AttributeError:
            #     other_has_bounds = False

            # if other_has_bounds:
            #     new_bounds = self.bounds._binary_operation(
            #         other.bounds, method)
            # else:
            if numpy_size(other) > 1:
                try:
                    other = other.insert_dimension(-1)
                except AttributeError:
                    other = numpy_expand_dims(other, -1)
            # --- End: if

            new_bounds = self.bounds._binary_operation(other, method)

            if not inplace:
                new.set_bounds(new_bounds, copy=False)
        # --- End: if

        if not bounds and new.has_bounds():
            new.del_bounds()

        if inplace:
            return self
        else:
            return new

    @_manage_log_level_via_verbosity
    def _equivalent_data(self, other, rtol=None, atol=None,
                         verbose=None):
        '''TODO

    Two real numbers ``x`` and ``y`` are considered equal if
    ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on absolute
    differences) and ``rtol`` (the tolerance on relative differences) are
    positive, typically very small numbers. See the *atol* and *rtol*
    parameters.

    :Parameters:

        atol: `float`, optional
            The tolerance on absolute differences between real
            numbers. The default value is set by the `ATOL` function.

        rtol: `float`, optional
            The tolerance on relative differences between real
            numbers. The default value is set by the `RTOL` function.

    :Returns:

        `bool`

        '''
        self_bounds = self.get_bounds(None)
        other_bounds = other.get_bounds(None)
        hasbounds = self_bounds is not None

        if hasbounds != (other_bounds is not None):
            # TODO: add traceback
            # TODO: improve message below
            logger.info(
                'One has bounds, the other does not')  # pragma: no cover
            return False

        try:
            direction0 = self.direction()
            direction1 = other.direction()
            if (direction0 != direction1 and
                    direction0 is not None and direction1 is not None):
                other = other.flip()
        except AttributeError:
            pass

        # Compare the data arrays
        if not super()._equivalent_data(
                other, rtol=rtol, atol=atol, verbose=verbose):
            # TODO: improve message below
            logger.info(
                'Non-equivalent data arrays')  # pragma: no cover
            return False

        if hasbounds:
            # Compare the bounds
            if not self_bounds._equivalent_data(
                    other_bounds, rtol=rtol, atol=atol, verbose=verbose):
                logger.info(
                    '{}: Non-equivalent bounds data: {!r}, {!r}'.format(
                        self.__class__.__name__, self_bounds.data,
                        other_bounds.data
                    )
                )  # pragma: no cover
                return False
        # --- End: if

        # Still here? Then the data are equivalent.
        return True

    def _YMDhms(self, attr):
        '''TODO
        '''
        out = super()._YMDhms(attr)
        out.del_bounds(None)
        return out

    def _matching_values(self, value0, value1, units=False):
        '''TODO

        '''
        if value1 is None:
            return False

        if units and isinstance(value0, str):
            return Units(value0).equals(Units(value1))

        if isinstance(value0, Query):
            return bool(value0.evaluate(value1))  # TODO vectors
        else:
            try:
                return value0.search(value1)
            except (AttributeError, TypeError):
                return self._equals(value1, value0)
        # --- End: if

        return False

    def _apply_superclass_data_oper(self, v, oper_name, *oper_args,
                                    bounds=True, interior_ring=False,
                                    **oper_kwargs):
        '''Define an operation that can be applied to the data array.

    .. versionadded:: 3.1.0

    :Parameters:

        v: the data array to apply the operations to (possibly in-place)

        oper_name: the string name for the desired operation, as it is
            defined (its method name) under the PropertiesData class, e.g.
            `sin` to apply PropertiesData.sin`.

            Note: there is no (easy) way to determine the name of a
            function/method within itself, without e.g. inspecting the stack
            (see rejected PEP 3130), so even though functions are named
            identically to those called  (e.g. both `sin`) the same
            name must be typed and passed into this method in each case.

            TODO: is there a way to prevent/bypass the above?

        oper_args, oper_kwargs: all of the arguments for `oper_name`.

        bounds: `bool`
            Whether or not there are cell bounds (to consider).

        interior_ring: `bool`
            Whether or not a geometry interior ring variable needs to
            be operated on.

        '''
        v = getattr(super(), oper_name)(*oper_args, **oper_kwargs)
        if v is None:  # from inplace operation in superclass method
            v = self

        # Now okay to mutate oper_kwargs as no longer needed in original form
        oper_kwargs.pop('inplace', None)
        if bounds:
            bounds = v.get_bounds(None)
            if bounds is not None:
                getattr(bounds, oper_name)(*oper_args, inplace=True,
                                           **oper_kwargs)
        # --- End: if

        if interior_ring:
            interior_ring = v.get_interior_ring(None)
            if interior_ring is not None:
                getattr(interior_ring, oper_name)(*oper_args, inplace=True,
                                                  **oper_kwargs)
        # --- End: if

        return v

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def cellsize(self):
        '''The cell sizes.

    If there are no cell bounds then the cell sizes are all zero.

    .. versionadded:: 2.0

    **Examples:**

    >>> print(c.bounds.array)
    [[-90. -87.]
     [-87. -80.]
     [-80. -67.]]
    >>> c.cellsize
    <CF Data(3,): [3.0, 7.0, 13.0] degrees_north>
    >>> print(d.cellsize.array)
    [  3.   7.  13.]
    >>> b = c.del_bounds()
    >>> c.cellsize
    <CF Data(3,): [0, 0, 0] degrees_north>

        '''
        data = self.get_bounds_data(None)
        if data is not None:
            if data.shape[-1] != 2:
                raise ValueError(
                    "Can only calculate cell sizes from bounds when there are "
                    "exactly two bounds per cell. Got {}".format(
                        data.shape[-1])
                )

            out = abs(data[..., 1] - data[..., 0])
            out.squeeze(-1, inplace=True)
            return out
        else:
            data = self.get_data(None)
            if data is not None:
                return Data.zeros(self.shape, units=self.Units)
        # --- End: if

        raise AttributeError(
            "Can't get cell sizes when there are no bounds nor coordinate data"
        )

    @property
    def dtype(self):
        '''Numpy data-type of the data array.

    .. versionadded:: 2.0

    **Examples:**

    >>> c.dtype
    dtype('float64')
    >>> import numpy
    >>> c.dtype = numpy.dtype('float32')

        '''
        data = self.get_data(None)
        if data is not None:
            return data.dtype

        bounds = self.get_bounds_data(None)
        if bounds is not None:
            return bounds.dtype

        raise AttributeError("{} doesn't have attribute 'dtype'".format(
            self.__class__.__name__))

    @dtype.setter
    def dtype(self, value):
        data = self.get_data(None)
        if data is not None:
            data.dtype = value

        bounds = self.get_bounds_data(None)
        if bounds is not None:
            bounds.dtype = value

    @property
    def isperiodic(self):
        '''TODO

    .. versionadded:: 2.0

    >>> print(c.period())
    None
    >>> c.isperiodic
    False
    >>> print(c.period(cf.Data(360, 'degeres_east')))
    None
    >>> c.isperiodic
    True
    >>> c.period(None)
    <CF Data(): 360 degrees_east>
    >>> c.isperiodic
    False

    '''
        period = self.period()
        if period is not None:
            return True

        bounds = self.get_bounds(None)
        if bounds is not None:
            return bounds.period is not None

#        return self._custom.get('period', None) is not None

    @property
    def lower_bounds(self):
        '''The lower bounds of cells.

    If there are no cell bounds then the coordinates are used as the
    lower bounds.

    .. versionadded:: 2.0

    .. seealso:: `upper_bounds`

    **Examples:**

    >>> print(c.array)
    [4  2  0]
    >>> print(c.bounds.array)
    [[ 5  3]
     [ 3  1]
     [ 1 -1]]
    >>> c.lower_bounds
    <CF Data(3): [3, 1, -1]>
    >>> b = c.del_bounds()
    >>> c.lower_bounds
    <CF Data(3): [4, 2, 0]>

        '''
        data = self.get_bounds_data(None)
        if data is not None:
            out = data.minimum(-1)
            out.squeeze(-1, inplace=True)
            return out
        else:
            data = self.get_data(None)
            if data is not None:
                return data.copy()
        # --- End: if

        raise AttributeError(
            "Can't get lower bounds when there are no bounds nor coordinate "
            "data"
        )

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
#        return super().Units

        data = self.get_data(None)
        if data is not None:
            # Return the units of the data
            return data.Units

#        print('TODO RECURISION HERE')
#        bounds = self.get_bounds(None)
#        if bounds is not None:
#            data = bounds.get_data(None)
#            if data is not None:
#                # Return the units of the bounds data
#                return data.Units
#        # --- End: if

        try:
            return self._custom['Units']
        except KeyError:
            # if bounds is None:
            self._custom['Units'] = _units_None
            return _units_None
#            else:
#                try:
#                    return bounds._custom['Units']
#                except KeyError:
#                    bounds._custom['Units'] = _units_None
#        # --- End: try

#        return _units_None

    @Units.setter
    def Units(self, value):
        PropertiesData.Units.fset(self, value)

        # Set the Units on the bounds
        bounds = self.get_bounds(None)
        if bounds is not None:
            bounds.Units = value

# Moved to parent class at v3.4.1
#        # Set the Units on the period
#        period = self._custom.get('period')
#        if period is not None:
#            period = period.copy()
#            period.Units = value
#            self._custom['period'] = period

    @Units.deleter
    def Units(self):
        PropertiesData.Units.fdel(self)

    @property
    def upper_bounds(self):
        '''The upper bounds of cells.

    If there are no cell bounds then the coordinates are used as the
    upper bounds.

    .. versionadded:: 2.0

    .. seealso:: `lower_bounds`

    **Examples:**

    >>> print(c.array)
    [4  2  0]
    >>> print(c.bounds.array)
    [[ 5  3]
     [ 3  1]
     [ 1 -1]]
    >>> c.upper_bounds
    <CF Data(3): [5, 3, 1]>
    >>> b = c.del_bounds()
    >>> c.upper_bounds
    <CF Data(3): [4, 2, 0]>

        '''
        data = self.get_bounds_data(None)
        if data is not None:
            out = data.maximum(-1)
            out.squeeze(-1, inplace=True)
            return out
        else:
            data = self.get_data(None)
            if data is not None:
                return data.copy()
        # --- End: if

        raise AttributeError(
            "Can't get upper bounds when there are no bounds nor coordinate "
            "data"
        )

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def mask_invalid(self, inplace=False, i=False):
        '''Mask the array where invalid values occur.

    Note that:

    * Invalid values are Nan or inf

    * Invalid values in the results of arithmetic operations only
      occur if the raising of `FloatingPointError` exceptions has been
      suppressed by `cf.Data.seterr`.

    * If the raising of `FloatingPointError` exceptions has been
      allowed then invalid values in the results of arithmetic
      operations it is possible for them to be automatically converted
      to masked values, depending on the setting of
      `cf.Data.mask_fpe`. In this case, such automatic conversion
      might be faster than calling `mask_invalid`.

    .. seealso:: `cf.Data.mask_fpe`, `cf.Data.seterr`

    :Parameters:

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.

    :Returns:

            The construct with masked elements.

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
        # Set bounds to True to bypass 'if bounds' check in call:
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'mask_invalid',
            bounds=True, inplace=inplace, i=i)

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
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
        try:
            return super().dtype
        except AttributeError as error:
            bounds = self.get_bounds(None)
            if bounds is not None:
                return bounds.dtype

            raise AttributeError(error)

    @dtype.setter
    def dtype(self, value):
        # DCH - allow dtype to be set before data c.f.  Units
        data = self.get_data(None)
        if data is not None:
            self.Data.dtype = value

    @dtype.deleter
    def dtype(self):
        data = self.get_data(None)
        if data is not None:
            del self.Data.dtype

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def ceil(self, bounds=True, inplace=False, i=False):
        '''The ceiling of the data, element-wise.

    The ceiling of ``x`` is the smallest integer ``n``, such that
     ``n >= x``.

    .. versionadded:: 1.0

    .. seealso:: `floor`, `rint`, `trunc`

    :Parameters:

        bounds: `bool`, optional
            If False then do not alter any bounds. By default any
            bounds are also altered.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'ceil',
            bounds=bounds, inplace=inplace, i=i)

    def chunk(self, chunksize=None):
        '''Partition the data array.

    :Parameters:

        chunksize: `int`, optional
            Set the new chunksize, in bytes.

    :Returns:

        `None`

    **Examples:**

    >>> c.chunksize()

    >>> c.chunksize(1e8)

        '''
        super().chunk(chunksize)

        # Chunk the bounds, if they exist.
        bounds = self.get_bounds(None)
        if bounds is not None:
            bounds.chunk(chunksize)

        # Chunk the interior ring, if it exists.
        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            interior_ring.chunk(chunksize)

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def clip(self, a_min, a_max, units=None, bounds=True,
             inplace=False, i=False):
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

        bounds: `bool`, optional
            If False then do not alter any bounds. By default any bounds
            are also altered.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'clip', a_min,
            a_max, bounds=bounds, inplace=inplace, i=i, units=units)

    def close(self):
        '''Close all files referenced by the construct.

    Note that a closed file will be automatically re-opened if its
    contents are subsequently required.

    .. seealso:: `files`

    :Returns:

        `None`

    **Examples:**

    >> c.close()

        '''
        super().close()

        bounds = self.get_bounds(None)
        if bounds is not None:
            bounds.close()

        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            interior_ring.close()

    @classmethod
    def concatenate(cls, variables, axis=0, _preserve=True):
        '''Join a sequence of variables together.

    :Parameters:

        variables: sequence of constructs

        axis: `int`, optional

    :Returns:

        TODO
        '''
        variable0 = variables[0]

        if len(variables) == 1:
            return variable0.copy()

        out = super().concatenate(variables, axis=axis, _preserve=_preserve)

        bounds = variable0.get_bounds(None)
        if bounds is not None:
            bounds = bounds.concatenate([v.get_bounds() for v in variables],
                                        axis=axis,
                                        _preserve=_preserve)
            out.set_bounds(bounds, copy=False)

        interior_ring = variable0.get_interior_ring(None)
        if interior_ring is not None:
            interior_ring = interior_ring.concatenate(
                [v.get_interior_ring() for v in variables],
                axis=axis,
                _preserve=_preserve)
            out.set_interior_ring(interior_ring, copy=False)

        return out

# AT2
#
#    @classmethod
#    def arctan2(cls, y, x, bounds=True):
#        '''Take the "two-argument" trigonometric inverse tangent
#    element-wise for `y`/`x`.
#
#    Explicitly this returns, for all corresponding elements, the angle
#    between the positive `x` axis and the line to the point (`x`, `y`),
#    where the signs of both `x` and `y` are taken into account to
#    determine the quadrant. Such knowledge of the signs of `x` and `y`
#    are lost when the quotient is input to the standard "one-argument"
#    `arctan` function, such that use of `arctan` leaves the quadrant
#    ambiguous. `arctan2` may therefore be preferred.
#
#    Units are ignored in the calculation. The result has units of radians.
#
#    .. versionadded:: 3.2.0
#
#    .. seealso:: `arctan`, `tan`
#
#    :Parameters:
#
#        y: `Data`
#            The data array to provide the numerator elements, corresponding
#            to the `y` coordinates in the `arctan2` definition.
#
#        x: `Data`
#            The data array to provide the denominator elements,
#            corresponding to the `x` coordinates in the `arctan2`
#            definition.
#
#        bounds: `bool`, optional
#            If False then do not alter any bounds. By default any
#            bounds are also altered. Note that bounds will only be changed
#            if both `x` and `y` have bounds to consider.
#
#    :Returns:
#
#        The construct with the "two-argument" trigonometric inverse tangent
#        of data values. If the operation was in-place then `None` is
#        returned.
#
#    **Examples:**
#
#    TODO
#
#        '''
#        out = super().arctan2(y, x)
#
#        if bounds:
#            bounds_y = y.get_bounds(None)
#            bounds_x = x.get_bounds(None)
#            if bounds_x is not None and bounds_y is not None:
#                bounds = Data.arctan2(x.get_bounds(), y.get_bounds())
#                out.set_bounds(bounds, copy=False)
#
#        return out

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def cos(self, bounds=True, inplace=False,  i=False):
        '''Take the trigonometric cosine of the data element-wise.

    Units are accounted for in the calculation, so that the cosine
    of 90 degrees_east is 0.0, as is the cosine of 1.57079632
    radians. If the units are not equivalent to radians (such as
    Kelvin) then they are treated as if they were radians.

    The output units are '1' (nondimensional).

    .. seealso:: `arccos`, `sin`, `tan`, `cosh`

    :Parameters:

        bounds: `bool`, optional
            If False then do not alter any bounds. By default any
            bounds are also altered.

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
    >>> g = f.cos()
    >>> g.Units
    <Units: 1>
    >>> print(g.array)
    [[0.0 1.0 0.0 --]]

    >>> f.Units
    <Units: m s-1>
    >>> print(f.array)
    [[1 2 3 --]]
    >>> f.cos(inplace=True)
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[0.540302305868 -0.416146836547 -0.9899924966 --]]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'cos', bounds=bounds)

    def creation_commands(self, representative_data=False,
                          namespace='cf', indent=0, string=True,
                          name='c', data_name='d', bounds_name='b',
                          interior_ring_name='i'):
        '''Return the commands that would create the construct.

    .. versionadded:: 3.2.0

    .. seealso:: `cf.Data.creation_commands`,
                 `cf.Field.creation_commands`

    :Parameters:

        representative_data: `bool`, optional
            Return one-line representations of `Data` instances, which
            are not executable code but prevent the data being
            converted in its entirety to a string representation.

        namespace: `str`, optional
            The namespace containing classes of the ``cf``
            package. This is prefixed to the class name in commands
            that instantiate instances of ``cf`` objects. By default,
            *namespace* is ``'cf'``, i.e. it is assumed that ``cf``
            was imported as ``import cf``.

            *Parameter example:*
              If ``cf`` was imported as ``import cf as cfp`` then set
              ``namespace='cfp'``

            *Parameter example:*
              If ``cf`` was imported as ``from cf import *`` then set
              ``namespace=''``

        indent: `int`, optional
            Indent each line by this many spaces. By default no
            indentation is applied. Ignored if *string* is False.

        string: `bool`, optional
            If False then return each command as an element of a
            `list`. By default the commands are concatenated into
            a string, with a new line inserted between each command.

    :Returns:

        `str` or `list`
            The commands in a string, with a new line inserted between
            each command. If *string* is False then the separate
            commands are returned as each element of a `list`.

    **Examples:**

        TODO

        '''
        if name in (data_name, bounds_name, interior_ring_name):
            raise ValueError(
                "'name' parameter can not have the same value as "
                "any of the 'data_name', 'bounds_name', or "
                "'interior_ring_name' parameters: {!r}".format(
                    name))

        if data_name in (name, bounds_name, interior_ring_name):
            raise ValueError(
                "'data_name' parameter can not have the same value as "
                "any of the 'name', 'bounds_name', or "
                "'interior_ring_name'parameters: {!r}".format(
                    data_name))

        out = super().creation_commands(
            representative_data=representative_data, indent=0,
            namespace=namespace, string=False, name=name,
            data_name=data_name)

        namespace0 = namespace
        if namespace0:
            namespace = namespace+"."
        else:
            namespace = ""

        indent = ' ' * indent

        # Geometry type
        geometry = self.get_geometry(None)
        if geometry is not None:
            out.append("{}.set_geometry({!r})".format(name, geometry))

        bounds = self.get_bounds(None)
        if bounds is not None:
            out.extend(bounds.creation_commands(
                representative_data=representative_data, indent=0,
                namespace=namespace0, string=False, name=bounds_name,
                data_name=data_name))

            out.append("{}.set_bounds({})".format(name, bounds_name))

        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            out.extend(interior_ring.creation_commands(
                representative_data=representative_data, indent=0,
                namespace=namespace0, string=False,
                name=interior_ring_name, data_name=data_name))

            out.append("{}.set_interior_ring({})".format(name,
                                                         interior_ring_name))

        if string:
            out[0] = indent+out[0]
            out = ('\n'+indent).join(out)

        return out

    def cyclic(self, axes=None, iscyclic=True):
        '''Set the cyclicity of axes of the data array.

    .. seealso:: `iscyclic`

    :Parameters:

        axes: (sequence of) `int`
            The axes to be set. Each axis is identified by its integer
            position. By default no axes are set.

        iscyclic: `bool`, optional
            If False then the axis is set to be non-cyclic. By default
            the axis is set to be cyclic.

    :Returns:

        `set`

    **Examples:**

        TODO

        '''
        out = super().cyclic(axes, iscyclic)

        if axes is None:
            return out

        bounds = self.get_bounds(None)
        if bounds is not None:
            axes = self._parse_axes(axes)
            bounds.cyclic(axes, iscyclic)

        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            axes = self._parse_axes(axes)
            interior_ring.cyclic(axes, iscyclic)

        return out

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
        self_special = self._private['special_attributes']
        other_special = other._private['special_attributes']
        if set(self_special) != set(other_special):
            if traceback:
                print("%s: Different attributes: %s" %
                      (self.__class__.__name__,
                       set(self_special).symmetric_difference(other_special)))
            return False

        for attr, x in self_special.items():
            y = other_special[attr]

            result = cf_equivalent(x, y, rtol=rtol, atol=atol,
                                   traceback=traceback)

            if not result:
                if traceback:
                    print("{}: Different {} attributes: {!r}, {!r}".format(
                        self.__class__.__name__, attr, x, y))
                return False
        # --- End: for

        # ------------------------------------------------------------
        # Check the data
        # ------------------------------------------------------------
        if not self._equivalent_data(other, rtol=rtol, atol=atol,
                                     traceback=traceback):
            # add traceback
            return False

        return True

    def contiguous(self, overlap=True):
        '''Return True if a construct has contiguous cells.

    A construct is contiguous if its cell boundaries match up, or
    overlap, with the boundaries of adjacent cells.

    In general, it is only possible for a zero, 1 or 2 dimensional
    construct with bounds to be contiguous. A size 1 construct with
    any number of dimensions is always contiguous.

    An exception occurs if the construct is multdimensional and has
    more than one element.

    .. versionadded:: 2.0

    :Parameters:

        overlap : bool, optional
            If False then 1-d cells with two vertices and with
            overlapping boundaries are not considered contiguous. By
            default such cells are not considered contiguous.

            .. note:: The value of the *overlap* parameter does not
                      affect any other types of cell, for which a
                      necessary (but not sufficient) condition for
                      contiguousness is that adjacent cells do not
                      overlap.

    :Returns:

        `bool`
            Whether or not the construct's cells are contiguous.

    **Examples:**

    >>> c.has_bounds()
    False
    >>> c.contiguous()
    False

    >>> print(c.bounds[:, 0])
    [  0.5   1.5   2.5   3.5 ]
    >>> print(c.bounds[:, 1])
    [  1.5   2.5   3.5   4.5 ]
    >>> c.contiuous()
    True

    >>> print(c.bounds[:, 0])
    [  0.5   1.5   2.5   3.5 ]
    >>> print(c.bounds[:, 1])
    [  2.5   3.5   4.5   5.5 ]
    >>> c.contiuous()
    True
    >>> c.contiuous(overlap=False)
    False

        '''
        bounds = self.get_bounds_data(None)
        if bounds is None:
            return False

        ndim = self.ndim
        nbounds = bounds.shape[-1]

        if self.size == 1:
            return True

        period = self.autoperiod().period()

        if ndim == 2:
            if nbounds != 4:
                raise ValueError("Can't tell if {}-d cells with {} vertices "
                                 "are contiguous".format(ndim, nbounds))

            # Check cells (j, i) and cells (j, i+1) are contiguous
            diff = bounds[:, :-1, 1] - bounds[:, 1:, 0]
            if period is not None:
                diff = diff % period

            if diff.any():
                return False

            diff = bounds[:, :-1, 2] - bounds[:, 1:, 3]
            if period is not None:
                diff = diff % period

            if diff.any():
                return False

            # Check cells (j, i) and (j+1, i) are contiguous
            diff = bounds[:-1, :, 3] - bounds[1:, :, 0]
            if period is not None:
                diff = diff % period

            if diff.any():
                return False

            diff = bounds[:-1, :, 2] - bounds[1:, :, 1]
            if period is not None:
                diff = diff % period

            if diff.any():
                return False

            return True

        if ndim > 2:
            raise ValueError("Can't tell if {}-d cells "
                             "are contiguous".format(ndim))

        if nbounds != 2:
            raise ValueError("Can't tell if {}-d cells with {} vertices "
                             "are contiguous".format(ndim, nbounds))

        lower = bounds[1:, 0]
        upper = bounds[:-1, 1]

        if not overlap:
            diff = lower - upper
            if period is not None:
                diff = diff % period

            return not diff.any()
        else:
            direction = self.direction()
            if direction is None:
                return (lower <= upper).all() or (lower >= upper).all()

            if direction:
                return (lower <= upper).all()
            else:
                return (lower >= upper).all()

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
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
            units days since the original reference time in the
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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            'convert_reference_time', inplace=inplace, i=i, units=units,
            calendar_months=calendar_months, calendar_years=calendar_years)

    def get_property(self, prop, default=ValueError(), bounds=False):
        '''Get a CF property.

    .. versionadded:: 3.2.0

    .. seealso:: `clear_properties`, `del_property`, `has_property`,
                 `properties`, `set_property`

    :Parameters:

        prop: `str`
            The name of the CF property.

            *Parameter example:*
              ``prop='long_name'``

        default: optional
            Return the value of the *default* parameter if the
            property does not exist. If set to an `Exception` instance
            then it will be raised instead.

        bounds: `bool`
            TODO 1.8

    :Returns:

            The value of the named property or the default value, if
            set.

    **Examples:**

    >>> f.set_property('project', 'CMIP7')
    >>> f.has_property('project')
    True
    >>> f.get_property('project')
    'CMIP7'
    >>> f.del_property('project')
    'CMIP7'
    >>> f.has_property('project')
    False
    >>> print(f.del_property('project', None))
    None
    >>> print(f.get_property('project', None))
    None

        '''
        out = super().get_property(prop, None)
        if out is not None:
            return out

        if bounds and self.has_bounds():
            out = self.get_bounds().get_property(prop, None)
            if out is not None:
                return out
        # --- End: if

        return super().get_property(prop, default)

    @_inplace_enabled
    def flatten(self, axes=None, inplace=False):
        '''Flatten axes of the data

    Any subset of the axes may be flattened.

    The shape of the data may change, but the size will not.

    The flattening is executed in row-major (C-style) order. For
    example, the array ``[[1, 2], [3, 4]]`` would be flattened across
    both dimensions to ``[1 2 3 4]``.

    .. versionadded:: 3.0.2

    .. seealso:: `insert_dimension`, `flip`, `swapaxes`, `transpose`

    :Parameters:

        axes: (sequence of) int or str, optional
            Select the axes.  By default all axes are flattened. The
            *axes* argument may be one, or a sequence, of:

              * An internal axis identifier. Selects this axis.

              * An integer. Selects the axis corresponding to the given
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
        # Note the 'axes' argument can change mid-method meaning it is not
        # possible to consolidate this method using a call to
        # _apply_superclass_data_operations, despite mostly the same logic.
        v = _inplace_enabled_define_and_cleanup(self)
        super(PropertiesDataBounds, v).flatten(axes, inplace=True)

        bounds = v.get_bounds(None)
        if bounds is not None:
            axes = self._parse_axes(axes)
            bounds.flatten(axes, inplace=True)

        interior_ring = v.get_interior_ring(None)
        if interior_ring is not None:
            axes = self._parse_axes(axes)
            interior_ring.flatten(axes, inplace=True)

        return v

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def floor(self, bounds=True, inplace=False, i=False):
        '''Floor the data array, element-wise.

    The floor of ``x`` is the largest integer ``n``, such that ``n <= x``.

    .. versionadded:: 1.0

    .. seealso:: `ceil`, `rint`, `trunc`

    :Parameters:

        bounds: `bool`, optional
            If False then do not alter any bounds. By default any
            bounds are also altered.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'floor',
            bounds=bounds, inplace=inplace, i=i)

    def direction(self):
        '''Return `None`, indicating that it is not specified whether the
    values are increasing or decreasing.

    .. versionadded:: 2.0

    :Returns:

        `None`

    **Examples:**

    >>> c.direction()
    None

        '''
        return

    def match_by_property(self, *mode, **properties):
        '''Determine whether or not a variable satisfies conditions.

    Conditions may be specified on the variable's attributes and CF
    properties.

    :Parameters:

    :Returns:

        `bool`
            Whether or not the variable matches the given criteria.

    **Examples:**

    TODO

        '''
        _or = False
        if mode:
            if len(mode) > 1:
                raise ValueError("Can provide at most one positional argument")

            x = mode[0]
            if x == 'or':
                _or = True
            elif x != 'and':
                raise ValueError(
                    "Positional argument, if provided, must one of 'or', 'and'"
                )
        # --- End: if

        if not properties:
            return True

        self_properties = self.properties()

        ok = True
        for name, value0 in properties.items():
            value1 = self_property.get(name)
            ok = self._matching_values(value0, value1, units=(name == 'units'))

            if _or:
                if ok:
                    break
            elif not ok:
                break
        # --- End: for

        return ok

    def match_by_identity(self, *identities):
        '''Determine whether or not a variable satisfies conditions.

    Conditions may be specified on the variable's attributes and CF
    properties.

    :Parameters:

    :Returns:

        `bool`
            Whether or not the variable matches the given criteria.

    **Examples:**

        TODO

        '''
        # Return all constructs if no identities have been provided
        if not identities:
            return True

        identities = self.identities()

        ok = False
        for value0 in identities:
            for value1 in self_identities:
                ok = self._matching_values(value0, value1)
                if ok:
                    break
            # --- End: for

            if ok:
                break
        # --- End: for

        return ok

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def override_calendar(self, calendar, inplace=False, i=False):
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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            'override_calendar', calendar, bounds=True,
            interior_ring=False, inplace=inplace, i=i)

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def override_units(self, units, inplace=False, i=False):
        '''Override the units.

    The new units need not be equivalent to the original ones, and the
    data array elements will not be changed to reflect the new
    units. Therefore, this method should only be used when it is known
    that the data array values are correct but the units have
    incorrectly encoded.

    Not to be confused with setting the `units` or `Units` attribute
    to units which are equivalent to the original units.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            'override_units', units, bounds=True, interior_ring=False,
            inplace=inplace, i=i)

    def get_filenames(self):
        '''Return the name of the file or files containing the data.

    The names of the file or files containing the bounds data are also
    returned.

    :Returns:

        `set`
            The file names in normalized, absolute form. If all of the
            data are in memory then an empty `set` is returned.

        '''
        out = super().get_filenames()

        data = self.get_bounds_data(None)
        if data is not None:
            out.update(data.get_filenames())

        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            data = interior_ring.get_data(None)
            if data is not None:
                out.update(interior_ring.get_filenames())
        # --- End: if

        return out

    @_inplace_enabled
    @_manage_log_level_via_verbosity
    def halo(self, size, axes=None, tripolar=None, fold_index=-1,
             inplace=False, verbose=None):
        '''Expand the data by adding a halo.

    The halo may be applied over a subset of the data dimensions and
    each dimension may have a different halo size (including
    zero). The halo region is populated with a copy of the proximate
    values from the original data.

    Corresponding axes exapnded in the bounds, if present.

    **Cyclic axes**

    A cyclic axis that is expanded with a halo of at least size 1 is
    no longer considered to be cyclic.

    **Tripolar domains**

    Data for global tripolar domains are a special case in that a halo
    added to the northern end of the "Y" axis must be filled with
    values that are flipped in "X" direction. Such domains need to be
    explicitly indicated with the *tripolar* parameter.

    .. versionadded:: 3.5.0

    :Parameters:

        size: `int` or `dict`
            Specify the size of the halo for each axis.

            If *size* is a non-negative `int` then this is the halo
            size that is applied to all of the axes defined by the
            *axes* parameter.

            Alternatively, halo sizes may be assigned to axes
            individually by providing a `dict` for which a key
            specifies an axis (defined by its integer position in the
            data) with a corresponding value of the halo size for that
            axis. Axes not specified by the dictionary are not
            expanded, and the *axes* parameter must not also be set.

            *Parameter example:*
              Specify a halo size of 1 for all otherwise selected
              axes: ``size=1``

            *Parameter example:*
              Specify a halo size of zero ``size=0``. This results in
              no change to the data shape.

            *Parameter example:*
              For data with three dimensions, specify a halo size of 3
              for the first dimension and 1 for the second dimension:
              ``size={0: 3, 1: 1}``. This is equivalent to ``size={0:
              3, 1: 1, 2: 0}``

            *Parameter example:*
              Specify a halo size of 2 for the first and last
              dimensions `size=2, axes=[0, -1]`` or equivalently
              ``size={0: 2, -1: 2}``.

        axes: (sequence of) `int`
            Select the domain axes to be expanded, defined by their
            integer positions in the data. By default, or if *axes* is
            `None`, all axes are selected. No axes are expanded if
            *axes* is an empty sequence.

        tripolar: `dict`, optional
            A dictionary defining the "X" and "Y" axes of a global
            tripolar domain. This is necessary because in the global
            tripolar case the "X" and "Y" axes need special treatment,
            as described above. It must have keys ``'X'`` and ``'Y'``,
            whose values identify the corresponding domain axis
            construct by their integer positions in the data.

            The "X" and "Y" axes must be a subset of those identified
            by the *size* or *axes* parameter.

            See the *fold_index* parameter.

            *Parameter example:*
              Define the "X" and Y" axes by positions 2 and 1
              respectively of the data: ``tripolar={'X': 2, 'Y': 1}``

        fold_index: `int`, optional
            Identify which index of the "Y" axis corresponds to the
            fold in "X" axis of a tripolar grid. The only valid values
            are ``-1`` for the last index, and ``0`` for the first
            index. By default it is assumed to be the last
            index. Ignored if *tripolar* is `None`.

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

        verbose: `int` or `None`, optional
            If an integer from ``0`` to ``3``, corresponding to increasing
            verbosity (else ``-1`` as a special case of maximal and extreme
            verbosity), set for the duration of the method call (only) as
            the minimum severity level cut-off of displayed log messages,
            regardless of the global configured `cf.LOG_LEVEL`.

            Else, if `None` (the default value), log messages will be
            filtered out, or otherwise, according to the value of the
            `cf.LOG_LEVEL` setting.

            Overall, the higher a non-negative integer that is set (up to
            a maximum of ``3``) the more description that is printed to
            convey information about the operation.

    :Returns:

            The expanded data, or `None` if the operation was
            in-place.

    **Examples:**

    TODO

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'halo',
            bounds=True, interior_ring=True, inplace=inplace,
            size=size, axes=axes, tripolar=tripolar,
            fold_index=fold_index, verbose=verbose)

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def flip(self, axes=None, inplace=False, i=False):
        '''Flip (reverse the direction of) data dimensions.

    .. seealso:: `insert_dimension`, `squeeze`, `transpose`, `unsqueeze`

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

            The construct with flipped axes, or `None` if the operation
            was in-place.

    **Examples:**

    >>> f.flip()
    >>> f.flip(1)
    >>> f.flip([0, 1])

    >>> g = f[::-1, :, ::-1]
    >>> f.flip([2, 0]).equals(g)
    True

        '''
        v = _inplace_enabled_define_and_cleanup(self)
        super(PropertiesDataBounds, v).flip(axes=axes, inplace=True)

        interior_ring = v.get_interior_ring(None)
        if interior_ring is not None:
            # --------------------------------------------------------
            # Flip the interior ring. Do this before flipping the
            # bounds because the axes argument might get changed
            # during that operation.
            # --------------------------------------------------------
            interior_ring.flip(axes, inplace=True)

        bounds = v.get_bounds(None)
        if bounds is not None:
            # --------------------------------------------------------
            # Flip the bounds.
            #
            # As per section 7.1 in the CF conventions: i) if the
            # variable is 0 or 1 dimensional then flip all dimensions
            # (including the trailing size 2 dimension); ii) if
            # the variable has 2 or more dimensions then do not flip
            # the trailing dimension.
            # --------------------------------------------------------
            ndim = bounds.ndim
            if ndim == 1:
                # Flip the bounds of a 0-d variable
                axes = (0,)
            elif ndim == 2:
                # Flip the bounds of a 1-d variable
                if axes in (0, 1):
                    axes = (0, 1)
                elif axes is not None:
                    axes = v._parse_axes(axes) + [-1]
            else:
                # Do not flip the bounds of an N-d variable (N >= 2)
                # nor a geometry variable
                axes = v._parse_axes(axes)

            bounds.flip(axes, inplace=True)

        return v

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def exp(self, bounds=True, inplace=False, i=False):
        '''The exponential of the data, element-wise.

    .. seealso:: `log`

    :Parameters:

        bounds: `bool`, optional
            If False then do not alter any bounds. By default any
            bounds are also altered.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'exp',
            bounds=bounds, inplace=inplace, i=i)

    def set_bounds(self, bounds, copy=True):
        '''Set the bounds.

    .. versionadded:: 3.0.0

    .. seealso: `del_bounds`, `get_bounds`, `has_bounds`, `set_data`

    :Parameters:

        bounds: `Bounds`
            The bounds to be inserted.

        copy: `bool`, optional
            If False then do not copy the bounds prior to
            insertion. By default the bounds are copied.

    :Returns:

        `None`

    **Examples:**

    >>> import numpy
    >>> b = cfdm.Bounds(data=cfdm.Data(numpy.arange(10).reshape(5, 2)))
    >>> c.set_bounds(b)
    >>> c.has_bounds()
    True
    >>> c.get_bounds()
    <Bounds: (5, 2) >
    >>> b = c.del_bounds()
    >>> b
    <Bounds: (5, 2) >
    >>> c.has_bounds()
    False
    >>> print(c.get_bounds(None))
    None
    >>> print(c.del_bounds(None))
    None

        '''
        data = self.get_data(None)

        if data is not None and bounds.shape[:data.ndim] != data.shape:
            # Check shape
            raise ValueError(
                "Can't set bounds: Incorrect shape: {0})".format(bounds.shape))

        if copy:
            bounds = bounds.copy()

        # Check units
        units = bounds.Units
        self_units = self.Units

        if data is not None and units and not units.equivalent(self_units):
            raise ValueError(
                "Can't set bounds: Bounds units of {!r} are not equivalent "
                "to {!r}".format(bounds.Units, self.Units)
            )

            bounds.Units = self_units

        if not units:
            bounds.override_units(self_units, inplace=True)

        # Copy selected properties to the bounds
        # for prop in ('standard_name', 'axis', 'positive',
        #              'leap_months', 'leap_years', 'month_lengths'):
        #     value = self.get_property(prop, None)
        #     if value is not None:
        #         bounds.set_property(prop, value)

        self._custom['direction'] = None

        super().set_bounds(bounds, copy=False)

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def sin(self, bounds=True, inplace=False, i=False):
        '''Take the trigonometric sine of the data element-wise.

    Units are accounted for in the calculation. For example, the
    sine of 90 degrees_east is 1.0, as is the sine of 1.57079632
    radians. If the units are not equivalent to radians (such as
    Kelvin) then they are treated as if they were radians.

    The Units are changed to '1' (nondimensional).

    .. seealso:: `arcsin`, `cos`, `tan`, `sinh`

    :Parameters:

        bounds: `bool`, optional
            If False then do not alter any bounds. By default any
            bounds are also altered.

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
    >>> g = f.sin()
    >>> g.Units
    <Units: 1>
    >>> print(g.array)
    [[-1.0 0.0 1.0 --]]

    >>> f.Units
    <Units: m s-1>
    >>> print(f.array)
    [[1 2 3 --]]
    >>> f.sin(inplace=True)
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[0.841470984808 0.909297426826 0.14112000806 --]]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'sin',
            bounds=bounds, inplace=inplace, i=i)

    # `arctan2`, AT2 seealso
    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def arctan(self, bounds=True, inplace=False):
        '''Take the trigonometric inverse tangent of the data element-wise.

    Units are ignored in the calculation. The result has units of radians.

    The "standard_name" and "long_name" properties are removed from
    the result.

    .. versionadded:: 3.0.7

    .. seealso:: `tan`

    :Parameters:

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

    :Returns:

            The construct with the trigonometric inverse tangent of data
            values. If the operation was in-place then `None` is returned.

    **Examples:**

    >>> print(f.array)
    [[0.5 0.7]
     [0.9 1.1]]
    >>> g = f.arctan()
    >>> g.Units
    <Units: radians>
    >>> print(g.array)
    [[0.46364761 0.61072596]
     [0.7328151  0.83298127]]

    >>> print(f.array)
    [1.2 1.0 0.8 0.6 --]
    >>> f.arctan(inplace=True)
    >>> print(f.array)
    [0.8760580505981934 0.7853981633974483 0.6747409422235527
     0.5404195002705842 --]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'arctan',
            inplace=inplace)

    @_inplace_enabled
    def arctanh(self, bounds=True, inplace=False):
        '''Take the inverse hyperbolic tangent of the data element-wise.

    Units are ignored in the calculation. The result has units of radians.

    The "standard_name" and "long_name" properties are removed from
    the result.

    .. versionadded:: 3.2.0

    .. seealso:: `tanh`, `arcsinh`, `arccosh`, `arctan`

    :Parameters:

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

    :Returns:

            The construct with the inverse hyperbolic tangent of data
            values. If the operation was in-place then `None` is returned.

    **Examples:**

    >>> print(f.array)
    [[0.5 0.7]
     [0.9 1.1]]
    >>> g = f.arctanh()
    >>> g.Units
    <Units: radians>
    >>> print(g.array)
    [[0.54930614 0.86730053]
     [1.47221949        nan]]

    >>> print(f.array)
    [1.2 1.0 0.8 0.6 --]
    >>> f.arctanh(inplace=True)
    >>> print(f.array)
    [nan inf 1.0986122886681098 0.6931471805599453 --]
    >>> f.mask_invalid(inplace=True)
    >>> print(f.array)
    [-- -- 1.0986122886681098 0.6931471805599453 --]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'arctanh',
            bounds=bounds, inplace=inplace)

    @_inplace_enabled
    def arcsin(self, bounds=True, inplace=False):
        '''Take the trigonometric inverse sine of the data element-wise.

    Units are ignored in the calculation. The result has units of radians.

    The "standard_name" and "long_name" properties are removed from
    the result.

    .. versionadded:: 3.2.0

    .. seealso:: `sin`, `arccos`, `arctan`, `arcsinh`

    :Parameters:

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

    :Returns:

            The construct with the trigonometric inverse sine of data
            values. If the operation was in-place then `None` is returned.

    **Examples:**

    >>> print(f.array)
    [[0.5 0.7]
     [0.9 1.1]]
    >>> g = f.arcsin()
    >>> g.Units
    <Units: radians>
    >>> print(g.array)
    [[0.52359878 0.7753975 ]
     [1.11976951        nan]]

    >>> print(f.array)
    [1.2 1.0 0.8 0.6 --]
    >>> f.arcsin(inplace=True)
    >>> print(f.array)
    [nan 1.5707963267948966 0.9272952180016123 0.6435011087932844 --]
    >>> f.mask_invalid(inplace=True)
    >>> print(f.array)
    [-- 1.5707963267948966 0.9272952180016123 0.6435011087932844 --]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'arcsin',
            bounds=bounds, inplace=inplace)

    @_inplace_enabled
    def arcsinh(self, bounds=True, inplace=False):
        '''Take the inverse hyperbolic sine of the data element-wise.

    Units are ignored in the calculation. The result has units of radians.

    The "standard_name" and "long_name" properties are removed from
    the result.

    .. versionadded:: 3.1.0

    .. seealso:: `sinh`

    :Parameters:

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

    :Returns:

            The construct with the inverse hyperbolic sine of data values.
            If the operation was in-place then `None` is returned.

    **Examples:**

    >>> print(f.array)
    [[0.5 0.7]
     [0.9 1.1]]
    >>> g = f.arcsinh()
    >>> g.Units
    <Units: radians>
    >>> print(g.array)
    [[0.48121183 0.65266657]
     [0.80886694 0.95034693]]

    >>> print(f.array)
    [1.2 1.0 0.8 0.6 --]
    >>> f.arcsinh(inplace=True)
    >>> print(f.array)
    [1.015973134179692 0.881373587019543 0.732668256045411 0.5688248987322475
     --]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'arcsinh',
            bounds=bounds, inplace=inplace)

    @_inplace_enabled
    def arccos(self, bounds=True, inplace=False):
        '''Take the trigonometric inverse cosine of the data element-wise.

    Units are ignored in the calculation. The result has units of radians.

    The "standard_name" and "long_name" properties are removed from
    the result.

    .. versionadded:: 3.2.0

    .. seealso:: `cos`, `arcsin`, `arctan`, `arccosh`

    :Parameters:

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

    :Returns:

            The construct with the trigonometric inverse cosine of data
            values. If the operation was in-place then `None` is returned.

    **Examples:**

    >>> print(f.array)
    [[0.5 0.7]
     [0.9 1.1]]
    >>> g = f.arccos()
    >>> g.Units
    <Units: radians>
    >>> print(g.array)
    [[1.04719755 0.79539883]
     [0.45102681        nan]]

    >>> print(f.array)
    [1.2 1.0 0.8 0.6 --]
    >>> f.arccos(inplace=True)
    >>> print(f.array)
    [nan 0.0 0.6435011087932843 0.9272952180016123 --]
    >>> f.mask_invalid(inplace=True)
    >>> print(f.array)
    [-- 0.0 0.6435011087932843 0.9272952180016123 --]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'arccos',
            bounds=bounds, inplace=inplace)

    @_inplace_enabled
    def arccosh(self, bounds=True, inplace=False):
        '''Take the inverse hyperbolic cosine of the data element-wise.

    Units are ignored in the calculation. The result has units of radians.

    The "standard_name" and "long_name" properties are removed from
    the result.

    .. versionadded:: 3.2.0

    .. seealso:: `cosh`, `arcsinh`, `arctanh`, `arccos`

    :Parameters:

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

    :Returns:

            The construct with the inverse hyperbolic cosine of data
            values. If the operation was in-place then `None` is returned.

    **Examples:**

    >>> print(f.array)
    [[0.5 0.7]
     [0.9 1.1]]
    >>> g = f.arccosh()
    >>> g.Units
    <Units: radians>
    >>> print(g.array)
    [[       nan        nan]
     [       nan 0.44356825]]

    >>> print(f.array)
    [1.2 1.0 0.8 0.6 --]
    >>> f.arccosh(inplace=True)
    >>> print(f.array)
    [0.6223625037147786 0.0 nan nan --]
    >>> f.mask_invalid(inplace=True)
    >>> print(f.array)
    [0.6223625037147786 0.0 -- -- --]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'arccosh',
            bounds=bounds, inplace=inplace)

    @_inplace_enabled
    def tanh(self, bounds=True, inplace=False):
        '''Take the hyperbolic tangent of the data element-wise.

    Units are accounted for in the calculation. If the units are not
    equivalent to radians (such as Kelvin) then they are treated as if
    they were radians. For example, the the hyperbolic tangent of 90
    degrees_east is 0.91715234, as is the hyperbolic tangent of
    1.57079632 radians.

    The output units are changed to '1' (nondimensional).

    The "standard_name" and "long_name" properties are removed from
    the result.

    .. versionadded:: 3.1.0

    .. seealso:: `arctanh`, `sinh`, `cosh`, `tan`


    :Parameters:

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

    :Returns:

            The construct with the hyperbolic tangent of data values. If the
            operation was in-place then `None` is returned.

    **Examples:**

    >>> f.Units
    <Units: degrees_north>
    >>> print(f.array)
    [[-90 0 90 --]]
    >>> g = f.tanh()
    >>> g.Units
    <Units: 1>
    >>> print(g.array)
    [[-0.9171523356672744 0.0 0.9171523356672744 --]]

    >>> f.Units
    <Units: m s-1>
    >>> print(f.array)
    [[1 2 3 --]]
    >>> f.tanh(inplace=True)
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[0.7615941559557649 0.9640275800758169 0.9950547536867305 --]]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'tanh',
            bounds=bounds, inplace=inplace)

    @_inplace_enabled
    def sinh(self, bounds=True, inplace=False):
        '''Take the hyperbolic sine of the data element-wise.

    Units are accounted for in the calculation. If the units are not
    equivalent to radians (such as Kelvin) then they are treated as if
    they were radians. For example, the the hyperbolic sine of 90
    degrees_north is 2.30129890, as is the hyperbolic sine of
    1.57079632 radians.

    The output units are changed to '1' (nondimensional).

    The "standard_name" and "long_name" properties are removed from
    the result.

    .. versionadded:: 3.1.0

    .. seealso:: `arcsinh`, `cosh`, `tanh`, `sin`

    :Parameters:

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

    :Returns:

            The construct with the hyperbolic sine of data values. If the
            operation was in-place then `None` is returned.

    **Examples:**

    >>> f.Units
    <Units: degrees_north>
    >>> print(f.array)
    [[-90 0 90 --]]
    >>> g = f.sinh(inplace=True)
    >>> g.Units
    <Units: 1>
    >>> print(g.array)
    [[-2.3012989023072947 0.0 2.3012989023072947 --]]

    >>> f.Units
    <Units: m s-1>
    >>> print(f.array)
    [[1 2 3 --]]
    >>> f.sinh(inplace=True)
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[1.1752011936438014 3.626860407847019 10.017874927409903 --]]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'sinh',
            bounds=bounds, inplace=inplace)

    @_inplace_enabled
    def cosh(self, bounds=True, inplace=False):
        '''Take the hyperbolic cosine of the data element-wise.

    Units are accounted for in the calculation. If the units are not
    equivalent to radians (such as Kelvin) then they are treated as if
    they were radians. For example, the the hyperbolic cosine of 0
    degrees_east is 1.0, as is the hyperbolic cosine of 1.57079632 radians.

    The output units are changed to '1' (nondimensional).

    The "standard_name" and "long_name" properties are removed from
    the result.

    .. versionadded:: 3.1.0

    .. seealso:: `arccosh`, `sinh`, `tanh`, `cos`

    :Parameters:

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

    :Returns:

            The construct with the hyperbolic cosine of data values. If the
            operation was in-place then `None` is returned.

    **Examples:**

    >>> f.Units
    <Units: degrees_north>
    >>> print(f.array)
    [[-90 0 90 --]]
    >>> g = f.cosh()
    >>> g.Units
    <Units: 1>
    >>> print(g.array)
    [[2.5091784786580567 1.0 2.5091784786580567 --]]

    >>> f.Units
    <Units: m s-1>
    >>> print(f.array)
    [[1 2 3 --]]
    >>> f.cosh(inplace=True)
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[1.5430806348152437 3.7621956910836314 10.067661995777765 --]]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'cosh',
            bounds=bounds, inplace=inplace)

    # `arctan2`, AT2 seealso
    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def tan(self, bounds=True, inplace=False, i=False):
        '''Take the trigonometric tangent of the data element-wise.

    Units are accounted for in the calculation, so that the
    tangent of 180 degrees_east is 0.0, as is the tangent of
    3.141592653589793 radians. If the units are not equivalent to
    radians (such as Kelvin) then they are treated as if they were
    radians.

    The Units are changed to '1' (nondimensional).

    .. seealso:: `arctan`, `cos`, `sin`, `tanh`

    :Parameters:

        bounds: `bool`, optional
            If False then do not alter any bounds. By default any
            bounds are also altered.

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

    :Returns:

            The construct with the tangent of data values. If the
            operation was in-place then `None` is returned.

    **Examples:**

    >>> f.Units
    <Units: degrees_north>
    >>> print(f.array)
    [[-45 0 45 --]]
    >>> g = f.tan()
    >>> g.Units
    <Units: 1>
    >>> print(g.array)
    [[-1.0 0.0 1.0 --]]

    >>> f.Units
    <Units: m s-1>
    >>> print(f.array)
    [[1 2 3 --]]
    >>> f.tan(inplace=True)
    >>> f.Units
    <Units: 1>
    >>> print(f.array)
    [[1.55740772465 -2.18503986326 -0.142546543074 --]]

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'tan',
            inplace=inplace, i=i)

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def log(self, base=None, bounds=True, inplace=False, i=False):
        '''The logarithm of the data array.

    By default the natural logarithm is taken, but any base may be
    specified.

    .. seealso:: `exp`

    :Parameters:

        base: number, optional
            The base of the logiarthm. By default a natural logiarithm
            is taken.

        bounds: `bool`, optional
            If False then do not alter any bounds. By default any
            bounds are also altered.

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.

    :Returns:

            The construct with the logarithm of data values. If the
            operation was in-place then `None` is returned.

    **Examples:**

    >>> f.data
    <CF Data(1, 2): [[1, 2]]>
    >>> f.log().data
    <CF Data: [[0.0, 0.69314718056]] ln(re 1)>

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
        # TODO: 'base' kwarg not used? why?
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'log',
            bounds=bounds, inplace=inplace, i=i)

    @_deprecated_kwarg_check('i')
    def squeeze(self, axes=None, inplace=False, i=False):
        '''Remove size one axes from the data array.

    By default all size one axes are removed, but particular size one
    axes may be selected for removal. Corresponding axes are also
    removed from the bounds data array, if present.

    .. seealso:: `flip`, `insert_dimension`, `transpose`

    :Parameters:

        axes: (sequence of) `int`
            The positions of the size one axes to be removed. By
            default all size one axes are removed. Each axis is
            identified by its original integer position. Negative
            integers counting from the last position are allowed.

            *Parameter example:*
              ``axes=0``

            *Parameter example:*
              ``axes=-2``

            *Parameter example:*
              ``axes=[2, 0]``

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

        i: deprecated at version 3.0.0
            Use *inplace* parameter instead.

    :Returns:

            The new construct with removed data axes. If the operation
            was in-place then `None` is returned.

    **Examples:**

    >>> f.shape
    (1, 73, 1, 96)
    >>> f.squeeze().shape
    (73, 96)
    >>> f.squeeze(0).shape
    (73, 1, 96)
    >>> g = f.squeeze([-3, 2])
    >>> g.shape
    (73, 96)
    >>> f.bounds.shape
    (1, 73, 1, 96, 4)
    >>> g.shape
    (73, 96, 4)

        '''
        return super().squeeze(axes=axes, inplace=inplace)

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def trunc(self, bounds=True, inplace=False, i=False):
        '''Truncate the data, element-wise.

    The truncated value of the scalar ``x``, is the nearest integer
    ``i`` which is closer to zero than ``x`` is. I.e. the fractional
    part of the signed number ``x`` is discarded.

    .. versionadded:: 1.0

    .. seealso:: `ceil`, `floor`, `rint`

    :Parameters:

        bounds: `bool`, optional
            If False then do not alter any bounds. By default any
            bounds are also altered.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'trunc',
            bounds=bounds, inplace=inplace, i=i)

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

    The identities of the bounds, if present, are included (with the
    exception of the bounds netCDF variable name).

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

    >>> f.properties()
    {}
    >>> f.bounds.properties()
    {'axis': 'Z',
     'units': 'm'}
    >>> f.identities()
    ['axis=Z', 'units=m', 'ncvar%z']

        '''
        identities = super().identities()

        bounds = self.get_bounds(None)
        if bounds is not None:
            identities.extend([i for i in bounds.identities()
                               if i not in identities])
# TODO ncvar AND?

        return identities

    @_deprecated_kwarg_check('relaxed_identity')
    def identity(self, default='', strict=False, relaxed=False,
                 nc_only=False, relaxed_identity=None, _ctype=True):
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

    If no identity can be found on the construct then the identity is
    taken from the bounds, if present (with the exception of the
    bounds netCDF variable name).

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

    >>> f.properties()
    {}
    >>> f.bounds.properties()
    {'axis': 'Z',
     'units': 'm'}
    >>> f.identity()
    'axis=Z'

        '''
        identity = super().identity(default=None, strict=strict,
                                    relaxed=relaxed, nc_only=nc_only,
                                    _ctype=_ctype)

# TODO: when coord has no standard name but bounds do - that standard name needs to be picked up.

        if identity is not None:
            return identity

        bounds = self.get_bounds(None)
        if bounds is not None:
            out = bounds.identity(default=None, strict=strict,
                                  relaxed=relaxed, nc_only=nc_only)

            if out is not None and not out.startswith('ncvar%'):
                return out
        # --- End: if

        return default

    def inspect(self):
        '''Inspect the object for debugging.

    .. seealso:: `cf.inspect`

    :Returns:

        `None`

        '''
        print(cf_inspect(self))  # pragma: no cover

    def period(self, *value):
        '''Return or set the period for cyclic values.

    .. seeslso:: `cyclic`

    :Parameters:

        value: optional
            The period. The absolute value is used.  May be set to any
            numeric scalar object, including `numpy` and `Data`
            objects. The units of the radius are assumed to be the
            same as the data, unless specified by a `Data` object.

            If *value* is `None` then any existing period is removed
            from the construct.

    :Returns:

        `Data` or `None`
            The period prior to the change, or the current period if
            no *value* was specified. `None` is always returned if the
            period had not been set previously.

    **Examples:**

    >>> print(c.period())
    None
    >>> c.Units
    <Units: degrees_east>
    >>> print(c.period(360))
    None
    >>> c.period()
    <CF Data(): 360.0 'degrees_east'>
    >>> import math
    >>> c.period(cf.Data(2*math.pi, 'radians'))
    <CF Data(): 360.0 degrees_east>
    >>> c.period()
    <CF Data(): 6.28318530718 radians>
    >>> c.period(None)
    <CF Data:() 6.28318530718 radians>
    >>> print(c.period())
    None
    >>> print(c.period(-360))
    None
    >>> c.period()
    <CF Data(): 360.0 degrees_east>

        '''
        old = super().period(*value)

        old2 = None

        bounds = self.get_bounds(None)
        if bounds is not None:
            old2 = bounds.period(*value)

        if old is None and old2 is not None:
            return old2

        return old

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def rint(self, bounds=True, inplace=False, i=False):
        '''Round the data to the nearest integer, element-wise.

    .. versionadded:: 1.0

    .. seealso:: `ceil`, `floor`, `trunc`

    :Parameters:

        bounds: `bool`, optional
            If False then do not alter any bounds. By default any
            bounds are also altered.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'rint',
            bounds=bounds, inplace=inplace, i=i)

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def round(self, decimals=0, bounds=True, inplace=False, i=False):
        '''Round the data to the given number of decimals.

    Data elements are evenly rounded to the given number of decimals.

    .. note:: Values exactly halfway between rounded decimal values
              are rounded to the nearest even value. Thus 1.5 and 2.5
              round to 2.0, -0.5 and 0.5 round to 0.0, etc. Results
              may also be surprising due to the inexact representation
              of decimal fractions in the IEEE floating point standard
              and errors introduced when scaling by powers of ten.

    .. versionadded:: 1.1.4

    .. seealso:: `ceil`, `floor`, `rint`, `trunc`

    :Parameters:

        decimals: `int`, optional
            Number of decimal places to round to (0 by default). If
            decimals is negative, it specifies the number of positions
            to the left of the decimal point.

        bounds: `bool`, optional
            If False then do not alter any bounds. By default any
            bounds are also altered.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'round',
            bounds=bounds, inplace=inplace, i=i, decimals=decimals)

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def roll(self, iaxis, shift, inplace=False, i=False):
        '''Roll the data along an axis.

    .. seealso:: `insert_dimension`, `flip`, `squeeze`, `transpose`

    :Parameters:

        iaxis: `int`
            TODO

        inplace: `bool`, optional
            If True then do the operation in-place and return `None`.

        i: deprecated at version 3.0.0
            Use *inplace* parmaeter instead.

    :Returns:

    TODO

    **Examples:**

    TODO

        '''
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), 'roll', iaxis,
            shift, interior_ring=True, inplace=inplace, i=i)

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    @property
    def hasbounds(self):
        '''Deprecated at version 3.0.0. Use method 'has_bounds' instead.

        '''
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, 'hasbounds',
            "Use method 'has_bounds' instead.")  # pragma: no cover

    def expand_dims(self, position=0, i=False):
        '''Insert a size 1 axis into the data array.

    Deprecated at version 3.0.0. Use method 'insert_dimension'
    instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'expand_dims',
            "Use method 'insert_dimension' instead.")  # pragma: no cover

    def files(self):
        '''Return the names of any files containing parts of the data array.

    Deprecated at version 3.4.0. Use method 'get_filenames' instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'expand_dims',
            "Use method 'get_filenames' instead.",
            version='3.4.0'
        )  # pragma: no cover

# --- End: class
