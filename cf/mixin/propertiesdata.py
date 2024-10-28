import logging
from itertools import chain
from os import sep

import numpy as np
from cfdm import is_log_level_info

from ..cfdatetime import dt
from ..data import Data
from ..data.data import _DEFAULT_CHUNKS
from ..decorators import (
    _deprecated_kwarg_check,
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
    _manage_log_level_via_verbosity,
)
from ..functions import (
    _DEPRECATION_ERROR_ATTRIBUTE,
    _DEPRECATION_ERROR_KWARGS,
    _DEPRECATION_ERROR_METHOD,
    abspath,
    default_netCDF_fillvals,
)
from ..functions import equivalent as cf_equivalent
from ..functions import inspect as cf_inspect
from ..units import Units
from . import Properties

_units_None = Units()

_month_units = ("month", "months")
_year_units = ("year", "years", "yr")

_relational_methods = (
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
)

logger = logging.getLogger(__name__)


class DeprecationError(Exception):
    pass


class PropertiesData(Properties):
    """Mixin class for a data array with metadata."""

    _special_properties = ("units", "calendar")

    def __array__(self, *dtype):
        """Returns a numpy array representation of the data."""
        data = self.get_data(None)
        if data is not None:
            return data.__array__(*dtype)

        raise ValueError(f"{self.__class__.__name__} has no data")

    def __contains__(self, value):
        """Called to implement membership test operators.

        x.__contains__(y) <==> y in x

        """
        data = self.get_data(None, _fill_value=None)
        if data is None:
            return False

        return value in data

    def __data__(self):
        """Returns a new reference to the data.

        Allows the construct to initialise a `Data` object.

        :Returns:

            `Data`

        **Examples**

        >>> f.data
        <CF Data(12): [14, ..., 56] km)
        >>> cf.Data(f)
        <CF Data(12): [14, ..., 56] km)
        >>> cf.Data.asdata(f)
        <CF Data(12): [14, ..., 56] km)

        """
        data = self.get_data(None)
        if data is not None:
            return data

        raise ValueError(f"{self.__class__.__name__} has no data")

    def __setitem__(self, indices, value):
        """Called to implement assignment to x[indices]

        x.__setitem__(indices, value) <==> x[indices]

        """
        data = self.get_data(None, _fill_value=None)
        if data is None:
            raise ValueError("Can't set elements when there is no data")

        try:
            value = value.get_data(_fill_value=None)
        except AttributeError:
            pass

        data[indices] = value

    def __add__(self, y):
        """The binary arithmetic operation ``+``

        x.__add__(y) <==> x+y

        """
        return self._binary_operation(y, "__add__")

    def __iadd__(self, y):
        """The augmented arithmetic assignment ``+=``

        x.__iadd__(y) <==> x+=y

        """
        return self._binary_operation(y, "__iadd__")

    def __radd__(self, y):
        """The binary arithmetic operation ``+`` with reflected
        operands.

        x.__radd__(y) <==> y+x

        """
        return self._binary_operation(y, "__radd__")

    def __sub__(self, y):
        """The binary arithmetic operation ``-``

        x.__sub__(y) <==> x-y

        """
        return self._binary_operation(y, "__sub__")

    def __isub__(self, y):
        """The augmented arithmetic assignment ``-=``

        x.__isub__(y) <==> x-=y

        """
        return self._binary_operation(y, "__isub__")

    def __rsub__(self, y):
        """The binary arithmetic operation ``-`` with reflected
        operands.

        x.__rsub__(y) <==> y-x

        """
        return self._binary_operation(y, "__rsub__")

    def __mul__(self, y):
        """The binary arithmetic operation ``*``

        x.__mul__(y) <==> x*y

        """
        return self._binary_operation(y, "__mul__")

    def __imul__(self, y):
        """The augmented arithmetic assignment ``*=``

        x.__imul__(y) <==> x*=y

        """
        return self._binary_operation(y, "__imul__")

    def __rmul__(self, y):
        """The binary arithmetic operation ``*`` with reflected
        operands.

        x.__rmul__(y) <==> y*x

        """
        return self._binary_operation(y, "__rmul__")

    def __div__(self, y):
        """The binary arithmetic operation ``/``

        x.__div__(y) <==> x/y

        """
        return self._binary_operation(y, "__div__")

    def __idiv__(self, y):
        """The augmented arithmetic assignment ``/=``

        x.__idiv__(y) <==> x/=y

        """
        return self._binary_operation(y, "__idiv__")

    def __rdiv__(self, y):
        """The binary arithmetic operation ``/`` with reflected
        operands.

        x.__rdiv__(y) <==> y/x

        """
        return self._binary_operation(y, "__rdiv__")

    def __floordiv__(self, y):
        """The binary arithmetic operation ``//``

        x.__floordiv__(y) <==> x//y

        """
        return self._binary_operation(y, "__floordiv__")

    def __ifloordiv__(self, y):
        """The augmented arithmetic assignment ``//=``

        x.__ifloordiv__(y) <==> x//=y

        """
        return self._binary_operation(y, "__ifloordiv__")

    def __rfloordiv__(self, y):
        """The binary arithmetic operation ``//`` with reflected
        operands.

        x.__rfloordiv__(y) <==> y//x

        """
        return self._binary_operation(y, "__rfloordiv__")

    def __truediv__(self, y):
        """The binary arithmetic operation ``/`` (true division)

        x.__truediv__(y) <==> x/y

        """
        return self._binary_operation(y, "__truediv__")

    def __itruediv__(self, y):
        """The augmented arithmetic assignment ``/=`` (true division)

        x.__itruediv__(y) <==> x/=y

        """
        return self._binary_operation(y, "__itruediv__")

    def __rtruediv__(self, y):
        """The binary arithmetic operation ``/`` (true division) with
        reflected operands.

        x.__rtruediv__(y) <==> y/x

        """
        return self._binary_operation(y, "__rtruediv__")

    def __pow__(self, y, modulo=None):
        """The binary arithmetic operations ``**`` and ``pow``

        x.__pow__(y) <==> x**y

        """
        if modulo is not None:
            raise NotImplementedError(
                "3-argument power not supported for %r"
                % self.__class__.__name__
            )

        return self._binary_operation(y, "__pow__")

    def __ipow__(self, y, modulo=None):
        """The augmented arithmetic assignment ``**=``

        x.__ipow__(y) <==> x**=y

        """
        if modulo is not None:
            raise NotImplementedError(
                "3-argument power not supported for %r"
                % self.__class__.__name__
            )

        return self._binary_operation(y, "__ipow__")

    def __rpow__(self, y, modulo=None):
        """The binary arithmetic operations ``**`` and ``pow`` with
        reflected operands.

        x.__rpow__(y) <==> y**x

        """
        if modulo is not None:
            raise NotImplementedError(
                "3-argument power not supported for %r"
                % self.__class__.__name__
            )

        return self._binary_operation(y, "__rpow__")

    def __mod__(self, y):
        """The binary arithmetic operation ``%``

        x.__mod__(y) <==> x % y

        .. versionadded:: 1.0

        """
        return self._binary_operation(y, "__mod__")

    def __imod__(self, y):
        """The binary arithmetic operation ``%=``

        x.__imod__(y) <==> x %= y

        .. versionadded:: 1.0

        """
        return self._binary_operation(y, "__imod__")

    def __rmod__(self, y):
        """The binary arithmetic operation ``%`` with reflected
        operands.

        x.__rmod__(y) <==> y % x

        .. versionadded:: 1.0

        """
        return self._binary_operation(y, "__rmod__")

    def __eq__(self, y):
        """The rich comparison operator ``==``

        x.__eq__(y) <==> x==y

        """
        return self._binary_operation(y, "__eq__")

    def __ne__(self, y):
        """The rich comparison operator ``!=``

        x.__ne__(y) <==> x!=y

        """
        return self._binary_operation(y, "__ne__")

    def __ge__(self, y):
        """The rich comparison operator ``>=``

        x.__ge__(y) <==> x>=y

        """
        return self._binary_operation(y, "__ge__")

    def __gt__(self, y):
        """The rich comparison operator ``>``

        x.__gt__(y) <==> x>y

        """
        return self._binary_operation(y, "__gt__")

    def __le__(self, y):
        """The rich comparison operator ``<=``

        x.__le__(y) <==> x<=y

        """
        return self._binary_operation(y, "__le__")

    def __lt__(self, y):
        """The rich comparison operator ``<``

        x.__lt__(y) <==> x<y

        """
        return self._binary_operation(y, "__lt__")

    def __and__(self, y):
        """The binary bitwise operation ``&``

        x.__and__(y) <==> x&y

        """
        return self._binary_operation(y, "__and__")

    def __iand__(self, y):
        """The augmented bitwise assignment ``&=``

        x.__iand__(y) <==> x&=y

        """
        return self._binary_operation(y, "__iand__")

    def __rand__(self, y):
        """The binary bitwise operation ``&`` with reflected operands.

        x.__rand__(y) <==> y&x

        """
        return self._binary_operation(y, "__rand__")

    def __or__(self, y):
        """The binary bitwise operation ``|``

        x.__or__(y) <==> x|y

        """
        return self._binary_operation(y, "__or__")

    def __ior__(self, y):
        """The augmented bitwise assignment ``|=``

        x.__ior__(y) <==> x|=y

        """
        return self._binary_operation(y, "__ior__")

    def __ror__(self, y):
        """The binary bitwise operation ``|`` with reflected operands.

        x.__ror__(y) <==> y|x

        """
        return self._binary_operation(y, "__ror__")

    def __xor__(self, y):
        """The binary bitwise operation ``^``

        x.__xor__(y) <==> x^y

        """
        return self._binary_operation(y, "__xor__")

    def __ixor__(self, y):
        """The augmented bitwise assignment ``^=``

        x.__ixor__(y) <==> x^=y

        """
        return self._binary_operation(y, "__ixor__")

    def __rxor__(self, y):
        """The binary bitwise operation ``^`` with reflected operands.

        x.__rxor__(y) <==> y^x

        """
        return self._binary_operation(y, "__rxor__")

    def __lshift__(self, y):
        """The binary bitwise operation ``<<``

        x.__lshift__(y) <==> x<<y

        """
        return self._binary_operation(y, "__lshift__")

    def __ilshift__(self, y):
        """The augmented bitwise assignment ``<<=``

        x.__ilshift__(y) <==> x<<=y

        """
        return self._binary_operation(y, "__ilshift__")

    def __rlshift__(self, y):
        """The binary bitwise operation ``<<`` with reflected operands.

        x.__rlshift__(y) <==> y<<x

        """
        return self._binary_operation(y, "__rlshift__")

    def __rshift__(self, y):
        """The binary bitwise operation ``>>``

        x.__lshift__(y) <==> x>>y

        """
        return self._binary_operation(y, "__rshift__")

    def __irshift__(self, y):
        """The augmented bitwise assignment ``>>=``

        x.__irshift__(y) <==> x>>=y

        """
        return self._binary_operation(y, "__irshift__")

    def __rrshift__(self, y):
        """The binary bitwise operation ``>>`` with reflected operands.

        x.__rrshift__(y) <==> y>>x

        """
        return self._binary_operation(y, "__rrshift__")

    def __abs__(self):
        """The unary arithmetic operation ``abs``

        x.__abs__() <==> abs(x)

        """
        return self._unary_operation("__abs__")

    def __neg__(self):
        """The unary arithmetic operation ``-``

        x.__neg__() <==> -x

        """
        return self._unary_operation("__neg__")

    def __invert__(self):
        """The unary bitwise operation ``~``

        x.__invert__() <==> ~x

        """
        return self._unary_operation("__invert__")

    def __pos__(self):
        """The unary arithmetic operation ``+``

        x.__pos__() <==> +x

        """
        return self._unary_operation("__pos__")

    def __query_isclose__(self, value, rtol, atol):
        """Query interface method for an "is close" condition.

        :Parameters:

            value:
                The object to test against.

            rtol: number
                The tolerance on relative numerical differences.

            atol: number
                The tolerance on absolute numerical differences.

        .. versionadded:: 3.15.2

        """
        data = self.get_data(None, _fill_value=None)
        if data is None:
            raise ValueError(
                "Can't apply '__query_isclose__' to a "
                f"{self.__class__.__name__} object with no data: {self!r}"
            )

        return data.isclose(value, rtol=rtol, atol=atol)

    def _binary_operation(self, y, method):
        """Implement binary arithmetic and comparison operations.

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

        **Examples**

        >>> w = u._binary_operation(u, '__add__')
        >>> w = u._binary_operation(v, '__lt__')
        >>> u._binary_operation(2, '__imul__')
        >>> u._binary_operation(v, '__idiv__')

        """
        if getattr(y, "_NotImplemented_RHS_Data_op", False):
            return NotImplemented

        data = self.get_data(None, _fill_value=None)
        if data is None:
            raise ValueError(
                f"Can't apply {method} to a {self.__class__.__name__} "
                f"object with no data: {self!r}"
            )

        inplace = method[2] == "i"

        units = self.Units
        sn = self.get_property("standard_name", None)
        ln = self.get_property("long_name", None)

        try:
            other_sn = y.get_property("standard_name", None)
            other_ln = y.get_property("long_name", None)
        except AttributeError:
            other_sn = None
            other_ln = None

        if isinstance(y, self.__class__):
            y = y.data
        elif y is None:
            y = Data(np.array(None, dtype=object))

        if not inplace:
            new = self.copy()  # data=False) TODO
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
                new.del_property("standard_name", None)
                new.del_property("long_name", None)
            elif other_sn is not None:
                new.set_property("standard_name", other_sn, copy=False)
                if other_ln is None:
                    new.del_property("long_name", None)
                else:
                    new.set_property("long_name", other_ln, copy=False)
        elif ln is None and other_ln is not None:
            new.set_property("long_name", other_ln, copy=False)

        new_units = new.Units
        if (
            method in _relational_methods
            or not units.equivalent(new_units)
            and not (units.isreftime and new_units.isreftime)
        ):
            new.del_property("standard_name", None)
            new.del_property("long_name", None)

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
    #    :Parameters:
    #
    #        dim_name_map: `dict`
    #
    #    :Returns:
    #
    #        `None`
    #
    #    **Examples**
    #
    #    >>> f._change_axis_names({'0': 'dim1', '1': 'dim2'})
    #
    #        '''
    #        data = self.get_data(None)
    #        if data is not None:
    #            data.change_axis_names(dim_name_map)

    def _conform_for_assignment(self, other):
        """Conform *other* for assignment broadcasting across *self*."""
        return other

    @_manage_log_level_via_verbosity
    def _equivalent_data(self, other, atol=None, rtol=None, verbose=None):
        """True if data is equivalent to other data, units considered.

        Two real numbers ``x`` and ``y`` are considered equal if
        ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on absolute
        differences) and ``rtol`` (the tolerance on relative differences)
        are positive, typically very small numbers. See the *atol* and
        *rtol* parameters.

        :Parameters:

            transpose: `dict`, optional

            atol: `float`, optional
                The tolerance on absolute differences between real
                numbers. The default value is set by the `atol` function.

            rtol: `float`, optional
                The tolerance on relative differences between real
                numbers. The default value is set by the `rtol` function.

        :Returns:

            `bool`
                Whether or not the two variables have equivalent data arrays.

        """
        if self.has_data() != other.has_data():
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Only one construct "
                    f"has data: {self!r}, {other!r}"
                )

            return False

        if not self.has_data():
            return True

        data0 = self.get_data(_fill_value=False)
        data1 = other.get_data(_fill_value=False)

        if data0.shape != data1.shape:
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Data have different shapes: "
                    f"{data0.shape}, {data1.shape}"
                )

            return False

        if not data0.Units.equivalent(data1.Units):
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Data have non-equivalent "
                    f"units: {data0.Units!r}, {data1.Units!r}"
                )

            return False

        if not data0.allclose(data1, rtol=rtol, atol=atol):
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Data have non-equivalent "
                    f"values: {data0!r}, {data1!r}"
                )

            return False

        return True

    #    def _parse_axes(self, axes):
    #        '''TODO
    #
    #
    #    :Returns:
    #
    #        `list`
    #
    #        '''
    #        ndim = self.ndim
    #
    #        if axes is None:
    #            return list(range(ndim))
    #
    #        if isinstance(axes, int):
    #            axes = (axes,)
    #
    #        return [(i + ndim if i < 0 else i) for i in axes]

    #    def _parse_match(self, match):
    #        '''Called by `match`
    #
    #    :Parameters:
    #
    #        match:
    #            As for the *match* parameter of `match` method.
    #
    #    :Returns:
    #
    #        `list`
    #        '''
    #        if not match:
    #            return ()
    #
    #        if isinstance(match, (str, dict, Query)):
    #            match = (match,)
    #
    #        matches = []
    #        for m in match:
    #            if isinstance(m, str):
    #                if '=' in m:
    #                    # CF property (string-valued)
    #                    m = m.split('=')
    #                    matches.append({m[0]: '='.join(m[1:])})
    #                else:
    #                    # Identity (string-valued) or python attribute
    #                    # (string-valued) or axis type
    #                    matches.append({None: m})
    #
    #            elif isinstance(m, dict):
    #                # Dictionary
    #                matches.append(m)
    #
    #            else:
    #                # Identity (not string-valued, e.g. cf.Query).
    #                matches.append({None: m})
    #
    #        return matches

    def _unary_operation(self, method):
        """Implement unary arithmetic operations on the data array.

        :Parameters:

            method: `str`
                The unary arithmetic method name (such as "__abs__").

        :Returns:

            `{{class}}`
                A new construct, or the same construct if the operation
                was in-place.

        **Examples**

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

        """
        data = self.get_data(None, _fill_value=False)
        if data is None:
            raise ValueError(
                f"Can't apply {method} to a {self.__class__.__name__} "
                "with no data"
            )

        new = self.copy(data=False)

        new_data = data._unary_operation(method)
        new.set_data(new_data, copy=False)

        return new

    def _YMDhms(self, attr):
        """Return some datetime component of the data array elements."""
        data = self.get_data(None, _fill_value=False)
        if data is None:
            raise ValueError(
                f"ERROR: Can't get {attr}s when there is no data array"
            )

        out = self.copy()  # data=False)

        out.set_data(getattr(data, attr), copy=False)

        out.del_property("standard_name", None)
        out.set_property("long_name", attr, copy=False)

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
    #
    #        raise ValueError(
    #            "ERROR: Can't get {0} when there is no data array".format(method))

    def _apply_data_oper(
        self, v, oper_name, oper_args=(), delete_props=False, **oper_kwargs
    ):
        """Define a data array operation and delete some properties.

        :Parameters:

            v: the data array to apply the operations to (possibly in-place)

            oper_name: the string name for the desired operation, as it is
                defined (its method name) under the Data class, e.g.
                `sin` to apply `Data.sin`.

                Note: there is no (easy) way to determine the name of a
                function/method within itself, without e.g. inspecting the stack
                (see rejected PEP 3130), so even though functions are named
                identically to those call in Data (e.g. both `sin`) the same
                name must be typed and passed into this method in each case.

                TODO: is there a way to prevent/bypass the above?

            oper_args, oper_kwargs: all of the arguments for *oper_name*.

            delete_props: whether or not to delete name properties.

        """
        # For explicitness on a per-method basis, apply inplace decorator
        # to individual methods calling this method, rather than decorating
        # only this and devolving the logic for inplace operations.
        if not oper_kwargs.get("inplace"):
            # Default for getattr below (preventing duplicate inplace kwarg)
            oper_kwargs["inplace"] = True

        data = v.get_data(None)
        if data is not None:
            getattr(data, oper_name)(*oper_args, **oper_kwargs)

        if delete_props:
            v.del_property("standard_name", None)
            v.del_property("long_name", None)

        return v

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def T(self):
        """`True` if and only if the data are coordinates for a CF 'T'
        axis.

        CF 'T' axis coordinates are defined by having units of reference
        time

        .. seealso:: `X`, `Y`, `Z`

        **Examples**

        >>> c.T
        False

        """
        return False

    @property
    def X(self):
        """Always False.

        .. seealso:: `T`, `Y`, `Z`

        **Examples**

        >>> print(f.X)
        False

        """
        return False

    @property
    def Y(self):
        """Always False.

        .. seealso:: `T`, `X`, `Z`

        **Examples**

        >>> print(f.Y)
        False

        """
        return False

    @property
    def Z(self):
        """Always False.

        .. seealso:: `T`, `X`, `Y`

        **Examples**

        >>> print(f.Z)
        False

        """
        return False

    @property
    def binary_mask(self):
        """A binary (0 and 1) missing data mask of the data array.

        The binary mask's data comprises dimensionless 32-bit integers
        that are 0 where the data has missing values and 1 otherwise.

        **Examples**

        >>> print(f.mask.array)
        [[ True  False  True False]]
        >>> b = f.binary_mask()
        >>> print(b.array)
        [[0 1 0 1]]

        """
        out = type(self)()
        out.set_propoerty("long_name", "binary_mask", copy=False)
        out.set_data(self.data.binary_mask(), copy=False)
        return out

    @property
    def data(self):
        """The `Data` object containing the data array.

        * ``f.data = x`` is equivalent to ``f.set_data(x, copy=False)``

        * ``x = f.data`` is equivalent to ``x = f.get_data()``

        * ``del f.data`` is equivalent to ``f.del_data()``

        * ``hasattr(f, 'data')`` is equivalent to ``f.has_data()``

        .. seealso:: `del_data`, `get_data`, `has_data`, `set_data`

        """
        return self.get_data()

    @data.setter
    def data(self, value):
        self.set_data(value, set_axes=False, copy=False)

    @data.deleter
    def data(self):
        return self.del_data()

    @property
    def isperiodic(self):
        """True if a given axis is periodic.

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

        """
        return self.period() is not None

    @property
    def reference_datetime(self):
        """The reference date-time of units of elapsed time.

        **Examples**

        >>> f.units
        'days since 2000-1-1'
        >>> f.reference_datetime
        cftime.DatetimeNoLeap(2000-01-01 00:00:00)

        """
        units = self.Units
        if not units.isreftime:
            raise AttributeError(
                f"{self.__class__.__name__} doesn't have attribute "
                "'reference_datetime'"
            )
        return dt(units.reftime, calendar=units._calendar)

    @reference_datetime.setter
    def reference_datetime(self, value):
        units = self.Units
        if not units.isreftime:
            raise AttributeError(
                "Can't set 'reference_datetime' for non reference date-time "
                f"units {self.__class__.__name__}"
            )

        units = units.units.split(" since ")
        try:
            self.units = f"{units[0]} since {value}"
        except (ValueError, TypeError):
            raise ValueError(
                "Can't override reference date-time "
                f"{units[1]!r} with {value!r}"
            )

    @property
    def Units(self):
        """The `cf.Units` object containing the units of the data array.

        Stores the units and calendar CF properties in an internally
        consistent manner. These are mirrored by the `units` and
        `calendar` CF properties respectively.

        **Examples**

        >>> f.Units
        <Units: K>

        >>> f.Units
        <Units: days since 2014-1-1 calendar=noleap>

        """
        data = self.get_data(None, _fill_value=False)
        if data is not None:
            return data.Units

        try:
            return self._custom["Units"]
        except KeyError:
            self._custom["Units"] = _units_None

        return _units_None

    @Units.setter
    def Units(self, value):
        data = self.get_data(None, _fill_value=False)
        if data is not None:
            data.Units = value
        else:
            self._custom["Units"] = value

        # Set the Units on the period
        period = self.period()
        if period is not None:
            period = period.copy()
            period.Units = value
            self._custom["period"] = period

        self._custom["direction"] = None

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
            f"Can't delete {self.__class__.__name__} attribute "
            "'Units'. Use the override_units method."
        )

    @property
    def year(self):
        """The year of each date-time data array element.

        Only applicable to data arrays with reference time units.

        .. seealso:: `month`, `day`, `hour`, `minute`, `second`

        **Examples**

        >>> print(f.datetime_array)
        [1950-11-15 00:00:00  1950-12-16 12:30:00  1951-01-16 12:00:45]
        >>> print(f.year.array)
        [1950  1950  1951]

        """
        return self._YMDhms("year")

    @property
    def month(self):
        """The month of each date-time data array element.

        Only applicable to data arrays with reference time units.

        .. seealso:: `year`, `day`, `hour`, `minute`, `second`

        **Examples**

        >>> print(f.datetime_array)
        [0450-11-15 00:00:00  0450-12-16 12:30:00  0451-01-16 12:00:45]
        >>> print(f.month.array)
        [11  12  1]

        """
        return self._YMDhms("month")

    @property
    def day(self):
        """The day of each date-time data array element.

        Only applicable to data arrays with reference time units.

        .. seealso:: `year`, `month`, `hour`, `minute`, `second`

        **Examples**

        >>> print(f.datetime_array)
        [0450-11-15 00:00:00  0450-12-16 12:30:00  0451-01-16 12:00:45]
        >>> print(f.day.array)
        [15  16  16]

        """
        return self._YMDhms("day")

    @property
    def hour(self):
        """The hour of each date-time data array element.

        Only applicable to data arrays with reference time units.

        .. seealso:: `year`, `month`, `day`, `minute`, `second`

        **Examples**

        >>> print(f.datetime_array)
        [0450-11-15 00:00:00  0450-12-16 12:30:00  0451-01-16 12:00:45]
        >>> print(f.hour.array)
        [ 0  12  12]

        """
        return self._YMDhms("hour")

    @property
    def minute(self):
        """The minute of each date-time data array element.

        Only applicable to data arrays with reference time units.

        .. seealso:: `year`, `month`, `day`, `hour`, `second`

        **Examples**

        >>> print(f.datetime_array)
        [0450-11-15 00:00:00  0450-12-16 12:30:00  0451-01-16 12:00:45]
        >>> print(f.minute.array)
        [ 0 30  0]

        """
        return self._YMDhms("minute")

    @property
    def second(self):
        """The second of each date-time data array element.

        Only applicable to data arrays with reference time units.

        .. seealso:: `year`, `month`, `day`, `hour`, `minute`

        **Examples**

        >>> print(f.datetime_array)
        [0450-11-15 00:00:00  0450-12-16 12:30:00  0451-01-16 12:00:45]
        >>> print(f.second.array)
        [ 0  0 45]

        """
        return self._YMDhms("second")

    @property
    def mask(self):
        """The mask of the data array.

        Values of True indicate masked elements.

        .. seealso:: `binary_mask`

        **Examples**

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

        """
        if not self.has_data():
            raise ValueError(
                "ERROR: Can't get mask when there is no data array"
            )

        out = self.copy()

        out.set_data(self.data.mask, copy=False)

        out.override_units(Units(), inplace=True)

        out.clear_properties()
        out.set_property("long_name", "mask", copy=False)

        out.nc_del_variable(default=None)

        return out

    # ----------------------------------------------------------------
    # CF properties
    # ----------------------------------------------------------------
    @property
    def add_offset(self):
        """The add_offset CF property.

        If present then this number is *subtracted* from the data prior to
        it being written to a file. If both `scale_factor` and
        `add_offset` properties are present, the offset is subtracted
        before the data are scaled. See
        http://cfconventions.org/latest.html for details.

        **Examples**

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

        """
        return self.get_property("add_offset", default=AttributeError())

    @add_offset.setter
    def add_offset(self, value):
        self.set_property("add_offset", value)
        self.dtype = np.result_type(self.dtype, np.array(value).dtype)

    @add_offset.deleter
    def add_offset(self):
        self.delprop("add_offset", default=AttributeError())
        if not self.has_property("scale_factor"):
            del self.dtype

    @property
    def calendar(self):
        """The calendar CF property.

        The calendar used for encoding time data. See
        http://cfconventions.org/latest.html for details.

        **Examples**

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

        """
        try:
            return self.Units.calendar
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__} doesn't have CF property "
                "'calendar'"
            )

    #        value = getattr(self.Units, "calendar", None)
    #        if value is None:
    #            raise AttributeError(
    #                "{} doesn't have CF property 'calendar'".format(
    #                    self.__class__.__name__
    #                )
    #            )
    #        return value

    @calendar.setter
    def calendar(self, value):
        self.Units = Units(getattr(self, "units", None), value)

    @calendar.deleter
    def calendar(self):
        if getattr(self, "calendar", None) is None:
            raise AttributeError(
                f"Can't delete non-existent {self.__class__.__name__} "
                "CF property 'calendar'"
            )

        self.Units = Units(getattr(self, "units", None))

    @property
    def _FillValue(self):
        """The _FillValue CF property.

        A value used to represent missing or undefined data.

        Note that this property is primarily for writing data to disk and
        is independent of the missing data mask. It may, however, get used
        when unmasking data array elements. See
        http://cfconventions.org/latest.html for details.

        The recommended way of retrieving the missing data value is with
        the `fill_value` method.

        .. seealso:: `fill_value`, `missing_value`,
                     `cf.default_netCDF_fillvals`

        **Examples**

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

        """
        return self.get_property("_FillValue", default=AttributeError())

    @_FillValue.setter
    def _FillValue(self, value):
        self.set_property("_FillValue", value)

    @_FillValue.deleter
    def _FillValue(self):
        self.del_property("_FillValue", default=AttributeError())

    @property
    def missing_value(self):
        """The missing_value CF property.

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

        **Examples**

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

        """
        return self.get_property("missing_value", default=AttributeError())

    @missing_value.setter
    def missing_value(self, value):
        self.set_property("missing_value", value)

    @missing_value.deleter
    def missing_value(self):
        self.del_property("missing_value", default=AttributeError())

    @property
    def scale_factor(self):
        """The scale_factor CF property.

        If present then the data are *divided* by this factor prior to it
        being written to a file. If both `scale_factor` and `add_offset`
        properties are present, the offset is subtracted before the data
        are scaled. See http://cfconventions.org/latest.html for details.

        **Examples**

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

        """
        return self.get_property("scale_factor", default=AttributeError())

    @scale_factor.setter
    def scale_factor(self, value):
        self.set_property("scale_factor", value)

    @scale_factor.deleter
    def scale_factor(self):
        self.del_property("scale_factor", default=AttributeError())

    @property
    def units(self):
        """The units CF property.

        The units of the data. The value of the `units` property is a
        string that can be recognised by UNIDATA's Udunits package
        (http://www.unidata.ucar.edu/software/udunits). See
        http://cfconventions.org/latest.html for details.

        **Examples**

        >>> f.units = 'K'
        >>> f.units
        'K'
        >>> del f.units

        >>> f.set_property('units', 'm.s-1')
        >>> f.get_property('units')
        'm.s-1'
        >>> f.has_property('units')
        True

        """
        try:
            return self.Units.units
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__} doesn't have CF property 'units'"
            )

    #        value = getattr(self.Units, "units", None)
    #        if value is None:
    #            raise AttributeError(
    #                f"{self.__class__.__name__} doesn't have CF property 'units'"
    #            )
    #
    #        return value

    @units.setter
    def units(self, value):
        self.Units = Units(value, getattr(self, "calendar", None))

    @units.deleter
    def units(self):
        if getattr(self, "units", None) is None:
            raise AttributeError(
                f"Can't delete non-existent {self.__class__.__name__} "
                "CF property 'units'"
            )

        self.Units = Units(None, getattr(self, "calendar", None))

    def add_file_location(self, location):
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

        >>> d.add_file_location('/data/model/')
        '/data/model'

        """
        data = self.get_data(None, _fill_value=False, _units=False)
        if data is not None:
            return data.add_file_location(location)

        return abspath(location).rstrip(sep)

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def mask_invalid(self, inplace=False, i=False):
        """Mask the array where invalid values occur.

        Deprecated at version 3.14.0. Use the method
        `masked_invalid` instead.

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


            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                TODO

        **Examples**

        >>> print(f.array)
        [ 0.  1.]
        >>> print(g.array)
        [ 1.  2.]
        >>> old = cf.Data.seterr('ignore')
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

        >>> old = cf.Data.seterr('raise')
        >>> old = cf.Data.mask_fpe(True)
        >>> print((g/f).array)
        [ --  2]
        >>> print((g**12345).array)
        [1.  -- ]

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "mask_invalid",
            message="Use the method 'masked_invalid' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @_inplace_enabled(default=False)
    def masked_invalid(self, inplace=False):
        """Mask the array where invalid values occur (NaN or inf).

        .. seealso:: `numpy.ma.masked_invalid`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with masked values, or `None` if the
                operation was in-place.


        **Examples**

        >>> print(f.array)
        [0 1 2]
        >>> print(g.array)
        [0 2 0]
        >>> h = f / g
        >>> print(h.array)
        [nan 0.5 inf]
        >>> i = h.masked_invalid()
        >>> print(i.array)
        [-- 0.5 --]

        """
        v = _inplace_enabled_define_and_cleanup(self)

        data = v.get_data(None, _fill_value=False)
        if data is not None:
            data.masked_invalid(inplace=True)

        return v

    def maximum(self):
        """The maximum of the data array.

        .. seealso:: `mean`, `mid_range`, `minimum`, `range`,
                     `sample_size`, `standard_deviation`, `sum`,
                     `variance`

        :Returns:

            `Data`
                The maximum of the data array.

        **Examples**

        >>> f.data
        <CF Data(12, 64, 128): [[[236.512756, ..., 256.93371]]] K>
        >>> f.maximum()
        <CF Data(): 311.343780 K>

        """
        data = self.get_data(None)
        if data is not None:
            return data.maximum(squeeze=True)

        raise ValueError(
            "ERROR: Can't get the maximum when there is no data array"
        )

    def mean(self):
        """The unweighted mean the data array.

        .. seealso:: `maximum`, `mid_range`, `minimum`, `range`,
                     `sample_size`, `standard_deviation`, `sum`,
                     `variance`

        :Returns:

            `Data`
                The unweighted mean the data array.

        **Examples**

        >>> f.data
        <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
        >>> f.mean()
        <CF Data(): 280.192227593 K>

        """
        data = self.get_data(None)
        if data is not None:
            return data.mean(squeeze=True)

        raise ValueError(
            "ERROR: Can't get the mean when there is no data array"
        )

    def mid_range(self):
        """The unweighted average of the maximum and minimum of the data
        array.

        .. seealso:: `maximum`, `mean`, `minimum`, `range`, `sample_size`,
                     `standard_deviation`, `sum`, `variance`

        :Returns:

            `Data`
                The unweighted average of the maximum and minimum of the
                data array.

        **Examples**

        >>> f.data
        <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
        >>> f.mid_range()
        <CF Data(): 255.08618927 K>

        """
        data = self.get_data(None)
        if data is not None:
            return data.mid_range(squeeze=True)

        raise ValueError(
            "ERROR: Can't get the mid-range when there is no data array"
        )

    def minimum(self):
        """The minimum of the data array.

        .. seealso:: `maximum`, `mean`, `mid_range`, `range`,
                     `sample_size`, `standard_deviation`, `sum`,
                     `variance`

        :Returns:

            `Data`
                The minimum of the data array.

        **Examples**

        >>> f.data
        <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
        >>> f.minimum()
        <CF Data(): 198.828598022 K>

        """
        data = self.get_data(None)
        if data is not None:
            return data.minimum(squeeze=True)

        raise ValueError(
            "ERROR: Can't get the minimum when there is no data array"
        )

    @_inplace_enabled(default=False)
    def pad_missing(self, axis, pad_width=None, to_size=None, inplace=False):
        """Pad an axis with missing data.

        :Parameters:

            axis: `int`
                Select the axis for which the padding is to be
                applied.

            {{pad_width: sequence of `int`, optional}}

            {{to_size: `int`, optional}}

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The {{class}} with padded data, or `None` if the
                operation was in-place.

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "pad_missing",
            axis=axis,
            pad_width=pad_width,
            to_size=to_size,
            inplace=inplace,
        )

    def period(self, *value, **config):
        """Return or set the period of the data.

        This is distinct from the cyclicity of individual axes.

        .. seealso:: `cyclic`, `iscyclic`, `isperiodic`

        :Parameters:

            value: optional
                The period. The absolute value is used.  May be set to any
                numeric scalar object, including `numpy` and `Data`
                objects. The units of the radius are assumed to be the
                same as the data, unless specified by a `Data` object.

                If *value* is `None` then any existing period is removed
                from the construct.

            config:
                Additional parameters for optimising the
                operation. See the code for details.

                .. versionadded:: 3.9.0

        :Returns:

            `Data` or `None`
                The period prior to the change, or the current period if
                no *value* was specified. `None` is always returned if the
                period had not been set previously.

        **Examples**

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

        """
        custom = self._custom
        old = custom.get("period")
        if old is not None:
            old = old.copy()

        period = config.get("period")
        if period is not None:
            custom["period"] = period.copy()
            return old

        if not value:
            return old

        value = value[0]

        if value is not None:
            value = Data.asdata(value)
            value_units = value.Units
            units = self.Units
            if not value_units:
                value = value.override_units(units)
            elif value_units != units:
                if value_units.equivalent(units):
                    value.Units = units
                else:
                    raise ValueError(
                        f"Period units {value_units!r} are not "
                        f"equivalent to data units {units!r}"
                    )

            value = abs(value)
            value.dtype = float

        custom["period"] = value

        return old

    @_inplace_enabled(default=False)
    def persist(self, inplace=False):
        """Persist the underlying dask array into memory.

        This turns an underlying lazy dask array into a equivalent
        chunked dask array, but now with the results fully computed.

        `persist` is particularly useful when using distributed
        systems, because the results will be kept in distributed
        memory, rather than returned to the local process.

        **Performance**

        `persist` causes all delayed operations to be computed.

        .. versionadded:: 3.14.0

        .. seealso:: `array`, `datetime_array`,
                     `dask.array.Array.persist`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with persisted data. If the operation
                was in-place then `None` is returned.

        **Examples**

        >>> g = f.persist()

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "persist",
            inplace=inplace,
            delete_props=False,
        )

    def range(self):
        """The absolute difference between the maximum and minimum of
        the data array.

        .. seealso:: `maximum`, `mean`, `mid_range`, `minimum`,
                     `sample_size`, `standard_deviation`, `sum`,
                     `variance`

        :Returns:

            `Data`
                The absolute difference between the maximum and minimum of
                the data array.

        **Examples**

        >>> f.data
        <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
        >>> f.range()
        <CF Data(): 112.515182495 K>

        """
        data = self.get_data(None)
        if data is not None:
            return data.range(squeeze=True)

        raise ValueError(
            "ERROR: Can't get the range when there is no data array"
        )

    def sample_size(self):
        """The number of non-missing data elements in the data array.

        .. seealso:: `count`, `maximum`, `mean`, `mid_range`, `minimum`,
                     `range`, `standard_deviation`, `sum`, `variance`

        :Returns:

            `Data`
                The number of non-missing data elements in the data array.

        **Examples**

        >>> f.data
        <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
        >>> f.sample_size()
        <CF Data(): 98304.0>

        """
        data = self.get_data(None)
        if data is not None:
            return data.sample_size(squeeze=True)

        raise ValueError(
            "ERROR: Can't get the sample size when there is no data array"
        )

    def standard_deviation(self):
        """The unweighted sample standard deviation of the data array.

        .. seealso:: `maximum`, `mean`, `mid_range`, `minimum`, `range`,
                     `sample_size`, `sum`, `variance`

        :Returns:

            `Data`
                The unweighted standard deviation of the data array.

        **Examples**

        >>> f.data
        <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
        >>> f.standard_deviation()
        <CF Data(): 22.685052535 K>

        """
        data = self.get_data(None)
        if data is not None:
            return data.sd(squeeze=True, ddof=0)

        raise ValueError(
            "ERROR: Can't get the standard deviation when there is no data "
            "array"
        )

    def sd(self):
        """Alias for `standard_deviation`"""
        return self.standard_deviation()

    def sum(self):
        """The sum of the data array.

        .. seealso:: `maximum`, `mean`, `mid_range`, `minimum`, `range`,
                     `sample_size`, `standard_deviation`, `variance`

        :Returns:

            `Data`
                The sum of the data array.

        **Examples**

        >>> f.data
        <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
        >>> f.sum()
        <CF Data(): 27544016.7413 K>

        """
        data = self.get_data(None)
        if data is not None:
            return data.sum(squeeze=True)

        raise ValueError(
            "ERROR: Can't get the sum when there is no data array"
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def swapaxes(self, axis0, axis1, inplace=False):
        """Interchange two axes of an array.

        .. seealso:: `flatten`, `flip`, `insert_dimension`, `squeeze`,
                     `transpose`

        :Parameters:

            axis0, axis1: `int`, `int`
                Select the axes to swap. Each axis is identified by its
                original integer position.

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with data with swapped axis positions. If
                the operation was in-place then `None` is returned.

        **Examples**

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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "swapaxes",
            (axis0, axis1),
            inplace=inplace,
            # TODODASKAPI - why not delete_props=False ??
            delete_props=True,
        )

    def variance(self):
        """The unweighted sample variance of the data array.

        .. seealso:: `maximum`, `mean`, `mid_range`, `minimum`, `range`,
                     `sample_size`, `standard_deviation`, `sum`

        :Returns:

            `Data`
                The unweighted variance of the data array.

        **Examples**

        >>> f.data
        <CF Data(12, 73, 96): [[[236.512756348, ..., 256.93371582]]] K>
        >>> f.variance()
        <CF Data(): 514.611608515 K2>

        """
        data = self.get_data(None)
        if data is None:
            raise ValueError(
                "ERROR: Can't get the variance when there is no data array"
            )

        return data.var(squeeze=True, ddof=0)

    def var(self):
        """Alias for `variance`"""
        return self.variance()

    @property
    def subspace(self):
        """Return a new variable whose data is subspaced.

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

        **Examples**

        TODO

        """
        return Subspace(self)

    @property
    def datetime_array(self):
        """An independent numpy array of date-time objects.

        Only applicable for data with reference time units.

        If the calendar has not been set then the CF default calendar will
        be used and the units will be updated accordingly.

        .. seealso:: `array`, `varray`

        **Examples**

        >>> f.units
        'days since 2000-01-01'
        >>> print(f.array)
        [ 0 31 60 91]
        >>> print(f.datetime_array)
        [cftime.DatetimeGregorian(2000-01-01 00:00:00)
         cftime.DatetimeGregorian(2000-02-01 00:00:00)
         cftime.DatetimeGregorian(2000-03-01 00:00:00)
         cftime.DatetimeGregorian(2000-04-01 00:00:00)]

        """
        data = self.get_data(None)
        if data is None:
            raise AttributeError(
                f"{self.__class__.__name__} has no data array"
            )

        return data.datetime_array

    @property
    def dtype(self):
        """The `numpy` data type of the data array.

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

        **Examples**

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

        """
        data = self.get_data(None, _fill_value=False)
        if data is None:
            raise AttributeError(
                f"{self.__class__.__name__} doesn't have attribute 'dtype'"
            )

        return data.dtype

    @dtype.setter
    def dtype(self, value):
        # DCH - allow dtype to be set before data c.f.  Units
        data = self.get_data(None, _fill_value=False)
        if data is not None:
            data.dtype = value

    @dtype.deleter
    def dtype(self):
        data = self.get_data(None, _fill_value=False)
        if data is not None:
            del data.dtype

    @property
    def hardmask(self):
        """Whether the mask is hard (True) or soft (False).

        When the mask is hard, masked elements of the data array can not
        be unmasked by assignment, but unmasked elements may be still be
        masked.

        When the mask is soft, masked entries of the data array may be
        unmasked by assignment and unmasked entries may be masked.

        By default, the mask is hard.

        .. seealso:: `where`, `subspace`, `__setitem__`

        **Examples**

        >>> f.hardmask = False
        >>> f.hardmask
        False

        """
        data = self.get_data(None, _fill_value=False)
        if data is None:
            raise AttributeError(
                f"{self.__class__.__name__} doesn't have attribute 'hardmask'"
            )

        return data.hardmask

    @hardmask.setter
    def hardmask(self, value):
        data = self.get_data(None, _fill_value=False)
        if data is None:
            raise AttributeError(
                f"{self.__class__.__name__} doesn't have any data"
            )

        data.hardmask = value

    @hardmask.deleter
    def hardmask(self):
        raise AttributeError(
            f"Won't delete {self.__class__.__name__} attribute 'hardmask'"
        )

    @property
    def array(self):
        """A numpy array deep copy of the data.

        Changing the returned numpy array does not change the data array.

        .. seealso:: `data`, `datetime_array`, `dask_array`

        **Examples**

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

        """
        data = self.get_data(None)
        if data is None:
            raise AttributeError(f"{self.__class__.__name__} has no data")

        return data.array

    @property
    def varray(self):
        """A numpy array view of the data.

        Deprecated at version 3.14.0.

        Changing the elements of the returned view changes the data array.

        .. seealso:: `array`, `data`, `datetime_array`, `dask_array`

        **Examples**

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

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "varray",
            message="Data are now stored as `dask` arrays for which, "
            "in general, a numpy array view is not robust.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @property
    def isscalar(self):
        """True if the data array is scalar.

        .. seealso:: `has_data`, `ndim`

        **Examples**

        >>> f.ndim
        0
        >>> f.isscalar
        True

        >>> f.ndim >= 1
        True
        >>> f.isscalar
        False

        >>> f.has_data()
        False
        >>> f.isscalar
        False

        """
        data = self.get_data(None, _fill_value=False)
        if data is None:
            return False

        return not data.ndim

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def ceil(self, inplace=False, i=False):
        """The ceiling of the data, element-wise.

        The ceiling of ``x`` is the smallest integer ``n``, such that
        ``n>=x``.

        .. versionadded:: 1.0

        .. seealso:: `floor`, `rint`, `trunc`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with the ceiling of the data. If the
                operation was in-place then `None` is returned.

        **Examples**

        >>> print(f.array)
        [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
        >>> print(f.ceil().array)
        [-1. -1. -1. -1.  0.  1.  2.  2.  2.]
        >>> f.ceil(inplace=True)
        >>> print(f.array)
        [-1. -1. -1. -1.  0.  1.  2.  2.  2.]

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "ceil",
            inplace=inplace,
            i=i,
            delete_props=True,
        )

    def cfa_update_file_substitutions(self, substitutions):
        """Set CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Parameters:

            {{cfa substitutions: `dict`}}

        :Returns:

            `None`

        **Examples**

        >>> f.cfa_update_file_substitutions({'base', '/data/model'})

        """
        data = self.get_data(None, _fill_value=False, _units=False)
        if data is not None:
            data.cfa_update_file_substitutions(substitutions)

    @_inplace_enabled(default=False)
    def cfa_clear_file_substitutions(self, inplace=False):
        """Remove all of the CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `dict`
                {{Returns cfa_clear_file_substitutions}}

        **Examples**

        >>> f.cfa_clear_file_substitutions()
        {}

        """
        data = self.get_data(None)
        if data is None:
            return {}

        return data.cfa_clear_file_substitutions({})

    def cfa_del_file_substitution(
        self,
        base,
    ):
        """Remove a CFA-netCDF file name substitution.

        .. versionadded:: 3.15.0

        :Parameters:

            `dict`
                {{Returns cfa_del_file_substitution}}

        **Examples**

        >>> f.cfa_del_file_substitution('base')

        """
        data = self.get_data(None, _fill_value=False, _units=False)
        if data is not None:
            data.cfa_del_file_substitution(base)

    def cfa_file_substitutions(
        self,
    ):
        """Return the CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Returns:

            `dict`
                {{Returns cfa_file_substitutions}}

        **Examples**

        >>> g = f.cfa_file_substitutions()

        """
        data = self.get_data(None)
        if data is None:
            return {}

        return data.cfa_file_substitutions({})

    def chunk(self, chunksize=None):
        """Partition the data array.

        Deprecated at version 3.14.0. Use the `rechunk` method
        instead.

        :Parameters:

            chunksize: `int`

        :Returns:

            `None`

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "chunk",
            "Use the 'rechunk' method instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def clip(self, a_min, a_max, units=None, inplace=False, i=False):
        """Limit the values in the data.

        Given an interval, values outside the interval are clipped to
        the interval edges. For example, if an interval of ``[0, 1]``
        is specified, values smaller than 0 become 0, and values
        larger than 1 become 1.

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
                Specify the units of *a_min* and *a_max*. By default
                the same units as the data are assumed.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with clipped data. If the operation was
                in-place then `None` is returned.

        **Examples**

        >>> g = f.clip(-90, 90)
        >>> g = f.clip(-90, 90, 'degrees_north')

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "clip",
            (a_min, a_max),
            inplace=inplace,
            i=i,
            units=units,
        )

    def close(self):
        """Close all files referenced by the construct.

        Deprecated at version 3.14.0. All files are now
        automatically closed when not being accessed.

        Note that a closed file will be automatically reopened if its
        contents are subsequently required.

        .. seealso:: `files`

        :Returns:

            `None`

        **Examples**

        >>> f.close()

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "close",
            "All files are now automatically closed when not being accessed.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @classmethod
    def concatenate(
        cls,
        variables,
        axis=0,
        cull_graph=False,
        relaxed_units=False,
        copy=True,
    ):
        """Join a sequence of variables together.

        .. seealso:: `Data.cull_graph`

        :Parameters:

            variables: sequence of constructs.

            axis: `int`, optional

            {{cull_graph: `bool`, optional}}

                .. versionadded:: 3.14.0

            {{relaxed_units: `bool`, optional}}

                .. versionadded:: 3.15.1

            copy: `bool`, optional
                If True (the default) then make copies of the
                {{class}} constructs, prior to the concatenation,
                thereby ensuring that the input constructs are not
                changed by the concatenation process. If False then
                some or all input constructs might be changed
                in-place, but the concatenation process will be
                faster.

                .. versionadded:: 3.15.1

        :Returns:

        TODO

        """
        out = variables[0]
        if copy:
            out = out.copy()

        if len(variables) == 1:
            return out

        data = Data.concatenate(
            [v.get_data(_fill_value=False) for v in variables],
            axis=axis,
            cull_graph=cull_graph,
            relaxed_units=relaxed_units,
            copy=copy,
        )
        out.set_data(data, copy=False)

        return out

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def cos(self, inplace=False, i=False):
        """Take the trigonometric cosine of the data element-wise.

        Units are accounted for in the calculation, so that the cosine
        of 90 degrees_east is 0.0, as is the cosine of 1.57079632
        radians. If the units are not equivalent to radians (such as
        Kelvin) then they are treated as if they were radians.

        The output units are '1' (nondimensional).

        The "standard_name" and "long_name" properties are removed from
        the result.

        .. seealso:: `arccos`, `sin`, `tan`, `cosh`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with the cosine of data values. If the
                operation was in-place then `None` is returned.

        **Examples**

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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "cos",
            inplace=inplace,
            i=i,
            delete_props=True,
        )

    def count(self):
        """Count the non-masked elements of the data.

        :Returns:

            `int`
                The number of non-masked elements.

        **Examples**

        >>> n = f.count()

        """
        data = self.get_data(None, _fill_value=False)
        if data is None:
            raise AttributeError("Can't count when there are data")

        return data.count()

    def count_masked(self):
        """Count the masked elements of the data.

        :Returns:

            `int`
                The number of masked elements.

        **Examples**

        >>> n = f.count_masked()

        """
        data = self.get_data(None, _fill_value=False)
        if data is None:
            raise AttributeError("Can't count masked when there are data")

        return data.count_masked()

    def cyclic(self, axes=None, iscyclic=True):
        """Get or set the cyclicity of an axis.

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

        **Examples**

        >>> f.cyclic()
        set()
        >>> f.cyclic(1)
        set()
        >>> f.cyclic()
        {1} TODO

        """
        data = self.get_data(None, _fill_value=False)
        if data is None:
            return set()

        return data.cyclic(axes, iscyclic)

    def datum(self, *index):
        """Return an element of the data array as a standard Python
        scalar.

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
                      If the data array shape is ``(2, 3, 6)`` then:
                        * ``f.datum(0)``  is equivalent to ``f.datum(0, 0, 0)``.
                        * ``f.datum(-1)`` is equivalent to ``f.datum(1, 2, 5)``.
                        * ``f.datum(16)`` is equivalent to ``f.datum(0, 2, 4)``.

                    If *index* is ``0`` or ``-1`` then the first or last
                    data array element respectively will be returned,
                    even if the data array is a scalar array or has two or
                    more dimensions.  ..

                  * Two or more integers. These arguments are interpreted
                    as a multidimensional index to the array. There must
                    be the same number of integers as data array
                    dimensions.  ..

                  * A tuple of integers. This argument is interpreted as a
                    multidimensional index to the array. There must be
                    the same number of integers as data array dimensions.

                    *Example:*
                      ``f.datum((0, 2, 4))`` is equivalent to ``f.datum(0,
                      2, 4)``; and ``f.datum(())`` is equivalent to
                      ``f.datum()``.

        :Returns:

                A copy of the specified element of the array as a suitable
                Python scalar.

        **Examples**

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

        """
        data = self.get_data(None, _fill_value=False)
        if data is None:
            raise ValueError(
                "ERROR: Can't return an element when there is no data array"
            )

        return data.datum(*index)

    def del_file_location(self, location):
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

        >>> f.del_file_location('/data/model/')
        '/data/model'

        """
        data = self.get_data(None, _fill_value=False, _units=False)
        if data is not None:
            return data.del_file_location(location)

        return abspath(location).rstrip(sep)

    @_manage_log_level_via_verbosity
    def equals(
        self,
        other,
        rtol=None,
        atol=None,
        verbose=None,
        ignore_data_type=False,
        ignore_fill_value=False,
        ignore_properties=None,
        ignore_compression=False,
        ignore_type=False,
    ):
        """Whether two instances are the same.

        Equality is strict by default. This means that:

        * the same descriptive properties must be present, with the
          same values and data types, and vector-valued properties
          must also have same the size and be element-wise equal (see
          the *ignore_properties* and *ignore_data_type* parameters),
          and

        ..

        * if there are data arrays then they must have same shape and
          data type, the same missing data mask, and be element-wise
          equal (see the *ignore_data_type* parameter).

        Two real numbers ``x`` and ``y`` are considered equal if
        ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on
        absolute differences) and ``rtol`` (the tolerance on relative
        differences) are positive, typically very small numbers. See
        the *atol* and *rtol* parameters.

        If data arrays are compressed then the compression type and
        the underlying compressed arrays must be the same, as well as
        the arrays in their uncompressed forms. See the
        *ignore_compression* parameter.

        Any type of object may be tested but, in general, equality is
        only possible with another object of the same type, or a
        subclass of one. See the *ignore_type* parameter.

        NetCDF elements, such as netCDF variable and dimension names,
        do not constitute part of the CF data model and so are not
        checked.

        .. versionadded:: 1.7.0

        :Parameters:

            other:
                The object to compare for equality.

            {{atol: number, optional}}

            {{rtol: number, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

            {{ignore_data_type: `bool`, optional}}

            {{ignore_fill_value: `bool`, optional}}

            {{ignore_properties: (sequence of) `str`, optional}}

            {{ignore_compression: `bool`, optional}}

            ignore_compression: `bool`, optional
                If True then any compression applied to the underlying
                arrays is ignored and only the uncompressed arrays are
                tested for equality. By default the compression type and,
                if applicable, the underlying compressed arrays must be
                the same, as well as the arrays in their uncompressed
                forms.

            {{ignore_type: `bool`, optional}}

        :Returns:

            `bool`
                Whether the two instances are equal.

        **Examples**

        >>> f.equals(f)
        True
        >>> f.equals(f.copy())
        True
        >>> f.equals('a string')
        False
        >>> f.equals(f - 1)
        False

        """
        # Check that each instance has the same Units
        try:
            if not self.Units.equals(other.Units):
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Different Units: "
                        f"{self.Units!r} != {other.Units!r}"
                    )

                return False
        except AttributeError:
            pass

        if not ignore_properties:
            ignore_properties = self._special_properties
        else:
            if isinstance(ignore_properties, str):
                ignore_properties = (ignore_properties,)

            ignore_properties = (
                tuple(ignore_properties) + self._special_properties
            )

        return super().equals(
            other,
            rtol=rtol,
            atol=atol,
            verbose=verbose,
            ignore_data_type=ignore_data_type,
            ignore_fill_value=ignore_fill_value,
            ignore_properties=ignore_properties,
            ignore_type=ignore_type,
        )

    def equivalent(self, other, rtol=None, atol=None, traceback=False):
        """True if two constructs are equal, False otherwise.

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
                numbers. The default value is set by the `atol` function.

            rtol: `float`, optional
                The tolerance on relative differences between real
                numbers. The default value is set by the `rtol` function.

        """
        if self is other:
            return True

        # Check that each instance is the same type
        if type(self) is not type(other):
            print(
                f"{self.__class__.__name__}: Different types: "
                f"{self.__class__.__name__}, {other.__class__.__name__}"
            )
            return False

        identity0 = self.identity()
        identity1 = other.identity()

        if identity0 is None or identity1 is None or identity0 != identity1:
            # add traceback
            return False

        # ------------------------------------------------------------
        # Check the special attributes
        # ------------------------------------------------------------
        self_special = self._private["special_attributes"]
        other_special = other._private["special_attributes"]
        if set(self_special) != set(other_special):
            if traceback:
                print(
                    "%s: Different attributes: %s"
                    % (
                        self.__class__.__name__,
                        set(self_special).symmetric_difference(other_special),
                    )
                )
            return False

        for attr, x in self_special.items():
            y = other_special[attr]

            result = cf_equivalent(
                x, y, rtol=rtol, atol=atol, traceback=traceback
            )

            if not result:
                if traceback:
                    print(
                        f"{self.__class__.__name__}: Different {attr} "
                        f"attributes: {x!r}, {x!r}"
                    )
                return False

        # ------------------------------------------------------------
        # Check the data
        # ------------------------------------------------------------
        if not self._equivalent_data(
            other, rtol=rtol, atol=atol, traceback=traceback
        ):
            # add traceback
            return False

        return True

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def convert_reference_time(
        self,
        units=None,
        calendar_months=False,
        calendar_years=False,
        inplace=False,
        i=False,
    ):
        """Convert reference time data values to have new units.

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

        **Performance**

        For conversions which do not require a change in the
        date-times implied by the data values, this method will be
        considerably slower than a simple reassignment of the
        units. For example, if the original units are ``'days since
        2000-12-1'`` then ``c.Units = cf.Units('days since
        1901-1-1')`` will give the same result and be considerably
        faster than ``c.convert_reference_time(cf.Units('days since
        1901-1-1'))``.

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

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with converted reference time data
                values, or `None` if the operation was in-place.

        **Examples**

        >>> print(f.array)
        [0 1 2 3]
        >>> f.Units
        <Units: months since 2004-1-1>
        >>> print(f.datetime_array)
        [cftime.DatetimeGregorian(2003, 12, 1, 0, 0, 0, 0, has_year_zero=False)
         cftime.DatetimeGregorian(2003, 12, 31, 10, 29, 3, 831223, has_year_zero=False)
         cftime.DatetimeGregorian(2004, 1, 30, 20, 58, 7, 662446, has_year_zero=False)
         cftime.DatetimeGregorian(2004, 3, 1, 7, 27, 11, 493670, has_year_zero=False)]
        >>> g = f.convert_reference_time(calendar_months=True)
        >>> g.Units
        <Units: days since 2004-1-1>
        >>> print(g.datetime_array)
        [cftime.DatetimeGregorian(2003, 12, 1, 0, 0, 0, 0, has_year_zero=False)
         cftime.DatetimeGregorian(2004, 1, 1, 0, 0, 0, 0, has_year_zero=False)
         cftime.DatetimeGregorian(2004, 2, 1, 0, 0, 0, 0, has_year_zero=False)
         cftime.DatetimeGregorian(2004, 3, 1, 0, 0, 0, 0, has_year_zero=False)]
        >>> print(g.array)
        [ 0 31 62 91]

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "convert_reference_time",
            inplace=inplace,
            units=units,
            calendar_months=calendar_months,
            calendar_years=calendar_years,
        )

    def file_locations(self):
        """The locations of files containing parts of the data.

        Returns the locations of any files that may be required to
        deliver the computed data array.

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
        data = self.get_data(None, _fill_value=False, _units=False)
        if data is not None:
            return data.file_locations()

        return set()

    @_inplace_enabled(default=False)
    def flatten(self, axes=None, inplace=False):
        """Flatten axes of the data.

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

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "flatten",
            (axes,),
            inplace=inplace,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def floor(self, inplace=False, i=False):
        """Floor the data array, element-wise.

        The floor of ``x`` is the largest integer ``n``, such that
        ``n<=x``.

        .. versionadded:: 1.0

        .. seealso:: `ceil`, `rint`, `trunc`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with floored data. If the operation was
                in-place then `None` is returned.

        **Examples**

        >>> print(f.array)
        [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
        >>> print(f.floor().array)
        [-2. -2. -2. -1.  0.  1.  1.  1.  1.]
        >>> f.floor(inplace=True)
        >>> print(f.array)
        [-2. -2. -2. -1.  0.  1.  1.  1.  1.]

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "floor",
            inplace=inplace,
            i=i,
            delete_props=True,
        )

    def match_by_naxes(self, *naxes):
        """Whether or not the data has a given dimensionality.

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

        **Examples**

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

        """
        if not naxes:
            return True

        data = self.get_data(None, _fill_value=False)
        if data is None:
            return False

        self_ndim = data.ndim
        for ndim in naxes:
            ok = ndim == self_ndim
            if ok:
                return True

        return False

    def match_by_units(self, *units, exact=True):
        """Whether or not the construct has given units.

        .. versionadded:: 3.0.0

        .. seealso:: `match`, `match_by_identity`, `match_by_property`,
                     `match_by_naxes`

        :Parameters:

            units: `str` or `re.Pattern` or `Units`, optional
                Units to be compared.

                Units are specified by a string (e.g. ``'m s-1'``), or a
                compiled regular expression (e.g. ``re.compile('^m')``),
                or a `Units` object (e.g. ``Units('m s-1')``).

                If no units are provided then there is always a match.

            exact: `bool`, optional
                If False then a match occurs if the construct's units are
                equivalent to any of those given by *units*. For example,
                metres and are equivalent to kilometres. By default, a
                match only occurs if the construct's units are exactly one
                of those given by *units*. Note that the format of the
                units is not important, i.e. 'm' is exactly the same as
                'metres' for this purpose.

        :Returns:

            `bool`
                Whether or not there is a match.

        **Examples**

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

        """
        if not units:
            return True

        self_units = self.Units

        ok = False
        for value in units:
            try:
                # re.Pattern object
                ok = value.search(self_units.units)
            except (AttributeError, TypeError):
                if exact:
                    ok = Units(value).equals(self_units)
                else:
                    ok = Units(value).equivalent(self_units)

            if ok:
                break

        return ok

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def all(self):
        """Test whether all data elements evaluate to True.

        Performs a logical "and" over the data array and returns the
        result. Masked values are considered as True during computation.

        .. seealso:: `allclose`, `any`

        :Returns:

            `bool`
                Whether to not all data elements evaluate to True.

        **Examples**

        >>> print(f.array)
        [[0  3  0]]
        >>> f.all()
        False

        >>> print(f.array)
        [[1  3  --]]
        >>> f.all()
        True

        """
        data = self.get_data(None)
        if data is not None:
            return data.all()

        return False

    def allclose(self, y, atol=None, rtol=None):
        """Test whether all data are element-wise equal to other,
        broadcastable data.

        Two real numbers ``x`` and ``y`` are considered equal if
        ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on
        absolute differences) and ``rtol`` (the tolerance on relative
        differences) are positive, typically very small numbers. See
        the *atol* and *rtol* parameters.

        .. seealso:: `all`, `any`, `isclose`

        :Parameters:

            y:
                The object to be compared with the data array. *y*
                must be broadcastable to the data array and if *y* has
                units then they must be compatible. May be any object
                that can be converted to a `Data` object (which
                includes numpy array and `Data` objects).

            atol: `float`, optional
                The tolerance on absolute differences between real
                numbers. The default value is set by the `atol`
                function.

            rtol: `float`, optional
                The tolerance on relative differences between real
                numbers. The default value is set by the `rtol`
                function.

        :Returns:

            `bool`
                Returns `True` if the data are equal within the given
                tolerance; `False` otherwise.

        **Examples**

        >>> x = f.allclose(g)

        """
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

        return data.allclose(y_data, rtol=rtol, atol=atol)

    def any(self):
        """Test whether any data elements evaluate to True.

        Performs a logical "or" over the data array and returns the
        result. Masked values are considered as False during
        computation.

        .. seealso:: `all`, `allclose`

        :Returns:

            `bool`
                Whether to not any data elements evaluate to `True`.

        **Examples**

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

        """
        data = self.get_data(None)
        if data is not None:
            return data.any()

        return False

    def fill_value(self, default=None):
        """Return the data array missing data value.

        This is the value of the `missing_value` CF property, or if
        that is not set, the value of the `_FillValue` CF property,
        else if that is not set, ``None``. In the last case the
        default `numpy` missing data value for the array's data type
        is assumed if a missing data value is required.

        .. seealso:: `cf.default_netCDF_fillvals`, `_FillValue`,
                     `missing_value`

        :Parameters:

            default: optional
                If the missing value is unset then return this
                value. By default, *default* is `None`. If *default*
                is the special value ``'netCDF'`` then return the
                netCDF default value appropriate to the data array's
                data type is used. These may be found with the
                `cf.default_netCDF_fillvals` function. For example:

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

                The missing data value or, if one has not been set, the
                value specified by *default*

        **Examples**

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

        """
        fillval = self.get_property("missing_value", None)
        if fillval is None:
            fillval = self.get_property("_FillValue", None)

        if fillval is None:
            if default == "netCDF":
                d = self.dtype
                fillval = default_netCDF_fillvals()[d.kind + str(d.itemsize)]
            else:
                fillval = default

        return fillval

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def flip(self, axes=None, inplace=False, i=False):
        """Flip (reverse the direction of) data dimensions.

        .. seealso:: `flatten`, `insert_dimension`, `squeeze`,
                     `transpose`, `unsqueeze`

        :Parameters:

            axes: optional
               Select the domain axes to flip. One, or a sequence, of:

                  * The position of the dimension in the data.

                If no axes are specified then all axes are flipped.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with flipped axes, or `None` if the
                operation was in-place.

        **Examples**

        >>> f.flip()
        >>> f.flip(1)
        >>> f.flip([0, 1])

        >>> g = f[::-1, :, ::-1]
        >>> f.flip([2, 0]).equals(g)
        True

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "flip",
            (axes,),
            inplace=inplace,
            i=i,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def exp(self, inplace=False, i=False):
        """The exponential of the data, element-wise.

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. seealso:: `log`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with the exponential of data values. If
                the operation was in-place then `None` is returned.

        **Examples**

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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "exp",
            inplace=inplace,
            i=i,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def sin(self, inplace=False, i=False):
        """Take the trigonometric sine of the data element-wise.

        Units are accounted for in the calculation. For example, the
        sine of 90 degrees_east is 1.0, as is the sine of 1.57079632
        radians. If the units are not equivalent to radians (such as
        Kelvin) then they are treated as if they were radians.

        The Units are changed to '1' (nondimensional).

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. seealso:: `cos`, `tan`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with the sine of data values. If the
                operation was in-place then `None` is returned.

        **Examples**

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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "sin",
            inplace=inplace,
            i=i,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def arctan(self, inplace=False):
        """Take the trigonometric inverse tangent of the data element-
        wise.

        Units are ignored in the calculation. The result has units of
        radians.

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. versionadded:: 3.0.7

        .. seealso:: `tan`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the trigonometric inverse tangent
                of data values. If the operation was in-place then
                `None` is returned.

        **Examples**

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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arctan",
            inplace=inplace,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def arctanh(self, inplace=False):
        """Take the inverse hyperbolic tangent of the data element-wise.

        Units are ignored in the calculation. The result has units of
        radians.

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. versionadded:: 3.2.0

        .. seealso:: `tanh`, `arcsinh`, `arccosh`, `arctan`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the inverse hyperbolic tangent of
                data values. If the operation was in-place then `None`
                is returned.

        **Examples**

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
        >>> f.masked_invalid(inplace=True)
        >>> print(f.array)
        [-- -- 1.0986122886681098 0.6931471805599453 --]

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arctanh",
            inplace=inplace,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def arcsin(self, inplace=False):
        """Take the trigonometric inverse sine of the data element-wise.

        Units are ignored in the calculation. The result has units of
        radians.

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. versionadded:: 3.2.0

        .. seealso:: `sin`, `arccos`, `arctan`, `arcsinh`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the trigonometric inverse sine of
                data values. If the operation was in-place then `None`
                is returned.

        **Examples**

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
        >>> f.masked_invalid(inplace=True)
        >>> print(f.array)
        [-- 1.5707963267948966 0.9272952180016123 0.6435011087932844 --]

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arcsin",
            inplace=inplace,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def arcsinh(self, inplace=False):
        """Take the inverse hyperbolic sine of the data element-wise.

        Units are ignored in the calculation. The result has units of
        radians.

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. versionadded:: 3.1.0

        .. seealso:: `sinh`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the inverse hyperbolic sine of data
                values.  If the operation was in-place then `None` is
                returned.

        **Examples**

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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arcsinh",
            inplace=inplace,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def arccos(self, inplace=False):
        """Take the trigonometric inverse cosine of the data element-
        wise.

        Units are ignored in the calculation. The result has units of
        radians.

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. versionadded:: 3.2.0

        .. seealso:: `cos`, `arcsin`, `arctan`, `arccosh`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the trigonometric inverse cosine of
                data values. If the operation was in-place then `None`
                is returned.

        **Examples**

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
        >>> f.masked_invalid(inplace=True)
        >>> print(f.array)
        [-- 0.0 0.6435011087932843 0.9272952180016123 --]

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arccos",
            inplace=inplace,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def arccosh(self, inplace=False):
        """Take the inverse hyperbolic cosine of the data element-wise.

        Units are ignored in the calculation. The result has units of
        radians.

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. versionadded:: 3.2.0

        .. seealso:: `cosh`, `arcsinh`, `arctanh`, `arccos`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the inverse hyperbolic cosine of
                data values. If the operation was in-place then `None`
                is returned.

        **Examples**

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
        >>> f.masked_invalid(inplace=True)
        >>> print(f.array)
        [0.6223625037147786 0.0 -- -- --]

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arccosh",
            inplace=inplace,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def tan(self, inplace=False, i=False):
        """Take the trigonometric tangent of the data element-wise.

        Units are accounted for in the calculation, so that the
        tangent of 180 degrees_east is 0.0, as is the tangent of
        3.141592653589793 radians. If the units are not equivalent to
        radians (such as Kelvin) then they are treated as if they were
        radians.

        The Units are changed to '1' (nondimensional).

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. seealso:: `arctan`, `cos`, `sin`, `tanh`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with the tangent of data values. If the
                operation was in-place then `None` is returned.

        **Examples**

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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "tan",
            inplace=inplace,
            i=i,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def tanh(self, inplace=False):
        """Take the hyperbolic tangent of the data array.

        Units are accounted for in the calculation. If the units are
        not equivalent to radians (such as Kelvin) then they are
        treated as if they were radians. For example, the the
        hyperbolic tangent of 90 degrees_east is 0.91715234, as is the
        hyperbolic tangent of 1.57079632 radians.

        The output units are changed to '1' (nondimensional).

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. versionadded:: 3.1.0

        .. seealso:: `sinh`, `cosh`


        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the hyperbolic tangent of data
                values. If the operation was in-place then `None` is
                returned.

        **Examples**

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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "tanh",
            inplace=inplace,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def sinh(self, inplace=False):
        """Take the hyperbolic sine of the data element-wise.

        Units are accounted for in the calculation. If the units are
        not equivalent to radians (such as Kelvin) then they are
        treated as if they were radians. For example, the the
        hyperbolic sine of 90 degrees_north is 2.30129890, as is the
        hyperbolic sine of 1.57079632 radians.

        The output units are changed to '1' (nondimensional).

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. versionadded:: 3.1.0

        .. seealso:: `arcsinh`, `cosh`, `tanh`, `sin`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the hyperbolic sine of data
                values. If the operation was in-place then `None` is
                returned.

        **Examples**

        >>> f.Units
        <Units: degrees_north>
        >>> print(f.array)
        [[-90 0 90 --]]
        >>> g = f.sinh()
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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "sinh",
            inplace=inplace,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def cosh(self, inplace=False):
        """Take the hyperbolic cosine of the data element-wise.

        Units are accounted for in the calculation. If the units are
        not equivalent to radians (such as Kelvin) then they are
        treated as if they were radians. For example, the the
        hyperbolic cosine of 0 degrees_east is 1.0, as is the
        hyperbolic cosine of 1.57079632 radians.

        The output units are changed to '1' (nondimensional).

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. versionadded:: 3.1.0

        .. seealso:: `arccosh`, `sinh`, `tanh`, `cos`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the hyperbolic cosine of data
                values. If the operation was in-place then `None` is
                returned.

        **Examples**

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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "cosh",
            inplace=inplace,
            delete_props=True,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def log(self, base=None, inplace=False, i=False):
        """The logarithm of the data array.

        By default the natural logarithm is taken, but any base may be
        specified.

        The "standard_name" and "long_name" properties are removed
        from the result.

        .. seealso:: `exp`

        :Parameters:

            base: number, optional
                The base of the logarithm. By default a natural logarithm
                is taken.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with the logarithm of data values, or
                `None` if the operation was in-place.

        **Examples**

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

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "log",
            (base,),
            inplace=inplace,
            i=i,
            delete_props=True,
        )

    def to_dask_array(self):
        """Convert the data to a `dask` array.

        .. versionadded:: 3.14.0

        .. seealso:: `cf.Data.to_dask_array`

        :Returns:

            `dask.array.Array`
                The dask array contained within the {{class}} instance.

        **Examples**

        >>> f.to_dask_array()
        dask.array<copy, shape=(10, 9), dtype=float64, chunksize=(10, 9), chunktype=numpy.ndarray>

        >>> f.to_dask_array() is f.data.to_dask_array()
        True

        """
        data = self.get_data(None)
        if data is None:
            raise ValueError("Can't get dask array when there is no data")

        return data.to_dask_array()

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def trunc(self, inplace=False, i=False):
        """Truncate the data, element-wise.

        The truncated value of the scalar ``x``, is the nearest
        integer ``i`` which is closer to zero than ``x`` is. I.e. the
        fractional part of the signed number ``x`` is discarded.

        .. versionadded:: 1.0

        .. seealso:: `ceil`, `floor`, `rint`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with truncated data. If the operation
                was in-place then `None` is returned.

        **Examples**

        >>> print(f.array)
        [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
        >>> print(f.trunc().array)
        [-1. -1. -1. -1.  0.  1.  1.  1.  1.]
        >>> f.trunc(inplace=True)
        >>> print(f.array)
        [-1. -1. -1. -1.  0.  1.  1.  1.  1.]

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "trunc",
            inplace=inplace,
            i=i,
        )

    def unique(self):
        """The unique elements of the data.

        :Returns:

            `Data`
                The unique data array values in a one dimensional `Data`
                object.

        **Examples**

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

        """
        data = self.get_data(None)
        if data is not None:
            return data.unique()

        raise ValueError(
            "ERROR: Can't get unique values when there is no data array"
        )

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

        n = self.get_property("standard_name", None)
        if n is not None:
            return str(n)

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

        for prop in ("cf_role", "axis", "long_name"):
            n = self.get_property(prop, None)
            if n is not None:
                return f"{prop}={n}"

        n = self.nc_get_variable(None)
        if n is not None:
            return f"ncvar%{n}"

        return default

    def identities(self, generator=False, **kwargs):
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

        TODO
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
        identities = super().identities(generator=True, **kwargs)

        i = getattr(self, "id", None)
        if i is None:
            g = identities
        else:
            g = chain((f"id%{i}",), identities)

        if generator:
            return g

        return list(g)

    def inspect(self):
        """Inspect the object for debugging.

        .. seealso:: `cf.inspect`

        :Returns:

            `None`

        """
        print(cf_inspect(self))  # pragma: no cover

    def iscyclic(self, axis):
        """Whether or not a given axis is cyclic.

        .. versionadded:: 3.5.0

        .. seealso:: `cyclic`, `period`, `isperiodic`

        :Parameters:

            axis: `int`, optional
               Select the axis by its position in the data dimensions.

        :Returns:

            `bool`
                `True` if the selected axis is cyclic, otherwise `False`.

        **Examples**

        >>> f.iscyclic('X')
        True
        >>> f.iscyclic('latitude')
        False

        >>> x = f.iscyclic('long_name=Latitude')
        >>> x = f.iscyclic('dimensioncoordinate1')
        >>> x = f.iscyclic('domainaxis2')
        >>> x = f.iscyclic('key%domainaxis2')
        >>> x = f.iscyclic('ncdim%y')
        >>> x = f.iscyclic(2)

        """
        axis = self._parse_axes(axis)
        if len(axis) != 1:
            raise ValueError(
                "Only one axis can be checked for cyclicity at once, but "
                f"multiple were selected: {axis}"
            )

        return axis[0] in self.cyclic()

    def get_data(self, default=ValueError(), _units=None, _fill_value=True):
        """Return the data.

        Note that a `Data` instance is returned. Use its `array`
        attribute to return the data as an independent `numpy` array.

        The units, calendar and fill value properties are, if set,
        inserted into the data.

        .. versionadded:: 1.7.0

        .. seealso:: `array`, `data`, `del_data`, `has_data`,
                     `set_data`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if data
                have not been set.

                {{default Exception}}

            _units: optional
                Ignored.

            _fill_value: optional

        :Returns:

                The data.

        **Examples**

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

        """
        return super().get_data(
            default=default, _units=False, _fill_value=_fill_value
        )

    @_inplace_enabled(default=False)
    @_manage_log_level_via_verbosity
    def halo(
        self,
        depth,
        axes=None,
        tripolar=None,
        fold_index=-1,
        inplace=False,
        verbose=None,
        size=None,
    ):
        """Expand the data by adding a halo.

        The halo may be applied over a subset of the data dimensions
        and each dimension may have a different halo size (including
        zero). The halo region is populated with a copy of the
        proximate values from the original data.

        **Cyclic axes**

        A cyclic axis that is expanded with a halo of at least size 1
        is no longer considered to be cyclic.

        **Tripolar domains**

        Data for global tripolar domains are a special case in that a
        halo added to the northern end of the "Y" axis must be filled
        with values that are flipped in "X" direction. Such domains
        need to be explicitly indicated with the *tripolar* parameter.

        .. versionadded:: 3.5.0

        :Parameters:

            depth: `int` or `dict`
                Specify the size of the halo for each axis.

                If *depth* is a non-negative `int` then this is the
                halo size that is applied to all of the axes defined
                by the *axes* parameter.

                Alternatively, halo sizes may be assigned to axes
                individually by providing a `dict` for which a key
                specifies an axis (defined by its integer position in
                the data) with a corresponding value of the halo size
                for that axis. Axes not specified by the dictionary
                are not expanded, and the *axes* parameter must not
                also be set.

                *Parameter example:*
                  Specify a halo size of 1 for all otherwise selected
                  axes: ``1``

                *Parameter example:*
                  Specify a halo size of zero: ``0``. This results in
                  no change to the data shape.

                *Parameter example:*
                  For data with three dimensions, specify a halo size
                  of 3 for the first dimension and 1 for the second
                  dimension: ``{0: 3, 1: 1}``. This is equivalent to
                  ``{0: 3, 1: 1, 2: 0}``

                *Parameter example:*
                  Specify a halo size of 2 for the first and last
                  dimensions `depth=2, axes=[0, -1]`` or equivalently
                  ``depth={0: 2, -1: 2}``.

            axes: (sequence of) `int`
                Select the domain axes to be expanded, defined by
                their integer positions in the data. By default, or if
                *axes* is `None`, all axes are selected. No axes are
                expanded if *axes* is an empty sequence.

            tripolar: `dict`, optional
                A dictionary defining the "X" and "Y" axes of a global
                tripolar domain. This is necessary because in the
                global tripolar case the "X" and "Y" axes need special
                treatment, as described above. It must have keys
                ``'X'`` and ``'Y'``, whose values identify the
                corresponding domain axis construct by their integer
                positions in the data.

                The "X" and "Y" axes must be a subset of those
                identified by the *depth* or *axes* parameter.

                See the *fold_index* parameter.

                *Parameter example:*
                  Define the "X" and Y" axes by positions 2 and 1
                  respectively of the data: ``{'X': 2, 'Y': 1}``

            fold_index: `int`, optional
                Identify which index of the "Y" axis corresponds to
                the fold in "X" axis of a tripolar grid. The only
                valid values are ``-1`` for the last index, and ``0``
                for the first index. By default it is assumed to be
                the last index. Ignored if *tripolar* is `None`.

            {{inplace: `bool`, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

            size: deprecated at version 3.14.0
                Use the *depth* parameter instead.

        :Returns:

            `{{class}}` or `None`
                The expanded construct, or `None` if the operation was
                in-place.

        **Examples**

        TODO

        """
        if size is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "halo",
                {"size": None},
                message="Use the 'depth' parameter instead.",
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        v = _inplace_enabled_define_and_cleanup(self)

        data = v.get_data(None)
        if data is not None:
            data.halo(
                depth,
                axes=axes,
                tripolar=tripolar,
                fold_index=fold_index,
                inplace=True,
                verbose=verbose,
            )

        return v

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def override_calendar(self, calendar, inplace=False, i=False):
        """Override the calendar of date-time units.

        The new calendar need not be equivalent to the original one,
        and the data array elements will not be changed to reflect the
        new units. Therefore, this method should only be used when it
        is known that the data array values are correct but the
        calendar has been incorrectly encoded.

        Not to be confused with setting the `calendar` or `Units`
        attributes to a calendar which is equivalent to the original
        calendar

        .. seealso:: `calendar`, `override_units`, `units`, `Units`

        :Parameters:

            calendar: `str`
                The new calendar.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                TODO

        **Examples**

        TODO

        >>> g = f.override_calendar('noleap')

        """
        v = _inplace_enabled_define_and_cleanup(self)

        data = v.get_data(None, _fill_value=False)
        if data is not None:
            data.override_calendar(calendar, inplace=True)
            v._custom["Units"] = data.Units
        else:
            if not v.Units.isreftime:
                raise ValueError(
                    "Can't override the calendar of non-reference-time "
                    f"units: {self.Units!r}"
                )

            PropertiesData.Units.fset(
                v, Units(getattr(v.Units, "units", None), calendar=calendar)
            )

        return v

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def override_units(self, units, inplace=False, i=False):
        """Override the units.

        The new units need not be equivalent to the original ones, and
        the data array elements will not be changed to reflect the new
        units. Therefore, this method should only be used when it is
        known that the data array values are correct but the units
        have incorrectly encoded.

        Not to be confused with setting the `units` or `Units`
        attributes to units which are equivalent to the original
        units.

        .. seealso:: `calendar`, `override_calendar`, `units`, `Units`

        :Parameters:

            units: `str` or `Units`
                The new units for the data array.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`

        TODO

        **Examples**

        >>> f.Units
        <Units: hPa>
        >>> f.datum(0)
        100000.0
        >>> f.override_units('km', inplace=True)
        >>> f.Units
        <Units: km>
        >>> f.datum(0)
        100000.0
        >>> f.override_units(Units('watts'), inplace=True)
        >>> f.Units
        <Units: watts>
        >>> f.datum(0)
        100000.0

        """
        v = _inplace_enabled_define_and_cleanup(self)

        units = Units(units)

        data = v.get_data(None, _fill_value=False)
        if data is not None:
            data.override_units(units, inplace=True)
        else:
            v._custom["Units"] = units

        # Override the Units on the period
        period = v.period()
        if period is not None:
            v.period(period=period.override_units(units))

        return v

    @_inplace_enabled(default=False)
    def rechunk(
        self,
        chunks=_DEFAULT_CHUNKS,
        threshold=None,
        block_size_limit=None,
        balance=False,
        bounds=True,
        interior_ring=True,
        inplace=False,
    ):
        """Change the chunk structure of the data.

        .. versionadded:: 3.14.0

        .. seealso:: `cf.Data.rechunk`

        :Parameters:

            {{chunks: `int`, `tuple`, `dict` or `str`, optional}}

            {{threshold: `int`, optional}}

            {{block_size_limit: `int`, optional}}

            {{balance: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with rechunked data, or `None` if the
                operation was in-place.

        **Examples**

        See `cf.Data.rechunk` for examples.

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "rechunk",
            inplace=inplace,
            chunks=chunks,
            threshold=threshold,
            block_size_limit=block_size_limit,
            balance=balance,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def rint(self, inplace=False, i=False):
        """Round the data to the nearest integer, element-wise.

        .. versionadded:: 1.0

        .. seealso:: `ceil`, `floor`, `trunc`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with rounded data. If the operation was
                in-place then `None` is returned.

        **Examples**

        >>> print(f.array)
        [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
        >>> print(f.rint().array)
        [-2. -2. -1. -1.  0.  1.  1.  2.  2.]
        >>> f.rint(inplace=True)
        >>> print(f.array)
        [-2. -2. -1. -1.  0.  1.  1.  2.  2.]

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "rint",
            inplace=inplace,
            i=i,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def round(self, decimals=0, inplace=False, i=False):
        """Round the data to the given number of decimals.

        Values exactly halfway between rounded decimal values are
        rounded to the nearest even value. Thus 1.5 and 2.5 round to
        2.0, -0.5 and 0.5 round to 0.0, etc. Results may also be
        surprising due to the inexact representation of decimal
        fractions in the IEEE floating point standard and errors
        introduced when scaling by powers of ten.

        .. versionadded:: 1.1.4

        .. seealso:: `ceil`, `floor`, `rint`, `trunc`

        :Parameters:

            decimals: `int`, optional
                Number of decimal places to round to (0 by
                default). If decimals is negative, it specifies the
                number of positions to the left of the decimal point.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with rounded data. If the operation was
                in-place then `None` is returned.

        **Examples**

        >>> print(f.array)
        [-1.81, -1.41, -1.01, -0.91,  0.09,  1.09,  1.19,  1.59,  1.99])
        >>> print(f.round().array)
        [-2., -1., -1., -1.,  0.,  1.,  1.,  2.,  2.]
        >>> print(f.round(1).array)
        [-1.8, -1.4, -1. , -0.9,  0.1,  1.1,  1.2,  1.6,  2. ]
        >>> print(f.round(-1).array)
        [-0., -0., -0., -0.,  0.,  0.,  0.,  0.,  0.]

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "round",
            inplace=inplace,
            i=i,
            decimals=decimals,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def roll(self, iaxis, shift, inplace=False, i=False):
        """Roll the data along one or more axes.

        Elements that roll beyond the last position are re-introduced
        at the first.

        .. seealso:: `flatten`, `insert_dimension`, `flip`, `squeeze`,
                     `transpose`

        :Parameters:

            axis: `int`, or `tuple` of `int`
                Axis or axes along which elements are shifted.

                *Parameter example:*
                  Roll the second axis: ``axis=1``.

                *Parameter example:*
                  Roll the last axis: ``axis=-1``.

                *Parameter example:*
                  Roll the first and last axes: ``axis=(0, -1)``.

            shift: `int`, or `tuple` of `int`
                The number of places by which elements are shifted.
                If a `tuple`, then *axis* must be a tuple of the same
                size, and each of the given axes is shifted by the
                corresponding number. If an `int` while *axis* is a
                `tuple` of `int`, then the same value is used for all
                given axes.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with rolled data. If the operation was
                in-place then `None` is returned.

        **Examples**

        >>> print(f.array)
        [ 0  1  2  3  4  5  6  7  8  9 10 11]
        >>> print(f.roll(0, 2).array)
        [10 11  0  1  2  3  4  5  6  7  8  9]
        >>> print(f.roll(0, -2).array)
        [ 2  3  4  5  6  7  8  9 10 11  0  1]

        """
        return self._apply_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "roll",
            (iaxis, shift),
            inplace=inplace,
            i=i,
        )

    def set_data(self, data, copy=True, inplace=True):
        """Set the data.

        The units, calendar and fill value of the incoming `Data`
        instance are removed prior to insertion.

        .. versionadded:: 3.0.0

        .. seealso:: `data`, `del_data`, `get_data`, `has_data`

        :Parameters:

            data: `Data`
                The data to be inserted.

                {{data_like}}

            copy: `bool`, optional
                If False then do not copy the data prior to
                insertion. By default the data are copied.

            {{inplace: `bool`, optional (default True)}}

                .. versionadded:: 3.7.0

        :Returns:

            `None` or `{{class}}`
                If the operation was in-place then `None` is returned,
                otherwise return a new `{{class}}` instance containing
                the new data.

        **Examples**

        >>> f = cf.{{class}}()
        >>> f.set_data([1, 2, 3])
        >>> f.has_data()
        True
        >>> f.get_data()
        <CF Data(3): [1, 2, 3]>
        >>> f.data
        <CF Data(3): [1, 2, 3]>
        >>> f.del_data()
        <CF Data(3): [1, 2, 3]>
        >>> g = f.set_data([4, 5, 6], inplace=False)
        >>> g.data
        <CF Data(3): [4, 5, 6]>
        >>> f.has_data()
        False
        >>> print(f.get_data(None))
        None
        >>> print(f.del_data(None))
        None

        """
        _Data = self._Data
        if not isinstance(data, _Data):
            data = _Data(data, copy=False)

        if not data.Units:
            units = self.Units
            if units is not None:
                if copy:
                    copy = False
                    data = data.override_units(units, inplace=False)
                else:
                    data.override_units(units, inplace=True)

        return super().set_data(data, copy=copy, inplace=inplace)

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def where(
        self, condition, x=None, y=None, inplace=False, i=False, verbose=None
    ):
        """Set data array elements depending on a condition.

        .. seealso:: `cf.masked`, `hardmask`, `subspace`

        :Parameters:

            TODO

        :Returns:

            TODO

        **Examples**

            TODO

        """
        v = _inplace_enabled_define_and_cleanup(self)

        data = v.get_data(None, _fill_value=False)
        if data is None:
            raise ValueError("ERROR: Can't set data in nonexistent data array")

        try:
            condition_data = condition.get_data(None)
        except AttributeError:
            pass
        else:
            if condition_data is None:
                raise ValueError(
                    "ERROR: Can't set data from "
                    f"{condition.__class__.__name__} with no data array"
                )

            condition = condition_data

        try:
            x_data = x.get_data(None, _fill_value=False)
        except AttributeError:
            pass
        else:
            if x_data is None:
                raise ValueError(
                    f"ERROR: Can't set data from {x.__class__.__name__} "
                    "with no data array"
                )

            x = x_data

        try:
            y_data = y.get_data(None, _fill_value=False)
        except AttributeError:
            pass
        else:
            if y_data is None:
                raise ValueError(
                    f"ERROR: Can't set data from {y.__class__.__name__} "
                    "with no data array"
                )

            y = y_data

        data.where(condition, x, y, inplace=True, verbose=verbose)

        return v

    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    @property
    def dtarray(self):
        """Alias for `datetime_array`."""
        return self.datetime_array

    def max(self, *args, **kwargs):
        """Alias for `maximum`."""
        return self.maximum(*args, **kwargs)

    def min(self, *args, **kwargs):
        """Alias for `minimum`."""
        return self.minimum(*args, **kwargs)

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    @property
    def attributes(self):
        """Deprecated at version 3.0.0."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, "attributes", version="3.0.0", removed_at="4.0.0"
        )

    @property
    def Data(self):
        """Deprecated at version 3.0.0, use `data` attribute or
        `get_data` method instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "Data",
            "Use 'data' attribute or 'get_data' method instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @Data.setter
    def Data(self, value):
        """Deprecated at version 3.0.0, use `set_data` method
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "Data",
            "Use 'data' attribute or 'set_data' method instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @Data.deleter
    def Data(self):
        """Deprecated at version 3.0.0, use `del_data` method
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "Data",
            "Use 'data' attribute or 'del_data' method instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def dtvarray(self):
        """Deprecated at version 3.0.0."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, "dtvarray", version="3.0.0", removed_at="4.0.0"
        )  # pragma: no cover

    @property
    def hasbounds(self):
        """Deprecated at version 3.0.0, use `has_bounds` method
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "hasbounds",
            "Use 'has_bounds' method instead",
            version="3.0.0",
            removed_at="4.0.0",
        )

    @property
    def hasdata(self):
        """Deprecated at version 3.0.0, use `has_data` method
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "hasdata",
            "Use 'has_data' method instead",
            version="3.0.0",
            removed_at="4.0.0",
        )

    @property
    def isauxiliary(self):
        """Deprecated at version 3.7.0, use `construct_type` attribute
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "isauxiliary",
            "Use 'construct_type'' attribute instead.",
            version="3.7.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def isdimension(self):
        """Deprecated at version 3.7.0, use `construct_type` attribute
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "isdimension",
            "Use 'construct_type'' attribute instead.",
            version="3.7.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def isdomainancillary(self):
        """Deprecated at version 3.7.0, use `construct_type` attribute
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "isdomainancillary",
            "Use 'construct_type'' attribute instead.",
            version="3.7.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def isfieldancillary(self):
        """Deprecated at version 3.7.0, use `construct_type` attribute
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "isfieldancillary",
            "Use 'construct_type'' attribute instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def ismeasure(self):
        """Deprecated at version 3.7.0, use `construct_type` attribute
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "ismeasure",
            "Use 'construct_type'' attribute instead.",
            version="3.7.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def unsafe_array(self):
        """Deprecated at version 3.0.0, use `array` attribute
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "unsafe_array",
            "Use 'array' attribute instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def asdatetime(self, i=False):
        """Deprecated at version 3.0.0."""
        _DEPRECATION_ERROR_METHOD(
            self, "asdatetime", version="3.0.0", removed_at="4.0.0"
        )  # pragma: no cover

    def asreftime(self, i=False):
        """Deprecated at version 3.0.0."""
        _DEPRECATION_ERROR_METHOD(
            self, "asreftime", version="3.0.0", removed_at="4.0.0"
        )  # pragma: no cover

    def expand_dims(self, position=0, i=False):
        """Deprecated at version 3.0.0, use `insert_dimension` method
        instead."""
        _DEPRECATION_ERROR_METHOD(
            self,
            "expand_dims",
            "Use method 'insert_dimension' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def insert_data(self, data, copy=True):
        """Deprecated at version 3.0.0, use `set_data` method
        instead."""
        _DEPRECATION_ERROR_METHOD(
            self,
            "insert_data",
            "Use method 'set_data' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def name(
        self, default=None, identity=False, ncvar=False, relaxed_identity=None
    ):
        """Deprecated at version 3.0.0, use method 'identity'
        instead."""
        _DEPRECATION_ERROR_METHOD(
            self,
            "name",
            "Use method 'identity' instead",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def remove_data(self):
        """Deprecated at version 3.0.0, use method `del_data`
        instead."""
        _DEPRECATION_ERROR_METHOD(
            self,
            "remove_data",
            "Use method 'del_data' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def select(self, *args, **kwargs):
        """Deprecated at version 3.0.0."""
        _DEPRECATION_ERROR_METHOD(
            self, "select", version="3.0.0", removed_at="4.0.0"
        )  # pragma: no cover


class Subspace:
    """Define a subspace of a field construct."""

    __slots__ = ("variable",)

    def __init__(self, variable):
        """Set the contained variable."""
        self.variable = variable

    def __getitem__(self, indices):
        """Called to implement evaluation of x[indices].

        x.__getitem__(indices) <==> x[indices]

        """
        return self.variable[indices]

    def __setitem__(self, indices, value):
        """Called to implement assignment to x[indices]

        x.__setitem__(indices, value) <==> x[indices]

        """
        if isinstance(value, self.__class__):
            value = value.data

        self.variable[indices] = value
