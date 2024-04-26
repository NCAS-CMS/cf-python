import logging

import numpy as np
from cfdm import is_log_level_debug, is_log_level_info

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
    bounds_combination_mode,
)
from ..functions import equivalent as cf_equivalent
from ..functions import inspect as cf_inspect
from ..functions import parse_indices
from ..functions import size as cf_size
from ..query import Query
from ..units import Units
from . import PropertiesData

_units_None = Units()

_month_units = ("month", "months")
_year_units = ("year", "years", "yr")

logger = logging.getLogger(__name__)


class PropertiesDataBounds(PropertiesData):
    """Mixin class for a data array with descriptive properties and cell
    bounds."""

    def __getitem__(self, indices):
        """Return a subspace of the field construct defined by indices.

        x.__getitem__(indices) <==> x[indices]

        """
        if indices is Ellipsis:
            return self.copy()

        # Parse the index
        if not isinstance(indices, tuple):
            indices = (indices,)

        arg0 = indices[0]
        if isinstance(arg0, str) and arg0 == "mask":
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
                        "Can't do a cyclic slice on a non-cyclic axis"
                    )

                new = new.roll(iaxis, shift)
        else:
            new = self.copy()  # data=False)

        #       data = self.data

        if auxiliary_mask:
            findices = tuple(auxiliary_mask) + tuple(indices)
        else:
            findices = tuple(indices)

        cname = self.__class__.__name__
        if is_log_level_debug(logger):
            logger.debug(
                f"{cname}.__getitem__: shape    = {self.shape}\n"
                f"{cname}.__getitem__: indices2 = {indices2}\n"
                f"{cname}.__getitem__: indices  = {indices}\n"
                f"{cname}.__getitem__: findices = {findices}"
            )  # pragma: no cover

        data = self.get_data(None, _fill_value=False)
        if data is not None:
            new_data = data[findices]
            new.set_data(new_data, copy=False)

            if 0 in new_data.shape:
                raise IndexError(
                    f"Indices {findices!r} result in a subspaced shape of "
                    f"{new_data.shape}, but can't create a subspace of "
                    f"{self.__class__.__name__} that has a size 0 axis"
                )

        # Subspace the interior ring array, if there is one.
        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            new.set_interior_ring(interior_ring[tuple(indices)], copy=False)

        # Subspace the bounds, if there are any
        bounds = self.get_bounds(None)
        if bounds is not None:
            bounds_data = bounds.get_data(None, _fill_value=False)
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

                if auxiliary_mask:
                    findices[1] = [
                        mask.insert_dimension(-1) for mask in findices[1]
                    ]

                if is_log_level_debug(logger):
                    logger.debug(
                        f"{self.__class__.__name__}.__getitem__: findices for "
                        f"bounds = {tuple(findices)}"
                    )  # pragma: no cover

                new.bounds.set_data(bounds_data[tuple(findices)], copy=False)

        # Remove the direction, as it may now be wrong
        new._custom.pop("direction", None)

        # Return the new bounded variable
        return new

    def __setitem__(self, indices, value):
        """Called to implement assignment to x[indices]

        x.__setitem__(indices, y) <==> x[indices] = y

        **Bounds**

        When assigning an object that has bounds to an object that also
        has bounds, then the bounds are also assigned, if possible. This
        is the only circumstance that allows bounds to be updated during
        assignment by index.

        Interior ring assignment can only occur if both ``x`` and ``y``
        have interior ring arrays. An exception will be raised only one of
        ``x`` and ``y`` has an interior ring array.

        """
        super().__setitem__(indices, value)

        # Set the interior ring, if present (added at v3.8.0).
        interior_ring = self.get_interior_ring(None)
        try:
            value_interior_ring = value.get_interior_ring(None)
        except AttributeError:
            value_interior_ring = None

        if interior_ring is not None and value_interior_ring is not None:
            indices = parse_indices(self.shape, indices)
            indices.append(slice(None))
            interior_ring[tuple(indices)] = value_interior_ring
        elif interior_ring is not None:
            raise ValueError(
                f"Can't assign {value!r} without an interior ring array to "
                f"{self!r} with an interior ring array"
            )
        elif value_interior_ring is not None:
            raise ValueError(
                f"Can't assign {value!r} with an interior ring array to "
                f"{self!r} without an interior ring array"
            )

        # Set the bounds, if present (added at v3.8.0).
        bounds = self.get_bounds(None)
        if bounds is not None:
            try:
                value_bounds = value.get_bounds(None)
            except AttributeError:
                value_bounds = None

            if value_bounds is not None:
                indices = parse_indices(self.shape, indices)
                indices.append(Ellipsis)
                bounds[tuple(indices)] = value_bounds

    def __eq__(self, y):
        """The rich comparison operator ``==``

        x.__eq__(y) <==> x==y

        """
        return self._binary_operation(y, "__eq__", False)

    def __ne__(self, y):
        """The rich comparison operator ``!=``

        x.__ne__(y) <==> x!=y

        """
        return self._binary_operation(y, "__ne__", False)

    def __ge__(self, y):
        """The rich comparison operator ``>=``

        x.__ge__(y) <==> x>=y

        """
        return self._binary_operation(y, "__ge__", False)

    def __gt__(self, y):
        """The rich comparison operator ``>``

        x.__gt__(y) <==> x>y

        """
        return self._binary_operation(y, "__gt__", False)

    def __le__(self, y):
        """The rich comparison operator ``<=``

        x.__le__(y) <==> x<=y

        """
        return self._binary_operation(y, "__le__", False)

    def __lt__(self, y):
        """The rich comparison operator ``<``

        x.__lt__(y) <==> x<y

        """
        return self._binary_operation(y, "__lt__", False)

    def __and__(self, other):
        """The binary bitwise operation ``&``

        x.__and__(y) <==> x&y

        """
        return self._binary_operation(other, "__and__", False)

    def __iand__(self, other):
        """The augmented bitwise assignment ``&=``

        x.__iand__(y) <==> x&=y

        """
        return self._binary_operation(other, "__iand__", False)

    def __rand__(self, other):
        """The binary bitwise operation ``&`` with reflected operands.

        x.__rand__(y) <==> y&x

        """
        return self._binary_operation(other, "__rand__", False)

    def __or__(self, other):
        """The binary bitwise operation ``|``

        x.__or__(y) <==> x|y

        """
        return self._binary_operation(other, "__or__", False)

    def __ior__(self, other):
        """The augmented bitwise assignment ``|=``

        x.__ior__(y) <==> x|=y

        """
        return self._binary_operation(other, "__ior__", False)

    def __ror__(self, other):
        """The binary bitwise operation ``|`` with reflected operands.

        x.__ror__(y) <==> y|x

        """
        return self._binary_operation(other, "__ror__", False)

    def __xor__(self, other):
        """The binary bitwise operation ``^``

        x.__xor__(y) <==> x^y

        """
        return self._binary_operation(other, "__xor__", False)

    def __ixor__(self, other):
        """The augmented bitwise assignment ``^=``

        x.__ixor__(y) <==> x^=y

        """
        return self._binary_operation(other, "__ixor__", False)

    def __rxor__(self, other):
        """The binary bitwise operation ``^`` with reflected operands.

        x.__rxor__(y) <==> y^x

        """
        return self._binary_operation(other, "__rxor__", False)

    def __lshift__(self, y):
        """The binary bitwise operation ``<<``

        x.__lshift__(y) <==> x<<y

        """
        return self._binary_operation(y, "__lshift__", False)

    def __ilshift__(self, y):
        """The augmented bitwise assignment ``<<=``

        x.__ilshift__(y) <==> x<<=y

        """
        return self._binary_operation(y, "__ilshift__", False)

    def __rlshift__(self, y):
        """The binary bitwise operation ``<<`` with reflected operands.

        x.__rlshift__(y) <==> y<<x

        """
        return self._binary_operation(y, "__rlshift__", False)

    def __rshift__(self, y):
        """The binary bitwise operation ``>>``

        x.__lshift__(y) <==> x>>y

        """
        return self._binary_operation(y, "__rshift__", False)

    def __irshift__(self, y):
        """The augmented bitwise assignment ``>>=``

        x.__irshift__(y) <==> x>>=y

        """
        return self._binary_operation(y, "__irshift__", False)

    def __rrshift__(self, y):
        """The binary bitwise operation ``>>`` with reflected operands.

        x.__rrshift__(y) <==> y>>x

        """
        return self._binary_operation(y, "__rrshift__", False)

    def __abs__(self):
        """The unary arithmetic operation ``abs``

        x.__abs__() <==> abs(x)

        """
        return self._unary_operation("__abs__", bounds=True)

    def __neg__(self):
        """The unary arithmetic operation ``-``

        x.__neg__() <==> -x

        """
        return self._unary_operation("__neg__", bounds=True)

    def __invert__(self):
        """The unary bitwise operation ``~``

        x.__invert__() <==> ~x

        """
        return self._unary_operation("__invert__", bounds=True)

    def __pos__(self):
        """The unary arithmetic operation ``+``

        x.__pos__() <==> +x

        """
        return self._unary_operation("__pos__", bounds=True)

    def _binary_operation(self, other, method, bounds=True):
        """Implement binary arithmetic and comparison operations.

        The operations act on the construct's data array with the numpy
        broadcasting rules.

        If the construct has bounds then they are operated on with the
        same data as the construct's data.

        It is intended to be called by the binary arithmetic and comparison
        methods, such as `!__sub__` and `!__lt__`.

        **Bounds**

        The flag returned by ``cf.bounds_combination_mode()`` is used to
        influence whether or not the result of a binary operation "op(x,
        y)", such as ``x + y``, ``x -= y``, ``x << y``, etc., will contain
        bounds, and if so how those bounds are calculated.

        The behaviour for the different flag values is described in the
        docstring of `cf.bounds_combination_mode`.

        :Parameters:

            other:

            method: `str`
                The binary arithmetic or comparison method name (such as
                ``'__imul__'`` or ``'__ge__'``).

            bounds: `bool`, optional
                If False then ignore the bounds and remove them from the
                result. By default the bounds are operated on as described
                above.

        :Returns:

            `{{class}}`
                A new construct, or the same construct if the operation
                was in-place.

        """
        if getattr(other, "_NotImplemented_RHS_Data_op", False):
            return NotImplemented

        inplace = method[2] == "i"

        bounds_AND = bounds and bounds_combination_mode() == "AND"
        bounds_OR = (
            bounds and not bounds_AND and bounds_combination_mode() == "OR"
        )
        bounds_XOR = (
            bounds
            and not bounds_AND
            and not bounds_OR
            and bounds_combination_mode() == "XOR"
        )
        bounds_NONE = (
            not bounds
            or not (bounds_AND or bounds_OR or bounds_XOR)
            or bounds_combination_mode() == "NONE"
        )

        if not bounds_NONE:
            geometry = self.get_geometry(None)
            try:
                other_geometry = other.get_geometry(None)
            except AttributeError:
                other_geometry = None

            if geometry != other_geometry:
                raise ValueError(
                    "Can't combine operands with different geometry types"
                )

            interior_ring = self.get_interior_ring(None)
            try:
                other_interior_ring = other.get_interior_ring(None)
            except AttributeError:
                other_interior_ring = None

            if interior_ring is not None or other_interior_ring is not None:
                raise ValueError(
                    "Can't combine operands with interior ring arrays"
                )

        has_bounds = self.has_bounds()

        if bounds and has_bounds and inplace and other is self:
            other = other.copy()

        try:
            other_bounds = other.get_bounds(None)
        except AttributeError:
            other_bounds = None

        if (
            (bounds_OR or bounds_XOR)
            and not has_bounds
            and other_bounds is not None
        ):
            # --------------------------------------------------------
            # If self has no bounds but other does, then copy self for
            # use in constructing new bounds.
            # --------------------------------------------------------
            original_self = self.copy()

        new = super()._binary_operation(other, method)

        if bounds_NONE:
            # --------------------------------------------------------
            # Remove any bounds from the result
            # --------------------------------------------------------
            new.del_bounds(None)

        elif has_bounds and other_bounds is not None:
            if bounds_AND or bounds_OR:
                # ----------------------------------------------------
                # Both self and other have bounds, so combine them for
                # the result.
                # ----------------------------------------------------
                new_bounds = self.bounds._binary_operation(
                    other_bounds, method
                )

                if not inplace:
                    new.set_bounds(new_bounds, copy=False)

            elif bounds_XOR:
                # ----------------------------------------------------
                # Both self and other have bounds, so remove the
                # bounds from the result
                # ----------------------------------------------------
                new.del_bounds(None)

        elif bounds_AND:
            # --------------------------------------------------------
            # At most one of self and other has bounds, so remove the
            # bounds from the result.
            # --------------------------------------------------------
            new.del_bounds(None)

        elif has_bounds:
            # --------------------------------------------------------
            # Only self has bounds, so combine the self bounds with
            # the other values.
            # --------------------------------------------------------
            if cf_size(other) > 1:
                for i in range(self.bounds.ndim - self.ndim):
                    try:
                        other = other.insert_dimension(-1)
                    except AttributeError:
                        other = np.expand_dims(other, -1)

            new_bounds = self.bounds._binary_operation(other, method)

            if not inplace:
                new.set_bounds(new_bounds, copy=False)

        elif other_bounds is not None:
            # --------------------------------------------------------
            # Only other has bounds, so combine self values with the
            # other bounds
            # --------------------------------------------------------
            new_bounds = self._Bounds(data=original_self.data, copy=True)
            for i in range(other_bounds.ndim - other.ndim):
                new_bounds = new_bounds.insert_dimension(-1)

            if inplace:
                # Can't do the operation in-place because we'll run
                # foul of the broadcasting rules (e.g. "ValueError:
                # non-broadcastable output operand with shape (12,1)
                # doesn't match the broadcast shape (12,2)")
                method2 = method.replace("__i", "__", 1)
            else:
                method2 = method

            new_bounds = new_bounds._binary_operation(other_bounds, method2)
            new.set_bounds(new_bounds, copy=False)

        new._custom["direction"] = None
        return new

    @_manage_log_level_via_verbosity
    def _equivalent_data(self, other, rtol=None, atol=None, verbose=None):
        """True if data is equivalent to other data, units considered.

        Two real numbers ``x`` and ``y`` are considered equal if
        ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on absolute
        differences) and ``rtol`` (the tolerance on relative differences)
        are positive, typically very small numbers. See the *atol* and
        *rtol* parameters.

        :Parameters:

            atol: `float`, optional
                The tolerance on absolute differences between real
                numbers. The default value is set by the `atol` function.

            rtol: `float`, optional
                The tolerance on relative differences between real
                numbers. The default value is set by the `rtol` function.

        :Returns:

            `bool`

        """
        self_bounds = self.get_bounds(None)
        other_bounds = other.get_bounds(None)
        hasbounds = self_bounds is not None

        if hasbounds != (other_bounds is not None):
            # TODO: add traceback
            # TODO: improve message below

            if is_log_level_info(logger):
                logger.info(
                    "One has bounds, the other does not"
                )  # pragma: no cover

            return False

        try:
            direction0 = self.direction()
            direction1 = other.direction()
            if (
                direction0 != direction1
                and direction0 is not None
                and direction1 is not None
            ):
                other = other.flip()
        except AttributeError:
            pass

        # Compare the data arrays
        if not super()._equivalent_data(
            other, rtol=rtol, atol=atol, verbose=verbose
        ):
            if is_log_level_info(logger):
                # TODO: improve message below
                logger.info("Non-equivalent data arrays")  # pragma: no cover

            return False

        if hasbounds:
            # Compare the bounds
            if not self_bounds._equivalent_data(
                other_bounds, rtol=rtol, atol=atol, verbose=verbose
            ):
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Non-equivalent bounds "
                        f"data: {self_bounds.data!r}, {other_bounds.data!r}"
                    )  # pragma: no cover

                return False

        # Still here? Then the data are equivalent.
        return True

    def _YMDhms(self, attr):
        """Return some datetime component of the data array elements."""
        out = super()._YMDhms(attr)
        out.del_bounds(None)
        return out

    def _matching_values(self, value0, value1, units=False, basic=False):
        """Whether two values match.

        The definition of "match" depends on the types of *value0* and
        *value1*.

        :Parameters:

            value0:
                The first value to be matched.

            value1:
                The second value to be matched.

            units: `bool`, optional
                If True then the units must be the same for values to be
                considered to match. By default, units are ignored in the
                comparison.

        :Returns:

            `bool`
                Whether or not the two values match.

        """
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
                return self._equals(value1, value0, basic=basic)

        return False

    def _apply_superclass_data_oper(
        self,
        v,
        oper_name,
        oper_args=(),
        bounds=True,
        interior_ring=False,
        **oper_kwargs,
    ):
        """Define an operation that can be applied to the data array.

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

        """
        v = getattr(super(), oper_name)(*oper_args, **oper_kwargs)
        if v is None:  # from inplace operation in superclass method
            v = self

        # Now okay to mutate oper_kwargs as no longer needed in original form
        oper_kwargs.pop("inplace", None)
        if bounds:
            bounds = v.get_bounds(None)
            if bounds is not None:
                getattr(bounds, oper_name)(
                    *oper_args, inplace=True, **oper_kwargs
                )

        if interior_ring:
            interior_ring = v.get_interior_ring(None)
            if interior_ring is not None:
                getattr(interior_ring, oper_name)(
                    *oper_args, inplace=True, **oper_kwargs
                )

        return v

    def _unary_operation(self, method, bounds=True):
        """Implement unary arithmetic operations on the data array and
        bounds.

        :Parameters:

            method: `str`
                The unary arithmetic method name (such as "__abs__").

            bounds: `bool`, optional
                If False then ignore the bounds and remove them from the
                result. By default the bounds are operated on as well.

        :Returns:

            `{{class}}`
                A new construct, or the same construct if the operation
                was in-place.

        """
        new = super()._unary_operation(method)

        self_bounds = self.get_bounds(None)
        if self_bounds is not None:
            if bounds:
                new_bounds = self_bounds._unary_operation(method)
                new.set_bounds(new_bounds)
            else:
                new.del_bounds()

        return new

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def cellsize(self):
        """The cell sizes.

        If there are no cell bounds then the cell sizes are all zero.

        .. versionadded:: 2.0

        **Examples**

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

        """
        data = self.get_bounds_data(None, _fill_value=None)
        if data is not None:
            if data.shape[-1] != 2:
                raise ValueError(
                    "Can only calculate cell sizes from bounds when there are "
                    f"exactly two bounds per cell. Got {data.shape[-1]}"
                )

            out = abs(data[..., 1] - data[..., 0])
            out.squeeze(-1, inplace=True)
            return out
        else:
            data = self.get_data(None)
            if data is not None:
                # Convert to "difference" units
                #
                # TODO: Think about temperature units in relation to
                #       https://github.com/cf-convention/discuss/issues/101,
                #       whenever that issue is resolved.
                units = self.Units
                if units.isreftime:
                    units = Units(units._units_since_reftime)

                return Data.zeros(self.shape, units=units)

        raise AttributeError(
            "Can't get cell sizes when there are no bounds nor coordinate data"
        )

    @property
    def dtype(self):
        """Numpy data-type of the data array.

        .. versionadded:: 2.0

        **Examples**

        >>> c.dtype
        dtype('float64')
        >>> import numpy
        >>> c.dtype = numpy.dtype('float32')

        """
        data = self.get_data(None, _fill_value=False)
        if data is not None:
            return data.dtype

        bounds = self.get_bounds_data(None, _fill_value=None)
        if bounds is not None:
            return bounds.dtype

        raise AttributeError(
            f"{self.__class__.__name__} doesn't have attribute 'dtype'"
        )

    @dtype.setter
    def dtype(self, value):
        data = self.get_data(None, _fill_value=False)
        if data is not None:
            data.dtype = value

        bounds = self.get_bounds_data(None, _fill_value=None)
        if bounds is not None:
            bounds.dtype = value

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
        period = self.period()
        if period is not None:
            return True

        bounds = self.get_bounds(None)
        if bounds is not None:
            return bounds.period is not None

    #        return self._custom.get('period', None) is not None

    @property
    def lower_bounds(self):
        """The lower bounds of cells.

        If there are no cell bounds then the coordinates are used as the
        lower bounds.

        .. versionadded:: 2.0

        .. seealso:: `upper_bounds`

        **Examples**

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

        """
        data = self.get_bounds_data(None)
        if data is not None:
            out = data.minimum(-1)
            out.squeeze(-1, inplace=True)
            return out
        else:
            data = self.get_data(None)
            if data is not None:
                return data.copy()

        raise AttributeError(
            "Can't get lower bounds when there are no bounds nor coordinate "
            "data"
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
        #        return super().Units

        data = self.get_data(None, _fill_value=False)
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

        try:
            return self._custom["Units"]
        except KeyError:
            # if bounds is None:
            self._custom["Units"] = _units_None
            return _units_None

    #            else:
    #                try:
    #                    return bounds._custom['Units']
    #                except KeyError:
    #                    bounds._custom['Units'] = _units_None

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
        """The upper bounds of cells.

        If there are no cell bounds then the coordinates are used as the
        upper bounds.

        .. versionadded:: 2.0

        .. seealso:: `lower_bounds`

        **Examples**

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

        """
        data = self.get_bounds_data(None)
        if data is not None:
            out = data.maximum(-1)
            out.squeeze(-1, inplace=True)
            return out
        else:
            data = self.get_data(None)
            if data is not None:
                return data.copy()

        raise AttributeError(
            "Can't get upper bounds when there are no bounds nor coordinate "
            "data"
        )

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
        data = self.get_data(None, _fill_value=False)
        if data is not None:
            data.dtype = value

    @dtype.deleter
    def dtype(self):
        data = self.get_data(None, _fill_value=False)
        if data is not None:
            del data.dtype

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
        location = super().add_file_location(location)

        bounds = self.get_bounds(None)
        if bounds is not None:
            bounds.add_file_location(location)

        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            interior_ring.add_file_location(location)

        return location

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def ceil(self, bounds=True, inplace=False, i=False):
        """The ceiling of the data, element-wise.

        The ceiling of ``x`` is the smallest integer ``n``, such that
         ``n >= x``.

        .. versionadded:: 1.0

        .. seealso:: `floor`, `rint`, `trunc`

        :Parameters:

            bounds: `bool`, optional
                If False then do not alter any bounds. By default any
                bounds are also altered.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with the ceiling of the data. If the operation was
                in-place then `None` is returned.

        **Examples**

        >>> print(f.array)
        [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
        >>> print(f.ceil().array)
        [-1. -1. -1. -1.  0.  1.  2.  2.  2.]
        >>> f.ceil(inplace=True)
        >>> print(f.array)
        [-1. -1. -1. -1.  0.  1.  2.  2.  2.]

        """
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "ceil",
            bounds=bounds,
            inplace=inplace,
            i=i,
        )

    def cfa_clear_file_substitutions(
        self,
    ):
        """Remove all of the CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Returns:

            `dict`
                {{Returns cfa_clear_file_substitutions}}

        **Examples**

        >>> f.cfa_clear_file_substitutions()
        {}

        """
        out = super().cfa_clear_file_substitutions()

        bounds = self.get_bounds(None)
        if bounds is not None:
            out.update(bounds.cfa_clear_file_substitutions())

        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            out.update(interior_ring.cfa_clear_file_substitutions())

        return out

    def cfa_del_file_substitution(self, base):
        """Remove a CFA-netCDF file name substitution.

        .. versionadded:: 3.15.0

        :Parameters:

            {{cfa base: `str`}}

        :Returns:

            `dict`
                {{Returns cfa_del_file_substitution}}

        **Examples**

        >>> c.cfa_del_file_substitution('base')

        """
        super().cfa_del_file_substitution(base)

        bounds = self.get_bounds(None)
        if bounds is not None:
            bounds.cfa_del_file_substitution(base)

        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            interior_ring.cfa_del_file_substitution(base)

    def cfa_file_substitutions(self):
        """Return the CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Returns:

            `dict`
                {{Returns cfa_file_substitutions}}

        **Examples**

        >>> c.cfa_file_substitutions()
        {}

        """
        out = super().cfa_file_substitutions()

        bounds = self.get_bounds(None)
        if bounds is not None:
            out.update(bounds.cfa_file_substitutions({}))

        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            out.update(interior_ring.cfa_file_substitutions({}))

        return out

    def cfa_update_file_substitutions(self, substitutions):
        """Set CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Parameters:

            {{cfa substitutions: `dict`}}

        :Returns:

            `None`

        **Examples**

        >>> c.cfa_add_file_substitutions({'base', '/data/model'})

        """
        super().cfa_update_file_substitutions(substitutions)

        bounds = self.get_bounds(None)
        if bounds is not None:
            bounds.cfa_update_file_substitutions(substitutions)

        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            interior_ring.cfa_update_file_substitutions(substitutions)

    def chunk(self, chunksize=None):
        """Partition the data array.

        Deprecated at version 3.14.0. Use the `rechunk` method
        instead.

        :Parameters:

            chunksize: `int`, optional
                Set the new chunksize, in bytes.

        :Returns:

            `None`

        **Examples**

        >>> c.chunksize()

        >>> c.chunksize(1e8)

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
    def clip(
        self, a_min, a_max, units=None, bounds=True, inplace=False, i=False
    ):
        """Limit the values in the data.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "clip",
            (a_min, a_max),
            bounds=bounds,
            inplace=inplace,
            i=i,
            units=units,
        )

    def close(self):
        """Close all files referenced by the construct.

        Deprecated at version 3.14.0. All files are now
        automatically closed when not being accessed.

        Note that a closed file will be automatically re-opened if its
        contents are subsequently required.

        .. seealso:: `files`

        :Returns:

            `None`

        **Examples**

        >> c.close()

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

            variables: sequence of constructs

            axis: `int`, optional

            {{cull_graph: `bool`, optional}}

                .. versionadded:: 3.14.0

            {{relaxed_units: `bool`, optional}}

                .. versionadded:: 3.15.1

            copy: `bool`, optional
                If True (the default) then make copies of the
                {{class}} objects, prior to the concatenation, thereby
                ensuring that the input constructs are not changed by
                the concatenation process. If False then some or all
                input constructs might be changed in-place, but the
                concatenation process will be faster.

                .. versionadded:: 3.15.1

        :Returns:

            TODO

        """
        variable0 = variables[0]
        if copy:
            variable0 = variable0.copy()

        if len(variables) == 1:
            return variable0

        out = super().concatenate(
            variables,
            axis=axis,
            cull_graph=cull_graph,
            relaxed_units=relaxed_units,
            copy=copy,
        )

        bounds = variable0.get_bounds(None)
        if bounds is not None:
            bounds = bounds.concatenate(
                [v.get_bounds() for v in variables],
                axis=axis,
                cull_graph=cull_graph,
                relaxed_units=relaxed_units,
                copy=copy,
            )
            out.set_bounds(bounds, copy=False)

        interior_ring = variable0.get_interior_ring(None)
        if interior_ring is not None:
            interior_ring = interior_ring.concatenate(
                [v.get_interior_ring() for v in variables],
                axis=axis,
                cull_graph=cull_graph,
                relaxed_units=relaxed_units,
                copy=copy,
            )
            out.set_interior_ring(interior_ring, copy=False)

        return out

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def cos(self, bounds=True, inplace=False, i=False):
        """Take the trigonometric cosine of the data element-wise.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self), "cos", bounds=bounds
        )

    def cyclic(self, axes=None, iscyclic=True):
        """Get or set the cyclicity of axes of the data array.

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

        **Examples**

            TODO

        """
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
                        f"attributes: {x!r}, {y!r}"
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

    def contiguous(self, overlap=True):
        """Return True if a construct has contiguous cells.

        A construct is contiguous if its cell boundaries match up, or
        overlap, with the boundaries of adjacent cells.

        In general, it is only possible for a zero, 1 or 2 dimensional
        construct with bounds to be contiguous. A size 1 construct with
        any number of dimensions is always contiguous.

        An exception occurs if the construct is multidimensional and has
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

        **Examples**

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

        """
        bounds = self.get_bounds_data(None, _fill_value=None)
        if bounds is None:
            return False

        ndim = self.ndim
        nbounds = bounds.shape[-1]

        if self.size == 1:
            return True

        period = self.autoperiod().period()

        if ndim == 2:
            if nbounds != 4:
                raise ValueError(
                    f"Can't tell if {ndim}-d cells with {nbounds} vertices "
                    "are contiguous"
                )

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
            raise ValueError(f"Can't tell if {ndim}-d cells are contiguous")

        if nbounds != 2:
            raise ValueError(
                f"Can't tell if {ndim}-d cells with {nbounds} vertices "
                "are contiguous"
            )

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

        Conversion is done by decoding the reference times to
        date-time objects and then re-encoding them for the new units.

        Any conversions are possible, but this method is primarily for
        conversions which require a change in the date-times
        originally encoded. For example, use this method to
        reinterpret data values in units of "months" since a reference
        time to data values in "calendar months" since a reference
        time. This is often necessary when units of "calendar months"
        were intended but encoded as "months", which have special
        definition. See the note and examples below for more details.

        .. note:: It is recommended that the units "year" and "month"
                  be used with caution, as explained in the following
                  excerpt from the CF conventions: "The Udunits
                  package defines a year to be exactly 365.242198781
                  days (the interval between 2 successive passages of
                  the sun through vernal equinox). It is not a
                  calendar year. Udunits includes the following
                  definitions for years: a common_year is 365 days, a
                  leap_year is 366 days, a Julian_year is 365.25 days,
                  and a Gregorian_year is 365.2425 days. For similar
                  reasons the unit ``month``, which is defined to be
                  exactly year/12, should also be used with caution.

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
                  If the original units are ``'months since
                  2000-1-1'`` in the Gregorian calendar then the
                  default units to convert to are ``'days since
                  2000-1-1'`` in the Gregorian calendar.

            calendar_months: `bool`, optional
                If True then treat units of ``'months'`` as if they
                were calendar months (in whichever calendar is
                originally specified), rather than a 12th of the
                interval between 2 successive passages of the sun
                through vernal equinox (i.e. 365.242198781/12 days).

            calendar_years: `bool`, optional
                If True then treat units of ``'years'`` as if they
                were calendar years (in whichever calendar is
                originally specified), rather than the interval
                between 2 successive passages of the sun through
                vernal equinox (i.e. 365.242198781 days).

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "convert_reference_time",
            inplace=inplace,
            bounds=True,
            interior_ring=False,
            units=units,
            calendar_months=calendar_months,
            calendar_years=calendar_years,
        )

    def get_property(self, prop, default=ValueError(), bounds=False):
        """Get a CF property.

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
                property has not been set.

                {{default Exception}}

            bounds: `bool`
                TODO

        :Returns:

                The value of the named property or the default value, if
                set.

        **Examples**

        >>> f = cf.{{class}}()
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

        """
        out = super().get_property(prop, None)
        if out is not None:
            return out

        if bounds and self.has_bounds():
            out = self.get_bounds().get_property(prop, None)
            if out is not None:
                return out

        return super().get_property(prop, default)

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
        out = super().file_locations()

        bounds = self.get_bounds(None)
        if bounds is not None:
            out.update(bounds.file_locations())

        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            out.update(interior_ring.file_locations())

        return out

    @_inplace_enabled(default=False)
    def flatten(self, axes=None, inplace=False):
        """Flatten axes of the data.

        Any subset of the axes may be flattened.

        The shape of the data may change, but the size will not.

        The flattening is executed in row-major (C-style) order. For
        example, the array ``[[1, 2], [3, 4]]`` would be flattened
        across both dimensions to ``[1 2 3 4]``.

        .. versionadded:: 3.0.2

        .. seealso:: `insert_dimension`, `flip`, `swapaxes`, `transpose`

        :Parameters:

            axes: (sequence of) int or str, optional
                Select the axes.  By default all axes are
                flattened. The *axes* argument may be one, or a
                sequence, of:

                  * An internal axis identifier. Selects this axis.

                  * An integer. Selects the axis corresponding to the
                    given position in the list of axes of the data
                    array.

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
        # Note the 'axes' argument can change mid-method meaning it is
        # not possible to consolidate this method using a call to
        # _apply_superclass_data_operations, despite mostly the same
        # logic.
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

        >>> c.del_file_location('/data/model/')
        '/data/model'

        """
        location = super().del_file_location(location)

        bounds = self.get_bounds(None)
        if bounds is not None:
            bounds.del_file_location(location)

        interior_ring = self.get_interior_ring(None)
        if interior_ring is not None:
            interior_ring.del_file_location(location)

        return location

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def floor(self, bounds=True, inplace=False, i=False):
        """Floor the data array, element-wise.

        The floor of ``x`` is the largest integer ``n``, such that ``n <= x``.

        .. versionadded:: 1.0

        .. seealso:: `ceil`, `rint`, `trunc`

        :Parameters:

            bounds: `bool`, optional
                If False then do not alter any bounds. By default any
                bounds are also altered.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "floor",
            bounds=bounds,
            inplace=inplace,
            i=i,
        )

    def direction(self):
        """Return `None`, indicating that it is not specified whether
        the values are increasing or decreasing.

        .. versionadded:: 2.0

        :Returns:

            `None`

        **Examples**

        >>> c.direction()
        None

        """
        return

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
                The construct with masked elements.

        **Examples**

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

        Invalid values in any bounds are also masked.

        .. seealso:: `numpy.ma.masked_invalid`

        :Parameters:

            {{inplace: `bool`, optional}}

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
        # Set bounds to True to bypass 'if bounds' check in call:
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "masked_invalid",
            bounds=True,
            inplace=inplace,
        )

    def match_by_property(self, *mode, **properties):
        """Determine whether or not a variable satisfies conditions.

        Conditions may be specified on the variable's attributes and CF
        properties.

        :Parameters:

        :Returns:

            `bool`
                Whether or not the variable matches the given criteria.

        **Examples**

        TODO

        """
        _or = False
        if mode:
            if len(mode) > 1:
                raise ValueError("Can provide at most one positional argument")

            x = mode[0]
            if x == "or":
                _or = True
            elif x != "and":
                raise ValueError(
                    "Positional argument, if provided, must one of 'or', 'and'"
                )

        if not properties:
            return True

        self_properties = self.properties()

        ok = True
        for name, value0 in properties.items():
            value1 = self_properties.get(name)
            ok = self._matching_values(value0, value1, units=(name == "units"))

            if _or:
                if ok:
                    break
            elif not ok:
                break

        return ok

    def match_by_identity(self, *identities):
        """Determine whether or not a variable satisfies conditions.

        Conditions may be specified on the variable's attributes and CF
        properties.

        :Parameters:

        :Returns:

            `bool`
                Whether or not the variable matches the given criteria.

        **Examples**

            TODO

        """
        # Return all constructs if no identities have been provided
        if not identities:
            return True

        self_identities = self.identities()

        ok = False
        for value0 in identities:
            for value1 in self_identities:
                ok = self._matching_values(value0, value1, basic=True)
                if ok:
                    break

            if ok:
                break

        return ok

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def override_calendar(self, calendar, inplace=False, i=False):
        """Override the calendar of date-time units.

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

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

        TODO

        **Examples**

        TODO

        >>> g = f.override_calendar('noleap')

        """
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "override_calendar",
            (calendar,),
            bounds=True,
            interior_ring=False,
            inplace=inplace,
            i=i,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def override_units(self, units, inplace=False, i=False):
        """Override the units.

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

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

                TODO

        **Examples**

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

        """
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "override_units",
            (units,),
            bounds=True,
            interior_ring=False,
            inplace=inplace,
            i=i,
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

        The halo may be applied over a subset of the data dimensions and
        each dimension may have a different halo size (including
        zero). The halo region is populated with a copy of the proximate
        values from the original data.

        Corresponding axes expanded in the bounds, if present.

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

            depth: `int` or `dict`
                Specify the size of the halo for each axis.

                If *depth* is a non-negative `int` then this is the halo
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
                  dimensions ``depth=2, axes=[0, -1]`` or equivalently
                  ``depth={0: 2, -1: 2}``.

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

                The "X" and "Y" axes must be a subset of those
                identified by the *depth* or *axes* parameter.

                See the *fold_index* parameter.

                *Parameter example:*
                  Define the "X" and Y" axes by positions 2 and 1
                  respectively of the data: ``{'X': 2, 'Y': 1}``

            fold_index: `int`, optional
                Identify which index of the "Y" axis corresponds to the
                fold in "X" axis of a tripolar grid. The only valid values
                are ``-1`` for the last index, and ``0`` for the first
                index. By default it is assumed to be the last
                index. Ignored if *tripolar* is `None`.

            {{inplace: `bool`, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

            size: deprecated at version 3.14.0
                Use the *depth* parameter instead.

        :Returns:

            `{{class}}` or `None`
                The expanded data, or `None` if the operation was
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

        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "halo",
            bounds=True,
            interior_ring=True,
            inplace=inplace,
            depth=depth,
            axes=axes,
            tripolar=tripolar,
            fold_index=fold_index,
            verbose=verbose,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def flip(self, axes=None, inplace=False, i=False):
        """Flip (reverse the direction of) data dimensions.

        .. seealso:: `insert_dimension`, `squeeze`, `transpose`,
        `unsqueeze`

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

        bounds = v.get_bounds_data(None, _fill_value=False)
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

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def exp(self, bounds=True, inplace=False, i=False):
        """The exponential of the data, element-wise.

        .. seealso:: `log`

        :Parameters:

            bounds: `bool`, optional
                If False then do not alter any bounds. By default any
                bounds are also altered.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with the exponential of data values. If the
                operation was in-place then `None` is returned.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "exp",
            bounds=bounds,
            inplace=inplace,
            i=i,
        )

    def set_bounds(self, bounds, copy=True):
        """Set the bounds.

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

        **Examples**

        >>> import numpy
        >>> b = {{package}}.Bounds(data=numpy.arange(10).reshape(5, 2))
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

        """
        data = self.get_data(None, _fill_value=False)

        if data is not None and bounds.shape[: data.ndim] != data.shape:
            # Check shape
            raise ValueError(
                f"Can't set bounds: Incorrect bounds shape {bounds.shape} "
                f"for data shape {data.shape}"
            )

        if copy:
            bounds = bounds.copy()

        # Check units
        units = bounds.Units
        self_units = self.Units

        if data is not None and units:
            if not units.equivalent(self_units):
                raise ValueError(
                    f"Can't set bounds: {bounds!r} units of {bounds.Units!r} "
                    f"are not equivalent to {self.Units!r}, the units of "
                    f"{self!r}"
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

        self._custom["direction"] = None

        super().set_bounds(bounds, copy=False)

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def sin(self, bounds=True, inplace=False, i=False):
        """Take the trigonometric sine of the data element-wise.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "sin",
            bounds=bounds,
            inplace=inplace,
            i=i,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def arctan(self, bounds=True, inplace=False):
        """Take the trigonometric inverse tangent of the data element-
        wise.

        Units are ignored in the calculation. The result has units of radians.

        The "standard_name" and "long_name" properties are removed from
        the result.

        .. versionadded:: 3.0.7

        .. seealso:: `tan`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the trigonometric inverse tangent of data
                values. If the operation was in-place then `None` is returned.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arctan",
            inplace=inplace,
        )

    @_inplace_enabled(default=False)
    def arctanh(self, bounds=True, inplace=False):
        """Take the inverse hyperbolic tangent of the data element-wise.

        Units are ignored in the calculation. The result has units of radians.

        The "standard_name" and "long_name" properties are removed from
        the result.

        .. versionadded:: 3.2.0

        .. seealso:: `tanh`, `arcsinh`, `arccosh`, `arctan`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the inverse hyperbolic tangent of data
                values. If the operation was in-place then `None` is returned.

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
        >>> f.mask_invalid(inplace=True)
        >>> print(f.array)
        [-- -- 1.0986122886681098 0.6931471805599453 --]

        """
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arctanh",
            bounds=bounds,
            inplace=inplace,
        )

    @_inplace_enabled(default=False)
    def arcsin(self, bounds=True, inplace=False):
        """Take the trigonometric inverse sine of the data element-wise.

        Units are ignored in the calculation. The result has units of radians.

        The "standard_name" and "long_name" properties are removed from
        the result.

        .. versionadded:: 3.2.0

        .. seealso:: `sin`, `arccos`, `arctan`, `arcsinh`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the trigonometric inverse sine of data
                values. If the operation was in-place then `None` is returned.

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
        >>> f.mask_invalid(inplace=True)
        >>> print(f.array)
        [-- 1.5707963267948966 0.9272952180016123 0.6435011087932844 --]

        """
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arcsin",
            bounds=bounds,
            inplace=inplace,
        )

    @_inplace_enabled(default=False)
    def arcsinh(self, bounds=True, inplace=False):
        """Take the inverse hyperbolic sine of the data element-wise.

        Units are ignored in the calculation. The result has units of radians.

        The "standard_name" and "long_name" properties are removed from
        the result.

        .. versionadded:: 3.1.0

        .. seealso:: `sinh`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the inverse hyperbolic sine of data values.
                If the operation was in-place then `None` is returned.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arcsinh",
            bounds=bounds,
            inplace=inplace,
        )

    @_inplace_enabled(default=False)
    def arccos(self, bounds=True, inplace=False):
        """Take the trigonometric inverse cosine of the data element-
        wise.

        Units are ignored in the calculation. The result has units of radians.

        The "standard_name" and "long_name" properties are removed from
        the result.

        .. versionadded:: 3.2.0

        .. seealso:: `cos`, `arcsin`, `arctan`, `arccosh`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the trigonometric inverse cosine of data
                values. If the operation was in-place then `None` is returned.

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
        >>> f.mask_invalid(inplace=True)
        >>> print(f.array)
        [-- 0.0 0.6435011087932843 0.9272952180016123 --]

        """
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arccos",
            bounds=bounds,
            inplace=inplace,
        )

    @_inplace_enabled(default=False)
    def arccosh(self, bounds=True, inplace=False):
        """Take the inverse hyperbolic cosine of the data element-wise.

        Units are ignored in the calculation. The result has units of
        radians.

        The "standard_name" and "long_name" properties are removed from
        the result.

        .. versionadded:: 3.2.0

        .. seealso:: `cosh`, `arcsinh`, `arctanh`, `arccos`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the inverse hyperbolic cosine of data
                values. If the operation was in-place then `None` is
                returned.

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
        >>> f.mask_invalid(inplace=True)
        >>> print(f.array)
        [0.6223625037147786 0.0 -- -- --]

        """
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "arccosh",
            bounds=bounds,
            inplace=inplace,
        )

    @_inplace_enabled(default=False)
    def tanh(self, bounds=True, inplace=False):
        """Take the hyperbolic tangent of the data element-wise.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "tanh",
            bounds=bounds,
            inplace=inplace,
        )

    @_inplace_enabled(default=False)
    def sinh(self, bounds=True, inplace=False):
        """Take the hyperbolic sine of the data element-wise.

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

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with the hyperbolic sine of data values. If
                the operation was in-place then `None` is returned.

        **Examples**

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

        """
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "sinh",
            bounds=bounds,
            inplace=inplace,
        )

    @_inplace_enabled(default=False)
    def cosh(self, bounds=True, inplace=False):
        """Take the hyperbolic cosine of the data element-wise.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "cosh",
            bounds=bounds,
            inplace=inplace,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def tan(self, bounds=True, inplace=False, i=False):
        """Take the trigonometric tangent of the data element-wise.

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

            {{inplace: `bool`, optional}}

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "tan",
            inplace=inplace,
            i=i,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def log(self, base=None, bounds=True, inplace=False, i=False):
        """The logarithm of the data array.

        By default the natural logarithm is taken, but any base may be
        specified.

        .. seealso:: `exp`

        :Parameters:

            base: number, optional
                The base of the logarithm. By default a natural logarithm
                is taken.

            bounds: `bool`, optional
                If False then do not alter any bounds. By default any
                bounds are also altered.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with the logarithm of data values. If the
                operation was in-place then `None` is returned.

        **Examples**

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

        """
        # TODO: 'base' kwarg not used? why?
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "log",
            (base,),
            bounds=bounds,
            inplace=inplace,
            i=i,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def squeeze(self, axes=None, inplace=False, i=False):
        """Remove size one axes from the data array.

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

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The new construct with removed data axes. If the operation
                was in-place then `None` is returned.

        **Examples**

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

        """
        return super().squeeze(axes=axes, inplace=inplace)

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def trunc(self, bounds=True, inplace=False, i=False):
        """Truncate the data, element-wise.

        The truncated value of the scalar ``x``, is the nearest integer
        ``i`` which is closer to zero than ``x`` is. I.e. the fractional
        part of the signed number ``x`` is discarded.

        .. versionadded:: 1.0

        .. seealso:: `ceil`, `floor`, `rint`

        :Parameters:

            bounds: `bool`, optional
                If False then do not alter any bounds. By default any
                bounds are also altered.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                The construct with truncated data. If the operation was
                in-place then `None` is returned.

        **Examples**

        >>> print(f.array)
        [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
        >>> print(f.trunc().array)
        [-1. -1. -1. -1.  0.  1.  1.  1.  1.]
        >>> f.trunc(inplace=True)
        >>> print(f.array)
        [-1. -1. -1. -1.  0.  1.  1.  1.  1.]

        """
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "trunc",
            bounds=bounds,
            inplace=inplace,
            i=i,
        )

    #   def identities(self, generator=False):
    #       """Return all possible identities.
    #
    #       The identities comprise:
    #
    #       * The "standard_name" property.
    #       * The "id" attribute, preceded by ``'id%'``.
    #       * The "cf_role" property, preceded by ``'cf_role='``.
    #       * The "axis" property, preceded by ``'axis='``.
    #       * The "long_name" property, preceded by ``'long_name='``.
    #       * All other properties (including "standard_name"), preceded by
    #         the property name and an ``'='``.
    #       * The coordinate type (``'X'``, ``'Y'``, ``'Z'`` or ``'T'``).
    #       * The netCDF variable name, preceded by ``'ncvar%'``.
    #
    #       The identities of the bounds, if present, are included (with the
    #       exception of the bounds netCDF variable name).
    #
    #       .. versionadded:: 3.0.0
    #
    #       .. seealso:: `id`, `identity`
    # ODO
    #       :Returns:
    #
    #           `list`
    #               The identities.
    #
    #       **Examples**
    #
    #       >>> f.properties()
    #       {'foo': 'bar',
    #        'long_name': 'Air Temperature',
    #        'standard_name': 'air_temperature'}
    #       >>> f.nc_get_variable()
    #       'tas'
    #       >>> f.identities()
    #       ['air_temperature',
    #        'long_name=Air Temperature',
    #        'foo=bar',
    #        'standard_name=air_temperature',
    #        'ncvar%tas']
    #
    #       >>> f.properties()
    #       {}
    #       >>> f.bounds.properties()
    #       {'axis': 'Z',
    #        'units': 'm'}
    #       >>> f.identities()
    #       ['axis=Z', 'units=m', 'ncvar%z']
    #
    #       """
    #       identities = super().identities()
    #
    #       bounds = self.get_bounds(None)
    #       if bounds is not None:
    #           identities.extend(
    #               [i for i in bounds.identities() if i not in identities]
    #           )
    #       # TODO ncvar AND?
    #
    #       return identities

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

        >>> f.properties()
        {}
        >>> f.bounds.properties()
        {'axis': 'Z',
         'units': 'm'}
        >>> f.identity()
        'axis=Z'

        """
        identity = super().identity(
            default=None,
            strict=strict,
            relaxed=relaxed,
            nc_only=nc_only,
            _ctype=_ctype,
        )

        # TODO: when coord has no standard name but bounds do - that standard name needs to be picked up.

        if identity is not None:
            return identity

        bounds = self.get_bounds(None)
        if bounds is not None:
            out = bounds.identity(
                default=None, strict=strict, relaxed=relaxed, nc_only=nc_only
            )

            if out is not None and not out.startswith("ncvar%"):
                return out

        return default

    def inspect(self):
        """Inspect the object for debugging.

        .. seealso:: `cf.inspect`

        :Returns:

            `None`

        """
        print(cf_inspect(self))  # pragma: no cover

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "pad_missing",
            bounds=True,
            interior_ring=True,
            axis=axis,
            pad_width=pad_width,
            to_size=to_size,
            inplace=inplace,
        )

    def period(self, *value, **config):
        """Return or set the period for cyclic values.

        .. seealso:: `cyclic`

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
        old = super().period(*value, **config)

        if old is not None:
            return old

        bounds = self.get_bounds(None)
        if bounds is None:
            return

        return bounds.period(*value, **config)

    @_inplace_enabled(default=False)
    def persist(self, bounds=True, inplace=False):
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

            bounds: `bool`, optional
                If False then do not persist any bounds data. By
                default any bound data are also persisted.

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The construct with persisted data. If the operation
                was in-place then `None` is returned.

        **Examples**

        >>> g = f.persist()

        """
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "persist",
            bounds=bounds,
            interior_ring=True,
            inplace=inplace,
        )

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

            bounds: `bool`, optional
                If True (the default) then rechunk the bounds, if
                they exist.

            interior_ring: `bool`, optional
                If True (the default) then rechunk an interior ring
                array, if one exists.

        :Returns:

            `{{class}}` or `None`
                The construct with rechunked data, or `None` if the
                operation was in-place.

        **Examples**

        See `cf.Data.rechunk` for examples.

        """
        if (bounds or interior_ring) and isinstance(chunks, dict):
            from dask.array.utils import validate_axis

            ndim = self.ndim
            chunks = {validate_axis(c, ndim): v for c, v in chunks.items()}

        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "rechunk",
            bounds=bounds,
            interior_ring=interior_ring,
            inplace=inplace,
            chunks=chunks,
            threshold=threshold,
            block_size_limit=block_size_limit,
            balance=balance,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def rint(self, bounds=True, inplace=False, i=False):
        """Round the data to the nearest integer, element-wise.

        .. versionadded:: 1.0

        .. seealso:: `ceil`, `floor`, `trunc`

        :Parameters:

            bounds: `bool`, optional
                If False then do not alter any bounds. By default any
                bounds are also altered.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "rint",
            bounds=bounds,
            inplace=inplace,
            i=i,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def round(self, decimals=0, bounds=True, inplace=False, i=False):
        """Round the data to the given number of decimals.

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
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "round",
            bounds=bounds,
            inplace=inplace,
            i=i,
            decimals=decimals,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def roll(self, iaxis, shift, inplace=False, i=False):
        """Roll the data along an axis.

        .. seealso:: `insert_dimension`, `flip`, `squeeze`, `transpose`

        :Parameters:

            iaxis: `int`
                TODO

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `{{class}}` or `None`
                TODO

        **Examples**

        TODO

        """
        return self._apply_superclass_data_oper(
            _inplace_enabled_define_and_cleanup(self),
            "roll",
            (iaxis, shift),
            bounds=True,
            interior_ring=True,
            inplace=inplace,
            i=i,
        )

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    @property
    def hasbounds(self):
        """Deprecated at version 3.0.0, use method `has_bounds`
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "hasbounds",
            "Use method 'has_bounds' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def expand_dims(self, position=0, i=False):
        """Deprecated at version 3.0.0, use method `insert_dimension`
        instead."""
        _DEPRECATION_ERROR_METHOD(
            self,
            "expand_dims",
            "Use method 'insert_dimension' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def files(self):
        """Deprecated at version 3.4.0, consider using the
        `get_original_filenames` method instead."""
        _DEPRECATION_ERROR_METHOD(
            self,
            "files",
            "Consider using the 'get_original_filenames' method instead.",
            version="3.4.0",
            removed_at="4.0.0",
        )  # pragma: no cover
