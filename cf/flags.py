import logging
from copy import deepcopy

import numpy as np
from cfdm import is_log_level_info

from .decorators import (
    _deprecated_kwarg_check,
    _display_or_return,
    _manage_log_level_via_verbosity,
)
from .functions import atol as cf_atol
from .functions import equals as cf_equals
from .functions import inspect as cf_inspect
from .functions import rtol as cf_rtol

logger = logging.getLogger(__name__)


class Flags:
    """Self-describing CF flag values.

    Stores the flag_values, flag_meanings and flag_masks CF attributes
    in an internally consistent manner.

    """

    def __init__(self, **kwargs):
        """**Initialisation**

        :Parameters:

            flag_values : optional
                The flag_values CF property. Sets the `flag_values`
                attribute.

            flag_meanings : optional
                The flag_meanings CF property. Sets the `flag_meanings`
                attribute.

            flag_masks : optional
                The flag_masks CF property. Sets the `flag_masks`
                attribute.

        """
        for attr, value in kwargs.items():
            if value is not None:
                setattr(self, attr, value)

    def __eq__(self, other):
        """x.__eq__(y) <==> x==y <==> x.equals(y)"""
        return self.equals(other)

    def __ne__(self, other):
        """x.__ne__(y) <==> x!=y <==> not x.equals(y)"""
        return not self.equals(other)

    def __hash__(self):
        """Return the hash value of the flags.

        Note that the flags will be sorted in place.

        :Returns:

            `int`
                The hash value.

        **Examples**

        >>> hash(f)
        -956218661958673979

        """
        self.sort()

        x = [
            tuple(getattr(self, attr, ()))
            for attr in ("_flag_meanings", "_flag_values", "_flag_masks")
        ]

        return hash(tuple(x))

    def __bool__(self):
        """x.__bool__() <==> x!=0."""
        for attr in ("_flag_meanings", "_flag_values", "_flag_masks"):
            if hasattr(self, attr):
                return True

        return False

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def flag_values(self):
        """The flag_values CF attribute.

        Stored as a 1-d numpy array but may be set as any array-like
        object.

        **Examples**

        >>> f.flag_values = ['a', 'b', 'c']
        >>> f.flag_values
        array(['a', 'b', 'c'], dtype='|S1')
        >>> f.flag_values = numpy.arange(4, dtype='int8')
        >>> f.flag_values
        array([1, 2, 3, 4], dtype=int8)
        >>> f.flag_values = 1
        >>> f.flag_values
        array([1])

        """
        try:
            return self._flag_values
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} has no attribute "
                "'flag_values'"
            )

    @flag_values.setter
    def flag_values(self, value):
        if not isinstance(value, np.ndarray):
            value = np.atleast_1d(value)
        self._flag_values = value

    @flag_values.deleter
    def flag_values(self):
        try:
            del self._flag_values
        except AttributeError:
            raise AttributeError(
                f"Can't delete {self.__class__.__name__!r} attribute "
                "'flag_values'"
            )

    # ----------------------------------------------------------------
    # Property attribute: flag_masks
    # ----------------------------------------------------------------
    @property
    def flag_masks(self):
        """The flag_masks CF attribute.

        Stored as a 1-d numpy array but may be set as array-like object.

        **Examples**

        >>> f.flag_masks = numpy.array([1, 2, 4], dtype='int8')
        >>> f.flag_masks
        array([1, 2, 4], dtype=int8)
        >>> f.flag_masks = 1
        >>> f.flag_masks
        array([1])

        """
        try:
            return self._flag_masks
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute "
                "'flag_masks'"
            )

    @flag_masks.setter
    def flag_masks(self, value):
        if not isinstance(value, np.ndarray):
            value = np.atleast_1d(value)

        self._flag_masks = value

    @flag_masks.deleter
    def flag_masks(self):
        try:
            del self._flag_masks
        except AttributeError:
            raise AttributeError(
                f"Can't delete {self.__class__.__name__!r} attribute "
                "'flag_masks'"
            )

    @property
    def flag_meanings(self):
        """The flag_meanings CF attribute.

        Stored as a 1-d numpy string array but may be set as a space
        delimited string or any array-like object.

        **Examples**

        >>> f.flag_meanings = 'low medium      high'
        >>> f.flag_meanings
        array(['low', 'medium', 'high'],
              dtype='|S6')
        >>> f.flag_meanings = ['left', 'right']
        >>> f.flag_meanings
        array(['left', 'right'],
              dtype='|S5')
        >>> f.flag_meanings = 'ok'
        >>> f.flag_meanings
        array(['ok'],
              dtype='|S2')
        >>> f.flag_meanings = numpy.array(['a', 'b'])
        >>> f.flag_meanings
        array(['a', 'b'],
              dtype='|S1')

        """
        try:
            return self._flag_meanings
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute "
                "'flag_meanings'"
            )

    @flag_meanings.setter
    def flag_meanings(self, value):
        if isinstance(value, str):
            value = np.atleast_1d(value.split())
        elif not isinstance(value, np.ndarray):
            value = np.atleast_1d(value)

        self._flag_meanings = value

    @flag_meanings.deleter
    def flag_meanings(self):
        try:
            del self._flag_meanings
        except AttributeError:
            raise AttributeError(
                f"Can't delete {self.__class__.__name__!r} attribute "
                "'flag_meanings'"
            )

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        string = []
        if hasattr(self, "flag_values"):
            string.append(f"flag_values={self.flag_values}")

        if hasattr(self, "flag_masks"):
            string.append(f"flag_masks={self.flag_masks}")

        if hasattr(self, "flag_meanings"):
            string.append(f"flag_meanings={self.flag_meanings}")

        x = ", ".join(string)
        return f"<CF {self.__class__.__name__}: {x}>"

    def copy(self):
        """Return a deep copy.

        Equivalent to ``copy.deepcopy(f)``

        :Returns:

                The deep copy.

        **Examples**

        >>> f.copy()

        """
        return deepcopy(self)

    @_display_or_return
    def dump(self, display=True, _level=0):
        """Return a string containing a full description of the
        instance.

        :Parameters:

            display : bool, optional
                If False then return the description as a string. By
                default the description is printed, i.e. ``f.dump()`` is
                equivalent to ``print(f.dump(display=False))``.

        :Returns:

            `None` or `str`
                A string containing the description.

        """
        indent0 = "    " * _level
        indent1 = "    " * (_level + 1)

        string = [f"{indent0}Flags:"]

        for attr in ("_flag_values", "_flag_meanings", "_flag_masks"):
            value = getattr(self, attr, None)
            if value is not None:
                string.append(f"{indent1}{attr[1:]} = {list(value)}")

        return "\n".join(string)

    @_deprecated_kwarg_check("traceback", version="3.0.0", removed_at="4.0.0")
    @_manage_log_level_via_verbosity
    def equals(
        self,
        other,
        rtol=None,
        atol=None,
        ignore_fill_value=False,
        verbose=None,
        traceback=False,
    ):
        """True if two groups of flags are logically equal, False
        otherwise.

        Note that both instances are sorted in place prior to the comparison.

        :Parameters:

            other:
                The object to compare for equality.

            atol: float, optional
                The absolute tolerance for all numerical comparisons, By
                default the value returned by the `atol` function is used.

            rtol: float, optional
                The relative tolerance for all numerical comparisons, By
                default the value returned by the `rtol` function is used.

            ignore_fill_value: bool, optional
                If True then data arrays with different fill values are
                considered equal. By default they are considered unequal.

            traceback: deprecated at version 3.0.0.
                Use *verbose* instead.

        :Returns:

            `bool`
                Whether or not the two instances are equal.

        **Examples**

        >>> f
        <CF Flags: flag_values=[1 0 2], flag_masks=[2 0 2], flag_meanings=['medium' 'low' 'high']>
        >>> g
        <CF Flags: flag_values=[2 0 1], flag_masks=[2 0 2], flag_meanings=['high' 'low' 'medium']>
        >>> f.equals(g)
        True
        >>> f
        <CF Flags: flag_values=[0 1 2], flag_masks=[0 2 2], flag_meanings=['low' 'medium' 'high']>
        >>> g
        <CF Flags: flag_values=[0 1 2], flag_masks=[0 2 2], flag_meanings=['low' 'medium' 'high']>

        """
        # Check that each instance is the same type
        if self.__class__ != other.__class__:
            if is_log_level_info(logger):
                cls = self.__class__.__name__
                logger.info(
                    f"{cls}: Different type: "
                    f"{cls}, {other.__class__.__name__}"
                )  # pragma: no cover

            return False

        self.sort()
        other.sort()

        # Set default tolerances
        if rtol is None:
            rtol = float(cf_rtol())

        if atol is None:
            atol = float(cf_atol())

        for attr in ("_flag_meanings", "_flag_values", "_flag_masks"):
            if hasattr(self, attr):
                if not hasattr(other, attr):
                    if is_log_level_info(logger):
                        logger.info(
                            f"{self.__class__.__name__}: "
                            f"Different attributes: {attr[1:]}"
                        )  # pragma: no cover

                    return False

                x = getattr(self, attr)
                y = getattr(other, attr)

                if x.shape != y.shape or not cf_equals(
                    x,
                    y,
                    rtol=rtol,
                    atol=atol,
                    ignore_fill_value=ignore_fill_value,
                    verbose=verbose,
                ):
                    if is_log_level_info(logger):
                        logger.info(
                            f"{self.__class__.__name__}: Different "
                            f"{attr[1:]!r}: {x!r}, {y!r}"
                        )  # pragma: no cover

                    return False

            elif hasattr(other, attr):
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Different attributes: "
                        f"{attr[1:]}"
                    )  # pragma: no cover

                return False

        return True

    def inspect(self):
        """Inspect the object for debugging.

        .. seealso:: `cf.inspect`

        :Returns:

            `None`

        """
        print(cf_inspect(self))  # pragma: no cover

    def sort(self):
        """Sort the flags in place.

        By default sort by flag values. If flag values are not present
        then sort by flag meanings. If flag meanings are not present then
        sort by flag_masks.

        :Returns:

            `None`

        **Examples**

        >>> f
        <CF Flags: flag_values=[2 0 1], flag_masks=[2 0 2], flag_meanings=['high' 'low' 'medium']>
        >>> f.sort()
        >>> f
        <CF Flags: flag_values=[0 1 2], flag_masks=[0 2 2], flag_meanings=['low' 'medium' 'high']>

        """
        if not self:
            return

        # Sort all three attributes
        for attr in ("flag_values", "_flag_meanings", "_flag_masks"):
            if hasattr(self, attr):
                indices = np.argsort(getattr(self, attr))
                break

        for attr in ("_flag_values", "_flag_meanings", "_flag_masks"):
            if hasattr(self, attr):
                array = getattr(self, attr).view()
                array[...] = array[indices]
