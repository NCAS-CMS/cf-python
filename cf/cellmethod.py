import logging
import re
from ast import literal_eval as ast_literal_eval

import cfdm
from cfdm import is_log_level_info

from .data import Data
from .data.utils import conform_units
from .decorators import (
    _deprecated_kwarg_check,
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
    _manage_log_level_via_verbosity,
)
from .functions import _DEPRECATION_ERROR_METHOD
from .functions import inspect as cf_inspect

logger = logging.getLogger(__name__)


_collapse_cell_methods = {
    "max": "maximum",
    "mean": "mean",
    "mid_range": "mid_range",
    "min": "minimum",
    "range": "range",
    "sd": "standard_deviation",
    "sum": "sum",
    "var": "variance",
    "sample_size": None,
    "sum_of_weights": None,
    "sum_of_weights2": None,
}


class CellMethod(cfdm.CellMethod):
    """A cell method construct of the CF data model.

    One or more cell method constructs describe how the cell values of
    the field construct represent the variation of the physical
    quantity within its cells, i.e. the structure of the data at a
    higher resolution.

    A single cell method construct consists of a set of axes, a
    "method" property which describes how a value of the field
    construct's data array describes the variation of the quantity
    within a cell over those axes (e.g. a value might represent the
    cell area average), and descriptive qualifiers serving to indicate
    more precisely how the method was applied (e.g. recording the
    spacing of the original data, or the fact that the method was
    applied only over El Nino years).

    """

    def __new__(cls, *args, **kwargs):
        """This must be overridden in subclasses.

        .. versionadded:: (cfdm) 3.7.0

        """
        instance = super().__new__(cls)
        instance._Data = Data
        return instance

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)

    @classmethod
    def create(cls, cell_methods_string=None):
        """Parse a CF-like cell_methods string.

        :Parameters:

            cell_methods_string: `str`
                A CF cell_methods string.

        :Returns:

            `list`

        **Examples**

        >>> c = CellMethod.create('lat: mean (interval: 1 hour)')

        """
        incorrect_interval = "Cell method interval is incorrectly formatted"

        out = []

        if not cell_methods_string:
            return out

        # ------------------------------------------------------------
        # Split the cell_methods string into a list of strings ready
        # for parsing. For example:
        #
        #   'lat: mean (interval: 1 hour)'
        #
        # would be split up into:
        #
        #   ['lat:', 'mean', '(', 'interval:', '1', 'hour', ')']
        # ------------------------------------------------------------
        cell_methods = re.sub(r"\((?=[^\s])", "( ", cell_methods_string)
        cell_methods = re.sub(r"(?<=[^\s])\)", " )", cell_methods).split()

        while cell_methods:
            cm = cls()

            axes = []
            while cell_methods:
                if not cell_methods[0].endswith(":"):
                    break

                # TODO Check that "name" ends with colon? How? ('lat:
                #      mean (area-weighted) or lat: mean (interval: 1
                #      degree_north comment: area-weighted)')

                axis = cell_methods.pop(0)[:-1]

                axes.append(axis)

            cm.set_axes(axes)

            if not cell_methods:
                out.append(cm)
                break

            # Method
            cm.set_method(cell_methods.pop(0))

            if not cell_methods:
                out.append(cm)
                break

            # Climatological statistics, and statistics which apply to
            # portions of cells
            while cell_methods[0] in ("within", "where", "over"):
                attr = cell_methods.pop(0)
                cm.set_qualifier(attr, cell_methods.pop(0))
                if not cell_methods:
                    break

            if not cell_methods:
                out.append(cm)
                break

            # interval and comment
            intervals = []
            if cell_methods[0].endswith("("):
                cell_methods.pop(0)

                if not (re.search("^(interval|comment):$", cell_methods[0])):
                    cell_methods.insert(0, "comment:")

                while not re.search(r"^\)$", cell_methods[0]):
                    term = cell_methods.pop(0)[:-1]

                    if term == "interval":
                        interval = cell_methods.pop(0)
                        if cell_methods[0] != ")":
                            units = cell_methods.pop(0)
                        else:
                            units = None

                        try:
                            parsed_interval = ast_literal_eval(interval)
                        except (SyntaxError, ValueError):
                            raise ValueError(
                                f"{incorrect_interval}: {interval!r}"
                            )

                        try:
                            data = Data(
                                array=parsed_interval, units=units, copy=False
                            )
                        except Exception:
                            raise ValueError(
                                f"{incorrect_interval}: {interval!r}"
                            )

                        intervals.append(data)
                        continue

                    if term == "comment":
                        comment = []
                        while cell_methods:
                            if cell_methods[0].endswith(")"):
                                break
                            if cell_methods[0].endswith(":"):
                                break
                            comment.append(cell_methods.pop(0))

                        cm.set_qualifier("comment", " ".join(comment))

                if cell_methods[0].endswith(")"):
                    cell_methods.pop(0)

            n_intervals = len(intervals)
            if n_intervals > 1 and n_intervals != len(axes):
                raise ValueError(
                    f"{incorrect_interval} (doesn't match axes): {interval!r}"
                )

            if intervals:
                cm.set_qualifier("interval", intervals)

            out.append(cm)

        return out

    def __hash__(self):
        """x.__hash__() <==> hash(x)"""
        return hash(str(self))

    def __eq__(self, y):
        """x.__eq__(y) <==> x==y."""
        return self.equals(y)

    def __ne__(self, other):
        """x.__ne__(y) <==> x!=y."""
        return not self.__eq__(other)

    @property
    def within(self):
        """The cell method's within qualifier.

        These describe how climatological statistics have been derived.

        .. seealso:: `over`

        **Examples**

        >>> c
        >>> c
        <CF CellMethod: time: minimum>
        >>> print(c.within)
        None
        >>> c.within = 'years'
        >>> c
        <CF CellMethod: time: minimum within years>
        >>> del c.within
        >>> c
        <CF CellMethod: time: minimum>

        """
        return self.get_qualifier("within", default=AttributeError())

    @within.setter
    def within(self, value):
        self.set_qualifier("within", value)

    @within.deleter
    def within(self):
        self.del_qualifier("within", default=AttributeError())

    @property
    def where(self):
        """The cell method's where qualifier.

        These describe how climatological statistics have been derived.

        .. seealso:: `over`

        **Examples**

        >>> c
        >>> c
        <CF CellMethod: time: minimum>
        >>> print(c.where)
        None
        >>> c.where = 'land'
        >>> c
        <CF CellMethod: time: minimum where years>
        >>> del c.where
        >>> c
        <CF CellMethod: time: minimum>

        """
        return self.get_qualifier("where", default=AttributeError())

    @where.setter
    def where(self, value):
        self.set_qualifier("where", value)

    @where.deleter
    def where(self):
        self.del_qualifier("where", default=AttributeError())

    @property
    def over(self):
        """The cell method's over qualifier.

        These describe how climatological statistics have been derived.

        .. seealso:: `within`

        **Examples**

        >>> c
        >>> c
        <CF CellMethod: time: minimum>
        >>> print(c.over)
        None
        >>> c.over = 'years'
        >>> c
        <CF CellMethod: time: minimum over years>
        >>> del c.over
        >>> c
        <CF CellMethod: time: minimum>

        """
        return self.get_qualifier("over", default=AttributeError())

    @over.setter
    def over(self, value):
        self.set_qualifier("over", value)

    @over.deleter
    def over(self):
        self.del_qualifier("over", default=AttributeError())

    @property
    def comment(self):
        """The cell method's comment qualifier."""
        return self.get_qualifier("comment", default=AttributeError())

    @comment.setter
    def comment(self, value):
        self.set_qualifier("comment", value)

    @comment.deleter
    def comment(self):
        self.del_qualifier("comment", default=AttributeError())

    @property
    def method(self):
        """The cell method's method qualifier.

        Describes how the cell values have been determined or derived.

        **Examples**

        >>> c
        <CF CellMethod: time: minimum>
        >>> c.method
        'minimum'
        >>> c.method = 'variance'
        >>> c
        <CF CellMethods: time: variance>
        >>> del c.method
        >>> c
        <CF CellMethod: time: >

        """
        return self.get_method(default=AttributeError())

    @method.setter
    def method(self, value):
        self.set_method(value)

    @method.deleter
    def method(self):
        self.del_method(default=AttributeError())

    @property
    def intervals(self):
        """The cell method's interval qualifier(s).

        **Examples**

        >>> c
        <CF CellMethod: time: minimum>
        >>> c.intervals
        ()
        >>> c.intervals = ['1 hr']
        >>> c
        <CF CellMethod: time: minimum (interval: 1 hr)>
        >>> c.intervals
        (<CF Data: 1 hr>,)
        >>> c.intervals = [cf.Data(7.5, 'minutes')]
        >>> c
        <CF CellMethod: time: minimum (interval: 7.5 minutes)>
        >>> c.intervals
        (<CF Data: 7.5 minutes>,)
        >>> del c.intervals
        >>> c
        <CF CellMethods: time: minimum>

        >>> c
        <CF CellMethod: lat: lon: mean>
        >>> c.intervals = ['0.2 degree_N', cf.Data(0.1, 'degree_E')]
        >>> c
        <CF CellMethod: lat: lon: mean (interval: 0.1 degree_N interval: 0.2 degree_E)>

        """
        return self.get_qualifier("interval", default=AttributeError())

    @intervals.setter
    def intervals(self, value):
        if not isinstance(value, (tuple, list)):
            raise ValueError(
                "intervals attribute must be a tuple or list, not "
                f"{value.__class__.__name__!r}"
            )

        # Parse the intervals
        values = []
        for interval in value:
            if isinstance(interval, str):
                i = interval.split()

                try:
                    x = ast_literal_eval(i.pop(0))
                except Exception:
                    raise ValueError(f"Unparseable interval: {interval!r}")

                if interval:
                    units = " ".join(i)
                else:
                    units = None

                try:
                    d = Data(x, units)
                except Exception:
                    raise ValueError(f"Unparseable interval: {interval!r}")
            else:
                try:
                    d = Data.asdata(interval, copy=True)
                except Exception:
                    raise ValueError(f"Unparseable interval: {interval!r}")

            if d.size != 1:
                raise ValueError(f"Unparseable interval: {interval!r}")

            if d.ndim > 1:
                d.squeeze(inplace=True)

            values.append(d)

        self.set_qualifier("interval", tuple(values))

    @intervals.deleter
    def intervals(self):
        self.del_qualifier("interval", default=AttributeError())

    @property
    def axes(self):
        """Return the axes of the cell method."""
        return self.get_axes(default=AttributeError())

    @axes.setter
    def axes(self, value):
        if not isinstance(value, (tuple, list)):
            raise ValueError(
                "axes attribute must be a tuple or list, not "
                f"{value.__class__.__name__}"
            )

        self.set_axes(tuple(value))

    @axes.deleter
    def axes(self):
        self.del_axes(default=AttributeError())

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def expand_intervals(self, inplace=False, i=False):
        """Expand the cell method interval to cover all of its axes."""
        c = _inplace_enabled_define_and_cleanup(self)
        n_axes = len(c.get_axes(()))
        intervals = c.get_qualifier("interval", ())
        if n_axes > 1 and len(intervals) == 1:
            c.set_qualifier("interval", intervals * n_axes)

        return c

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def change_axes(self, axis_map, inplace=False, i=False):
        """Change the axes of the cell method according to a given
        mapping.

        :Parameters:

            axis_map: `dict`

            inplace: `bool`

        """
        c = _inplace_enabled_define_and_cleanup(self)

        if not axis_map:
            return c

        c.set_axes([axis_map.get(axis, axis) for axis in self.get_axes(())])

        return c

    @_deprecated_kwarg_check("traceback", version="3.0.0", removed_at="4.0.0")
    @_manage_log_level_via_verbosity
    def equivalent(
        self, other, rtol=None, atol=None, verbose=None, traceback=False
    ):
        """True if two cell methods are equivalent, False otherwise.

        The `axes` and `intervals` attributes are ignored in the
        comparison.

        :Parameters:

            other:
                The object to compare for equality.

            {{atol: number, optional}}

            {{rtol: number, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

            traceback: deprecated at version 3.0.0
                Use the *verbose* parameter instead.

        :Returns:

            `bool`
                Whether or not the two instances are equivalent.

        **Examples**

        >>> a = cf.example_field(1)
        >>> a.cell_methods()
        <CF Constructs: cell_method(2)>
        >>> print(a.cell_methods())
        Constructs:
        {'cellmethod0': <CF CellMethod: domainaxis1: domainaxis2: mean where land (interval: 0.1 degrees)>,
         'cellmethod1': <CF CellMethod: domainaxis3: maximum>}
        >>> c0 = a.cell_method('cellmethod0')
        >>> c1 = a.cell_method('cellmethod1')
        >>> c0.equivalent(c0)
        True
        >>> c0.equivalent(c1)
        False
        >>> c1.equivalent(c0)
        False
        >>> c1.equivalent(c1)
        True

        """
        if self is other:
            return True

        # Check that each instance is the same type
        if self.__class__ != other.__class__:
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Different types: "
                    f"{self.__class__.__name__} != {other.__class__.__name__}"
                )  # pragma: no cover

            return False

        axes0 = self.get_axes(())
        axes1 = other.get_axes(())

        if len(axes0) != len(axes1) or set(axes0) != set(axes1):
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Non-equivalent axes: "
                    f"{axes0!r}, {axes1!r}"
                )  # pragma: no cover

            return False

        argsort = [axes1.index(axis0) for axis0 in axes0]
        other1 = other.sorted(indices=argsort)

        if not self.equals(
            other1, rtol=rtol, atol=atol, ignore_qualifiers=("interval",)
        ):
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Non-equivalent: "
                    f"{self!r}, {other!r}"
                )  # pragma: no cover

            return False

        self1 = self
        if len(self1.get_qualifier("interval", ())) != len(
            other1.get_qualifier("interval", ())
        ):
            self1 = self1.expand_intervals()
            other1.expand_intervals(inplace=True)
            if len(self1.get_qualifier("interval", ())) != len(
                other1.get_qualifier("interval", ())
            ):
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Different numbers of "
                        "intervals: "
                        f"{self1.get_qualifier('interval', ())!r} != "
                        f"{other1.get_qualifier('interval', ())!r}"
                    )  # pragma: no cover

                return False

        intervals0 = self1.get_qualifier("interval", ())
        if intervals0:
            for data0, data1 in zip(
                intervals0, other1.get_qualifier("interval", ())
            ):
                data1 = conform_units(data1, data0.Units)
                if not data0.equals(
                    data1,
                    rtol=rtol,
                    atol=atol,
                    ignore_data_type=True,
                    ignore_fill_value=True,
                ):
                    if is_log_level_info(logger):
                        logger.info(
                            f"{self.__class__.__name__}: Different interval "
                            f"data: {self.intervals!r} != {other.intervals!r}"
                        )  # pragma: no cover

                    return False

        # Still here? Then they are equivalent
        return True

    def inspect(self):
        """Inspect the attributes.

        .. seealso:: `cf.inspect`

        :Returns:

            `None`

        """
        print(cf_inspect(self))

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    def write(self, axis_map=None):
        """Return a string of the cell method.

        Deprecated at version 3.0.0. Use 'str(cell_method)' instead.

        """
        # Unsafe to set mutable '{}' as default in the func signature.
        if axis_map is None:  # distinguish from falsy '{}'
            axis_map = {}
        _DEPRECATION_ERROR_METHOD(
            self,
            "write",
            "Use 'str(cell_method)' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def remove_axes(self, axes):
        """Deprecated at version 3.0.0.

        Use method 'del_axes' instead."

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "remove_axes",
            "Use method 'del_axes' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover
