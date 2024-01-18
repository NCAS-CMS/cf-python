import logging

import cfdm

from .functions import _DEPRECATION_ERROR_METHOD
from .query import Query

logger = logging.getLogger(__name__)


class Constructs(cfdm.Constructs):
    """A container for metadata constructs.

    The container has similarities to a `dict` in that it presents the
    metadata constructs as key/value pairs, where the key is the
    unique identifier that corresponds to a metadata construct value;
    is indexable by metadata construct identifier; and provides a
    subset of the usual dictionary methods: `get`, `items`, `keys`,
    and `values`. The number of constructs (which may be zero) can be
    found via the Python `len` function. The container can be
    converted to an actual `dict` with the `todict` method.

    **Filtering**

    A subset of the metadata constructs can be defined and returned in
    a new `Constructs` instance by using the various filter
    methods. See `filter` for more details.

    **Calling**

    Calling a `Constructs` instance also creates a new `Constructs`
    instance that contains a subset of the metadata constructs,
    primarily selecting by construct identity. For instance, selecting
    constructs that have an identity of 'latitude' could be done by
    either ``x = c('latitude')`` or ``x =
    c.filter_by_identity('latitude')``. More generally,
    ``c(*identities, **filter_kwargs)`` is equivalent to
    ``c.filter(filter_by_identity=identities, **filter_kwargs)``.

    **Metadata constructs**

    The following metadata constructs can be included:

    * auxiliary coordinate constructs
    * coordinate reference constructs
    * cell measure constructs
    * dimension coordinate constructs
    * domain ancillary constructs
    * domain axis constructs
    * domain topology constructs
    * cell connectivity constructs
    * cell method constructs
    * field ancillary constructs

    .. versionadded:: 3.0.0

    """

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)

    @classmethod
    def _matching_values(cls, value0, construct, value1, basic=False):
        """Whether two values match according to equality on a given
        construct.

        The definition of "match" depends on the types of *value0* and
        *value1*.

        :Parameters:

            value0:
                The first value to be matched.

            construct:
                The construct whose `_equals` method is used to
                determine whether values can be considered to match.

            value1:
                The second value to be matched.

            basic: `bool`
                If True then value0 and value1 will be compared with
                the basic ``==`` operator.

        :Returns:

            `bool`
                Whether or not the two values match.

        """
        if isinstance(value0, Query):
            return value0.evaluate(value1)

        return super()._matching_values(value0, construct, value1, basic=basic)

    def _flip(self, axes):
        """Flip (reverse the direction of) axes of the constructs in-
        place.

        .. versionadded:: 3.11.0

        :Parameters:

            axes: sequence of `str`
                Select the domain axes to flip, defined by the domain axis
                identifiers. The sequence may be empty.

        :Returns:

            `None`

        """
        data_axes = self.data_axes()

        # Flip any constructs which span the given axes
        for key, construct in self.filter_by_data().items():
            construct_axes = data_axes[key]
            construct_flip_axes = axes.intersection(construct_axes)
            if construct_flip_axes:
                iaxes = [
                    construct_axes.index(axis) for axis in construct_flip_axes
                ]
                construct.flip(iaxes, inplace=True)

    def close(self):
        """Close all files referenced by the metadata constructs.

        Deprecated at version 3.14.0. All files are now
        automatically closed when not being accessed.

        Note that a closed file will be automatically reopened if its
        contents are subsequently required.

        :Returns:

            `None`

        **Examples**

        >>> c.close()

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "close",
            "All files are now automatically closed when not being accessed.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def _filter_by_identity(self, arg, identities, todict, _config):
        """Worker function for `filter_by_identity` and `filter`.

        See `filter_by_identity` for details.

        .. versionadded:: 3.9.0

        """
        ctypes = [i for i in "XTYZ" if i in identities]

        config = {"identities_kwargs": {"ctypes": ctypes}}
        if _config:
            config.update(_config)

        return super()._filter_by_identity(arg, identities, todict, config)

    @classmethod
    def _short_iteration(cls, x):
        """The default short circuit test.

        If this method returns True then only the first identity
        returned by the construct's `!identities` method will be
        checked.

        See `_filter_by_identity` for details.

        .. versionadded:: 3.9.0

        :Parameters:

            x: `str`
                The value against which the construct's identities are
                being compared.

        :Returns:

            `bool`
                 Returns `True` if a construct's `identities` method
                 is to short circuit after the first identity is
                 computed, otherwise `False`.

        """
        if not isinstance(x, str):
            return False

        if x in "XTYZ" or x.startswith("measure:") or x.startswith("id%"):
            return True

        return "=" not in x and ":" not in x and "%" not in x
