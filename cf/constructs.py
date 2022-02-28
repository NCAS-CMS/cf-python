import logging

import cfdm

from .query import Query

logger = logging.getLogger(__name__)


class Constructs(cfdm.Constructs):
    """A container for metadata constructs.

    The following metadata constructs can be included:

    * auxiliary coordinate constructs
    * coordinate reference constructs
    * cell measure constructs
    * dimension coordinate constructs
    * domain ancillary constructs
    * domain axis constructs
    * cell method constructs
    * field ancillary constructs

    The container may be used by `Field` and `Domain` instances. In
    the latter case cell method and field ancillary constructs must be
    flagged as "ignored" (see the *_ignore* parameter).

    The container is like a dictionary in many ways, in that it stores
    key/value pairs where the key is the unique construct key with
    corresponding metadata construct value, and provides some of the
    usual dictionary methods.

    **Calling**

    Calling a `Constructs` instance selects metadata constructs by
    identity and is an alias for the `filter_by_identity` method. For
    example, to select constructs that have an identity of
    'air_temperature': ``d = c('air_temperature')``.

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

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def close(self):
        """Close all files referenced by the metadata constructs.

        Note that a closed file will be automatically reopened if its
        contents are subsequently required.

        :Returns:

            `None`

        **Examples:**

        >>> c.close()

        """
        # TODODASK - is this method still needed?

        for construct in self.filter_by_data().values():
            construct.close()

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

    #    def _filter_by_coordinate_type(self, arg, ctypes, todict):
    #        """Worker function for `filter_by_identity` and `filter`.
    #
    #        See `filter_by_identity` for details.
    #
    #        .. versionadded:: 3.9.0
    #
    #        """
    #        out, pop = self._filter_preprocess(
    #            arg,
    #            filter_applied={"filter_by_identity": ctypes},
    #            todict=todict,
    #        )
    #
    #        if not ctypes:
    #            # Return all constructs if no coordinate types have been
    #            # provided
    #            return out
    #
    #        for cid, construct in tuple(out.items()):
    #            ok = False
    #            for ctype in ctypes:
    #                if getattr(construct, ctype, False):
    #                    ok = True
    #                    break
    #
    #            if not ok:
    #                pop(cid)
    #
    #        return out

    @classmethod
    def _short_iteration(cls, x):
        """The default short circuit test.

        If this method returns True then only the first identity
        returned by the construct's `!identities` method will be
        checked.

        See `_filter_by_identity` for details.

        .. versionadded:: (cfdm) 1.8.9.0

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
