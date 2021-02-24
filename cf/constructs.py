import logging

import cfdm

from . import mixin

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

    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------
    def _matching_values(self, value0, construct, value1):
        """Whether two values match according to equality on a given
        construct.

        The definition of "match" depends on the types of *value0* and
        *value1*.

        :Parameters:

            value0:
                The first value to be matched.

            construct:
                The construct whose `equals` method is used to determine whether
                values can be considered to match.

            value1:
                The second value to be matched.

        :Returns:

            `bool`
                Whether or not the two values match.

        """
        if isinstance(value0, Query):
            return value0.evaluate(value1)

        return super()._matching_values(value0, construct, value1)

    def _flip(self, axes):
        """Flip (reverse the direction of) axes of the constructs in-place.

        .. versionadded:: 3.TODO.0

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

    def filter_by_identity(self, *identities):
        """Select metadata constructs by identity.

        .. versionadded:: 3.0.0

        .. seealso:: `filter_by_axis`, `filter_by_data`, `filter_by_key`,
                     `filter_by_measure`, `filter_by_method`,
                     `filter_by_naxes`, `filter_by_ncdim`,
                     `filter_by_ncvar`, `filter_by_property`,
                     `filter_by_size`, `filter_by_type`,
                     `filters_applied`, `inverse_filter`, `unfilter`

        :Parameters:

            identities: optional
                Select constructs that have any of the given identities or
                construct keys.

                An identity is specified by a string (e.g. ``'latitude'``,
                ``'long_name=time'``, etc.); or a compiled regular
                expression (e.g. ``re.compile('^atmosphere')``), for which
                all constructs whose identities match (via `re.search`)
                are selected.

                If no identities are provided then all constructs are selected.

                Each construct has a number of identities, and is selected
                if any of them match any of those provided. A construct's
                identities are those returned by its `!identities`
                method. In the following example, the construct ``x`` has
                five identities:

                   >>> x.identities()
                   ['time', 'long_name=Time', 'foo=bar', 'T', 'ncvar%t']

                A construct key may optionally have the ``'key%'``
                prefix. For example ``'dimensioncoordinate2'`` and
                ``'key%dimensioncoordinate2'`` are both acceptable keys.

                Note that the identifiers of a metadata construct in the
                output of a `print` or `!dump` call are always one of its
                identities, and so may always be used as an *identities*
                argument.

                Domain axis constructs may also be identified by their
                position in the field construct's data array. Positions
                are specified by either integers.

                .. note:: This is an extension to the functionality of
                          `cfdm.Constucts.filter_by_identity`.

        :Returns:

            `Constructs`
                The selected constructs and their construct keys.

        **Examples:**

        Select constructs that have a "standard_name" property of
        'latitude':

        >>> d = c.filter_by_identity('latitude')

        Select constructs that have a "long_name" property of 'Height':

        >>> d = c.filter_by_identity('long_name=Height')

        Select constructs that have a "standard_name" property of
        'latitude' or a "foo" property of 'bar':

        >>> d = c.filter_by_identity('latitude', 'foo=bar')

        Select constructs that have a netCDF variable name of 'time':

        >>> d = c.filter_by_identity('ncvar%time')

        """
        #        field_data_axes = self._field_data_axes
        #
        #        if field_data_axes is not None:
        #            # Allows integer data domain axis positions, do we want this? TODO
        #            new_identities = []
        #            for i in identities:
        #                try:
        #                    _ = field_data_axes[i]
        #                except IndexError:
        #                    new_identities.append(i)
        #                else:
        #                    if isinstance(_, str):
        #                        new_identities.append('key%'+_)
        #                    else:
        #                        new_identities.extend(['key%'+axis for axis in _])
        #        else:
        #            new_identities = identities
        #

        # Allow keys without the 'key%' prefix
        identities = list(identities)
        for n, identity in enumerate(identities):
            if identity in self:
                identities[n] = f"key%{identity}"

        return super().filter_by_identity(*identities)
