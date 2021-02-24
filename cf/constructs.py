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
        """TODO"""
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
                Identify the metadata constructs by one or more of

                * A metadata construct identity.

                  {{construct selection identity}}

                * The key of a metadata construct

                *Parameter example:*
                  ``identity='latitude'``

                *Parameter example:*
                  ``'T'

                *Parameter example:*
                  ``'latitude'``

                *Parameter example:*
                  ``'long_name=Cell Area'``

                *Parameter example:*
                  ``'cellmeasure1'``

                *Parameter example:*
                  ``'measure:area'``

                *Parameter example:*
                  ``cf.eq('time')'``

                *Parameter example:*
                  ``re.compile('^lat')``

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
        # Allow keys without the 'key%' prefix
        identities = list(identities)
        for n, identity in enumerate(identities):
            if identity in self:
                identities[n] = f"key%{identity}"

        return super().filter_by_identity(*identities)
