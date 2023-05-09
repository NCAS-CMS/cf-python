from . import ConstructList, mixin
from .functions import _DEPRECATION_ERROR_METHOD


class FieldList(mixin.FieldDomainList, ConstructList):
    """An ordered sequence of field constructs.

    A field list supports the python list-like operations (such as
    indexing and methods like `!append`). These methods provide
    functionality similar to that of a built-in list. The main
    difference is that when a field construct element needs to be
    assessed for equality its `~cf.Field.equals` method is used, rather
    than the ``==`` operator.

    """

    def __init__(self, fields=None):
        """**Initialisation**

        :Parameters:

            fields: (sequence of) `Field`, optional
                 Create a new list with these field constructs.

        """
        super().__init__(constructs=fields)

    def concatenate(
        self, axis=0, cull_graph=False, relaxed_units=False, copy=True
    ):
        """Join the sequence of fields within the field list together.

        This is different to `cf.aggregate` because it does not
        account for all metadata. For example, it assumes that the
        axis order is the same in each field.

        .. versionadded:: 1.0

        .. seealso:: `cf.aggregate`, `Field.concatenate`,
                     `Data.concatenate`

        :Parameters:

            axis: `int`, optional
                The axis along which the arrays will be joined. The
                default is 0. Note that scalar arrays are treated as
                if they were one dimensional.

            {{cull_graph: `bool`, optional}}

                .. versionadded:: 3.14.0

            {{relaxed_units: `bool`, optional}}

                .. versionadded:: 3.15.1

            copy: `bool`, optional
                If True (the default) then make copies of the Field
                constructs, prior to the concatenation, thereby
                ensuring that the input constructs are not changed by
                the concatenation process. If False then some or all
                input constructs might be changed in-place, but the
                concatenation process will be faster.

                .. versionadded:: 3.15.1

        :Returns:

            `Field`
                The field generated from the concatenation of all of
                the fields contained in the input field list.

        """
        return self[0].concatenate(
            self,
            axis=axis,
            cull_graph=cull_graph,
            relaxed_units=relaxed_units,
            copy=copy,
        )

    def select_by_naxes(self, *naxes):
        """Select field constructs by property.

        To find the inverse of the selection, use a list comprehension
        with `~cf.Field.match_by_naxes` method of the construct
        elements. For example, to select all constructs which do *not*
        have 3-dimensional data:

           >>> gl = cf.FieldList(
           ...     f for f in fl if not f.match_by_naxes(3)
           ... )

        .. versionadded:: 3.0.0

        .. seealso:: `select`, `select_by_identity`,
                     `select_by_construct`, `select_by_property`,
                     `select_by_rank`, `select_by_units`

        :Parameters:

            naxes: `int`, optional
                Select field constructs whose data spans a particular
                number of domain axis constructs.

                A number of domain axis constructs is given by an
                `int`.

                If no numbers are provided then all field constructs
                are selected.

        :Returns:

            `FieldList`
                The matching field constructs.

        **Examples**

        >>> f = cf.read("file.nc")
        >>> f
        [<CF Field: specific_humidity(latitude(5), longitude(8)) 1>,
         <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>]
        >>> f.select_by_naxes()
        [<CF Field: specific_humidity(latitude(5), longitude(8)) 1>,
        <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>]
        >>> f.select_by_naxes(1)
        []
        >>> f.select_by_naxes(2)
        [<CF Field: specific_humidity(latitude(5), longitude(8)) 1>]
        >>> f.select_by_naxes(3)
        [<CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>]

        """
        return type(self)(f for f in self if f.match_by_naxes(*naxes))

    def select_by_units(self, *units, exact=True):
        """Select field constructs by units.

        To find the inverse of the selection, use a list comprehension
        with `~cf.Field.match_by_units` method of the construct
        elements. For example, to select all constructs whose units
        are *not* ``'km'``:

           >>> gl = cf.FieldList(
           ...     f for f in fl if not f.match_by_units('km')
           ... )

        .. versionadded:: 3.0.0

        .. seealso:: `select`, `select_by_identity`,
                     `select_by_construct`, `select_by_naxes`,
                     `select_by_rank`, `select_by_property`

        :Parameters:

            units: optional
                Select field constructs. By default all field
                constructs are selected. May be one or more of:

                  * The units of a field construct.

                Units are specified by a string or compiled regular
                expression (e.g. 'km', 'm s-1',
                ``re.compile('^kilo')``, etc.) or a `Units` object
                (e.g. ``Units('km')``, ``Units('m s-1')``, etc.).

            exact: `bool`, optional
                If `False` then select field constructs whose units
                are equivalent to any of those given by *units*. For
                example, metres and are equivalent to kilometres. By
                default, field constructs whose units are exactly one
                of those given by *units* are selected. Note that the
                format of the units is not important, i.e. 'm' is
                exactly the same as 'metres' for this purpose.

        :Returns:

            `FieldList`
                The matching field constructs.

        **Examples**

        >>> gl = fl.select_by_units('metres')
        >>> gl = fl.select_by_units('m')
        >>> gl = fl.select_by_units('m', 'kilogram')
        >>> gl = fl.select_by_units(Units('m'))
        >>> gl = fl.select_by_units('km', exact=False)
        >>> gl = fl.select_by_units(Units('km'), exact=False)
        >>> gl = fl.select_by_units(re.compile('^met'))
        >>> gl = fl.select_by_units(Units('km'))
        >>> gl = fl.select_by_units(Units('kg m-2'))

        """
        return type(self)(
            f for f in self if f.match_by_units(*units, exact=exact)
        )

    def select_field(self, *identities, default=ValueError()):
        """Select a unique field construct by its identity.

        .. versionadded:: 3.0.4

        .. seealso:: `select`, `select_by_identity`

        :Parameters:

            identities: optional
                Select the field construct by one or more of

                * A construct identity.

                  {{construct selection identity}}

            default: optional
                Return the value of the *default* parameter if a
                unique field construct can not be found.

                {{default Exception}}

        :Returns:

            `Field`
                The unique matching field construct.

        **Examples**

        >>> fl
        [<CF Field: specific_humidity(latitude(73), longitude(96)) 1>,
         <CF Field: specific_humidity(latitude(73), longitude(96)) 1>,
         <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>]
        >>> fl.select_field('air_temperature')
        <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
        >>> f.select_field('specific_humidity')
        ValueError: Multiple fields found
        >>> f.select_field('specific_humidity', 'No unique field')
        'No unique field'
        >>> f.select_field('snowfall_amount')
        ValueError: No fields found

        """
        out = self.select_by_identity(*identities)

        if not out:
            if default is None:
                return

            return self._default(
                default, "select_field() can't return 0 fields"
            )

        n = len(out)
        if n > 1:
            if default is None:
                return

            return self._default(
                default, f"select_field() can't return {n} fields"
            )

        return out[0]

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    def _parameters(self, d):
        """Deprecated at version 3.0.0."""
        _DEPRECATION_ERROR_METHOD(
            self, "_parameters", version="3.0.0", removed_at="4.0.0"
        )  # pragma: no cover

    def _deprecated_method(self, name):
        """Deprecated at version 3.0.0."""
        _DEPRECATION_ERROR_METHOD(
            self, "_deprecated_method", version="3.0.0", removed_at="4.0.0"
        )  # pragma: no cover

    def set_equals(
        self,
        other,
        rtol=None,
        atol=None,
        ignore_data_type=False,
        ignore_fill_value=False,
        ignore_properties=(),
        ignore_compression=False,
        ignore_type=False,
        traceback=False,
    ):
        """Deprecated at version 3.0.0.

        Use method 'equals' with unordered=True instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "set_equals",
            "Use method 'equals' with unordered=True instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def select1(self, *args, **kwargs):
        """Deprecated at version 3.0.0.

        Use method 'fl.select_field' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "select1",
            "Use method 'fl.select_field' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover
