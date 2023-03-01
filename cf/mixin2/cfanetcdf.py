"""This class is not in the cf.mixin package because it needs to be
imported by cf.Data, and some of the other mixin classes in cf.mixin
themsleves import cf.Data, which would lead to a circular import
situation.

"""
from cfdm.mixin import NetCDFMixin


class CFANetCDF(NetCDFMixin):
    """Mixin class for accessing CFA-netCDF aggregation instruction terms.

    Must be used in conjunction with `NetCDF`

    .. versionadded:: TODOCFAVER

    """

    def cfa_del_aggregated_data(self, default=ValueError()):
        """Remove the CFA-netCDF aggregation instruction terms.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_get_aggregated_data`,
                     `cfa_has_aggregated_data`,
                     `cfa_set_aggregated_data`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if the
                CFA-netCDF aggregation terms have not been set. If set
                to an `Exception` instance then it will be raised
                instead.

        :Returns:

            `dict`
                The removed CFA-netCDF aggregation instruction terms.

        **Examples**

        >>> f.cfa_set_aggregated_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.cfa_has_aggregated_data()
        True
        >>> f.cfa_get_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'c ',
         'tracking_id': 'tracking_id'}
        >>> f.cfa_del_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.cfa_has_aggregated_data()
        False
        >>> print(f.cfa_get_aggregated_data(None))
        None
        None

        """
        return self._nc_del("cfa_aggregated_data", default=default)

    def cfa_get_aggregated_data(self, default=ValueError()):
        """Return the CFA-netCDF aggregation instruction terms.

        .. versifragement onadsded:: TODOCFAVER

        .. seealso:: `scfa_del_aggregated_data`,
                     `cfa_has_aggregated_data`,
                     `cfa_set_aggregated_data`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if the
                CFA-netCDF aggregation terms have not been set. If set
                to an `Exception` instance then it will be raised
                instead.

        :Returns:

            `dict`
         he aggregation instruction terms and their
                corresponding netCDF variable names.

        **Examples**

        >>> f.cfa_set_aggregated_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.cfa_has_aggregated_data()
        True
        >>> f.cfa_get_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.cfa_del_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.cfa_has_aggregated_data()
        False
        >>> print(f.cfa_get_aggregated_data(None))
        None
        >>> print(f.cfa_del_aggregated_data(None))
        None

        """
        out = self._nc_get("cfa_aggregated_data", default=None)
        if out is not None:
            return out.copy()

        if default is None:
            return default

        return self._default(
            default,
            f"{self.__class__.__name__} has no CFA-netCDF aggregation terms",
        )

    def cfa_has_aggregated_data(self):
        """Whether any CFA-netCDF aggregation instruction terms have been set.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_del_aggregated_data`,
                     `cfa_get_aggregated_data`,
                     `cfa_set_aggregated_data`

        :Returns:

            `bool`
                `True` if the CFA-netCDF aggregation instruction terms
                have been set, otherwise `False`.

        **Examples**

        >>> f.cfa_set_aggregated_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.cfa_has_aggregated_data()
        True
        >>> f.cfa_get_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.cfa_del_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.cfa_has_aggregated_data()
        False
        >>> print(f.cfa_get_aggregated_data(None))
        None
        >>> print(f.cfa_del_aggregated_data(None))
        None

        """
        return self._nc_has("cfa_aggregated_data")

    def cfa_set_aggregated_data(self, value):
        """Set the CFA-netCDF aggregation instruction terms.

        If there are any ``/`` (slash) characters in the netCDF
        variable names then these act as delimiters for a group
        hierarchy. By default, or if the name starts with a ``/``
        character and contains no others, the name is assumed to be in
        the root group.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_del_aggregated_data`,
                     `cfa_get_aggregated_data`,
                     `cfa_has_aggregated_data`

        :Parameters:

            value: `dict`
                The aggregation instruction terms and their
                corresponding netCDF variable names.

        :Returns:

            `None`

        **Examples**

        >>> f.cfa_set_aggregated_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.cfa_has_aggregated_data()
        True
        >>> f.cfa_get_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.cfa_del_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.cfa_has_aggregated_data()
        False
        >>> print(f.cfa_get_aggregated_data(None))
        None
        >>> print(f.cfa_del_aggregated_data(None))
        None

        """
        if value:
            self._nc_set("cfa_aggregated_data", value.copy())

    def cfa_clear_file_substitutions(self):
        """Remove the CFA-netCDF file name substitutions.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_get_file_substitutions`,
                     `cfa_has_file_substitutions`,
                     `cfa_set_file_substitutions`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if
                CFA-netCDF file name substitutions have not been
                set. If set to an `Exception` instance then it will be
                raised instead.

        :Returns:

            `dict`
                The removed CFA-netCDF file name substitutions.

        **Examples**

        >>> f.cfa_set_file_substitutions({'${base}': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_get_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_del_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_has_file_substitutions()
        False
        >>> print(f.cfa_get_file_substitutions(None))
        None
        >>> print(f.cfa_del_file_substitutions(None))
        None

        """
        return self._nc_del("cfa_file_substitutions", {}).copy()

    def cfa_del_file_substitution(self, base, default=ValueError()):
        """Remove the CFA-netCDF file name substitutions.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_get_file_substitutions`,
                     `cfa_has_file_substitutions`,
                     `cfa_set_file_substitutions`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if
                CFA-netCDF file name substitutions have not been
                set. If set to an `Exception` instance then it will be
                raised instead.

        :Returns:

            `dict`
                The removed CFA-netCDF file name substitutions.

        **Examples**

        >>> f.cfa_set_file_substitutions({'${base}': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_get_file_substitutions()
        {'base': 'file:///data/'}
        >>> f.cfa_del_file_substitutions()
        {'base': 'file:///data/'}
        >>> f.cfa_has_file_substitutions()
        False
        >>> print(f.cfa_get_file_substitutions(None))
        None
        >>> print(f.cfa_del_file_substitutions(None))
        None

        """
        if not (base.startswith("${") and base.endswith("}")):
            base = f"${{{base}}}"

        subs = self.cfa_file_substitutions({})
        if base not in subs:
            if default is None:
                return

            return self._default(
                default,
                f"{self.__class__.__name__} has no netCDF {base!r} "
                "CFA file substitution",
            )

        out = {base: subs.pop(base)}
        if subs:
            self._nc_set("cfa_file_substitutions", subs)
        else:
            self._nc_del("cfa_file_substitutions", None)

        return out

    def cfa_get_file_substitutions(self):
        """Return the CFA-netCDF file name substitutions.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_del_file_substitutions`,
                     `cfa_get_file_substitutions`,
                     `cfa_set_file_substitutions`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if
                CFA-netCDF file name substitutions have not been
                set. If set to an `Exception` instance then it will be
                raised instead.

        :Returns:

            value: `dict`
                The CFA-netCDF file name substitutions.

        **Examples**

        >>> f.cfa_set_file_substitutions({'${base}': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_get_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_del_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_has_file_substitutions()
        False
        >>> print(f.cfa_get_file_substitutions(None))
        None
        >>> print(f.cfa_del_file_substitutions(None))
        None

        """
        out = self._nc_get("cfa_file_substitutions", default=None)
        if out is not None:
            return out.copy()

        return {}

    def cfa_has_file_substitutions(self):
        """Whether any CFA-netCDF file name substitutions have been set.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_del_file_substitutions`,
                     `cfa_get_file_substitutions`,
                     `cfa_set_file_substitutions`

        :Returns:

            `bool`
                `True` if any CFA-netCDF file name substitutions have
                been set, otherwise `False`.

        **Examples**

        >>> f.cfa_set_file_substitutions({'${base}': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_get_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_del_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_has_file_substitutions()
        False
        >>> print(f.cfa_get_file_substitutions(None))
        None
        >>> print(f.cfa_del_file_substitutions(None))
        None

        """
        return self._nc_has("cfa_file_substitutions")

    def cfa_set_file_substitutions(self, value):
        """Set the CFA-netCDF file name substitutions.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_del_file_substitutions`,
                     `cfa_get_file_substitutions`,
                     `cfa_has_file_substitutions`

        :Parameters:

            value: `dict`
                The new CFA-netCDF file name substitutions.

        :Returns:

            `None`

        **Examples**

        >>> f.cfa_set_file_substitutions({'base': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_get_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_del_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_has_file_substitutions()
        False
        >>> print(f.cfa_get_file_substitutions(None))
        None
        >>> print(f.cfa_del_file_substitutions(None))
        None

        """
        if not value:
            return

        value = value.copy()
        for base, sub in value.items():
            if not (base.startswith("${") and base.endswith("}")):
                value[f"${{{base}}}"] = value.pop(base)

        subs = self.cfa_get_file_substitutions()
        subs.update(value)
        self._nc_set("cfa_file_substitutions", subs)
