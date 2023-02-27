"""This class is not in the cf.mixin package because it needs to be
imported by cf.Data, and some of the other mixin classes in cf.mixin
themsleves import cf.Data, which would lead to a circular import
situation.

"""


class CFANetCDF:
    """Mixin class for accessing CFA-netCDF aggregation instruction terms.

    .. versionadded:: TODOCFAVER

    """

    def nc_del_cfa_aggregated_data(self, default=ValueError()):
        """Remove the CFA-netCDF aggregation instruction terms.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_get_cfa_aggregated_data`,
                     `nc_has_cfa_aggregated_data`,
                     `nc_set_cfa_aggregated_data`

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

        >>> f.nc_set_cfa_aggregated_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.nc_has_cfa_aggregated_data()
        True
        >>> f.nc_get_cfa_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_del_cfa_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_has_cfa_aggregated_data()
        False
        >>> print(f.nc_get_cfa_aggregated_data(None))
        None
        >>> print(f.nc_del_cfa_aggregated_data(None))
        None

        """
        return self._nc_del("cfa_aggregated_data", default=default)

    def nc_get_cfa_aggregated_data(self, default=ValueError()):
        """Return the CFA-netCDF aggregation instruction terms.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_del_cfa_aggregated_data`,
                     `nc_has_cfa_aggregated_data`,
                     `nc_set_cfa_aggregated_data`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if the
                CFA-netCDF aggregation terms have not been set. If set
                to an `Exception` instance then it will be raised
                instead.

        :Returns:

            `dict`
                The aggregation instruction terms and their
                corresponding netCDF variable names.

        **Examples**

        >>> f.nc_set_cfa_aggregated_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.nc_has_cfa_aggregated_data()
        True
        >>> f.nc_get_cfa_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_del_cfa_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_has_cfa_aggregated_data()
        False
        >>> print(f.nc_get_cfa_aggregated_data(None))
        None
        >>> print(f.nc_del_cfa_aggregated_data(None))
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

    def nc_has_cfa_aggregated_data(self):
        """Whether any CFA-netCDF aggregation instruction terms have been set.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_del_cfa_aggregated_data`,
                     `nc_get_cfa_aggregated_data`,
                     `nc_set_cfa_aggregated_data`

        :Returns:

            `bool`
                `True` if the CFA-netCDF aggregation instruction terms
                have been set, otherwise `False`.

        **Examples**

        >>> f.nc_set_cfa_aggregated_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.nc_has_cfa_aggregated_data()
        True
        >>> f.nc_get_cfa_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_del_cfa_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_has_cfa_aggregated_data()
        False
        >>> print(f.nc_get_cfa_aggregated_data(None))
        None
        >>> print(f.nc_del_cfa_aggregated_data(None))
        None

        """
        return self._nc_has("cfa_aggregated_data")

    def nc_set_cfa_aggregated_data(self, value):
        """Set the CFA-netCDF aggregation instruction terms.

        If there are any ``/`` (slash) characters in the netCDF
        variable names then these act as delimiters for a group
        hierarchy. By default, or if the name starts with a ``/``
        character and contains no others, the name is assumed to be in
        the root group.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_del_cfa_aggregated_data`,
                     `nc_get_cfa_aggregated_data`,
                     `nc_has_cfa_aggregated_data`

        :Parameters:

            value: `dict`
                The aggregation instruction terms and their
                corresponding netCDF variable names.

        :Returns:

            `None`

        **Examples**

        >>> f.nc_set_cfa_aggregated_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.nc_has_cfa_aggregated_data()
        True
        >>> f.nc_get_cfa_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_del_cfa_aggregated_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_has_cfa_aggregated_data()
        False
        >>> print(f.nc_get_cfa_aggregated_data(None))
        None
        >>> print(f.nc_del_cfa_aggregated_data(None))
        None

        """
        return self._nc_set("cfa_aggregated_data", value.copy())

    def nc_del_cfa_file_substitutions(self, value, default=ValueError()):
        """Remove the CFA-netCDF file name substitutions.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_get_cfa_file_substitutions`,
                     `nc_has_cfa_file_substitutions`,
                     `nc_set_cfa_file_substitutions`

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

        >>> f.nc_set_cfa_file_substitutions({'${base}': 'file:///data/'})
        >>> f.nc_has_cfa_file_substitutions()
        True
        >>> f.nc_get_cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.nc_del_cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.nc_has_cfa_file_substitutions()
        False
        >>> print(f.nc_get_cfa_file_substitutions(None))
        None
        >>> print(f.nc_del_cfa_file_substitutions(None))
        None

        """
        return self._nc_del("cfa_file_substitutions", default=default)

    def nc_get_cfa_file_substitutions(self, default=ValueError()):
        """Return the CFA-netCDF file name substitutions.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_del_cfa_file_substitutions`,
                     `nc_get_cfa_file_substitutions`,
                     `nc_set_cfa_file_substitutions`

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

        >>> f.nc_set_cfa_file_substitutions({'${base}': 'file:///data/'})
        >>> f.nc_has_cfa_file_substitutions()
        True
        >>> f.nc_get_cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.nc_del_cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.nc_has_cfa_file_substitutions()
        False
        >>> print(f.nc_get_cfa_file_substitutions(None))
        None
        >>> print(f.nc_del_cfa_file_substitutions(None))
        None

        """
        out = self._nc_get("cfa_file_substitutions", default=None)
        if out is not None:
            return out.copy()

        if default is None:
            return default

        return self._default(
            default,
            f"{self.__class__.__name__} has no CFA-netCDF file name substitutions",
        )

    def nc_has_cfa_file_substitutions(self):
        """Whether any CFA-netCDF file name substitutions have been set.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_del_cfa_file_substitutions`,
                     `nc_get_cfa_file_substitutions`,
                     `nc_set_cfa_file_substitutions`

        :Returns:

            `bool`
                `True` if any CFA-netCDF file name substitutions have
                been set, otherwise `False`.

        **Examples**

        >>> f.nc_set_cfa_file_substitutions({'${base}': 'file:///data/'})
        >>> f.nc_has_cfa_file_substitutions()
        True
        >>> f.nc_get_cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.nc_del_cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.nc_has_cfa_file_substitutions()
        False
        >>> print(f.nc_get_cfa_file_substitutions(None))
        None
        >>> print(f.nc_del_cfa_file_substitutions(None))
        None

        """
        return self._nc_has("cfa_file_substitutions")

    def nc_set_cfa_file_substitutions(self, value):
        """Set the CFA-netCDF file name substitutions.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_del_cfa_file_substitutions`,
                     `nc_get_cfa_file_substitutions`,
                     `nc_has_cfa_file_substitutions`

        :Parameters:

            value: `dict`
                The CFA-netCDF file name substitutions.

        :Returns:

            `None`

        **Examples**

        >>> f.nc_set_cfa_file_substitutions({'${base}': 'file:///data/'})
        >>> f.nc_has_cfa_file_substitutions()
        True
        >>> f.nc_get_cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.nc_del_cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.nc_has_cfa_file_substitutions()
        False
        >>> print(f.nc_get_cfa_file_substitutions(None))
        None
        >>> print(f.nc_del_cfa_file_substitutions(None))
        None

        """
        return self._nc_set("cfa_file_substitutions", value.copy())
