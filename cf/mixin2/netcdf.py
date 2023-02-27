"""This class is not in the cf.mixin package because it needs to be
imported by cf.Data, and some of the other mixin classes in cf.mixin
themsleves import cf.Data, which would lead to a circular import
situation.

"""


class CFANetCDF:
    """Mixin class for accessing CFA-netCDF aggregation instruction terms.

    .. versionadded:: TODOCFAVER

    """

    def nc_del_cfa_aggregation_data(self, default=ValueError()):
        """Remove the CFA-netCDF aggregation instruction terms.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_get_cfa_aggregation_data`,
                     `nc_has_cfa_aggregation_data`,
                     `nc_set_cfa_aggregation_data`

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

        >>> f.nc_set_cfa_aggregation_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.nc_has_cfa_aggregation_data()
        True
        >>> f.nc_get_cfa_aggregation_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_del_cfa_aggregation_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_has_cfa_aggregation_data()
        False
        >>> print(f.nc_get_cfa_aggregation_data(None))
        None
        >>> print(f.nc_del_cfa_aggregation_data(None))
        None

        """
        return self._nc_del("dimension", default=default)

    def nc_get_cfa_aggregation_data(self, default=ValueError()):
        """Return the CFA-netCDF aggregation instruction terms.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_del_cfa_aggregation_data`,
                     `nc_has_cfa_aggregation_data`,
                     `nc_set_cfa_aggregation_data`

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

        >>> f.nc_set_cfa_aggregation_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.nc_has_cfa_aggregation_data()
        True
        >>> f.nc_get_cfa_aggregation_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_del_cfa_aggregation_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_has_cfa_aggregation_data()
        False
        >>> print(f.nc_get_cfa_aggregation_data(None))
        None
        >>> print(f.nc_del_cfa_aggregation_data(None))
        None

        """
        out = self._nc_get("cfa_aggregation_data", default=None)
        if out is not None:
            return out.copy()

        if default is None:
            return default

        return self._default(
            default,
            f"{self.__class__.__name__} has no CFA-netCDF aggregation terms",
        )

    def nc_has_cfa_aggregation_data(self):
        """Whether the CFA-netCDF aggregation instruction terms have been set.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_del_cfa_aggregation_data`,
                     `nc_get_cfa_aggregation_data`,
                     `nc_set_cfa_aggregation_data`

        :Returns:

            `bool`
                `True` if the CFA-netCDF aggregation instruction terms
                have been set, otherwise `False`.

        **Examples**

        >>> f.nc_set_cfa_aggregation_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.nc_has_cfa_aggregation_data()
        True
        >>> f.nc_get_cfa_aggregation_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_del_cfa_aggregation_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_has_cfa_aggregation_data()
        False
        >>> print(f.nc_get_cfa_aggregation_data(None))
        None
        >>> print(f.nc_del_cfa_aggregation_data(None))
        None

        """
        return self._nc_has("cfa_aggregation_data")

    def nc_set_cfa_aggregation_data(self, value):
        """Set the CFA-netCDF aggregation instruction terms.

        If there are any ``/`` (slash) characters in the netCDF
        variable names then these act as delimiters for a group
        hierarchy. By default, or if the name starts with a ``/``
        character and contains no others, the name is assumed to be in
        the root group.

        .. versionadded:: TODOCFAVER

        .. seealso:: `nc_del_cfa_aggregation_data`,
                     `nc_get_cfa_aggregation_data`,
                     `nc_has_cfa_aggregation_data`

        :Parameters:

            value: `dict`
                The aggregation instruction terms and their
                corresponding netCDF variable names.

        :Returns:

            `None`

        **Examples**

        >>> f.nc_set_cfa_aggregation_data(
        ...     {'location': 'cfa_location',
        ...      'file': 'cfa_file',
        ...      'address': 'cfa_address',
        ...      'format': 'cfa_format',
        ...      'tracking_id': 'tracking_id'}
        ... )
        >>> f.nc_has_cfa_aggregation_data()
        True
        >>> f.nc_get_cfa_aggregation_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_del_cfa_aggregation_data()
        {'location': 'cfa_location',
         'file': 'cfa_file',
         'address': 'cfa_address',
         'format': 'cfa_format',
         'tracking_id': 'tracking_id'}
        >>> f.nc_has_cfa_aggregation_data()
        False
        >>> print(f.nc_get_cfa_aggregation_data(None))
        None
        >>> print(f.nc_del_cfa_aggregation_data(None))
        None

        """
        return self._nc_set("cfa_aggregation_data", value.copy())
