"""This class is not in the cf.mixin package because it needs to be
imported by cf.Data, and some of the other mixin classes in cf.mixin
themsleves import cf.Data, which would lead to a circular import
situation.

"""

from re import split

from cfdm.mixin import NetCDFMixin


class CFANetCDF(NetCDFMixin):
    """Mixin class for CFA-netCDF.

    .. versionadded:: 3.15.0

    """

    def cfa_del_aggregated_data(self):
        """Remove the CFA-netCDF aggregation instruction terms.

        The aggregation instructions are stored in the
        ``aggregation_data`` attribute of a CFA-netCDF aggregation
        variable.

        .. versionadded:: 3.15.0

        .. seealso:: `cfa_get_aggregated_data`,
                     `cfa_has_aggregated_data`,
                     `cfa_set_aggregated_data`

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
        >>> f.cfa_del_aggregated_data()
        {}
        >>> f.cfa_get_aggregated_data()
        {}

        """
        return self._nc_del("cfa_aggregated_data", {}).copy()

    def cfa_get_aggregated_data(self):
        """Return the CFA-netCDF aggregation instruction terms.

        The aggregation instructions are stored in the
        ``aggregation_data`` attribute of a CFA-netCDF aggregation
        variable.

        .. versionadded:: 3.15.0

        .. seealso:: `cfa_del_aggregated_data`,
                     `cfa_has_aggregated_data`,
                     `cfa_set_aggregated_data`

        :Returns:

            `dict`
                The aggregation instruction terms and their
                corresponding netCDF variable names in a dictionary
                whose key/value pairs are the aggregation instruction
                terms and their corresponding variable names.

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
        >>> f.cfa_del_aggregated_data()
        {}
        >>> f.cfa_get_aggregated_data()
        {}

        """
        out = self._nc_get("cfa_aggregated_data", default=None)
        if out is not None:
            return out.copy()

        return {}

    def cfa_has_aggregated_data(self):
        """Whether any CFA-netCDF aggregation instruction terms have been set.

        The aggregation instructions are stored in the
        ``aggregation_data`` attribute of a CFA-netCDF aggregation
        variable.

        .. versionadded:: 3.15.0

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
        >>> f.cfa_del_aggregated_data()
        {}
        >>> f.cfa_get_aggregated_data()
        {}

        """
        return self._nc_has("cfa_aggregated_data")

    def cfa_set_aggregated_data(self, value):
        """Set the CFA-netCDF aggregation instruction terms.

        The aggregation instructions are stored in the
        ``aggregation_data`` attribute of a CFA-netCDF aggregation
        variable.

        If there are any ``/`` (slash) characters in the netCDF
        variable names then these act as delimiters for a group
        hierarchy. By default, or if the name starts with a ``/``
        character and contains no others, the name is assumed to be in
        the root group.

        .. versionadded:: 3.15.0

        .. seealso:: `cfa_del_aggregated_data`,
                     `cfa_get_aggregated_data`,
                     `cfa_has_aggregated_data`

        :Parameters:

            value: `str` or `dict`
                The aggregation instruction terms and their
                corresponding netCDF variable names. Either a
                CFA-netCDF-compliant string value of an
                ``aggregated_data`` attribute, or a dictionary whose
                key/value pairs are the aggregation instruction terms
                and their corresponding variable names.

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
        >>> f.cfa_del_aggregated_data()
        {}
        >>> f.cfa_get_aggregated_data()
        {}

        """
        if value:
            if isinstance(value, str):
                v = split("\s+", value)
                value = {term[:-1]: var for term, var in zip(v[::2], v[1::2])}
            else:
                # 'value' is a dictionary
                value = value.copy()

            self._nc_set("cfa_aggregated_data", value)

    def cfa_clear_file_substitutions(self):
        """Remove all of the CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        .. seealso:: `cfa_del_file_substitution`,
                     `cfa_file_substitutions`,
                     `cfa_has_file_substitutions`,
                     `cfa_update_file_substitutions`

        :Returns:

            `dict`
                {{Returns cfa_clear_file_substitutions}}

        **Examples**

        >>> f.cfa_update_file_substitutions({'base': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_update_file_substitutions({'${base2}': '/home/data/'})
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/', '${base2}': '/home/data/'}
        >>> f.cfa_update_file_substitutions({'${base}': '/new/location/'})
        >>> f.cfa_file_substitutions()
        {'${base}': '/new/location/', '${base2}': '/home/data/'}
        >>> f.cfa_del_file_substitution('${base}')
        {'${base}': '/new/location/'}
        >>> f.cfa_clear_file_substitutions()
        {'${base2}': '/home/data/'}
        >>> f.cfa_has_file_substitutions()
        False
        >>> f.cfa_file_substitutions()
        {}
        >>> f.cfa_clear_file_substitutions()
        {}
        >>> print(f.cfa_del_file_substitution('base', None))
        None

        """
        return self._nc_del("cfa_file_substitutions", {}).copy()

    def cfa_del_file_substitution(self, base):
        """Remove a CFA-netCDF file name substitution.

        .. versionadded:: 3.15.0

        .. seealso:: `cfa_clear_file_substitutions`,
                     `cfa_file_substitutions`,
                     `cfa_has_file_substitutions`,
                     `cfa_update_file_substitutions`

        :Parameters:

            {{cfa base: `str`}}

        :Returns:

            `dict`
                {{Returns cfa_del_file_substitution}}

        **Examples**

        >>> f.cfa_update_file_substitutions({'base': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_update_file_substitutions({'${base2}': '/home/data/'})
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/', '${base2}': '/home/data/'}
        >>> f.cfa_update_file_substitutions({'${base}': '/new/location/'})
        >>> f.cfa_file_substitutions()
        {'${base}': '/new/location/', '${base2}': '/home/data/'}
        >>> f.cfa_del_file_substitution('${base}')
        {'${base}': '/new/location/'}
        >>> f.cfa_clear_file_substitutions()
        {'${base2}': '/home/data/'}
        >>> f.cfa_has_file_substitutions()
        False
        >>> f.cfa_file_substitutions()
        {}
        >>> f.cfa_clear_file_substitutions()
        {}
        >>> print(f.cfa_del_file_substitution('base'))
        {}

        """
        if not (base.startswith("${") and base.endswith("}")):
            base = f"${{{base}}}"

        subs = self.cfa_file_substitutions()
        if base not in subs:
            return {}

        out = {base: subs.pop(base)}
        if subs:
            self._nc_set("cfa_file_substitutions", subs)
        else:
            self._nc_del("cfa_file_substitutions", None)

        return out

    def cfa_file_substitutions(self):
        """Return the CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        .. seealso:: `cfa_clear_file_substitutions`,
                     `cfa_del_file_substitution`,
                     `cfa_has_file_substitutions`,
                     `cfa_update_file_substitutions`
        :Returns:

            `dict`
                The CFA-netCDF file name substitutions.

        **Examples**

        >>> f.cfa_update_file_substitutions({'base': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_update_file_substitutions({'${base2}': '/home/data/'})
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/', '${base2}': '/home/data/'}
        >>> f.cfa_update_file_substitutions({'${base}': '/new/location/'})
        >>> f.cfa_file_substitutions()
        {'${base}': '/new/location/', '${base2}': '/home/data/'}
        >>> f.cfa_del_file_substitution('${base}')
        {'${base}': '/new/location/'}
        >>> f.cfa_clear_file_substitutions()
        {'${base2}': '/home/data/'}
        >>> f.cfa_has_file_substitutions()
        False
        >>> f.cfa_file_substitutions()
        {}
        >>> f.cfa_clear_file_substitutions()
        {}
        >>> print(f.cfa_del_file_substitution('base', None))
        None

        """
        out = self._nc_get("cfa_file_substitutions", default=None)
        if out is not None:
            return out.copy()

        return {}

    def cfa_has_file_substitutions(self):
        """Whether any CFA-netCDF file name substitutions have been set.

        .. versionadded:: 3.15.0

        .. seealso:: `cfa_clear_file_substitutions`,
                     `cfa_del_file_substitution`,
                     `cfa_file_substitutions`,
                     `cfa_update_file_substitutions`

        :Returns:

            `bool`
                `True` if any CFA-netCDF file name substitutions have
                been set, otherwise `False`.

        **Examples**

        >>> f.cfa_update_file_substitutions({'base': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_update_file_substitutions({'${base2}': '/home/data/'})
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/', '${base2}': '/home/data/'}
        >>> f.cfa_update_file_substitutions({'${base}': '/new/location/'})
        >>> f.cfa_file_substitutions()
        {'${base}': '/new/location/', '${base2}': '/home/data/'}
        >>> f.cfa_del_file_substitution('${base}')
        {'${base}': '/new/location/'}
        >>> f.cfa_clear_file_substitutions()
        {'${base2}': '/home/data/'}
        >>> f.cfa_has_file_substitutions()
        False
        >>> f.cfa_file_substitutions()
        {}
        >>> f.cfa_clear_file_substitutions()
        {}
        >>> print(f.cfa_del_file_substitution('base', None))
        None

        """
        return self._nc_has("cfa_file_substitutions")

    def cfa_update_file_substitutions(self, substitutions):
        """Set CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        .. seealso:: `cfa_clear_file_substitutions`,
                     `cfa_del_file_substitution`,
                     `cfa_file_substitutions`,
                     `cfa_has_file_substitutions`

        :Parameters:

            {{cfa substitutions: `dict`}}

        :Returns:

            `None`

        **Examples**

        >>> f.cfa_update_file_substitutions({'base': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.cfa_update_file_substitutions({'${base2}': '/home/data/'})
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/', '${base2}': '/home/data/'}
        >>> f.cfa_update_file_substitutions({'${base}': '/new/location/'})
        >>> f.cfa_file_substitutions()
        {'${base}': '/new/location/', '${base2}': '/home/data/'}
        >>> f.cfa_del_file_substitution('${base}')
        {'${base}': '/new/location/'}
        >>> f.cfa_clear_file_substitutions()
        {'${base2}': '/home/data/'}
        >>> f.cfa_has_file_substitutions()
        False
        >>> f.cfa_file_substitutions()
        {}
        >>> f.cfa_clear_file_substitutions()
        {}
        >>> print(f.cfa_del_file_substitution('base', None))
        None

        """
        if not substitutions:
            return

        substitutions = substitutions.copy()
        for base, sub in tuple(substitutions.items()):
            if not (base.startswith("${") and base.endswith("}")):
                substitutions[f"${{{base}}}"] = substitutions.pop(base)

        subs = self.cfa_file_substitutions()
        subs.update(substitutions)
        self._nc_set("cfa_file_substitutions", subs)
