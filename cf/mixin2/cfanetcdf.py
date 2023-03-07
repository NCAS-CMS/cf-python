"""This class is not in the cf.mixin package because it needs to be
imported by cf.Data, and some of the other mixin classes in cf.mixin
themsleves import cf.Data, which would lead to a circular import
situation.

"""
from re import split

from cfdm.mixin import NetCDFMixin


class CFANetCDF(NetCDFMixin):
    """Mixin class for CFA-netCDF.

    .. versionadded:: TODOCFAVER

    """

    def cfa_del_aggregated_data(self, default=ValueError()):
        """Remove the CFA-netCDF aggregation instruction terms.

        The aggregation instructions are stored in the
        `aggregation_data` attribute of a CFA-netCDF aggregation
        variable.

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

        The aggregation instructions are stored in the
        `aggregation_data` attribute of a CFA-netCDF aggregation
        variable.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_del_aggregated_data`,
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

        The aggregation instructions are stored in the
        `aggregation_data` attribute of a CFA-netCDF aggregation
        variable.

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

        The aggregation instructions are stored in the
        `aggregation_data` attribute of a CFA-netCDF aggregation
        variable.

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
        >>> print(f.cfa_get_aggregated_data(None))
        None
        >>> print(f.cfa_del_aggregated_data(None))
        None

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
        """Remove the CFA-netCDF file name substitutions.

        The file substitutions are stored in the `substitutions`
        attribute of a CFA-netCDF `file` aggregation aggregation
        instruction term.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_del_file_substitution`,
                     `cfa_file_substitutions`,
                     `cfa_has_file_substitutions`,
                     `cfa_set_file_substitutions`

        :Returns:

            `dict`
                The removed CFA-netCDF file name substitutions.

        **Examples**

        >>> f.`cfa_set_file_substitutions({'base': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.`cfa_set_file_substitutions({'${base2}': '/home/data/'})
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/', '${base2}': '/home/data/'}
        >>> f.`cfa_set_file_substitutions({'${base}': '/new/location/'})
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

    def cfa_del_file_substitution(self, base, default=ValueError()):
        """Remove the CFA-netCDF file name substitutions.

        The file substitutions are stored in the `substitutions`
        attribute of a CFA-netCDF `file` aggregation aggregation
        instruction term.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_clear_file_substitutions`,
                     `cfa_file_substitutions`,
                     `cfa_has_file_substitutions`,
                     `cfa_set_file_substitutions`

        :Parameters:

            base: `str`
                The substition definition to be removed. May be
                specified with or without the ``${...}`` syntax. For
                instance, the following are equivalent: ``'base'``,
                ``'${base}'``.

            default: optional
                Return the value of the *default* parameter if file
                name substitution has not been set. If set to an
                `Exception` instance then it will be raised instead.

        :Returns:

            `dict`
                The removed CFA-netCDF file name substitutions.

        **Examples**

        >>> f.cfa_set_file_substitutions({'base': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.`cfa_set_file_substitutions({'${base2}': '/home/data/'})
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/', '${base2}': '/home/data/'}
        >>> f.`cfa_set_file_substitutions({'${base}': '/new/location/'})
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
        if not (base.startswith("${") and base.endswith("}")):
            base = f"${{{base}}}"

        subs = self.cfa_file_substitutions()
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

    def cfa_file_substitutions(self):
        """Return the CFA-netCDF file name substitutions.

        The file substitutions are stored in the `substitutions`
        attribute of a CFA-netCDF `file` aggregation aggregation
        instruction term.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_clear_file_substitutions`,
                     `cfa_del_file_substitution`,
                     `cfa_file_substitutions`,
                     `cfa_set_file_substitution`
        :Returns:

            value: `dict`
                The CFA-netCDF file name substitutions.

        **Examples**

        >>> f.`cfa_set_file_substitutions({'base': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.`cfa_set_file_substitutions({'${base2}': '/home/data/'})
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/', '${base2}': '/home/data/'}
        >>> f.`cfa_set_file_substitutions({'${base}': '/new/location/'})
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

        The file substitutions are stored in the `substitutions`
        attribute of a CFA-netCDF `file` aggregation aggregation
        instruction term.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_clear_file_substitutions`,
                     `cfa_del_file_substitution`,
                     `cfa_file_substitutions`,
                     `cfa_set_file_substitutions`

        :Returns:

            `bool`
                `True` if any CFA-netCDF file name substitutions have
                been set, otherwise `False`.

        **Examples**

        >>> f.`cfa_set_file_substitutions({'base': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.`cfa_set_file_substitutions({'${base2}': '/home/data/'})
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/', '${base2}': '/home/data/'}
        >>> f.`cfa_set_file_substitutions({'${base}': '/new/location/'})
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

    def cfa_set_file_substitutions(self, value):
        """Set CFA-netCDF file name substitutions.

        The file substitutions are stored in the `substitutions`
        attribute of a CFA-netCDF `file` aggregation aggregation
        instruction term.

        .. versionadded:: TODOCFAVER

        .. seealso:: `cfa_clear_file_substitutions`,
                     `cfa_del_file_substitution`,
                     `cfa_file_substitutions`,
                     `cfa_has_file_substitutions`

        :Parameters:

            value: `str` or `dict`
                The substition definitions in a dictionary whose
                key/value pairs are the file name parts to be
                substituted and their corresponding substitution text.

                The substition definition may be specified with or
                without the ``${...}`` syntax. For instance, the
                following are equivalent: ``{'base': 'sub'}``,
                ``{'${base}': 'sub'}``.

        :Returns:

            `None`

        **Examples**

        >>> f.`cfa_set_file_substitutions({'base': 'file:///data/'})
        >>> f.cfa_has_file_substitutions()
        True
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/'}
        >>> f.`cfa_set_file_substitutions({'${base2}': '/home/data/'})
        >>> f.cfa_file_substitutions()
        {'${base}': 'file:///data/', '${base2}': '/home/data/'}
        >>> f.`cfa_set_file_substitutions({'${base}': '/new/location/'})
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
        if not value:
            return

        value = value.copy()
        for base, sub in tuple(value.items()):
            if not (base.startswith("${") and base.endswith("}")):
                value[f"${{{base}}}"] = value.pop(base)

        subs = self.cfa_file_substitutions()
        subs.update(value)
        self._nc_set("cfa_file_substitutions", subs)
