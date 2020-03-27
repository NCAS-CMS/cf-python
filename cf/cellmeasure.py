import cfdm

from . import mixin

from .decorators import _deprecated_kwarg_check


class CellMeasure(mixin.PropertiesData,
                  cfdm.CellMeasure):
    '''A cell measure construct of the CF data model.

    A cell measure construct provides information that is needed about
    the size or shape of the cells and that depends on a subset of the
    domain axis constructs. Cell measure constructs have to be used
    when the size or shape of the cells cannot be deduced from the
    dimension or auxiliary coordinate constructs without special
    knowledge that a generic application cannot be expected to have.

    The cell measure construct consists of a numeric array of the
    metric data which spans a subset of the domain axis constructs,
    and properties to describe the data. The cell measure construct
    specifies a "measure" to indicate which metric of the space it
    supplies, e.g. cell horizontal areas, and must have a units
    property consistent with the measure, e.g. square metres. It is
    assumed that the metric does not depend on axes of the domain
    which are not spanned by the array, along which the values are
    implicitly propagated. CF-netCDF cell measure variables correspond
    to cell measure constructs.

    **NetCDF interface**

    The netCDF variable name of the construct may be accessed with the
    `nc_set_variable`, `nc_get_variable`, `nc_del_variable` and
    `nc_has_variable` methods.

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def ismeasure(self):
        '''Always True.

    .. seealso:: `isauxiliary`, `isdimension`

    **Examples:**

    >>> c.ismeasure
    True

        '''
        return True

    @property
    def measure(self):
        '''TODO
        '''
        return self.get_measure(default=AttributeError())

    @measure.setter
    def measure(self, value): self.set_measure(value)

    @measure.deleter
    def measure(self):        self.del_measure(default=AttributeError())

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def creation_commands(self, representative_data=False,
                          namespace='cf', indent=0, string=True,
                          name='c', data_name='d'):
        '''Return the commands that would create the cell measure construct.

    .. versionadded:: 3.2.0

    .. seealso:: `cf.Data.creation_commands`,
                 `cf.Field.creation_commands`

    :Parameters:

        representative_data: `bool`, optional
            Return one-line representations of `Data` instances, which
            are not executable code but prevent the data being
            converted in its entirety to a string representation.

        namespace: `str`, optional
            The namespace containing classes of the ``cf``
            package. This is prefixed to the class name in commands
            that instantiate instances of ``cf`` objects. By default,
            *namespace* is ``'cf'``, i.e. it is assumed that ``cf``
            was imported as ``import cf``.

            *Parameter example:*
              If ``cf`` was imported as ``import cf as cfp`` then set
              ``namespace='cfp'``

            *Parameter example:*
              If ``cf`` was imported as ``from cf import *`` then set
              ``namespace=''``

        indent: `int`, optional
            Indent each line by this many spaces. By default no
            indentation is applied. Ignored if *string* is False.

        string: `bool`, optional
            If False then return each command as an element of a
            `list`. By default the commands are concatenated into
            a string, with a new line inserted between each command.

    :Returns:

        `str` or `list`
            The commands in a string, with a new line inserted between
            each command. If *string* is False then the separate
            commands are returned as each element of a `list`.

    **Examples:**

        TODO

        '''
        out = super().creation_commands(
            representative_data=representative_data, indent=indent,
            namespace=namespace, string=False, name=name,
            data_name=data_name)

        measure = self.get_measure(None)
        if measure is not None:
            out.append("{}.set_measure({!r})".format(name, measure))

        if string:
            out[0] = indent+out[0]
            out = ('\n'+indent).join(out)

        return out

    @_deprecated_kwarg_check('relaxed_identity')
    def identity(self, default='', strict=None, relaxed=False,
                 nc_only=False, relaxed_identity=None):
        '''Return the canonical identity.

    By default the identity is the first found of the following:

    * The `measure` attribute, preceeded by ``'measure:'``.
    * The `standard_name` property.
    * The `id` attribute, preceeded by ``'id%'``.
    * The `long_name` property, preceeded by ``'long_name='``.
    * The netCDF variable name, preceeded by ``'ncvar%'``.
    * The value of the *default* parameter.

    .. versionadded:: 3.0.0

    .. seealso:: `id`, `identities`, `long_name`, `measure`,
                 `nc_get_variable`, `standard_name`

    :Parameters:

        default: optional
            If no identity can be found then return the value of the
            default parameter.

        strict: `bool`, optional
            If True then the identity is the first found of only the
            "measure" arttribute, "standard_name" property or the "id"
            attribute.

        relaxed: `bool`, optional
            If True then the identity is the first found of only the
            "measure" arttribute, the "standard_name" property, the
            "id" attribute, the "long_name" property or the netCDF
            variable name.

        nc_only: `bool`, optional
            If True then only take the identity from the netCDF
            variable name.

    :Returns:

            The identity.

    **Examples:**

    >>> c.measure
    'area'
    >>> c.properties()
    {'long_name': 'cell_area',
     'foo': 'bar'}
    >>> c.nc_get_variable()
    'areacello'
    >>> c.identity()
    'measure:area'
    >>> del c.measure
    >>> c.identity()
    'long_name=cell_area'
    >>> del c.long_name
    >>> c.identity()
    'ncvar%areacello'
    >>> c.nc_del_variable()
    'areacello'
    >>> c.identity()
    ''
    >>> c.identity('no identity')
    'no identity'

        '''
        err_msg = "%{0} and 'nc_only' parameters cannot both be True"
        if nc_only:
            if strict:
                raise ValueError(err_msg.format("'strict'"))

            if relaxed:
                raise ValueError(err_msg.format("'relaxed'"))

            n = self.nc_get_variable(None)
            if n is not None:
                return 'ncvar%{0}'.format(n)

            return default

        n = self.get_measure(default=None)
        if n is not None:
            return 'measure:{0}'.format(n)

        n = self.get_property('standard_name', None)
        if n is not None:
            return '{0}'.format(n)

        n = getattr(self, 'id', None)
        if n is not None:
            return 'id%{0}'.format(n)

        if relaxed:
            n = self.get_property('long_name', None)
            if n is not None:
                return 'long_name={0}'.format(n)

            n = self.nc_get_variable(None)
            if n is not None:
                return 'ncvar%{0}'.format(n)

            return default

        if strict:
            return default

        for prop in ('long_name',):
            n = self.get_property(prop, None)
            if n is not None:
                return '{0}={1}'.format(prop, n)
        # --- End: for

        n = self.nc_get_variable(None)
        if n is not None:
            return 'ncvar%{0}'.format(n)

        return default


# --- End: class
