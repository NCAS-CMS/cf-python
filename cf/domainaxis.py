import cfdm

from .functions import inspect as cf_inspect


class DomainAxis(cfdm.DomainAxis):
    '''A domain axis construct of the CF data model.

    A domain axis construct specifies the number of points along an
    independent axis of the domain. It comprises a positive integer
    representing the size of the axis. In CF-netCDF it is usually
    defined either by a netCDF dimension or by a scalar coordinate
    variable, which implies a domain axis of size one. The field
    construct's data array spans the domain axis constructs of the
    domain, with the optional exception of size one axes, because
    their presence makes no difference to the order of the elements.

    **NetCDF interface**

    The netCDF dimension name of the construct may be accessed with
    the `nc_set_dimension`, `nc_get_dimension`, `nc_del_dimension` and
    `nc_has_dimension` methods.

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)

    def __hash__(self):
        '''TODO

        '''
        return hash((self.__class__.__name__,
                     self.get_size(None),
                     self.nc_get_dimension()))

    def __eq__(self, other):
        '''The rich comparison operator ``==``

    x.__eq__(y) <==> x.size==y

        '''
        return self.get_size() == int(other)

    def __ne__(self, other):
        '''The rich comparison operator ``!=``

    x.__ne__(y) <==> x.size!=y

        '''
        return self.get_size() != int(other)

    def __gt__(self, other):
        '''The rich comparison operator ``>``

    x.__gt__(y) <==> x.size>y

        '''
        return self.get_size() > int(other)

    def __ge__(self, other):
        '''The rich comparison operator ``>=``

    x.__ge__(y) <==> x.size>=y

        '''
        return self.get_size() >= int(other)

    def __lt__(self, other):
        '''The rich comparison operator ``<``

    x.__lt__(y) <==> x.size<y

        '''
        return self.get_size() < int(other)

    def __le__(self, other):
        '''The rich comparison operator ``<=``

    x.__le__(y) <==> x.size<=y

        '''
        return self.get_size() <= int(other)

    def __add__(self, other):
        '''TODO
        '''
        new = self.copy()
        new.set_size(self.get_size() + int(other))
        return new

    def __radd__(self, other):
        '''TODO
        '''
        return self + other

    def __iadd__(self, other):
        '''TODO
        '''
        self.set_size(self.get_size() + int(other))
        return self

    def __sub__(self, other):
        '''TODO
        '''
        new = self.copy()
        new.set_size(self.get_size() - int(other))
        return new

    def __isub__(self, other):
        '''TODO
        '''
        self.set_size(self.get_size() - int(other))
        return self

    def __int__(self):
        '''TODO

    x.__int__() <==> int(x)
        '''
        return self.get_size()

    @property
    def size(self):
        '''The domain axis size.

    .. seealso:: `del_size`, `get_size`, `has_size`, `set_size`

    **Examples:**

    >>> d.size = 96
    >>> d.size
    96
    >>> del d.size
    >>> hasattr(d, 'size')
    False

        '''
        return self.get_size(default=AttributeError())

    @size.setter
    def size(self, value):
        self.set_size(value)

    @size.deleter
    def size(self):
        self.del_size(default=AttributeError())

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def creation_commands(self, representative_data=False,
                          namespace='cf', indent=0, string=True,
                          name='c'):
        '''Return the commands that would create the domain axis construct.

    .. versionadded:: 3.2.0

    .. seealso:: `cf.Field.creation_commands`

    :Parameters:

        representative_data: `bool`, optional
            Ignored.

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
        namespace0 = namespace
        if namespace0:
            namespace = namespace+"."
        else:
            namespace = ""

        indent = ' ' * indent

        out = []
        out.append("# {}: {}".format(self.construct_type, self.identity()))
        out.append("{} = {}{}()".format(name, namespace,
                                        self.__class__.__name__))

        size = self.get_size(None)
        if size is not None:
            out.append("{}.set_size({})".format(name, size))

        nc = self.nc_get_dimension(None)
        if nc is not None:
            out.append("{}.nc_set_dimension({!r})".format(name, nc))

        if self.nc_is_unlimited():
            out.append("c.nc_set_unlimited({})".format(True))

        if string:
            out[0] = indent+out[0]
            out = ('\n'+indent).join(out)

        return out

    def inspect(self):
        '''Inspect the object for debugging.

    .. seealso:: `cf.inspect`

    :Returns:

        `None`

    **Examples:**

    >>> d.inspect()

        '''
        print(cf_inspect(self))  # pragma: no cover

# --- End: class
