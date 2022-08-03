import cfdm

from .functions import inspect as cf_inspect


class DomainAxis(cfdm.DomainAxis):
    """A domain axis construct of the CF data model.

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

    """

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)

    def __hash__(self):
        """Returns the hash value of the domain axis."""
        return hash(
            (
                self.__class__.__name__,
                self.get_size(None),
                self.nc_get_dimension(),
            )
        )

    def __eq__(self, other):
        """The rich comparison operator ``==``.

        x.__eq__(y) <==> x.size==y

        """
        return self.get_size() == int(other)

    def __ne__(self, other):
        """The rich comparison operator ``!=``.

        x.__ne__(y) <==> x.size!=y

        """
        return self.get_size() != int(other)

    def __gt__(self, other):
        """The rich comparison operator ``>``.

        x.__gt__(y) <==> x.size>y

        """
        return self.get_size() > int(other)

    def __ge__(self, other):
        """The rich comparison operator ``>=``.

        x.__ge__(y) <==> x.size>=y

        """
        return self.get_size() >= int(other)

    def __lt__(self, other):
        """The rich comparison operator ``<``.

        x.__lt__(y) <==> x.size<y

        """
        return self.get_size() < int(other)

    def __le__(self, other):
        """The rich comparison operator ``<=``.

        x.__le__(y) <==> x.size<=y

        """
        return self.get_size() <= int(other)

    def __add__(self, other):
        """The binary arithmetic operation ``+``.

        x.__add__(y) <==> x+y

        """
        new = self.copy()
        new.set_size(self.get_size() + int(other))
        return new

    def __radd__(self, other):
        """The binary arithmetic operation ``+`` with reflected
        operands.

        x.__radd__(y) <==> y+x

        """
        return self + other

    def __iadd__(self, other):
        """The augmented arithmetic assignment ``+=``.

        x.__iadd__(y) <==> x+=y

        """
        self.set_size(self.get_size() + int(other))
        return self

    def __sub__(self, other):
        """The binary arithmetic operation ``-``.

        x.__sub__(y) <==> x - y

        """
        new = self.copy()
        new.set_size(self.get_size() - int(other))
        return new

    def __isub__(self, other):
        """The augmented arithmetic assignment ``-=``.

        x.__isub__(y) <==> x -= y

        """
        self.set_size(self.get_size() - int(other))
        return self

    def __int__(self):
        """Implements the built-in function `int`.

        x.__int__() <==> int(x)

        """
        return self.get_size()

    @property
    def size(self):
        """The domain axis size.

        .. seealso:: `del_size`, `get_size`, `has_size`, `set_size`

        **Examples**

        >>> d.size = 96
        >>> d.size
        96
        >>> del d.size
        >>> hasattr(d, 'size')
        False

        """
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
    def inspect(self):
        """Inspect the object for debugging.

        .. seealso:: `cf.inspect`

        :Returns:

            `None`

        **Examples**

        >>> d.inspect()

        """
        print(cf_inspect(self))  # pragma: no cover
