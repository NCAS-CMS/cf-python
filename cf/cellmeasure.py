import cfdm

from . import mixin
from .decorators import _deprecated_kwarg_check


class CellMeasure(mixin.PropertiesData, cfdm.CellMeasure):
    """A cell measure construct of the CF data model.

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

    {{netCDF variable}}

    """

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def measure(self):
        """Measure which indicates the metric of space supplied."""
        return self.get_measure(default=AttributeError())

    @measure.setter
    def measure(self, value):
        self.set_measure(value)

    @measure.deleter
    def measure(self):
        self.del_measure(default=AttributeError())

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    @_deprecated_kwarg_check(
        "relaxed_identity", version="3.0.0", removed_at="4.0.0"
    )
    def identity(
        self,
        default="",
        strict=None,
        relaxed=False,
        nc_only=False,
        relaxed_identity=None,
    ):
        """Return the canonical identity.

        By default the identity is the first found of the following:

        * The `measure` attribute, preceded by ``'measure:'``.
        * The `standard_name` property.
        * The `id` attribute, preceded by ``'id%'``.
        * The `long_name` property, preceded by ``'long_name='``.
        * The netCDF variable name, preceded by ``'ncvar%'``.
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
                "measure" attribute, "standard_name" property or the "id"
                attribute.

            relaxed: `bool`, optional
                If True then the identity is the first found of only the
                "measure" attribute, the "standard_name" property, the
                "id" attribute, the "long_name" property or the netCDF
                variable name.

            nc_only: `bool`, optional
                If True then only take the identity from the netCDF
                variable name.

        :Returns:

                The identity.

        **Examples**

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

        """
        err_msg = "%{0} and 'nc_only' parameters cannot both be True"
        if nc_only:
            if strict:
                raise ValueError(err_msg.format("'strict'"))

            if relaxed:
                raise ValueError(err_msg.format("'relaxed'"))

            n = self.nc_get_variable(None)
            if n is not None:
                return f"ncvar%{n}"

            return default

        n = self.get_measure(default=None)
        if n is not None:
            return f"measure:{n}"

        n = self.get_property("standard_name", None)
        if n is not None:
            return f"{n}"

        n = getattr(self, "id", None)
        if n is not None:
            return f"id%{n}"

        if relaxed:
            if strict:
                raise ValueError(
                    "'relaxed' and 'strict' parameters cannot both be True"
                )

            n = self.get_property("long_name", None)
            if n is not None:
                return f"long_name={n}"

            n = self.nc_get_variable(None)
            if n is not None:
                return f"ncvar%{n}"

            return default

        if strict:
            return default

        n = self.get_property("long_name", None)
        if n is not None:
            return f"long_name={n}"

        n = self.nc_get_variable(None)
        if n is not None:
            return f"ncvar%{n}"

        return default
