from .. import _found_ESMF


if _found_ESMF:
    try:
        import ESMF
    except Exception as error:
        print(f"WARNING: Can not import ESMF for regridding: {error}")

regrid_method_map = {
    "linear": ESMF.RegridMethod.BILINEAR,  # see comment below...
    "bilinear": ESMF.RegridMethod.BILINEAR,  # (for back compat)
    "conservative": ESMF.RegridMethod.CONSERVE,
    "conservative_1st": ESMF.RegridMethod.CONSERVE,
    "conservative_2nd": ESMF.RegridMethod.CONSERVE_2ND,
    "nearest_dtos": ESMF.RegridMethod.NEAREST_DTOS,
    "nearest_stod": ESMF.RegridMethod.NEAREST_STOD,
    "patch": ESMF.RegridMethod.PATCH,
}
# ... diverge from ESMF with respect to name for bilinear method by
# using 'linear' because 'bi' implies 2D linear interpolation, which
# could mislead or confuse for Cartesian regridding in 1D or 3D.

regrid_method_map_inverse = {v: k for k, v in regrid_method_map.items()}

conservative_regridding_methods = (
    "conservative",
    "conservative_1st",
    "conservative_2nd",
)

regridding_methods = (
    "linear",  # prefer over 'bilinear' as of v3.2.0
    "bilinear",  # only for backward compatibility, use & document 'linear'
    "patch",
    "nearest_stod",
    "nearest_dtos",
) + conservative_regridding_methods


class RegridOperator:
    """A regridding operator between two fields.

    Stores an `ESMF.Regrid` regridding operator and associated
    parameters that describe the complete coordinate system of the
    destination grid.

    """

    def __init__(self, regrid, name, **parameters):
        """**Initialization**

        :Parameters:

            regrid: `ESMF.Regrid`
                The `ESMF` regridding operator between two fields.

            name: `str`
                A name that defines the context of the destination
                grid parameters.

            parameters: `dict`
               Parameters that describe the complete coordinate system
               of the destination grid.

               Any parameter names and values are allowed, and it is
               assumed that the these are well defined during the
               creation and subsequent use of a `RegridOperator`
               instance.

        """
        self._regrid = regrid
        self._name = name
        self._parameters = parameters

    def __del__(self):
        """Calls the `ESMF.Regrid` destroy method."""
        self._regrid.destroy()

    def __repr__(self):
        return (
            f"<CF {self.__class__.__name__}: "
            f"{self._name}, method={self.method}>"
        )

    @property
    def method(self):
        """The regridding method.

        **Examples:**

        >>> r.method
        'conservative'

        """
        method = regrid_method_map_inverse[self._regrid.regrid_method]
        if method == "bilinear":
            method = "linear"

        return method

    @property
    def name(self):
        """The name of the regrid method.

        **Examples:**

        >>> r.name
        'regrids'

        """
        return self._name

    @property
    def parameters(self):
        """The parameters that describe the destination grid.

        Any parameter names and values are allowed, and it is assumed
        that the these are well defined during the creation and
        subsequent use of a `RegridOperator` instance.

        **Examples:**

        >>> type(r.parameters)
        dict

        """
        return self._parameters

    @property
    def regrid(self):
        """The contained regridding operator.

        **Examples:**

        >>> type(r.regrid)
        ESMF.api.regrid.Regrid

        """
        return self._regrid

    def check_method(self, method):
        """Whether the given method is equivalent to the regridding
        method.

        :Parameters:

            method: `str`
                A regridding method, such as ``'conservative'``.

        :Returns:

            `bool`
                Whether or not method is equivalent to the regridding
                method.

        **Examples:**

        >>> r.method
        'conservative'
        >>> r.check_method('conservative')
        True
        >>> r.check_method('conservative_1st')
        True
        >>> r.check_method('conservative_2nd')
        False

        """
        return regrid_method_map.get(method) == self._regrid.regrid_method

    def copy(self):
        """Return a copy.

        The contained `ESMF.Regrid` instance (see `regrid`) is shallow
        copied and the "dst" parameter, which is a `Field` or `dict`
        instance, is copied with its `!copy` method.

        :Returns:

            `RegridOperator`
                The copy.

        **Examples:**

        >>> s = r.copy()

        """
        parameters = self._parameters
        if "dst" in parameters:
            parameters = parameters.copy()
            parameters["dst"] = parameters["dst"].copy()

        return type(self)(
            regrid=self._regrid.copy(), name=self._name, **parameters
        )

    def destroy(self):
        """Free the memory allocated by the `ESMF.Regrid` instance.

        **Examples:**

        >>> r.destroy()

        """
        self._regrid.destroy()

    def get_parameter(self, parameter):
        """Return a regrid operation parameter.

        **Examples:**

        >>> r.get_parameter('ignore_degenerate')
        True
        >>> r.get_parameter('x')
        Traceback
            ...
        ValueError: RegridOperator has no 'x' parameter

        """
        try:
            return self._parameters[parameter]
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__} has no {parameter!r} parameter"
            )
