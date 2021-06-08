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

    Stores an `ESMF.Regrid` regridding operator and its associated
    destination `Field` construct.

    """

    def __init__(self, regrid, dst):
        """**Initialization**

        :Parameters:

            regrid: `ESMF.Regrid`
                The `ESMF.Regrid` regridding operator between two
                fields.

            dst: `Field`
                The destination field construct associated with the
                `ESMF.Rerid` regriddinging operator.

        """
        self._regrid = regrid
        self._dst = dst

    def __del__(self):
        """Call the `ESMF.Regrid` destroy method."""
        self._regrid.destroy()

    @property
    def dst(self):
        """The contained destination field of the regridding operator.

        **Examples:**

        >>> type(r.dst)
        cf.field.Field

        """
        return self._dst

    @property
    def method(self):
        """The name of the regrid method.

        **Examples:**

        >>> r.method
        'conservative'

        """
        method = regrid_method_map_inverse[self._regrid.regrid_method]
        if method == "bilinear":
            method = "linear"

        return method

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
        copied and the contained `Field` instance (see `dst`) is deep
        copied.

        :Returns:

            `RegridOperator`

                The copy.

        **Examples:**

        >>> s = r.copy()

        """
        return type(self)(regrid=self._regrid.copy(), dst=self._dst.copy())

    def destroy(self):
        """Free the memory allocated by the contained `ESMF.Regrid`
        instance.

        **Examples:**

        >>> r.destroy()

        """
        self._regrid.destroy()
