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


class RegridOperator:
    """Class containing all the methods required for accessing ESMF
    regridding through ESMPY and the associated utility methods."""

    def __init__(self, regrid, dst):
        """Creates a handle for regridding fields from a source grid to
        a destination grid that can then be used by the run_regridding
        method.

        :Parameters:

            regrid: `ESMF.Regrid`
                The source field with an associated grid to be used
                for regridding.

            dst: `Field`
                The destination field construct with an associated
                grid to be used for regridding.

        """
        self._regrid = regrid
        self._dst = dst

    def __del__(self):
        """TODO"""
        self._regrid.destroy()

    @property
    def dst(self):
        """The contained destination field of the regridding operator.

        **Examples:**

        >>> type(r.dst)
        TODO

        """
        return self._dst

    @property
    def method(self):
        """The name of the regrid method.

        **Examples:**

        >>> r.method
        'conservative'

        """
        return regrid_method_map_inverse[self._regrid.regrid_method]

    @property
    def regrid(self):
        """The contained regridding operator.

        **Examples:**

        >>> type(r.regrid)
        TODO

        """
        return self._regrid

    def check_method(self, method):
        """Check the given method against the ESMF regrid method.

        :Parameters:

            method: `str`
                TODO

        :Returns:

            `bool`
                TODO

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
        """Free the memory allocated by the contained `ESMF.Regrid` instance.

        **Examples:**

        >>> r.destroy()

        """
        self._regrid.destroy()
