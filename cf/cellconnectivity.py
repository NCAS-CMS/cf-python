import cfdm

from . import mixin


class CellConnectivity(mixin.PropertiesData, cfdm.CellConnectivity):
    """A cell connectivity construct of the CF data model.

    A cell connectivity construct defines explicitly how cells
    arranged in two or three dimensions in real space but indexed by a
    single domain (discrete) axis are connected. Connectivity can only
    be provided when the domain axis construct also has a domain
    topology construct, and two cells can only be connected if they
    also have a topological relationship. For instance, the
    connectivity of two-dimensional face cells could be characterised
    by whether or not they have shared edges, where the edges are
    defined by connected nodes of the domain topology construct.

    The cell connectivity construct consists of an array recording the
    connectivity, and properties to describe the data. There must be a
    property indicating the condition by which the connectivity is
    derived from the domain topology. The array spans the domain axis
    construct with the addition of a ragged dimension. For each cell,
    the first element along the ragged dimension contains the unique
    identity of the cell, and the following elements contain in
    arbitrary order the identities of all the other cells to which the
    cell is connected. Note that the connectivity array for point
    cells is, by definition, equivalent to the array of the domain
    topology construct.

    When cell connectivity constructs are present they are considered
    to define the connectivity of the cells. Exactly the same
    connectivity information could be derived from the domain topology
    construct. Connectivity information inferred from inspection of
    any other constructs is not guaranteed to be the same.

    In CF-netCDF a cell topology construct can only be provided by a
    UGRID mesh topology variable. The construct array is supplied
    either indirectly by any of the UGRID variables that are used to
    define a domain topology construct, or directly by the UGRID
    "face_face_connectivity" variable (for face cells). In the direct
    case, the integer indices contained in the UGRID variable may be
    used as the cell identities, although the CF data model attaches
    no significance to the values other than the fact that some values
    are the same as others.

    Restricting the types of connectivity to those implied by the
    geospatial topology of the cells precludes connectivity derived
    from any other sources, but is consistent with UGRID encoding
    within CF-netCDF.

    See CF Appendix I "The CF Data Model".

    **NetCDF interface**

    {{netCDF variable}}

    .. versionadded:: 3.16.0

    """

    @property
    def connectivity(self):
        """The connectivity type.

        {{cell connectivity type}}

        .. versionadded:: 3.16.0

        .. seealso:: `del_connectivity`, `get_connectivity`,
                     `has_connectivity`, `set_connectivity`

        """
        return self.get_connectivity(default=AttributeError())

    @connectivity.setter
    def connectivity(self, value):
        self.set_connectivity(value)

    @connectivity.deleter
    def connectivity(self):
        self.del_connectivity(default=AttributeError())

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

        * The connectivity type type, preceded by ``'connectivity:'``.
        * The `standard_name` property.
        * The `id` attribute, preceded by ``'id%'``.
        * The `long_name` property, preceded by ``'long_name='``.
        * The netCDF variable name, preceded by ``'ncvar%'``.
        * The value of the *default* parameter.

        .. versionadded:: 3.16.0

        .. seealso:: `id`, `identities`, `long_name`, `connectivity`,
                     `nc_get_variable`, `standard_name`

        :Parameters:

            default: optional
                If no identity can be found then return the value of the
                default parameter.

            strict: `bool`, optional
                If True then the identity is the first found of only
                the "connectivity" attribute, "standard_name" property
                or the "id" attribute.

            relaxed: `bool`, optional
                If True then the identity is the first found of only
                the "connectivity" attribute, the "standard_name"
                property, the "id" attribute, the "long_name" property
                or the netCDF variable name.

            nc_only: `bool`, optional
                If True then only take the identity from the netCDF
                variable name.

        :Returns:

                The identity.

        """
        if nc_only:
            if strict:
                raise ValueError(
                    "'strict' and 'nc_only' parameters cannot both be True"
                )

            if relaxed:
                raise ValueError(
                    "'relaxed' and 'nc_only' parameters cannot both be True"
                )

            n = self.nc_get_variable(None)
            if n is not None:
                return f"ncvar%{n}"

            return default

        n = self.get_connectivity(default=None)
        if n is not None:
            return f"connectivity:{n}"

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
