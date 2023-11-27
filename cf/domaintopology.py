import cfdm

from . import mixin


class DomainTopology(mixin.PropertiesData, cfdm.DomainTopology):
    """A domain topology construct of the CF data model.

    A domain topology construct defines the geospatial topology of
    cells arranged in two or three dimensions in real space but
    indexed by a single (discrete) domain axis construct, and at most
    one domain topology construct may be associated with any such
    domain axis. The topology describes topological relationships
    between the cells - spatial relationships which do not depend on
    the cell locations - and is represented by an undirected graph,
    i.e. a mesh in which pairs of nodes are connected by links. Each
    node has a unique arbitrary identity that is independent of its
    spatial location, and different nodes may be spatially co-located.

    The topology may only describe cells that have a common spatial
    dimensionality, one of:

    * Point: A point is zero-dimensional and has no boundary vertices.
    * Edge: An edge is one-dimensional and corresponds to a line
            connecting two boundary vertices.
    * Face: A face is two-dimensional and corresponds to a surface
            enclosed by a set of edges.

    Each type of cell implies a restricted topology for which only
    some kinds of mesh are allowed. For point cells, every node
    corresponds to exactly one cell; and two cells have a topological
    relationship if and only if their nodes are connected by a mesh
    link. For edge and face cells, every node corresponds to a
    boundary vertex of a cell; the same node can represent vertices in
    multiple cells; every link in the mesh connects two cell boundary
    vertices; and two cells have a topological relationship if and
    only if they share at least one node.

    A domain topology construct contains an array defining the mesh,
    and properties to describe it. There must be a property indicating
    the spatial dimensionality of the cells. The array values comprise
    the node identities, and all array elements that refer to the same
    node must contain the same value, which must differ from any other
    value in the array. The array spans the domain axis construct and
    also has a ragged dimension, whose function depends on the spatial
    dimensionality of the cells.

    For each point cell, the first element along the ragged dimension
    contains the node identity of the cell, and the following elements
    contain in arbitrary order the identities of all the cells to
    which it is connected by a mesh link.

    For each edge or face cell, the elements along the ragged
    dimension contain the node identities of the boundary vertices of
    the cell, in the same order that the boundary vertices are stored
    by the auxiliary coordinate constructs. Each boundary vertex
    except the last is connected by a mesh link to the next vertex
    along the ragged dimension, and the last vertex is connected to
    the first.

    When a domain topology construct is present it is considered to be
    definitive and must be used in preference to the topology implied
    by inspection of any other constructs, which is not guaranteed to
    be the same.

    In CF-netCDF a domain topology construct can only be provided for
    a UGRID mesh topology variable. The information in the construct
    array is supplied by the UGRID "edge_nodes_connectivity" variable
    (for edge cells) or "face_nodes_connectivity" variable (for face
    cells). The topology for node cells may be provided by any of
    these three UGRID variables. The integer indices contained in the
    UGRID variable may be used as the mesh node identities, although
    the CF data model attaches no significance to the values other
    than the fact that some values are the same as others. The spatial
    dimensionality property is provided by the "location" attribute of
    a variable that references the UGRID mesh topology variable,
    i.e. a data variable or a UGRID location index set variable.

    A single UGRID mesh topology defines multiple domain constructs
    and defines how they relate to each other. For instance, when
    "face_node_connectivity" and "edge_node_connectivity" variables
    are both present there are three implied domain constructs - one
    each for face, edge and point cells - all of which have the same
    mesh and so are explicitly linked (e.g. it is known which edge
    cells define each face cell). The CF data model has no mechanism
    for explicitly recording such relationships between multiple
    domain constructs, however whether or not two domains have the
    same mesh may be reliably deternined by inspection, thereby
    allowing the creation of netCDF datasets containing UGRID mesh
    topology variables.

    The restrictions on the type of mesh that may be used with a given
    cell spatial dimensionality excludes some meshes which can be
    described by an undirected graph, but is consistent with UGRID
    encoding within CF-netCDF. UGRID also describes meshes for
    three-dimensional volume cells that correspond to a volume
    enclosed by a set of faces, but how the nodes relate to volume
    boundary vertices is undefined and so volume cells are currently
    omitted from the CF data model.

    See CF Appendix I "The CF Data Model".

    **NetCDF interface**

    The netCDF variable name of the UGRID mesh topology variable may
    be accessed with the `nc_set_variable`, `nc_get_variable`,
    `nc_del_variable`, and `nc_has_variable` methods.

    .. versionadded::  3.16.0

    """

    @property
    def cell(self):
        """The cell type.

        {{cell type}}

        .. versionadded:: 3.16.0

        .. seealso:: `del_cell`, `get_cell`, `has_cell`, `set_cell`

        """
        return self.get_cell(default=AttributeError())

    @cell.setter
    def cell(self, value):
        self.set_cell(value)

    @cell.deleter
    def cell(self):
        self.del_cell(default=AttributeError())

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

        * The cell type type, preceded by ``'cell:'``.
        * The `standard_name` property.
        * The `id` attribute, preceded by ``'id%'``.
        * The `long_name` property, preceded by ``'long_name='``.
        * The netCDF variable name, preceded by ``'ncvar%'``.
        * The value of the *default* parameter.

        .. versionadded:: 3.16.0

        .. seealso:: `id`, `identities`, `long_name`, `cell`,
                     `nc_get_variable`, `standard_name`

        :Parameters:

            default: optional
                If no identity can be found then return the value of the
                default parameter.

            strict: `bool`, optional
                If True then the identity is the first found of only
                the `cell` attribute, "standard_name" property or the
                "id" attribute.

            relaxed: `bool`, optional
                If True then the identity is the first found of only
                the `cell` attribute, the "standard_name" property,
                the "id" attribute, the "long_name" property or the
                netCDF variable name.

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

        n = self.get_cell(default=None)
        if n is not None:
            return f"cell:{n}"

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
