import cfdm

from .mixin_container import Container


class Weights(Container, cfdm.Container):
    """Worker functions for `cf.Field.weights`.

    .. versionadded:: 3.16.0

    """

    @classmethod
    def area_XY(
        cls,
        f,
        weights,
        weights_axes,
        auto=False,
        measure=False,
        radius=None,
        methods=False,
    ):
        """Calculate area weights from X and Y dimension coordinate
        constructs.

        .. versionadded:: 3.16.0

        :Parameters:

            {{weights weights: `dict`}}

            {{weights weights_axes: `set`}}

            {{weights auto: `bool`, optional}}

            {{weights measure: `bool`, optional}}

            {{radius: optional}}

            {{weights methods: `bool`, optional}}

        :Returns:

            `bool`
                `True` if weights were created, otherwise `False`.

        """
        from .units import Units

        xkey, xcoord = f.dimension_coordinate(
            "X", item=True, default=(None, None)
        )
        ykey, ycoord = f.dimension_coordinate(
            "Y", item=True, default=(None, None)
        )

        if xcoord is None or ycoord is None:
            if auto:
                return False

            raise ValueError(
                "No unique X and Y dimension coordinate constructs for "
                "calculating area weights"
            )

        radians = Units("radians")
        metres = Units("metres")

        if xcoord.Units.equivalent(radians) and ycoord.Units.equivalent(
            radians
        ):
            pass
        elif xcoord.Units.equivalent(metres) and ycoord.Units.equivalent(
            metres
        ):
            radius = None
        else:
            if auto:
                return False

            raise ValueError(
                "Insufficient coordinate constructs for calculating "
                "area weights"
            )

        xaxis = f.get_data_axes(xkey)[0]
        yaxis = f.get_data_axes(ykey)[0]

        for axis in (xaxis, yaxis):
            if axis in weights_axes:
                if auto:
                    return False

                raise ValueError(
                    "Multiple weights specifications for "
                    f"{f.constructs.domain_axis_identity(axis)!r} axis"
                )

        if measure and radius is not None:
            radius = f.radius(default=radius)

        if measure or xcoord.size > 1:
            if not xcoord.has_bounds():
                if auto:
                    return False

                raise ValueError(
                    "Can't create area weights: No bounds for "
                    f"{xcoord.identity()!r} axis"
                )

            if methods:
                weights[(xaxis,)] = f"linear {xcoord.identity()}"
            else:
                cells = xcoord.cellsize
                if xcoord.Units.equivalent(radians):
                    cells.Units = radians
                    if measure:
                        cells = cells * radius
                        cells.override_units(radius.Units, inplace=True)
                else:
                    cells.Units = metres

                weights[(xaxis,)] = cells

            weights_axes.add(xaxis)

        if measure or ycoord.size > 1:
            if not ycoord.has_bounds():
                if auto:
                    return False

                raise ValueError(
                    "Can't create area weights: No bounds for "
                    f"{ycoord.identity()!r} axis"
                )

            if ycoord.Units.equivalent(radians):
                ycoord = ycoord.clip(-90, 90, units=Units("degrees"))
                ysin = ycoord.sin()
                if methods:
                    weights[(yaxis,)] = f"linear sine {ycoord.identity()}"
                else:
                    cells = ysin.cellsize
                    if measure:
                        cells = cells * radius

                    weights[(yaxis,)] = cells
            else:
                if methods:
                    weights[(yaxis,)] = f"linear {ycoord.identity()}"
                else:
                    cells = ycoord.cellsize
                    weights[(yaxis,)] = cells

            weights_axes.add(yaxis)

        return True

    @classmethod
    def data(
        cls,
        f,
        w,
        weights,
        weights_axes,
        axes=None,
        data=False,
        components=False,
        methods=False,
    ):
        """Create weights from `Data`.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            w: `Data`
                Create weights from this data, that must be
                broadcastable to the data of *f*, unless the *axes*
                parameter is also set.

            axes: (sequence of) `int` or `str`, optional
               If not `None` then weights are created only for the
               specified axes.

            {{weights weights: `dict`}}

            {{weights weights_axes: `set`}}

            data: `bool`, optional
                If True then create weights in a `Data` instance that
                is broadcastable to the original data.

            components: `bool`, optional
                If True then the *weights* dictionary is updated with
                orthogonal weights components.

            {{weights methods: `bool`, optional}}

        :Returns:

            `bool`
                `True` if weights were created, otherwise `False`.

        """
        # ------------------------------------------------------------
        # Data weights
        # ------------------------------------------------------------
        field_data_axes = f.get_data_axes()

        if axes is not None:
            if isinstance(axes, (str, int)):
                axes = (axes,)

            if len(axes) != w.ndim:
                raise ValueError(
                    "'axes' parameter must provide an axis identifier "
                    f"for each weights data dimension. Got {axes!r} for "
                    f"{w.ndim} dimension(s)."
                )

            iaxes = [
                field_data_axes.index(f.domain_axis(axis, key=True))
                for axis in axes
            ]

            if data:
                for i in range(f.ndim):
                    if i not in iaxes:
                        w = w.insert_dimension(position=i)
                        iaxes.insert(i, i)

                w = w.transpose(iaxes)

                if w.ndim > 0:
                    while w.shape[0] == 1:
                        w = w.squeeze(0)

        else:
            iaxes = list(range(f.ndim - w.ndim, f.ndim))

        if not (components or methods):
            if not f._is_broadcastable(w.shape):
                raise ValueError(
                    f"The 'Data' weights (shape {w.shape}) are not "
                    "broadcastable to the field construct's data "
                    f"(shape {f.shape})."
                )

            axes0 = field_data_axes[f.ndim - w.ndim :]
        else:
            axes0 = [field_data_axes[i] for i in iaxes]

        for axis0 in axes0:
            if axis0 in weights_axes:
                raise ValueError(
                    "Multiple weights specified for "
                    f"{f.constructs.domain_axis_identity(axis0)!r} axis"
                )

        if methods:
            weights[tuple(axes0)] = "custom data"
        else:
            weights[tuple(axes0)] = w

        weights_axes.update(axes0)
        return True

    @classmethod
    def field(cls, f, field_weights, weights, weights_axes):
        """Create weights from other field constructs.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            field_weights: sequence of `Field`
                The field constructs from which to create weights.

            {{weights weights: `dict`}}

            {{weights weights_axes: `set`}}

        :Returns:

            `bool`
                `True` if weights were created, otherwise `False`.

        """
        s = f.analyse_items()

        domain_axes = f.domain_axes(todict=True)
        for w in field_weights:
            t = w.analyse_items()

            if t["undefined_axes"]:
                w_domain_axes_1 = w.domain_axes(
                    filter_by_size=(1,), todict=True
                )
                if set(w_domain_axes_1).intersection(t["undefined_axes"]):
                    raise ValueError(
                        f"Weights field {w} has at least one undefined "
                        f"domain axes: {w_domain_axes_1}."
                    )

            w = w.squeeze()

            w_domain_axes = w.domain_axes(todict=True)

            axis1_to_axis0 = {}

            coordinate_references = f.coordinate_references(todict=True)
            w_coordinate_references = w.coordinate_references(todict=True)

            for axis1 in w.get_data_axes():
                identity = t["axis_to_id"].get(axis1, None)
                if identity is None:
                    raise ValueError(
                        "Weights field has unmatched, size > 1 "
                        f"{w.constructs.domain_axis_identity(axis1)!r} axis"
                    )

                axis0 = s["id_to_axis"].get(identity, None)
                if axis0 is None:
                    raise ValueError(
                        f"Weights field has unmatched, size > 1 {identity!r} "
                        "axis"
                    )

                w_axis_size = w_domain_axes[axis1].get_size()
                f_axis_size = domain_axes[axis0].get_size()

                if w_axis_size != f_axis_size:
                    raise ValueError(
                        f"Weights field has incorrectly sized {identity!r} "
                        f"axis ({w_axis_size} != {f_axis_size})"
                    )

                axis1_to_axis0[axis1] = axis0

                # Check that the defining coordinate data arrays are
                # compatible
                key0 = s["axis_to_coord"][axis0]
                key1 = t["axis_to_coord"][axis1]

                if not f._equivalent_construct_data(
                    w, key0=key0, key1=key1, s=s, t=t
                ):
                    raise ValueError(
                        f"Weights field has incompatible {identity!r} "
                        "coordinates"
                    )

                # Still here? Then the defining coordinates have
                # equivalent data arrays

                # If the defining coordinates are attached to
                # coordinate references then check that those
                # coordinate references are equivalent
                refs0 = [
                    key
                    for key, ref in coordinate_references.items()
                    if key0 in ref.coordinates()
                ]
                refs1 = [
                    key
                    for key, ref in w_coordinate_references.items()
                    if key1 in ref.coordinates()
                ]

                nrefs = len(refs0)
                if nrefs > 1 or nrefs != len(refs1):
                    # The defining coordinates are associated with
                    # different numbers of coordinate references
                    equivalent_refs = False
                elif not nrefs:
                    # Neither defining coordinate is associated with a
                    # coordinate reference
                    equivalent_refs = True
                else:
                    # Each defining coordinate is associated with
                    # exactly one coordinate reference
                    equivalent_refs = f._equivalent_coordinate_references(
                        w, key0=refs0[0], key1=refs1[0], s=s, t=t
                    )

                if not equivalent_refs:
                    raise ValueError(
                        "Input weights field has an incompatible "
                        "coordinate reference"
                    )

            axes0 = tuple(
                [axis1_to_axis0[axis1] for axis1 in w.get_data_axes()]
            )

            for axis in axes0:
                if axis in weights_axes:
                    raise ValueError(
                        "Multiple weights specified for "
                        f"{f.constructs.domain_axis_identity(axis)!r} "
                        "axis"
                    )

            weights[tuple(axes0)] = w.data
            weights_axes.update(axes0)

        return True

    @classmethod
    def field_scalar(cls, f):
        """Return a scalar field of weights with long name ``'weight'``.

        .. versionadded:: 3.16.0

        :Parameter:

            f: `Field`
                The field for which the weights are being created.

        :Returns:

            `Field`
                The scalar weights field, with a single array value of
                ``1``.

        **Examples**

        >>> s = w.field_scalar(f)
        >> print(s)
        Field: long_name=weight
        -----------------------
        Data            : long_name=weight 1
        >>> print(s.array)
        1.0

        """
        data = f._Data(1.0, "1")

        f = type(f)()
        f.set_data(data, copy=False)
        f.long_name = "weight"
        f.comment = f"Weights for {f!r}"
        return f

    @classmethod
    def polygon_area(
        cls,
        f,
        domain_axis,
        weights,
        weights_axes,
        auto=False,
        measure=False,
        radius=None,
        great_circle=False,
        return_areas=False,
        methods=False,
    ):
        """Creates area weights for polygon geometry cells.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            domain_axis: `str` or `None`
                If set to a domain axis identifier
                (e.g. ``'domainaxis1'``) then only accept cells that
                recognise the given axis. If `None` then the cells may
                span any axis.

            {{weights weights: `dict`}}

            {{weights weights_axes: `set`}}

            {{weights auto: `bool`, optional}}

            {{weights measure: `bool`, optional}}

            {{radius: optional}}

            great_circle: `bool`, optional
                If True then assume that the edges of spherical
                polygons are great circles.

            {{weights methods: `bool`, optional}}

        :Returns:

            `bool` or `Data`
                `True` if weights were created, otherwise `False`. If
                *return_areas* is True and weights were created, then
                the weights are returned.

        """
        from .units import Units

        axis, aux_X, aux_Y, aux_Z, ugrid = cls._geometry_ugrid_cells(
            f, domain_axis, "polygon", auto=auto
        )

        if axis is None:
            if auto:
                return False

            if domain_axis is None:
                raise ValueError("No polygon cells")

            raise ValueError(
                "No polygon cells for "
                f"{f.constructs.domain_axis_identity(domain_axis)!r} axis"
            )

        if axis in weights_axes:
            if auto:
                return False

            raise ValueError(
                "Multiple weights specifications for "
                f"{f.constructs.domain_axis_identity(axis)!r} axis"
            )

        x = aux_X.bounds.data
        y = aux_Y.bounds.data

        radians = Units("radians")
        metres = Units("metres")

        if x.Units.equivalent(radians) and y.Units.equivalent(radians):
            if not great_circle:
                raise ValueError(
                    "Must set great_circle=True to allow the derivation of "
                    "area weights from spherical polygons composed from "
                    "great circle segments."
                )

            if methods:
                weights[(axis,)] = "area spherical polygon"
                return True

            spherical = True
            x.Units = radians
        elif x.Units.equivalent(metres) and y.Units.equivalent(metres):
            if methods:
                weights[(axis,)] = "area plane polygon"
                return True

            spherical = False
        else:
            return False

        y.Units = x.Units
        x = x.persist()
        y = y.persist()

        # Find the number of nodes per polygon
        n_nodes = x.count(axis=-1, keepdims=False).array
        if (y.count(axis=-1, keepdims=False) != n_nodes).any():
            raise ValueError(
                "Can't create area weights for "
                f"{f.constructs.domain_axis_identity(axis)!r} axis: "
                f"{aux_X!r} and {aux_Y!r} have inconsistent bounds "
                "specifications"
            )

        if ugrid:
            areas = cls._polygon_area_ugrid(f, x, y, n_nodes, spherical)
        else:
            areas = cls._polygon_area_geometry(
                f, x, y, aux_X, aux_Y, n_nodes, spherical
            )

        del x, y, n_nodes

        if not measure:
            areas.override_units(Units("1"), inplace=True)
        elif spherical:
            areas = cls._spherical_area_measure(f, areas, aux_Z, radius)

        if return_areas:
            return areas

        weights[(axis,)] = areas
        weights_axes.add(axis)
        return True

    @classmethod
    def _polygon_area_geometry(cls, f, x, y, aux_X, aux_Y, n_nodes, spherical):
        """Creates area weights for polygon geometry cells.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            x: `Data`
                The X coordinates of the polygon nodes, with units of
                radians or equivalent to metres.

            y: `Data`
                The Y coordinates of the polygon nodes, with the same
                units as *x*.

            aux_X: `AuxiliaryCoordinate`
                The X coordinate construct of *f*.

            aux_Y: `AuxiliaryCoordinate`
                The Y coordinate construct of *f*.

            n_nodes: `numpy.ndarray`
                The number of nodes per polygon, equal to the number
                of non-missing values in the trailing dimension of
                *x*.

            spherical: `bool`
                True for spherical lines.

        :Returns:

            `Data`
                The area of the geometry polygon cells, in units of
                square metres.

        """
        import numpy as np

        x_units = x.Units
        y_units = y.Units

        # Check for interior rings
        interior_ring_X = aux_X.get_interior_ring(None)
        interior_ring_Y = aux_Y.get_interior_ring(None)
        if interior_ring_X is None and interior_ring_Y is None:
            interior_rings = None
        elif interior_ring_X is None:
            raise ValueError(
                "Can't find weights: X coordinates have missing "
                "interior ring variable"
            )
        elif interior_ring_Y is None:
            raise ValueError(
                "Can't find weights: Y coordinates have missing "
                "interior ring variable"
            )
        elif not interior_ring_X.data.equals(interior_ring_Y.data):
            raise ValueError(
                "Can't find weights: Interior ring variables for X and Y "
                "coordinates have different data values"
            )
        else:
            interior_rings = interior_ring_X.data
            if interior_rings.shape != aux_X.bounds.shape[:-1]:
                raise ValueError(
                    "Can't find weights: Interior ring variables have "
                    f"incorrect shape. Got {interior_rings.shape}, "
                    f"expected {aux_X.bounds.shape[:-1]}"
                )

        rows = np.arange(x.shape[0])
        n_nodes_m1 = n_nodes - 1

        duplicate = x[0, 0, 0].isclose(x[0, 0, n_nodes_m1[0, 0]]) & y[
            0, 0, 0
        ].isclose(y[0, 0, n_nodes_m1[0, 0]])
        duplicate = duplicate.array

        y = y.array

        # Pad out the boundary vertex array for each cell with first
        # and last bounds neighbours
        empty_col_x = np.ma.masked_all(x.shape[:-1] + (1,), dtype=x.dtype)
        empty_col_y = np.ma.masked_all(y.shape[:-1] + (1,), dtype=y.dtype)

        if not duplicate.any():
            # The first and last boundary vertices are different in
            # all polygons, i.e. No. nodes = No. edges.
            #
            # Insert two new Y columns that duplicate the first and
            # last nodes.
            #
            # E.g. for triangle cells:
            # [[[4, 5, 6]]]     -> [[[6, 4, 5, 6, 4]]]
            # [[[4, 5, 6, --]]] -> [[[6, 4, 5, 6, 4, --]]]
            n_nodes_p1 = n_nodes + 1
            y = np.ma.concatenate((empty_col_y, y, empty_col_y), axis=-1)
            for i in range(x.shape[1]):
                y[:, i, 0] = y[rows, i, n_nodes[:, i]]
                y[rows, i, n_nodes_p1[:, i]] = y[rows, i, 1]

            if spherical:
                # Spherical polygons defined by great circles: Also
                # insert the columns into X.
                x = x.array
                x = np.ma.concatenate((empty_col_x, x, empty_col_x), axis=-1)
                for i in range(x.shape[1]):
                    x[:, i, 0] = x[rows, i, n_nodes[:, i]]
                    x[rows, i, n_nodes_p1[:, i]] = x[rows, i, 1]

            # The number of edges in each polygon
            N = n_nodes

            del n_nodes_p1
        elif duplicate.all():
            # The first and last boundary vertices coincide in all
            # cells, i.e. No. nodes = No. edges + 1.
            #
            # E.g. for triangle cells:
            # [[[4, 5, 6, 4]]]     -> [[[6, 4, 5, 6, 4]]]
            # [[[4, 5, 6, 4, --]]] -> [[[6, 4, 5, 6, 4, --]]]
            if not spherical:
                raise ValueError(
                    "Can't (yet) calculate weights for plane geometry "
                    "polygons with identical first and last nodes"
                )

            y = np.ma.concatenate((empty_col_y, y), axis=-1)
            for i in range(x.shape[1]):
                y[:, i, 0] = y[rows, i, n_nodes_m1[:, i]]

            x = x.array
            x = np.ma.concatenate((empty_col_x, x), axis=-1)
            for i in range(x.shape[1]):
                x[:, i, 0] = x[rows, i, n_nodes_m1[:, i]]

            # The number of edges in each polygon
            N = n_nodes_m1
        else:
            raise ValueError(
                "Can't calculate spherical geometry polygon cell areas "
                "when the first and last boundary vertices coincide in "
                "some cells but not all"
            )

        del duplicate, rows, n_nodes_m1

        y = f._Data(y, units=y_units)
        if spherical:
            x = f._Data(x, units=x_units)

        if spherical:
            # Spherical polygons defined by great circles
            all_areas = cls._spherical_polygon_areas(
                f, x, y, N, interior_rings
            )
        else:
            # Plane polygons defined by straight lines.
            all_areas = cls._plane_polygon_areas(x, y)

        # Sum the areas of each geometry polygon part to get the total
        # area of each cell
        areas = all_areas.sum(-1, squeeze=True)
        return areas

    @classmethod
    def _polygon_area_ugrid(cls, f, x, y, n_nodes, spherical):
        """Creates area weights for UGRID face cells.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            x: `Data`
                The X coordinates of the polygon nodes, with units of
                radians or equivalent to metres.

            y: `Data`
                The Y coordinates of the polygon nodes, with the same
                units as *x*.

            n_nodes: `numpy.ndarray`
                The number of nodes per polygon, equal to the number
                of non-missing values in the trailing dimension of
                *x*.

            spherical: `bool`
                True for spherical polygons.

        :Returns:

            `Data`
                The area of the UGRID face cells, in units of square
                metres.

        """
        import numpy as np

        y_units = y.Units

        n_nodes0 = n_nodes.item(0)
        if (n_nodes == n_nodes0).all():
            # All cells have the same number of nodes, so we can use
            # an integer and a slice in place of two integer arrays,
            # which is much faster.
            n_nodes = n_nodes0
            rows = slice(None)
        else:
            rows = np.arange(x.shape[0])

        n_nodes_p1 = n_nodes + 1

        # The first and last boundary vertices are different in
        # all polygons, i.e. No. nodes = No. edges.
        #
        # Insert two new Y columns that duplicate the first and last
        # nodes.
        #
        # E.g. for triangle cells:
        # [[4, 5, 6]]     -> [[6, 4, 5, 6, 4]]
        # [[4, 5, 6, --]] -> [[6, 4, 5, 6, 4, --]]
        y = y.array
        empty_col_y = np.ma.masked_all((y.shape[0], 1), dtype=y.dtype)
        y = np.ma.concatenate((empty_col_y, y, empty_col_y), axis=1)
        y[:, 0] = y[rows, n_nodes]
        y[rows, n_nodes_p1] = y[rows, 1]
        y = f._Data(y, units=y_units)

        if spherical:
            # Spherical polygons defined by great circles: Also insert
            # the columns into X.
            x_units = x.Units
            x = x.array
            empty_col_x = np.ma.masked_all((x.shape[0], 1), dtype=x.dtype)
            x = np.ma.concatenate((empty_col_x, x, empty_col_x), axis=1)
            x[:, 0] = x[rows, n_nodes]
            x[rows, n_nodes_p1] = x[rows, 1]
            x = f._Data(x, units=x_units)

            areas = cls._spherical_polygon_areas(f, x, y, n_nodes)
        else:
            # Plane polygons defined by straight lines
            areas = cls._plane_polygon_areas(x, y)

        return areas

    @classmethod
    def line_length(
        cls,
        f,
        domain_axis,
        weights,
        weights_axes,
        auto=False,
        measure=False,
        radius=None,
        great_circle=False,
        methods=False,
    ):
        """Creates line-length weights for line geometries.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            domain_axis: `str` or `None`
                If set to a domain axis identifier
                (e.g. ``'domainaxis1'``) then only recognise cells
                that span the given axis. If `None` then the cells may
                span any axis.

            {{weights weights: `dict`}}

            {{weights weights_axes: `set`}}

            {{weights auto: `bool`, optional}}

            {{weights measure: `bool`, optional}}

            {{radius: optional}}

            great_circle: `bool`, optional
                If True then assume that the spherical lines are great
                circles.

            {{weights methods: `bool`, optional}}

        :Returns:

            `bool`
                `True` if weights were created, otherwise `False`.

        """
        from .units import Units

        axis, aux_X, aux_Y, aux_Z, ugrid = cls._geometry_ugrid_cells(
            f, domain_axis, "line", auto=auto
        )

        if axis is None:
            if auto:
                return False

            if domain_axis is None:
                raise ValueError("No line cells")

            raise ValueError(
                "No line cells for "
                f"{f.constructs.domain_axis_identity(domain_axis)!r} axis"
            )

        if axis in weights_axes:
            if auto:
                return False

            raise ValueError(
                "Multiple weights specifications for "
                f"{f.constructs.domain_axis_identity(axis)!r} axis"
            )

        radians = Units("radians")
        metres = Units("metres")

        x = aux_X.bounds.data
        y = aux_Y.bounds.data

        if x.Units.equivalent(radians) and y.Units.equivalent(radians):
            if not great_circle:
                raise ValueError(
                    "Must set great_circle=True to allow the derivation of "
                    "area weights from spherical polygons composed from "
                    "great circle segments."
                )

            if methods:
                weights[(axis,)] = "linear spherical line"
                return True

            spherical = True
            x.Units = radians
        elif x.Units.equivalent(metres) and y.Units.equivalent(metres):
            if methods:
                weights[(axis,)] = "linear plane line"
                return True

            spherical = False
        else:
            return False

        y.Units = x.Units

        if ugrid:
            lengths = cls._line_length_ugrid(f, x, y, spherical)
        else:
            lengths = cls._line_length_geometry(f, x, y, spherical)

        if not measure:
            lengths.override_units(Units("1"), inplace=True)
        elif spherical:
            r = f.radius(default=radius)
            r = r.override_units(Units("1"))
            lengths = lengths * r

        weights[(axis,)] = lengths
        weights_axes.add(axis)
        return True

    @classmethod
    def _line_length_geometry(cls, f, x, y, spherical):
        """Creates line-length weights for line geometries.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            x: `Data`
                The X coordinates of the line nodes.

            y: `Data`
                The Y coordinates of the line nodes.

            spherical: `bool`
                True for spherical lines.

        :Returns:

            `Data`
                The length of the geometry line cells, in units of
                metres.

        """
        from .units import Units

        if spherical:
            all_lengths = cls._central_angles(f, x, y)
            all_lengths.override_units(Units("m"), inplace=True)
        else:
            delta_x = x.diff(axis=-1)
            delta_y = y.diff(axis=-1)
            all_lengths = (delta_x**2 + delta_y**2) ** 0.5

        # Sum the lengths of each part to get the total length of each
        # cell
        lengths = all_lengths.sum(axes=(-2, -1), squeeze=True)
        return lengths

    @classmethod
    def _line_length_ugrid(cls, f, x, y, spherical):
        """Creates line-length weights for UGRID edges.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            x: `Data`
                The X coordinates of the line nodes.

            y: `Data`
                The Y coordinates of the line nodes.

            spherical: `bool`
                True for spherical lines.

        :Returns:

            `Data`
                The length of the UGRID edge cells, in units of
                metres.

        """
        from .units import Units

        if spherical:
            lengths = cls._central_angles(f, x, y)
            lengths.override_units(Units("m"), inplace=True)
        else:
            delta_x = x.diff(axis=-1)
            delta_y = y.diff(axis=-1)
            lengths = (delta_x**2 + delta_y**2) ** 0.5

        lengths.squeeze(axes=-1, inplace=True)
        return lengths

    @classmethod
    def geometry_volume(
        cls,
        f,
        domain_axis,
        weights,
        weights_axes,
        auto=False,
        measure=False,
        radius=None,
        great_circle=False,
        methods=False,
    ):
        """Creates volume weights for polygon geometry cells.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            domain_axis: `str` or `None`
                If set to a domain axis identifier
                (e.g. ``'domainaxis1'``) then only recognise cells
                that span the given axis. If `None` then the cells may
                span any axis.

            {{weights weights: `dict`}}

            {{weights weights_axes: `set`}}

            {{weights auto: `bool`, optional}}

            {{weights measure: `bool`, optional}}

            {{radius: optional}}

            great_circle: `bool`, optional
                If True then assume that the edges of spherical
                polygons are great circles.

            {{weights methods: `bool`, optional}}

        :Returns:

            `bool`
                `True` if weights were created, otherwise `False`.

        """
        from .units import Units

        axis, aux_X, aux_Y, aux_Z, ugrid = cls._geometry_ugrid_cells(
            f, domain_axis, "polygon", auto=auto
        )

        if axis is None and auto:
            return False

        if axis in weights_axes:
            if auto:
                return False

            raise ValueError(
                "Multiple weights specifications for "
                f"{f.constructs.domain_axis_identity(axis)!r} axis"
            )

        x = aux_X.bounds.data
        y = aux_Y.bounds.data
        z = aux_Z.bounds.data

        radians = Units("radians")
        metres = Units("metres")

        if not z.Units.equivalent(metres):
            if auto:
                return False

            raise ValueError(
                "Z coordinate bounds must have units equivalent to "
                f"metres for volume calculations. Got {z.Units!r}."
            )

        if not methods:
            # Initialise cell volumes as the cell areas
            areas = cls.polygon_area(
                f,
                weights,
                weights_axes,
                auto=auto,
                measure=measure,
                radius=radius,
                great_circle=great_circle,
                methods=False,
                return_areas=True,
            )
            if measure:
                delta_z = abs(z[..., 1] - z[..., 0])
                delta_z.squeeze(axis=-1, inplace=True)

        if x.Units.equivalent(metres) and y.Units.equivalent(metres):
            # --------------------------------------------------------
            # Plane polygons defined by straight lines.
            # --------------------------------------------------------
            if methods:
                weights[(axis,)] = "volume plane polygon geometry"
                return True

            if measure:
                volumes = areas * delta_z
            else:
                volumes = areas

        elif x.Units.equivalent(radians) and y.Units.equivalent(radians):
            # --------------------------------------------------------
            # Spherical polygons defined by great circles
            # --------------------------------------------------------
            if not great_circle:
                raise ValueError(
                    "Must set great_circle=True to allow the derivation "
                    "of volume weights from spherical polygons composed "
                    "from great circle segments."
                )

            if methods:
                weights[(axis,)] = "volume spherical polygon geometry"
                return True

            if measure:
                r = f.radius(default=radius)

                # actual_volume =
                #    [actual_area/(4*pi*r**2)]
                #    * (4/3)*pi*[(r+delta_z)**3 - r**3)]
                volumes = areas * (
                    delta_z**3 / (3 * r**2) + delta_z**2 / r + delta_z
                )
            else:
                volumes = areas
        else:
            raise ValueError(
                "X and Y coordinate bounds must both have units "
                "equivalent to either metres (for plane polygon) or "
                "radians (for spherical polygon) volume calculations. Got "
                f"{x.Units!r} and {y.Units!r}."
            )

        weights[(axis,)] = volumes
        weights_axes.add(axis)
        return True

    @classmethod
    def _interior_angles(cls, f, lon, lat, interior_rings=None):
        """Find the interior angles at spherical polygon vertices.

        The interior angle at a vertex is that formed by two adjacent
        sides. Each interior angle is therefore a function of three
        vertices - the vertex at which the interior angle is required and
        the two adjacent vertices either side of it.

        **Method**

        Bevis, M., Cambareri, G. Computing the area of a spherical polygon
          of arbitrary shape. Math Geol 19, 335–346 (1987).

        http://bemlar.ism.ac.jp/zhuang/Refs/Refs/bevis1987mathgeol.pdf

        Abstract: An algorithm for determining the area of a spherical
          polygon of arbitrary shape is presented. The kernel of the
          problem is to compute the interior angle at each vertex of the
          spherical polygon; a well-known relationship between the area of
          a spherical polygon and the sum of its interior angles then may
          be exploited. The algorithm has been implemented as a FORTRAN
          subroutine and a listing is provided.

        .. versionadded:: 3.16.0

        .. seealso:: `_central_angles`

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            lon: `Data`
                Longitudes. Must have units of radians, which is not
                checked.

            lat: `Data`
                Latitudes. Must have units of radians, which is not
                checked.

            interior_ring: `Data`, optional
                The interior ring indicators for parts of polygon
                geometry cells. If set must have shape
                ``lon.shape[:-1]``.

        :Returns:

            `Data`
                The interior angles of each spherical polygon, in
                units of radians.

        """
        import numpy as np

        from .query import lt

        # P denotes a vertex at which the interior angle is required, A
        # denotes the adjacent point clockwise from P, and B denotes the
        # adjacent point anticlockwise from P.
        lon_A = lon[..., :-2]
        lon_P = lon[..., 1:-1]
        lon_B = lon[..., 2:]

        lon_A_minus_lon_P = lon_A - lon_P
        lon_B_minus_lon_P = lon_B - lon_P

        cos_lat = lat.cos()
        cos_lat_A = cos_lat[..., :-2]
        cos_lat_P = cos_lat[..., 1:-1]
        cos_lat_B = cos_lat[..., 2:]

        sin_lat = lat.sin()
        sin_lat_A = sin_lat[..., :-2]
        sin_lat_P = sin_lat[..., 1:-1]
        sin_lat_B = sin_lat[..., 2:]

        y = lon_A_minus_lon_P.sin() * cos_lat_A
        x = (
            sin_lat_A * cos_lat_P
            - cos_lat_A * sin_lat_P * lon_A_minus_lon_P.cos()
        )
        lat_A_primed = f._Data.arctan2(y, x)

        y = lon_B_minus_lon_P.sin() * cos_lat_B
        x = (
            sin_lat_B * cos_lat_P
            - cos_lat_B * sin_lat_P * lon_B_minus_lon_P.cos()
        )
        lat_B_primed = f._Data.arctan2(y, x)

        # The CF vertices here are, in general, given in anticlockwise
        # order, so we do "alpha_P = lat_B_primed - lat_A_primed",
        # rather than the "alpha_P = lat_A_primed - lat_B_primed"
        # given in Bevis and Cambareri, which assumes clockwise order.
        alpha_P = lat_B_primed - lat_A_primed

        if interior_rings is not None:
            # However, interior rings *are* given in clockwise order in
            # CF, so we need to negate alpha_P in these cases.
            alpha_P.where(interior_rings, -alpha_P, inplace=True)

        # Add 2*pi to negative values
        alpha_P = alpha_P.where(lt(0), alpha_P + 2 * np.pi, alpha_P)
        return alpha_P

    @classmethod
    def _central_angles(cls, f, lon, lat):
        r"""Find the central angles for spherical great circle line segments.

        The central angle of two points on the sphere is the angle
        subtended from the centre of the sphere by the two points on its
        surface. It is calculated with a special case of the Vincenty
        formula that is accurate for all spherical distances:

        **Method**

        \Delta \sigma =\arctan {\frac {\sqrt {\left(\cos \phi
        _{2}\sin(\Delta \lambda )\right)^{2}+\left(\cos \phi _{1}\sin
        \phi _{2}-\sin \phi _{1}\cos \phi _{2}\cos(\Delta \lambda
        )\right)^{2}}}{\sin \phi _{1}\sin \phi _{2}+\cos \phi _{1}\cos
        \phi _{2}\cos(\Delta \lambda )}}

        The quadrant for \Delta \sigma should be governed by the signs of
        the numerator and denominator of the right hand side using the
        atan2 function.

        (https://en.wikipedia.org/wiki/Great-circle_distance)

        .. versionadded:: 3.16.0

        .. seealso:: `_interior_angles`

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            lon: `Data`
                Longitudes with units of radians.

            lat: `Data`
                Latitudes with units of radians.

        :Returns:

            `Data`
                The central angles in units of radians.

        """
        # A and B denote adjacent vertices that define each line segment
        delta_lon = lon.diff(axis=-1)

        cos_lat = lat.cos()
        sin_lat = lat.sin()

        cos_lat_A = cos_lat[..., :-1]
        cos_lat_B = cos_lat[..., 1:]

        sin_lat_A = sin_lat[..., :-1]
        sin_lat_B = sin_lat[..., 1:]

        cos_delta_lon = delta_lon.cos()
        sin_delta_lon = delta_lon.sin()

        y = (
            (cos_lat_B * sin_delta_lon) ** 2
            + (cos_lat_A * sin_lat_B - sin_lat_A * cos_lat_B * cos_delta_lon)
            ** 2
        ) ** 0.5
        x = sin_lat_A * sin_lat_B + cos_lat_A * cos_lat_B * cos_delta_lon

        delta_sigma = f._Data.arctan2(y, x)
        return delta_sigma

    @classmethod
    def linear(
        cls,
        f,
        axis,
        weights,
        weights_axes,
        auto=False,
        measure=False,
        methods=False,
    ):
        """1-d linear weights from dimension coordinate constructs.

        .. versionadded:: 3.16.0

        :Parameters:

            axis: `str`
                The identity of the domain axis over which to find the
                weights.

            {{weights weights: `dict`}}

            {{weights weights_axes: `set`}}

            {{weights auto: `bool`, optional}}

            {{weights measure: `bool`, optional}}

            {{weights methods: `bool`, optional}}

        :Returns:

            `bool`
                `True` if weights were created, otherwise `False`.

        """
        da_key = f.domain_axis(axis, key=True, default=None)
        if da_key is None:
            if auto:
                return False

            raise ValueError(
                "Can't create weights: Can't find domain axis matching "
                f"{axis!r}"
            )

        dim = f.dimension_coordinate(filter_by_axis=(da_key,), default=None)
        if dim is None:
            if auto:
                return False

            raise ValueError(
                f"Can't create linear weights for {axis!r} axis: Can't find "
                "dimension coordinate construct."
            )

        if not measure and dim.size == 1:
            return False

        if da_key in weights_axes:
            if auto:
                return False

            raise ValueError(
                f"Can't create linear weights for {axis!r} axis: Multiple "
                "axis specifications"
            )

        if not dim.has_bounds():
            # Dimension coordinate has no bounds
            if auto:
                return False

            raise ValueError(
                f"Can't create linear weights for {axis!r} axis: No bounds"
            )
        else:
            # Bounds exist
            if methods:
                weights[(da_key,)] = (
                    f"linear {f.constructs.domain_axis_identity(da_key)}"
                )
            else:
                weights[(da_key,)] = dim.cellsize

        weights_axes.add(da_key)
        return True

    @classmethod
    def cell_measure(
        cls, f, measure, weights, weights_axes, methods=False, auto=False
    ):
        """Cell measure weights.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            {{weights weights: `dict`}}

            {{weights weights_axes: `set`}}

            {{weights methods: `bool`, optional}}

            {{weights auto: `bool`, optional}}

        :Returns:

            `bool`
                `True` if weights were created, otherwise `False`.

        """
        m = f.cell_measures(filter_by_measure=(measure,), todict=True)
        len_m = len(m)

        if not len_m:
            if measure == "area":
                return False

            if auto:
                return False

            raise ValueError(
                f"Can't find weights: No {measure!r} cell measure"
            )

        elif len_m > 1:
            if auto:
                return False

            raise ValueError(
                f"Can't find weights: Multiple {measure!r} cell measures"
            )

        key, clm = m.popitem()

        if not clm.has_data():
            if auto:
                return False

            raise ValueError(
                f"Can't find weights: Cell measure {m!r} has no data, "
                "possibly because it is external. "
                "Consider setting cell_measures=False"
            )

        clm_axes0 = f.get_data_axes(key)

        clm_axes = tuple(
            [axis for axis, n in zip(clm_axes0, clm.data.shape) if n > 1]
        )

        for axis in clm_axes:
            if axis in weights_axes:
                if auto:
                    return False

                raise ValueError(
                    "Multiple weights specifications for "
                    f"{f.constructs.domain_axis_identity(axis)!r} axis"
                )

        clm = clm.get_data(_fill_value=False).copy()
        if clm_axes != clm_axes0:
            iaxes = [clm_axes0.index(axis) for axis in clm_axes]
            clm.squeeze(iaxes, inplace=True)

        if methods:
            weights[tuple(clm_axes)] = measure + " cell measure"
        else:
            weights[tuple(clm_axes)] = clm

        weights_axes.update(clm_axes)
        return True

    @classmethod
    def scale(cls, w, scale):
        """Scale the weights so that they are <= scale.

        .. versionadded:: 3.16.0

        :Parameters:

            w: `Data`
                The weights to be scaled.

            scale: number or `None`
                The maximum value of the scaled weights. If `None`
                then no scaling is applied.

        :Returns:

            `Data`
                The scaled weights.

        """
        if scale is None:
            return w

        if scale < 0:
            raise ValueError(
                "Can't set 'scale' parameter to a negatve number. Got "
                f"{scale!r}"
            )

        w = w / w.max()
        if scale != 1:
            w = w * scale

        return w

    @classmethod
    def _geometry_ugrid_cells(cls, f, domain_axis, cell_type, auto=False):
        """Checks whether weights for geometry or UGRID cells can be created.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            domain_axis: `str` or `None`
                If set to a domain axis identifier
                (e.g. ``'domainaxis1'``) then only recognise cells
                that span the given axis. If `None` then the cells may
                span any axis.

            cell_type: `str`
                Either ``'polygon'`` or ``'line'``.

            {{weights auto: `bool`, optional}}

        :Returns:

           5-`tuple`
                If no appropriate geometry/UGRID cells were found then
                a `tuple` of all `None` is returned. Otherwise the
                `tuple` comprises:

                * The domain axis identifier of the cells, or `None`.
                * The X coordinate construct of *f*, or `None`.
                * The Y coordinate construct of *f*, or `None`.
                * The Z coordinate construct of *f*, or `None`.
                * `True` if the cells are a UGRID mesh, `False` if
                  they are geometry cells, or `None`.

        """
        n_return = 5
        aux_X = None
        aux_Y = None
        aux_Z = None
        x_axis = None
        y_axis = None
        z_axis = None
        ugrid = None

        if cell_type == "polygon":
            ugrid_cell_type = "face"
        else:
            ugrid_cell_type = "edge"

        auxiliary_coordinates_1d = f.auxiliary_coordinates(
            filter_by_naxes=(1,), todict=True
        )

        for key, aux in auxiliary_coordinates_1d.items():
            if str(aux.ctype) not in "XYZ":
                continue

            aux_axis = f.get_data_axes(key)[0]

            ugrid = f.domain_topology(default=None, filter_by_axis=(aux_axis,))
            if ugrid is not None:
                cell = ugrid.get_cell(None)
            else:
                cell = None

            if not (
                cell == ugrid_cell_type or aux.get_geometry(None) == cell_type
            ):
                continue

            # Still here? Then this X, Y, or Z auxiliary coordinate is
            # for either UGRID or geometry cells.
            if aux.X:
                aux_X = aux.copy()
                x_axis = aux_axis
                if domain_axis is not None and x_axis != domain_axis:
                    aux_X = None
                    continue
            elif aux.Y:
                aux_Y = aux.copy()
                y_axis = aux_axis
                if domain_axis is not None and y_axis != domain_axis:
                    aux_Y = None
                    continue
            elif aux.Z:
                aux_Z = aux.copy()
                z_axis = aux_axis
                if domain_axis is not None and z_axis != domain_axis:
                    aux_Z = None
                    continue

        if aux_X is None or aux_Y is None:
            if auto:
                return (None,) * n_return

            raise ValueError(
                "Can't create weights: Need both X and Y nodes to "
                f"calculate {cell_type} cell weights"
            )

        if x_axis != y_axis:
            if auto:
                return (None,) * n_return

            raise ValueError(
                "Can't create weights: X and Y cells span different domain "
                f"axes: {x_axis} != {y_axis}"
            )

        axis = x_axis

        if aux_X.get_bounds(None) is None or aux_Y.get_bounds(None) is None:
            # Not both X and Y coordinates have bounds
            if auto:
                return (None,) * n_return

            raise ValueError(
                "Can't create weights: Not both X and Y coordinates have "
                "bounds"
            )

        if aux_X.bounds.shape != aux_Y.bounds.shape:
            if auto:
                return (None,) * n_return

            raise ValueError(
                "Can't create weights: UGRID/geometry X and Y coordinate "
                "bounds must have the same shape. "
                f"Got {aux_X.bounds.shape} and {aux_Y.bounds.shape}"
            )

        if aux_Z is None:
            for key, aux in auxiliary_coordinates_1d.items():
                if aux.Z:
                    aux_Z = aux.copy()
                    z_axis = f.get_data_axes(key)[0]

        # Check Z coordinates
        if aux_Z is not None:
            if z_axis != x_axis:
                if auto:
                    return (None,) * n_return

                raise ValueError(
                    "Z coordinates span different domain axis to X and Y "
                    "geometry coordinates"
                )

        return axis, aux_X, aux_Y, aux_Z, bool(ugrid)

    @classmethod
    def _spherical_area_measure(cls, f, areas, aux_Z, radius=None):
        """Convert spherical polygon weights to cell measures.

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            areas: `Data`
                The area of the polygon cells on the unit sphere, in
                units of square metres.

            aux_Z: `AuxiliaryCoordinate`
                The Z coordinate construct of *f*.

            {{radius: optional}}

        :Returns:

            `Data`
                The area of each polygon on the surface of the defined
                sphere, in units of square metres.

        """
        # Multiply by radius squared, accounting for any Z
        # coordinates, to get the actual area
        from .units import Units

        radius = f.radius(default=radius)
        if aux_Z is None:
            # No Z coordinates
            r = radius
        else:
            z = aux_Z.get_data(None, _fill_value=False)
            if z is None:
                # No Z coordinates
                r = radius
            else:
                if not z.Units.equivalent(Units("metres")):
                    raise ValueError(
                        "Z coordinates must have units equivalent to "
                        f"metres for area calculations. Got {z.Units!r}"
                    )

                positive = aux_Z.get_property("positive", None)
                if positive is None:
                    raise ValueError(
                        "Z coordinate 'positive' property is not defined"
                    )

                if positive.lower() == "up":
                    r = radius + z
                elif positive.lower() == "down":
                    r = radius - z
                else:
                    raise ValueError(
                        "Bad value of Z coordinate 'positive' property: "
                        f"{positive!r}."
                    )

        r = r.override_units(Units("1"))
        areas = areas * r**2
        return areas

    @classmethod
    def _plane_polygon_areas(cls, x, y):
        r"""Calculate the areas of plane polygons.

        The area, A, of a plane polygon is given by the shoelace
        formula:

        A={\frac {1}{2}}\sum _{i=1}^{n}x_{i}(y_{i+1}-y_{i-1})}

        (https://en.wikipedia.org/wiki/Shoelace_formula).

        The formula gives a positive area for polygon nodes stored in
        anticlockwise order as viewed from above, and a negative area
        for polygon nodes stored in clockwise order. Note that
        interior ring polygons are stored in clockwise order.

        .. versionadded:: 3.16.0

        :Parameters:

            x: `Data`
                The X coordinates of the polygon nodes, with no
                duplication of the first and last nodes (i.e. the
                polygons are represented by ``N`` values, where
                ``N`` is the number of edges).

            y: `Data`
                The Y coordinates of the polygon nodes, with
                wrap-around duplication of the first and last nodes
                (i.e. the polygons are represented by ``N + 2``
                values, where ``N`` is the number of edges).

        :Returns:

            `Data`
                The area of each plane polygon defined by the trailing
                dimensions of *x* and *y*, in units of square metres.

        """
        areas = 0.5 * (x * (y[..., 2:] - y[..., :-2])).sum(-1, squeeze=True)
        return areas

    @classmethod
    def _spherical_polygon_areas(cls, f, x, y, N, interior_rings=None):
        r"""Calculate the areas of polygons on the unit sphere.

        The area, A, of a polygon on the unit sphere, whose sides are
        great circles, is given by (Todhunter):

        A=\left(\sum _{n=1}^{N}A_{n}\right)-(N-2)\pi

        where A_{n} is the n-th interior angle, and N is the number of
        sides (https://en.wikipedia.org/wiki/Spherical_trigonometry).

        .. versionadded:: 3.16.0

        :Parameters:

            f: `Field`
                The field for which the weights are being created.

            x: array_like
                The X coordinates of the polygon nodes, with
                wrap-around duplication of the first and last nodes
                (i.e. the polygons are represented by ``N + 2``
                values, where ``N`` is the number of edges).

            y: array_like
                The Y coordinates of the polygon nodes, with
                wrap-around duplication of the first and last nodes
                (i.e. the polygons are represented by ``N + 2``
                values, where ``N`` is the number of edges).

            N: array_like
                The number of edges in each polygon.

            interior_rings: array_like, optional
                The interior ring indicators for parts of polygon
                geometry cells. If set must have shape
                ``x.shape[:-1]``.

        :Returns:

            `Data`
                The area on the unit sphere of each polygon defined by
                the trailing dimensions of *x* and *y*, in units of
                square metres.

        """
        from numpy import pi

        from .units import Units

        interior_angles = cls._interior_angles(f, x, y, interior_rings)
        areas = interior_angles.sum(-1, squeeze=True) - (N - 2) * pi
        areas.override_units(Units("m2"), inplace=True)
        return areas
