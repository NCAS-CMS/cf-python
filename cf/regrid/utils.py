"""Worker functions for regridding."""
import logging

import numpy as np

from ..decorators import (
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
)
from ..functions import regrid_logging

try:
    import ESMF
except Exception:
    ESMF_imported = False
else:
    ESMF_imported = True


logger = logging.getLogger(__name__)

# Mapping of regrid method strings to ESMF method codes. Gets
# populated the first time `ESMF_initialise` is run.
ESMF_method_map = {}


@_inplace_enabled(default=False)
def regrid(
        f,
        dst,
        coord_system,
        method,
        src_cyclic=None,
        dst_cyclic=None,
        use_src_mask=True,
        use_dst_mask=False,
        src_axes=None,
        dst_axes=None,
        axes=None,
        ignore_degenerate=True,
        return_operator=False,
        check_regrid_operator=False,
        inplace=False,
):
    """TODO.
    
    .. versionadded:: TODODASK

    :Parameters:

        f: `Field`

        dst: `Field`, `Domain`, `dict` or `RegridOperator`

        coord_system: `str`

        method: `str`
            Specify which interpolation method to use during
            regridding.

        src_cyclic: `None` or `bool`, optional
            For spherical regridding, specifies whether or not the
            source grid longitude axis is cyclic (i.e. the first and
            last cells are adjacent). If `None` then the cyclicity
            will be inferred from the source grid coordinates,
            defaulting to `False` if it can not be determined.

        dst_cyclic: `None` or `bool`, optional
            For spherical regridding, specifies whether or not the
            destination grid longitude axis is cyclic (i.e. the first
            and last cells are adjacent). If `None` then the
            cyclicity will be inferred from the destination grid
            coordinates, defaulting to `False` if it can not be
            determined.

        use_src_mask: `bool`, optional    
            For the ``"nearest_stod"`` regridding method, if True (the
            default) then cells in the result that are nearest to a
            masked source point are masked. Otherwise destination grid
            cells are only mapped to from unmasked source cells.
          
            For all other regridding methods, *use_src_mask* must be
            True, otherwise an exception is raised.

            If *dst* is a regrid operator then *use_src_mask* is
            ignored.

        use_dst_mask: `bool`, optional
            If *dst* is a `Field` and *use_dst_mask* is False (the
            default) then the mask of data on the destination grid is
            not taken into account when performing regridding. If
            *use_dst_mask* is True then any masked cells in *dst* are
            guaranteed to be transferred to the result. If *dst* has
            more dimensions than are being regridded, then the mask of
            the destination grid is taken as the subspace defined by
            index ``0`` in all of the non-regridding dimensions.
            
            For any other type of *dst*, *use_dst_mask* is ignored.
  
        src_axes=None,

        dst_axes=None,

        axes=None,

        ignore_degenerate: `bool`, optional
            For conservative regridding methods, if True (the default)
            then degenerate cells (those for which enough vertices
            collapse to leave a cell either a line or a point) are
            skipped, not producing a result. Otherwise an error will
            be produced if degenerate cells are found, that will be
            present in the ESMF log files if `cf.regrid_logging` is
            set to True.

            For all other regridding methods, degenerate cells are
            always skipped, regardless of the value of
            *ignore_degenerate*.

            If *dst* is a regrid operator then *ignore_degenerate* is
            ignored.

        return_operator: `bool`, optional

        check_regrid_operator: `bool`, optional

            {{inplace: `bool`, optional}}

    :Returns:

        `Field`, `None` or `RegridOperator`

    """
    if return_operator:
        f = s.copy()
    else:
        f = _inplace_enabled_define_and_cleanup(f)

    create_regrid_operator = True
    if isinstance(dst, RegridOperator):
        create_regrid_operator = False

        regrid_operator = dst
        dst = regrid_operator.get_parameter("dst").copy()
        dst_axes = regrid_operator.get_parameter("dst_axes")
        dst_cyclic = regrid_operator.dst_cyclic
        method = regrid_operator.method
    else:
        check_regrid_operator = False

    check_method(method)
    check_use_src_mask(use_src_mask, method)

    if coord_system == "Cartesian":
        # Parse the axes
        if axes is None:
            raise ValueError(
                "Must set the 'axes' parameter for Cartesian regridding"
            )

        if isinstance(axes, str):
            axes = (axes,)

        n_axes = len(axes)
        if not 1 <= n_axes <= 3:
            raise ValueError(
                "Between 1 and 3 axes must be individually specified "
                "for Cartesian regridding"
            )

        if method == "patch" and n_axes != 2:
            raise ValueError(
                "The patch recovery method is only available for "
                "2-d regridding."
            )

        src_axes = axes
        if dst_axes is None:
            dst_axes = axes

    if isinstance(dst, dict):
        # Convert a dictionary containing the destination grid to a
        # Domain object
        dst, dst_axes = dict_to_domain(
            coord_system, dst, cyclic=dst_cyclic, axes=dst_axes, field=f
        )
        use_dst_mask = False
    elif create_regrid_operator and not (
        isinstance(dst, f.__class__)
        or isinstance(dst, f._Domain.__class__)
    ):
        raise TypeError(
            "'dst' parameter must be of type Field, Domain, dict "
            f"or RegridOperator. Got: {type(dst)}"
        )

    (
        dst_axis_keys,
        dst_axis_sizes,
        dst_axis_indices,
        dst_coords,
        dst_bounds,
        dst_cyclic,
    ) = get_coords(
        coord_system, f, "destination", method, dst_cyclic, axes=dst_axes
    )

    (
        src_axis_keys,
        src_axis_sizes,
        src_axis_indices,
        src_coords,
        src_bounds,
        src_cyclic,
    ) = get_coords(
        coord_system, f, "source", method, src_cyclic, axes=src_axes
    )

    # Check the consistency of the source and destination coordinate
    # units.
    comform_coordinate_units( coord_system, src_coords, src_bounds,
                              dst_coords, dst_bounds )

    if check_regrid_operator:
        # Check that the given regrid operator is compatable
        check_operator(
            coord_system,
            f,
            src_cyclic,
            src_coords,
            src_bounds,
            regrid_operator,
        )

#        # Get the axis indices and their order for the source field
#        # (this might change 'f' in place).
#        src_axis_indices = get_axis_indices(f, src_axis_keys)

    if create_regrid_operator:
        # --------------------------------------------------------
        #
        # --------------------------------------------------------
        ESMF_manager = ESMF_initialise()

        # Create a mask for the destination grid
        dst_mask = None
        grid_dst_mask = None
        if use_dst_mask:
            dst = dst.copy()
  #                dst_axis_indices = get_axis_indices(dst, dst_axis_keys)
            dst_mask = get_mask(dst, dst_axis_indices)
            if method == "nearest_stod":
                # For nearest source to desination regridding, the
                # desination mask needs to be taken into account
                # during the ESMF calculation of the regrid
                # weights. For other methods, the destination mask
                # gets applied during the compute of `Data._regrid`.
                grid_dst_mask = np.array(dst_mask.transpose())
                dst_mask = None

        # Create a mask for the source grid
        src_mask = None
        grid_src_mask = None
        if method in ("patch", "conservative_2d"):
            # For patch recovery and second-order conservative
            # regridding, the source mask needs to be taken into
            # account during the ESMF calculation of the regrid
            # weights. For other methods, the source mask gets applied
            # during the compute of `Data._regrid`.
            src_mask = get_mask(f, src_axis_indices)
            grid_src_mask = np.array(src_mask.transpose())
            if not grid_src_mask.any():
                # There are no masked source cells, so we can collapse
                # the mask that gets stored in the regrid operator.
                src_mask = np.array(False)

        # Create the destination ESMF.Grid
        dst_grid = create_ESMF_grid(
            coord_system,
            name="destination",
            method=method,
            coords=dst_coords,
            bounds=dst_bounds,
            cyclic=dst_cyclic,
            mask=grid_dst_mask,
        )
        del grid_dst_mask

        # Create the source ESMF.Grid
        src_grid = create_ESMF_grid(
            coord_system,
            name="source",
            method=method,
            coords=src_coords,
            bounds=src_bounds,
            cyclic=src_cyclic,
            mask=grid_src_mask,
        )
        del grid_src_mask

        # Create regrid weights matrix
        quarter = (
            coord_system == "Cartesian"
            and n_axes == 1
            and len(src_coords) == 2
            and not src_bounds
        )

        weights, row, col = create_ESMF_weights(
            method,
            src_grid,
            dst_grid,
            unmapped_action=unmapped_action,
            ignore_degenerate=ignore_degenerate,
            quarter=quarter,
        )

        # We've finished with ESMF, so finalise the ESMF manager. This
        # is done to free up any Persistent Execution Threads (PETs)
        # created by the ESMF Virtual Machine
        # (https://earthsystemmodeling.org/esmpy_doc/release/latest/html/api.html#resource-allocation).
        del ESMF_mananger

        # Create regrid operator
        regrid_operator = RegridOperator(
            weights,
            row,
            col,
            method=method,
            src_shape=src_shape,
            dst_shape=dst_shape,
            src_cyclic=src_cyclic,
            dst_cyclic=dst_cyclic,
            src_mask=src_mask,
            dst_mask=dst_mask,
            src_coords=src_coords,
            src_bounds=src_bounds,
            coord_sys=coord_sys,
            parameters={"dst": dst.copy(), "dst_axes": dst_axes},
        )

        if return_operator:
            return regrid_operator

    # ----------------------------------------------------------------
    # Still here? Then do the regridding
    # ----------------------------------------------------------------
    regridded_sizes = regridded_axes_sizes(
        src_axis_indices, dst_axis_sizes
    )

    data = f.data._regrid(
        operator=regrid_operator,
        regrid_axes=src_axis_indices,  # in the order expected by the op
        regridded_sizes=regridded_sizes,
    )

    # Set regridded metadata
    update_non_coordinates(
        coord_system,
        f,
        regrid_operator,
        src_axis_keys=src_axis_keys,
        dst_axis_keys=dst_axis_keys,
        dst_axis_sizes=dst_axis_sizes,
    )

    update_coordinates(
        f,
        dst,
        src_axis_keys=src_axis_keys,
        dst_axis_keys=dst_axis_keys,
        dst_axis_sizes=dst_axis_sizes,
    )

    # Insert regridded data into the new field
    f.set_data(new_data, axes=f.get_data_axes(), copy=False)

    if coord_system == "spherical":
        # Set the cyclicity of the longitude axis of the new field
        key, x = f.dimension_coordinate(
            "X", default=(None, None), item=True
        )
        if x is not None and x.Units.equivalent(Units("degrees")):
            f.cyclic(
                key,
                iscyclic=dst_cyclic,
                config={"coord": x, "period": Data(360.0, "degrees")},
            )

    return f


def dict_to_domain(coord_system, d, cyclic=None, axes=None, field=None):
    """Convert a dictionary grid definition to a `Domain`.

    See `spherical_dict_to_domain` and `Cartesian_dict_to_domain` for
    details.

    .. versionadded:: TODODASK

    """
    domain_class = field._Domain
    if coord_system == "spherical":
        return spherical_dict_to_domain(d, cyclic=cyclic, domain_class=domain_class)

    # Cartesian
    return Cartesian_dict_to_domain(d, axes=axes, domain_class=domain_class)

def get_coords(
    coord_system, f, name=None, method=None, cyclic=None, axes=None
):
    """Get axis and coordinate information for regridding.

    See `get_spherical_coords` and `get_Cartesian_coords` for details.

    .. versionadded:: TODODASK

    """
    if coord_system == "spherical":
        func = get_spherical_coords
    else:
        func = get_Cartesian_coords

    return func(f, name=name, method=method, cyclic=cyclic, axes=axes)


def get_spherical_coords(
    f, name=None, method=None, cyclic=None, axes=None
):
    """Get latitude and longitude coordinate information.

    Retrieve the latitude and longitude coordinates of a field, as
    well as some associated information. If 1-d lat/lon coordinates
    are found then these are returned. Otherwise if 2-d lat/lon
    coordinates found then these are returned.

    :Parameters:

        f: `Field`
            The source or destination field from which to get the
            information.

        name: `str`
            A name to identify the field in error messages. Either
            ``'source'`` or ``'destination'``.

        method: `str`
            The regridding method.

        axes: `dict`, optional
            A dictionary specifying the X and Y axes, with keys
            ``'X'`` and ``'Y'``.

            *Parameter example:*
              ``axes={'X': 'ncdim%x', 'Y': 'ncdim%y'}``

            *Parameter example:*
              ``axes={'X': 1, 'Y': 0}``

    :Returns:

        `list`, `list`, `dict`, `dict`, `bool`
            * The keys of the X and Y dimension coordinates.

            * The sizes of the X and Y dimension coordinates.

            * The keys of the X and Y coordinate (1-d dimension
              coordinate, or 2-d auxilliary coordinates).

            * The X and Y coordinates (1-d dimension coordinates or
              2-d auxilliary coordinates).

            * True if 2-d auxiliary coordinates are returned or if 1-d
              X and Y coordinates are returned, which are not
              lon/lat.

    """
    data_axes = f.constructs.data_axes()

    coords_1d = False
    coords_2d = False

    # Look for 1-d X/Y dimension coordinates
    lon_key, lon = f.dimension_coordinate("X", item=True, default=(None, None))
    lat_key, lat = f.dimension_coordinate("Y", item=True, default=(None, None))

    if (
        lon is not None
        and lat is not None
        and x.Units.islongitude
        and y.Units.islatitude
    ):
        # Found 1-d latitude/longitude dimension coordinates
        coords_1d = True

        x_axis = data_axes[lon_key][0]
        y_axis = data_axes[lat_key][0]
    else:
        # Look for 2-d X/Y auxiliary coordinates
        lon_key, lon = f.auxiliary_coordinate(
            "X", filter_by_naxes=(2,), item=True, default=(None, None)
        )
        lat_key, lat = f.auxiliary_coordinate(
            "Y", filter_by_naxes=(2,), item=True, default=(None, None)
        )
        if (
            lon is not None
            and lat is not None
            and lon.Units.islongitude
            and lat.Units.islatitude
        ):
            # Found 2-d latitude/longitude auxiliary coordinates
            coords_2d = True

            lon_axes = data_axes[lon_key]
            lat_axes = data_axes[lat_key]

            if not axes or ("X" not in axes or "Y" not in axes):
                raise ValueError("TODO")

            if set((axes["X"], axes["Y"])) == set((0, 1)):
                if lon_axes != lat_axes:
                    raise ValueError("TODO")

                x_axis = lon_axes[axes["X"]]
                y_axis = lat_axes[axes["Y"]]
            else:
                x_axis = f.domain_axis(axes["X"], key=True, defualt=None)
                y_axis = f.domain_axis(axes["Y"], key=True, default=None)
                if x_axis is None or y_axis is None:
                    raise ValueError("TODO")

                if set(lon_axes) != set((x_axis, y_axis)):
                    raise ValueError("TODO")

                if set(lat_axes) != set((x_axis, y_axis)):
                    raise ValueError("TODO")

    if not (coords_1d or coords_2d):
        raise ValueError(
            "Could not find 1-d nor 2-d latitude and longitude coordinates"
        )

    if x_axis == y_axis:
        raise ValueError(
            "The X and Y axes must be distinct, but they are "
            "the same for {name} field {f!r}."
        )

    # Check for size 1 latitude or longitude dimensions if source grid
    # (a size 1 dimension is only problematic for the source grid in ESMF)
    if (
        name == "source"
        and method in ("linear", "bilinear", "patch")
        and (x_size == 1 or y_size == 1)
    ):
        raise ValueError(
            f"Neither the longitude nor latitude dimensions of the {name}"
            f"field {f!r} can be of size 1 for {method!r} regridding."
        )

    coords = {0: lon, 1: lat}  # ESMF order

    # Bounds
    if regridding_is_conservative(method):
        bounds = get_bounds(coords)
        if len(bounds) < len(coords):
            raise ValueError("TODO")
    else:
        bounds = {}

    # Convert 2-d coordinate arrays to ESMF axis order = [X, Y]
    if coords_2d:
        for dim, coord_key in {0: lon_key, 1: lat_key}.items():
            coord_axes = data_axes[coord_key]
            ESMF_order = [coord_axes.index(axis) for axis in (x_axis, y_axis)]
            coords[dim] = coords[dim].transpose(ESMF_order)
            if bounds:
                bounds[dim] = bounds[dim].transpose(ESMF_order + [-1])

    # Set cyclicity of X axis
    if cyclic is None:
        cyclic = f.iscyclic(x_axis)
    else:
        cyclic = False

    # Get X/Y axis sizes
    domain_axes = f.domain_axes(todict=True)
    x_size = domain_axes[x_axis].size
    y_size = domain_axes[y_axis].size

    axis_keys = [y_axis, x_axis]
    axis_sizes = [y_size, x_size]
    axis_indices = get_axis_indices(f, axis_keys)

    return (axis_keys, axis_sizes, axis_indices, coords, bounds, cyclic)


def get_Cartesian_coords(
    f, name=None, method=None, cyclic=None, axes=None
):
    """Retrieve the specified Cartesian dimension coordinates of the
    field and their corresponding keys.

    :Parameters:

        f: `Field`
           The field from which to get the coordinates.

        name: `str`
            A name to identify the field in error messages.

        axes: sequence of `str`
            Specifiers for the dimension coordinates to be
            retrieved. See `cf.Field.domain_axes` for details.

    :Returns:

        `list`, `list`
            A list of the keys of the dimension coordinates; and a
            list of the dimension coordinates retrieved.

    """
    # Find the axis keys, sizes and indices of the regrid axes, in the
    # order that they have been given by the 'axes' parameters.
    axis_sizes = []
    for dim, axis in enumerate(axes):
        key, domain_axis = f.domain_axis(axis, item=True, default=None, None))
        if key is None:
            raise ValueError("TODO")

        axis_keys.append(key)
        axis_sizes.append(domain_axis.size)

    data_axes = f.get_data_axes()
    axis_indices = [data_axes.index(key) for key in axis_keys]

    # Reorder the axis keys, sizes and indices so that they are in the
    # same relative order as they occur in the field's data array.
    reorder = np.argsort(axis_indices)
    axis_keys = np.array(axis_keys)[reorder].tolist()
    axis_sizes = np.array(axis_sizes)[reorder].tolist()
    axis_indices = np.array(axis_indices)[reorder].tolist()

    # Find the grid coordinates for the regrid axes, in the *reverse*
    # order to 'axis_keys'.
    coords = {}
    for dim, axis_key in enumerate(axes_keys[::-1]):
        coord = f.dimension_coordinate(filter_by_axis=(axis_key,), default=None)
        if coord is None:
            raise ValueError(
                f"No unique {name} dimension coordinate "
                f"matches key {axis!r}."
            )
        
        coords[dim] = coord

#    axis_keys = []
#    axis_sizes = []
#    coords = {}
#    for dim, axis in enumerate(axes):
#        key = f.domain_axis(axis, key=True)
#        coord = f.dimension_coordinate(filter_by_axis=(key,), default=None)
#        if coord is None:
#            raise ValueError(
#                f"No unique {name} dimension coordinate "
#                f"matches key {axis!r}."
#            )
#
#        axis_keys.append(key)
#        axis_sizes.append(coord.size)
#        coords[dim] = coord

    bounds = []
    if regridding_is_conservative(method):
        bounds = get_bounds(coords)
        if len(bounds) < len(coords):
            raise ValueError("TODO")

    if len(coords) == 1:
        # Create a dummy axis because ESMF doesn't like creating
        # weights for 1-d regridding
        data = np.array([np.finfo(float).epsneg, np.finfo(float).eps])
        if bounds:
            # Size 1
            coords[1] = np.array([0.0])
            bounds[1] = data
        else:
            # Size 2
            coords[1] = data

    return axis_keys, axis_sizes, axis_indices, coords, bounds, bool(cyclic)


def spherical_dict_to_domain(d, cyclic=None, domain_class=None):
    """Convert a dictionary spherical grid definition to a `Domain`.
    
    .. versionadded:: TODODASK

    :Parameters:

        d: `dict`

        cyclic: `bool` or `None`

        field: `Field`
    
    :Returns:

        `Field`, `list`
            The new field containing the grid; and a list of the
            domain axis identifiers of the regrid axes, in the
            relative order expected by the regridding algorithm.

    """
    try:
        coords = {"lat": d["latitude"].copy(), "lon": d["longitude"].copy()}
    except KeyError:
        raise ValueError(
            "Dictionary keys 'longitude' and 'latitude' must be "
            "specified for the destination grid"
        )

    coords_1d = False

    if coords["lat"].ndim == 1:
        coords_1d = True
        axis_sizes = [coords["lat"].size, coords["lon"].size]
        if coords["lon"].ndim != 1:
            raise ValueError(
                "Longitude and latitude coordinates for the "
                "destination grid must have the same number of dimensions."
            )
    elif coords["lat"].ndim == 2:
        try:
            axis_order = tuple(d["axes"])
        except KeyError:
            raise ValueError(
                "Dictionary key 'axes' must be specified when providing "
                "2-d latitude and longitude coordinates"
            )

        axis_sizes = coords["lat"].shape
        if axis_order == ("X", "Y"):
            axis_sizes = axis_sizes[::-1]
        elif axis_order != ("Y", "X"):
            raise ValueError(
                "Dictionary key 'axes' must either be ('X', 'Y') or "
                f"('Y', 'X'). Got {axis_order!r}"
            )

        if coords["lat"].shape != coords["lon"].shape:
            raise ValueError(
                "2-d longitude and latitude coordinates for the "
                "destination grid must have the same shape."
            )
    else:
        raise ValueError(
            "Longitude and latitude coordinates for the "
            "destination grid must be 1-d or 2-d"
        )

    # Set coordinate identities
    coords["lat"].standard_name = "latitude"
    coords["lon"].standard_name = "longitude"

    # Create field
    f = type(domain_class)()

    # Set domain axes
    axis_keys = []
    for size in axis_sizes:
        key = f.set_construct(f._DomainAxis(size), copy=False)
        axis_keys.append(key)

    if coord_1d:
        # Set 1-d coordinates
        for key, axis in zip(("lat", "lon"), axis_keys):
            f.set_construct(coords[key], axes=axis, copy=False)
    else:
        # Set 2-d coordinates
        axes = axis_keys
        if axis_order == ("X", "Y"):
            axes = axes[::-1]

        for coord in coords.values():
            f.set_construct(coord, axes=axis_keys, copy=False)

    # Set X axis cyclicity
    if cyclic is not None:
        f.cyclic(axis_keys[1], iscyclic=cyclic, period=360)

    if coords_1d:
        yx_axes = {}
    else:
        yx_axes = {"Y": axis_keys[0], "X": axis_keys[1]}

    return f, yx_axes


def Cartesian_dict_to_domain(d, axes=None, domain_class=None):
    """Convert a dictionary Cartesian grid definition to a `Domain`.
    
    .. versionadded:: TODODASK

    :Parameters:

        d: `dict`

        axes: 

        field: `Field`

    :Returns:

        `Field`, `list`
            The new field containing the grid; and a list of the
            domain axis identifiers of the regrid axes, in the
            relative order expected by the regridding algorithm.

    """
    if axes is None:
        axes = d.keys()

    f = type(domain_class)()

    axis_keys = []
    for dim, axis in enumerate(axes):
        coord = d[axis]
        key = f.set_construct(f._DomainAxis(coord.size), copy=False)
        f.set_construct(coord, axes=key, copy=True)
        axis_keys.append(key)

    return f, axes_keys


# def create_spherical_ESMF_grid(
#    name=None, method=None, coords=None, cyclic=None, bounds=None, mask=None
# ):
#    """Create a spherical `ESMF.Grid`.
#
#    :Parameters:
#
#        coords: sequence of array-like
#            The coordinates if not Cartesian it is assume that the
#            first is longitude and the second is latitude.
#
#        cyclic: `bool`
#            Whether or not the longitude axis is cyclic.
#
#        bounds: `dict` or `None`, optional
#            The coordinates if not Cartesian it is assume that the
#            first is longitude and the second is latitude.
#
#        coord_ESMF_order: `dict`, optional
#            Two tuples one indicating the order of the x and y axes
#            for 2-d longitude, one for 2-d latitude. Ignored if
#            coordinates are 1-d.
#
#        mask: optional
#
#    :Returns:
#
#        `ESMF.Grid`
#            The spherical grid.
#
#    """
#    #    lon = np.asanyarray(coords[0])
#    #    lat = np.asanyarray(coords[1])
#    coords_1d = lon.ndim == 1
#
#    # Parse coordinates for the Grid, and get its shape.
#    if coords_1d:
#        coords = {dim: np.asanyarray(c) for dim, c in coords.items()}
#        shape = [c.size for dim, c in sorted(coords.items())]
#        for dim, c in coords.items():
#            coords[dim] = c.reshape(
#                [c.size if i == dim else 1 for i in range(n_axes)]
#            )
#
#    #        shape = (lon.size, lat.size)
#    #        lon = lon.reshape(lon.size, 1)
#    #        lat = lat.reshape(1, lat.size)
#    #        coords[0] = lon
#    #        coords[1]  = lat
#    else:
#        shape = lon.shape
#        if lat.shape != shape:
#            raise ValueError(
#                f"The {name} 2-d longitude and latitude coordinates "
#                "must have the same shape."
#            )
#
#    # Parse bounds for the Grid
#    if bounds:
#        lon_bounds = np.asanyarray(bounds[0])
#        lat_bounds = np.asanyarray(bounds[1])
#        lat_bounds = np.clip(lat_bounds, -90, 90)
#
#        contiguous_bounds((lon_bounds, lat_bounds), name, cyclic, period=360)
#
#        if coords_1d:
#            if cyclic:
#                x = lon_bounds[:, 0:1]
#            else:
#                n = lon_bounds.shape[0]
#                x = np.empty((n + 1, 1), dtype=lon_bounds.dtype)
#                x[:n, 0] = lon_bounds[:, 0]
#                x[n, 0] = lon_bounds[-1, 1]
#
#            m = lat_bounds.shape[0]
#            y = np.empty((1, m + 1), dtype=lat_bounds.dtype)
#            y[0, :m] = lat_bounds[:, 0]
#            y[0, m] = lat_bounds[-1, 1]
#        else:
#            n, m = x_bounds.shape[0:2]
#
#            x = np.empty((n + 1, m + 1), dtype=lon_bounds.dtype)
#            x[:n, :m] = lon_bounds[:, :, 0]
#            x[:n, m] = lon_bounds[:, -1, 1]
#            x[n, :m] = lon_bounds[-1, :, 3]
#            x[n, m] = lon_bounds[-1, -1, 2]
#
#            y = np.empty((n + 1, m + 1), dtype=lat_bounds.dtype)
#            y[:n, :m] = lat_bounds[:, :, 0]
#            y[:n, m] = lat_bounds[:, -1, 1]
#            y[n, :m] = lat_bounds[-1, :, 3]
#            y[n, m] = lat_bounds[-1, -1, 2]
#
#        bounds[0] = x
#        bounds[1] = y
#
#    #        lon_bounds = x
#    #        lat_bounds = y
#
#    # Create empty Grid
#    X, Y = 0, 1
#
#    max_index = np.array(shape, dtype="int32")
#    if bounds:
#        staggerlocs = [ESMF.StaggerLoc.CORNER, ESMF.StaggerLoc.CENTER]
#    else:
#        staggerlocs = [ESMF.StaggerLoc.CENTER]
#
#    if cyclic:
#        grid = ESMF.Grid(
#            max_index,
#            num_peri_dims=1,
#            periodic_dim=X,
#            pole_dim=Y,
#            staggerloc=staggerlocs,
#        )
#    else:
#        grid = ESMF.Grid(max_index, staggerloc=staggerlocs)
#
#    # Populate Grid centres
#    for dim, c in coords.items():
#        gridCentre[...] = c
#
#    #    c = grid.get_coords(X, staggerloc=ESMF.StaggerLoc.CENTER)
#    #    c[...] = lon
#    #    c = grid.get_coords(Y, staggerloc=ESMF.StaggerLoc.CENTER)
#    #    c[...] = lat
#
#    # Populate Grid corners
#    if bounds:
#        for dim, b in bounds.items():
#            gridCorner = grid.get_coords(dim, staggerloc=staggerloc)
#            gridCorner[...] = b
#
#    #        c = grid.get_coords(X, staggerloc=ESMF.StaggerLoc.CORNER)
#    #        c[...] = lon_bounds
#    #        c = grid.get_coords(Y, staggerloc=ESMF.StaggerLoc.CORNER)
#    #        c[...] = lat_bounds
#
#    # Add a mask
#    if mask is not None:
#        add_mask(grid, mask)
#
#    return grid


def comform_coordinate_units(
    coord_system, src_coords, src_bounds, dst_coords, dst_bounds
):
    """Make the source and destination coordinates have the same units.

    .. versionadded:: TODODASK

    .. seealso:: `get_coords`, `regrid`

    :Parameters:

        coord_system: `str`
            The coordinate system of the source and destination grids.

        src_coords: `dict`
            The source grid coordinates, as returned by `get_coords`.
    
        src_bounds: `dict`
            The source grid coordinate bounds, as returned by
            `get_coords`.
    
        dst_coords: `dict`
            The destination grid coordinates, as returned by
            `get_coords`.
   
        dst_bounds: `dict`
            The destination grid coordinate bounds, as returned by
            `get_coords`.

    :Returns:

        `None`

    """
    if coord_system == "spherical":
        # For spherical coordinate systems, the units have already
        # been checked in `get_coords`.
        return

    for src, dst in zip((src_coords, src_bounds), (dst_coords, dst_bounds)):
        for dim, s in src.items():
            d = dst[dim]

            s_units = s.getattr("Units", None)
            d_units = d.getattr("Units", None)
            if s_units is None or d_units is None:
                continue

            if s_units == d_units:
                continue

            if not s_units.equivalent(d_units):
                raise ValueError(
                    "Units of source and destination coordinates "
                    f"are not equivalent: {s!r}, {d!r}"
                )

            s = s.copy()
            s.Units = d_units
            src[dim] = s


def check_operator(
    coord_system, f, src_cyclic, src_coords, src_bounds, regrid_operator
):
    """Whether two `ESMF.Grid` instances have identical coordinates.

    :Parameters:

        grid0, grid1: `ESMF.Grid`, `ESMF.Grid`
            The `ESMF` Grid instances to be compared

    :Returns:

        `str`
            

    """
    if coord_system != regrid_operator.coord_system:
        raise ValueError(
            f"Can't regrid {f!r} with {regrid_operator!r}: "
            "Coordinate system mismatch"
        )

    if bool(src_cyclic) != regrid_operator.src_cyclic:
        raise ValueError(
            f"Can't regrid {f!r} with {regrid_operator!r}: "
            "Cyclicity of axes mismatch"
        )

    message = (
        f"Can't regrid {f!r} with {regrid_operator!r}: "
        "Source grid coordinates mismatch"
    )

    op_coords = regrid_operator.src_coords
    if src_coords.keys() != op_coords.keys():
        raise ValueError(message)

    for dim, a in src_coords.items():
        a = np.asanyarray(a)
        b = np.asanyarray(op_coords[dim])
        if not np.array_equal(a, b):
            raise ValueError(message)

    message = (
        f"Can't regrid {f!r} with {regrid_operator!r}: "
        "Source grid coordinate bounds mismatch"
    )

    op_bounds = regrid_operator.src_bounds
    if src_bounds.keys() != op_bounds.keys():
        raise ValueError(message)

    for dim, a in src_bounds.items():
        a = np.asanyarray(a)
        b = np.asanyarray(op_bounds[dim])
        if not np.array_equal(a, b):
            raise ValueError(message)


def ESMF_initialise():
    """Initialise `ESMF`.

    Initialises the `ESMF` manager, unless it has already been
    initialised.

    Whether logging is enabled or not is determined by
    `cf.regrid_logging`. If it is then logging takes place after every
    call to `ESMF`.

    Also initialises the global 'ESMF_method_map' dictionary, unless
    it has already been initialised.

    :Returns:

        `ESMF.Manager`
            A singleton instance of the `ESMF` manager.

    """
    if not ESMF_imported:
        raise RuntimeError(
            "Regridding will not work unless the ESMF library is installed"
        )

    # Update the global 'ESMF_method_map' dictionary
    if not ESMF_method_map:
        ESMF_method_map.update(
            {
                "linear": ESMF.RegridMethod.BILINEAR,  # see comment below...
                "bilinear": ESMF.RegridMethod.BILINEAR,  # (for back compat)
                "conservative": ESMF.RegridMethod.CONSERVE,
                "conservative_1st": ESMF.RegridMethod.CONSERVE,
                "conservative_2nd": ESMF.RegridMethod.CONSERVE_2ND,
                "nearest_dtos": ESMF.RegridMethod.NEAREST_DTOS,
                "nearest_stod": ESMF.RegridMethod.NEAREST_STOD,
                "patch": ESMF.RegridMethod.PATCH,
            }
        )
        # ... diverge from ESMF with respect to name for bilinear
        # method by using 'linear' because 'bi' implies 2D linear
        # interpolation, which could mislead or confuse for Cartesian
        # regridding in 1D or 3D.

    return ESMF.Manager(debug=bool(regrid_logging()))


def create_ESMF_grid(
    coord_system,
    name=None,
    method=None,
    coords=None,
    bounds=None,
    cyclic=None,
    mask=None,
):
    """Create an `ESMF` Grid.

    .. versionadded:: TODODASK

    :Parameters:

        coord_system: `str`

        name: `str`

        method: `str`

        coords: `dict`

        bounds: `dict`, optional

        cyclic: `bool` or `None`, optional

        mask: array_like, optional

    :Returns:

        `ESMF.Grid`

    """
    num_peri_dims = 0
    periodic_dim = 0
    if coord_system == "spherical":
        lon, lat = 0, 1
        coord_sys = ESMF.CoordSys.SPH_DEG
        if cyclic:
            num_peri_dims = 1
            periodic_dim = lon
    else:
        # Cartesian
        coord_sys = ESMF.CoordSys.CART

    # Parse coordinates for the Grid and get its shape
    n_axes = len(coords)
    coords_1d = coords[0].ndim == 1

    coords = {dim: np.asanyarray(c) for dim, c in coords.items()}
    if coords_1d:
        # 1-d coordinates for N-d regridding
        shape = [c.size for dim, c in sorted(coords.items())]
        for dim, c in coords.items():
            coords[dim] = c.reshape(
                [c.size if i == dim else 1 for i in range(n_axes)]
            )
    elif n_axes == 2:
        # 2-d coordinates for 2-d regridding
        shape = coords[0].shape
    else:
        raise ValueError(
            "Coordinates must be 1-d, or possibly 2-d for 2-d regridding"
        )

    # Parse bounds for the Grid
    if bounds:
        bounds = {dim: np.asanyarray(b) for dim, b in bounds.items()}

        if coord_system == "spherical":
            bounds[lat] = np.clip(bounds[lat], -90, 90)

        if not contiguous_bounds(bounds.values(), cyclic=cyclic, period=360):
            raise ValueError(
                f"The {name} coordinates must have contiguous, "
                f"non-overlapping bounds for {method} regridding."
            )

        if coords_1d:
            # Bounds for 1-d coordinates
            for dim, b in bounds.items():
                if coord_system == "spherical" and cyclic and dim == lon:
                    tmp = b[:, 0]
                else:
                    n = b.shape[0]
                    tmp = np.empty((n + 1,), dtype=b.dtype)
                    tmp[:n] = b[:, 0]
                    tmp[n] = b[-1, 1]

                tmp = tmp.reshape(
                    [tmp.size if i == dim else 1 for i in range(n_axes)]
                )
                bounds[dim] = tmp
        else:
            # Bounds for 2-d coordinates
            for dim, b in bounds.items():
                n, m = b.shape[0:2]
                tmp = np.empty((n + 1, m + 1), dtype=b.dtype)
                tmp[:n, :m] = b[:, :, 0]
                tmp[:n, m] = b[:, -1, 1]
                tmp[n, :m] = b[-1, :, 3]
                tmp[n, m] = b[-1, -1, 2]
                bounds[dim] = tmp

    # Define the stagger locations
    max_index = np.array(shape, dtype="int32")
    if bounds:
        if n_axes == 3:
            staggerlocs = [
                ESMF.StaggerLoc.CENTER_VCENTER,
                ESMF.StaggerLoc.CORNER_VFACE,
            ]
        else:
            staggerlocs = [ESMF.StaggerLoc.CORNER, ESMF.StaggerLoc.CENTER]

    else:
        if n_axes == 3:
            staggerlocs = [ESMF.StaggerLoc.CENTER_VCENTER]
        else:
            staggerlocs = [ESMF.StaggerLoc.CENTER]

    # Create an empty Grid
    grid = ESMF.Grid(
        max_index,
        coord_sys=coord_sys,
        num_peri_dims=num_peri_dims,
        periodic_dim=periodic_dim,
        staggerloc=staggerlocs,
    )

    # Populate the Grid centres
    for dim, c in coords.items():
        if n_axes == 3:
            gridCentre = grid.get_coords(
                dim, staggerloc=ESMF.StaggerLoc.CENTER_VCENTER
            )
        else:
            gridCentre = grid.get_coords(
                dim, staggerloc=ESMF.StaggerLoc.CENTER
            )

        gridCentre[...] = c

    # Populate the Grid corners
    if bounds:
        if n_axes == 3:
            staggerloc = ESMF.StaggerLoc.CORNER_VFACE
        else:
            staggerloc = ESMF.StaggerLoc.CORNER

        for dim, b in bounds.items():
            gridCorner = grid.get_coords(dim, staggerloc=staggerloc)
            gridCorner[...] = b

    # Add a Grid mask
    if mask is not None:
        add_mask(grid, mask)

    return grid


def create_ESMF_weights(
    method,
    src_grid,
    dst_grid,
    unmapped_action,
    ignore_degenerate,
    quarter=False,
):
    """Create an `ESMF` regrid operator.

    .. versionadded:: TODODASK

    :Parameters:

        ignore_degenerate: `bool`
            Whether to check for degenerate points.

        quarter: `bool`, optional
            If True then only return weights corresponding to the top
            left hand corner of the weights matrix. This is necessary
            for 1-d regridding when the weights were generated for a
            2-d grid for which one of the dimensions is a size 2 dummy
            dimension.

            .. seealso:: `get_Cartesian_coords`

    :Returns:

        3-`tuple` of `numpy.ndarray`

    """
    method = ESMF_method_map.get(method)
    if method is None:
        raise ValueError("TODO")

    if unmapped_action == "ignore" or unmapped_action is None:
        unmapped_action = ESMF.UnmappedAction.IGNORE
    elif unmapped_action == "error":
        unmapped_action = ESMF.UnmappedAction.ERROR

    # Add a mask to the source grid and create the source field
    src_field = ESMF.Field(src_grid, "src")
    dst_field = ESMF.Field(dst_grid, "dst")

    # Create the regrid operator
    r = ESMF.Regrid(
        src_field,
        dst_field,
        regrid_method=method,
        unmapped_action=unmapped_action,
        ignore_degenerate=bool(ignore_degenerate),
        src_mask_values=np.array([0], dtype="int32"),
        dst_mask_values=np.array([0], dtype="int32"),
        norm_type=ESMF.api.constants.NormType.FRACAREA,
        factors=True,
    )

    weights = r.get_weights_dict(deep_copy=True)
    row = weights["row_dst"]
    col = weights["col_src"]
    weights = weights["weights"]

    if quarter:
        # Find the indices that define the weights for the just the
        # top left corner of the weights matrix
        index = np.where(
            (row <= dst_field.data.size // 2)
            & (col <= src_field.data.size // 2)
        )
        weights = weights[index]
        row = row[index]
        col = col[index]

    destroy_Regrid(r)

    return weights, row, col


def add_mask(grid, mask):
    """Add a mask to an `ESMF.Grid`.

    .. versionadded:: TODODASK
    
    :Parameters:

        grid: `ESMF.Grid`
            An `ESMF` grid.

        mask: `np.ndarray` or `None`
            The mask to add to the grid. If `None` then no mask is
            added. If an array, then it must either be masked array,
            in which case it is replaced by its mask, or a boolean
            array. If the mask contains no True elements then no mask
            is added.

    :Returns:

        `None`

    """
    mask = np.asanyarray(mask)

    m = None
    if mask.dtype == bool and m.any():
        m = mask
    elif np.ma.is_masked(mask):
        m = mask.mask

    if m is not None:
        grid_mask = grid.add_item(ESMF.GridItem.MASK)
        grid_mask[...] = np.invert(mask).astype("int32")


def contiguous_bounds(bounds, cyclic=None, period=None):
    """TODODASK"""
    for b in bounds:
        ndim = b.ndim - 1
        if ndim == 1:
            # 1-d cells
            diff = b[1:, 0] - b[:-1, 1]
            if cyclic and period is not None:
                diff = diff % period

            if diff.any():
                return False

        elif ndim == 2:
            # 2-d cells
            nbounds = b.shape[-1]
            if nbounds != 4:
                raise ValueError(
                    f"Can't tell if {ndim}-d cells with {nbounds} vertices "
                    "are contiguous"
                )

            # Check cells (j, i) and cells (j, i+1) are contiguous
            diff = b[:, :-1, 1] - b[:, 1:, 0]
            if cyclic and period is not None:
                diff = diff % period

            if diff.any():
                return False

            diff = b[:, :-1, 2] - b[:, 1:, 3]
            if cyclic and period is not None:
                diff = diff % period

            if diff.any():
                return False

            # Check cells (j, i) and (j+1, i) are contiguous
            diff = b[:-1, :, 3] - b[1:, :, 0]
            if cyclic and period is not None:
                diff = diff % period

            if diff.any():
                return False

            diff = b[:-1, :, 2] - b[1:, :, 1]
            if cyclic and period is not None:
                diff = diff % period

            if diff.any():
                return False

    return True


def destroy_Regrid(regrid, src=True, dst=True):
    """Release the memory allocated to an `ESMF.Regrid` operator.
    It does not matter if the base regrid operator has already been
    destroyed.
    .. versionadded:: TODODASK
    :Parameters:
        regrid: `ESMF.Regrid`
            The regrid operator to be destroyed.
        src: `bool`
            By default the source fields and grid are destroyed. If
            False then they are not.
        dst: `bool`
            By default the destination fields and grid are
            destroyed. If False then they are not.
    :Returns:
    
        `None`
    """
    if src:
        regrid.srcfield.grid.destroy()
        regrid.srcfield.destroy()
        regrid.src_frac_field.destroy()

    if dst:
        regrid.dstfield.grid.destroy()
        regrid.dstfield.destroy()
        regrid.dst_frac_field.destroy()

    regrid.destroy()


def get_bounds(coords):
    """TODODASK"""
    bounds = {key: c.get_bounds(None) for key, c in coords.items()}
    bounds = {key: b for key, b in bounds if b is not None}
    return bounds


def check_method(method):
    """Check that the regrid method is valid.

    If it is not valid then an exception is raised.

    :Parameters:

        method: `str`
            The regridding method.

    :Returns:

        `None`

    """
    regridding_methods = (
        "linear",
        "bilinear",
        "conservative",
        "conservative_1st",
        "conservative_2nd",
        "nearest_dtos",
        "nearest_stod",
        "patch",
    )
    if method not in regridding_methods:
        raise ValueError(
            "Can't regrid: Must set a valid regridding method from "
            f"{regridding_methods}. Got: {method!r}"
        )
    elif method == "bilinear":
        logger.info(
            "Note the 'bilinear' method argument has been renamed to "
            "'linear' at version 3.2.0. It is still supported for now "
            "but please use 'linear' in future. "
            "'bilinear' will be removed at version 4.0.0."
        )


def check_use_src_mask(use_src_mask, method):
    """Check the setting of the use_src_mask parameter.

    An exception is raised is the setting is incorrect relative to the
    regridding method.

    :Parameters:

        use_src_mask: `bool`
            Whether to use the source mask in regridding.

        method: `str`
            The regridding method.

    :Returns:

        `None`

    """
    if not use_src_mask and not method == "nearest_stod":
        raise ValueError(
            "The 'use_src_mask' parameter can only be False when "
            f"using the {method} regridding method."
        )


def regridded_axes_sizes(src_axis_indices, dst_axis_sizes):
    """TODODASK"""
    return {axis: size for axis, size in zip(src_axis_indices, dst_axis_sizes)}


def regridding_is_conservative(method):
    return method in ("conservative", "conservative_1st", "conservative_2nd")


def get_axis_indices(f, axis_keys):
    """Get axis indices and their orders in rank of this field.

    The indices will be returned in the same order as expected by the
    regridding operator.

    For instance, for spherical regridding, if *f* has shape ``(96,
    12, 73)`` for axes (Longitude, Time, Latitude) then the axis
    indices will be ``[2, 0]``.
 
    :Parameters:

        f: `Field`
            The source or destination field. This field might get
            size-1 dimensions inserted into its data in-place.

        axis_keys: sequence
            A sequence of domain axis identifiers for the axes being
            regridded, in the order expected by `Data._regrid`.
            
    :Returns:

        `list`

    """
    # Make sure that the data array spans all of the regridding
    # axes. This might change 'f' in-place.
    data_axes = f.get_data_axes()
    for key in axis_keys:
        if key not in data_axes:
            f.insert_dimension(axis_key, position=-1, inplace=True)

    # The indices of the regridding axes, in the order expected by
    # `Data._regrid`.
    data_axes = f.get_data_axes()
    axis_indices = [data_axes.index(key) for key in axis_keys]

    return axis_indices


def get_mask(f, regrid_axes):
    """Get the mask of the grid.

    The mask dimensions will ordered as expected by the regridding
    operator.

    :Parameters:

        f: `Field`
            The  field.

        dst_axis_keys: sequence of `int`
            The positions of the regridding axes in the destination
            field.

        dst_order: sequence, optional
            The order of the destination axes.

        src_order: sequence, optional
            The order of the source axes.

    :Returns:

        `dask.array.Array`
            The mask.

    """
    index = [slice(None) if i in regrid_axes else 0 for i in range(f.ndim)]

    mask = da.ma.getmaskarray(f)
    mask = mask[tuple(index)]

    # Reorder the destination mask axes to match those of the
    # reordered source data
    mask = da.transpose(mask, np.argsort(regrid_axes))

    return mask


def update_coordinates(
    f, dst, src_axis_keys=None, dst_axis_keys=None, dst_axis_sizes=None
):
    """Update the coordinates of the regridded field.

    :Parameters:

        f: `Field`
            The regridded field. Updated in-place.

        dst: Field or `dict`
            The object containing the destination grid.

        dst_coords: sequence
            Ignored if *dst* is a `Field`. The destination
            coordinates. Assumed to be copies of those in *dst*.

        src_axis_keys: sequence
            The keys of the regridding axes in the source field.

        dst_axis_keys: sequence
            The keys of the regridding axes in the destination field.

        dst_axis_sizes: sequence, optional
            The sizes of the destination axes.


    :Returns:

        `None`

    """
    # Remove the source coordinates of new field
    for key in f.coordinates(
        filter_by_axis=src_axis_keys, axis_mode="or", todict=True
    ):
        f.del_construct(key)

    # Domain axes
    f_domain_axes = f.domain_axes(todict=True)
    dst_domain_axes = dst.domain_axes(todict=True)
    for src_axis, dst_axis in zip(src_axis_keys, dst_axis_keys):
        f_domain_axis = f_domain_axes[src_axis]
        dst_domain_axis = dst_domain_axes[dst_axis]

        f_domain_axis.set_size(dst_domain_axis.size)

        ncdim = dst_domain_axis.nc_get_dimension(None)
        if ncdim is not None:
            f_domain_axis.nc_set_dimension(ncdim)

    # Coordinates
    axis_map = {
        dst_axis: src_axis
        for dst_axis, src_axis in zip(dst_axis_keys, src_axis_keys)
    }
    dst_data_axes = dst.constructs.data_axes()

    for key, aux in dst.coordinates(
        filter_by_axis=dst_axis_keys, axis_mode="subset", todict=True
    ).items():
        axes = [axis_map[axis] for axis in dst_data_axes[key]]
        f.set_construct(aux, axes=axes)


def update_non_coordinates(
    coord_system,
    f,
    regrid_operator,
    src_axis_keys=None,
    dst_axis_keys=None,
    dst_axis_sizes=None,
):
    """Update the coordinate references of the regridded field.

    :Parameters:

        f: `Field`
            The regridded field. Updated in-place.

        regrid_operator:

        src_axis_keys: sequence of `str`
            The keys of the source regridding axes.

        dst_axis_sizes: sequence, optional
            The sizes of the destination axes.

    :Returns:

        `None`

    """
    dst = operator.get_parameter("dst")

    domain_ancillaries = f.domain_ancillaries(todict=True)

    # Initialise cached value for domain_axes
    domain_axes = None

    data_axes = f.constructs.data_axes()

    # ----------------------------------------------------------------
    # Delete source coordinate references (and all of their domain
    # ancillaries) whose *coordinates* span any of the regridding
    # axes.
    # ----------------------------------------------------------------
    for ref_key, ref in f.coordinate_references(todict=True).items():
        ref_axes = []
        for c_key in ref.coordinates():
            ref_axes.extend(data_axes[c_key])

        if set(ref_axes).intersection(src_axis_keys):
            f.del_coordinate_reference(ref_key)

    # ----------------------------------------------------------------
    # Delete source cell measures and field ancillaries that span any
    # of the regridding axes
    # ----------------------------------------------------------------
    for key in f.constructs(
        filter_by_type=("cell_measure", "field_ancillary"), todict=True
    ):
        if set(data_axes[key]).intersection(src_axis_keys):
            f.del_construct(key)

    # ----------------------------------------------------------------
    # Regrid any remaining source domain ancillaries that span all of
    # the regridding axes
    # ----------------------------------------------------------------
    if coord_system == "spherical":
        regrid = "regrids"
        kwargs = {"src_cyclic": regrid_operator.src_cyclic}  # include others?
    else:
        regrid = "regridc"
        kwargs = {"axes": src_axis_keys}  ### ??? should this be 'axes'

    for da_key in f.domain_ancillaries(todict=True):
        da_axes = data_axes[da_key]
        if not set(da_axes).issuperset(src_axis_keys):
            # Delete any source domain ancillary that doesn't span all
            # of the regidding axes
            f.del_construct(da_key)
            continue

        # Convert the domain ancillary to a field, without any
        # non-coordinate metadata (to prevent them being unnecessarily
        # processed during the regridding of the domain ancillary
        # field, and especially to avoid potential
        # regridding-of-domian-ancillaries infinite recursion).
        da_field = f.convert(key)

        for key in da_field.constructs(
            filter_by_type=(
                "coordinate_reference",
                "domain_ancillary",
                "cell_measure",
            ),
            todict=True,
        ):
            da_field.del_construct(key)

        # Regrid the field containing the domain ancillary
        getattr(da_field, regrid)(regrid_operator, inplace=True, **kwargs)

        # Set sizes of regridded axes
        domain_axes = f.domain_axes(cached=domain_axes, todict=True)
        for axis, new_size in zip(src_axis_keys, dst_axis_sizes):
            domain_axes[axis].set_size(new_size)

        # Put the regridded domain ancillary back into the field
        f.set_construct(
            f._DomainAncillary(source=da_field),
            key=da_key,
            axes=da_axes,
            copy=False,
        )

    # ----------------------------------------------------------------
    # Copy selected coordinate references from the desination grid
    # ----------------------------------------------------------------
    dst_data_axes = dst.constructs.data_axes()

    for ref in dst.coordinate_references(todict=True).values():
        axes = set()
        for c_key in ref.coordinates():
            axes.update(dst_data_axes[c_key])

        if axes and set(axes).issubset(dst_axis_keys):
            f.set_coordinate_reference(ref, parent=dst, strict=True)


# def convert_mask_to_ESMF(mask):
#    """Convert a numpy boolean mask to an ESMF binary mask.
#
#    .. versionadded:: TODODASK
#
#    :Parameters:
#
#        mask: boolean array_like
#            The numpy mask. Must be of boolean data type, but this is
#            not checked.
#
#    :Returns:
#
#        `numpy.ndarray`
#
#    **Examples**
#
#    >>> cf.regrid.utils.convert_mask_to_ESMF([True, False])
#    array([0, 1], dtype=int32)
#
#    """
#    return np.invert(mask).astype('int32')

