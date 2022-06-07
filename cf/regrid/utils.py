"""Worker functions for regridding."""

import logging
from operator import itemgetter

import numpy as np

from .. import _found_ESMF
from ..data import Data
from ..dimensioncoordinate import DimensionCoordinate
from ..functions import regrid_logging

if _found_ESMF:
    try:
        import ESMF
    except Exception:
        _found_ESMF = False


logger = logging.getLogger(__name__)


def regrid_coords(
    regrid_type, f, name=None, method=None, cyclic=None, axes=None
):
    """TODODASK"""
    if regrid_type == "spherical":
        func = regrid_spherical_coords
    else:
        func = regrid_Cartesian_coords

    return func(f, name=name, method=method, cyclic=cyclic, axes=axes)


def regrid_dict_to_field(regrid_type, d, cyclic=None, axes=None, field=None):
    """TODODASK"""
    if regrid_type == "spherical":
        return regrid_spherical_dict_to_field(d, cyclic, field=None)

    # Cartesian
    return regrid_Cartesian_dict_to_field(d, axes, field=None)


def regrid_ESMF_grid(
    regrid_type,
    name=None,
    method=None,
    coords=None,
    cyclic=None,
    bounds=None,
    mask=None,
):
    """TODODASK"""
    if regrid == "spherical":
        func = regrid_spherical_ESMF_grid
    else:
        func = regrid_Cartesian_ESMF_grid

    return func(
        name=name,
        method=method,
        coords=coords,
        cyclic=cyclic,
        bounds=bounds,
        mask=mask,
    )


def regrid_spherical_coords(f, name=None, method=None, cyclic=None, axes=None):
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

        `list`, `list`, `list`, `list`, `bool`
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

    coords_1D = False

    if axes is None:
        # Retrieve the field construct's X and Y dimension
        # coordinates
        x_key, x = f.dimension_coordinate(
            "X",
            item=True,
            default=ValueError(
                f"No unique X dimension coordinate found for the {name} "
                f"field {f!r}. If none is present you "
                "may need to specify the axes keyword."
            ),
        )
        y_key, y = f.dimension_coordinate(
            "Y",
            item=True,
            default=ValueError(
                f"No unique Y dimension coordinate found for the {name} "
                f"field {f!r}. If none is present you "
                "may need to specify the axes keyword."
            ),
        )

        x_axis = data_axes[x_key][0]
        y_axis = data_axes[y_key][0]

        x_size = x.size
        y_size = y.size

        coords_1D = True
    else:
        # --------------------------------------------------------
        # Axes have been provided
        # --------------------------------------------------------
        for key in ("X", "Y"):
            if key not in axes:
                raise ValueError(
                    f"Key {key!r} must be specified for axes of {name} "
                    f"field {f!r}."
                )

        if axes["X"] in (1, 0) and axes["Y"] in (0, 1):
            # Axes specified by integer position in dimensions of
            # lat and lon 2-d auxiliary coordinates
            if axes["X"] == axes["Y"]:
                raise ValueError(
                    "The X and Y axes must be distinct, but they are the same "
                    "for {name} field {f!r}."
                )

            lon_key, lon = f.auxiliary_coordinate(
                "X", item=True, filter_by_naxes=(2,), default=(None, None)
            )
            lat_key, lat = f.auxiliary_coordinate(
                "Y", item=True, filter_by_naxes=(2,), default=(None, None)
            )
            if lon is None:
                raise ValueError(
                    "The X axis does not correspond to a longitude coordinate "
                    f"for {name} field {f!r}."
                )
            if lat is None:
                raise ValueError(
                    "The Y axis does not correspond to a latitude coordinate "
                    f"for {name} field {f!r}."
                )

            if lat.shape != lon.shape:
                raise ValueError(
                    "The shape of the latitude and longitude coordinates "
                    "must be equal but they differ for {name} field {f!r}."
                )

            lon_axes = data_axes[lon_key]
            lat_axes = data_axes[lat_key]
            if lat_axes != lon_axes:
                raise ValueError(
                    "The domain axis constructs spanned by the latitude and "
                    "longitude coordinates should be the same, but they "
                    "differ for {name} field {f!r}."
                )

            x_axis = lon_axes[axes["X"]]
            y_axis = lat_axes[axes["Y"]]

        else:
            x_axis = f.domain_axis(
                axes["X"],
                key=True,
                default=ValueError(
                    f"'X' axis specified for {name} {f!r} field not found."
                ),
            )

            y_axis = f.domain_axis(
                axes["Y"],
                key=True,
                default=ValueError(
                    f"'Y' axis specified for {name} field {f!r} not found."
                ),
            )

        domain_axes = f.domain_axes(todict=True)
        x_size = domain_axes[x_axis].get_size()
        y_size = domain_axes[y_axis].get_size()

    axis_keys = [y_axis, x_axis]
    axis_sizes = [y_size, x_size]

    # If 1-d latitude and longitude coordinates for the field are
    # not found search for 2-d auxiliary coordinates.
    if axes is not None or not x.Units.islongitude or not y.Units.islatitude:
        lon_found = False
        lat_found = False

        for key, aux in f.auxiliary_coordinates(
            filter_by_naxes=(2,), todict=True
        ).items():
            if aux.Units.islongitude:
                if lon_found:
                    raise ValueError(
                        "The 2-d auxiliary longitude coordinate "
                        f"of the {name} field {f!r} is not unique."
                    )
                else:
                    lon_found = True
                    x = aux
                    x_key = key

            if aux.Units.islatitude:
                if lat_found:
                    raise ValueError(
                        "The 2-d auxiliary latitude coordinate "
                        f"of the {name} field {f!r} is not unique."
                    )
                else:
                    lat_found = True
                    y = aux
                    y_key = key

        if not lon_found or not lat_found:
            raise ValueError(
                "Both longitude and latitude coordinates "
                f"were not found for the {name} field {f!r}."
            )

        if axes is not None:
            if set(axis_keys) != set(data_axes[x_key]):
                raise ValueError(
                    "Axes of longitude do not match "
                    f"those specified for {name} field {f!r}."
                )

            if set(axis_keys) != set(data_axes[y_key]):
                raise ValueError(
                    "Axes of latitude do not match "
                    f"those specified for {name} field {f!r}."
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

    coord_keys = {"lat": y_key, "lon": x_key}
    coords = {"lat": y.copy(), "lon": x.copy()}

    # Bounds
    if regridding_is_conservative(method):
        bounds = get_bounds(coords)
        if len(bounds) < len(coords):
            raise ValueError("TODO")
    else:
        bounds = {}

    # Cyclicity
    if cyclic is None:
        cyclic = f.iscyclic(axis_keys[1])

    if not coords_1D:
        for key, coord_key in coord_keys.items():
            coord_axes = f.get_data_axes(coord_key)
            ESMF_order = [
                coord_axes.index(axis_key) for axis_key in (x_axis, y_axis)
            ]
            coords[key] = coords[key].transpose(ESMF_order)
            if key in bounds:
                bounds[key] = bounds[key].transpose(ESMF_order + [-1])

    return axis_keys, axis_sizes, coords, bounds, cyclic


def regrid_Cartesian_coords(f, name=None, method=None, cyclic=None, axes=None):
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
    axis_keys = []
    axis_sizes = []
    coords = {}
    for dim, axis in enumerate(axes):
        key = f.domain_axis(axis, key=True)
        coord = f.dimension_coordinate(filter_by_axis=(key,), default=None)
        if coord is None:
            raise ValueError(
                f"No unique {name} dimension coordinate "
                f"matches key {axis!r}."
            )

        axis_keys.append(key)
        axis_sizes.append(coord.size)
        coords[dim] = coord.copy()

    bounds = []
    if regridding_is_conservative(method):
        bounds = get_bounds(coords)
        if len(bounds) < len(coords):
            raise ValueError("TODO")

    if len(coords) == 1:
        # dummy axis
        data = np.array([np.finfo(float).epsneg, np.finfo(float).eps])
        if regridding_is_conservative(method):
            coords[1] = np.array([0.0])
            bounds[1] = data
        else:
            coords[1] = data

    return axis_keys, axis_sizes, coords, bounds


def regrid_spherical_dict_to_field(d, cyclic=None, field=None):
    """TODODASK"""
    try:
        coords = {"lat": d["latitude"].copy(), "lon": d["longitude"].copy()}
    except KeyError:
        raise ValueError(
            "Dictionary keys 'longitude' and 'latitude' must be "
            "specified for the destination grid"
        )

    coords_1D = False

    if coords["lat"].ndim == 1:
        coords_1D = True
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

    # Set coordiante identities
    coords["lat"].standard_name = "latitude"
    coords["lon"].standard_name = "longitude"

    # Create field
    f = type(field)()

    # Set domain axes
    axis_keys = []
    for size in axis_sizes:
        key = f.set_construct(f._DomainAxis(size), copy=False)
        axis_keys.append(key)

    if coord_1D:
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
    if cyclic:
        f.cyclic(axis_keys[1], period=360)

    if coords_1D:
        yx_axes = None
    else:
        yx_axes = {"Y": axis_keys[0], "X": axis_keys[1]}

    return f, yx_axes


def regrid_Cartesian_dict_to_field(d, axes=None, field=None):
    """TODODASK"""
    if axes is None:
        axes = d.keys()

    f = type(field)()

    axis_keys = []
    for dim, axis in enumerate(axes):
        coord = d[axis]
        key = f.set_construct(f._DomainAxis(coord.size), copy=False)
        f.set_construct(coord, axes=key, copy=False)
        axis_keys.append(key)

    return f, axes_keys


def regrid_spherical_ESMF_grid(
    name=None, method=None, coords=None, cyclic=None, bounds=None, mask=None
):
    """Create a speherical `ESMF` Grid.

    :Parameters:

        coords: sequence of array-like
            The coordinates if not Cartesian it is assume that the
            first is longitude and the second is latitude.

        cyclic: `bool`
            Whether or not the longitude axis is cyclic.

        bounds: `dict` or `None`, optional
            The coordinates if not Cartesian it is assume that the
            first is longitude and the second is latitude.

        coord_ESMF_order: `dict`, optional
            Two tuples one indicating the order of the x and y axes
            for 2-d longitude, one for 2-d latitude. Ignored if
            coordinates are 1-d.

        mask: optional

    :Returns:

        `ESMF.Grid`
            The `ESMF` grid.

    """
    lon = np.asanyarray(coords["lon"])
    lat = np.asanyarray(coords["lat"])
    coords_1D = lon.ndim == 1

    # Parse coordinates for the Grid, and get its shape.
    if coords_1D:
        shape = (lon.size, lat.size)
        lon = lon.reshape(lon.size, 1)
        lat = lat.reshape(1, lat.size)
    else:
        shape = lon.shape
        if lat.shape != shape:
            raise ValueError(
                f"The {name} longitude and latitude coordinates "
                "must have the same shape."
            )

    # Parse bounds for the Grid
    if bounds:
        lon_bounds = np.asanyarray(bounds["lon"])
        lat_bounds = np.asanyarray(bounds["lat"])
        lat_bounds = np.clip(lat_bounds, -90, 90)

        contiguous_bounds((lon_bounds, lat_bounds), name, cyclic, period=360)

        if coords_1D:
            if cyclic:
                x = lon_bounds[:, 0:1]
            else:
                n = lon_bounds.shape[0]
                x = np.empty((n + 1, 1), dtype=lon_bounds.dtype)
                x[:n, 0] = lon_bounds[:, 0]
                x[n, 0] = lon_bounds[-1, 1]

            m = lat_bounds.shape[0]
            y = np.empty((1, m + 1), dtype=lat_bounds.dtype)
            y[0, :m] = lat_bounds[:, 0]
            y[0, m] = lat_bounds[-1, 1]
        else:
            n, m = x_bounds.shape[0:2]

            x = np.empty((n + 1, m + 1), dtype=lon_bounds.dtype)
            x[:n, :m] = lon_bounds[:, :, 0]
            x[:n, m] = lon_bounds[:, -1, 1]
            x[n, :m] = lon_bounds[-1, :, 3]
            x[n, m] = lon_bounds[-1, -1, 2]

            y = np.empty((n + 1, m + 1), dtype=lat_bounds.dtype)
            y[:n, :m] = lat_bounds[:, :, 0]
            y[:n, m] = lat_bounds[:, -1, 1]
            y[n, :m] = lat_bounds[-1, :, 3]
            y[n, m] = lat_bounds[-1, -1, 2]

        lon_bounds = x
        lat_bounds = y

    # Create empty Grid
    X, Y = 0, 1

    max_index = np.array(shape, dtype="int32")
    if bounds:
        staggerlocs = [ESMF.StaggerLoc.CORNER, ESMF.StaggerLoc.CENTER]
    else:
        staggerlocs = [ESMF.StaggerLoc.CENTER]

    if cyclic:
        grid = ESMF.Grid(
            max_index,
            num_peri_dims=1,
            periodic_dim=X,
            pole_dim=Y,
            staggerloc=staggerlocs,
        )
    else:
        grid = ESMF.Grid(max_index, staggerloc=staggerlocs)

    # Populate Grid centres
    c = grid.get_coords(X, staggerloc=ESMF.StaggerLoc.CENTER)
    c[...] = lon
    c = grid.get_coords(Y, staggerloc=ESMF.StaggerLoc.CENTER)
    c[...] = lat

    # Populate Grid corners
    if bounds:
        c = grid.get_coords(X, staggerloc=ESMF.StaggerLoc.CORNER)
        c[...] = lon_bounds
        c = grid.get_coords(Y, staggerloc=ESMF.StaggerLoc.CORNER)
        c[...] = lat_bounds

    # Add a mask
    if mask is not None:
        add_mask(grid, mask)

    return grid


def regrid_Cartesian_ESMF_grid(
    name=None, method=None, coords=None, cyclic=None, bounds=None, mask=None
):
    """Create a Cartesian `ESMF` Grid.

    :Parameters:

        coords: `dict`
            With keys
            X, Y and Z (or some subset thereof)

        bounds: `dict`
            With keys
            X, Y and Z (or some subset thereof)

    :Returns:

        `ESMF.Grid`
            The `ESMF` grid.

    """
    n_axes = len(coords)

    # Parse coordinates for the Grid, and get its shape in "0 1 2"
    # order.
    coords = {dim: np.asanyarray(c) for dim, c in coords.items()}
    shape = [c.size for dim, c in sorted(coords.items())]
    for dim, c in coords.items():
        coords[dim] = c.reshape(
            [c.size if i == dim else 1 for i in range(n_axes)]
        )

    # Parse bounds for the Grid
    if bounds:
        bounds = {dim: np.asanyarray(b) for dim, b in bounds.items()}

        contiguous_bounds(bounds.values(), name, cyclic=False, period=None)

        for dim, b in bounds.items():
            n = b.shape[0]
            if n > 1:
                tmp = np.empty((n + 1,), dtype=b.dtype)
                tmp[0:-1] = b[:, 0]
                tmp[-1] = b[-1, 1]
                b = tmp.reshape(
                    [tmp.size if i == dim else 1 for i in range(n_axes)]
                )
                bounds[dim] = b
                del tmp

    # Create empty Grid
    max_index = np.array(shape, dtype="int32")
    if bounds:
        if n_axes < 3:
            staggerlocs = [ESMF.StaggerLoc.CORNER, ESMF.StaggerLoc.CENTER]
        else:
            staggerlocs = [
                ESMF.StaggerLoc.CENTER_VCENTER,
                ESMF.StaggerLoc.CORNER_VFACE,
            ]
    else:
        if n_axes < 3:
            staggerlocs = [ESMF.StaggerLoc.CENTER]
        else:
            staggerlocs = [ESMF.StaggerLoc.CENTER_VCENTER]

    grid = ESMF.Grid(
        max_index, coord_sys=ESMF.CoordSys.CART, staggerloc=staggerlocs
    )

    # Populate Grid centres
    for dim, c in coords.items():
        if n_axes < 3:
            gridCentre = grid.get_coords(
                dim, staggerloc=ESMF.StaggerLoc.CENTER
            )
        else:
            gridCentre = grid.get_coords(
                dim, staggerloc=ESMF.StaggerLoc.CENTER_VCENTER
            )

        gridCentre[...] = c

    # Populate Grid corners
    if bounds:
        if n_axes < 3:
            staggerloc = ESMF.StaggerLoc.CORNER
        else:
            staggerloc = ESMF.StaggerLoc.CORNER_VFACE

        for dim, b in bounds.items():
            gridCorner = grid.get_coords(dim, staggerloc=staggerloc)
            gridCorner[...] = b

    # Add a mask
    if mask is not None:
        add_mask(grid, mask)

    return grid


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


def check_source_grid(f, method, src_coords, src_bounds, operator):
    """Whether two `ESMF.Grid` instances have identical coordinates.

    :Parameters:

        grid0, grid1: `ESMF.Grid`, `ESMF.Grid`
            The `ESMF` Grid instances to be compared

    :Returns:

        `str`
            

    """
    if bool(src_cyclic) != bool(operator.src_cyclic):
        raise ValueError(
            f"Can't regrid {f!r} with {operator!r}: "
            "Cyclicity of the source field longitude axis "
            "does not match that of the regridding operator."
        )

    if method != operator.method:
        raise ValueError("TOTODASK")

    message = (
        f"Can't regrid {f!r} with {operator!r}: "
        "Source grid coordinates do not match those of "
        "the regridding operator."
    )


def contiguous_bounds(bounds, name, cyclic=None, period=None):
    """TODODASK"""
    message = (
        f"The {name} coordinates must have contiguous, "
        f"non-overlapping bounds for {method} regridding."
    )

    for b in bounds:
        ndim = b.ndim - 1
        if ndim == 1:
            # 1-d cells
            diff = b[1:, 0] - b[:-1, 1]
            if cyclic and period is not None:
                diff = diff % period

            if diff.any():
                raise ValueError(message)

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
                raise ValueError(message)

            diff = b[:, :-1, 2] - b[:, 1:, 3]
            if cyclic and period is not None:
                diff = diff % period

            if diff.any():
                raise ValueError(message)

            # Check cells (j, i) and (j+1, i) are contiguous
            diff = b[:-1, :, 3] - b[1:, :, 0]
            if cyclic and period is not None:
                diff = diff % period

            if diff.any():
                raise ValueError(message)

            diff = b[:-1, :, 2] - b[1:, :, 1]
            if cyclic and period is not None:
                diff = diff % period

            if diff.any():
                raise ValueError(message)

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


def regrid_check_method(method):
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


def regrid_check_use_src_mask(use_src_mask, method):
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


def regrid_get_axis_indices(f, axis_keys):
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

        spherical: `bool`, optional
            TODODAS
            
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


def regrid_get_mask(f, regrid_axes):
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

        Cartesian: `bool`, optional
            Whether the regridding is Cartesian or spherical.

        coords_ext: sequence, optional
            In the case of Cartesian regridding, extension coordinates
            (see _regrid_check_bounds for details).

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


def regrid_update_coordinates(
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

        Cartesian: `bool`, optional
            Whether regridding is Cartesian of spherical, False by
            default.

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


def regrid_update_non_coordinates(
    regrid_type,
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

        Cartesian: `bool`, optional
            Whether to do Cartesian regridding or spherical

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
    if regrid_type == "spherical":
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
