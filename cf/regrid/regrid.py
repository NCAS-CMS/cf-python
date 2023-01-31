"""Worker functions for regridding."""
import logging
from dataclasses import dataclass, field
from typing import Any

import dask.array as da
import numpy as np

from ..functions import DeprecationError, regrid_logging
from ..units import Units
from .regridoperator import RegridOperator

try:
    import ESMF
except Exception:
    ESMF_imported = False
else:
    ESMF_imported = True

logger = logging.getLogger(__name__)

# Mapping of regrid method strings to ESMF method codes. The values
# get replaced with `ESMF.RegridMethod` constants the first time
# `ESMF_initialise` is run.
ESMF_methods = {
    "linear": None,
    "bilinear": None,
    "conservative": None,
    "conservative_1st": None,
    "conservative_2nd": None,
    "nearest_dtos": None,
    "nearest_stod": None,
    "patch": None,
}


@dataclass()
class Grid:
    """A source or destination grid definition.

    .. versionadded:: 3.14.0

    """

    # The domain axis identifiers of the regrid axes, in the order
    # expected by `Data._regrid`. E.g. ['domainaxis3', 'domainaxis2']
    axis_keys: list = field(default_factory=list)
    # The positions of the regrid axes, in the order expected by
    # `Data._regrid`. E.g. [3, 2]
    axis_indices: list = field(default_factory=list)
    # The value of the *src_axes* or *dst_axes* parameter, as
    # appropriate.
    axes: Any = None
    # The sizes of the regrid axes, in the order expected by
    # `Data._regrid`. E.g. [73, 96]
    shape: tuple = None
    # The regrid axis coordinates, in the order expected by
    # `ESMF.Grid`. If the coordinates are 2-d (or more) then the axis
    # order of each coordinate object must be as expected by
    # `ESMF.Grid`.
    coords: list = field(default_factory=list)
    # The regrid axis coordinate bounds, in the order expected by
    # `ESMF.Grid`. If the coordinates are 2-d (or more) then the axis
    # order of each bounds object must be as expected by `ESMF.Grid`.
    bounds: list = field(default_factory=list)
    # For spherical regridding, whether or not the longitude axis is
    # cyclic.
    cyclic: Any = None
    # The coordinate system of the grid.
    coord_sys: str = ""
    # The regridding method.
    method: str = ""
    # Identify the grid as 'source' or 'destination'.
    name: str = ""
    # If True then, for 1-d regridding, the ESMF weights are generated
    # for a 2-d grid for which one of the dimensions is a size 2 dummy
    # dimension.
    dummy_size_2_dimension: bool = False


def regrid(
    coord_sys,
    src,
    dst,
    method=None,
    src_cyclic=None,
    dst_cyclic=None,
    use_src_mask=True,
    use_dst_mask=False,
    src_axes=None,
    dst_axes=None,
    axes=None,
    ignore_degenerate=True,
    return_operator=False,
    check_coordinates=False,
    min_weight=None,
    inplace=False,
    return_ESMF_regrid_operator=False,
):
    """Regrid a field to a new spherical or Cartesian grid.

    This is a worker function primarily intended to be called by
    `cf.Field.regridc` and `cf.Field.regrids`.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Field.regridc`, `cf.Field.regrids`,
                 `cf.data.dask_regrid.regrid`.

    :Parameters:

        coord_sys: `str`
            The name of the coordinate system of the source and
            destination grids. Either ``'spherical'`` or
            ``'Cartesian'``.

        src: `Field`
            The source field to be regridded.

        dst: `Field`, `Domain`, `RegridOperator` or sequence of `Coordinate`
            The definition of the destination grid.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

        method: `str`
            Specify which interpolation method to use during
            regridding. This parameter must be set unless *dst* is a
            `RegridOperator`, when the *method* is ignored.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

        src_cyclic: `None` or `bool`, optional
            For spherical regridding, specifies whether or not the
            source grid longitude axis is cyclic.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

        dst_cyclic: `None` or `bool`, optional
            For spherical regridding, specifies whether or not the
            destination grid longitude axis is cyclic.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

        use_src_mask: `bool`, optional
            Whether or not to use the source grid mask.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

        use_dst_mask: `bool`, optional
            Whether or not to use the source grid mask.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

        src_axes: `dict` or sequence or `None`, optional
            Define the source grid axes to be regridded.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

        dst_axes: `dict` or sequence or `None`, optional
            Define the destination grid axes to be regridded.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.
            Ignored for Cartesian regridding.

        axes: sequence, optional
            Define the axes to be regridded for the source grid and,
            if *dst* is a `Field` or `Domain`, the destination
            grid. Ignored for spherical regridding.

            See `cf.Field.regridc` for details.

        ignore_degenerate: `bool`, optional
            Whether or not to  ignore degenerate cells.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.
            Ignored for Cartesian regridding.

        return_operator: `bool`, optional
            If True then do not perform the regridding, rather return
            the `RegridOperator` instance that defines the regridding
            operation, and which can be used in subsequent calls.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

        check_coordinates: `bool`, optional
            Whether or not to check the regrid operator source grid
            coordinates. Ignored unless *dst* is a `RegridOperator`.

            If True and then the source grid coordinates defined by
            the regrid operator are checked for compatibility against
            those of the *src* field. By default this check is not
            carried out. See the *dst* parameter for details.

            If False then only the computationally cheap tests are
            performed (checking that the coordinate system, cyclicity
            and grid shape are the same).

        inplace: `bool`, optional
            If True then modify *src* in-place and return `None`.

        return_ESMF_regrid_operator: `bool`, optional
            If True then *src* is not regridded, but the `ESMF.Regrid`
            instance for the operation is returned instead. This is
            useful for checking that the field has been regridded
            correctly.

    :Returns:

        `Field` or `None` or `RegridOperator` or `ESMF.Regrid`
            The regridded field construct; or `None` if the operation
            was in-place; or the regridding operator if
            *return_operator* is True.

            If *return_ESMF_regrid_operator* is True then *src* is not
            regridded, but the `ESMF.Regrid` instance for the
            operation is returned instead.

    """
    if not inplace:
        src = src.copy()

    # ----------------------------------------------------------------
    # Parse and check parameters
    # ----------------------------------------------------------------
    if isinstance(dst, RegridOperator):
        regrid_operator = dst
        dst = regrid_operator.dst
        method = regrid_operator.method
        dst_cyclic = regrid_operator.dst_cyclic
        dst_axes = regrid_operator.dst_axes
        if regrid_operator.coord_sys == "Cartesian":
            src_axes = regrid_operator.src_axes

        create_regrid_operator = False
    else:
        create_regrid_operator = True

    if method not in ESMF_methods:
        raise ValueError(
            "Can't regrid: Must set a valid regridding method from "
            f"{sorted(ESMF_methods)}. Got: {method!r}"
        )
    elif method == "bilinear":
        logger.info(
            "Note the 'bilinear' method argument has been renamed to "
            "'linear' at version 3.2.0. It is still supported for now "
            "but please use 'linear' in future. "
            "'bilinear' will be removed at version 4.0.0."
        )

    if not use_src_mask and not method == "nearest_stod":
        raise ValueError(
            "The 'use_src_mask' parameter can only be False when "
            f"using the {method!r} regridding method."
        )

    # Parse the axes
    if coord_sys == "spherical":
        n_axes = 2
    elif not create_regrid_operator:
        n_axes = len(dst_axes)
    elif coord_sys == "Cartesian":
        if isinstance(axes, str):
            axes = (axes,)

        n_axes = len(axes)

        if src_axes is None:
            src_axes = axes
        elif axes is not None:
            raise ValueError(
                "Can't set both the 'axes' and 'src_axes' parameters"
            )
        elif isinstance(src_axes, str):
            src_axes = (src_axes,)

        if dst_axes is None:
            dst_axes = axes
        elif axes is not None:
            raise ValueError(
                "Can't set both the 'axes' and 'dst_axes' parameters"
            )
        elif isinstance(dst_axes, str):
            dst_axes = (dst_axes,)

    if n_axes != 2 and method in ("conservative_2nd", "patch"):
        raise ValueError(
            f"{method!r} regridding is only available for 2-d regridding, "
            f"but {n_axes}-d regridding has been requested."
        )

    if isinstance(dst, src._Domain):
        use_dst_mask = False
        dst = dst.copy()
    elif isinstance(dst, src.__class__):
        dst = dst.copy()
    elif isinstance(dst, RegridOperator):
        pass
    else:
        if isinstance(dst, dict):
            raise DeprecationError(
                "Setting the 'dst' parameter to a dictionary was "
                "deprecated at version 3.14.0. Use a sequence of "
                "Coordinate instances instead. See the docs for details."
            )

        try:
            dst[0]
        except TypeError:
            raise TypeError(
                "The 'dst' parameter must be one of Field, Domain, "
                f"RegridOperator, or a sequence of Coordinate. Got {dst!r}"
            )
        except IndexError:
            # This error will get trapped in one of the following
            # *_coords_to_domain calls
            pass

        # Convert a sequence of Coordinates that define the
        # destination grid into a Domain object
        use_dst_mask = False
        if coord_sys == "spherical":
            dst, dst_axes = spherical_coords_to_domain(
                dst,
                dst_axes=dst_axes,
                cyclic=dst_cyclic,
                domain_class=src._Domain,
            )
        else:
            # Cartesian
            dst, dst_axes = Cartesian_coords_to_domain(
                dst, domain_class=src._Domain
            )

    # ----------------------------------------------------------------
    # Create descriptions of the source and destination grids
    # ----------------------------------------------------------------
    dst_grid = get_grid(
        coord_sys, dst, "destination", method, dst_cyclic, axes=dst_axes
    )

    src_grid = get_grid(
        coord_sys, src, "source", method, src_cyclic, axes=src_axes
    )

    conform_coordinate_units(src_grid, dst_grid)

    if create_regrid_operator:
        # ------------------------------------------------------------
        # Create a new regrid operator
        # ------------------------------------------------------------
        ESMF_manager = ESMF_initialise()  # noqa: F841

        # Create a mask for the destination grid
        dst_mask = None
        grid_dst_mask = None
        if use_dst_mask:
            dst_mask = get_mask(dst, dst_grid)
            if (
                method in ("patch", "conservative_2nd", "nearest_stod")
                or return_ESMF_regrid_operator
            ):
                # For these regridding methods, the destination mask
                # must be taken into account during the ESMF
                # calculation of the regrid weights, rather than the
                # mask being applied retrospectively to weights that
                # have been calculated assuming no destination grid
                # mask. See `cf.data.dask_regrid`.
                grid_dst_mask = np.array(dst_mask.transpose())
                dst_mask = None

        # Create the destination ESMF.Grid
        dst_ESMF_grid = create_ESMF_grid(dst_grid, mask=grid_dst_mask)
        del grid_dst_mask

        # Create a mask for the source grid
        src_mask = None
        grid_src_mask = None
        if use_src_mask and (
            method in ("patch", "conservative_2nd", "nearest_stod")
            or return_ESMF_regrid_operator
        ):
            # For patch recovery and second-order conservative
            # regridding, the source mask needs to be taken into
            # account during the ESMF calculation of the regrid
            # weights, rather than the mask being applied
            # retrospectively to weights that have been calculated
            # assuming no source grid mask. See `cf.data.dask_regrid`.
            src_mask = get_mask(src, src_grid)
            grid_src_mask = np.array(src_mask.transpose())
            if not grid_src_mask.any():
                # There are no masked source cells, so we can collapse
                # the mask that gets stored in the regrid operator.
                src_mask = np.array(False)
                grid_src_mask = src_mask

        # Create the source ESMF.Grid
        src_ESMF_grid = create_ESMF_grid(src_grid, mask=grid_src_mask)

        del grid_src_mask

        ESMF_regrid_operator = [] if return_ESMF_regrid_operator else None

        # Create regrid weights
        weights, row, col, start_index = create_ESMF_weights(
            method,
            src_ESMF_grid,
            dst_ESMF_grid,
            ignore_degenerate=ignore_degenerate,
            quarter=src_grid.dummy_size_2_dimension,
            ESMF_regrid_operator=ESMF_regrid_operator,
        )

        if return_ESMF_regrid_operator:
            # Return the equivalent ESMF.Regrid operator
            return ESMF_regrid_operator[-1]

        # Still here? Then we've finished with ESMF, so finalise the
        # ESMF manager. This is done to free up any Persistent
        # Execution Threads (PETs) created by the ESMF Virtual Machine
        # (https://earthsystemmodeling.org/esmpy_doc/release/latest/html/api.html#resource-allocation).
        del ESMF_manager

        if src_grid.dummy_size_2_dimension:
            # We have a dummy size_2 dimension, so remove its
            # coordinates so that they don't end up in the regrid
            # operator.
            src_grid.coords.pop()
            if src_grid.bounds:
                src_grid.bounds.pop()

        # Create regrid operator
        regrid_operator = RegridOperator(
            weights,
            row,
            col,
            coord_sys=coord_sys,
            method=method,
            src_shape=src_grid.shape,
            dst_shape=dst_grid.shape,
            src_cyclic=src_grid.cyclic,
            dst_cyclic=dst_grid.cyclic,
            src_coords=src_grid.coords,
            src_bounds=src_grid.bounds,
            src_mask=src_mask,
            dst_mask=dst_mask,
            start_index=start_index,
            src_axes=src_axes,
            dst_axes=dst_axes,
            dst=dst,
        )
    else:
        # ------------------------------------------------------------
        # Check that the given regrid operator is compatible with the
        # source field's grid
        # ------------------------------------------------------------
        check_operator(
            src, src_grid, regrid_operator, check_coordinates=check_coordinates
        )

    if return_operator:
        return regrid_operator

    # ----------------------------------------------------------------
    # Still here? Then do the regridding
    # ----------------------------------------------------------------
    regridded_axis_sizes = {
        axis: size for axis, size in zip(src_grid.axis_indices, dst_grid.shape)
    }

    regridded_data = src.data._regrid(
        method=method,
        operator=regrid_operator,
        regrid_axes=src_grid.axis_indices,
        regridded_sizes=regridded_axis_sizes,
        min_weight=min_weight,
    )

    # ----------------------------------------------------------------
    # Update the regridded metadata
    # ----------------------------------------------------------------
    update_non_coordinates(
        src, dst, regrid_operator, src_grid=src_grid, dst_grid=dst_grid
    )

    update_coordinates(src, dst, src_grid=src_grid, dst_grid=dst_grid)

    # ----------------------------------------------------------------
    # Insert regridded data into the new field
    # ----------------------------------------------------------------
    src.set_data(regridded_data, axes=src.get_data_axes(), copy=False)

    if coord_sys == "spherical":
        # Set the cyclicity of the longitude axis of the new field
        key, x = src.dimension_coordinate("X", default=(None, None), item=True)
        if x is not None and x.Units.equivalent(Units("degrees")):
            src.cyclic(
                key, iscyclic=dst_grid.cyclic, period=360, config={"coord": x}
            )

    # Return the regridded source field
    if inplace:
        return

    return src


def spherical_coords_to_domain(
    dst, dst_axes=None, cyclic=None, domain_class=None
):
    """Convert a sequence of `Coordinate` to spherical grid definition.

    .. versionadded:: 3.14.0

    :Parameters:

        dst: sequence of `Coordinate`
            Two 1-d dimension coordinate constructs or two 2-d
            auxiliary coordinate constructs that define the spherical
            latitude and longitude grid coordinates (in any order) of
            the destination grid.

        dst_axes: `dict` or `None`
            When *d* contains 2-d latitude and longitude coordinates
            then the X and Y dimensions must be identified with the
            *dst_axes* dictionary, with keys ``'X'`` and ``'Y'``. The
            dictionary values identify a unique domain axis by its
            position in the 2-d coordinates' data arrays, i.e. the
            dictionary values must be ``0`` and ``1``. Ignored if *d*
            contains 1-d coordinates.

        cyclic: `bool` or `None`
            Specifies whether or not the destination grid longitude
            axis is cyclic (i.e. the first and last cells of the axis
            are adjacent). If `None` then the cyclicity will be
            inferred from the coordinates, defaulting to `False` if it
            can not be determined.

        domain_class: `Domain` class
            The domain class used to create the new `Domain` instance.

    :Returns:

        `Domain`, `dict`
            The new domain containing the grid; and a dictionary
            identifying the domain axis identifiers of the X and Y
            regrid axes (as defined by the *dst_axes* parameter of
            `cf.Field.regrids`).

    """
    coords = {}
    for c in dst:
        try:
            c = c.copy()
            if c.Units.islatitude:
                c.standard_name = "latitude"
                coords["lat"] = c
            elif c.Units.islongitude:
                c.standard_name = "longitude"
                coords["lon"] = c
        except AttributeError:
            pass

    if len(coords) != 2:
        raise ValueError(
            "When 'dst' is a sequence it must be of latitude and "
            f"longitude coordinate constructs. Got: {dst!r}"
        )

    coords_1d = False

    if coords["lat"].ndim == 1:
        coords_1d = True
        axis_sizes = [coords["lat"].size, coords["lon"].size]
        if coords["lon"].ndim != 1:
            raise ValueError(
                "When 'dst' is a sequence of latitude and longitude "
                "coordinates, they must have the same number of dimensions."
            )
    elif coords["lat"].ndim == 2:
        message = (
            "When 'dst' is a sequence of 2-d latitude and longitude "
            "coordinates, then 'dst_axes' must be either "
            "{'X': 0, 'Y': 1} or {'X': 1, 'Y': 0}"
        )

        if dst_axes is None:
            raise ValueError(message)

        axis_sizes = coords["lat"].shape
        if dst_axes["X"] == 0:
            axis_sizes = axis_sizes[::-1]
        elif dst_axes["Y"] != 0:
            raise ValueError(message)

        if coords["lat"].shape != coords["lon"].shape:
            raise ValueError(
                "When 'dst' is a sequence of latitude and longitude "
                "coordinates, they must have the same shape."
            )
    else:
        raise ValueError(
            "When 'dst' is a sequence of latitude and longitude "
            "coordinates, they must be either 1-d or 2-d."
        )

    d = domain_class()

    # Set domain axes
    axis_keys = []
    for size in axis_sizes:
        key = d.set_construct(d._DomainAxis(size), copy=False)
        axis_keys.append(key)

    if coords_1d:
        # Set 1-d coordinates
        for key, axis in zip(("lat", "lon"), axis_keys):
            d.set_construct(coords[key], axes=axis, copy=False)
    else:
        # Set 2-d coordinates
        coord_axes = axis_keys
        if dst_axes["X"] == 0:
            coord_axes = coord_axes[::-1]

        for coord in coords.values():
            d.set_construct(coord, axes=coord_axes, copy=False)

    if cyclic is not None:
        # Reset X axis cyclicity
        d.cyclic(axis_keys[1], iscyclic=cyclic, period=360)

    dst_axes = {"Y": axis_keys[0], "X": axis_keys[1]}

    return d, dst_axes


def Cartesian_coords_to_domain(dst, domain_class=None):
    """Convert a sequence of `Coordinate` to Cartesian grid definition.

    .. versionadded:: 3.14.0

    :Parameters:

        dst: sequence of `DimensionCoordinate`
            Between one and three 1-d dimension coordinate constructs
            that define the coordinates of the grid. The order of the
            coordinate constructs **must** match the order of source
            field regridding axes defined elsewhere by the *src_axes*
            or *axes* parameter.

        domain_class: `Domain` class
            The domain class used to create the new `Domain` instance.

    :Returns:

        `Domain`, `list`
            The new domain containing the grid; and a list identifying
            the domain axis identifiers of the regrid axes (as defined
            by the *dst_axes* parameter of `cf.Field.regridc`).

    """
    if not dst:
        pass

    d = domain_class()

    axis_keys = []
    for coord in dst:
        axis = d.set_construct(d._DomainAxis(coord.size), copy=False)
        d.set_construct(coord, axes=axis, copy=True)
        axis_keys.append(axis)

    return d, axis_keys


def get_grid(coord_sys, f, name=None, method=None, cyclic=None, axes=None):
    """Get axis and coordinate information for regridding.

    See `spherical_grid` and `Cartesian_grid` for details.

    .. versionadded:: 3.14.0

        coord_sys: `str`
            The coordinate system of the source and destination grids.

    """
    if coord_sys == "spherical":
        return spherical_grid(
            f, name=name, method=method, cyclic=cyclic, axes=axes
        )

    # Cartesian
    return Cartesian_grid(f, name=name, method=method, axes=axes)


def spherical_grid(f, name=None, method=None, cyclic=None, axes=None):
    """Get latitude and longitude coordinate information.

    Retrieve the latitude and longitude coordinates of a field, as
    well as some associated information. If 1-d lat/lon coordinates
    are found then these are returned. Otherwise if 2-d lat/lon
    coordinates found then these are returned.

    .. versionadded:: 3.14.0

    .. seealso:: `Cartesian_grid`

    :Parameters:

        f: `Field` or `Domain`
            The construct from which to get the grid information.

        name: `str`
            A name to identify the grid.

        method: `str`
            The regridding method.

        cyclic: `None` or `bool`
            Specifies whether or not the source grid longitude axis is
            cyclic.

            See `cf.Field.regrids` for details.

        axes: `dict`, optional
            A dictionary identifying the X and Y axes of the domain,
            with keys ``'X'`` and ``'Y'``.

            *Parameter example:*
              ``axes={'X': 'ncdim%x', 'Y': 'ncdim%y'}``

            *Parameter example:*
              When *f* is a `Field`, ``axes={'X': 1, 'Y': 0}``, where
              ``0`` and ``1`` are axis positions in the 2-d
              coordinates of *f*.

    :Returns:

        `Grid`
            The grid definition.

    """
    data_axes = f.constructs.data_axes()

    coords_1d = False
    coords_2d = False

    # Look for 1-d X and Y dimension coordinates
    lon_key_1d, lon_1d = f.dimension_coordinate(
        "X", item=True, default=(None, None)
    )
    lat_key_1d, lat_1d = f.dimension_coordinate(
        "Y", item=True, default=(None, None)
    )

    if lon_1d is not None and lat_1d is not None:
        x_axis = data_axes[lon_key_1d][0]
        y_axis = data_axes[lat_key_1d][0]
        if lon_1d.Units.islongitude and lat_1d.Units.islatitude:
            # Found 1-d latitude and longitude dimension coordinates
            coords_1d = True
            lon = lon_1d
            lat = lat_1d

    if not coords_1d:
        # Look for 2-d X and Y auxiliary coordinates
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
            # Found 2-d latitude and longitude auxiliary coordinates
            coords_2d = True

            lon_axes = data_axes[lon_key]
            lat_axes = data_axes[lat_key]

            if lon_1d is None or lat_1d is None:
                if not axes or ("X" not in axes or "Y" not in axes):
                    raise ValueError(
                        f"The {name} latitude and longitude coordinates "
                        "are 2-d but the X and Y axes could not be identified "
                        "from dimension coordinates nor from the "
                        f"{'src_axes' if name == 'source' else 'dst_axes'!r}"
                        "parameter"
                    )

                x_axis = f.domain_axis(axes["X"], key=True, default=None)
                y_axis = f.domain_axis(axes["Y"], key=True, default=None)
                if x_axis is None or y_axis is None:
                    raise ValueError(
                        f"Two {name} domain axes can be found from the "
                        "identifiers given by the "
                        f"{'src_axes' if name == 'source' else 'dst_axes'!r}"
                        f"parameter: {axes!r}"
                    )

            if not set(lon_axes) == set(lat_axes) == set((x_axis, y_axis)):
                raise ValueError(
                    f"The {name} domain axes identified by the "
                    f"{'src_axes' if name == 'source' else 'dst_axes'!r}"
                    f"parameter: {axes!r} do not match the domain axes "
                    f"spanned by the {name} grid latitude and longitude "
                    "coordinates"
                )

    if x_axis == y_axis:
        raise ValueError(
            "The X and Y axes must be distinct, but they are "
            f"the same for {name} field {f!r}."
        )

    if not (coords_1d or coords_2d):
        raise ValueError(
            "Could not find 1-d nor 2-d latitude and longitude coordinates"
        )

    # Get X and Y axis sizes
    domain_axes = f.domain_axes(todict=True)
    x_size = domain_axes[x_axis].size
    y_size = domain_axes[y_axis].size

    # Source grid size 1 dimensions are problematic for ESMF for some
    # methods
    if (
        name == "source"
        and method in ("linear", "bilinear", "patch")
        and (x_size == 1 or y_size == 1)
    ):
        raise ValueError(
            f"Neither the X nor Y dimensions of the {name} field"
            f"{f!r} can be of size 1 for spherical {method!r} regridding."
        )

    coords = [lon, lat]  # ESMF order

    # Convert 2-d coordinate arrays to ESMF axis order = [X, Y]
    if coords_2d:
        for dim, coord_key in enumerate((lon_key, lat_key)):
            coord_axes = data_axes[coord_key]
            ESMF_order = [coord_axes.index(axis) for axis in (x_axis, y_axis)]
            coords[dim] = coords[dim].transpose(ESMF_order)

    bounds = get_bounds(method, coords)

    # Set cyclicity of X axis
    if cyclic is None:
        cyclic = f.iscyclic(x_axis)

    axis_keys = [y_axis, x_axis]
    axis_sizes = [y_size, x_size]

    if f.construct_type == "domain":
        axis_indices = [0, 1]
    else:
        # Make sure that the data array spans all of the regridding
        # axes. This might change 'f' in-place.
        data_axes = f.get_data_axes()
        for axis_key in axis_keys:
            if axis_key not in data_axes:
                f.insert_dimension(axis_key, position=-1, inplace=True)

        # The indices of the regridding axes, in the order expected by
        # `Data._regrid`.
        data_axes = f.get_data_axes()
        axis_indices = [data_axes.index(key) for key in axis_keys]

    return Grid(
        axis_keys=axis_keys,
        axis_indices=axis_indices,
        axes={"X": x_axis, "Y": y_axis},
        shape=tuple(axis_sizes),
        coords=coords,
        bounds=bounds,
        cyclic=bool(cyclic),
        coord_sys="spherical",
        method=method,
        name=name,
    )


def Cartesian_grid(f, name=None, method=None, axes=None):
    """Retrieve the specified Cartesian dimension coordinates of the
    field and their corresponding keys.

    .. versionadded:: 3.14.0

    .. seealso:: `spherical_grid`

    :Parameters:

        f: `Field` or `Domain`
           The field or domain construct from which to get the
           coordinates.

        name: `str`
            A name to identify the grid.

        axes: sequence of `str`
            Specifiers for the dimension coordinates to be
            retrieved. See `cf.Field.domain_axes` for details.

    :Returns:

        `Grid`
            The grid definition.

    """
    if not axes:
        if name == "source":
            raise ValueError(
                "Must set either the 'axes' or 'src_axes' parameter"
            )
        else:
            raise ValueError(
                "When 'dst' is a Field or Domain, must set either the "
                "'axes' or 'dst_axes' parameter"
            )

    n_axes = len(axes)
    if not 1 <= n_axes <= 3:
        raise ValueError(
            "Between 1 and 3 axes must be individually specified "
            f"for Cartesian regridding. Got: {axes!r}"
        )

    # Find the axis keys, sizes and indices of the regrid axes, in the
    # order that they have been given by the 'axes' parameters.
    axis_keys = []
    axis_sizes = []
    for axis in axes:
        key, domain_axis = f.domain_axis(axis, item=True, default=(None, None))
        if key is None:
            raise ValueError(
                f"No {name} grid domain axis could be found from {axis!r}"
            )

        axis_keys.append(key)
        axis_sizes.append(domain_axis.size)

    if f.construct_type == "domain":
        axis_indices = list(range(len(axis_keys)))
    else:
        # Make sure that the data array spans all of the regridding
        # axes. This might change 'f' in-place.
        data_axes = f.get_data_axes()
        for axis_key in axis_keys:
            if axis_key not in data_axes:
                f.insert_dimension(axis_key, position=-1, inplace=True)

        # The indices of the regridding axes, in the order expected by
        # `Data._regrid`.
        data_axes = f.get_data_axes()
        axis_indices = [data_axes.index(key) for key in axis_keys]

    coords = []
    for key in axis_keys[::-1]:
        coord = f.dimension_coordinate(filter_by_axis=(key,), default=None)
        if coord is None:
            raise ValueError(
                f"No unique {name} dimension coordinate for domain axis "
                f"{key!r}."
            )

        coords.append(coord)

    bounds = get_bounds(method, coords)

    dummy_size_2_dimension = False
    if len(coords) == 1:
        # Create a dummy axis because ESMF doesn't like creating
        # weights for 1-d regridding
        data = np.array([-1.0, 1.0])
        if conservative_regridding(method):
            # For conservative regridding the extra dimension can be
            # size 1
            coords.append(np.array([0.0]))
            bounds.append(np.array([data]))
        else:
            # For linear regridding the extra dimension must be size 2
            coords.append(data)
            dummy_size_2_dimension = True

    return Grid(
        axis_keys=axis_keys,
        axis_indices=axis_indices,
        axes=axis_keys,
        shape=tuple(axis_sizes),
        coords=coords,
        bounds=bounds,
        cyclic=False,
        coord_sys="Cartesian",
        method=method,
        name=name,
        dummy_size_2_dimension=dummy_size_2_dimension,
    )


def conform_coordinate_units(src_grid, dst_grid):
    """Make the source and destination coordinates have the same units.

    Modifies *src_grid* in-place so that its coordinates and bounds
    have the same units as the coordinates and bounds of *dst_grid*.

    .. versionadded:: 3.14.0

    .. seealso:: `get_spherical_grid`, `regrid`

    :Parameters:

        src_grid: `Grid`
            The definition of the source grid.

        dst_grid: `Grid`
            The definition of the destination grid.

    :Returns:

        `None`

    """
    if src_grid.coord_sys == "spherical":
        # For spherical coordinate systems, the units will have
        # already been checked in `get_spherical_grid`.
        return

    for src, dst in zip(
        (src_grid.coords, src_grid.bounds), (dst_grid.coords, dst_grid.bounds)
    ):
        for dim, (s, d) in enumerate(zip(src, dst)):
            s_units = getattr(s, "Units", None)
            d_units = getattr(d, "Units", None)
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


def check_operator(src, src_grid, regrid_operator, check_coordinates=False):
    """Check compatibility of the source field and regrid operator.

    .. versionadded:: 3.14.0

    .. seealso:: `regrid`

    :Parameters:

        src: `Field`
            The field to be regridded.

        src_grid: `Grid`
            The definition of the source grid.

        regrid_operator: `RegridOperator`
            The regrid operator.

        check_coordinates: `bool`, optional
            Whether or not to check the regrid operator source grid
            coordinates. Ignored unless *dst* is a `RegridOperator`.

            If True and then the source grid coordinates defined by
            the regrid operator are checked for compatibility against
            those of the *src* field. By default this check is not
            carried out. See the *dst* parameter for details.

            If False then only the computationally cheap tests are
            performed (checking that the coordinate system, cyclicity
            and grid shape are the same).

    :Returns:

        `bool`
            Returns `True` if the source grid coordinates and bounds
            match those of the regrid operator. Otherwise an exception
            is raised.

    """
    if regrid_operator.coord_sys != src_grid.coord_sys:
        raise ValueError(
            f"Can't regrid {src!r} with {regrid_operator!r}: "
            "Coordinate system mismatch"
        )

    if regrid_operator.src_cyclic != src_grid.cyclic:
        raise ValueError(
            f"Can't regrid {src!r} with {regrid_operator!r}: "
            "Source grid cyclicity mismatch"
        )

    if regrid_operator.src_shape != src_grid.shape:
        raise ValueError(
            f"Can't regrid {src!r} with {regrid_operator!r}: "
            "Source grid shape mismatch: "
            f"{src_grid.shape} != {regrid_operator.src_shape}"
        )

    if not check_coordinates:
        return True

    # Still here? Then check also the coordinates and bounds
    for a, b in zip(regrid_operator.src_coords, src_grid.coords):
        a = np.asanyarray(a)
        b = np.asanyarray(b)
        if not np.array_equal(a, b):
            raise ValueError(
                f"Can't regrid {src!r} with {regrid_operator!r}: "
                "Source grid coordinates mismatch"
            )

    for a, b in zip(regrid_operator.src_bounds, src_grid.bounds):
        a = np.asanyarray(a)
        b = np.asanyarray(b)
        if not np.array_equal(a, b):
            raise ValueError(
                f"Can't regrid {src!r} with {regrid_operator!r}: "
                "Source grid coordinate bounds mismatch"
            )

    return True


def ESMF_initialise():
    """Initialise the `ESMF` manager.

    The is a null operation if the manager has already been
    initialised.

    Whether ESMF logging is enabled or not is determined by
    `cf.regrid_logging`.

    Also initialises the global 'ESMF_methods' dictionary, unless it
    has already been initialised.

    :Returns:

        `ESMF.Manager`
            The `ESMF` manager.

    """
    if not ESMF_imported:
        raise RuntimeError(
            "Regridding will not work unless the ESMF library is installed"
        )

    # Update the global 'ESMF_methods' dictionary
    if ESMF_methods["linear"] is None:
        ESMF_methods.update(
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


def create_ESMF_grid(grid=None, mask=None):
    """Create an `ESMF` Grid.

    .. versionadded:: 3.14.0

    :Parameters:

        grid: `Grid`

        mask: array_like, optional
            The grid mask. If `None` (the default) then there are no
            masked cells, other must be a Boolean array, with True for
            masked elements, that broadcasts to the ESMF N-d
            coordinates.

    :Returns:

        `ESMF.Grid`

    """
    coords = grid.coords
    bounds = grid.bounds
    cyclic = grid.cyclic

    num_peri_dims = 0
    periodic_dim = 0
    spherical = False
    if grid.coord_sys == "spherical":
        spherical = True
        lon, lat = 0, 1
        coord_sys = ESMF.CoordSys.SPH_DEG
        if cyclic:
            num_peri_dims = 1
            periodic_dim = lon
    else:
        # Cartesian
        coord_sys = ESMF.CoordSys.CART

    # Parse coordinates for the ESMF.Grid and get its shape
    n_axes = len(coords)
    coords_1d = coords[0].ndim == 1

    coords = [np.asanyarray(c) for c in coords]
    if coords_1d:
        # 1-d coordinates for N-d regridding
        shape = [c.size for c in coords]
        coords = [
            c.reshape([c.size if i == dim else 1 for i in range(n_axes)])
            for dim, c in enumerate(coords)
        ]
    elif n_axes == 2:
        # 2-d coordinates for 2-d regridding
        shape = coords[0].shape
    else:
        raise ValueError(
            "Coordinates must be 1-d, or possibly 2-d for 2-d regridding"
        )

    # Parse bounds for the ESMF.Grid
    if bounds:
        bounds = [np.asanyarray(b) for b in bounds]

        if spherical:
            bounds[lat] = np.clip(bounds[lat], -90, 90)
            if not contiguous_bounds(bounds[lat]):
                raise ValueError(
                    f"The {grid.name} latitude coordinates must have "
                    f"contiguous, non-overlapping bounds for {grid.method} "
                    "regridding."
                )

            if not contiguous_bounds(bounds[lon], cyclic=cyclic, period=360):
                raise ValueError(
                    f"The {grid.name} longitude coordinates must have "
                    f"contiguous, non-overlapping bounds for {grid.method} "
                    "regridding."
                )
        else:
            # Cartesian
            for b in bounds:
                if not contiguous_bounds(b):
                    raise ValueError(
                        f"The {grid.name} coordinates must have contiguous, "
                        "non-overlapping bounds for "
                        f"{grid.method} regridding."
                    )

        # Convert each bounds to a grid with no repeated values.
        if coords_1d:
            # Bounds for 1-d coordinates.
            #
            # E.g. if the ESMF.Grid is (X, Y) then for non-cyclic
            #      bounds <CF Bounds: longitude(96, 2) degrees_east>
            #      we create a new bounds array with shape (97, 1);
            #      and for non-cyclic bounds <CF Bounds: latitude(73,
            #      2) degrees_north> we create a new bounds array with
            #      shape (1, 74). When multiplied, these arrays would
            #      create the 2-d (97, 74) bounds grid expected by
            #      ESMF.Grid.
            #
            #      Note that if the X axis were cyclic, then its new
            #      bounds array would have shape (96, 1).
            for dim, b in enumerate(bounds):
                if spherical and cyclic and dim == lon:
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
            #
            # E.g. if the ESMF.Grid is (X, Y) then for bounds <CF
            #      Bounds: latitude(96, 73, 2) degrees_north> with a
            #      non-cyclic X axis, we create a new bounds array
            #      with shape (97, 74).
            #
            #      Note that if the X axis were cyclic, then the new
            #      bounds array would have shape (96, 74).
            if spherical and cyclic:
                for dim, b in enumerate(bounds):
                    n, m = b.shape[0:2]
                    tmp = np.empty((n, m + 1), dtype=b.dtype)
                    tmp[:, :m] = b[:, :, 0]
                    if dim == lon:
                        tmp[:, m] = b[:, -1, 0]
                    else:
                        tmp[:, m] = b[:, -1, 1]

                    bounds[dim] = tmp
            else:
                for dim, b in enumerate(bounds):
                    n, m = b.shape[0:2]
                    tmp = np.empty((n + 1, m + 1), dtype=b.dtype)
                    tmp[:n, :m] = b[:, :, 0]
                    tmp[:n, m] = b[:, -1, 1]
                    tmp[n, :m] = b[-1, :, 3]
                    tmp[n, m] = b[-1, -1, 2]
                    bounds[dim] = tmp

    # Define the ESMF.Grid stagger locations
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

    # Create an empty ESMF.Grid
    esmf_grid = ESMF.Grid(
        max_index=np.array(shape, dtype="int32"),
        coord_sys=coord_sys,
        num_peri_dims=num_peri_dims,
        periodic_dim=periodic_dim,
        staggerloc=staggerlocs,
    )

    # Populate the ESMF.Grid centres
    for dim, c in enumerate(coords):
        if n_axes == 3:
            grid_centre = esmf_grid.get_coords(
                dim, staggerloc=ESMF.StaggerLoc.CENTER_VCENTER
            )
        else:
            grid_centre = esmf_grid.get_coords(
                dim, staggerloc=ESMF.StaggerLoc.CENTER
            )

        grid_centre[...] = c

    # Populate the ESMF.Grid corners
    if bounds:
        if n_axes == 3:
            staggerloc = ESMF.StaggerLoc.CORNER_VFACE
        else:
            staggerloc = ESMF.StaggerLoc.CORNER

        for dim, b in enumerate(bounds):
            grid_corner = esmf_grid.get_coords(dim, staggerloc=staggerloc)
            grid_corner[...] = b

    # Add an ESMF.Grid mask
    if mask is not None:
        if mask.dtype != bool:
            raise ValueError(
                "'mask' must be None or a Boolean array. Got: "
                f"dtype={mask.dtype}"
            )

        if not mask.any():
            mask = None

        if mask is not None:
            grid_mask = esmf_grid.add_item(ESMF.GridItem.MASK)
            if len(grid.coords) == 2 and mask.ndim == 1:
                # ESMF grid has a dummy size 1 dimension, so we need to
                # include this in the mask as well.
                mask = np.expand_dims(mask, 1)

            # Note: 'mask' has True/False for masked/unmasked
            #       elements, but the ESMF mask requires 0/1 for
            #       masked/unmasked elements.
            grid_mask[...] = np.invert(mask).astype("int32")

    return esmf_grid


def create_ESMF_weights(
    method,
    src_ESMF_grid,
    dst_ESMF_grid,
    ignore_degenerate,
    quarter=False,
    ESMF_regrid_operator=None,
):
    """Create the `ESMF` regridding weights.

    .. versionadded:: 3.14.0

    :Parameters:

        method: `str`
            The regridding method.

        src_ESMF_grid: `ESMF.Grid`
            The source grid.

        dst_ESMF_grid: `ESMF.Grid`
            The destination grid.

        ignore_degenerate: `bool`, optional
            Whether or not to ignore degenerate cells.

            See `cf.Field.regrids` for details.

        quarter: `bool`, optional
            If True then only return weights corresponding to the top
            left hand quarter of the weights matrix. This is necessary
            for 1-d regridding, for which the ESMF weights need to be
            generated for a 2-d grid for which one of the dimensions
            is a size 2 dummy dimension.

            .. seealso:: `Cartesian_grid`

         ESMF_regrid_operator: `None` or `list`, optional
            If a `list` then the `ESMF.Regrid` instance that created
            the instance is made available as the list's last
            element.

    :Returns:

        4-`tuple` of `numpy.ndarray`
            * weights: The 1-d array of the regridding weights.
            * row: The 1-d array of the row indices of the regridding
                   weights in the dense weights matrix, which has J
                   rows and I columns, where J and I are the total
                   number of cells in the destination and source grids
                   respectively. The start index is 1.
            * col: The 1-d array of column indices of the regridding
                   weights in the dense weights matrix, which has J
                   rows and I columns, where J and I are the total
                   number of cells in the destination and source grids
                   respectively. The start index is 1.
            * start_index: The non-negative integer start index of the
                   row and column indices.

    """
    src_ESMF_field = ESMF.Field(src_ESMF_grid, "src")
    dst_ESMF_field = ESMF.Field(dst_ESMF_grid, "dst")

    mask_values = np.array([0], dtype="int32")

    # Create the ESMF.regrid operator
    r = ESMF.Regrid(
        src_ESMF_field,
        dst_ESMF_field,
        regrid_method=ESMF_methods.get(method),
        unmapped_action=ESMF.UnmappedAction.IGNORE,
        ignore_degenerate=bool(ignore_degenerate),
        src_mask_values=mask_values,
        dst_mask_values=mask_values,
        norm_type=ESMF.api.constants.NormType.FRACAREA,
        factors=True,
    )

    weights = r.get_weights_dict(deep_copy=True)
    row = weights["row_dst"]
    col = weights["col_src"]
    weights = weights["weights"]

    if quarter:
        # The weights were created with a dummy size 2 dimension such
        # that the weights for each dummy axis element are
        # identical. The duplicate weights need to be removed.
        #
        # To do this, only retain the indices that correspond to the
        # top left quarter of the weights matrix in dense form. I.e.
        # if w is the NxM (N, M both even) dense form of the weights,
        # then this is equivalent to w[:N//2, :M//2].
        index = np.where(
            (row <= dst_ESMF_field.data.size // 2)
            & (col <= src_ESMF_field.data.size // 2)
        )
        weights = weights[index]
        row = row[index]
        col = col[index]

    if ESMF_regrid_operator is None:
        # Destroy ESMF objects
        src_ESMF_grid.destroy()
        dst_ESMF_grid.destroy()
        src_ESMF_field.destroy()
        dst_ESMF_field.destroy()
        r.srcfield.grid.destroy()
        r.srcfield.destroy()
        r.dstfield.grid.destroy()
        r.dstfield.destroy()
        r.destroy()
    else:
        # Make the Regrid instance available via the
        # 'ESMF_regrid_operator' list
        ESMF_regrid_operator.append(r)

    return weights, row, col, 1


def contiguous_bounds(b, cyclic=False, period=None):
    """Determine whether or not bounds are contiguous.

    :Parameters:

        b: array_like
            The bounds.

        cyclic: `bool`, optional
            Specifies whether or not the bounds are cyclic.

        period: number, optional
            If *cyclic* is True then define the period of the cyclic
            axis.

    :Returns:

        `bool`
            Whether or not the bounds are contiguous.

    """
    ndim = b.ndim - 1
    if ndim == 1:
        # 1-d cells
        diff = b[1:, 0] - b[:-1, 1]
        if cyclic:
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
        if cyclic:
            diff = diff % period

        if diff.any():
            return False

        diff = b[:, :-1, 2] - b[:, 1:, 3]
        if cyclic:
            diff = diff % period

        if diff.any():
            return False

        # Check cells (j, i) and (j+1, i) are contiguous
        diff = b[:-1, :, 3] - b[1:, :, 0]
        if cyclic:
            diff = diff % period

        if diff.any():
            return False

        diff = b[:-1, :, 2] - b[1:, :, 1]
        if cyclic:
            diff = diff % period

        if diff.any():
            return False

    return True


def get_bounds(method, coords):
    """Get coordinate bounds needed for defining an `ESMF.Grid`.

    .. versionadded:: 3.14.0

    :Parameters:

        method: `str`
            The regridding method.

        coords: sequence of `Coordinate`
            The coordinates that define an `ESMF.Grid`.

    :Returns:

        `list`
            The coordinate bounds. Will be an empty list if the
            regridding method is not conservative.

    """
    if not conservative_regridding(method):
        bounds = []
    else:
        bounds = [c.get_bounds(None) for c in coords]
        for c, b in zip(coords, bounds):
            if b is None:
                raise ValueError(
                    f"All coordinates must have bounds for {method!r} "
                    f"regridding: {c!r}"
                )

    return bounds


def conservative_regridding(method):
    """Whether or not a regridding method is conservative.

    .. versionadded:: 3.14.0

    :Parameters:

        method: `str`
            A regridding method.

    :Returns:

        `bool`
            True if the method is first or second order
            conservative. Otherwise False.

    """
    return method in ("conservative", "conservative_1st", "conservative_2nd")


def get_mask(f, grid):
    """Get the mask of the grid.

    The mask dimensions will ordered as expected by the regridding
    operator.

    :Parameters:

        f: `Field`
            The field providing the mask.

        grid: `Grid`

    :Returns:

        `dask.array.Array`
            The Boolean mask.

    """
    regrid_axes = grid.axis_indices

    index = [slice(None) if i in regrid_axes else 0 for i in range(f.ndim)]

    mask = da.ma.getmaskarray(f.data.to_dask_array())
    mask = mask[tuple(index)]

    # Reorder the mask axes to grid.axes_keys
    mask = da.transpose(mask, axes=np.argsort(regrid_axes).tolist())

    return mask


def update_coordinates(src, dst, src_grid, dst_grid):
    """Update the regrid axis coordinates.

    Replace the existing coordinate constructs that span the regridding
    axes with those from the destination grid.

    :Parameters:

        src: `Field`
            The regridded source field. Updated in-place.

        dst: `Field` or `Domain`
            The field or domain containing the destination grid.

        src_grid: `Grid`
            The definition of the source grid.

        dst_grid: `Grid`
            The definition of the destination grid.

    :Returns:

        `None`

    """
    src_axis_keys = src_grid.axis_keys
    dst_axis_keys = dst_grid.axis_keys

    # Remove the source coordinates of new field
    for key in src.coordinates(
        filter_by_axis=src_axis_keys, axis_mode="or", todict=True
    ):
        src.del_construct(key)

    # Domain axes
    src_domain_axes = src.domain_axes(todict=True)
    dst_domain_axes = dst.domain_axes(todict=True)
    for src_axis, dst_axis in zip(src_axis_keys, dst_axis_keys):
        src_domain_axis = src_domain_axes[src_axis]
        dst_domain_axis = dst_domain_axes[dst_axis]

        src_domain_axis.set_size(dst_domain_axis.size)

        ncdim = dst_domain_axis.nc_get_dimension(None)
        if ncdim is not None:
            src_domain_axis.nc_set_dimension(ncdim)

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
        src.set_construct(aux, axes=axes)


def update_non_coordinates(
    src, dst, regrid_operator, src_grid=None, dst_grid=None
):
    """Update the coordinate references of the regridded field.

    :Parameters:

        src: `Field`
            The regridded field. Updated in-place.

        dst: `Field` or `Domain`
            The field or domain containing the destination grid.

        regrid_operator: `RegridOperator`
            The regrid operator.

        src_grid: `Grid`
            The definition of the source grid.

        dst_grid: `Grid`
            The definition of the destination grid.

    :Returns:

        `None`

    """
    src_axis_keys = src_grid.axis_keys
    dst_axis_keys = dst_grid.axis_keys

    # Initialise cached value for domain_axes
    domain_axes = None

    data_axes = src.constructs.data_axes()

    # ----------------------------------------------------------------
    # Delete source grid coordinate references whose `coordinates`
    # span any of the regridding axes. Also delete the corresponding
    # domain ancillaries.
    # ----------------------------------------------------------------
    for ref_key, ref in src.coordinate_references(todict=True).items():
        ref_axes = []
        for c_key in ref.coordinates():
            ref_axes.extend(data_axes[c_key])

        if set(ref_axes).intersection(src_axis_keys):
            src.del_coordinate_reference(ref_key)

    # ----------------------------------------------------------------
    # Delete source grid cell measures and field ancillaries that span
    # any of the regridding axes
    # ----------------------------------------------------------------
    for key in src.constructs(
        filter_by_type=("cell_measure", "field_ancillary"), todict=True
    ):
        if set(data_axes[key]).intersection(src_axis_keys):
            src.del_construct(key)

    # ----------------------------------------------------------------
    # Regrid any remaining source domain ancillaries that span all of
    # the regridding axes
    # ----------------------------------------------------------------
    for da_key in src.domain_ancillaries(todict=True):
        da_axes = data_axes[da_key]

        # Ignore any remaining source domain ancillary that spans none
        # of the regridding axes
        if not set(da_axes).intersection(src_axis_keys):
            continue

        # Delete any any remaining source domain ancillary that spans
        # some but not all of the regridding axes
        if not set(da_axes).issuperset(src_axis_keys):
            src.del_construct(da_key)
            continue

        # Convert the domain ancillary to a field, without any
        # non-coordinate metadata (to prevent them being unnecessarily
        # processed during the regridding of the domain ancillary
        # field, and especially to avoid potential
        # regridding-of-domain-ancillaries infinite recursion).
        da_field = src.convert(da_key)

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
        regrid(
            src_grid.coord_sys,
            da_field,
            regrid_operator,
            inplace=True,
        )

        # Set sizes of regridded axes
        domain_axes = src.domain_axes(cached=domain_axes, todict=True)
        for axis, new_size in zip(src_axis_keys, dst_grid.shape):
            domain_axes[axis].set_size(new_size)

        # Put the regridded domain ancillary back into the field
        src.set_construct(
            src._DomainAncillary(source=da_field),
            key=da_key,
            axes=da_axes,
            copy=False,
        )

    # ----------------------------------------------------------------
    # Copy selected coordinate references from the destination grid
    # ----------------------------------------------------------------
    dst_data_axes = dst.constructs.data_axes()

    for ref in dst.coordinate_references(todict=True).values():
        axes = set()
        for c_key in ref.coordinates():
            axes.update(dst_data_axes[c_key])

        if axes and set(axes).issubset(dst_axis_keys):
            src.set_coordinate_reference(ref, parent=dst, strict=True)
