"""Worker functions for regridding."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import dask.array as da
import numpy as np
from cfdm import is_log_level_debug

from ..functions import DeprecationError, regrid_logging
from ..units import Units
from .regridoperator import RegridOperator

# ESMF renamed its Python module to `esmpy` at ESMF version 8.4.0. Allow
# either for now for backwards compatibility.
esmpy_imported = False
try:
    import esmpy

    esmpy_imported = True
except ImportError:
    try:
        # Take the new name to use in preference to the old one.
        import ESMF as esmpy

        esmpy_imported = True
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Mapping of regrid method strings to esmpy method codes. The values
# get replaced with `esmpy.RegridMethod` constants the first time
# `esmpy_initialise` is run.
esmpy_methods = {
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

    # Identify the grid as 'source' or 'destination'.
    name: str = ""
    # The coordinate system of the grid.
    coord_sys: str = ""
    # The type of the grid. E.g. 'structured grid'
    type: str = ""
    # The domain axis identifiers of the regrid axes, in the order
    # expected by `Data._regrid`. E.g. ['domainaxis3', 'domainaxis2']
    axis_keys: list = field(default_factory=list)
    # The positions of the regrid axes, in the order expected by
    # `Data._regrid`. E.g. [3, 2] or [3]
    axis_indices: list = field(default_factory=list)
    # The domain axis identifiers of the regridding axes. E.g. {'X':
    # 'domainaxis2', 'Y': 'domainaxis1'} or ['domainaxis1',
    # 'domainaxis2'], or {'X': 'domainaxis2', 'Y': 'domainaxis2'}
    axes: Any = None
    # The number of regrid axes.
    n_regrid_axes: int = 0
    # The dimensionality of the regridding on this grid. Generally the
    # same as *n_regrid_axes*, but for a regridding a UGRID mesh axis
    # *n_regrid_axes* is 1 and *dimensionality* is 2.
    dimensionality: int = 0
    # The shape of the regridding axes, in the same order as the
    # 'axis_keys' attribute. E.g. (96, 73) or (1243,)
    shape: tuple = None
    # The regrid axis coordinates, in the order expected by
    # `esmpy`. If the coordinates are 2-d (or more) then the axis
    # order of each coordinate object must be as expected by `esmpy`.
    coords: list = field(default_factory=list)
    # The regrid axis coordinate bounds, in the order expected by
    # `esmpy`. If the coordinates are 2-d (or more) then the axis
    # order of each bounds object must be as expected by `esmpy`.
    bounds: list = field(default_factory=list)
    # Only used if `mesh` is False. For spherical regridding, whether
    # or not the longitude axis is cyclic.
    cyclic: Any = None
    # The regridding method.
    method: str = ""
    # If True then, for 1-d regridding, the esmpy weights are generated
    # for a 2-d grid for which one of the dimensions is a size 2 dummy
    # dimension.
    dummy_size_2_dimension: bool = False
    # Whether or not the grid is a structured grid.
    is_grid: bool = False
    # Whether or not the grid is a UGRID mesh.
    is_mesh: bool = False
    # Whether or not the grid is a location stream.
    is_locstream: bool = False
    # The type of grid.
    type: str = "unknown"
    # The location on a UGRID mesh topology of the grid cells. An
    # empty string means that the grid is not a UGRID mesh
    # topology. E.g. '' or 'face'.
    mesh_location: str = ""
    # A domain topology construct that spans the regrid axis. A value
    # of None means that the grid is not a UGRID mesh topology.
    domain_topology: Any = None
    # The featureType of a discrete sampling geometry grid. An empty
    # string means that the grid is not a DSG. E.g. '' or
    # 'trajectory'.
    featureType: str = ""
    # The domain axis identifiers of new axes that result from the
    # regridding operation changing the number of data dimensions
    # (e.g. by regridding a source UGRID (1-d) grid to a destination
    # non-UGRID (2-d) grid, or vice versa). An empty list means that
    # the regridding did not change the number of data axes. E.g. [],
    # ['domainaxis3', 'domainaxis4'], ['domainaxis4']
    new_axis_keys: list = field(default_factory=list)
    # Specify vertical regridding coordinates. E.g. 'air_pressure',
    # 'domainaxis0'
    z: Any = None
    # Whether or not to use ln(z) when calculating vertical weights
    ln_z: bool = False
    # The integer position in *coords* of a vertical coordinate. If
    # `None` then there are no vertical coordinates.
    z_index: Any = None


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
    src_z=None,
    dst_z=None,
    z=None,
    ln_z=None,
    ignore_degenerate=True,
    return_operator=False,
    check_coordinates=False,
    min_weight=None,
    weights_file=None,
    return_esmpy_regrid_operator=False,
    inplace=False,
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

        return_esmpy_regrid_operator: `bool`, optional
            If True then *src* is not regridded, but the
            `esmpy.Regrid` instance for the operation is returned
            instead. This is useful for checking that the field has
            been regridded correctly.

        weights_file: `str` or `None`, optional
            Provide a netCDF file that contains, or will contain, the
            regridding weights.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

            .. versionadded:: 3.15.2

        src_z: optional
            The identity of the source grid vertical coordinates used
            to calculate the weights. If `None` then no vertical axis
            is identified, and in the spherical case regridding will
            be 2-d.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

            .. versionadded:: 3.16.2

        dst_z: optional
            The identity of the destination grid vertical coordinates
            used to calculate the weights. If `None` then no vertical
            axis is identified, and in the spherical case regridding
            will be 2-d.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

            .. versionadded:: 3.16.2

        z: optional
            The *z* parameter is a convenience that may be used to
            replace both *src_z* and *dst_z* when they would contain
            identical values.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

            .. versionadded:: 3.16.2

        ln_z: `bool` or `None`, optional
            Whether or not the weights are to be calculated with the
            natural logarithm of vertical coordinates.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

            .. versionadded:: 3.16.2

    :Returns:

        `Field` or `None` or `RegridOperator` or `esmpy.Regrid`
            The regridded field construct; or `None` if the operation
            was in-place; or the regridding operator if
            *return_operator* is True.

            If *return_esmpy_regrid_operator* is True then *src* is
            not regridded, but the `esmpy.Regrid` instance for the
            operation is returned instead.

    """
    if not inplace:
        src = src.copy()

    spherical = coord_sys == "spherical"
    cartesian = not spherical

    # ----------------------------------------------------------------
    # Parse and check parameters
    # ----------------------------------------------------------------
    if isinstance(dst, RegridOperator):
        regrid_operator = dst
        dst = regrid_operator.dst.copy()
        method = regrid_operator.method
        dst_cyclic = regrid_operator.dst_cyclic
        dst_axes = regrid_operator.dst_axes
        dst_z = regrid_operator.dst_z
        src_z = regrid_operator.src_z
        ln_z = regrid_operator.ln_z
        if regrid_operator.coord_sys == "Cartesian":
            src_axes = regrid_operator.src_axes

        create_regrid_operator = False
    else:
        create_regrid_operator = True

        # Parse the z, src_z, dst_z, and ln_z parameters
        if z is not None:
            if dst_z is None:
                dst_z = z
            else:
                raise ValueError("Can't set both 'z' and 'dst_z'")

            if src_z is None:
                src_z = z
            else:
                raise ValueError("Can't set both 'z' and 'src_z'")

        elif (src_z is None) != (dst_z is None):
            raise ValueError(
                "Must set both 'src_z' and 'dst_z', or neither of them"
            )

        if ln_z is None and src_z is not None:
            raise ValueError(
                "When 'z', 'src_z', or 'dst_z' have been set, "
                "'ln_z' cannot be None."
            )

        ln_z = bool(ln_z)

    if method not in esmpy_methods:
        raise ValueError(
            "Can't regrid: Must set a valid regridding method from "
            f"{sorted(esmpy_methods)}. Got: {method!r}"
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

    if cartesian:
        if isinstance(axes, str):
            axes = (axes,)

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
                f"RegridOperator, or a sequence of Coordinate. Got: {dst!r}"
            )
        except IndexError:
            # This error will get trapped in one of the following
            # *_coords_to_domain calls
            pass

        # Convert a sequence of Coordinates that define the
        # destination grid into a Domain object
        use_dst_mask = False
        if spherical:
            dst, dst_axes, dst_z = spherical_coords_to_domain(
                dst,
                dst_axes=dst_axes,
                cyclic=dst_cyclic,
                dst_z=dst_z,
                domain_class=src._Domain,
            )
        else:
            # Cartesian
            dst, dst_axes, dst_z = Cartesian_coords_to_domain(
                dst, dst_z=dst_z, domain_class=src._Domain
            )

    # ----------------------------------------------------------------
    # Create descriptions of the source and destination grids
    # ----------------------------------------------------------------
    dst_grid = get_grid(
        coord_sys,
        dst,
        "destination",
        method,
        dst_cyclic,
        axes=dst_axes,
        z=dst_z,
        ln_z=ln_z,
    )
    src_grid = get_grid(
        coord_sys,
        src,
        "source",
        method,
        src_cyclic,
        axes=src_axes,
        z=src_z,
        ln_z=ln_z,
    )

    if is_log_level_debug(logger):
        logger.debug(
            f"Source Grid:\n{src_grid}\n\nDestination Grid:\n{dst_grid}\n"
        )  # pragma: no cover

    conform_coordinates(src_grid, dst_grid)

    if method in ("conservative_2nd", "patch"):
        if not (src_grid.dimensionality >= 2 and dst_grid.dimensionality >= 2):
            raise ValueError(
                f"{method!r} regridding is not available for 1-d regridding"
            )
    elif method in ("nearest_dtos", "nearest_stod"):
        if not has_coordinate_arrays(src_grid) and not has_coordinate_arrays(
            dst_grid
        ):
            raise ValueError(
                f"{method!r} regridding is only available when both the "
                "source and destination grids have coordinate arrays"
            )

        if method == "nearest_dtos":
            if src_grid.is_mesh is not dst_grid.is_mesh:
                raise ValueError(
                    f"{method!r} regridding is (at the moment) only available "
                    "when neither or both the source and destination grids "
                    "are a UGRID mesh"
                )

            if src_grid.is_locstream or dst_grid.is_locstream:
                raise ValueError(
                    f"{method!r} regridding is (at the moment) only available "
                    "when neither the source and destination grids are "
                    "DSG featureTypes."
                )

    elif cartesian and (src_grid.is_mesh or dst_grid.is_mesh):
        raise ValueError(
            "Cartesian regridding is (at the moment) not available when "
            "either the source or destination grid is a UGRID mesh"
        )

    elif cartesian and (src_grid.is_locstream or dst_grid.is_locstream):
        raise ValueError(
            "Cartesian regridding is (at the moment) not available when "
            "either the source or destination grid is a DSG featureType"
        )

    if create_regrid_operator:
        # ------------------------------------------------------------
        # Create a new regrid operator
        # ------------------------------------------------------------
        esmpy_manager = esmpy_initialise()  # noqa: F841

        # Create a mask for the destination grid
        dst_mask = None
        grid_dst_mask = None
        if use_dst_mask:
            dst_mask = get_mask(dst, dst_grid)
            if (
                method in ("patch", "conservative_2nd", "nearest_stod")
                or return_esmpy_regrid_operator
            ):
                # For these regridding methods, the destination mask
                # must be taken into account during the esmpy
                # calculation of the regrid weights, rather than the
                # mask being applied retrospectively to weights that
                # have been calculated assuming no destination grid
                # mask. See `cf.data.dask_regrid`.
                grid_dst_mask = np.array(dst_mask.transpose())
                dst_mask = None

        # Create the destination esmpy.Grid
        dst_esmpy_grid = create_esmpy_grid(dst_grid, grid_dst_mask)
        del grid_dst_mask

        # Create a mask for the source grid
        src_mask = None
        grid_src_mask = None
        if use_src_mask and (
            method in ("patch", "conservative_2nd", "nearest_stod")
            or return_esmpy_regrid_operator
        ):
            # For patch recovery and second-order conservative
            # regridding, the source mask needs to be taken into
            # account during the esmpy calculation of the regrid
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

        # Create the source esmpy.Grid
        src_esmpy_grid = create_esmpy_grid(src_grid, grid_src_mask)
        del grid_src_mask

        if is_log_level_debug(logger):
            logger.debug(
                f"Source ESMF Grid:\n{src_esmpy_grid}\n\nDestination ESMF Grid:\n{dst_esmpy_grid}\n"
            )  # pragma: no cover

        esmpy_regrid_operator = [] if return_esmpy_regrid_operator else None

        # Create regrid weights
        weights, row, col, start_index, from_file = create_esmpy_weights(
            method,
            src_esmpy_grid,
            dst_esmpy_grid,
            src_grid=src_grid,
            dst_grid=dst_grid,
            ignore_degenerate=ignore_degenerate,
            quarter=src_grid.dummy_size_2_dimension,
            esmpy_regrid_operator=esmpy_regrid_operator,
            weights_file=weights_file,
        )

        if return_esmpy_regrid_operator:
            # Return the equivalent esmpy.Regrid operator
            return esmpy_regrid_operator[-1]

        # Still here? Then we've finished with esmpy, so finalise the
        # esmpy manager. This is done to free up any Persistent
        # Execution Threads (PETs) created by the esmpy Virtual
        # Machine:
        # https://earthsystemmodeling.org/esmpy_doc/release/latest/html/api.html#resource-allocation
        del esmpy_manager

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
            dimensionality=src_grid.dimensionality,
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
            weights_file=weights_file if from_file else None,
            src_mesh_location=src_grid.mesh_location,
            dst_featureType=dst_grid.featureType,
            src_z=src_grid.z,
            dst_z=dst_grid.z,
            ln_z=ln_z,
        )
    else:
        if weights_file is not None:
            raise ValueError(
                "Can't provide a weights file when 'dst' is a RegridOperator"
            )

        # Check that the given regrid operator is compatible with the
        # source field's grid
        check_operator(
            src, src_grid, regrid_operator, check_coordinates=check_coordinates
        )

    if return_operator:
        regrid_operator.tosparse()
        return regrid_operator

    # ----------------------------------------------------------------
    # Still here? Then do the regridding
    # ----------------------------------------------------------------
    if src_grid.n_regrid_axes == dst_grid.n_regrid_axes:
        regridded_axis_sizes = {
            src_iaxis: (dst_size,)
            for src_iaxis, dst_size in zip(
                src_grid.axis_indices, dst_grid.shape
            )
        }
    elif src_grid.n_regrid_axes == 1:
        # Fewer source grid axes than destination grid axes (e.g. mesh
        # regridded to lat/lon).
        regridded_axis_sizes = {src_grid.axis_indices[0]: dst_grid.shape}
    elif dst_grid.n_regrid_axes == 1:
        # More source grid axes than destination grid axes
        # (e.g. lat/lon regridded to mesh).
        src_axis_indices = sorted(src_grid.axis_indices)
        regridded_axis_sizes = {src_axis_indices[0]: (dst_grid.shape[0],)}
        for src_iaxis in src_axis_indices[1:]:
            regridded_axis_sizes[src_iaxis] = ()

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
    update_non_coordinates(src, dst, src_grid, dst_grid, regrid_operator)

    update_coordinates(src, dst, src_grid, dst_grid)

    # ----------------------------------------------------------------
    # Insert regridded data into the new field
    # ----------------------------------------------------------------
    update_data(src, regridded_data, src_grid)

    if coord_sys == "spherical" and dst_grid.is_grid:
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
    dst, dst_axes=None, cyclic=None, dst_z=None, domain_class=None
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

        dst_z: optional
            If `None`, the default, then it assumed that none of the
            coordinate consructs in *dst* are vertical coordinates.

            Otherwise identify the destination grid vertical
            coordinate construct as the unique construct returned by
            ``d.coordinate(dst_z)``, where ``d`` is the `Domain`
            returned by this function.

            .. versionadded:: 3.16.2

        domain_class: `Domain` class
            The domain class used to create the new `Domain` instance.

    :Returns:

        3-`tuple`
            * The new domain containing the grid
            * A dictionary identifying the domain axis identifiers of
              the regrid axes (as defined by the *dst_axes* parameter
              of `cf.Field.regrids`)
            * The value of *dst_z*. Either `None`, or replaced with
              its construct identifier in the output `Domain`.

    """
    if dst_z is None:
        if len(dst) != 2:
            raise ValueError(
                "Expected a sequence of latitude and longitude "
                f"coordinate constructs. Got: {dst!r}"
            )
    elif len(dst) != 3:
        raise ValueError(
            "Expected a sequence of latitude, longitude, and vertical "
            f"coordinate constructs. Got: {dst!r}"
        )

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
            elif dst_z is not None:
                coords["Z"] = c
        except AttributeError:
            pass

    if "lat" not in coords or "lon" not in coords:
        raise ValueError(
            "Expected a sequence that includes latitude and longitude "
            f"coordinate constructs. Got: {dst!r}"
        )

    if dst_z is not None and "Z" not in coords:
        raise ValueError(
            "Expected a sequence that includes vertical "
            f"coordinate constructs. Got: {dst!r}"
        )

    coords_1d = False

    if coords["lat"].ndim == 1:
        coords_1d = True
        axis_sizes = [coords["lat"].size, coords["lon"].size]
        if coords["lon"].ndim != 1:
            raise ValueError(
                "When 'dst' is a sequence containing latitude and "
                "longitude coordinate constructs, they must have the "
                f"same shape. Got: {dst!r}"
            )
    elif coords["lat"].ndim == 2:
        message = (
            "When 'dst' is a sequence containing 2-d latitude and longitude "
            "coordinate constructs, 'dst_axes' must be dictionary with at "
            "least the keys {'X': 0, 'Y': 1} or {'X': 1, 'Y': 0}. "
            f"Got: {dst_axes!r}"
        )

        if dst_axes is None:
            raise ValueError(message)

        axis_sizes = coords["lat"].shape
        if dst_axes.get("X") == 0 and dst_axes.get("Y") == 1:
            axis_sizes = axis_sizes[::-1]
        elif not (dst_axes.get("X") == 1 and dst_axes.get("Y") == 0):
            raise ValueError(message)

        if coords["lat"].shape != coords["lon"].shape:
            raise ValueError(
                "When 'dst' is a sequence of latitude and longitude "
                f"coordinates, they must have the same shape. Got: {dst!r}"
            )
    elif coords["lat"].ndim > 2:
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

    coord_axes = []
    if coords_1d:
        # Set 1-d coordinates
        coord_axes = []
        for key, axis in zip(("lat", "lon"), axis_keys):
            da_key = d.set_construct(coords[key], axes=axis, copy=False)
            coord_axes.append(da_key)

        if dst_axes and "X" in dst_axes and dst_axes["X"] == 0:
            coord_axes = coord_axes[::-1]
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

    if dst_z is not None:
        # ------------------------------------------------------------
        # Deal with Z coordinates
        # ------------------------------------------------------------
        z_coord = coords["Z"]
        if z_coord.ndim == 1:
            z_axis = d.set_construct(d._DomainAxis(z_coord.size), copy=False)
            d.set_construct(z_coord, axes=z_axis, copy=False)
        elif z_coord.ndim == 3:
            if dst_axes is None or "Z" not in dst_axes or dst_axes["Z"] != 2:
                raise ValueError(
                    "When 'dst' is a sequence containing a 3-d vertical "
                    "coordinate construct, 'dst_axes' must be either "
                    "{'X': 0, 'Y': 1, 'Z': 2} or {'X': 1, 'Y': 0, 'Z': 2}. "
                    f"Got: {dst_axes!r}"
                )

            z_axis = d.set_construct(
                d._DomainAxis(z_coord.shape[2]), copy=False
            )
            z_key = d.set_construct(
                z_coord, axes=coord_axes + (z_axis,), copy=False
            )

        # Check that z_coord is indeed a vertical coordinate
        # construct, and replace 'dst_z' with its construct
        # identifier.
        key = d.coordinate(dst_z, key=True, default=None)
        if key != z_key:
            raise ValueError(
                f"Could not find destination {dst_z!r} vertical coordinates"
            )

        dst_z = key
        dst_axes["Z"] = z_axis

    return d, dst_axes, dst_z


def Cartesian_coords_to_domain(dst, dst_z=None, domain_class=None):
    """Convert a sequence of `Coordinate` to Cartesian grid definition.

    .. versionadded:: 3.14.0

    :Parameters:

        dst: sequence of `DimensionCoordinate`
            Between one and three 1-d dimension coordinate constructs
            that define the coordinates of the grid. The order of the
            coordinate constructs **must** match the order of source
            field regridding axes defined elsewhere by the *src_axes*
            or *axes* parameter.

        dst_z: optional
            If `None`, the default, then it assumed that none of the
            coordinate consructs in *dst* are vertical coordinates.

            Otherwise identify the destination grid vertical
            coordinate construct as the unique construct returned by
            ``d.coordinate(dst_z)``, where ``d`` is the `Domain`
            returned by this function.

            .. versionadded:: 3.16.2

        domain_class: `Domain` class
            The domain class used to create the new `Domain` instance.

    :Returns:

        3-`tuple`
            * The new domain containing the grid
            * A list identifying the domain axis identifiers of the
              regrid axes (as defined by the *dst_axes* parameter of
              `cf.Field.regridc`)
            * The value of *dst_z*. Either `None`, or replaced with
              its construct identifier in the output `Domain`.

    """
    d = domain_class()

    axis_keys = []
    for coord in dst:
        axis = d.set_construct(d._DomainAxis(coord.size), copy=False)
        d.set_construct(coord, axes=axis, copy=True)
        axis_keys.append(axis)

    if dst_z is not None:
        # Check that there are vertical coordinates, and replace
        # 'dst_z' with the identifier of its domain axis construct.
        z_key = d.coordinate(dst_z, key=True, default=None)
        if z_key is None:
            raise ValueError(
                f"Could not find destination {dst_z!r} vertical coordinates"
            )

        dst_z = d.get_data_axes(z_key)[0]

    return d, axis_keys, dst_z


def get_grid(
    coord_sys,
    f,
    name=None,
    method=None,
    cyclic=None,
    axes=None,
    z=None,
    ln_z=None,
):
    """Get axis and coordinate information for regridding.

    See `spherical_grid` and `Cartesian_grid` for details.

    .. versionadded:: 3.14.0

        coord_sys: `str`
            The coordinate system of the source and destination grids.

        ln_z: `bool` or `None`, optional
            Whether or not the weights are to be calculated with the
            natural logarithm of vertical coordinates.

            See `cf.Field.regrids` (for spherical regridding) or
            `cf.Field.regridc` (for Cartesian regridding) for details.

            .. versionadded:: 3.16.2

    """
    if coord_sys == "spherical":
        return spherical_grid(
            f,
            name=name,
            method=method,
            cyclic=cyclic,
            axes=axes,
            z=z,
            ln_z=ln_z,
        )

    # Cartesian
    return Cartesian_grid(
        f, name=name, method=method, axes=axes, z=z, ln_z=ln_z
    )


def spherical_grid(
    f,
    name=None,
    method=None,
    cyclic=None,
    axes=None,
    z=None,
    ln_z=None,
):
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

        z: optional
            If `None`, the default, then the regridding is 2-d in
            the latitude-longitude plane.

            If not `None` then 3-d spherical regridding is enabled by
            identifying the grid vertical coordinates from which to
            derive the vertical component of the regridding
            weights. The vertical coordinate construct may be 1-d or
            3-d and is defined by the unique construct returned by
            ``f.coordinate(src_z)``

            .. versionadded:: 3.16.2

        ln_z: `bool` or `None`, optional
            Whether or not the weights are to be calculated with the
            natural logarithm of vertical coordinates.

            .. versionadded:: 3.16.2

    :Returns:

        `Grid`
            The grid definition.

    """
    data_axes = f.constructs.data_axes()

    dim_coords_1d = False
    aux_coords_2d = False
    aux_coords_1d = False
    domain_topology, mesh_location, axis1 = get_mesh(f)

    if not mesh_location:
        featureType, axis1 = get_dsg(f)
    else:
        featureType = None

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
            dim_coords_1d = True
            lon = lon_1d
            lat = lat_1d

    if not dim_coords_1d:
        domain_topology, mesh_location, mesh_axis = get_mesh(f)

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
            aux_coords_2d = True

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

        elif mesh_location is not None or featureType is not None:
            if mesh_location and mesh_location not in ("face", "point"):
                raise ValueError(
                    f"Can't regrid {'from' if name == 'source' else 'to'} "
                    f"a {name} unstructured mesh of "
                    f"{mesh_location!r} cells"
                )

            if featureType and conservative_regridding(method):
                raise ValueError(
                    f"Can't do {method} regridding "
                    f"{'from' if name == 'source' else 'to'} "
                    f"a {name} DSG featureType"
                )

            lon = f.auxiliary_coordinate(
                "X",
                filter_by_axis=(axis1,),
                axis_mode="exact",
                default=None,
            )
            lat = f.auxiliary_coordinate(
                "Y",
                filter_by_axis=(axis1,),
                axis_mode="exact",
                default=None,
            )
            if (
                lon is not None
                and lat is not None
                and lon.Units.islongitude
                and lat.Units.islatitude
            ):
                # Found 1-d latitude and longitude auxiliary
                # coordinates for a UGRID mesh topology
                aux_coords_1d = True
                x_axis = axis1
                y_axis = axis1

    if not (dim_coords_1d or aux_coords_2d or aux_coords_1d):
        raise ValueError(
            "Could not find 1-d nor 2-d latitude and longitude coordinates"
        )

    if not (mesh_location or featureType) and x_axis == y_axis:
        raise ValueError(
            "The X and Y axes must be distinct, but they are "
            f"the same for {name} field {f!r}."
        )

    # Get X and Y axis sizes
    domain_axes = f.domain_axes(todict=True)
    x_size = domain_axes[x_axis].size
    y_size = domain_axes[y_axis].size

    # Source grid size 1 dimensions are problematic for esmpy for some
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

    coords = [lon, lat]  # esmpy order

    # Convert 2-d coordinate arrays to esmpy axis order = [X, Y]
    if aux_coords_2d:
        for dim, coord_key in enumerate((lon_key, lat_key)):
            coord_axes = data_axes[coord_key]
            esmpy_order = [coord_axes.index(axis) for axis in (x_axis, y_axis)]
            coords[dim] = coords[dim].transpose(esmpy_order)

    # Set cyclicity of X axis
    if mesh_location or featureType:
        cyclic = None
    elif cyclic is None:
        cyclic = f.iscyclic(x_axis)
    else:
        cyclic = bool(cyclic)

    axes = {"X": x_axis, "Y": y_axis}
    axis_keys = [y_axis, x_axis]
    shape = (y_size, x_size)

    if mesh_location or featureType:
        axis_keys = axis_keys[0:1]
        shape = shape[0:1]

    n_regrid_axes = len(axis_keys)
    regridding_dimensionality = n_regrid_axes
    if mesh_location or featureType:
        regridding_dimensionality += 1

    if z is not None:
        # ------------------------------------------------------------
        # 3-d spherical regridding
        # ------------------------------------------------------------
        if conservative_regridding(method):
            raise ValueError(f"Can't do {method} 3-d spherical regridding")

        if mesh_location:
            raise ValueError(
                "Can't do 3-d spherical regridding "
                f"{'from' if name == 'source' else 'to'} a {name} "
                f"unstructured mesh of {mesh_location!r} cells"
            )
        elif featureType:
            # Look for 1-d Z auxiliary coordinates
            z_key, z_1d = f.auxiliary_coordinate(
                z,
                filter_by_axis=(axis1,),
                axis_mode="exact",
                item=True,
                default=(None, None),
            )
            if z_1d is None:
                raise ValueError(
                    f"Could not find {name} DSG {featureType} 1-d "
                    f"{z!r} coordinates"
                )

            z_coord = z_1d
            z_axis = axis1
        else:
            # Look for 1-d Z dimension coordinates
            z_key, z_1d = f.dimension_coordinate(
                z, item=True, default=(None, None)
            )
            if z_1d is not None:
                z_axis = data_axes[z_key][0]
                z_coord = z_1d
            else:
                # Look for 3-d Z auxiliary coordinates
                z_key, z_3d = f.auxiliary_coordinate(
                    z,
                    filter_by_naxes=(3,),
                    axis_mode="exact",
                    item=True,
                    default=(None, None),
                )
                if z_3d is None:
                    raise ValueError(
                        f"Could not find {name} structured grid 1-d or 3-d "
                        f"{z!r} coordinates"
                    )

                coord_axes = data_axes[z_key]
                if x_axis not in coord_axes or y_axis not in coord_axes:
                    raise ValueError(
                        f"The {name} structured grid 3-d {z!r} "
                        "coordinates do not span the latitude and longitude "
                        "domain axes"
                    )

                z_axis = [
                    axis for axis in coord_axes if axis not in (x_axis, y_axis)
                ][0]

                # Re-order 3-d Z coordinates to ESMF order
                esmpy_order = [
                    coord_axes.index(axis) for axis in (x_axis, y_axis, z_axis)
                ]
                z_coord = z_3d.transpose(esmpy_order)

        coords.append(z_coord)  # esmpy order

        if not (mesh_location or featureType):
            axes["Z"] = z_axis
            axis_keys.insert(0, z_axis)
            z_size = domain_axes[z_axis].size
            shape = (z_size,) + shape

        regridding_dimensionality += 1
        z_index = 2
    else:
        z_index = None

    if f.construct_type == "domain":
        axis_indices = list(range(n_regrid_axes))
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

    is_mesh = bool(mesh_location)
    is_locstream = bool(featureType)
    is_grid = not is_mesh and not is_locstream

    grid = Grid(
        name=name,
        coord_sys="spherical",
        method=method,
        axis_keys=axis_keys,
        axis_indices=axis_indices,
        axes=axes,
        n_regrid_axes=n_regrid_axes,
        dimensionality=regridding_dimensionality,
        shape=shape,
        coords=coords,
        bounds=get_bounds(method, coords, mesh_location),
        cyclic=cyclic,
        is_grid=is_grid,
        is_mesh=is_mesh,
        is_locstream=is_locstream,
        mesh_location=mesh_location,
        domain_topology=domain_topology,
        featureType=featureType,
        z=z,
        ln_z=ln_z,
        z_index=z_index,
    )

    set_grid_type(grid)
    return grid


def Cartesian_grid(f, name=None, method=None, axes=None, z=None, ln_z=None):
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

        z: optional
            If not `None` then *src_z* specifies the identity of a
            vertical coordinate construct of the source grid.

            .. versionadded:: 3.16.2

        ln_z: `bool` or `None`, optional
            Whether or not the weights are to be calculated with the
            natural logarithm of vertical coordinates.

            .. versionadded:: 3.16.2

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

    if z is not None and z not in axes:
        raise ValueError(
            f"Z coordinate {z!r} must match exactly an "
            f"element of 'axes' ({axes!r})"
        )

    # Find the axis keys, sizes and indices of the regrid axes, in the
    # order that they have been given by the 'axes' parameters.
    axis_keys = []
    axis_sizes = []
    for i, axis in enumerate(axes):
        key, domain_axis = f.domain_axis(axis, item=True, default=(None, None))
        if key is None:
            raise ValueError(
                f"No {name} grid domain axis could be found from {axis!r}"
            )

        axis_keys.append(key)
        axis_sizes.append(domain_axis.size)

    domain_topology, mesh_location, axis1 = get_mesh(f)
    if not mesh_location:
        featureType, axis1 = get_dsg(f)
    else:
        featureType, axis1 = None

    if mesh_location or featureType:
        # There is a domain topology axis
        if tuple(set(axis_keys)) == (axis1,):
            # There is a unique regridding axis, and it's the discrete
            # axis.
            if mesh_location and mesh_location not in ("face", "point"):
                raise ValueError(
                    f"Can't do Cartesian regridding "
                    f"{'from' if name == 'source' else 'to'} "
                    f"a {name} unstructured mesh of "
                    f"{mesh_location!r} cells"
                )

            if featureType and conservative_regridding(method):
                raise ValueError(
                    f"Can't do {method} Cartesian regridding "
                    f"{'from' if name == 'source' else 'to'} "
                    f"a {name} DSG featureType"
                )

            axis_keys = axis_keys[0:1]
            axis_sizes = axis_sizes[0:1]
        elif axis1 in axis_keys:
            raise ValueError(
                "Can't do Cartesian regridding for a combination of "
                f"discrete and non-discrete axes: {axis_keys}"
            )
        else:
            # None of the regridding axes have a domain topology or
            # featureType
            domain_topology = None
            featureType = None
            mesh_location = None
            axis1 = None

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

    cyclic = False
    coord_ids = axes[::-1]
    coords = []
    if mesh_location or featureType:
        raise ValueError(
            "Cartesian regridding is (at the moment) not available when "
            "either the source or destination grid is a UGRID mesh or "
            "a DSG featureType"
        )

        # This code may get used if we remove the above exception:
        for coord_id in coord_ids:
            aux = f.auxiliary_coordinate(
                coord_id,
                filter_by_axis=(axis1,),
                axis_mode="exact",
                default=None,
            )
            if aux is None:
                raise ValueError(
                    f"Could not find {coord_id!r} 1-d auxiliary coordinates"
                )

            coords.append(aux)
    else:
        for key in axis_keys[::-1]:
            dim = f.dimension_coordinate(filter_by_axis=(key,), default=None)
            if dim is None:
                raise ValueError(
                    f"No unique {name} dimension coordinate for domain axis "
                    f"{key!r}."
                )

            coords.append(dim)

    if z is not None:
        z_index = coord_ids.index(z)
    else:
        z_index = None

    bounds = get_bounds(method, coords, mesh_location)

    dummy_size_2_dimension = False
    if not (mesh_location or featureType) and len(coords) == 1:
        # Create a dummy axis because esmpy doesn't like creating
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

    n_regrid_axes = len(axis_keys)
    regridding_dimensionality = n_regrid_axes
    if mesh_location or featureType:
        regridding_dimensionality += 1

    is_mesh = bool(mesh_location)
    is_locstream = bool(featureType)
    is_grid = not is_mesh and not is_locstream

    grid = Grid(
        name=name,
        coord_sys="Cartesian",
        method=method,
        axis_keys=axis_keys,
        axis_indices=axis_indices,
        axes=axis_keys,
        n_regrid_axes=n_regrid_axes,
        dimensionality=regridding_dimensionality,
        shape=tuple(axis_sizes),
        coords=coords,
        bounds=bounds,
        cyclic=cyclic,
        dummy_size_2_dimension=dummy_size_2_dimension,
        is_mesh=is_mesh,
        is_locstream=is_locstream,
        is_grid=is_grid,
        mesh_location=mesh_location,
        domain_topology=domain_topology,
        featureType=featureType,
        z=z,
        ln_z=ln_z,
        z_index=z_index,
    )

    set_grid_type(grid)
    return grid


def conform_coordinates(src_grid, dst_grid):
    """Make the source and destination coordinates have the same units.

    Modifies *src_grid* in-place so that its coordinates and bounds
    have the same units as the coordinates and bounds of *dst_grid*.

    .. versionadded:: 3.14.0

    .. seealso:: `regrid`

    :Parameters:

        src_grid: `Grid`
            The definition of the source grid.

        dst_grid: `Grid`
            The definition of the destination grid.

    :Returns:

        `None`

    """
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

    # Take the natural logarithm of spherical vertical Z coordinates
    for grid in (src_grid, dst_grid):
        index = grid.z_index
        if grid.ln_z and index is not None:
            grid.coords[index] = grid.coords[index].log()
            if grid.bounds:
                grid.bounds[index] = grid.bounds[index].log()


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
            performed (checking that the coordinate system, cyclicity,
            grid shape, and mesh location are the same).

    :Returns:

        `bool`
            Returns `True` if the source grid coordinates and bounds
            match those of the regrid operator. Otherwise an exception
            is raised.

    """
    if regrid_operator.coord_sys != src_grid.coord_sys:
        raise ValueError(
            f"Can't regrid {src!r} with {regrid_operator!r}: "
            "Coordinate system mismatch: "
            f"{src_grid.coord_sys!r} != {regrid_operator.coord_sys!r}"
        )

    if regrid_operator.src_cyclic != src_grid.cyclic:
        raise ValueError(
            f"Can't regrid {src!r} with {regrid_operator!r}: "
            "Source grid cyclicity mismatch: "
            f"{src_grid.cyclic!r} != {regrid_operator.src_cyclic!r}"
        )

    if regrid_operator.src_shape != src_grid.shape:
        raise ValueError(
            f"Can't regrid {src!r} with {regrid_operator!r}: "
            "Source grid shape mismatch: "
            f"{src_grid.shape} != {regrid_operator.src_shape}"
        )

    if regrid_operator.src_mesh_location != src_grid.mesh_location:
        raise ValueError(
            f"Can't regrid {src!r} with {regrid_operator!r}: "
            "Source grid mesh location mismatch: "
            f"{src_grid.mesh_location} != {regrid_operator.src_mesh_location}"
        )

    if regrid_operator.src_featureType != src_grid.featureType:
        raise ValueError(
            f"Can't regrid {src!r} with {regrid_operator!r}: "
            "Source grid DSG featureType mismatch: "
            f"{src_grid.featureType} != {regrid_operator.src_featureType}"
        )

    if regrid_operator.dimensionality != src_grid.dimensionality:
        raise ValueError(
            f"Can't regrid {src!r} with {regrid_operator!r}: "
            "Source grid regridding dimensionality: "
            f"{src_grid.dimensionality} != {regrid_operator.dimensionality}"
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


def esmpy_initialise():
    """Initialise the `esmpy` manager.

    The is a null operation if the manager has already been
    initialised.

    Whether esmpy logging is enabled or not is determined by
    `cf.regrid_logging`.

    Also initialises the global 'esmpy_methods' dictionary, unless it
    has already been initialised.

    :Returns:

        `esmpy.Manager`
            The `esmpy` manager.

    """
    if not esmpy_imported:
        raise RuntimeError(
            "Regridding will not work unless the esmpy library is installed"
        )

    # Update the global 'esmpy_methods' dictionary
    if esmpy_methods["linear"] is None:
        esmpy_methods.update(
            {
                "linear": esmpy.RegridMethod.BILINEAR,  # see comment below...
                "bilinear": esmpy.RegridMethod.BILINEAR,  # (for back compat)
                "conservative": esmpy.RegridMethod.CONSERVE,
                "conservative_1st": esmpy.RegridMethod.CONSERVE,
                "conservative_2nd": esmpy.RegridMethod.CONSERVE_2ND,
                "nearest_dtos": esmpy.RegridMethod.NEAREST_DTOS,
                "nearest_stod": esmpy.RegridMethod.NEAREST_STOD,
                "patch": esmpy.RegridMethod.PATCH,
            }
        )
        reverse = {value: key for key, value in esmpy_methods.items()}
        esmpy_methods.update(reverse)

        # ... diverge from esmpy with respect to name for bilinear
        # method by using 'linear' because 'bi' implies 2D linear
        # interpolation, which could mislead or confuse for Cartesian
        # regridding in 1D or 3D.

    return esmpy.Manager(debug=bool(regrid_logging()))


def create_esmpy_grid(grid, mask=None):
    """Create an `esmpy.Grid` or `esmpy.Mesh`.

    .. versionadded:: 3.14.0

    :Parameters:

        grid: `Grid`
            The definition of the source or destination grid.

        mask: array_like, optional
            The grid mask. If `None` (the default) then there are no
            masked cells, otherwise must be a Boolean array, with True
            for masked elements, that broadcasts to the esmpy
            coordinates.

    :Returns:

        `esmpy.Grid` or `esmpy.Mesh`
            The `esmpy.Grid` or `esmpy.Mesh` derived from *grid*.

    """
    if grid.is_mesh:
        # Create an `esmpy.Mesh`
        return create_esmpy_mesh(grid, mask)

    if grid.is_locstream:
        # Create an `esmpy.LocStream`
        return create_esmpy_locstream(grid, mask)

    # Create an `esmpy.Grid`
    coords = grid.coords
    bounds = grid.bounds
    cyclic = grid.cyclic

    num_peri_dims = 0
    periodic_dim = 0
    if grid.coord_sys == "spherical":
        spherical = True
        lon, lat, z = 0, 1, 2
        coord_sys = esmpy.CoordSys.SPH_DEG
        if cyclic:
            num_peri_dims = 1
            periodic_dim = lon
    else:
        # Cartesian
        spherical = False
        coord_sys = esmpy.CoordSys.CART

    # Parse coordinates for the esmpy.Grid, and get its shape.
    n_axes = len(coords)
    coords = [np.asanyarray(c) for c in coords]
    shape = [None] * n_axes
    for dim, c in enumerate(coords[:]):
        ndim = c.ndim
        if ndim == 1:
            # 1-d
            shape[dim] = c.size
            c = c.reshape([c.size if i == dim else 1 for i in range(n_axes)])
        elif ndim == 2:
            # 2-d lat or lon
            shape[:ndim] = c.shape
            if n_axes == 3:
                c = c.reshape(c.shape + (1,))
        elif ndim == 3:
            # 3-d Z
            shape[:ndim] = c.shape
        else:
            raise ValueError(
                f"Can't create an esmpy.Grid from coordinates with {ndim} "
                f"dimensions: {c!r}"
            )

        coords[dim] = c

    # Parse bounds for the esmpy.Grid
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

        # Convert each bounds to a grid with no repeated values
        for dim, b in enumerate(bounds[:]):
            ndim = b.ndim
            if ndim == 2:
                # Bounds for 1-d coordinates.
                #
                # E.g. if the esmpy.Grid is (X, Y) then for non-cyclic
                #      bounds <CF Bounds: longitude(96, 2)
                #      degrees_east> we create a new bounds array with
                #      shape (97, 1); and for non-cyclic bounds <CF
                #      Bounds: latitude(73, 2) degrees_north> we
                #      create a new bounds array with shape (1,
                #      74). When multiplied, these arrays would create
                #      the 2-d (97, 74) bounds grid expected by
                #      esmpy.Grid.
                #
                #      Note that if the X axis were cyclic, then its
                #      new bounds array would have shape (96, 1).
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
            elif ndim == 3:
                # Bounds for 2-d coordinates
                #
                # E.g. if the esmpy.Grid is (X, Y) then for bounds <CF
                #      Bounds: latitude(96, 73, 2) degrees_north> with
                #      a non-cyclic X axis, we create a new bounds
                #      array with shape (97, 74).
                #
                #      Note that if the X axis were cyclic, then the
                #      new bounds array would have shape (96, 74).
                n, m = b.shape[:2]
                if spherical and cyclic:
                    tmp = np.empty((n, m + 1), dtype=b.dtype)
                    tmp[:, :m] = b[:, :, 0]
                    if dim == lon:
                        tmp[:, m] = b[:, -1, 0]
                    else:
                        tmp[:, m] = b[:, -1, 1]
                else:
                    tmp = np.empty((n + 1, m + 1), dtype=b.dtype)
                    tmp[:n, :m] = b[:, :, 0]
                    tmp[:n, m] = b[:, -1, 1]
                    tmp[n, :m] = b[-1, :, 3]
                    tmp[n, m] = b[-1, -1, 2]

                if n_axes == 3:
                    tmp = tmp.reshape(tmp.shape + (1,))

            elif ndim == 4:
                # Bounds for 3-d coordinates
                raise ValueError(
                    f"Can't do {grid.method} 3-d {grid.coord_sys} regridding "
                    f"with {grid.coord_sys} 3-d coordinates "
                    f"{coords[z].identity!r}."
                )

            bounds[dim] = tmp

    # Define the esmpy.Grid stagger locations. For details see
    #
    # 2-d:
    # https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node5.html#fig:gridstaggerloc2d
    #
    # 3-d:
    # https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node5.html#fig:gridstaggerloc3d
    if bounds:
        if n_axes == 3:
            staggerlocs = [
                esmpy.StaggerLoc.CENTER_VCENTER,
                esmpy.StaggerLoc.CORNER_VFACE,
            ]
        else:
            staggerlocs = [esmpy.StaggerLoc.CORNER, esmpy.StaggerLoc.CENTER]
    else:
        if n_axes == 3:
            staggerlocs = [esmpy.StaggerLoc.CENTER_VCENTER]
        else:
            staggerlocs = [esmpy.StaggerLoc.CENTER]

    # Create an empty esmpy.Grid
    esmpy_grid = esmpy.Grid(
        max_index=np.array(shape, dtype="int32"),
        coord_sys=coord_sys,
        num_peri_dims=num_peri_dims,
        periodic_dim=periodic_dim,
        staggerloc=staggerlocs,
    )

    # Populate the esmpy.Grid centres
    for dim, c in enumerate(coords):
        if n_axes == 3:
            grid_centre = esmpy_grid.get_coords(
                dim, staggerloc=esmpy.StaggerLoc.CENTER_VCENTER
            )
        else:
            grid_centre = esmpy_grid.get_coords(
                dim, staggerloc=esmpy.StaggerLoc.CENTER
            )

        grid_centre[...] = c

    # Populate the esmpy.Grid corners
    if bounds:
        if n_axes == 3:
            staggerloc = esmpy.StaggerLoc.CORNER_VFACE
        else:
            staggerloc = esmpy.StaggerLoc.CORNER

        for dim, b in enumerate(bounds):
            grid_corner = esmpy_grid.get_coords(dim, staggerloc=staggerloc)
            grid_corner[...] = b

    # Add an esmpy.Grid mask
    if mask is not None:
        if mask.dtype != bool:
            raise ValueError(
                "'mask' must be None or a Boolean array. Got: "
                f"dtype={mask.dtype}"
            )

        if not mask.any():
            mask = None

        if mask is not None:
            grid_mask = esmpy_grid.add_item(esmpy.GridItem.MASK)
            if len(grid.coords) == 2 and mask.ndim == 1:
                # esmpy grid has a dummy size 1 dimension, so we need to
                # include this in the mask as well.
                mask = np.expand_dims(mask, 1)

            # Note: 'mask' has True/False for masked/unmasked
            #       elements, but the esmpy mask requires 0/1 for
            #       masked/unmasked elements.
            grid_mask[...] = np.invert(mask).astype("int32")

    return esmpy_grid


def create_esmpy_mesh(grid, mask=None):
    """Create an `esmpy.Mesh`.

    .. versionadded:: 3.16.0

    .. seealso:: `create_esmpy_grid`

    :Parameters:

        grid: `Grid`
            The definition of the source or destination grid.

        mask: array_like, optional
            The mesh mask. If `None` (the default) then there are no
            masked cells, otherwise must be a 1-d Boolean array, with
            True for masked elements.

    :Returns:

        `esmpy.Mesh`
            The `esmpy.Mesh` derived from *grid*.

    """
    if grid.mesh_location != "face":
        raise ValueError(
            f"Can't regrid {'from' if grid.name == 'source' else 'to'} "
            f"a {grid.name} grid of UGRID {grid.mesh_location!r} cells"
        )

    if grid.coord_sys == "spherical":
        coord_sys = esmpy.CoordSys.SPH_DEG
    else:
        # Cartesian
        coord_sys = esmpy.CoordSys.CART

    # Create an empty esmpy.Mesh
    esmpy_mesh = esmpy.Mesh(
        parametric_dim=2, spatial_dim=2, coord_sys=coord_sys
    )

    element_conn = grid.domain_topology.normalise().array
    element_count = element_conn.shape[0]
    element_types = np.ma.count(element_conn, axis=1)
    element_conn = np.ma.compressed(element_conn)

    # Element coordinates
    if grid.coords:
        try:
            element_coords = [c.array for c in grid.coords]
        except AttributeError:
            # The coordinate constructs have no data
            element_coords = None
        else:
            element_coords = np.stack(element_coords, axis=-1)
    else:
        element_coords = None

    node_ids, index = np.unique(element_conn, return_index=True)
    node_coords = [b.data.compressed().array[index] for b in grid.bounds]
    node_coords = np.stack(node_coords, axis=-1)
    node_count = node_ids.size
    node_owners = np.zeros(node_count)

    # Make sure that node IDs are >= 1, as needed by newer versions of
    # esmpy.
    min_id = node_ids.min()
    if min_id < 1:
        node_ids += min_id + 1

    # Add nodes. This must be done before `add_elements`.
    esmpy_mesh.add_nodes(
        node_count=node_count,
        node_ids=node_ids,
        node_coords=node_coords,
        node_owners=node_owners,
    )

    # Mask
    if mask is not None:
        if mask.dtype != bool:
            raise ValueError(
                "'mask' must be None or a Boolean array. "
                f"Got: dtype={mask.dtype}"
            )

        # Note: 'mask' has True/False for masked/unmasked elements,
        #       but the esmpy mask requires 0/1 for masked/unmasked
        #       elements.
        mask = np.invert(mask).astype("int32")
        if mask.all():
            # There are no masked elements
            mask = None

    # Add elements. This must be done after `add_nodes`.
    #
    # Note: The element_ids should start at 1, since when writing the
    #       weights to a file, these ids are used for the column
    #       indices.
    esmpy_mesh.add_elements(
        element_count=element_count,
        element_ids=np.arange(1, element_count + 1),
        element_types=element_types,
        element_conn=element_conn,
        element_mask=mask,
        element_area=None,
        element_coords=element_coords,
    )
    return esmpy_mesh


def create_esmpy_locstream(grid, mask=None):
    """Create an `esmpy.LocStream`.

    .. versionadded:: 3.16.2

    .. seealso:: `create_esmpy_grid`, `create_esmpy_mesh`

    :Parameters:

        grid: `Grid`
            The definition of the source or destination grid.

        mask: array_like, optional
            The mesh mask. If `None` (the default) then there are no
            masked cells, otherwise must be a 1-d Boolean array, with
            True for masked elements.

    :Returns:

        `esmpy.LocStream`
            The `esmpy.LocStream` derived from *grid*.

    """
    if grid.coord_sys == "spherical":
        coord_sys = esmpy.CoordSys.SPH_DEG
        keys = ("ESMF:Lon", "ESMF:Lat", "ESMF:Radius")
    else:
        # Cartesian
        coord_sys = esmpy.CoordSys.CART
        keys = ("ESMF:X", "ESMF:Y", "ESMF:Z")

    # Create an empty esmpy.LocStream
    location_count = grid.shape[0]
    esmpy_locstream = esmpy.LocStream(
        location_count=location_count,
        coord_sys=coord_sys,
        name=grid.featureType,
    )

    # Add coordinates (must be of type float64)
    for coord, key in zip(grid.coords, keys):
        esmpy_locstream[key] = coord.array.astype(float)

    # Add mask (always required, and must be of type int32)
    if mask is not None:
        if mask.dtype != bool:
            raise ValueError(
                "'mask' must be None or a Boolean array. "
                f"Got: dtype={mask.dtype}"
            )

        # Note: 'mask' has True/False for masked/unmasked elements,
        #       but the esmpy mask requires 0/1 for masked/unmasked
        #       elements.
        mask = np.invert(mask).astype("int32")
    else:
        # No masked points
        mask = np.full((location_count,), 1, dtype="int32")

    esmpy_locstream["ESMF:Mask"] = mask

    return esmpy_locstream


def create_esmpy_weights(
    method,
    src_esmpy_grid,
    dst_esmpy_grid,
    src_grid,
    dst_grid,
    ignore_degenerate,
    quarter=False,
    esmpy_regrid_operator=None,
    weights_file=None,
):
    """Create the `esmpy` regridding weights.

    .. versionadded:: 3.14.0

    :Parameters:

        method: `str`
            The regridding method.

        src_esmpy_grid: `esmpy.Grid`
            The source grid.

        dst_esmpy_grid: `esmpy.Grid`
            The destination grid.

        src_grid: `Grid`
            The definition of the source grid.

        dst_grid: `Grid`
            The definition of the destination grid.

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

        esmpy_regrid_operator: `None` or `list`, optional
            If a `list` then the `esmpy.Regrid` instance that created
            the instance is made available as the list's last element.

        weights_file: `str` or `None`, optional
            Provide a netCDF file that contains, or will contain, the
            regridding weights. If `None` (the default) then the
            weights are computed in memory for regridding between the
            source and destination grids, and no file is created.

            If set to a file path that does not exist then the
            weights will be computed and also written to that file.

            If set to a file path that already exists then the weights
            will be read from this file, instead of being computed.

            .. versionadded:: 3.15.2

    :Returns:

        5-`tuple`
            * weights: Either the 1-d `numpy` array of the regridding
                   weights. Or `None` if the regridding weights are to
                   be read from a file.
            * row: The 1-d `numpy` array of the row indices of the
                   regridding weights in the dense weights matrix,
                   which has J rows and I columns, where J and I are
                   the total number of cells in the destination and
                   source grids respectively. The start index is 1. Or
                   `None` if the indices are to be read from a file.
            * col: The 1-d `numpy` array of column indices of the
                   regridding weights in the dense weights matrix,
                   which has J rows and I columns, where J and I are
                   the total number of cells in the destination and
                   source grids respectively. The start index is 1. Or
                   `None` if the indices are to be read from a file.
            * start_index: The non-negative integer start index of the
                   row and column indices.
            * from_file: `True` if the weights were read from a file,
                   otherwise `False`.

    """
    start_index = 1

    compute_weights = True
    if esmpy_regrid_operator is None and weights_file is not None:
        from os.path import isfile

        if isfile(weights_file):
            # The regridding weights and indices will be read from a
            # file
            compute_weights = False
            weights = None
            row = None
            col = None

    from_file = True
    if compute_weights or esmpy_regrid_operator is not None:
        # Create the weights using ESMF
        from_file = False

        src_mesh_location = src_grid.mesh_location
        if src_mesh_location == "face":
            src_meshloc = esmpy.api.constants.MeshLoc.ELEMENT
        elif src_mesh_location == "point":
            src_meshloc = esmpy.api.constants.MeshLoc.NODE
        elif not src_mesh_location:
            src_meshloc = None

        dst_mesh_location = dst_grid.mesh_location
        if dst_mesh_location == "face":
            dst_meshloc = esmpy.api.constants.MeshLoc.ELEMENT
        elif dst_mesh_location == "point":
            dst_meshloc = esmpy.api.constants.MeshLoc.NODE
        elif not dst_mesh_location:
            dst_meshloc = None

        src_esmpy_field = esmpy.Field(
            src_esmpy_grid, name="src", meshloc=src_meshloc
        )
        dst_esmpy_field = esmpy.Field(
            dst_esmpy_grid, name="dst", meshloc=dst_meshloc
        )

        mask_values = np.array([0], dtype="int32")

        # Create the esmpy.Regrid operator
        r = esmpy.Regrid(
            src_esmpy_field,
            dst_esmpy_field,
            regrid_method=esmpy_methods.get(method),
            unmapped_action=esmpy.UnmappedAction.IGNORE,
            ignore_degenerate=bool(ignore_degenerate),
            src_mask_values=mask_values,
            dst_mask_values=mask_values,
            norm_type=esmpy.api.constants.NormType.FRACAREA,
            factors=True,
        )

        weights = r.get_weights_dict(deep_copy=True)
        row = weights["row_dst"]
        col = weights["col_src"]
        weights = weights["weights"]

        if quarter:
            # The weights were created with a dummy size 2 dimension
            # such that the weights for each dummy axis element are
            # identical. The duplicate weights need to be removed.
            #
            # To do this, only retain the indices that correspond to
            # the top left quarter of the weights matrix in dense
            # form. I.e. if w is the NxM dense form of the weights (N,
            # M both even), then this is equivalent to w[:N//2,
            # :M//2].
            index = np.where(
                (row <= dst_esmpy_field.data.size // 2)
                & (col <= src_esmpy_field.data.size // 2)
            )
            weights = weights[index]
            row = row[index]
            col = col[index]

        if weights_file is not None:
            # Write the weights to a netCDF file (copying the
            # dimension and variable names and structure of a weights
            # file created by ESMF).
            from netCDF4 import Dataset

            from .. import __version__
            from ..data.array.netcdfarray import _lock

            if (
                max(dst_esmpy_field.data.size, src_esmpy_field.data.size)
                <= np.iinfo("int32").max
            ):
                i_dtype = "i4"
            else:
                i_dtype = "i8"

            upper_bounds = src_esmpy_grid.upper_bounds
            if len(upper_bounds) > 1:
                upper_bounds = upper_bounds[0]

            src_shape = tuple(upper_bounds)

            upper_bounds = dst_esmpy_grid.upper_bounds
            if len(upper_bounds) > 1:
                upper_bounds = upper_bounds[0]

            dst_shape = tuple(upper_bounds)

            regrid_method = f"{src_grid.coord_sys} {src_grid.method}"
            if src_grid.ln_z:
                regrid_method += f", ln {src_grid.method} in vertical"

            _lock.acquire()
            nc = Dataset(weights_file, "w", format="NETCDF4")

            nc.title = (
                f"Regridding weights from source {src_grid.type} "
                f"with shape {src_shape} to destination "
                f"{dst_grid.type} with shape {dst_shape}"
            )
            nc.source = f"cf v{__version__}, esmpy v{esmpy.__version__}"
            nc.history = f"Created at {datetime.now()}"
            nc.regrid_method = regrid_method
            nc.ESMF_unmapped_action = r.unmapped_action
            nc.ESMF_ignore_degenerate = int(r.ignore_degenerate)

            nc.createDimension("n_s", weights.size)
            nc.createDimension("src_grid_rank", src_esmpy_grid.rank)
            nc.createDimension("dst_grid_rank", dst_esmpy_grid.rank)

            v = nc.createVariable("src_grid_dims", i_dtype, ("src_grid_rank",))
            v.long_name = "Source grid shape"
            v[...] = src_shape

            v = nc.createVariable("dst_grid_dims", i_dtype, ("dst_grid_rank",))
            v.long_name = "Destination grid shape"
            v[...] = dst_shape

            v = nc.createVariable("S", weights.dtype, ("n_s",))
            v.long_name = "Weights values"
            v[...] = weights

            v = nc.createVariable("row", i_dtype, ("n_s",), zlib=True)
            v.long_name = "Destination/row indices"
            v.start_index = start_index
            v[...] = row

            v = nc.createVariable("col", i_dtype, ("n_s",), zlib=True)
            v.long_name = "Source/col indices"
            v.start_index = start_index
            v[...] = col

            nc.close()
            _lock.release()

    if esmpy_regrid_operator is None:
        # Destroy esmpy objects (the esmpy.Grid objects exist even if
        # we didn't create any weights using esmpy.Regrid).
        src_esmpy_grid.destroy()
        dst_esmpy_grid.destroy()
        if compute_weights:
            src_esmpy_field.destroy()
            dst_esmpy_field.destroy()
            r.srcfield.grid.destroy()
            r.srcfield.destroy()
            r.dstfield.grid.destroy()
            r.dstfield.destroy()
            r.destroy()
    else:
        # Make the Regrid instance available via the
        # 'esmpy_regrid_operator' list
        esmpy_regrid_operator.append(r)

    return weights, row, col, start_index, from_file


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


def get_bounds(method, coords, mesh_location):
    """Get coordinate bounds needed for defining an `esmpy.Grid`.

    .. versionadded:: 3.14.0

    :Parameters:

        method: `str`
            The regridding method.

        coords: sequence of `Coordinate`
            The coordinates that define an `esmpy.Grid`.

        mesh_location: `str`
            The UGRID mesh element of the source grid
            (e.g. ``'face'``). An empty string should be used for a
            non-UGRID source grid.

            .. versionadded:: 3.16.0

    :Returns:

        `list`
            The coordinate bounds. Will be an empty list if there are
            no bounds, or if bounds are not needed for the regridding.

    """
    if not mesh_location and not conservative_regridding(method):
        return []

    if mesh_location == "point":
        # Node cells have no bounds
        return []

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

    # Reorder the mask axes to grid.axis_keys
    axes = np.argsort(regrid_axes).tolist()
    if len(axes) > 1:
        mask = da.transpose(mask, axes=axes)

    return mask


def update_coordinates(src, dst, src_grid, dst_grid):
    """Update the regrid axis coordinates.

    Replace the existing coordinate constructs that span the regridding
    axes with those from the destination grid.

    Also, if the source grid is a mesh, remove the existing domain
    topology and cell connectivity constructs that span the regridding
    axis; and if the destination grid is a mesh copy domain topology
    and cell connectivity constructs from the destination grid.

    .. versionadded:: 3.14.0

    :Parameters:

        src: `Field`
            The regridded source field. Updated in-place.

        dst: `Field` or `Domain`
            The field or domain containing the destination grid.

        src_grid: `Grid` or `Mesh`
            The definition of the source grid.

        dst_grid: `Grid` or `Mesh`
            The definition of the destination grid.

    :Returns:

        `None`

    """
    src_axis_keys = src_grid.axis_keys
    dst_axis_keys = dst_grid.axis_keys

    # Remove the source coordinate, domain topology and cell
    # connectivity constructs from regridded field.
    for key in src.constructs(
        filter_by_type=(
            "dimension_coordinate",
            "auxiliary_coordinate",
            "domain_topology",
            "cell_connectivity",
        ),
        filter_by_axis=src_axis_keys,
        axis_mode="or",
        todict=True,
    ):
        src.del_construct(key)

    # Domain axes
    src_domain_axes = src.domain_axes(todict=True)
    dst_domain_axes = dst.domain_axes(todict=True)
    if src_grid.n_regrid_axes == dst_grid.n_regrid_axes:
        # Change the size of the regridded domain axes
        for src_axis, dst_axis in zip(src_axis_keys, dst_axis_keys):
            src_domain_axis = src_domain_axes[src_axis]
            dst_domain_axis = dst_domain_axes[dst_axis]

            src_domain_axis.set_size(dst_domain_axis.size)

            ncdim = dst_domain_axis.nc_get_dimension(None)
            if ncdim is not None:
                src_domain_axis.nc_set_dimension(ncdim)
    else:
        # The regridding has changed the number of data axes (e.g. by
        # regridding a source mesh grid to a destination non-mesh
        # grid, or vice versa), so insert new domain axis constructs
        # for all of the new axes.
        src_axis_keys = [
            src.set_construct(dst_domain_axes[dst_axis].copy())
            for dst_axis in dst_axis_keys
        ]
        src_grid.new_axis_keys = src_axis_keys

    axis_map = {
        dst_axis: src_axis
        for dst_axis, src_axis in zip(dst_axis_keys, src_axis_keys)
    }
    dst_data_axes = dst.constructs.data_axes()

    # Copy coordinates constructs from the destination grid
    for key, coord in dst.coordinates(
        filter_by_axis=dst_axis_keys, axis_mode="subset", todict=True
    ).items():
        axes = [axis_map[axis] for axis in dst_data_axes[key]]
        src.set_construct(coord, axes=axes)

    # Copy domain topology and cell connectivity constructs from the
    # destination grid
    if dst_grid.is_mesh:
        for key, topology in dst.constructs(
            filter_by_type=("domain_topology", "cell_connectivity"),
            filter_by_axis=dst_axis_keys,
            axis_mode="exact",
            todict=True,
        ).items():
            axes = [axis_map[axis] for axis in dst_data_axes[key]]
            src.set_construct(topology, axes=axes)


def update_non_coordinates(src, dst, src_grid, dst_grid, regrid_operator):
    """Update the coordinate references of the regridded field.

    .. versionadded:: 3.14.0

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
    # Delete source grid cell measure and field ancillary constructs
    # that span any of the regridding axes.
    # ----------------------------------------------------------------
    for key in src.constructs(
        filter_by_type=("cell_measure", "field_ancillary"),
        filter_by_axis=src_axis_keys,
        axis_mode="or",
        todict=True,
    ):
        #        if set(data_axes[key]).intersection(src_axis_keys):
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


def update_data(src, regridded_data, src_grid):
    """Insert the regridded field data.

    .. versionadded:: 3.16.0

    .. seealso: `update_coordinates`, `update_non_coordinates`

    :Parameters:

        src`: `Field`
            The regridded field construct, that will be updated
            in-place.

        regridded_data: `numpy.ndarray`
            The regridded array

        src_grid: `Grid`
            The definition of the source grid.

    :Returns:

        `None`

    """
    data_axes = src.get_data_axes()
    if src_grid.new_axis_keys:
        # The regridding has changed the number of data axes (e.g. by
        # regridding a source UGRID grid to a destination non-UGRID
        # grid, or vice versa) => delete the old, superceded domain
        # axis construct and update the list of data axes.
        data_axes = list(data_axes)
        index = data_axes.index(src_grid.axis_keys[0])
        for axis in src_grid.axis_keys:
            data_axes.remove(axis)
            src.del_construct(axis)

        data_axes[index:index] = src_grid.new_axis_keys

    src.set_data(regridded_data, axes=data_axes, copy=False)


def get_mesh(f):
    """Get domain topology mesh information.

    .. versionadded:: 3.16.0

    :Parameters:

        f: `Field` or `Domain`
            The construct from which to get the mesh information.

    :Returns:

       3-`tuple`
           If the field or domain has no domain topology construct
           then ``(None, None, None)`` is returned. Otherwise the
           tuple contains:

           * The domain topology construct
           * The mesh location of the domain topology (e.g. ``'face'``)
           * The identifier of domain axis construct that is spanned
             by the domain topology construct

    """
    key, domain_topology = f.domain_topology(item=True, default=(None, None))
    if domain_topology is None:
        return (None, None, None)

    return (
        domain_topology,
        domain_topology.get_cell(""),
        f.get_data_axes(key)[0],
    )


def get_dsg(f):
    """Get domain discrete sampling geometry information.

    .. versionadded:: 3.16.2

    :Parameters:

        f: `Field` or `Domain`
            The construct from which to get the DSG information.

    :Returns:

       2-`tuple`
           If the field or domain is not a DSG then ``(None, None)``
           is returned. Otherwise the tuple contains:

           * The featureType (e.g. ``'trajectory'``)
           * The identifier of domain axis construct that is spanned
             by the DSG.

    """
    featureType = f.get_property("featureType", None)
    if featureType is None or f.ndim != 1:
        return (None, None)

    return (
        featureType,
        f.get_data_axes()[0],
    )


def has_coordinate_arrays(grid):
    """Whether all grid coordinates have representative arrays.

    .. versionadded:: 3.16.0

    :Parameters:

        grid: `Grid`
            The definition of the grid.

    :Returns:

        `bool`
            True if and only if there are grid coordinates and they
            all have representative arrays.

    """
    if not grid.coords:
        return False

    for coord in grid.coords:
        try:
            has_data = coord.has_data()
        except AttributeError:
            # 'coord' is not a construct, because it doesn't have a
            # `has_data` attribute, and so must be something that
            # certainly has data (e.g. a numpy array).
            has_data = True

        if not has_data:
            return False

    return True


def set_grid_type(grid):
    """Set the ``type`` attribute of a `Grid` instance in-place.

    .. versionadded:: 3.16.2

    :Parameters:

        grid: `Grid`
            The definition of the grid.

    :Returns:

        `None`

    """
    if grid.is_grid:
        grid.type = "structured grid"
    elif grid.is_mesh:
        grid.type = f"UGRID {grid.mesh_location} mesh"
    elif grid.is_locstream:
        grid.type = f"DSG {grid.featureType}"
