"""Worker functions for regridding."""
from operator import itemgetter

import numpy as np

from ..functions import regrid_logging
from ..dimensioncoordinate import DimensionCoordinate
from ..data import Data
from .. import _found_ESMF

if _found_ESMF:
    try:
        import ESMF
    except Exception as error:
        print(f"WARNING: Can not import ESMF for regridding: {error}")

from .regridoperator import (
    RegridOperator,
    conservative_regridding_methods,
    regrid_method_map,
    regridding_methods,
)


def regrid_compute_mass_grid(
    valuefield,
    areafield,
    dofrac=False,
    fracfield=None,
    uninitval=422397696.0,
):
    """Compute the mass of an `ESMF` Field.

    :Parameters:

        valuefield: `ESMF.Field`
            This contains data values of a field built on the cells of
            a grid.

        areafield: `ESMF.Field`
            This contains the areas associated with the grid cells.

        fracfield: `ESMF.Field`
            This contains the fractions of each cell which contributed
            to a regridding operation involving 'valuefield.

        dofrac: `bool`
            This gives the option to not use the 'fracfield'.

        uninitval: `float`
            The value uninitialised cells take.

    :Returns:

        `float`
            The mass of the data field.

    """
    mass = 0.0
    areafield.get_area()

    ind = np.where(valuefield.data != uninitval)

    if dofrac:
        mass = np.sum(
            areafield.data[ind] * valuefield.data[ind] * fracfield.data[ind]
        )
    else:
        mass = np.sum(areafield.data[ind] * valuefield.data[ind])

    return mass


def regrid_get_latlon(f, name, axes=None):
    """Retrieve the latitude and longitude coordinates of this field and
    associated information. If 1-d lat/long coordinates are found then
    these are returned. Otherwise, 2-d lat/long coordinates are searched
    for and if found returned.

    :Parameters:

        f: `Field`
            The source or destination field.

        name: `str`
            A name to identify the field in error messages. Either
            ``'source'`` or ``'destination'``.

        axes: `dict`, optional
            A dictionary specifying the X and Y axes, with keys
            ``'X'`` and ``'Y'``.

            *Parameter example:*
              ``axes={'X': 'ncdim%x', 'Y': 'ncdim%y'}``

            *Parameter example:*
              ``axes={'X': 1, 'Y': 0}``

    :Returns:

        `list`, `list`, `list`, `list`, `bool`
            * The keys of the x and y dimension coordinates.

            * The sizes of the x and y dimension coordinates.

            * The keys of the x and y coordinate (1-d dimension
              coordinate, or 2-d auxilliary coordinates).

            * The x and y coordinates (1-d dimension coordinates or
              2-d auxilliary coordinates).

            * True if 2-d auxiliary coordinates are returned or if 1-d
              X and Y coordinates are returned, which are not
              long/lat.

    """
    data_axes = f.constructs.data_axes()

    if axes is None:
        # Retrieve the field construct's X and Y dimension
        # coordinates
        x_key, x = f.dimension_coordinate(
            "X",
            item=True,
            default=ValueError(
                f"No unique X dimension coordinate found for the {name} "
                "field. If none is present you "
                "may need to specify the axes keyword."
            ),
        )
        y_key, y = f.dimension_coordinate(
            "Y",
            item=True,
            default=ValueError(
                f"No unique Y dimension coordinate found for the {name} "
                "field. If none is present you "
                "may need to specify the axes keyword."
            ),
        )

        x_axis = data_axes[x_key][0]
        y_axis = data_axes[y_key][0]

        x_size = x.size
        y_size = y.size
    else:
        # --------------------------------------------------------
        # Axes have been provided
        # --------------------------------------------------------
        for key in ("X", "Y"):
            if key not in axes:
                raise ValueError(
                    f"Key {key!r} must be specified for axes of {name} "
                    "field."
                )

        if axes["X"] in (1, 0) and axes["Y"] in (0, 1):
            # Axes specified by integer position in dimensions of
            # lat and lon 2-d auxiliary coordinates
            if axes["X"] == axes["Y"]:
                raise ValueError("TODO 0")

            lon_key, lon = f.auxiliary_coordinate(
                "X", item=True, filter_by_naxes=(2,), default=(None, None)
            )
            lat_key, lat = f.auxiliary_coordinate(
                "Y", item=True, filter_by_naxes=(2,), default=(None, None)
            )
            if lon is None:
                raise ValueError("TODO x")
            if lat is None:
                raise ValueError("TODO y")

            if lat.shape != lon.shape:
                raise ValueError("TODO 222222")

            lon_axes = data_axes[lon_key]
            lat_axes = data_axes[lat_key]
            if lat_axes != lon_axes:
                raise ValueError("TODO 3333333")

            x_axis = lon_axes[axes["X"]]
            y_axis = lat_axes[axes["Y"]]
        else:
            x_axis = f.domain_axis(
                axes["X"],
                key=True,
                default=ValueError(
                    f"'X' axis specified for {name} field not found."
                ),
            )

            y_axis = f.domain_axis(
                axes["Y"],
                key=True,
                default=ValueError(
                    f"'Y' axis specified for {name} field not found."
                ),
            )

        domain_axes = f.domain_axes(todict=True)
        x_size = domain_axes[x_axis].get_size()
        y_size = domain_axes[y_axis].get_size()

    axis_keys = [x_axis, y_axis]
    axis_sizes = [x_size, y_size]

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
                        f"of the {name} field is not unique."
                    )
                else:
                    lon_found = True
                    x = aux
                    x_key = key

            if aux.Units.islatitude:
                if lat_found:
                    raise ValueError(
                        "The 2-d auxiliary latitude coordinate "
                        f"of the {name} field is not unique."
                    )
                else:
                    lat_found = True
                    y = aux
                    y_key = key

        if not lon_found or not lat_found:
            raise ValueError(
                "Both longitude and latitude coordinates "
                f"were not found for the {name} field."
            )

        if axes is not None:
            if set(axis_keys) != set(data_axes[x_key]):
                raise ValueError(
                    "Axes of longitude do not match "
                    f"those specified for {name} field."
                )

            if set(axis_keys) != set(data_axes[y_key]):
                raise ValueError(
                    "Axes of latitude do not match "
                    f"those specified for {name} field."
                )

        coords_2D = True
    else:
        coords_2D = False
        # Check for size 1 latitude or longitude dimensions
        if x_size == 1 or y_size == 1:
            raise ValueError(
                "Neither the longitude nor latitude dimension coordinates "
                f"of the {name} field can be of size 1."
            )

    coord_keys = [x_key, y_key]
    coords = [x, y]

    return axis_keys, axis_sizes, coord_keys, coords, coords_2D


def get_cartesian_coords(f, name, axes):
    """Retrieve the specified cartesian dimension coordinates of the
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
    for axis in axes:
        key = f.domain_axis(axis, key=True)
        axis_keys.append(key)

    coords = []
    for key in axis_keys:
        d = f.dimension_coordinate(filter_by_axis=(key,), default=None)
        if d is None:
            raise ValueError(
                f"No unique {name} dimension coordinate "
                f"matches key {key!r}."
            )

        coords.append(d.copy())

    return axis_keys, coords


def regrid_get_axis_indices(f, axis_keys):
    """Get axis indices and their orders in rank of this field.

    :Parameters:

         f: `Field`
            The source or destination field. This field might get
            size-1 dimensions inserted into its data in-place.

        axis_keys: sequence
            A sequence of axis specifiers.

    :Returns:

        `list`, `numpy.ndarray`
            A list of the indices of the specified axes; and the rank
            order of the axes.

    """
    data_axes = f.get_data_axes()

    # Get the positions of the axes
    axis_indices = []
    for axis_key in axis_keys:
        try:
            axis_index = data_axes.index(axis_key)
        except ValueError:
            f.insert_dimension(axis_key, position=0, inplace=True)
            axis_index = data_axes.index(axis_key)

        axis_indices.append(axis_index)

    # Get the rank order of the positions of the axes
    tmp = np.array(axis_indices)
    tmp = tmp.argsort()
    n = len(tmp)
    order = np.empty((n,), dtype=int)
    order[tmp] = np.arange(n)

    return axis_indices, order


def regrid_get_coord_order(f, axis_keys, coord_keys):
    """Get the ordering of the axes for each N-d auxiliary coordinate.

    :Parameters:

         f: `Field`
            The source or destination field.

        axis_keys: sequence
            A sequence of axis keys.

        coord_keys: sequence
            A sequence of keys for each of the N-d auxiliary
            coordinates.

    :Returns:

        `list`
            A list of lists specifying the ordering of the axes for
            each N-d auxiliary coordinate.

    """
    coord_axes = [f.get_data_axes(coord_key) for coord_key in coord_keys]
    coord_order = [
        [coord_axis.index(axis_key) for axis_key in axis_keys]
        for coord_axis in coord_axes
    ]

    return coord_order


def regrid_get_section_shape(src, axis_sizes, axis_indices):
    """Get the shape of each regridded section.

    :Parameters:

        src: `Field`
            The source field.

        axis_sizes: sequence
            A sequence of the sizes of each axis along which the
            section.  will be taken

        axis_indices: sequence
            A sequence of the same length giving the axis index of
            each axis.

    :Returns:

        `list`
            A list of integers defining the shape of each section.

    """
    shape = [1] * src.ndim
    for i, axis_index in enumerate(axis_indices):
        shape[axis_index] = axis_sizes[i]

    return shape


def regrid_check_bounds(src_coords, dst_coords, method, ext_coords=None):
    """Check the bounds of the coordinates for regridding.

    :Parameters:

        src_coords: sequence
            A sequence of the source coordinates.

        dst_coords: sequence
            A sequence of the destination coordinates.

        method: `str`
            A string indicating the regrid method.

        ext_coords: `None` or sequence
            If a sequence of extension coordinates is present these
            are also checked. Only used for cartesian regridding when
            regridding only 1 (only 1!) dimension of an N>2
            dimensional field. In this case we need to provided the
            coordinates of the dimensions that aren't being regridded
            (that are the same in both src and dst grids) so that we
            can create a sensible ESMF grid object.

    :Returns:

        `None`

    """
    if method not in conservative_regridding_methods:
        # Bounds are only not needed for conservative methods
        return

    for name, coords in zip(
        ("Source", "Destination"), (src_coords, dst_coords)
    ):
        for coord in coords:
            if not coord.has_bounds():
                raise ValueError(
                    f"{name} {coord!r} coordinates must have bounds "
                    "for conservative regridding."
                )

            if not coord.contiguous(overlap=False):
                raise ValueError(
                    f"{name} {coord!r} coordinates must have "
                    "contiguous, non-overlapping bounds "
                    "for conservative regridding."
                )

    if ext_coords is not None:
        for coord in ext_coords:
            if not coord.has_bounds():
                raise ValueError(
                    f"{coord!r} dimension coordinates must have "
                    "bounds for conservative regridding."
                )
            if not coord.contiguous(overlap=False):
                raise ValueError(
                    f"{coord!r} dimension coordinates must have "
                    "contiguous, non-overlapping bounds "
                    "for conservative regridding."
                )


def regrid_check_method(method):
    """Check that the regrid method is valid.

    If it is not valid then an exception is raised.

    :Parameters:

        method: `str`
            The regridding method.

    :Returns:

        `None`

    """
    if method not in regridding_methods:
        raise ValueError(
            "Can't regrid: Must set a valid regridding method from "
            f"{regridding_methods}. Got: {method!r}"
        )
    elif method == "bilinear":  # TODO use logging.info() once have logging
        print(
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
            "The use_src_mask parameter can only be False when using the "
            "'nearest_stod' regridding method."
        )


def regrid_get_reordered_sections(
    src, axis_order, regrid_axes, regrid_axis_indices
):
    """Get a dictionary of the data sections for regridding and a list
    of its keys reordered if necessary so that they will be looped over
    in the order specified in axis_order.

    :Parameters:

        src: `Field`
            The source field.

        axis_order: `None` or sequence of axes specifiers.
            If `None` then the sections keys will not be reordered. If
            a particular axis is one of the regridding axes or is not
            found then a ValueError will be raised.

        regrid_axes: sequence
            A sequence of the keys of the regridding axes.

        regrid_axis_indices: sequence
            A sequence of the indices of the regridding axes.

    :Returns:

        `list`, `dict`
            An ordered list of the section keys; and a dictionary of
            the data sections for regridding.

    """
    # If we had dynamic masking, we wouldn't need this method, we
    # could sdimply replace it in the clling function with a call to
    # Data.section. However, we don't have it, so this allows us to
    # possibibly reduce the number of trasnistions between different
    # masks - each change is slow.
    data_axes = src.get_data_axes()

    axis_indices = []
    if axis_order is not None:
        for axis in axis_order:
            axis_key = src.dimension_coordinate(
                filter_by_axis=(axis,),
                default=None,
                key=True,
            )
            if axis_key is not None:
                if axis_key in regrid_axes:
                    raise ValueError("Cannot loop over regridding axes.")

                try:
                    axis_indices.append(data_axes.index(axis_key))
                except ValueError:
                    # The axis has been squeezed so do nothing
                    pass

            else:
                raise ValueError(f"Source field axis not found: {axis!r}")

    # Section the data
    sections = src.data.section(regrid_axis_indices)

    # Reorder keys correspondingly if required
    if axis_indices:
        section_keys = sorted(sections.keys(), key=itemgetter(*axis_indices))
    else:
        section_keys = sections.keys()

    return section_keys, sections


def regrid_get_destination_mask(
    f, dst_order, axes=("X", "Y"), cartesian=False, coords_ext=None
):
    """Get the mask of the destination field.

    :Parameters:

        f: `Field`
            The destination field.

        dst_order: sequence, optional
            The order of the destination axes.

        axes: optional
            The axes the data is to be sectioned along.

        cartesian: `bool`, optional
            Whether the regridding is Cartesian or spherical.

        coords_ext: sequence, optional
            In the case of Cartesian regridding, extension coordinates
            (see _regrid_check_bounds for details).

    :Returns:

        `numpy.ndarray`
            The mask.

    """
    data_axes = f.get_data_axes()

    indices = {axis: [0] for axis in data_axes if axis not in axes}

    g = f.subspace(**indices)
    g = g.squeeze(tuple(indices)).transpose(dst_order)

    dst_mask = g.mask.array

    if cartesian:
        tmp = []
        for coord in coords_ext:
            tmp.append(coord.size)
            dst_mask = np.tile(dst_mask, tmp + [1] * dst_mask.ndim)

    return dst_mask


def regrid_fill_fields(src_data, srcfield, dstfield, fill_value):
    """Fill the source and destination Fields.

    Fill the source Field with data and the destination Field with
    fill values.

    :Parameters:

        src_data: ndarray
            The data to fill the source field with.

        srcfield: `ESMF.Field`
            The source field.

        dstfield: `ESMF.Field`
            The destination field. This get always gets initialised with
            missing values.

        fill_value:
            The fill value with which to fill *dstfield*

    :Returns:

        `None`

    """
    srcfield.data[...] = np.ma.MaskedArray(src_data, copy=False).filled(
        fill_value
    )
    dstfield.data[...] = fill_value


def regrid_compute_field_mass(
    f,
    _compute_field_mass,
    k,
    srcgrid,
    srcfield,
    srcfracfield,
    dstgrid,
    dstfield,
):
    """Compute the field mass for conservative regridding.

    The mass should be the same before and after regridding.

    :Parameters:

        f: `Field`
            The source field.

        _compute_field_mass: `dict`
            A dictionary for the results.

        k: `tuple`
            A key identifying the section of the field being regridded.

        srcgrid: `ESMF.Grid`
            The source grid.

        srcfield: `ESMF.Grid`
            The source field.

        srcfracfield: `ESMF.Field`
            Information about the fraction of each cell of the source
            field used in regridding.

        dstgrid: `ESMF.Grid`
            The destination grid.

        dstfield: `ESMF.Field`
            The destination field.

    :Returns:

        `None`

    """
    if not isinstance(_compute_field_mass, dict):
        raise ValueError("Expected _compute_field_mass to be a dictionary.")

    fill_value = f.fill_value(default="netCDF")

    # Calculate the mass of the source field
    srcareafield = create_Field(srcgrid, "srcareafield")
    srcmass = regrid_compute_mass_grid(
        srcfield,
        srcareafield,
        dofrac=True,
        fracfield=srcfracfield,
        uninitval=fill_value,
    )

    # Calculate the mass of the destination field
    dstareafield = create_Field(dstgrid, "dstareafield")
    dstmass = regrid_compute_mass_grid(
        dstfield, dstareafield, uninitval=fill_value
    )

    # Insert the two masses into the dictionary for comparison
    _compute_field_mass[k] = (srcmass, dstmass)


def regrid_get_regridded_data(f, method, fracfield, dstfield, dstfracfield):
    """Get the regridded data.

    :Parameters:

        f: `Field`
            The source field.

        method: `str`
            The regridding method.

        fracfield: `bool`
            Whether to return the frac field or not in the case of
            conservative regridding.

        dstfield: `ESMF.Field`
            The destination field.

        dstfracfield: `ESMF.Field`
            Information about the fraction of each of the destination
            field cells involved in the regridding. For conservative
            regridding this must be taken into account.

    :Returns:

        `numpy.ndarray`
            The regridded data.

    """
    if method in conservative_regridding_methods:
        frac = dstfracfield.data.copy()
        if fracfield:
            regridded_data = frac
        else:
            frac[frac == 0.0] = 1.0
            regridded_data = np.ma.MaskedArray(
                dstfield.data / frac,
                mask=(dstfield.data == f.fill_value(default="netCDF")),
            )
    else:
        regridded_data = np.ma.MaskedArray(
            dstfield.data.copy(),
            mask=(dstfield.data == f.fill_value(default="netCDF")),
        )

    return regridded_data


def regrid_update_coordinate_references(
    f,
    dst,
    src_axis_keys,
    dst_axis_sizes,
    method,
    use_dst_mask,
    cartesian=False,
    axes=("X", "Y"),
    n_axes=2,
    src_cyclic=False,
    dst_cyclic=False,
):
    """Update the coordinate references of the regridded field.

    :Parameters:

        f: `Field`
            The regridded field. Updated in-place.

        dst: `Field` or `dict`
            The object with the destination grid for regridding.

        src_axis_keys: sequence of `str`
            The keys of the source regridding axes.

        dst_axis_sizes: sequence, optional
            The sizes of the destination axes.

        method: `bool`
            The regridding method.

        use_dst_mask: `bool`
            Whether to use the destination mask in regridding.

        cartesian: `bool`, optional
            Whether to do Cartesian regridding or spherical

        axes: sequence, optional
            Specifiers for the regridding axes.

        n_axes: `int`, optional
            The number of regridding axes.

        src_cyclic: `bool`, optional
            Whether the source longitude is cyclic for spherical
            regridding.

        dst_cyclic: `bool`, optional
            Whether the destination longitude is cyclic for spherical
            regridding.

    :Returns:

        `None`

    """
    domain_ancillaries = f.domain_ancillaries(todict=True)

    # Initialise cached value for domain_axes
    domain_axes = None

    data_axes = f.constructs.data_axes()

    for key, ref in f.coordinate_references(todict=True).items():
        ref_axes = []
        for k in ref.coordinates():
            ref_axes.extend(data_axes[k])

        if set(ref_axes).intersection(src_axis_keys):
            f.del_construct(key)
            continue

        for (
            term,
            value,
        ) in ref.coordinate_conversion.domain_ancillaries().items():
            if value not in domain_ancillaries:
                continue

            key = value

            # If this domain ancillary spans both X and Y axes
            # then regrid it, otherwise remove it
            x = f.domain_axis("X", key=True)
            y = f.domain_axis("Y", key=True)
            if (
                f.domain_ancillary(
                    filter_by_axis=(x, y),
                    axis_mode="exact",
                    key=True,
                    default=None,
                )
                is not None
            ):
                # Convert the domain ancillary into an independent
                # field
                value = f.convert(key)
                try:
                    if cartesian:
                        value.regridc(
                            dst,
                            axes=axes,
                            method=method,
                            use_dst_mask=use_dst_mask,
                            inplace=True,
                        )
                    else:
                        value.regrids(
                            dst,
                            src_cyclic=src_cyclic,
                            dst_cyclic=dst_cyclic,
                            method=method,
                            use_dst_mask=use_dst_mask,
                            inplace=True,
                        )
                except ValueError:
                    ref.coordinate_conversion.set_domain_ancillary(term, None)
                    f.del_construct(key)
                else:
                    ref.coordinate_conversion.set_domain_ancillary(term, key)
                    d_axes = data_axes[key]

                    domain_axes = f.domain_axes(
                        cached=domain_axes, todict=True
                    )

                    for k_s, new_size in zip(src_axis_keys, dst_axis_sizes):
                        domain_axes[k_s].set_size(new_size)

                    f.set_construct(
                        f._DomainAncillary(source=value),
                        key=key,
                        axes=d_axes,
                        copy=False,
                    )


def regrid_copy_coordinate_references(f, dst, dst_axis_keys):
    """Copy coordinate references from the destination field to the new,
    regridded field.

    :Parameters:

        f: `Field`
            The source field.

        dst: `Field`
            The destination field.

        dst_axis_keys: sequence of `str`
            The keys of the regridding axes in the destination field.

    :Returns:

        `None`

    """
    dst_data_axes = dst.constructs.data_axes()

    for ref in dst.coordinate_references(todict=True).values():
        axes = set()
        for key in ref.coordinates():
            axes.update(dst_data_axes[key])

        if axes and set(axes).issubset(dst_axis_keys):
            # This coordinate reference's coordinates span the X
            # and/or Y axes
            f.set_coordinate_reference(ref, parent=dst, strict=True)


def regrid_use_bounds(method):
    """Returns whether to use the bounds or not in regridding. This is
    only the case for conservative regridding.

    :Parameters:

        method: `str`
            The regridding method

    :Returns:

        `bool`

    """
    return method in conservative_regridding_methods


def regrid_update_coordinates(
    f,
    dst,
    dst_dict,
    dst_coords,
    src_axis_keys,
    dst_axis_keys,
    cartesian=False,
    dst_axis_sizes=None,
    dst_coords_2D=False,
    dst_coord_order=None,
):
    """Update the coordinates of the regridded field.

    :Parameters:

        f: `Field`
            The regridded field. Updated in-place.

        dst: Field or `dict`
            The object containing the destination grid.

        dst_dict: `bool`
            Whether dst is a dictionary.

        dst_coords: sequence
            The destination coordinates.

        src_axis_keys: sequence
            The keys of the regridding axes in the source field.

        dst_axis_keys: sequence
            The keys of the regridding axes in the destination field.

        cartesian: `bool`, optional
            Whether regridding is Cartesian of spherical, False by
            default.

        dst_axis_sizes: sequence, optional
            The sizes of the destination axes.

        dst_coords_2D: `bool`, optional
            Whether the destination coordinates are 2-d, currently only
            applies to spherical regridding.

        dst_coord_order: `list`, optional
            A list of lists specifying the ordering of the axes for
            each 2-d destination coordinate.

    :Returns:

        `None`

    """
    # NOTE: May be common ground between cartesian and spherical that
    # could save some lines of code.

    # Remove the source coordinates of new field
    for key in f.coordinates(
        filter_by_axis=src_axis_keys, axis_mode="or", todict=True
    ):
        f.del_construct(key)

    domain_axes = f.domain_axes(todict=True)

    if cartesian:
        # Insert coordinates from dst into new field
        if dst_dict:
            for k_s, coord in zip(src_axis_keys, dst_coords):
                domain_axes[k_s].set_size(coord.size)
                f.set_construct(coord, axes=[k_s])
        else:
            axis_map = {
                key_d: key_s
                for key_s, key_d in zip(src_axis_keys, dst_axis_keys)
            }

            for key_d in dst_axis_keys:
                dim = dst.dimension_coordinate(filter_by_axis=(key_d,))
                key_s = axis_map[key_d]
                domain_axes[key_s].set_size(dim.size)
                f.set_construct(dim, axes=[key_s])

            dst_data_axes = dst.constructs.data_axes()

            for aux_key, aux in dst.auxiliary_coordinates(
                filter_by_axis=dst_axis_keys,
                axis_mode="subset",
                todict=True,
            ).items():
                aux_axes = [
                    axis_map[key_d] for key_d in dst_data_axes[aux_key]
                ]
                f.set_construct(aux, axes=aux_axes)
    else:
        # Give destination grid latitude and longitude standard names
        dst_coords[0].standard_name = "longitude"
        dst_coords[1].standard_name = "latitude"

        # Insert 'X' and 'Y' coordinates from dst into new field
        for axis_key, axis_size in zip(src_axis_keys, dst_axis_sizes):
            domain_axes[axis_key].set_size(axis_size)

        if dst_dict:
            if dst_coords_2D:
                for coord, coord_order in zip(dst_coords, dst_coord_order):
                    axis_keys = [src_axis_keys[index] for index in coord_order]
                    f.set_construct(coord, axes=axis_keys)
            else:
                for coord, axis_key in zip(dst_coords, src_axis_keys):
                    f.set_construct(coord, axes=[axis_key])

        else:
            for src_axis_key, dst_axis_key in zip(
                src_axis_keys, dst_axis_keys
            ):
                dim_coord = dst.dimension_coordinate(
                    filter_by_axis=(dst_axis_key,), default=None
                )
                if dim_coord is not None:
                    f.set_construct(dim_coord, axes=[src_axis_key])

                for aux in dst.auxiliary_coordinates(
                    filter_by_axis=(dst_axis_key,),
                    axis_mode="exact",
                    todict=True,
                ).values():
                    f.set_construct(aux, axes=[src_axis_key])

            for aux_key, aux in dst.auxiliary_coordinates(
                filter_by_axis=dst_axis_keys,
                axis_mode="subset",
                todict=True,
            ).items():
                aux_axes = dst.get_data_axes(aux_key)
                if aux_axes == tuple(dst_axis_keys):
                    f.set_construct(aux, axes=src_axis_keys)
                else:
                    f.set_construct(aux, axes=src_axis_keys[::-1])

    # Copy names of dimensions from destination to source field
    if not dst_dict:
        dst_domain_axes = dst.domain_axes(todict=True)
        for src_axis_key, dst_axis_key in zip(src_axis_keys, dst_axis_keys):
            ncdim = dst_domain_axes[dst_axis_key].nc_get_dimension(None)
            if ncdim is not None:
                domain_axes[src_axis_key].nc_set_dimension(ncdim)


def grids_have_same_masks(grid0, grid1):
    """Whether two `ESMF.Grid` instances have identical masks.

    :Parameters:

        grid0, grid1: `ESMF.Grid`, `ESMF.Grid`
            The `ESMF` Grid instances to be compared

    :Returns:

        `bool`
            Whether or not the Grids have identical masks.

    """
    mask0 = grid0.mask
    mask1 = grid1.mask

    if len(mask0) != len(mask1):
        return False

    for a, b in zip(mask0, mask1):
        if not np.array_equal(a, b):
            return False

    return True


def grids_have_same_coords(grid0, grid1):
    """Whether two `ESMF.Grid` instances have identical coordinates.

    :Parameters:

        grid0, grid1: `ESMF.Grid`, `ESMF.Grid`
            The `ESMF` Grid instances to be compared

    :Returns:

        `bool`
            Whether or not the Grids have identical coordinates.

    """
    coords0 = grid0.coords
    coords1 = grid1.coords

    if len(coords0) != len(coords1):
        return False

    for c, d in zip(coords0, coords1):
        if len(c) != len(d):
            return False

        for a, b in zip(c, d):
            if not np.array_equal(a, b):
                return False

    return True


def regrid_initialize():
    """Initialize `ESMF`.

    Initialise the `ESMF` manager. Whether logging is enabled or not
    is determined by cf.regrid_logging. If it is then logging takes
    place after every call to `ESMF`.

    :Returns:

        `ESMF.Manager`
            A singleton instance of the `ESMF` manager.

    """
    return ESMF.Manager(debug=bool(regrid_logging()))


def create_Regrid(
    srcfield,
    dstfield,
    srcfracfield,
    dstfracfield,
    method,
    ignore_degenerate,
):
    """Create an `ESMF` regrid operator.

    :Parameters:

        srcfield: `ESMF.Field`
            The source Field with an associated grid to be used
            for regridding.

        dstfield: `ESMF.Field`
            The destination Field with an associated grid to be
            used for regridding.

        srcfracfield: `ESMF.Field`
            A Field to hold the fraction of the source field that
            contributes to conservative regridding.

        dstfracfield: `ESMF.Field`
            A Field to hold the fraction of the source field that
            contributes to conservative regridding.

        method: `str`
            Setting as follows gives a corresponding result of:

            ===================  =================================
            *method*             Description
            ===================  =================================
            'conservative_1st'   Use first-order conservative
                                 regridding.

            'conservative'       Alias for 'conservative_1st'.

            'conservative_2nd'   Use second-order conservative
                                 regridding.

            'linear'             Use (multi)linear interpolation.

            'bilinear'           Deprecated alias for 'linear'.

            'patch'              Use higher-order patch recovery.

            'nearest_stod'       Use nearest source to
                                 destination interpolation.

            'nearest_dtos'       Use nearest destination to
                                 source interpolation.
            ===================  =================================

        ignore_degenerate: `bool`
            Whether to check for degenerate points.

    :Returns:

        `ESMF.Regrid`
            The `ESMF` regrid operator.

    """
    regrid_method = regrid_method_map.get(method)
    if regrid_method is None:
        raise ValueError(f"Regrid method {method!r} not recognised.")

    # Initialise the regridder. This also creates the weights
    # needed for the regridding.
    return ESMF.Regrid(
        srcfield,
        dstfield,
        regrid_method=regrid_method,
        src_mask_values=np.array([0], dtype="int32"),
        dst_mask_values=np.array([0], dtype="int32"),
        src_frac_field=srcfracfield,
        dst_frac_field=dstfracfield,
        unmapped_action=ESMF.UnmappedAction.IGNORE,
        ignore_degenerate=ignore_degenerate,
    )


def create_Grid(
    coords,
    use_bounds,
    mask=None,
    cartesian=False,
    cyclic=False,
    coords_2D=False,
    coord_order=None,
):
    """Create an `ESMF` Grid.

    :Parameters:

        coords: sequence
            The coordinates if not Cartesian it is assume that the
            first is longitude and the second is latitude.

        use_bounds: `bool`
            Whether to populate the grid corners with information from
            the bounds or not.

        mask: `numpy.ndarray`, optional
            An optional numpy array of booleans containing the grid
            points to mask. Where the elements of mask are True the
            output grid is masked.

        cartesian: `bool`, optional
            Whether to create a Cartesian grid or a spherical one,
            False by default.

        cyclic: `bool`, optional
            Whether or not the longitude (if present) is cyclic. If
            None the a check for cyclicity is made from the
            bounds. None by default.

        coords_2D: `bool`, optional
            Whether the coordinates are 2-d or not. Presently only
            works for spherical coordinates. False by default.

        coord_order: sequence, optional
            Two tuples one indicating the order of the x and y axes
            for 2-d longitude, one for 2-d latitude.

    :Returns:

        `ESMF.Grid`
            The `ESMF` grid.

    """
    if not cartesian:
        lon = coords[0]
        lat = coords[1]
        if not coords_2D:
            # Get the shape of the grid
            shape = [lon.size, lat.size]
        else:
            x_order = coord_order[0]
            y_order = coord_order[1]
            # Get the shape of the grid
            shape = lon.transpose(x_order).shape
            if lat.transpose(y_order).shape != shape:
                raise ValueError(
                    "The longitude and latitude coordinates"
                    " must have the same shape."
                )

        if use_bounds:
            if not coords_2D:
                # Get the bounds
                x_bounds = lon.get_bounds()
                y_bounds = lat.get_bounds().clip(-90, 90, "degrees").array

                # If cyclic not set already, check for cyclicity
                if cyclic is None:
                    cyclic = abs(
                        x_bounds.datum(-1) - x_bounds.datum(0)
                    ) == Data(360, "degrees")

                x_bounds = x_bounds.array
            else:
                # Get the bounds
                x_bounds = lon.get_bounds()
                y_bounds = lat.get_bounds().clip(-90, 90, "degrees")
                n = x_bounds.shape[0]
                m = x_bounds.shape[1]
                x_bounds = x_bounds.array
                y_bounds = y_bounds.array

                tmp_x = np.empty((n + 1, m + 1))
                tmp_x[:n, :m] = x_bounds[:, :, 0]
                tmp_x[:n, m] = x_bounds[:, -1, 1]
                tmp_x[n, :m] = x_bounds[-1, :, 3]
                tmp_x[n, m] = x_bounds[-1, -1, 2]

                tmp_y = np.empty((n + 1, m + 1))
                tmp_y[:n, :m] = y_bounds[:, :, 0]
                tmp_y[:n, m] = y_bounds[:, -1, 1]
                tmp_y[n, :m] = y_bounds[-1, :, 3]
                tmp_y[n, m] = y_bounds[-1, -1, 2]

                x_bounds = tmp_x
                y_bounds = tmp_y

        else:
            if not coords_2D:
                # If cyclicity not set already, check for cyclicity
                if cyclic is None:
                    try:
                        x_bounds = lon.get_bounds()
                        cyclic = abs(
                            x_bounds.datum(-1) - x_bounds.datum(0)
                        ) == Data(360, "degrees")
                    except ValueError:
                        pass

        # Create empty grid
        max_index = np.array(shape, dtype="int32")
        if use_bounds:
            staggerLocs = [ESMF.StaggerLoc.CORNER, ESMF.StaggerLoc.CENTER]
        else:
            staggerLocs = [ESMF.StaggerLoc.CENTER]

        if cyclic:
            grid = ESMF.Grid(
                max_index, num_peri_dims=1, staggerloc=staggerLocs
            )
        else:
            grid = ESMF.Grid(max_index, staggerloc=staggerLocs)

        # Populate grid centres
        x, y = 0, 1
        gridXCentre = grid.get_coords(x, staggerloc=ESMF.StaggerLoc.CENTER)
        gridYCentre = grid.get_coords(y, staggerloc=ESMF.StaggerLoc.CENTER)
        if not coords_2D:
            gridXCentre[...] = lon.array.reshape((lon.size, 1))
            gridYCentre[...] = lat.array.reshape((1, lat.size))
        else:
            gridXCentre[...] = lon.transpose(x_order).array
            gridYCentre[...] = lat.transpose(y_order).array

        # Populate grid corners if there are bounds
        if use_bounds:
            gridCorner = grid.coords[ESMF.StaggerLoc.CORNER]
            if not coords_2D:
                if cyclic:
                    gridCorner[x][...] = x_bounds[:, 0].reshape(lon.size, 1)
                else:
                    n = x_bounds.shape[0]
                    tmp_x = np.empty(n + 1)
                    tmp_x[:n] = x_bounds[:, 0]
                    tmp_x[n] = x_bounds[-1, 1]
                    gridCorner[x][...] = tmp_x.reshape(lon.size + 1, 1)

                n = y_bounds.shape[0]
                tmp_y = np.empty(n + 1)
                tmp_y[:n] = y_bounds[:, 0]
                tmp_y[n] = y_bounds[-1, 1]
                gridCorner[y][...] = tmp_y.reshape(1, lat.size + 1)
            else:
                gridCorner = grid.coords[ESMF.StaggerLoc.CORNER]
                x_bounds = x_bounds.transpose(x_order)
                y_bounds = y_bounds.transpose(y_order)
                if cyclic:
                    x_bounds = x_bounds[:-1, :]
                    y_bounds = y_bounds[:-1, :]

                gridCorner[x][...] = x_bounds
                gridCorner[y][...] = y_bounds
    else:
        # Test the dimensionality of the list of coordinates
        ndim = len(coords)
        if ndim < 1 or ndim > 3:
            raise ValueError(
                "Cartesian grid must have between 1 and 3 dimensions."
            )

        # For 1-d conservative regridding add an extra dimension of
        # size 1
        if ndim == 1:
            if not use_bounds:
                # For 1-d non-conservative regridding, the extra
                # dimension should already have been added in the
                # calling function.
                raise ValueError(
                    "Cannot create a Cartesian grid from "
                    "one dimension coordinate with no bounds."
                )
            coords = [
                DimensionCoordinate(
                    data=Data(0),
                    bounds=Data(
                        [
                            np.finfo("float32").epsneg,
                            np.finfo("float32").eps,
                        ]
                    ),
                )
            ] + coords
            if mask is not None:
                mask = mask[None, :]
            ndim = 2

        shape = list()
        for coord in coords:
            shape.append(coord.size)

        # Initialise the grid
        max_index = np.array(shape, dtype="int32")
        if use_bounds:
            if ndim < 3:
                staggerLocs = [
                    ESMF.StaggerLoc.CORNER,
                    ESMF.StaggerLoc.CENTER,
                ]
            else:
                staggerLocs = [
                    ESMF.StaggerLoc.CENTER_VCENTER,
                    ESMF.StaggerLoc.CORNER_VFACE,
                ]
        else:
            if ndim < 3:
                staggerLocs = [ESMF.StaggerLoc.CENTER]
            else:
                staggerLocs = [ESMF.StaggerLoc.CENTER_VCENTER]

        grid = ESMF.Grid(
            max_index, coord_sys=ESMF.CoordSys.CART, staggerloc=staggerLocs
        )

        # Populate the grid centres
        for d in range(0, ndim):
            if ndim < 3:
                gridCentre = grid.get_coords(
                    d, staggerloc=ESMF.StaggerLoc.CENTER
                )
            else:
                gridCentre = grid.get_coords(
                    d, staggerloc=ESMF.StaggerLoc.CENTER_VCENTER
                )
            gridCentre[...] = coords[d].array.reshape(
                [shape[d] if x == d else 1 for x in range(0, ndim)]
            )

        # Populate grid corners
        if use_bounds:
            if ndim < 3:
                gridCorner = grid.coords[ESMF.StaggerLoc.CORNER]
            else:
                gridCorner = grid.coords[ESMF.StaggerLoc.CORNER_VFACE]

            for d in range(0, ndim):
                # boundsD = coords[d].get_bounds(create=True).array
                boundsD = coords[d].get_bounds(None)
                if boundsD is None:
                    boundsD = coords[d].create_bounds()

                boundsD = boundsD.array

                if shape[d] > 1:
                    tmp = np.empty(shape[d] + 1)
                    tmp[0:-1] = boundsD[:, 0]
                    tmp[-1] = boundsD[-1, 1]
                    boundsD = tmp

                gridCorner[d][...] = boundsD.reshape(
                    [shape[d] + 1 if x == d else 1 for x in range(0, ndim)]
                )

    # Add the mask if appropriate
    if mask is not None:
        gmask = grid.add_item(ESMF.GridItem.MASK)
        gmask[...] = 1
        gmask[mask] = 0

    return grid


def create_Field(grid, name):
    """Create an `ESMF` Field.

    :Parameters:

        grid: `ESMF.Grid`
            The `ESMF` grid to use in creating the field.

        name: `str`
            The name to give the `ESMF` Field.

    :Returns:

        `ESMF.Field`
            The `ESMF` Field for use as a source or destination Field
            in regridding.

    """
    return ESMF.Field(grid, name)


def run_Regrid(regrid, srcfield, dstfield):
    """Call an `ESMF.Regrid` instance to perform regridding.

    :Parameters:

        regrid: `ESMF.Regrid`
            The `ESMF` regrid operator

        srcfield: `ESMF.Field`
            The source Field with an associated grid to be used for
            regridding.

        dstfield: `ESMF.Field`
            The destination Field with an associated grid to be used
            for regridding.

    :Returns:

        `ESMF.Field`
            The regridded Field

    """
    return regrid(srcfield, dstfield, zero_region=ESMF.Region.SELECT)


def regrid_create_operator(regrid, name, parameters):
    """Create a new `RegridOperator` instance.

    :Parameters:

        regrid: `ESMF.Regrid`
            The `ESMF` regridding operator between two fields.

        name: `str`
            A descriptive name for the operator.

        parameters: `dict`
            Parameters that describe the complete coordinate system of
            the destination grid.

    :Returns:

        `RegridOperator`
            The new regrid operator.

    """
    return RegridOperator(regrid, name, **parameters)


def regrid_get_operator_method(operator, method):
    """Return the regridding method of a regridding operator.

    :Parameters:

        operator: `RegridOperator`
            The regridding operator.

        method: `str` or `None`
            A regridding method. If `None` then ignored. If a `str`
            then an exception is raised if it not equivalent to the
            regridding operator's method.

    :Returns:

        `string`
            The operator's regridding method.

    """
    if method is None:
        method = operator.method
    elif not operator.check_method(method):
        raise ValueError(
            f"Method {method!r} does not match the method of the "
            f"regridding operator: {operator.method!r}"
        )

    return method
