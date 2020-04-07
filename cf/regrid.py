# -*- coding: utf-8 -*-
from numpy import array as numpy_array
from numpy import empty as numpy_empty
from numpy import where as numpy_where
from numpy import sum as numpy_sum
from numpy import finfo as numpy_finfo
from .data.data import Data
from .dimensioncoordinate import DimensionCoordinate
from .functions import REGRID_LOGGING
from . import _found_ESMF
if _found_ESMF:
    try:
        import ESMF
    except Exception as error:
        print("WARNING: Can not import ESMF for regridding: {0}".format(error))


class Regrid:
    '''Class containing all the methods required for accessing ESMF
    regridding through ESMPY and the associated utility methods.

    '''

    def __init__(self, srcfield, dstfield, srcfracfield, dstfracfield,
                 method='conservative_1st', ignore_degenerate=False):
        '''Creates a handle for regridding fields from a source grid to a
    destination grid that can then be used by the run_regridding method.

    :Parameters:

        srcfield: ESMF.Field
            The source field with an associated grid to be used for
            regridding.

        dstfield: ESMF.Field
            The destination field with an associated grid to be used
            for regridding.

        srcfracfield: ESMF.Field
            A field to hold the fraction of the source field that
            contributes to conservative regridding.

        dstfracfield: ESMF.Field
            A field to hold the fraction of the source field that
            contributes to conservative regridding.

        method: `str`, optional
            By default the regridding method is set to
            'conservative_1st'. In this case or if it is set to
            'conservative' first-order conservative regridding is
            used. If it is set to 'conservative_2nd' second order
            conservative regridding is used. If it is set to
            'linear' then (multi)linear interpolation is used.  If
            it is set to 'patch' then higher-order patch recovery is
            used.  If it is set to 'nearest_stod' then nearest source
            to destination interpolation is used. If it is set to
            'nearest_dtos' then nearest destination to source
            interpolation is used.

        ignore_degenerate: `bool`, optional
            Whether to check for degenerate points.

        '''
        # create a handle to the regridding method
        regrid_method_map = {
            'linear': ESMF.RegridMethod.BILINEAR,  # see comment below...
            'bilinear': ESMF.RegridMethod.BILINEAR,  # (for back compat)
            'conservative': ESMF.RegridMethod.CONSERVE,
            'conservative_1st': ESMF.RegridMethod.CONSERVE,
            'conservative_2nd': ESMF.RegridMethod.CONSERVE_2ND,
            'nearest_dtos': ESMF.RegridMethod.NEAREST_DTOS,
            'nearest_stod': ESMF.RegridMethod.NEAREST_STOD,
            'patch': ESMF.RegridMethod.PATCH,
        }
        # ... diverge from ESMF with respect to name for bilinear method by
        # using 'linear' because 'bi' implies 2D linear interpolation, which
        # could mislead or confuse for Cartesian regridding in 1D or 3D.
        regrid_method = regrid_method_map.get(
            method, ValueError('Regrid method not recognised.'))
        # Initialise the regridder. This also creates the
        # weights needed for the regridding.
        self.regridSrc2Dst = ESMF.Regrid(
            srcfield, dstfield, regrid_method=regrid_method,
            src_mask_values=numpy_array([0], dtype='int32'),
            dst_mask_values=numpy_array([0], dtype='int32'),
            src_frac_field=srcfracfield, dst_frac_field=dstfracfield,
            unmapped_action=ESMF.UnmappedAction.IGNORE,
            ignore_degenerate=ignore_degenerate)

    def destroy(self):
        '''Free the memory associated with the ESMF.Regrid instance.

        '''
        self.regridSrc2Dst.destroy()

    @staticmethod
    def initialize():
        '''Check whether ESMF has been found. If not raise an import
    error. Initialise the ESMPy manager. Whether logging is enabled or
    not is determined by cf.REGRID_LOGGING. If it is then logging
    takes place after every call to ESMPy.

    :Returns:

        `ESMF.Manager`
            A singleton instance of the ESMPy manager.

        '''
        if not _found_ESMF:
            raise ImportError(
                'The ESMF package is needed to support regridding.')

        manager = ESMF.Manager(debug=REGRID_LOGGING())

        return manager

    @staticmethod
    def create_grid(coords, use_bounds, mask=None, cartesian=False,
                    cyclic=False, coords_2D=False, coord_order=None):
        '''Create an ESMPy grid given a sequence of coordinates for use as a
    source or destination grid in regridding. Optionally the grid may
    have an associated mask.

    :Parameters:

        coords: sequence
            The coordinates if not Cartesian it is assume that the
            first is longitude and the second is latitude.

        use_bounds: `bool`
            Whether to populate the grid corners with information from
            the bounds or not.

        mask: `numpy.ndarray`, optional

            An optional numpy array of booleans containing the grid
            points to mask.  Where the elements of mask are True the
            output grid is masked.

        cartesian: `bool`, optional
            Whether to create a Cartesian grid or a spherical one,
            False by default.

        cyclic: `bool`, optional
            Whether or not the longitude (if present) is cyclic. If
            None the a check for cyclicity is made from the
            bounds. None by default.

        coords_2D: `bool`, optional
            Whether the coordinates are 2D or not. Presently only
            works for spherical coordinates. False by default.

        coord_order: sequence, optional
            Two tuples one indicating the order of the x and y axes
            for 2D longitude, one for 2D latitude.

    :Returns:

        `ESMF.Grid`
            The resulting ESMPy grid for use as a source or destination
            grid in regridding.

        '''

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
                        'The longitude and latitude coordinates'
                        ' must have the same shape.'
                    )
            # --- End: if

            if use_bounds:
                if not coords_2D:
                    # Get the bounds
                    x_bounds = lon.get_bounds()
                    y_bounds = lat.get_bounds().clip(-90, 90, 'degrees').array

                    # If cyclic not set already, check for cyclicity
                    if cyclic is None:
                        cyclic = abs(x_bounds.datum(-1)
                                     - x_bounds.datum(0)) == Data(360,
                                                                  'degrees')

                    x_bounds = x_bounds.array
                else:
                    # Get the bounds
                    x_bounds = lon.get_bounds()
                    y_bounds = lat.get_bounds().clip(-90, 90, 'degrees')
                    n = x_bounds.shape[0]
                    m = x_bounds.shape[1]
                    x_bounds = x_bounds.array
                    y_bounds = y_bounds.array

                    tmp_x = numpy_empty((n + 1, m + 1))
                    tmp_x[:n, :m] = x_bounds[:, :, 0]
                    tmp_x[:n, m] = x_bounds[:, -1, 1]
                    tmp_x[n, :m] = x_bounds[-1, :, 3]
                    tmp_x[n, m] = x_bounds[-1, -1, 2]

                    tmp_y = numpy_empty((n + 1, m + 1))
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
                            cyclic = abs(x_bounds.datum(-1) -
                                         x_bounds.datum(0)) == Data(
                                             360, 'degrees')
                        except ValueError:
                            pass
            # --- End: if

            # Create empty grid
            max_index = numpy_array(shape, dtype='int32')
            if use_bounds:
                staggerLocs = [ESMF.StaggerLoc.CORNER, ESMF.StaggerLoc.CENTER]
            else:
                staggerLocs = [ESMF.StaggerLoc.CENTER]

            if cyclic:
                grid = ESMF.Grid(max_index, num_peri_dims=1,
                                 staggerloc=staggerLocs)
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
                        gridCorner[x][...] = x_bounds[:, 0].reshape(
                            lon.size, 1)
                    else:
                        n = x_bounds.shape[0]
                        tmp_x = numpy_empty(n + 1)
                        tmp_x[:n] = x_bounds[:, 0]
                        tmp_x[n] = x_bounds[-1, 1]
                        gridCorner[x][...] = tmp_x.reshape(lon.size + 1, 1)

                    n = y_bounds.shape[0]
                    tmp_y = numpy_empty(n + 1)
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
            # --- End: if
        else:
            # Test the dimensionality of the list of coordinates
            ndim = len(coords)
            if ndim < 1 or ndim > 3:
                raise ValueError(
                    'Cartesian grid must have between 1 and 3 dimensions.'
                )

            # For 1D conservative regridding add an extra dimension of size 1
            if ndim == 1:
                if not use_bounds:
                    # For 1D nonconservative regridding the extra dimension
                    # should already have been added in cf.Field.regridc.
                    raise ValueError(
                        'Cannot create a Cartesian grid from '
                        'one dimension coordinate with no bounds.'
                    )
                coords = [DimensionCoordinate(data=Data(0),
                          bounds=Data([numpy_finfo('float32').epsneg,
                                       numpy_finfo('float32').eps]))] + coords
                if mask is not None:
                    mask = mask[None, :]
                ndim = 2

            shape = list()
            for coord in coords:
                shape.append(coord.size)

            # Initialise the grid
            max_index = numpy_array(shape, dtype='int32')
            if use_bounds:
                if ndim < 3:
                    staggerLocs = [ESMF.StaggerLoc.CORNER,
                                   ESMF.StaggerLoc.CENTER]
                else:
                    staggerLocs = [ESMF.StaggerLoc.CENTER_VCENTER,
                                   ESMF.StaggerLoc.CORNER_VFACE]
            else:
                if ndim < 3:
                    staggerLocs = [ESMF.StaggerLoc.CENTER]
                else:
                    staggerLocs = [ESMF.StaggerLoc.CENTER_VCENTER]
            # --- End: if
            grid = ESMF.Grid(max_index, coord_sys=ESMF.CoordSys.CART,
                             staggerloc=staggerLocs)

            # Populate the grid centres
            for d in range(0, ndim):
                if ndim < 3:
                    gridCentre = grid.get_coords(
                        d, staggerloc=ESMF.StaggerLoc.CENTER)
                else:
                    gridCentre = grid.get_coords(
                        d, staggerloc=ESMF.StaggerLoc.CENTER_VCENTER)
                gridCentre[...] = coords[d].array.reshape(
                    [shape[d] if x == d else 1 for x in range(0, ndim)])
            # --- End: for

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
                        tmp = numpy_empty(shape[d] + 1)
                        tmp[0:-1] = boundsD[:, 0]
                        tmp[-1] = boundsD[-1, 1]
                        boundsD = tmp

                    gridCorner[d][...] = boundsD.reshape(
                        [shape[d] + 1 if x == d else 1
                         for x in range(0, ndim)])
            # --- End: if
        # --- End: if

        # Add the mask if appropriate
        if mask is not None:
            gmask = grid.add_item(ESMF.GridItem.MASK)
            gmask[...] = 1
            gmask[mask] = 0

        return grid

    @staticmethod
    def create_field(grid, name):
        '''Create an ESMPy field for use as a source or destination field in
    regridding given an ESMPy grid and a name.

    :Parameters:

        grid: ESMF.Grid
            The ESMPy grid to use in creating the field.

        name: `str`
            The name to give the field.

    :Returns:

        `ESMF.Field`
            The resulting ESMPy field for use as a source or
            destination field in regridding.

        '''
        field = ESMF.Field(grid, name)
        return field

    def run_regridding(self, srcfield, dstfield):
        '''

        '''
        dstfield = self.regridSrc2Dst(srcfield, dstfield,
                                      zero_region=ESMF.Region.SELECT)
        return dstfield

    @staticmethod
    def concatenate_data(data_list, axis):
        '''Concatenates a list of Data objects into a single Data object along
    the specified access (see cf.Data.concatenate for details). In the
    case that the list contains only one element, that element is
    simply returned.

    :Parameters:

        data_list: `list`
            The list of data objects to concatenate.

        axis: `int`
            The axis along which to perform the concatenation.

    :Returns:

        `Data`
            The resulting single Data object.

        '''
        if len(data_list) > 1:
            data = Data.concatenate(data_list, axis=axis)
            if data.fits_in_one_chunk_in_memory(data.dtype.itemsize):
                data.varray
            return data
        else:
            assert len(data_list) == 1
            return data_list[0]

    @staticmethod
    def reconstruct_sectioned_data(sections):
        '''Expects a dictionary of Data objects with ordering information as
    keys, as output by the section method when called with a Data
    object. Returns a reconstructed cf.Data object with the sections
    in the original order.

    :Parameters:

        sections: `dict`
            The dictionary or Data objects with ordering information
            as keys.

    :Returns:

        `Data`
            The resulting reconstructed Data object.

        '''
        ndims = len(sections.keys()[0])
        for i in range(ndims - 1, -1, -1):
            keys = sorted(sections.keys())
            if i == 0:
                if keys[0][i] is None:
                    assert len(keys) == 1
                    return sections.values()[0]
                else:
                    data_list = []
                    for k in keys:
                        data_list.append(sections[k])

                    return Regrid.concatenate_data(data_list, i)

            else:
                if keys[0][i] is None:
                    pass
                else:
                    new_sections = {}
                    new_key = keys[0][:i]
                    data_list = []
                    for k in keys:
                        if k[:i] == new_key:
                            data_list.append(sections[k])
                        else:
                            new_sections[new_key] = Regrid.concatenate_data(
                                data_list, i)
                            new_key = k[:i]
                            data_list = [sections[k]]
                    # --- End: for

                    new_sections[new_key] = Regrid.concatenate_data(
                        data_list, i)
                    sections = new_sections
        # --- End: for

    @staticmethod
    def compute_mass_grid(valuefield, areafield, dofrac=False, fracfield=None,
                          uninitval=422397696.):
        '''Compute the mass of a data field.

    :Parameters:


        valuefield: ESMF.Field
            This contains data values of a field built on the cells of
            a grid.

        areafield: ESMF.Field
            This contains the areas associated with the grid cells.

        fracfield: ESMF.Field
            This contains the fractions of each cell which contributed
            to a regridding operation involving 'valuefield.

        dofrac: `bool`
            This gives the option to not use the 'fracfield'.

        uninitval: `float`
            The value uninitialised cells take.

    :Returns:

        `float`
            The mass of the data field is computed.

        '''
        mass = 0.0
        areafield.get_area()

        ind = numpy_where(valuefield.data != uninitval)

        if dofrac:
            mass = numpy_sum(
                areafield.data[ind] * valuefield.data[ind] *
                fracfield.data[ind]
            )
        else:
            mass = numpy_sum(areafield.data[ind] * valuefield.data[ind])

        return mass


# --- End: class
