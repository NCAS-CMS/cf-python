from copy import deepcopy

import numpy as np
from cfdm import Container

from ..decorators import _display_or_return
from ..functions import _DEPRECATION_ERROR_ATTRIBUTE, _DEPRECATION_ERROR_METHOD
from ..mixin_container import Container as mixin_Container


class RegridOperator(mixin_Container, Container):
    """A regridding operator between two grids.

    Regridding is the process of interpolating from one grid
    resolution to a different grid resolution.

    The regridding operator stores the regridding weights; auxiliary
    information, such as the grid shapes; the CF metadata for the
    destination grid; and the source grid coordinates.

    .. versionadded:: 3.10.0

    """

    def __init__(
        self,
        weights=None,
        row=None,
        col=None,
        coord_sys=None,
        method=None,
        src_shape=None,
        dst_shape=None,
        src_cyclic=None,
        dst_cyclic=None,
        src_mask=None,
        dst_mask=None,
        src_coords=None,
        src_bounds=None,
        start_index=0,
        src_axes=None,
        dst_axes=None,
        dst=None,
        weights_file=None,
        src_mesh_location=None,
        dst_mesh_location=None,
        src_featureType=None,
        dst_featureType=None,
        dimensionality=None,
        src_z=None,
        dst_z=None,
        ln_z=False,
    ):
        """**Initialisation**

        :Parameters:

            weights: array_like
                The 1-d array of regridding weights for locations in
                the 2-d dense weights matrix. The locations are defined
                by the *row* and *col* parameters.

            row, col: array_like, array_like
                The 1-d arrays of the row and column indices of the
                regridding weights in the dense weights matrix, which
                has J rows and I columns, where J and I are the total
                number of cells in the destination and source grids
                respectively. See the *start_index* parameter.

            coord_sys: `str`
                The name of the coordinate system of the source and
                destination grids. Either ``'spherical'`` or
                ``'Cartesian'``.

            method: `str`
                The name of the regridding method.

            src_shape: sequence of `int`
                The shape of the source grid.

            dst_shape: sequence of `int`
                The shape of the destination grid.

            src_cyclic: `bool`
                For spherical regridding, specifies whether or not the
                source grid longitude axis is cyclic.

            dst_cyclic: `bool`
                For spherical regridding, specifies whether or not the
                destination grid longitude axis is cyclic.

            src_mask: `numpy.ndarray` or `None`, optional
                If a `numpy.ndarray` with shape *src_shape* then this
                is the source grid mask that was used during the
                creation of the *weights*. If *src_mask* is a scalar
                array with value `False`, then this is equivalent to a
                source grid mask with shape *src_shape* entirely
                populated with `False`.

                If `None` (the default), then the weights are assumed
                to have been created assuming no source grid masked
                cells.

            dst_mask: `numpy.ndarray` or `None`, optional
                A destination grid mask to be applied to the weights
                matrix, in addition to those destination grid cells
                that have no non-zero weights. If `None` (the default)
                then no additional destination grid cells are
                masked. If a Boolean `numpy` array then it must have
                shape *dst_shape*, and a value of `True` signifies a
                masked destination grid cell.

            start_index: `int`, optional
                Specify whether the *row* and *col* parameters use 0-
                or 1-based indexing. By default 0-based indexing is
                used.

                If *row* and *col* are to be read from a weights file
                and their netCDF variables have ``start_index``
                attributes, then these will be used in preference to
                *start_index*.

            parameters: Deprecated at version 3.14.0
                Use keyword parameters instead.

            dst: `Field` or `Domain`
                The definition of the destination grid.

            dst_axes: `dict` or sequence or `None`, optional
                The destination grid axes to be regridded.

            src_axes: `dict` or sequence or `None`, optional
                The source grid axes to be regridded.

            weights_file: `str` or `None`, optional
                 Path to a netCDF file that contained the regridding
                 weights. If `None`, the default, then the weights
                 were computed rather than read from a file.

                 .. versionadded:: 3.15.2

            src_mesh_location: `str`, optional
                The UGRID mesh element of the source grid
                (e.g. ``'face'``).

                .. versionadded:: 3.16.0

            dst_mesh_location: `str`, optional
                The UGRID mesh element of the destination grid
                (e.g. ``'face'``).

                .. versionadded:: 3.16.2

            src_featureType: `str`, optional
                The discrete sampling geometry (DSG) featureType of
                the source grid (e.g. ``'trajectory'``).

                .. versionadded:: 3.16.2

            dst_featureType: `str`, optional
                The DSG featureType of the destination grid
                (e.g. ``'trajectory'``).

                .. versionadded:: 3.16.2

            src_z: optional
                The identity of the source grid vertical coordinates
                used to calculate the weights. If `None` then no
                source grid vertical axis is identified.

                .. versionadded:: 3.16.2

            dst_z: optional
                The identity of the destination grid vertical
                coordinates used to calculate the weights. If `None`
                then no destination grid vertical axis is identified.

                .. versionadded:: 3.16.2

            ln_z: `bool`, optional
                Whether or not the weights were calculated with the
                natural logarithm of vertical coordinates.

                .. versionadded:: 3.16.2

            dimensionality: `int`, optional
                The number of physical regridding dimensions. This may
                differ from the corresponding number of storage
                dimensions in the source or destination grids, if
                either has an unstructured mesh or a DSG featureType.

                .. versionadded:: 3.16.2

        """
        super().__init__()

        if weights is None and weights_file is None:
            # This to allow a no-arg init!
            return

        self._set_component("weights", weights, copy=False)
        self._set_component("row", row, copy=False)
        self._set_component("col", col, copy=False)
        self._set_component("coord_sys", coord_sys, copy=False)
        self._set_component("method", method, copy=False)
        self._set_component("src_mask", src_mask, copy=False)
        self._set_component("dst_mask", dst_mask, copy=False)
        self._set_component("src_cyclic", bool(src_cyclic), copy=False)
        self._set_component("dst_cyclic", bool(dst_cyclic), copy=False)
        self._set_component("src_shape", tuple(src_shape), copy=False)
        self._set_component("dst_shape", tuple(dst_shape), copy=False)
        self._set_component("src_coords", tuple(src_coords), copy=False)
        self._set_component("src_bounds", tuple(src_bounds), copy=False)
        self._set_component("start_index", int(start_index), copy=False)
        self._set_component("src_axes", src_axes, copy=False)
        self._set_component("dst_axes", dst_axes, copy=False)
        self._set_component("dst", dst, copy=False)
        self._set_component("weights_file", weights_file, copy=False)
        self._set_component("src_mesh_location", src_mesh_location, copy=False)
        self._set_component("dst_mesh_location", dst_mesh_location, copy=False)
        self._set_component("src_featureType", src_featureType, copy=False)
        self._set_component("dst_featureType", dst_featureType, copy=False)
        self._set_component("dimensionality", dimensionality, copy=False)
        self._set_component("src_z", src_z, copy=False)
        self._set_component("dst_z", dst_z, copy=False)
        self._set_component("ln_z", bool(ln_z), copy=False)

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        return (
            f"<CF {self.__class__.__name__}: {self.coord_sys} {self.method}>"
        )

    @property
    def col(self):
        """The 1-d array of the column indices of the regridding
        weights.

        .. versionadded:: 3.14.0

        """
        return self._get_component("col")

    @property
    def coord_sys(self):
        """The name of the regridding coordinate system.

        .. versionadded:: 3.14.0

        """
        return self._get_component("coord_sys")

    @property
    def dimensionality(self):
        """The number of physical regridding dimensions.

        .. versionadded:: 3.16.2

        """
        return self._get_component("dimensionality")

    @property
    def dst(self):
        """The definition of the destination grid.

        Either a `Field` or` Domain`.

        .. versionadded:: 3.14.0

        """
        return self._get_component("dst")

    @property
    def dst_axes(self):
        """The destination grid axes to be regridded.

        If not required then this will be `None`.

        .. versionadded:: 3.14.0

        """
        return self._get_component("dst_axes")

    @property
    def dst_cyclic(self):
        """Whether or not the destination grid longitude axis is
        cyclic."""
        return self._get_component("dst_cyclic")

    @property
    def dst_featureType(self):
        """The DSG featureType of the destination grid.

        .. versionadded:: 3.16.2

        """
        return self._get_component("dst_featureType")

    @property
    def dst_mask(self):
        """A destination grid mask to be applied to the weights matrix.

        If `None` then no additional destination grid cells are
        masked.

        If a Boolean `numpy` array then it is required that this mask
        is applied to the weights matrix prior to use in a regridding
        operation. The mask must have shape `!dst_shape`, and a value
        of `True` signifies a masked destination grid cell.

        .. versionadded:: 3.14.0

        """
        return self._get_component("dst_mask")

    @property
    def dst_mesh_location(self):
        """The UGRID mesh element of the destination grid.

        .. versionadded:: 3.16.0

        """
        return self._get_component("dst_mesh_location")

    @property
    def dst_shape(self):
        """The shape of the destination grid.

        .. versionadded:: 3.14.0

        """
        return self._get_component("dst_shape")

    @property
    def dst_z(self):
        """The identity of the destination grid vertical coordinates.

        .. versionadded:: 3.16.2

        """
        return self._get_component("dst_z")

    @property
    def ln_z(self):
        """Whether or not vertical weights are based on ln(z).

        .. versionadded:: 3.16.2

        """
        return self._get_component("ln_z")

    @property
    def method(self):
        """The name of the regridding method.

        .. versionadded:: 3.14.0

        """
        return self._get_component("method")

    @property
    def name(self):
        """The name of the regridding method."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "name",
            version="3.14.0",
            removed_at="5.0.0",
        )

    @property
    def row(self):
        """The 1-d array of the row indices of the regridding weights.

        .. versionadded:: 3.14.0

        """
        return self._get_component("row")

    @property
    def src_axes(self):
        """The source grid axes to be regridded.

        If not required then this will be `None`.


        .. versionadded:: 3.14.0

        """
        return self._get_component("src_axes")

    @property
    def src_bounds(self):
        """The bounds of the source grid cells.

        .. versionadded:: 3.14.0

        """
        return self._get_component("src_bounds")

    @property
    def src_coords(self):
        """The coordinates of the source grid cells.

        .. versionadded:: 3.14.0

        """
        return self._get_component("src_coords")

    @property
    def src_cyclic(self):
        """Whether or not the source grid longitude axis is cyclic.

        .. versionadded:: 3.14.0

        """
        return self._get_component("src_cyclic")

    @property
    def src_featureType(self):
        """The DSG featureType of the source grid.

        .. versionadded:: 3.16.2

        """
        return self._get_component("src_featureType")

    @property
    def src_mask(self):
        """The source grid mask that was applied during the weights
        creation.

        If `None` then the weights are assumed to have been created
        assuming no source grid masked cells.

        If a Boolean `numpy.ndarray` with shape `!src_shape` then this
        is the source grid mask that was used during the creation of
        the *weights*.

        .. versionadded:: 3.14.0

        """
        return self._get_component("src_mask")

    @property
    def src_mesh_location(self):
        """The UGRID mesh element of the source grid.

        .. versionadded:: 3.16.0

        """
        return self._get_component("src_mesh_location")

    @property
    def src_shape(self):
        """The shape of the source grid.

        .. versionadded:: 3.14.0

        """
        return self._get_component("src_shape")

    @property
    def src_z(self):
        """The identity of the source grid vertical coordinates.

        .. versionadded:: 3.16.2

        """
        return self._get_component("src_z")

    @property
    def start_index(self):
        """The start index of the row and column indices.

        If `row` and `col` are to be read from a weights file and
        their netCDF variables have ``start_index`` attributes, then
        these will be used in preference to `start_index`.

        .. versionadded:: 3.14.0

        """
        return self._get_component("start_index")

    @property
    def weights(self):
        """The 1-d array of the regridding weights, or `None`.

        If and only if it is a `scipy` sparse array that combines the
        weights and the row and column indices (as opposed to a
        `numpy` array of just the weights) then the `dst_mask`
        attribute will have been updated to `True` for destination
        grid points for which the weights are all zero.

        .. versionadded:: 3.14.0

        """
        return self._get_component("weights")

    @property
    def weights_file(self):
        """The file which contains the weights, or `None`.

        .. versionadded:: 3.15.2

        """
        return self._get_component("weights_file")

    def copy(self):
        """Return a deep copy.

        :Returns:

            `RegridOperator`
                The deep copy.

        """
        row = self.row
        if row is not None:
            row = row.copy()

        col = self.col
        if col is not None:
            col = col.copy()

        return type(self)(
            self.weights.copy(),
            row,
            col,
            method=self.method,
            src_shape=self.src_shape,
            dst_shape=self.dst_shape,
            src_cyclic=self.src_cyclic,
            dst_cyclic=self.dst_cyclic,
            src_mask=deepcopy(self.src_mask),
            dst_mask=deepcopy(self.dst_mask),
            src_coords=deepcopy(self.src_coords),
            src_bounds=deepcopy(self.src_bounds),
            coord_sys=self.coord_sys,
            start_index=self.start_index,
            src_axes=self.src_axes,
            dst_axes=self.dst_axes,
            dst=self.dst.copy(),
            weights_file=self.weights_file,
            src_mesh_location=self.src_mesh_location,
            dst_mesh_location=self.dst_mesh_location,
            src_featureType=self.src_featureType,
            dst_featureType=self.dst_featureType,
            src_z=self.src_z,
            dst_z=self.dst_z,
            ln_z=self.ln_z,
        )

    @_display_or_return
    def dump(self, display=True):
        """A full description of the regrid operator.

        Returns a description of all properties, including
        the weights and their row and column indices.

        .. versionadded:: 3.14.0

        :Parameters:

            display: `bool`, optional
                If False then return the description as a string. By
                default the description is printed.

        :Returns:

            {{returns dump}}

        """
        _title = repr(self)[4:-1]
        line = f"{''.ljust(len(_title), '-')}"
        string = [line, _title, line]

        for attr in (
            "coord_sys",
            "method",
            "dimensionality",
            "src_shape",
            "dst_shape",
            "src_cyclic",
            "dst_cyclic",
            "src_mask",
            "dst_mask",
            "src_coords",
            "src_bounds",
            "start_index",
            "src_axes",
            "dst_axes",
            "src_mesh_location",
            "dst_mesh_location",
            "src_featureType",
            "dst_featureType",
            "src_z",
            "dst_z",
            "ln_z",
            "dst",
            "weights",
            "row",
            "col",
            "weights_file",
        ):
            string.append(f"{attr}: {getattr(self, attr)!r}")

        return "\n".join(string)

    def get_parameter(self, parameter, *default):
        """Return a regrid operation parameter.

        Deprecated at version 3.14.0.

        :Parameters:

            parameter: `str`
                The name of the parameter.

            default: optional
                Return the value of the *default* parameter if the
                parameter has not been set.

                If set to an `Exception` instance then it will be
                raised instead.

                .. versionadded:: 3.14.0

        :Returns:

            The value of the named parameter or the default value, if
            set.

        **Examples**

        >>> r.get_parameter('dst_axes')
        ['domainaxis1', 'domainaxis0']
        >>> r.get_parameter('x')
        Traceback
            ...
        ValueError: RegridOperator has no 'x' parameter
        >>> r.get_parameter('x', 'missing')
        'missing'

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "get_parameter",
            message="Use attributes directly.",
            version="3.14.0",
            removed_at="5.0.0",
        )

    def parameters(self):
        """Get the CF metadata parameters for the destination grid.

        Deprecated at version 3.14.0.

        Any parameter names and values are allowed, and it is assumed
        that these are well defined during the creation and
        subsequent use of a `RegridOperator` instance.

        :Returns:

            `dict`
                The parameters.

        **Examples**

        >>> r.parameters()
        {'dst': <CF Domain: {latitude(5), longitude(8), time(1)}>,
         'dst_axes': ['domainaxis1', 'domainaxis2'],
         'src_axes': None}

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "parameters",
            version="3.14.0",
            removed_at="5.0.0",
        )

    def tosparse(self):
        """Convert the weights to `scipy` sparse array format in-place.

        The `weights` attribute is set to a Compressed Sparse Row
        (CSR) array (i.e. a `scipy.sparse._arrays.csr_array` instance)
        that combines the weights and the row and column indices, and
        the `row` and `col` attributes are set to `None`.

        The `dst_mask` attribute is also updated to `True` for
        destination grid points for which the weights are all zero.

        A CSR array is used as the most efficient sparse array type
        given that we expect no changes to the sparsity structure, and
        any further modification of the weights to account for missing
        values in the source grid will always involve row-slicing.

        If the weights are already in a sparse array format then no
        action is taken.

        .. versionadded:: 3.13.0

        :Returns:

            `None`

        """
        weights = self.weights
        row = self.row
        col = self.col
        if weights is not None and row is None and col is None:
            # Weights are already in sparse array format
            return

        from math import prod

        from scipy.sparse import csr_array

        start_index = self.start_index
        col_start_index = None
        row_start_index = None

        if weights is None:
            weights_file = self.weights_file
            if weights_file is not None:
                # Read the weights from the weights file
                from netCDF4 import Dataset

                from ..data.array.netcdfarray import _lock

                _lock.acquire()
                nc = Dataset(weights_file, "r")
                weights = nc.variables["S"][...]
                row = nc.variables["row"][...]
                col = nc.variables["col"][...]

                try:
                    col_start_index = nc.variables["col"].start_index
                except AttributeError:
                    col_start_index = 1

                try:
                    row_start_index = nc.variables["row"].start_index
                except AttributeError:
                    row_start_index = 1

                nc.close()
                _lock.release()
            else:
                raise ValueError(
                    "Conversion to sparse array format requires at least "
                    "one of the 'weights' or 'weights_file' attributes to "
                    "be set"
                )

        # Convert to sparse array format
        if col_start_index:
            col = col - col_start_index
        elif start_index:
            col = col - start_index

        if row_start_index:
            row = row - row_start_index
        elif start_index:
            row = row - start_index

        src_size = prod(self.src_shape)
        dst_size = prod(self.dst_shape)

        weights = csr_array((weights, (row, col)), shape=[dst_size, src_size])

        self._set_component("weights", weights, copy=False)
        self._set_component("row", None, copy=False)
        self._set_component("col", None, copy=False)
        del row, col

        # Set the destination grid mask to True where the weights for
        # destination grid points are all zero
        dst_mask = self.dst_mask
        if dst_mask is not None:
            if dst_mask.dtype != bool or dst_mask.shape != self.dst_shape:
                raise ValueError(
                    f"The {self.__class__.__name__}.dst_mask attribute must "
                    "be None or a Boolean numpy array with shape "
                    f"{self.dst_shape}. Got: dtype={dst_mask.dtype}, "
                    f"shape={dst_mask.shape}"
                )

            dst_mask = np.array(dst_mask).reshape((dst_size,))
        else:
            dst_mask = np.zeros((dst_size,), dtype=bool)

        # Performance note:
        #
        # It is much more efficient to access 'weights.indptr' and
        # 'weights.data' directly, rather than iterating over rows of
        # 'weights' and using 'weights.getrow'.

        indptr = weights.indptr.tolist()
        data = weights.data
        for j, (i0, i1) in enumerate(zip(indptr[:-1], indptr[1:])):
            if not data[i0:i1].size:
                dst_mask[j] = True

        if not dst_mask.any():
            dst_mask = None
        else:
            dst_mask = dst_mask.reshape(self.dst_shape)

        self._set_component("dst_mask", dst_mask, copy=False)
