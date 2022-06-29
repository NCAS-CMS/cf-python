from copy import deepcopy


class RegridOperator:
    """A regridding operator between two grids.

    Regridding is the process of interpolating from one grid
    resolution to a different grid resolution.

    The regridding operator stores the regridding weights, auxiliary
    information, such as the grid shapes, the CF metadata for the
    destination grid, and the source grid coordinates.

    """

    def __init__(
        self,
        weights,
        row,
        col,
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
        parameters=None,
    ):
        """**Initialization**

        :Parameters:

            weights: array_like
                The 1-d array of the regridding weights.

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
                is the reference source grid mask that was used during
                the creation of the *weights*. If *src_mask* is a
                scalar array with value `False`, then this is
                equivalent to a source grid mask with shape
                *src_shape* entirely populated with `False`.
    
                If `None` (the default), then the weights are assumed
                to have been created assuming no source grid mask.

            dst_mask: `numpy.ndarray` or `None`, optional
                A destination grid mask to be applied to the weights
                matrix, in addition to those destination grid cells
                that have no non-zero weights. If `None` (the default)
                then no additional destination grid cells are
                masked. If a Boolean `numpy` array then it must have
                shape *dst_shape*, and `True` signifies as masked
                cell.

            start_index: `int`, optional
                Specify whether the *row* and *col* parameters use 0-
                or 1-based indexing. By default 0-based indexing is
                used.

            parameters: `dict`, optional
                Parameters that describe the CF metadata for the
                destination grid.

                Any parameter names and values are allowed, and it is
                assumed that the these are well defined during the
                creation and subsequent use of a `RegridOperator`
                instance.

        """
        self._weights = weights
        self._row = row
        self._col = col
        self._coord_sys = coord_sys
        self._method = method
        self._src_shape = tuple(src_shape)
        self._dst_shape = tuple(dst_shape)
        self._src_cyclic = bool(src_cyclic)
        self._dst_cyclic = bool(dst_cyclic)
        self._src_mask = src_mask
        self._dst_mask = dst_mask
        self._src_coords = src_coords
        self._src_bounds = src_bounds
        self._start_index = int(start_index)

        if parameters is None:
            self._parameters = {}
        else:
            self._parameters = parameters.copy()

    def __repr__(self):
        return (
            f"<CF {self.__class__.__name__}: {self.coord_sys} {self.method}>"
        )

    @property
    def col(self):
        """The 1-d array of the column indices of the regridding weights.

        """
        return self._col

    @property
    def coord_sys(self):
        """The name of the coordinate system.

        """
        return self._coord_sys

    @property
    def dst_cyclic(self):
        """Whether or not the destination grid longitude axis is cyclic.

        """
        return self._dst_cyclic

    @property
    def dst_mask(self):
        """A destination grid mask to be applied to the weights matrix.
        """
        return self._dst_mask

    @property
    def dst_shape(self):
        """The shape of the destination grid.

        """
        return self._dst_shape

    @property
    def method(self):
        """The name of the regridding method.

        """
        return self._method

    @property
    def row(self):
        """The 1-d array of the row indices of the regridding weights.

        """
        return self._row

    @property
    def src_bounds(self):
        """TODODASK
        """
        return self._src_bounds

    @property
    def src_coords(self):
        """TODODASK
        """
        return self._src_coords

    @property
    def src_cyclic(self):
        """Whether or not the source grid longitude axis is cyclic.

        """
        return self._src_cylcic

    @property
    def src_mask(self):
        """The source grid mask that was applied during the weights creation.

        """
        return self._src_mask

    @property
    def src_shape(self):
        """The shape of the source grid.
        """
        return self._src_shape

    @property
    def start_index(self):
        """The start index of the row and column indices.

        """
        return self._start_index

    @property
    def weights(self):
        """The 1-d array of the regridding weights.

        """
        return self._weights

    def copy(self):
        """Return a deep copy.

        :Returns:

            `RegridOperator`
                The deep copy.

        """
        return type(self)(
            self.weights.copy(),
            self.row.copy(),
            self.col.copy(),
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
            parameters=deepcopy(parameters),
        )

    def get_parameter(self, parameter, *default):
        """Return a regrid operation parameter.

        :Parameters:
    
            parameter: `str`
                The name of the parameter.
            
            default: optional
                Return the value of the *default* parameter if the
                parameter has not been set.
            
                If set to an `Exception` instance then it will be
                raised instead.

                .. versionadded:: TODODASK

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
        try:
            return self._parameters[parameter]
        except KeyError:
            if default:
                return default[0]

            raise ValueError(
                f"{self.__class__.__name__} has no {parameter!r} parameter"
            )

    def parameters(self):
        """Parameters that describe the CF metadata for the destination grid.
    
        Any parameter names and values are allowed, and it is assumed
        that the these are well defined during the creation and
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
        return self._parameters.copy()

    def todense(self, order="C"):
        """Return the weights in dense format.
        
        .. versionadded:: TODODASK

        .. seealso:: `tosparse`

        :Parameters:

            order: `str`, optional
                Specify the memory layout of the returned weights
                matrix. ``'C'`` (the default) means C order
                (row-major), ``'F'`` means Fortran (column-major)
                order.

        :Returns:
        
            `numpy.ndarray`
                The full array of weights, with zeros at locations not
                The sparse array of weightsdefined by `row` and `col`.

        """
        return self.sparse_array().todense(order=order)

    def tosparse(self):
        """Return the weights in sparse COOrdinate format.
        
        .. versionadded:: TODODASK

        .. seealso:: `todense`

        :Returns:
        
            `scipy.sparse._arrays.coo_array`
                The sparse array of weights.

        """
        from math import prod
        from scipy.sparse import coo_array

        row = self.row
        col = self.col
        start_index = self.start_index
        if start_index:
            row = row - start_index
            col = col - start_index

        src_size = prod(self.src_shape)
        dst_size = prod(self.dst_shape)

        return coo_array(
            (self.weights, (row, col)), shape=[dst_size, src_size]
        )
