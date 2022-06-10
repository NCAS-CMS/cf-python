from copy import deepcopy


class RegridOperator:
    """A regridding operator between two fields.

    TODODASK

    Stores an `ESMF.Regrid` regridding operator and associated
    parameters that describe the complete coordinate system of the
    destination grid.

    """

    def __init__(self, weights, row, col, method=None, src_shape=None,
                 dst_shape=None, src_cyclic=None, dst_cyclic=None,
                 src_mask=None, dst_mask=None, src_coords=None,
                 src_bounds=None, coord_sys=None, parameters=None):
        """**Initialization**

        :Parameters:

            regrid: `ESMF.Regrid`
                The `ESMF` regridding operator between two fields.

            name: `str`, optional
                A name that defines the context of the destination
                grid parameters.

            parameters: `dict`, optional
               Parameters that describe the complete coordinate system
               of the destination grid.

               Any parameter names and values are allowed, and it is
               assumed that the these are well defined during the
               creation and subsequent use of a `RegridOperator`
               instance.

        """
        if parameters is None:
            parameters = {}
            
        self._weights = weights
        self._row = row
        self._col = col
        self._method = method
        self._src_shape = tuple(src_shape)
        self._dst_shape = tuple(dst_shape)
        self._src_cyclic = bool(src_cyclic)
        self._dst_cyclic = bool(dst_cyclic)
        self._src_mask = src_mask
        self._dst_mask = dst_mask
        self._src_coords = tuple(src_coords)
        self._src_bounds = tuple(src_bounds)
        self._coord_sys = coord_sys
        self._parameters = parameters.copy()
    
    def __repr__(self):
        return (
            f"<CF {self.__class__.__name__}: {self.coord_sys} {self.method}>"
        )

    @property
    def coord_sys(self):
        """TODODASK

        """
        return self._coord_sys

    @property
    def dst_cyclic(self):
        """TODODASK
        """
        return self._dst_cyclic

    @property
    def dst_mask(self):
        """TODODASK
        """
        return self._dst_mask

    @property
    def dst_shape(self):
        """TODODASK
        """
        return self._dst_shape

    @property
    def method(self):
        """The regridding method.

        **Examples**

        >>> r.method
        'patch'

        """
        return self._method

    @property
    def parameters(self):
        """The parameters that describe the destination grid.

        Any parameter names and values are allowed, and it is assumed
        that the these are well defined during the creation and
        subsequent use of a `RegridOperator` instance.

        **Examples**

        >>> type(r.parameters)
        dict

        """
        return self._parameters.copy()

    @property
    def quarter(self):
        """TODODASK
        """
        return self._quarter

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
        """TODODASK
        """
        return self._src_cylcic

    @property
    def src_mask(self):
        """TODODASK
        """
        return self._src_mask

    @property
    def src_shape(self):
        """TODODASK
        """
        return self._src_shape

    @property
    def weights(self):
        """The contained regridding operator.

        """
        return self._weights

    def copy(self):
        """Return a deep copy.

        :Returns:

            `RegridOperator`

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
            quarter=self.quarter,
            parameters=deepcopy(parameters),
        )
    
    def todense(self, order="C"):
        """Return the weights in dense format.
        
        .. versionadded:: TODODASK

        .. seealso:: `tosparse`

        """
        return self.sparse_array().todense(order=order)
 
    def get_parameter(self, parameter):
        """Return a regrid operation parameter.

        **Examples**

        >>> r.get_parameter('ignore_degenerate')
        True
        >>> r.get_parameter('x')
        Traceback
            ...
        ValueError: RegridOperator has no 'x' parameter

        """
        try:
            return self._parameters[parameter]
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__} has no {parameter!r} parameter"
            )

    def tosparse(self):
        """Return the weights in sparse COOrdinate format.
        
        .. versionadded:: TODODASK

        .. seealso:: `todense`

        """
        from math import prod

        data = self.weights
        i = self.row - 1
        j = self.col - 1

        src_size = prod(self.src_shape)
        dst_size = prod(self.dst_shape)
        shape = [dst_size, src_size]

        if self.quarter:
             from scipy.sparse import csr_array

             w = csr_array((data, (i, j)), shape=shape)
             w = w[: dst_size / 2, : src_size / 2]
        else:
            from scipy.sparse import coo_array
            
            w = coo_array((data, (i, j)), shape=shape)

        return w
