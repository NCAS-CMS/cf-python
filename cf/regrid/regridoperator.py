from copy import deepcopy

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
    ):
        """**Initialization**

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

            parameters: Deprecated at version 3.14.0
                Use keyword parameters instead.

            dst: `Field` or `Domain`
                The definition of the destination grid.

            dst_axes: `dict` or sequence or `None`, optional
                The destination grid axes to be regridded.

            src_axes: `dict` or sequence or `None`, optional
                The source grid axes to be regridded.

        """
        super().__init__()

        if weights is None or row is None or col is None:
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
    def dst_shape(self):
        """The shape of the destination grid.

        .. versionadded:: 3.14.0

        """
        return self._get_component("dst_shape")

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
    def src_shape(self):
        """The shape of the source grid.

        .. versionadded:: 3.14.0

        """
        return self._get_component("src_shape")

    @property
    def start_index(self):
        """The start index of the row and column indices.

        .. versionadded:: 3.14.0

        """
        return self._get_component("start_index")

    @property
    def weights(self):
        """The 1-d array of the regridding weights.

        .. versionadded:: 3.14.0

        """
        return self._get_component("weights")

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
            src_axes=self.src_axes,
            dst_axes=self.dst_axes,
            dst=self.dst.copy(),
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
            "dst",
            "weights",
            "row",
            "col",
        ):
            string.append(f"{attr}: {getattr(self, attr)!r}")

        return "\n".join(string)

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
            message="Using attributes instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )

        try:
            return self._get_component("parameters")[parameter]
        except KeyError:
            if default:
                return default[0]

            raise ValueError(
                f"{self.__class__.__name__} has no {parameter!r} parameter"
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

    def todense(self, order="C"):
        """Return the weights in dense format.

        .. versionadded:: 3.14.0

        .. seealso:: `tosparse`

        :Parameters:

            order: `str`, optional
                Specify the memory layout of the returned weights
                matrix. ``'C'`` (the default) means C order
                (row-major), and``'F'`` means Fortran order
                (column-major).

        :Returns:

            `numpy.ndarray`
                The 2-d dense weights matrix, an array with with shape
                ``(J, I)``, where ``J`` is the number of destination
                grid cells and ``I`` is the number of source grid
                cells.

        """
        return self.tosparse().todense(order=order)

    def tosparse(self):
        """Return the weights in sparse COOrdinate format.

        See `scipy.sparse._arrays.coo_array` for sparse format
        details.

        .. versionadded:: 3.14.0

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
