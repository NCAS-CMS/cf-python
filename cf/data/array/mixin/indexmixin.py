import numpy as np
from dask.base import is_dask_collection

from ....functions import indices_shape, parse_indices


class IndexMixin:
    """Mixin class for lazy subspacing of a data array.

    .. versionadded:: NEXTVERSION

    """

    def __array__(self, *dtype):
        """Convert the ``{{class}}` into a `numpy` array.

        .. versionadded:: (cfdm) NEXTVERSION

        :Parameters:

            dtype: optional
                Typecode or data-type to which the array is cast.

        :Returns:

            `numpy.ndarray`
                An independent `numpy` array of the subspace of the
                data defined by the `indices` attribute.

        """
        array = self._get_array()
        if dtype:
            return array.astype(dtype[0], copy=False)

        return array

    def __getitem__(self, index):
        """Returns a subspace of the array as a new `{{class}}`.

        x.__getitem__(indices) <==> x[indices]

        The new `{{class}}` may be converted to a `numpy` array with
        its `__array__` method.

        Consecutive subspaces are lazy, with only the final data
        elements retrieved from the data when `__array__` is called.

        For example, if the original data has shape ``(12, 145, 192)``
        and consecutive subspaces of ``[8:9, 10:20:3, [15, 1, 4, 12]``
        and ``[[0], [True, False, True], ::-2]`` are applied, then
        only the elements defined by subspace ``[[8], [10, 16], [12,
        1]]`` will be retrieved from the data when `__array__` is
        called.

        Indexing is similar to `numpy` indexing. The only difference
        to `numpy` indexing (given the restrictions on the type of
        indices allowed) is:

          * When two or more dimension's indices are sequences of
            integers then these indices work independently along each
            dimension (similar to the way vector subscripts work in
            Fortran).

        .. versionadded:: NEXTVERSION

        .. seealso:: `index`, `original_shape`, `__array__`,
                     `__getitem__`

        :Returns:

            `{{class}}`
                The subspaced array.

        """
        shape = self.shape
        index0 = self.index
        original_shape = self.original_shape

        index = parse_indices(shape, index, keepdims=False)

        new = self.copy()
        new_indices = []
        new_shape = []
        new_original_shape = []

        for ind0, ind, size, original_size in zip(
            index0, index, shape, original_shape
        ):
            keepdim = True
            if isinstance(ind, slice) and ind == slice(None):
                new_index = ind0
                new_size = size
            else:
                if is_dask_collection(ind):
                    # Note: This will never occur when __getitem__ is
                    #       being called from within a Dask graph, because
                    #       any lazy indices will have already been
                    #       computed as part of the whole graph execution;
                    #       i.e. we don't have to worry about a
                    #       compute-within-a-compute situation. (If this
                    #       were not the case then we could get round it
                    #       by wrapping the compute inside a `with
                    #       dask.config.set({"scheduler":
                    #       "synchronous"}):` clause.)
                    ind = ind.compute()
    
                if isinstance(ind0, slice):
                    if isinstance(ind, slice):
                        # 'ind0' is slice; 'ind' is slice
                        start, stop, step = ind0.indices(size)
                        size0 = indices_shape((ind0,), (original_size,))[0]
                        start1, stop1, step1 = ind.indices(size0 + 1)
                        size1, mod1 = divmod(stop1 - start1, step1)
    
                        if mod1 != 0:
                            size1 += 1
    
                        start += start1 * step
                        step *= step1
                        stop = start + (size1 - 1) * step
    
                        if step > 0:
                            stop += 1
                        else:
                            stop -= 1
    
                        if stop < 0:
                            stop = None
    
                        new_index = slice(start, stop, step)
                    elif np.iterable(ind):
                        # 'ind0' is slice; 'ind' is array of int/bool
                        new_index = np.arange(*ind0.indices(original_size))[ind]
                    else:
                        # 'ind' is Integral. Remove the dimension.
                        new_index = ind
                        keepdim = False
                else:
                    # 'ind0' is array of int
                    new_index = np.asanyarray(ind0)[ind]

            new_indices.append(new_index)
            if keepdim:
                new_original_shape.append(original_size)
            
        new_shape = indices_shape(new_indices, original_shape, keepdims=False)

        new._set_component("shape", tuple(new_shape), copy=False)

        new._custom["original_shape"] =  tuple(new_original_shape)
        new._custom["index"] = tuple(new_indices)
        return new

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        self.original_shape
        out = super().__repr__()
        return f"{out[:-1]}{self._custom['original_shape0']}>"

    @property
    def __asanyarray__(self):
        """Whether the array is accessed by conversion to a `numpy` array.

        Always returns `True`.

        .. versionadded:: NEXTVERSION

        """
        return True

    def _get_array(self):
        """Returns a subspace of the data.

        The subspace is defined by the indices stored in the `index`
        attribute, and may be the result of multiple `__getitem__`
        calls.

        .. versionadded:: NEXTVERSION

        .. seealso:: `__array__`, `index`

        :Returns:

            `numpy.ndarray`
                The subspace.

        """
        return NotImplementedError(
            f"Must implement {self.__class__.__name__}._get_array"
        )

    @property
    def index(self):
        """The index to be applied when converting to a `numpy` array.

        .. versionadded:: NEXTVERSION

        .. seealso:: `original_shape`, `shape`

        **Examples**

        >>> x.index
        (slice(None), slice(None), slice(None))
        >>> x.shape
        (12, 145, 192)
        >>> x = x[8:9, 10:20:3, [15, 1,  4, 12]]
        >>> x.index
        (slice(8, 9), slice(10, 20, 3), [15, 1,  4, 12])
        >>> x.shape
        (1, 3, 4)
        >>> x = x[[0], [True, False, True], ::-2]
        >>> x.index
        ([8], [10, 16], [12, 1])
        >>> x.shape
        (1, 2, 2)

        """
        ind = self._custom.get("index")
        if ind is None:
            ind = (slice(None),) * self.ndim
            self._custom["index"] = ind

        return ind

    @property
    def original_shape(self):
        """The original shape of the data.

        .. versionadded:: NEXTVERSION

        .. seealso:: `index`, `shape`

        """
        shape = self._custom.get("original_shape")
        if shape is None:
            shape = self.shape
            self._custom["original_shape0"] = shape
            self._custom["original_shape"] = shape

        return shape
