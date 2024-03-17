import numpy as np
from dask.base import is_dask_collection

from ....functions import indices_shape, parse_indices


class IndexMixin:
    """TODO xMixin class for an array stored in a file.

    .. versionadded:: NEXTVERSION

    """

    def __array__(self, *dtype):
        """Convert the ``{{class}}` into a `numpy` array.

        TODO stored indices

        .. versionadded:: (cfdm) NEXTVERSION

        :Parameters:

            dtype: optional
                Typecode or data-type to which the array is cast.

        :Returns:

            `numpy.ndarray`
                An independent numpy array of the data.

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
        elements read from the dataset when `__array__` is called.

        For example, if a dataset variable has shape ``(12, 145,
        192)`` and consecutive subspaces of ``[8:9, 10:20:3, [15, 1,
        4, 12]`` and ``[[0], [True, False, True], ::-2]`` are applied
        then only the elements defined by subspace ``[[8], [10, 16],
        [12, 1]]`` will be retrieved from the dataset when `__array__`
        is called.

        Indexing is similar to `numpy` indexing. The only difference
        to numpy indexing (given the restrictions on the type of
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

        for ind0, ind, size, original_size in zip(
            index0, index, shape, original_shape
        ):
            if isinstance(ind, slice) and ind == slice(None):
                new_indices.append(ind0)
                new_shape.append(size)
                continue

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
                    raise ValueError(
                        f"Can't subspace {self!r} with index {ind} that "
                        "removes a dimension"
                    )
            else:
                # 'ind0' is array of int
                new_index = np.asanyarray(ind0)[ind]

            new_indices.append(new_index)

        new_shape = indices_shape(new_indices, original_shape, keepdims=True)
        new._set_component("shape", tuple(new_shape), copy=False)

        new._custom["index"] = tuple(new_indices)
        return new

    def __repr__(self):
        """TODO"""
        out = super().__repr__()
        return f"{out[:-1]}{self.original_shape}>"

    @property
    def _dask_asanyarray(self):
        """TODO

        .. versionadded:: NEXTVERSION

        """
        return True

    def _get_array(self):
        """Returns a subspace of the dataset variable.

        The subspace is defined by the indices stored in the `index`
        attribute.

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

        :Returns:

            `tuple`

        **Examples**

        >>> x.index
        (slice(None, None, None), slice(None, None, None), slice(None, None, None))
        >>> x.index
        (slice(None, None, None), slice(None, None, None), slice(None, None, None))
        >>> x = x[[0], 10:20:2, :]
        >>> x.index

        TODO


        """
        ind = self._custom.get("index")
        if ind is None:
            ind = (slice(None),) * self.ndim
            self._custom["index"] = ind

        return ind

    @property
    def original_shape(self):
        """TODO

        .. versionadded:: NEXTVERSION

        """
        shape = self._custom.get("original_shape")
        if shape is None:
            shape = self.shape
            self._custom["original_shape"] = shape

        return shape
