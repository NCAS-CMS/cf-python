from math import ceil

import numpy as np
from dask.base import is_dask_collection

from ....functions import parse_indices, indices_shape


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
        """TODO Returns a subspace of the array as a numpy array.

        x.__getitem__(indices) <==> x[indices]

        The indices that define the subspace must be either `Ellipsis` or
        a sequence that contains an index for each dimension. In the
        latter case, each dimension's index must either be a `slice`
        object or a sequence of two or more integers.

        Indexing is similar to numpy indexing. The only difference to
        numpy indexing (given the restrictions on the type of indices
        allowed) is:

          * When two or more dimension's indices are sequences of integers
            then these indices work independently along each dimension
            (similar to the way vector subscripts work in Fortran).

        .. versionadded:: NEXTVERSION

        :Returns:

            `{{class}}`
                TODO

        """
        shape = self.shape
        index0 = self.index
        original_shape = self.original_shape

        index = parse_indices(shape, index, keepdims=True)
        
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
                #       computed as part of the whole graph execution
                #       - i.e. we don't have to worry about a
                #       compute-withn-a-compute situation. (If this
                #       were not the case then we could get round it
                #       by wrapping the compute inside a `with
                #       dask.config.set({"scheduler":
                #       "synchronous"}):` claus.)
                ind = ind.compute()

            if isinstance(ind0, slice):
                if isinstance(ind, slice):
                    # 'ind0' is slice, 'ind' is slice
                    start, stop, step = ind0.indices(size)
                    size0, _ = divmod(stop - start - 1, step)
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
                else:
                    # 'ind0' is slice, 'ind' is array of int/bool
                    new_index = np.arange(*ind0.indices(original_size))[ind]
            else:
                # 'ind0' is array of int
                new_index = np.asanyarray(ind0)[ind]

            new_indices.append(new_index)

        new_shape = indices_shape(new_indices, original_shape, keepdims=False)
        new._set_component("shape", tuple(new_shape), copy=False)
        
        new._custom["index"] =  tuple(new_indices)
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
        """TODO Returns a subspace of the array as a numpy array.

        The indices that define the subspace must be either `Ellipsis` or
        a sequence that contains an index for each dimension. In the
        latter case, each dimension's index must either be a `slice`
        object or a sequence of two or more integers.

        Indexing is similar to numpy indexing. The only difference to
        numpy indexing (given the restrictions on the type of indices
        allowed) is:

          * When two or more dimension's indices are sequences of integers
            then these indices work independently along each dimension
            (similar to the way vector subscripts work in Fortran).

        .. versionadded:: NEXTVERSION

        """
        return NotImplementedError(
            f"Must implement {self.__class__.__name__}._get_array"
        )

    @property
    def index(self):
        """TODO

        .. versionadded:: NEXTVERSION

        """
        ind = self._custom.get("index")
        if ind is None:
            ind = (slice(None),) * self.ndim
            self._custom["index"]= ind       

        return ind

    @property
    def original_shape(self):
        """TODO

        .. versionadded:: NEXTVERSION

        """
        shape = self._custom.get('original_shape')
        if shape is None:
            shape =  self.shape
            self._custom["original_shape"] = shape
            
        return shape
            
