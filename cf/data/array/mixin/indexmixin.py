from math import ceil

import numpy as np
from dask.base import is_dask_collection

from ....functions import parse_indices


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
        new = self.copy()

        shape0 = self.shape
        index0 = self.index
        index = parse_indices(shape0, index, keepdims=False)

        new_indices = []
        new_shape = []
        for ind0, ind, size in zip(index0, index, shape0):
            if ind == slice(None):
                new_indices.append(ind0)
                new_shape.append(size)
                continue

            if is_dask_collection(ind):
                # I think that this will never occur when __getitem__
                # is being called from within a Dask graph. Otherwise
                # we'll need to run the `compute` inside a `with
                # dask.config.set({"scheduler": "synchronous"}):`
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
                    new_size = ceil((stop - start) / step)
                else:
                    # 'ind0' is slice, 'ind' is (array of) int/bool
                    new_index = np.arange(*ind0.indices(size0))[ind]
                    new_size = new.size
            else:
                # 'ind0' is (array of) int
                new_index = np.asanyarray(ind0)[ind]
                new_size = new.size

            new_indices.append(new_index)
            new_shape.append(new_size)

        new._set_component("index", tuple(new_indices), copy=False)
        new._set_component("shape", tuple(new_shape), copy=False)

        print (index0, index, new_indices)
 
        return new

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
        ind = self._get_component("index", None)
        if ind is None:
            ind = (slice(None),) * self.ndim
            self._set_component("index", ind, copy=False)

        return ind
