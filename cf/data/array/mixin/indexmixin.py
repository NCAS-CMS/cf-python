from math import ceil
from os import sep
from os.path import basename, dirname, join

import numpy as np

from ....functions import _DEPRECATION_ERROR_ATTRIBUTE, abspath


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
        array = np.asanyarray(self._get_array())
        if not dtype:
            return array
        else:
            return array.astype(dtype[0], copy=False)

    def __getitem__(self, index)
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

        """
        shape0 = self.shape
        index = parse_indices(shape0, index, keepdims=False, bool_as_int=True)
        
        index0 = self._get_component('index', None)
        if index0 is None:
            self._set_component('index', index, copy=False)
            return
                
        new_index = []
        for ind0, ind, size0 in zip(index0, index, shape0):            
            if index == slice(None):
                new_index.append(ind0)
                new_shape.apepend(size0)
                continue
            
            if isinstance(ind0, slice):
                if isinstance(ind, slice):
                    # 'ind0' is slice, 'ind' is slice
                    start, stop, step = ind0.indices(size0)
                    size1, mod = divmod(stop - start - 1, step)
                    start1, stop1, step1 = ind.indices(size1 + 1)
                    size2, mod = divmod(stop1 - start1, step1)

                    if mod != 0:
                        size2 += 1

                    start += start1 * step
                    step *= step1
                    stop = start + (size2 - 1) * step

                    if step > 0:
                        stop += 1
                    else:
                        stop -= 1
                        
                    if stop < 0:
                        stop = None
                        
                    new = slice(start, stop, step)
                    new_size = ceil((stop - start)/step)
                else:
                    # 'ind0' is slice, 'ind' is numpy array of int
                    new = np.arange(*ind0.indices(size0))[ind]
                    new_size = new.size
            else:
                # 'ind0' is numpy array of int
                new = ind0[ind]
                new_size = new.size
                    
            new_index.append(new)
            new_shape.apepend(new_size)

        self._set_component('index', tuple(new_index), copy=False)
        self._set_component('shape', tuple(new_shape), copy=False)
        
    def _get_array(self)
        """Returns a subspace of the array as a numpy array.

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

        """
        return NotImplementedError(
            f"Must implement {self.__class__.__name__}._get_array"
        )
