import numpy

from numpy import array       as numpy_array
from numpy import asscalar    as numpy_asscalar
from numpy import ndenumerate as numpy_ndenumerate
from numpy import empty       as numpy_empty
from numpy import expand_dims as numpy_expand_dims
from numpy import squeeze     as numpy_squeeze

from copy import deepcopy

from operator import mul

from .partition import Partition

from ..functions import (_DEPRECATION_WARNING_METHOD,
                         _DEPRECATION_ERROR_METHOD)

from ..decorators import (_inplace_enabled,
                          _inplace_enabled_define_and_cleanup)


_empty_matrix = numpy_empty((), dtype=object)


class PartitionMatrix:
    '''

A hyperrectangular partition matrix of a master data array.

Each of elements (called partitions) span all or part of exactly one
sub-array of the master data array.

Normal numpy basic and advanced indexing is supported, but size 1
dimensions are always removed from the output array, i.e. a partition
rather than a partition matrix is returned if the output array has
size 1.


**Attributes**

==========  ===========================================================
Attribute   Description
==========  ===========================================================
`!axes`
`!matrix`
`!ndim`     The number of partition dimensions in the partition matrix.
`!shape`    List of the partition matrix's dimension sizes.
`!size`     The number of partitions in the partition matrix.
==========  ===========================================================

'''

    def __init__(self, matrix, axes):
        '''**Initialization**

    :Parameters:

        matrix: `numpy.ndarray`
            An array of Partition objects.

        axes: `list`
            The identities of the partition axes of the partition
            array. If the partition matrix is a scalar array then it
            is an empty list. DO NOT UPDATE INPLACE.

    **Examples:**

    >>> pm = PartitionMatrix(
    ...          numpy.array(Partition(
    ...              location = [(0, 1), (2, 4)],
    ...              shape = [1, 2],
    ...              _dimensions = ['dim2', 'dim0'],
    ...              Units = cf.Units('m'),
    ...              part = [],
    ...              data = numpy.array([[5, 6], [7, 8]])
    ...          ), dtype=object),
    ...          axes=[]
    ...      )

        '''
        self.matrix = matrix
        self.axes = axes

    def __deepcopy__(self, memo):
        '''Used if copy.deepcopy is called on the variable.

        '''
        return self.copy()

    def __getitem__(self, indices):
        '''x.__getitem__(indices) <==> x[indices]

    Normal numpy basic and advanced indexing is supported, but size 1
    dimensions are always removed from the output array, i.e. a
    partition rather than a partition matrix is returned if the output
    array has size 1.

    Returns either a partition or a partition matrix.

    **Examples:**

    >>> pm.shape
    (5, 3)
    >>> pm[0, 1]
    <cf.data.partition.Partition at 0x1934c80>
    >>> pm[:, 1]
    <CF PartitionMatrix: 1 partition dimensions>
    >>> pm[:, 1].shape
    (5,)
    >>> pm[1:4, slice(2, 0, -1)].shape
    (3, 2)

    >>> pm.shape
    ()
    >>> pm[()]
    <cf.data.partition.Partition at 0x1934c80>
    >>> pm[...]
    <cf.data.partition.Partition at 0x1934c80>

        '''
        out = self.matrix[indices]

        if isinstance(out, Partition):
            return out

        if out.size == 1:
            return self.matrix.item()

        axes = [axis for axis, n in zip(self.axes, out.shape) if n != 1]

        return type(self)(numpy_squeeze(out), axes)

    def __repr__(self):
        '''x.__repr__() <==> repr(x)

        '''
        return '<CF %s: %s>' % (self.__class__.__name__, self.shape)

    def __setitem__(self, indices, value):
        '''x.__setitem__(indices, y) <==> x[indices]=y

    Indices must be an integer, a slice object or a tuple. If a slice
    object is given then the value being assigned must be an
    iterable. If a tuple of integers (or slices equivalent to an
    integer) is given then there must be one index per partition
    matrix dimension.

    **Examples:**

    >>> pm.shape
    (3,)
    >>> pm[2] = p1
    >>> pm[:] = [p1, p2, p3]

    >>> pm.shape
    (2, 3)
    >>> pm[0, 2] = p1


    >>> pm.shape
    ()
    >>> pm[()] = p1
    >>> pm[...] = p1

        '''
        self.matrix[indices] = value

    def __str__(self):
        '''x.__str__() <==> str(x)

        '''
        return str(self.matrix)
        out = []
        for partition in self.matrix.flat:
            out.append(str(partition))

        return '\n'.join(out)

    def change_axis_names(self, axis_map):
        '''Change the axis names.

    The axis names are arbitrary, so mapping them to another arbitrary
    collection does not change the data array values, units, nor axis
    order.

    :Parameters:

        axis_map: `dict`

    :Returns:

        `None`

        '''
        # Partition dimensions
        axes = self.axes
        self.axes = [axis_map[axis] for axis in axes]

        # Partitions. Note that a partition may have dimensions which
        # are not in self.axes and that these must also be in
        # axis_name_map.
        for partition in self.matrix.flat:
            partition.change_axis_names(axis_map)

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def flat(self):
        '''A flat iterator over the partitions in the partition matrix.

    **Examples:**

    >>> pm.shape
    [2, 2]
    >>> for partition in pm.flat:
    ...     print(repr(partition.Units))
    ...
    <CF Units: m s-1>
    <CF Units: km hr-1>
    <CF Units: miles day-1>
    <CF Units: mm minute-1>

    >>> pm.flat
    <numpy.flatiter at 0x1e14840>

    >>> flat = pm.flat
    >>> next(flat)
    <Partition at 0x1934c80>
    >>> next(flat)
    <Partition at 0x784b347>

        '''
        return self.matrix.flat

    @property
    def ndim(self):
        '''The number of partition dimensions in the partition matrix.

    Not to be confused with the number of dimensions of the master
    data array.

    **Examples:**

    >>> pm.shape
    (8, 4)
    >>> pm.ndim
    2

    >>> pm.shape
    ()
    >>> pm.ndim
    0

        '''
        return self.matrix.ndim

    @property
    def shape(self):
        '''List of the partition matrix's dimension sizes.

    Not to be confused with the sizes of the master data array's
    dimensions.

    **Examples:**

    >>> pm.ndim
    2
    >>> pm.size
    32
    >>> pm.shape
    (8, 4)

    >>> pm.ndim
    0
    >>> pm.shape
    ()

        '''
        return self.matrix.shape

    @property
    def size(self):
        '''The number of partitions in the partition matrix.

    Not to be confused with the number of elements in the master data
    array.

    **Examples:**

    >>> pm.shape
    (8, 4)
    >>> pm.size
    32

    >>> pm.shape
    ()
    >>> pm.size
    1

        '''
        return self.matrix.size

    def add_partitions(self, adimensions, master_flip, extra_boundaries, axis):
        '''Add partition boundaries.

    :Parameters:

        adimensions: `list`
            The ordered axis names of the master array.

        master_flip: `list`

        extra_boundaries: `list` of `int`
            The boundaries of the new partitions.

        axis: `str`
            The name of the axis to have the new partitions.

        '''
        def _update_p(matrix, location, master_index,
                      part, master_axis_to_position, master_flip):
            '''TODO

        :Parameters:

            matrix: numpy array of `Partition` objects

            location: `list`

            master_index: `int`

            part: `list`

            master_axis_to_position: `dict`

            master_flip: `list`

        :Returns:

            numpy array of `Partition` objects

            '''
            for partition in matrix.flat:
                partition.location = partition.location[:]
                partition.shape = partition.shape[:]

                partition.location[master_index] = location
                partition.shape[master_index] = shape

                partition.new_part(part,
                                   master_axis_to_position,
                                   master_flip)
            # --- End: for

            return matrix

        # If no extra boundaries have been provided, just return
        # without doing anything
        if not extra_boundaries:
            return

        master_index = adimensions.index(axis)
        index = self.axes.index(axis)

        # Find the position of the extra-boundaries dimension in the
        # list of master array dimensions
        extra_boundaries = extra_boundaries[:]

        # Create the master_axis_to_position dictionary required by
        # Partition.new_part
        master_axis_to_position = {}
        for i, data_axis in enumerate(adimensions):
            master_axis_to_position[data_axis] = i

        matrix = self.matrix
        shape = matrix.shape

        # Initialize the new partition matrix
        new_shape = list(shape)
        new_shape[index] += len(extra_boundaries)
        new_matrix = numpy_empty(new_shape, dtype=object)

        part = [slice(None)] * len(adimensions)
        indices = [slice(None)] * matrix.ndim
        new_indices = indices[:]
        new_indices[index] = slice(0, 1)  # 0

        # Find the first extra boundary
        x = extra_boundaries.pop(0)

        for i in range(shape[index]):
            indices[index] = slice(i, i+1)
            sub_matrix = matrix[tuple(indices)]
#            (r0, r1) = next(sub_matrix.flat).location[master_index]
            (r0, r1) = sub_matrix.item(0,).location[master_index]

# Could do better, perhaps, by assigning in blocks
            if not r0 < x < r1:
                # This new boundary is *not* within the span of this
                # sub-matrix. Therefore, just copy the sub-matrix
                # straight into the new matrix
                new_matrix[tuple(new_indices)] = sub_matrix
#                new_indices[index] += 1
                new_indices[index] = slice(
                    new_indices[index].start+1, new_indices[index].stop+1)
                continue

            # --------------------------------------------------------
            # Still here? Then this new boundary *is* within the span
            # of this sub-matrix.
            # --------------------------------------------------------

            # Find the new extent of the original partition(s)
            location = (r0, x)
            shape = x - r0
            part[master_index] = slice(0, shape)

            # Create new partition(s) in place of the original ones(s)
            # and set the location, shape and part attributes
            new_matrix[tuple(new_indices)] = _update_p(
                deepcopy(sub_matrix), location, master_index, part,
                master_axis_to_position, master_flip)
#            new_indices[index] += 1
            new_indices[index] = slice(
                new_indices[index].start+1, new_indices[index].stop+1)

            while x < r1:
                # Find the extent of the new partition(s)
                if not extra_boundaries:
                    # There are no more new boundaries, so the new
                    # partition(s) run to the end of the original
                    # partition(s) in which they lie.
                    location1 = r1
                else:
                    # There are more new boundaries, so this
                    # new partition runs either to the next
                    # new boundary or to the end of the
                    # original partition, which comes first.
                    location1 = min(extra_boundaries[0], r1)

                location = (x, location1)
                shape = location1 - x
                offset = x - r0
                part[master_index] = slice(offset, offset + shape)

                # Create the new partition(s) and set the
                # location, shape and part attributes
                new_matrix[tuple(new_indices)] = _update_p(
                    deepcopy(sub_matrix), location, master_index, part,
                    master_axis_to_position, master_flip
                )

                new_indices[index] = slice(
                    new_indices[index].start+1, new_indices[index].stop+1)
#                new_indices[index] += 1

                if not extra_boundaries:
                    # There are no more extra boundaries, so we can
                    # return now.
                    #  new_indices[index] = slice(new_indices[index],
                    #  None)
                    new_indices[index] = slice(new_indices[index].start,
                                               None)
                    indices[index] = slice(i+1, None)

                    new_matrix[tuple(new_indices)] = matrix[tuple(indices)]
                    self.matrix = new_matrix

                    return

                # Still here? Then move on to the next new boundary
                x = extra_boundaries.pop(0)
            # --- End: while
        # --- End: for

        self.matrix = new_matrix

    def copy(self):
        '''Return a deep copy.

    ``pm.copy()`` is equivalent to ``copy.deepcopy(pm)``.

    :Returns:

            The deep copy.

    **Examples:**

    >>> pm.copy()

        '''
        # ------------------------------------------------------------
        # NOTE: 15 May 2013. It is necesary to treat
        #       self.matrix.ndim==0 as a special case since there is a
        #       bug (feature?) in numpy <= v1.7 (at least):
        #       http://numpy-discussion.10968.n7.nabble.com/bug-in-deepcopy-of-rank-zero-arrays-td33705.html
        # ------------------------------------------------------------
        matrix = self.matrix

        if not matrix.ndim:
            new_matrix = _empty_matrix.copy()  # numpy_empty((), dtype=object)
            new_matrix[()] = matrix.item().copy()
            return type(self)(new_matrix, [])
        else:
            new_matrix = numpy.empty(matrix.size, dtype=object)
            new_matrix[...] = [partition.copy() for partition in matrix.flat]
            new_matrix.resize(matrix.shape)
            return type(self)(new_matrix, self.axes)

    # 0
    @_inplace_enabled
    def insert_dimension(self, axis, inplace=False):
        '''Insert a new size 1 axis in place.

    The new axis is always inserted at position 0, i.e. it becomes the
    new slowest varying axis.

    .. seealso:: `flip`, `squeezes`, `swapaxes`, `transpose`

    :Parameters:

        axis: `str`
            The internal identity of the new axis.

    :Returns:

        `PartitionMatrix`

    **Examples:**

    >>> pm.shape
    (2, 3)
    >>> pm.insert_dimension('dim2')
    >>> pm.shape
    (1, 2, 3)

        '''
        p = _inplace_enabled_define_and_cleanup(self)

        p.matrix = numpy_expand_dims(p.matrix, 0)
        p.axes = [axis] + p.axes

        return p

    def ndenumerate(self):
        '''Return an iterator yielding pairs of array indices and values.

    :Returns:

        `numpy.ndenumerate`
            An iterator over the array coordinates and values.

    **Examples:**

    >>> pm.shape
    (2, 3)
    >>> for i, partition in pm.ndenumerate():
    ...     print(i, repr(partition))
    ...
    (0, 0) <cf.data.partition.Partition object at 0x13a4490>
    (0, 1) <cf.data.partition.Partition object at 0x24a4650>
    (0, 2) <cf.data.partition.Partition object at 0x35a4590>
    (1, 0) <cf.data.partition.Partition object at 0x46a4789>
    (1, 1) <cf.data.partition.Partition object at 0x57a3456>
    (1, 2) <cf.data.partition.Partition object at 0x68a9872>

        '''
        return numpy_ndenumerate(self.matrix)

    # 0
    def partition_boundaries(self, data_axes):
        '''Return the partition boundaries for each dimension.

    :Parameters:

        data_axes: sequence

    :Returns:

        `dict`

        '''
        boundaries = {}

        matrix = self.matrix
        indices = [0] * self.ndim

        for i, axis in enumerate(self.axes):
            indices[i] = slice(None)
            j = data_axes.index(axis)

            sub_matrix = matrix[tuple(indices)]

            b = [partition.location[j][0] for partition in sub_matrix.flat]

            # Python3: can't access variables from within previous
            # list comprehension
            last_partition = sub_matrix.flat[-1]

            b.append(last_partition.location[j][1])
            boundaries[axis] = b

            indices[i] = 0
        # --- End: for

        return boundaries

    # 0
    @_inplace_enabled
    def swapaxes(self, axis0, axis1, inplace=False):
        '''Swap the positions of two axes.

    Note that this does not change the master data array.

    .. seealso:: `insert_dimension`, `flip`, `squeeze`, `transpose`

    :Parameters:

        axis0, axis1: `int`, `int`
            Select the axes to swap. Each axis is identified by its
            original integer position.

    :Returns:

        `PartitionMatrix`

    **Examples:**

    >>> pm.shape
    (2, 3, 4, 5)
    >>> pm.swapaxes(1, 2)
    >>> pm.shape
    (2, 4, 3, 5)
    >>> pm.swapaxes(1, -1)
    >>> pm.shape
    (2, 5, 3, 4)

        '''
        p = _inplace_enabled_define_and_cleanup(self)

        if axis0 != axis1:
            iaxes = list(range(p.matrix.ndim))
            iaxes[axis1], iaxes[axis0] = iaxes[axis0], iaxes[axis1]
            p.transpose(iaxes, inplace=True)

        return p

    # 0
    def set_location_map(self, data_axes, ns=None):
        '''Set the `!location` attribute of each partition of the partition
    matrix in place.

    :Parameters:

        data_axes: sequence
            The axes of the master data array.

        ns: sequence of `int`, optional

    :Returns:

        `None`

    **Examples:**

    >>> pm.set_location_map(['dim1', 'dim0'])
    >>> pm.set_location_map([])

        '''
        matrix = self.matrix

        shape = matrix.shape
        axes = self.axes

        slice_None = slice(None)

        indices = [slice_None] * matrix.ndim

        # Never update location in-place
        for partition in matrix.flat:
            partition.location = partition.location[:]

        if ns is None:
            ns = range(len(data_axes))

        for axis, n in zip(data_axes, ns):

            if axis in axes:
                # ----------------------------------------------------
                # This data array axis is also a partition matrix axis
                # ----------------------------------------------------
                m = axes.index(axis)
                start = 0
                for i in range(shape[m]):
                    indices[m] = slice(i, i+1)  # i
                    flat = matrix[tuple(indices)].flat

                    partition = next(flat)
                    stop = start + partition.shape[n]
                    location = (start, stop)

                    partition.location[n] = location

                    for partition in flat:
                        partition.location[n] = location

                    start = stop

                # --- End: for
                indices[m] = slice_None
            else:
                # ----------------------------------------------------
                # This data array axis is not a partition matrix axis
                # ----------------------------------------------------
                flat = matrix.flat
                partition = next(flat)
                location = (0, partition.shape[n])

                partition.location[n] = location

                for partition in flat:
                    partition.location[n] = location
        # --- End: for

    # 0
    @_inplace_enabled
    def squeeze(self, inplace=False):
        '''Remove all size 1 axes.

    Note that this does not change the master data array.

    .. seealso:: `insert_dimension`, `flip`, `swapaxes`, `transpose`

    :Returns:

        `PartitionMatrix`

    **Examples:**

    >>> pm.shape
    (1, 2, 1, 2)
    >>> pm.squeeze()
    >>> pm.shape
    (2, 2)

    >>> pm.shape
    (1,)
    >>> pm.squeeze()
    >>> pm.shape
    ()
    >>> pm.squeeze()
    >>> pm.shape
    ()

        '''
        p = _inplace_enabled_define_and_cleanup(self)

        matrix = p.matrix
        shape = matrix.shape

        if 1 in shape:
            p.matrix = matrix.squeeze()

            axes = p.axes
            p.axes = [axis for axis, size in zip(axes, shape) if size > 1]

        return p

    @_inplace_enabled
    def transpose(self, axes, inplace=False):
        '''Permute the partition dimensions of the partition matrix in place.

    Note that this does not change the master data array.

    .. seealso:: `insert_dimension`, `flip`, `squeeze`, `swapaxes`

    :Parameters:

        axes: sequence of `int`
            Permute the axes according to the values given.

    :Returns:

        `PartitionMatrix`

    **Examples:**

    >>> pm.ndim
    3
    >>> pm.transpose((2, 0, 1))

        '''
        p = _inplace_enabled_define_and_cleanup(self)

        matrix = p.matrix
        if list(axes) != list(range(matrix.ndim)):
            p.matrix = matrix.transpose(axes)
            p_axes = p.axes
            p.axes = [p_axes[i] for i in axes]

        return p

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    def expand_dims(self, axis, inplace=False):
        '''Insert a new size 1 axis in place.

    Deprecated at version 3.0.0. Use method 'insert_dimension'
    instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'expand_dims', "Use method 'insert_dimension' instead."
        )  # pragma: no cover


# --- End: class
