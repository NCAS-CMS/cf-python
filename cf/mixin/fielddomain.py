import logging

from numpy import argmax as numpy_argmax
from numpy import array as numpy_array
from numpy import ndarray as numpy_ndarray
from numpy import ones as numpy_ones

from ..query import Query

from ..functions import parse_indices
from ..functions import (_DEPRECATION_ERROR_KWARGS,)

from ..decorators import (
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
    _manage_log_level_via_verbosity,
    _deprecated_kwarg_check,
)

logger = logging.getLogger(__name__)


class FieldDomainMixin:
    '''TODO

    '''
    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------
    def _create_auxiliary_mask_component(self, mask_shape, ind,
                                         compress):
        '''TODO

    .. versionadded:: 3.TODO.0

    :Parameters:

        mask_shape: `tuple`
            The shape of the mask component to be created. This will
            contain `None` for axes not spanned by the *ind*
            parameter.

              *Parameter example*
                  ``mask_shape=(3, 11, None)``

        ind: `numpy.ndarray`
            As returned by a single argument call of
            ``numpy.array(numpy[.ma].where(....))``.

        compress: `bool`
            If True then remove whole slices which only contain masked
            points.

    :Returns:

        `Data`

        '''
#        # --------------------------------------------------------
#        # Find the shape spanned by ind
#        # --------------------------------------------------------
#        shape = [n for n in mask_shape if n]

        # Note that, for now, auxiliary_mask has to be numpy array
        # (rather than a cf.Data object) because we're going to index
        # it with fancy indexing which a cf.Data object might not
        # support - namely a non-monotonic list of integers.
        auxiliary_mask = numpy_ones(mask_shape, dtype=bool)

        auxiliary_mask[tuple(ind)] = False

        if compress:
            # For compressed indices, remove slices which only
            # contain masked points. (Note that we only expect to
            # be here if there were N-d item criteria.)
            for iaxis, (index, n) in enumerate(zip(ind, mask_shape)):
                index = set(index)
                if len(index) < n:
                    auxiliary_mask = auxiliary_mask.take(
                        sorted(index), axis=iaxis
                    )
        # --- End: if

#        # Add missing size 1 axes to the auxiliary mask
#        if auxiliary_mask.ndim < ndim:
#            i = [
#                slice(None) if n else numpy_newaxis
#                for n in mask_shape
#            ]
#            auxiliary_mask = auxiliary_mask[tuple(i)]

        return self._Data(auxiliary_mask)

    def _indices(self, mode, data_axes, **kwargs):
        '''Create indices that define a subspace of the field construct. TODO

    The subspace is defined by identifying indices based on the
    metadata constructs.

    Metadata constructs are selected conditions are specified on their
    data. Indices for subspacing are then automatically inferred from
    where the conditions are met.

    The returned tuple of indices may be used to created a subspace by
    indexing the original field construct with them.

    Metadata constructs and the conditions on their data are defined
    by keyword parameters.

    * Any domain axes that have not been identified remain unchanged.

    * Multiple domain axes may be subspaced simultaneously, and it
      doesn't matter which order they are specified in.

    * Subspace criteria may be provided for size 1 domain axes that
      are not spanned by the field construct's data.

    * Explicit indices may also be assigned to a domain axis
      identified by a metadata construct, with either a Python `slice`
      object, or a sequence of integers or booleans.

    * For a dimension that is cyclic, a subspace defined by a slice or
      by a `Query` instance is assumed to "wrap" around the edges of
      the data.

    * Conditions may also be applied to multi-dimensional metadata
      constructs. The "compress" mode is still the default mode (see
      the positional arguments), but because the indices may not be
      acting along orthogonal dimensions, some missing data may still
      need to be inserted into the field construct's data.

    **Auxiliary masks**

    When creating an actual subspace with the indices, if the first
    element of the tuple of indices is ``'mask'`` then the extent of
    the subspace is defined only by the values of elements three and
    onwards. In this case the second element contains an "auxiliary"
    data mask that is applied to the subspace after its initial
    creation, in order to set unselected locations to missing data.

    .. versionadded:: 3.TODO.0

    :Parameters:

        mode: sequence of `str`, optional
            There are three modes of operation, each of which provides
            indices for a different type of subspace:

            ==============  ==========================================
            *mode*          Description
            ==============  ==========================================
            ``'compress'``  This is the default mode. Unselected
                            locations are removed to create the
                            returned subspace. Note that if a
                            multi-dimensional metadata construct is
                            being used to define the indices then some
                            missing data may still be inserted at
                            unselected locations.

            ``'envelope'``  The returned subspace is the smallest that
                            contains all of the selected
                            indices. Missing data is inserted at
                            unselected locations within the envelope.

            ``'full'``      The returned subspace has the same domain
                            as the original field construct. Missing
                            data is inserted at unselected locations.
            ==============  ==========================================

            At most one mode can be given.

        kwargs: *optional*
            A keyword name is an identity of a metadata construct, and
            the keyword value provides a condition for inferring
            indices that apply to the dimension (or dimensions)
            spanned by the metadata construct's data. Indices are
            created that select every location for which the metadata
            construct's data satisfies the condition.

    :Returns:

        `tuple`
            The indices meeting the conditions.

    **Examples:**

    >>> q = cf.example_field(0)
    >>> print(q)
    Field: specific_humidity (ncvar%q)
    ----------------------------------
    Data            : specific_humidity(latitude(5), longitude(8)) 1
    Cell methods    : area: mean
    Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : time(1) = [2019-01-01 00:00:00]
    >>> indices = q.indices(X=112.5)
    >>> print(indices)
    (slice(0, 5, 1), slice(2, 3, 1))
    >>> q[indicies]
    <CF Field: specific_humidity(latitude(5), longitude(1)) 1>
    >>> q.indices(X=112.5, latitude=cf.gt(-60))
    (slice(1, 5, 1), slice(2, 3, 1))
    >>> q.indices(latitude=cf.eq(-45) | cf.ge(20))
    (array([1, 3, 4]), slice(0, 8, 1))
    >>> q.indices(X=[1, 2, 4], Y=slice(None, None, -1))
    (slice(4, None, -1), array([1, 2, 4]))
    >>> q.indices(X=cf.wi(-100, 200))
    (slice(0, 5, 1), slice(-2, 4, 1))
    >>> q.indices(X=slice(-2, 4))
    (slice(0, 5, 1), slice(-2, 4, 1))
    >>> q.indices('compress', X=[1, 2, 4, 6])
    (slice(0, 5, 1), array([1, 2, 4, 6]))
    >>> q.indices(Y=[True, False, True, True, False])
    (array([0, 2, 3]), slice(0, 8, 1))
    >>> q.indices('envelope', X=[1, 2, 4, 6])
    ('mask', [<CF Data(1, 6): [[False, ..., False]]>], slice(0, 5, 1), slice(1, 7, 1))
    >>> indices = q.indices('full', X=[1, 2, 4, 6])
    ('mask', [<CF Data(1, 8): [[True, ..., True]]>], slice(0, 5, 1), slice(0, 8, 1))
    >>> print(indices)
    >>> print(q)
    <CF Field: specific_humidity(latitude(5), longitude(8)) 1>

    >>> print(a)
    Field: air_potential_temperature (ncvar%air_potential_temperature)
    ------------------------------------------------------------------
    Data            : air_potential_temperature(time(120), latitude(5), longitude(8)) K
    Cell methods    : area: mean
    Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                    : latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : air_pressure(1) = [850.0] hPa
    >>> a.indices(T=410.5)
    (slice(2, 3, 1), slice(0, 5, 1), slice(0, 8, 1))
    >>> a.indices(T=cf.dt('1960-04-16'))
    (slice(4, 5, 1), slice(0, 5, 1), slice(0, 8, 1))
    >>> indices = a.indices(T=cf.wi(cf.dt('1962-11-01'),
    ...                             cf.dt('1967-03-17 07:30')))
    >>> print(indices)
    (slice(35, 88, 1), slice(0, 5, 1), slice(0, 8, 1))
    >>> a[indices]
    <CF Field: air_potential_temperature(time(53), latitude(5), longitude(8)) K>

    >>> print(t)
    Field: air_temperature (ncvar%ta)
    ---------------------------------
    Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K
    Cell methods    : grid_latitude(10): grid_longitude(9): mean where land (interval: 0.1 degrees) time(1): maximum
    Field ancils    : air_temperature standard_error(grid_latitude(10), grid_longitude(9)) = [[0.76, ..., 0.32]] K
    Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
                    : grid_latitude(10) = [2.2, ..., -1.76] degrees
                    : grid_longitude(9) = [-4.7, ..., -1.18] degrees
                    : time(1) = [2019-01-01 00:00:00]
    Auxiliary coords: latitude(grid_latitude(10), grid_longitude(9)) = [[53.941, ..., 50.225]] degrees_N
                    : longitude(grid_longitude(9), grid_latitude(10)) = [[2.004, ..., 8.156]] degrees_E
                    : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., b'kappa']
    Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
    Coord references: grid_mapping_name:rotated_latitude_longitude
                    : standard_name:atmosphere_hybrid_height_coordinate
    Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                    : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                    : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m
    >>> indices = t.indices(latitude=cf.wi(51, 53))
    >>> print(indices)
    ('mask', [<CF Data(1, 5, 9): [[[False, ..., False]]]>], slice(0, 1, 1), slice(3, 8, 1), slice(0, 9, 1))
    >>> t[indices]
    <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(5), grid_longitude(9)) K>

        '''
        if len(mode) > 1:
            raise ValueError(
                "Can't provide more than one positional argument.")

        envelope = 'envelope' in mode
        full = 'full' in mode
        compress = 'compress' in mode or not (envelope or full)

        logger.debug(
            "indices:\n"
            "    envelope, full, compress = {} {} {}\n".format(
                envelope, full, compress
            )
        )  # pragma: no cover

        domain_axes = self.domain_axes
        constructs = self.constructs.filter_by_data()

#        domain_rank = self.rank
#
#        ordered_axes = self._parse_axes(axes)
#        ordered_axes = sorted(domain_axes)
#        if ordered_axes is None or len(ordered_axes) != domain_rank:
#            raise ValueError(
#                "Must provide an ordered sequence of all domain axes "
#                "as the last positional argument. Got {!r}".format(axes)
#            )
#
#        domain_shape = tuple(
#            [domain_axes[axis].get_size(None) for axis in ordered_axes]
#        )
#        if None in domain_shape:
#            raise ValueError(
#                "Can't find indices when a domain axis has no size"
#            )
#
        # Initialize indices
#        indices = [slice(None)] * domain_rank
        indices = {axis: slice(None) for axis in domain_axes}

        parsed = {}
        unique_axes = set()
        n_axes = 0
        for identity, value in kwargs.items():
            if identity in domain_axes:
                axes = (identity,)
                key = None
                construct = None
            else:
                c = constructs.filter_by_identity(identity)
                if len(c) != 1:
                    raise ValueError(
                        "Can't find indices: Ambiguous axis or axes: "
                        "{!r}".format(identity)
                    )

                key, construct = dict(c).popitem()

                axes = self.get_data_axes(key)

            if axes in parsed:
                parsed[axes].append((axes, key, construct, value))
            else:
                y = set(axes)
                for x in parsed:
                    if set(x) == set(y):
                        parsed[x].append((axes, key, construct, value))
                        break

                if axes not in parsed:
                    n_axes += len(axes)
                    parsed[axes] = [(axes, key, construct, value)]
            # --- End: if

            unique_axes.update(axes)

#            sorted_axes = tuple(sorted(axes))
#            if sorted_axes not in parsed:
#                n_axes += len(sorted_axes)
#
#            parsed.setdefault(sorted_axes, []).append(
#                (axes, key, construct, value))
#
#             unique_axes.update(sorted_axes)
        # --- End: for

        if len(unique_axes) < n_axes:
            raise ValueError(
                "Can't find indices: Multiple constructs with incompatible "
                "domain axes"
            )

        auxiliary_mask = {}

        for canonical_axes, axes_key_construct_value in parsed.items():

            axes, keys, constructs, points = list(
                zip(*axes_key_construct_value)
            )

            n_items = len(constructs)
            n_axes = len(canonical_axes)

            if n_items > n_axes:
                if n_axes == 1:
                    a = 'axis'
                else:
                    a = 'axes'

                raise ValueError(
                    "Error: Can't specify {} conditions for {} {}: {}".format(
                        n_items, n_axes, a, points
                    )
                )

            create_mask = False

            item_axes = axes[0]

            logger.debug(
                "    item_axes = {!r}\n"
                "    keys      = {!r}".format(item_axes, keys)
            )  # pragma: no cover

            if n_axes == 1:
                # ----------------------------------------------------
                # 1-d construct
                # ----------------------------------------------------
                ind = None

                axis = item_axes[0]
                item = constructs[0]
                value = points[0]

                logger.debug(
                    "    {} 1-d constructs: {!r}\n"
                    "    axis      = {!r}\n"
                    "    value     = {!r}".format(
                        n_items, constructs, axis, value
                    )
                )  # pragma: no cover

                if isinstance(value, (list, slice, tuple, numpy_ndarray)):
                    # ------------------------------------------------
                    # 1-dimensional CASE 1: Value is already an index,
                    #                       e.g. [0], (0,3),
                    #                       slice(0,4,2),
                    #                       numpy.array([2,4,7]),
                    #                       [True, False, True]
                    # ------------------------------------------------
                    logger.debug('    1-d CASE 1: ')  # pragma: no cover

                    index = value

                    if envelope or full:
                        size = domain_axes[axis].get_size()
                        d = self._Data(range(size))
                        ind = (d[value].array,)
                        index = slice(None)

                elif (
                        item is not None
                        and isinstance(value, Query)
                        and value.operator in ('wi', 'wo')
                        and item.construct_type == 'dimension_coordinate'
                        #                        and self.iscyclic(axis)
                ):
                    # ------------------------------------------------
                    # 1-dimensional CASE 2: Axis is cyclic and
                    #                       subspace criterion is a
                    #                       'within' or 'without'
                    #                       Query instance
                    # ------------------------------------------------
                    logger.debug('    1-d CASE 2: ')  # pragma: no cover

                    if item.increasing:
                        anchor0 = value.value[0]
                        anchor1 = value.value[1]
                    else:
                        anchor0 = value.value[1]
                        anchor1 = value.value[0]

                    a = self.anchor(axis, anchor0, dry_run=True)['roll']
                    b = self.flip(axis).anchor(
                            axis, anchor1, dry_run=True)['roll']

                    size = item.size
                    if abs(anchor1 - anchor0) >= item.period():
                        if value.operator == 'wo':
                            set_start_stop = 0
                        else:
                            set_start_stop = -a

                        start = set_start_stop
                        stop = set_start_stop
                    elif a + b == size:
                        b = self.anchor(axis, anchor1, dry_run=True)['roll']
                        if (b == a and value.operator == 'wo') or not (
                                b == a or value.operator == 'wo'):
                            set_start_stop = -a
                        else:
                            set_start_stop = 0

                        start = set_start_stop
                        stop = set_start_stop
                    else:
                        if value.operator == 'wo':
                            start = b - size
                            stop = -a + size
                        else:
                            start = -a
                            stop = b - size

                    index = slice(start, stop, 1)

                    if full:
                        # index = slice(start, start+size, 1)
                        d = self._Data(list(range(size)))
                        d.cyclic(0)
                        ind = (d[index].array,)

                        index = slice(None)

                elif item is not None:
                    # ------------------------------------------------
                    # 1-dimensional CASE 3: All other 1-d cases
                    # ------------------------------------------------
                    logger.debug('    1-d CASE 3:')  # pragma: no cover

                    item_match = (value == item)

                    if not item_match.any():
                        raise ValueError(
                            "No {!r} axis indices found from: {}".format(
                                identity, value)
                        )

                    index = numpy_asanyarray(item_match)

                    if envelope or full:
                        if numpy_ma_isMA(index):
                            ind = numpy_ma_where(index)
                        else:
                            ind = numpy_where(index)

                        index = slice(None)

                else:
                    raise ValueError(
                        "Must specify a domain axis construct or a construct "
                        "with data for which to create indices"
                    )

                logger.debug(
                    '    index = {}'.format(index))  # pragma: no cover

                # Put the index into the correct place in the list of
                # indices.
                #
                # Note that we might overwrite it later if there's an
                # auxiliary mask for this axis.
#                if axis in ordered_axes:
#                    indices[ordered_axes.index(axis)] = index
                indices[axis] = index

            else:
                # ----------------------------------------------------
                # N-dimensional constructs
                # ----------------------------------------------------
                logger.debug(
                    "    {} N-d constructs: {!r}\n"
                    "    {} points        : {!r}\n"
                    "    shape          : {}".format(
                        n_items, constructs, len(points), points,
                    )
                )  # pragma: no cover

                # Make sure that each N-d item has the same axis order
                for construct, construct_axes in zip(constructs, axes):
                    if construct_axes != canonical_axes:
                        iaxes = [construct_axes.index(axis)
                                 for axis in canonical_axes]
                        construct = construct.transpose(iaxes)

                    transposed_constructs.append(construct)

#                item_axes = sorted_axes

#                g = self.transpose(ordered_axes, constructs=True)
#
#                item_axes = g.get_data_axes(keys[0])
#
#               constructs = [g.constructs[key] for key in keys]
                logger.debug(
                    "    transposed N-d constructs: {!r}".format(
                        transposed_constructs
                    )
                )  # pragma: no cover

                item_matches = [
                    (value == construct).data
                    for value, construct in zip(points, transposed_constructs)
                ]

                item_match = item_matches.pop()

                for m in item_matches:
                    item_match &= m

                item_match = item_match.array  # LAMA alert

                if numpy_ma_isMA:
                    ind = numpy_ma_where(item_match)
                else:
                    ind = numpy_where(item_match)

                logger.debug(
                    "    item_match  = {}\n"
                    "    ind         = {}".format(item_match, ind)
                )  # pragma: no cover

                bounds = [
                    item.bounds.array[ind]
                    for item in transposed_constructs
                    if item.has_bounds()
                ]

                contains = False
                if bounds:
                    points2 = []
                    for v, construct in zip(points, transposed_constructs):
                        if isinstance(v, Query):
                            if v.operator == 'contains':
                                contains = True
                                v = v.value
                            elif v.operator == 'eq':
                                v = v.value
                            else:
                                contains = False
                                break
                        # --- End: if

                        v = Data.asdata(v)
                        if v.Units:
                            v.Units = construct.Units

                        points2.append(v.datum())
                # --- End: if

                if contains:
                    # The coordinates have bounds and the condition is
                    # a 'contains' Query object. Check each
                    # potentially matching cell for actually including
                    # the point.
                    try:
                        Path
                    except NameError:
                        raise ImportError(
                            "Must install matplotlib to create indices based "
                            "on {}-d constructs and a 'contains' Query "
                            "object".format(transposed_constructs[0].ndim)
                        )

                    if n_items != 2:
                        raise ValueError(
                            "Can't index for cell from {}-d coordinate "
                            "objects".format(n_axes)
                        )

                    if 0 < len(bounds) < n_items:
                        raise ValueError("bounds alskdaskds TODO")

                    # Remove grid cells if, upon closer inspection,
                    # they do actually contain the point.
                    delete = [n for n, vertices in
                              enumerate(zip(*zip(*bounds))) if not
                              Path(zip(*vertices)).contains_point(points2)]

                    if delete:
                        ind = [numpy_delete(ind_1d, delete) for ind_1d in ind]
            # --- End: if

            if ind is not None:
                mask_shape = [] #[None] * domain_rank
                masked_subspace_size = 1
                ind = numpy_array(ind)
                logger.debug('    ind = {}'.format(ind))  # pragma: no cover

                for i, (axis, start, stop) in enumerate(
                        zip(canonical_axes, ind.min(axis=1), ind.max(axis=1))
                ):
                    if data_axes and axis not in data_axes:
                        continue

#                    position = ordered_axes.index(axis)

#                    if indices[position] == slice(None):
                    if indices[axis] == slice(None):
                        if compress:
                            # Create a compressed index for this axis
                            size = stop - start + 1
                            index = sorted(set(ind[i]))
                        elif envelope:
                            # Create an envelope index for this axis
                            stop += 1
                            size = stop - start
                            index = slice(start, stop)
                        elif full:
                            # Create a full index for this axis
                            start = 0
                            stop = self.domain_axes[axis].get_size()
                            size = stop - start
                            index = slice(start, stop)
                        else:
                            raise ValueError(
                                "Must have full, envelope or compress"
                            )  # pragma: no cover

#                        indices[position] = index
                        indices[axis] = index

#                    mask_shape[position] = size
                    mask_shape.append(size)
                    masked_subspace_size *= size
                    ind[i] -= start
                # --- End: for

                create_mask = (data_axes
                               and ind.shape[1] < masked_subspace_size)
            else:
                create_mask = False

            # --------------------------------------------------------
            # Create an auxiliary mask for these axes
            # --------------------------------------------------------
            logger.debug(
                "    create_mask = {}".format(create_mask)
            )  # pragma: no cover

            if create_mask:
                logger.debug(
                    "    mask_shape  = {}".format(mask_shape)
                )  # pragma: no cover

                mask = self._create_auxiliary_mask_component(
                    mask_shape, ind, compress
                )
                auxiliary_mask[canonical_axes] = mask
                logger.debug(
                    "    mask_shape  = {}\n"
                    "    mask.shape  = {}".format(mask_shape, mask.shape)
                )  # pragma: no cover
        # --- End: for

        for axis, index in tuple(indices.items()):
            indices[axis] = parse_indices(
                (domain_axes[axis].get_size(),), (index,)
            )[0]

#        indices = parse_indices(domain_shape, tuple(indices))

        # Include the auxiliary mask
        indices = {
            'indices': indices,
            'auxiliary_mask': auxiliary_mask,
        }

        # Return the indices and the auxiliary mask
        return indices

    def _roll_constructs(self, axis, shift):
        '''Roll the constructs along a cyclic axis.

    A unique axis is selected with the axes and kwargs parameters.

    .. versionadded:: 3.TODO.0

    .. seealso:: `anchor`, `axis`, `cyclic`, `iscyclic`, `period`

    :Parameters:

        axis:
            The cyclic axis to be rolled, defined by that which would
            be selected by passing the given axis description to a
            call of the field construct's `domain_axis` method. For
            example, for a value of ``'X'``, the domain axis construct
            returned by ``f.domain_axis('X')`` is selected.

        shift: `int`
            The number of places by which the selected cyclic axis is
            to be rolled.

    :Returns:

        `None`

    **Examples:**

    Roll the data of the "X" axis one elements to the right:

    >>> f.roll('X', 1)

    Roll the data of the "X" axis three elements to the left:

    >>> f.roll('X', -3)

        '''
        dim = self.dimension_coordinates.filter_by_axis(
            'exact', axis
        ).value(None)
        
        if dim is not None and dim.period() is None:
            raise ValueError(
                "Can't roll: {!r} axis has non-periodic dimension "
                "coordinate construct".format(dim.identity())
            )

        data_axes = self.constructs.data_axes()

        for key, construct in self.constructs.filter_by_data().items():
            axes = data_axes.get(key, ())
            if axis in axes:
                construct.roll(axes.index(axis), shift, inplace=True)
        # --- End: for

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    @_deprecated_kwarg_check('i')
    @_inplace_enabled(default=False)
    def anchor(self, axis, value, inplace=False, dry_run=False,
               i=False, **kwargs):
        '''Roll a cyclic axis so that the given value lies in the first
    coordinate cell.

    A unique axis is selected with the *axes* and *kwargs* parameters.

    .. versionadded:: 3.TODO.0

    .. seealso:: `axis`, `cyclic`, `iscyclic`, `roll`

    :Parameters:

        axis:
            The cyclic axis to be rolled, defined by that which would
            be selected by passing the given axis description to a
            call of the field construct's `domain_axis` method. For
            example, for a value of ``'X'``, the domain axis construct
            returned by ``f.domain_axis('X')`` is selected.

        value:
            Anchor the dimension coordinate values for the selected
            cyclic axis to the *value*. May be any numeric scalar
            object that can be converted to a `Data` object (which
            includes `numpy` and `Data` objects). If *value* has units
            then they must be compatible with those of the dimension
            coordinates, otherwise it is assumed to have the same
            units as the dimension coordinates. The coordinate values
            are transformed so that *value* is "equal to or just
            before" the new first coordinate value. More specifically:

              * Increasing dimension coordinates with positive period,
                P, are transformed so that *value* lies in the
                half-open range (L-P, F], where F and L are the
                transformed first and last coordinate values,
                respectively.

        ..

              * Decreasing dimension coordinates with positive period,
                P, are transformed so that *value* lies in the
                half-open range (L+P, F], where F and L are the
                transformed first and last coordinate values,
                respectively.

            *Parameter example:*
              If the original dimension coordinates are ``0, 5, ...,
              355`` (evenly spaced) and the period is ``360`` then
              ``value=0`` implies transformed coordinates of ``0, 5,
              ..., 355``; ``value=-12`` implies transformed
              coordinates of ``-10, -5, ..., 345``; ``value=380``
              implies transformed coordinates of ``380, 385, ...,
              715``.

            *Parameter example:*
              If the original dimension coordinates are ``355, 350,
              ..., 0`` (evenly spaced) and the period is ``360`` then
              ``value=355`` implies transformed coordinates of ``355,
              350, ..., 0``; ``value=0`` implies transformed
              coordinates of ``0, -5, ..., -355``; ``value=392``
              implies transformed coordinates of ``390, 385, ...,
              30``.

        {{inplace: `bool`, optional}}

        dry_run: `bool`, optional
            Return a dictionary of parameters which describe the
            anchoring process. The field is not changed, even if *i*
            is True.

        {{i: deprecated at version 3.0.0}}

        kwargs: deprecated at version 3.0.0

    :Returns:

        `dict`

    **Examples:**

    >>> f.iscyclic('X')
    True
    >>> f.dimension_coordinate('X').data
    <CF Data(8): [0, ..., 315] degrees_east> TODO
    >>> print(f.dimension_coordinate('X').array)
    [  0  45  90 135 180 225 270 315]
    >>> g = f.anchor('X', 230)
    >>> print(g.dimension_coordinate('X').array)
    [270 315   0  45  90 135 180 225]
    >>> g = f.anchor('X', cf.Data(590, 'degreesE'))
    >>> print(g.dimension_coordinate('X').array)
    [630 675 360 405 450 495 540 585]
    >>> g = f.anchor('X', cf.Data(-490, 'degreesE'))
    >>> print(g.dimension_coordinate('X').array)
    [-450 -405 -720 -675 -630 -585 -540 -495]

    >>> f.iscyclic('X')
    True
    >>> f.dimension_coordinate('X').data
    <CF Data(8): [0.0, ..., 357.1875] degrees_east>
    >>> f.anchor('X', 10000).dimension_coordinate('X').data
    <CF Data(8): [10001.25, ..., 10358.4375] degrees_east>
    >>> d = f.anchor('X', 10000, dry_run=True)
    >>> d
    {'axis': 'domainaxis2',
     'nperiod': <CF Data(1): [10080.0] 0.0174532925199433 rad>,
     'roll': 28}
    >>> (f.roll(d['axis'], d['roll']).dimension_coordinate(
    ...     d['axis']) + d['nperiod']).data
    <CF Data(8): [10001.25, ..., 10358.4375] degrees_east>

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'anchor', kwargs)  # pragma: no cover

        axis = self.domain_axis(axis, key=True)

        if dry_run:
            f = self
        else:
            f = _inplace_enabled_define_and_cleanup(self)

        dimension_coordinates = self.dimension_coordinates

        dim = dimension_coordinates.filter_by_axis('and', axis).value(
            default=None
        )
        if dim is None:
            raise ValueError(
                "Can't anchor non-cyclic {!r} axis".format(
                    f.constructs.domain_axis_identity(axis))
            )

        period = dim.period()
        if period is None:
            raise ValueError(
                "Cyclic {!r} axis has no period".format(dim.identity()))

        value = self._Data.asdata(value)
        if not value.Units:
            value = value.override_units(dim.Units)
        elif not value.Units.equivalent(dim.Units):
            raise ValueError(
                "Anchor value has incompatible units: {!r}".format(
                    value.Units)
            )

        axis_size = f.domain_axes[axis].get_size()

        if axis_size <= 1:
            # Don't need to roll a size one axis
            return {'axis': axis, 'roll': 0, 'nperiod': 0}

        c = dim.get_data()

        if dim.increasing:
            # Adjust value so it's in the range [c[0], c[0]+period)
            n = ((c[0] - value) / period).ceil()
            value1 = value + n * period

            shift = axis_size - numpy_argmax((c - value1 >= 0).array)
            if not dry_run:
                f.roll(axis, shift, inplace=True)

            dim = dimension_coordinates.filter_by_axis('and', axis).value()
            n = ((value - dim.data[0]) / period).ceil()
        else:
            # Adjust value so it's in the range (c[0]-period, c[0]]
            n = ((c[0] - value) / period).floor()
            value1 = value + n * period

            shift = axis_size - numpy_argmax((value1 - c >= 0).array)

            if not dry_run:
                f.roll(axis, shift, inplace=True)

            dim = f.dimension_coordinate(axis)
            n = ((value - dim.data[0]) / period).floor()
        # --- End: if

        if dry_run:
            return {'axis': axis, 'roll': shift, 'nperiod': n * period}

        if n:
            np = n * period
            dim += np
            bounds = dim.get_bounds(None)
            if bounds is not None:
                bounds += np
        # --- End: if

        return f

    @_manage_log_level_via_verbosity
    def autocyclic(self, verbose=None):
        '''Set dimensions to be cyclic.

    A dimension is set to be cyclic if it has a unique longitude (or
    grid longitude) dimension coordinate construct with bounds and the
    first and last bounds values differ by 360 degrees (or an
    equivalent amount in other units).

    .. versionadded:: 1.0

    .. seealso:: `cyclic`, `iscyclic`, `period`

    :Parameters:

        {{verbose: `int` or `str` or `None`, optional}}

    :Returns:

       `bool`

    **Examples:**

    >>> f.autocyclic()

        '''
        dims = self.dimension_coordinates('X')

        if len(dims) != 1:
            logger.debug(
                "Not one 'X' dimension coordinate construct: {}".format(
                    len(dims))
            )  # pragma: no cover
            return False

        key, dim = dict(dims).popitem()
        if not dim.Units.islongitude:
            logger.debug(0)
            if dim.get_property('standard_name', None) not in (
                    'longitude', 'grid_longitude'):
                self.cyclic(key, iscyclic=False)
                logger.debug(1)  # pragma: no cover
                return False
        # --- End: if

        bounds = dim.get_bounds(None)
        if bounds is None:
            self.cyclic(key, iscyclic=False)
            logger.debug(2)  # pragma: no cover
            return False

        bounds_data = bounds.get_data(None)
        if bounds_data is None:
            self.cyclic(key, iscyclic=False)
            logger.debug(3)  # pragma: no cover
            return False

        bounds = bounds_data.array

        period = self._Data(360.0, units='degrees')

        period.Units = bounds_data.Units

        if abs(bounds[-1, -1] - bounds[0, 0]) != period.array:
            self.cyclic(key, iscyclic=False)
            logger.debug(4)  # pragma: no cover
            return False

        self.cyclic(key, iscyclic=True, period=period)
        logger.debug(5)  # pragma: no cover

        return True

    def del_construct(self, identity=None, default=ValueError()):
        '''Remove a metadata construct.

    If a domain axis construct is selected for removal then it can't
    be spanned by any metadata construct's data. See `del_domain_axis`
    for more options in this case.

    A domain ancillary construct may be removed even if it is
    referenced by coordinate reference construct. In this case the
    reference is replace with `None`.

    .. versionadded:: 3.TODO.0

    .. seealso:: `constructs`, `get_construct`, `has_construct`,
                 `set_construct`, `del_domain_axis`,
                 `del_coordinate_reference`

    :Parameters:

        identity: optional
            Select the construct by one of

            * A metadata construct identity.

              {{construct selection identity}}

            * The key of a metadata construct

            * `None`. This is the default, which selects the metadata
              construct when there is only one of them.

            *Parameter example:*
              ``identity='latitude'``

            *Parameter example:*
              ``identity='T'

            *Parameter example:*
              ``identity='long_name=Cell Area'``

            *Parameter example:*
              ``identity='cellmeasure1'``

            *Parameter example:*
              ``identity='measure:area'``

            *Parameter example:*
              ``identity=cf.eq('time')'``

            *Parameter example:*
              ``identity=re.compile('^lat')``

            Select the construct to removed. Must be

              * The identity or key of a metadata construct.

            A construct identity is specified by a string
            (e.g. ``'latitude'``, ``'long_name=time'``,
            ``'ncvar%lat'``, etc.); a `Query` object
            (e.g. ``cf.eq('longitude')``); or a compiled regular
            expression (e.g. ``re.compile('^atmosphere')``) that
            selects the relevant constructs whose identities match via
            `re.search`.

            A construct has a number of identities, and is selected if
            any of them match any of those provided. A construct's
            identities are those returned by its `!identities`
            method. In the following example, the construct ``x`` has
            six identities:

               >>> x.identities()
               ['time',
                'long_name=Time',
                'foo=bar',
                'standard_name=time',
                'ncvar%t',
                'T']

            A construct key may optionally have the ``'key%'``
            prefix. For example ``'dimensioncoordinate2'`` and
            ``'key%dimensioncoordinate2'`` are both acceptable keys.

            Note that in the output of a `print` call or `!dump`
            method, a construct is always described by one of its
            identities, and so this description may always be used as
            an *identity* argument.

            *Parameter example:*
              ``identity='measure:area'``

            *Parameter example:*
              ``identity='cell_area'``

            *Parameter example:*
              ``identity='long_name=Cell Area'``

            *Parameter example:*
              ``identity='cellmeasure1'``

        default: optional
            Return the value of the *default* parameter if the
            construct can not be removed, or does not exist. If set to
            an `Exception` instance then it will be raised instead.

    :Returns:

            The removed metadata construct.

    **Examples:**

    >>> f.del_construct('X')
    <CF DimensionCoordinate: grid_latitude(111) degrees>

        '''
        key = self.construct_key(identity, default=None)
        if key is None:
            return self._default(
                default,
                "Can't identify construct to delete from {!r}".format(identity)
            )

        return super().del_construct(key, default=default)

    def del_coordinate_reference(self, identity=None, construct=None,
                                 default=ValueError()):
        '''Remove a coordinate reference construct and all of its domain
    ancillary constructs.

    .. versionadded:: 3.0.0

    .. seealso:: `del_construct`

    :Parameters:

        identity: optional
            Select the coordinate reference construct by one of:

            * The identity of a coordinate reference construct.

              {{construct selection identity}}

            * The key of a coordinate reference construct

            * `None`. This is the default, which selects the
              coordinate reference construct when there is only one of
              them.

            *Parameter example:*
              ``identity='standard_name:atmosphere_hybrid_height_coordinate'``

            *Parameter example:*
              ``identity='grid_mapping_name:rotated_latitude_longitude'``

            *Parameter example:*
              ``identity='transverse_mercator'``

            *Parameter example:*
              ``identity='coordinatereference1'``

            *Parameter example:*
              ``identity='key%coordinatereference1'``

            *Parameter example:*
              ``identity='ncvar%lat_lon'``

            *Parameter example:*
              ``identity=cf.eq('rotated_pole')'``

            *Parameter example:*
              ``identity=re.compile('^rotated')``

        construct: optional
            TODO

        default: optional
            Return the value of the *default* parameter if the
            construct can not be removed, or does not exist.

            {{default Exception}}

    :Returns:

            The removed coordinate reference construct.

    **Examples:**

    >>> f.del_coordinate_reference('rotated_latitude_longitude')
    <CF CoordinateReference: rotated_latitude_longitude>

        '''
        if construct is None:
            if identity is None:
                raise ValueError("TODO")

            key = self.coordinate_reference(identity, key=True, default=None)
            if key is None:
                return self._default(
                    default,
                    "Can't identify construct from {!r}".format(identity))

            ref = self.del_construct(key)

            for da_key in (
                    ref.coordinate_conversion.domain_ancillaries().values()):
                self.del_construct(da_key, default=None)

            return ref
        elif identity is not None:
            raise ValueError("TODO")

        out = []

        c_key = self.construct(construct, key=True, default=None)
        if c_key is None:
            return self._default(
                default,
                "Can't identify construct from {!r}".format(construct))

        for key, ref in tuple(self.coordinate_references.items()):
            if c_key in ref.coordinates():
                self.del_coordinate_reference(key, construct=None,
                                              default=default)
                out.append(ref)
                continue

            if (c_key in
                    ref.coordinate_conversion.domain_ancillaries().values()):
                self.del_coordinate_reference(key, construct=None,
                                              default=default)
                out.append(ref)
                continue
        # --- End: for

        return out

    def del_domain_axis(self, identity=None, squeeze=False,
                        default=ValueError()):
        '''Remove a domain axis construct.

    In general, a domain axis construct can only be removed if it is
    not spanned by any construct's data. However, a size 1 domain axis
    construct can be removed in any case if the *squeeze* parameter is
    set to `True`. In this case, a metadata construct whose data spans
    only the removed domain axis construct will also be removed.

    .. versionadded:: 3.6.0

    .. seealso:: `del_construct`

    :Parameters:

        identity: optional
            Select the domain axis construct by one of:

            * An identity or key of a 1-d dimension or auxiliary
              coordinate construct that whose data spans the domain
              axis construct.

              {{construct selection identity}}
       
            * A domain axis construct identity.

              {{domain axis selection identity}}
        
            * The key of a domain axis construct.

            * `None`. This is the default, which selects the domain
              axis construct when there is only one of them.
            ``'key%dimensioncoordinate2'`` are both acceptable keys.

            *Parameter example:*
              ``identity='long_name=Latitude'``

            *Parameter example:*
              ``identity='dimensioncoordinate1'``

            *Parameter example:*
              ``identity='domainaxis2'``

            *Parameter example:*
              ``identity='key%domainaxis2'``

            *Parameter example:*
              ``identity='ncdim%y'``

        squeeze: `bool`, optional
            If True then allow the removal of a size 1 domain axis
            construct that is spanned by any data array and squeeze
            the corresponding dimension from those arrays.

        default: optional
            Return the value of the *default* parameter if the
            construct can not be removed, or does not exist.

            {{default Exception}}

    :Returns:

        `DomainAxis`
            The removed domain axis construct.

    **Examples:**

    >>> f = cf.example_field(0)
    >>> g = f[0]
    Field: specific_humidity (ncvar%q)
    ----------------------------------
    Data            : specific_humidity(latitude(1), longitude(8)) 1
    Cell methods    : area: mean
    Dimension coords: latitude(1) = [-75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : time(1) = [2019-01-01 00:00:00]
    >>> g.del_domain_axis('Y', squeeze=True)
    <CF DomainAxis: size(1)>
    >>> print(g)
    Field: specific_humidity (ncvar%q)
    ----------------------------------
    Data            : specific_humidity(longitude(8)) 1
    Cell methods    : area: mean
    Dimension coords: longitude(8) = [22.5, ..., 337.5] degrees_east
                    : time(1) = [2019-01-01 00:00:00]
    >>> g.del_domain_axis('T', squeeze=True)
    <CF DomainAxis: size(1)>
    >>> print(g)
    Field: specific_humidity (ncvar%q)
    ----------------------------------
    Data            : specific_humidity(longitude(8)) 1
    Cell methods    : area: mean
    Dimension coords: longitude(8) = [22.5, ..., 337.5] degrees_east

        '''
        dakey = self.domain_axis(identity, key=True)
        domain_axis = self.constructs[dakey]

        if not squeeze:
            return self.del_construct(dakey)

        if dakey in self.get_data_axes(default=()):
            self.squeeze(dakey, inplace=True)

        for ckey, construct in self.constructs.filter_by_data().items():
            data = construct.get_data(None)
            if data is None:
                continue

            construct_axes = self.get_data_axes(ckey)
            if dakey not in construct_axes:
                continue

            i = construct_axes.index(dakey)
            construct.squeeze(i, inplace=True)
            construct_axes = list(construct_axes)
            construct_axes.remove(dakey)
            self.set_data_axes(axes=construct_axes, key=ckey)

            if not construct_axes:
                self.del_construct(ckey)
        # --- End: for

        return domain_axis

    def auxiliary_coordinate(self, identity=None,
                             default=ValueError(), key=False):
        '''Return an auxiliary coordinate construct, or its key.

    .. versionadded:: 3.0.0

    .. seealso:: `construct`, `auxiliary_coordinates`, `cell_measure`,
                 `cell_method`, `coordinate`, `coordinate_reference`,
                 `dimension_coordinate`, `domain_ancillary`,
                 `domain_axis`, `field_ancillary`

    :Parameters:

        identity: optional
            Select the auxiliary coordinate construct by one of:

            * The identity of a auxiliary coordinate construct.

              {{construct selection identity}}

            * The key of a auxiliary coordinate construct

            * `None`. This is the default, which selects the auxiliary
              coordinate construct when there is only one of them.

            *Parameter example:*
              ``identity='Y'``

            *Parameter example:*
              ``identity='latitude'``

            *Parameter example:*
              ``identity='long_name=Latitude'``

            *Parameter example:*
              ``identity='auxiliarycoordinate1'``

            *Parameter example:*
              ``identity='ncdim%y'``

            *Parameter example:*
              ``identity=cf.eq('time')'``

            *Parameter example:*
              ``identity=re.compile('^lat')``

        key: `bool`, optional
            If True then return the selected construct key. By
            default the construct itself is returned.

        default: optional
            Return the value of the *default* parameter if a construct
            can not be found.

            {{default Exception}}

    :Returns:

        `AuxiliaryCoordinate` or `str`
            The selected auxiliary coordinate construct, or its key.

    **Examples:**

    TODO

        '''
        c = self.auxiliary_coordinates

        if identity is not None:
            c = c(identity)
            if not c:
                da_key = self.domain_axis(identity, key=True, default=None)
                if da_key is not None:
                    c = self.auxiliary_coordinates.filter_by_axis(
                        'exact', da_key)
        # --- End: if

        if key:
            return c.key(default=default)

        return c.value(default=default)

    def construct(self, identity=None, default=ValueError(), key=False):
        '''Select a metadata construct by its identity.

    .. seealso:: `del_construct`, `get_construct`, `has_construct`,
                 `set_construct`

    :Parameters:

        identity: optional
            Select the construct by one of

            * A metadata construct identity.

              {{construct selection identity}}

            * The key of a metadata construct

            * `None`. This is the default, which selects the metadata
              construct when there is only one of them.

            *Parameter example:*
              ``identity='latitude'``

            *Parameter example:*
              ``identity='T'

            *Parameter example:*
              ``identity='long_name=Cell Area'``

            *Parameter example:*
              ``identity='cellmeasure1'``

            *Parameter example:*
              ``identity='measure:area'``

            *Parameter example:*
              ``identity=cf.eq('time')'``

            *Parameter example:*
              ``identity=re.compile('^lat')``

        key: `bool`, optional
            If True then return the selected construct key. By
            default the construct itself is returned.

        default: optional
            Return the value of the *default* parameter if a construct
            can not be found.

            {{default Exception}}

    :Returns:

            The selected coordinate construct, or its key.

    **Examples:**

    >>> f = cf.example_field(1)
    >>> print(f)
    Field: air_temperature (ncvar%ta)
    ---------------------------------
    Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K
    Cell methods    : grid_latitude(10): grid_longitude(9): mean where land (interval: 0.1 degrees) time(1): maximum
    Field ancils    : air_temperature standard_error(grid_latitude(10), grid_longitude(9)) = [[0.76, ..., 0.32]] K
    Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
                    : grid_latitude(10) = [2.2, ..., -1.76] degrees
                    : grid_longitude(9) = [-4.7, ..., -1.18] degrees
                    : time(1) = [2019-01-01 00:00:00]
    Auxiliary coords: latitude(grid_latitude(10), grid_longitude(9)) = [[53.941, ..., 50.225]] degrees_N
                    : longitude(grid_longitude(9), grid_latitude(10)) = [[2.004, ..., 8.156]] degrees_E
                    : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., b'kappa']
    Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
    Coord references: grid_mapping_name:rotated_latitude_longitude
                    : standard_name:atmosphere_hybrid_height_coordinate
    Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                    : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                    : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m

    >>> f.construct('long_name=Grid latitude name')
    <CF AuxiliaryCoordinate: long_name=Grid latitude name(10) >
    >>> f.construct('ncvar%a')
    <CF DomainAncillary: ncvar%a(1) m>
    >>> f.construct('measure:area')
    <CF CellMeasure: measure:area(9, 10) km2>
    >>> f.construct('domainaxis0')
    <CF DomainAxis: size(1)>
    >>> f.construct('height')
    Traceback (most recent call last):
        ...
    ValueError: Can't return zero constructs
    >>> f.construct('height', default=False)
    False
    >>> f.construct('height', default=TypeError("No height coordinates"))
    Traceback (most recent call last):
        ...
    TypeError: No height coordinates

        '''
        c = self.constructs(identity)

        if key:
            return c.key(default=default)

        return c.value(default=default)

    def cell_measure(self, identity=None, default=ValueError(), key=False):
        '''Select a cell measure construct by its identity.

    .. versionadded:: 3.0.0

    .. seealso:: `construct`, `auxiliary_coordinate`, `cell_measures`,
                 `cell_method`, `coordinate`, `coordinate_reference`,
                 `dimension_coordinate`, `domain_ancillary`,
                 `domain_axis`, `field_ancillary`

    :Parameters:

        identity: optional
            Select the cell measure construct by one of:

            * The identity of a cell measure construct.

              {{construct selection identity}}

            * The key of a cell measure construct

            * `None`. This is the default, which selects the cell
              measure construct when there is only one of them.

            *Parameter example:*
              ``identity='measure:area'``

            *Parameter example:*
              ``identity='cell_area'``

            *Parameter example:*
              ``identity='long_name=Cell Area'``

            *Parameter example:*
              ``identity='cellmeasure1'``

            *Parameter example:*
              ``identity=cf.eq('cell_area')'``

            *Parameter example:*
              ``identity=re.compile('^cell')``

        key: `bool`, optional
            If True then return the selected construct key. By
            default the construct itself is returned.

        default: optional
            Return the value of the *default* parameter if a construct
            can not be found.

            {{default Exception}}

    :Returns:

        `CellMeasure`or `str`
            The selected cell measure construct, or its key.

    **Examples:**

    TODO

        '''
        c = self.cell_measures

        if identity is not None:
            c = c(identity)
            if not c:
                da_key = self.domain_axis(identity, key=True, default=None)
                if da_key is not None:
                    c = self.cell_measures.filter_by_axis('exact', da_key)
        # --- End: if

        if key:
            return c.key(default=default)

        return c.value(default=default)

    def coordinate(self, identity=None, default=ValueError(),
                   key=False):
        '''Return a dimension or auxiliary coordinate construct, or its key.

    .. versionadded:: 3.0.0

    .. seealso:: `construct`, `auxiliary_coordinate`, `coordinates`,
                 `dimension_coordinate`

    :Parameters:

        identity: optional
            Select the coordinate construct by one of:
                            
            * The identity of a dimension or auxiliary coordinate
              construct.

              {{construct selection identity}}

            * The key of a dimension or auxiliary coordinate construct

            * `None`. This is the default, which selects the dimension
              or auxiliary coordinate construct when there is only one
              of them.

            *Parameter example:*
              ``identity='Y'``

            *Parameter example:*
              ``identity='latitude'``

            *Parameter example:*
              ``identity='long_name=Latitude'``

            *Parameter example:*
              ``identity='auxiliarycoordinate1'``

            *Parameter example:*
              ``identity='ncdim%y'``

            *Parameter example:*
              ``identity=cf.eq('time')'``

            *Parameter example:*
              ``identity=re.compile('^lat')``

        key: `bool`, optional
            If True then return the selected construct key. By
            default the construct itself is returned.

        default: optional
            Return the value of the *default* parameter if a construct
            can not be found.

            {{default Exception}}

    :Returns:

        `DimensionCoordinate` or `AuxiliaryCoordinate` or `str`
            The selected dimension or auxiliary coordinate construct,
            or its key.

    **Examples:**

    TODO

        '''
        c = self.coordinates

        if identity is not None:
            c = c(identity)
            if not c:
                da_key = self.domain_axis(identity, key=True, default=None)
                if da_key is not None:
                    c = self.coordinates.filter_by_axis('exact', da_key)
        # --- End: if

        if key:
            return c.key(default=default)

        return c.value(default=default)

    def coordinate_reference(self, identity=None,
                             default=ValueError(), key=False):
        '''Return a coordinate reference construct, or its key.

    .. versionadded:: 3.0.0

    .. seealso:: `construct`, `auxiliary_coordinate`, `cell_measure`,
                 `cell_method`, `coordinate`, `coordinate_references`,
                 `dimension_coordinate`, `domain_ancillary`,
                 `domain_axis`, `field_ancillary`

    :Parameters:

        identity: optional
            Select the coordinate reference construct by one of:

            * The identity of a coordinate reference construct.

              {{construct selection identity}}

            * The key of a coordinate reference construct

            * `None`. This is the default, which selects the
              coordinate reference construct when there is only one of
              them.

            *Parameter example:*
              ``identity='standard_name:atmosphere_hybrid_height_coordinate'``

            *Parameter example:*
              ``identity='grid_mapping_name:rotated_latitude_longitude'``

            *Parameter example:*
              ``identity='transverse_mercator'``

            *Parameter example:*
              ``identity='coordinatereference1'``

            *Parameter example:*
              ``identity='key%coordinatereference1'``

            *Parameter example:*
              ``identity='ncvar%lat_lon'``

            *Parameter example:*
              ``identity=cf.eq('rotated_pole')'``

            *Parameter example:*
              ``identity=re.compile('^rotated')``

        key: `bool`, optional
            If True then return the selected construct key. By
            default the construct itself is returned.

        default: optional
            Return the value of the *default* parameter if a construct
            can not be found.

            {{default Exception}}

    :Returns:

        `CoordinateReference` or `str`
            The selected coordinate reference construct, or its key.

    **Examples:**

    TODO

        '''
        c = self.coordinate_references

        if identity is not None:
            c = c.filter_by_identity(identity)
            for cr_key, cr in self.coordinate_references.items():
                if cr.match(identity):
                    c._set_construct(cr, key=cr_key, copy=False)
        # --- End: if

        if key:
            return c.key(default=default)

        return c.value(default=default)

    def coordinate_reference_domain_axes(self, identity=None):
        '''Return the domain axes that apply to a coordinate reference
    construct.

    :Parameters:

        identity: optional
            Select the coordinate reference construct by one of:

            * The identity of a coordinate reference construct.

              {{construct selection identity}}

            * The key of a coordinate reference construct

            * `None`. This is the default, which selects the
              coordinate reference construct when there is only one of
              them.

            *Parameter example:*
              ``identity='standard_name:atmosphere_hybrid_height_coordinate'``

            *Parameter example:*
              ``identity='grid_mapping_name:rotated_latitude_longitude'``

            *Parameter example:*
              ``identity='transverse_mercator'``

            *Parameter example:*
              ``identity='coordinatereference1'``

            *Parameter example:*
              ``identity='key%coordinatereference1'``

            *Parameter example:*
              ``identity='ncvar%lat_lon'``

            *Parameter example:*
              ``identity=cf.eq('rotated_pole')'``

            *Parameter example:*
              ``identity=re.compile('^rotated')``

    :Returns:

        `set`
            The identifiers of the domain axis constructs that san
            the data of all coordinate and domain ancillary constructs
            used by the selected coordinate reference construct.

    **Examples:**

    >>> f.coordinate_reference_domain_axes('coordinatereference0')
    {'domainaxis0', 'domainaxis1', 'domainaxis2'}

    >>> f.coordinate_reference_domain_axes(
    ...     'atmosphere_hybrid_height_coordinate')
    {'domainaxis0', 'domainaxis1', 'domainaxis2'}

        '''
        cr = self.coordinate_reference(identity)

        domain_axes = tuple(self.domain_axes)
        data_axes = self.constructs.data_axes()

        axes = []
        for i in cr.coordinates() | set(
                cr.coordinate_conversion.domain_ancillaries().values()):
            i = self.construct_key(i, None)
            axes.extend(data_axes.get(i, ()))

        return set(axes)

    def dimension_coordinate(self, identity=None, key=False,
                             default=ValueError()):
        '''Return a dimension coordinate construct, or its key.

    .. versionadded:: 3.0.0

    .. seealso:: `construct`, `auxiliary_coordinate`, `cell_measure`,
                 `cell_method`, `coordinate_reference`,
                 `dimension_coordinates`, `domain_ancillary`,
                 `domain_axis`, `field_ancillary`

    :Parameters:

        identity: optional
            Select the dimension coordinate construct by one of:

            * The identity of a dimension coordinate construct.

              {{construct selection identity}}

            * The key of a dimension coordinate construct

            * `None`. This is the default, which selects the dimension
              coordinate construct when there is only one of them.

            *Parameter example:*
              ``identity='Y'``

            *Parameter example:*
              ``identity='latitude'``

            *Parameter example:*
              ``identity='long_name=Latitude'``

            *Parameter example:*
              ``identity='dimensioncoordinate1'``

            *Parameter example:*
              ``identity='ncdim%y'``

            *Parameter example:*
              ``identity=cf.eq('time')'``

            *Parameter example:*
              ``identity=re.compile('^lat')``

        key: `bool`, optional
            If True then return the selected construct key. By default
            the construct itself is returned.

        default: optional
            Return the value of the *default* parameter if a construct
            can not be found.

            {{default Exception}}

    :Returns:

        `DimensionCoordinate` or `str`
            The selected dimension coordinate construct, or its key.

    **Examples:**

    TODO

        '''
        c = self.dimension_coordinates

        if identity is not None:
            c = c(identity)
            if not c:
                da_key = self.domain_axis(identity, key=True, default=None)
                if da_key is not None:
                    c = self.dimension_coordinates.filter_by_axis(
                        'exact', da_key)
        # --- End: if

        if key:
            return c.key(default=default)

        return c.value(default=default)

    @_deprecated_kwarg_check('axes')
    def direction(self, identity=None, axes=None, **kwargs):
        '''Whether or not a domain axis is increasing.

    An domain axis is considered to be increasing if its dimension
    coordinate values are increasing in index space or if it has no
    dimension coordinate.

    .. seealso:: `directions`

    :Parameters:

        identity: optional
            Select the domain axis construct by one of:
            
            * An identity or key of a 1-d dimension or auxiliary
              coordinate construct that whose data spans the domain
              axis construct.
            
               {{construct selection identity}}
            
            * A domain axis construct identity
            
              {{construct selection identity}}
            
            * The integer position of the domain axis construct in the
              field construct's data.
            
            * `None`. This is the default, which selects the domain
               construct when there is only one of them.

        axes: deprecated at version 3.0.0
            Use the *identity* parmeter instead.

        size:  deprecated at version 3.0.0

        kwargs: deprecated at version 3.0.0

    :Returns:

        `bool`
            Whether or not the domein axis is increasing.

    **Examples:**

    >>> print(f.dimension_coordinate('X').array)
    array([  0  30  60])
    >>> f.direction('X')
    True
    >>> g = f.flip('X')
    >>> g.direction('X')
    False

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'direction', kwargs)  # pragma: no cover

        axis = self.domain_axis(identity, key=True, default=None)
        if axis is None:
            return True

        for key, coord in self.dimension_coordinates.items():
            if axis == self.get_data_axes(key)[0]:
                return coord.direction()
        # --- End: for

        return True

    def directions(self):
        '''Return a dictionary mapping all domain axes to their directions.

    .. seealso:: `direction`

    :Returns:

        `dict`
            A dictionary whose key/value pairs are domain axis keys
            and their directions.

    **Examples:**

    >>> d.directions()
    {'domainaxis1': True, 'domainaxis1': False}

        '''
        out = {key: True for key in self.domain_axes.keys()}

        for key, dc in self.dimension_coordinates.items():
            direction = dc.direction()
            if not direction:
                axis = self.get_data_axes(key)[0]
                out[axis] = dc.direction()
        # --- End: for

        return out

    def domain_ancillary(self, identity=None, default=ValueError(),
                         key=False):
        '''Return a domain ancillary construct, or its key.

    .. versionadded:: 3.0.0

    .. seealso:: `construct`, `auxiliary_coordinate`, `cell_measure`,
                 `cell_method`, `coordinate`, `coordinate_reference`,
                 `dimension_coordinate`, `domain_ancillaries`,
                 `domain_axis`, `field_ancillary`

    :Parameters:

        identity: optional
            Select the domain ancillary construct by one of:

            * A domin ancillary construct identity

              {{construct selection identity}}

            * The key of a domain ancillary construct

            * `None`. This is the default, which selects the domain
              construct when there is only one of them.

            *Parameter example:*
              ``identity='Y'``

            *Parameter example:*
              ``identity='latitude'``

            *Parameter example:*
              ``identity='long_name=Latitude'``

            *Parameter example:*
              ``identity='domainancillary1'``

            *Parameter example:*
              ``identity='ncdim%y'``

            *Parameter example:*
              ``identity=cf.eq('latitude')'``

            *Parameter example:*
              ``identity=re.compile('^lat')``

        key: `bool`, optional
            If True then return the selected construct key. By
            default the construct itself is returned.

        default: optional
            Return the value of the *default* parameter if a construct
            can not be found.

            {{default Exception}}

    :Returns:

        `DomainAncillary` or `str`
            The selected domain ancillary coordinate construct, or its
            key.

    **Examples:**

    TODO

        '''
        c = self.domain_ancillaries

        if identity is not None:
            c = c(identity)
            if not c:
                da_key = self.domain_axis(identity, key=True, default=None)
                if da_key is not None:
                    c = self.domain_ancillaries.filter_by_axis(
                        'exact', da_key)
        # --- End: if

        if key:
            return c.key(default=default)

        return c.value(default=default)

    def get_coordinate_reference(self, identity=None, key=False,
                                 construct=None, default=ValueError()):
        '''TODO

    .. versionadded:: 3.0.2

    .. seealso:: `construct`

    :Parameters:

        identity: optional
            Select the coordinate reference construct by one of:

            * The identity of a coordinate reference construct.

              {{construct selection identity}}

            * The key of a coordinate reference construct

            * `None`. This is the default, which selects the
              coordinate reference construct when there is only one of
              them.

            *Parameter example:*
              ``identity='standard_name:atmosphere_hybrid_height_coordinate'``

            *Parameter example:*
              ``identity='grid_mapping_name:rotated_latitude_longitude'``

            *Parameter example:*
              ``identity='transverse_mercator'``

            *Parameter example:*
              ``identity='coordinatereference1'``

            *Parameter example:*
              ``identity='key%coordinatereference1'``

            *Parameter example:*
              ``identity='ncvar%lat_lon'``

            *Parameter example:*
              ``identity=cf.eq('rotated_pole')'``

            *Parameter example:*
              ``identity=re.compile('^rotated')``

        construct: optional
            TODO

        key: `bool`, optional
            If True then return the selected construct key. By
            default the construct itself is returned.

        default: optional
            Return the value of the *default* parameter if a construct
            can not be found.

            {{default Exception}}

    :Returns:

        `CoordinateReference` or `str`
            The selected coordinate reference construct, or its key.

    **Examples:**

    TODO
        '''
        if construct is None:
            return self.coordinate_reference(identity=identity,
                                             key=key, default=default)

        out = []

        c_key = self.construct(construct, key=True, default=None)
        if c_key is None:
            return self._default(
                default,
                "Can't identify construct from {!r}".format(construct))

        for cr_key, ref in tuple(self.coordinate_references.items()):
            if c_key in (
                    ref.coordinates(),
                    ref.coordinate_conversion.domain_ancillaries().values()
            ):
                if key:
                    if cr_key not in out:
                        out.append(cr_key)
                elif ref not in out:
                    out.append(ref)

                continue
        # --- End: for

        return out

    def has_construct(self, identity=None):
        '''Whether a metadata construct exists.

    .. versionadded:: 3.4.0

    .. seealso:: `construct`, `del_construct`, `get_construct`,
                 `set_construct`

    :Parameters:

        identity: optional
            Select the metadata construct by one of:

            * The identity of a metadata construct.

              {{construct selection identity}}

            * The key of a metadata construct

            * `None`. This is the default, which selects the metadata
              construct when there is only one of them.

            *Parameter example:*
              ``identity='T'

            *Parameter example:*
              ``identity='measure:area'``

            *Parameter example:*
              ``identity='cell_area'``

            *Parameter example:*
              ``identity='long_name=Cell Area'``

            *Parameter example:*
              ``identity='cellmeasure1'``

            *Parameter example:*
              ``identity=cf.eq('air_temperature')'``

            *Parameter example:*
              ``identity=re.compile('^air')``

    :Returns:

        `bool`
            `True` if the construct exists, otherwise `False`.

    **Examples:**

    >>> f = cf.example_field(0)
    >>> print(f)
    Field: specific_humidity (ncvar%q)
    ----------------------------------
    Data            : specific_humidity(latitude(5), longitude(8)) 1
    Cell methods    : area: mean
    Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : time(1) = [2019-01-01 00:00:00]

    >>> f.has_construct('T')
    True
    >>> f.has_construct('longitude')
    True
    >>> f.has_construct('Z')
    False

        '''
        return bool(self.construct(identity, default=False))

    def iscyclic(self, identity=None):
        '''Returns True if the given axis is cyclic.

    .. versionadded:: 3.TODO.0

    .. seealso:: `axis`, `cyclic`, `period`

    :Parameters:

        identity: optional
            Select the domain axis construct by one of:

            * An identity or key of a 1-d dimension or auxiliary
              coordinate construct that whose data spans the domain
              axis construct.

              {{construct selection identity}}
       
            * A domain axis construct identity.

              {{domain axis selection identity}}
        
            * The key of a domain axis construct.

            * `None`. This is the default, which selects the domain
              axis construct when there is only one of them.

            *Parameter example:*
              ``identity='time'``

            *Parameter example:*
              ``identity='domainaxis2'``

            *Parameter example:*
              ``identity='ncdim%y'``

    :Returns:

        `bool`
            True if the selected axis is cyclic, otherwise False.

    **Examples:**

    >>> f.iscyclic('X')
    True
    >>> f.iscyclic('latitude')
    False

    >>> x = f.iscyclic('long_name=Latitude')
    >>> x = f.iscyclic('dimensioncoordinate1')
    >>> x = f.iscyclic('domainaxis2')
    >>> x = f.iscyclic('key%domainaxis2')
    >>> x = f.iscyclic('ncdim%y')
    >>> x = f.iscyclic(2)

        '''
        axis = self.domain_axis(identity, key=True)
        return axis in self.cyclic()

    def match_by_rank(self, *ranks):
        '''Whether or not the number of domain axis constructs satisfies
    conditions.

    .. versionadded:: 3.0.0

    .. seealso:: `match`, `match_by_property`, `match_by_identity`,
                 `match_by_ncvar`, `match_by_construct`

    :Parameters:

        ranks: optional
            Define conditions on the number of domain axis constructs.

            A condition is one of:

              * `int`
              * a `Query` object

            The condition is satisfied if the number of domain axis
            constructs equals the condition value.

            *Parameter example:*
              To see if the field construct has 4 domain axis
              constructs: ``4``

            *Parameter example:*
              To see if the field construct has at least 3 domain axis
              constructs: ``cf.ge(3)``

    :Returns:

        `bool`
            Whether or not at least one of the conditions are met.

    **Examples:**

    >>> f.match_by_rank(3, 4)

    >>> f.match_by_rank(cf.wi(2, 4))

    >>> f.match_by_rank(1, cf.gt(3))

        '''
        if not ranks:
            return True

        n_domain_axes = len(self.domain_axes)
        for rank in ranks:
            ok = (rank == n_domain_axes)
            if ok:
                return True
        # --- End: for

        return False

    def replace_construct(self, identity, construct, copy=True):
        '''Replace a metadata construct.

    Replacement assigns the same construct key and, if applicable, the
    domain axes of the original construct to the new, replacing
    construct.

    .. versionadded:: 3.0.0

    .. seealso:: `set_construct`

    :Parameters:

        identity: optional
            Select the construct by one of

            * A metadata construct identity.

              {{construct selection identity}}

            * The key of a metadata construct

            * `None`. This is the default, which selects the metadata
              construct when there is only one of them.

            *Parameter example:*
              ``identity='latitude'``

            *Parameter example:*
              ``identity='T'

            *Parameter example:*
              ``identity='long_name=Cell Area'``

            *Parameter example:*
              ``identity='cellmeasure1'``

            *Parameter example:*
              ``identity='measure:area'``

            *Parameter example:*
              ``identity=cf.eq('time')'``

            *Parameter example:*
              ``identity=re.compile('^lat')``

        construct:
           The new construct to replace the one selected by the
           *identity* parameter.

        copy: `bool`, optional
            If True then set a copy of the new construct. By default
            the construct is copied.

    :Returns:

            The construct that was replaced.

    **Examples:**

    >>> f.replace_construct('X', new_X_construct)

        '''
        key = self.construct(identity, key=True, default=ValueError('TODO a'))
        c = self.constructs[key]

        set_axes = True

        if not isinstance(construct, c.__class__):
            raise ValueError('TODO')

        axes = self.get_data_axes(key, None)
        if axes is not None:
            shape0 = getattr(c, 'shape', None)
            shape1 = getattr(construct, 'shape', None)
            if shape0 != shape1:
                raise ValueError('TODO')
        # --- End: if

        self.set_construct(construct, key=key, axes=axes, copy=copy)

        return c

    def set_coordinate_reference(self, coordinate_reference, key=None,
                                 field=None, strict=True):
        '''Set a coordinate reference construct.

    By default, this is equivalent to using the `set_construct`
    method. If, however, the *field* parameter has been set then it is
    assumed to be a field construct that contains the new coordinate
    reference construct. In this case, existing coordinate and domain
    ancillary constructs will be referenced by the inserted coordinate
    reference construct, based on those which are referenced from the
    other parent field construct (given by the *field* parameter).

    .. versionadded:: 3.0.0

    .. seealso:: `set_construct`

    :Parameters:

        coordinate_reference: `CoordinateReference`
            The coordinate reference construct to be inserted.

        key: `str`, optional
            The construct identifier to be used for the construct. If
            not set then a new, unique identifier is created
            automatically. If the identifier already exisits then the
            exisiting construct will be replaced.

            *Parameter example:*
              ``key='coordinatereference1'``

        field: `Field`, optional
            A parent field construct that contains the new coordinate
            reference construct.

        strict: `bool`, optional
            If False then allow non-strict identities for
            identifying coordinate and domain ancillary metadata
            constructs.

    :Returns:

        `str`
            The construct identifier for the coordinate refernece
            construct.

        '''
        if field is None:
            return self.set_construct(
                coordinate_reference, key=key, copy=True)

        # Still here?
        ref = coordinate_reference.copy()

        ckeys = []
        for value in coordinate_reference.coordinates():
            if value in field.coordinates:
                identity = field.coordinates[value].identity(strict=strict)
                ckeys.append(
                    self.coordinate(identity, key=True, default=None))
        # --- End: for

        ref.clear_coordinates()
        ref.set_coordinates(ckeys)

        coordinate_conversion = coordinate_reference.coordinate_conversion

        dakeys = {}
        for term, value in coordinate_conversion.domain_ancillaries().items():
            if value in field.domain_ancillaries:
                identity = field.domain_ancillaries[value].identity(
                    strict=strict)
                dakeys[term] = self.domain_ancillary(
                    identity, key=True, default=None)
            else:
                dakeys[term] = None
        # --- End: for

        ref.coordinate_conversion.clear_domain_ancillaries()
        ref.coordinate_conversion.set_domain_ancillaries(dakeys)

        return self.set_construct(ref, key=key, copy=False)

    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    def aux(self, identity, default=ValueError(), key=False, **kwargs):
        '''Alias for `cf.{{class}}.auxiliary_coordinate`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'aux', kwargs,
                "Use methods of the 'auxiliary_coordinates' attribute instead."
            )  # pragma: no cover

        return self.auxiliary_coordinate(identity, key=key, default=default)

    def auxs(self, *identities, **kwargs):
        '''Alias for `cf.{{class}}.auxiliary_coordinates`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'auxs', kwargs,
                "Use methods of the 'auxiliary_coordinates' attribute "
                "instead."
            )  # pragma: no cover

        for i in identities:
            if isinstance(i, dict):
                _DEPRECATION_ERROR_DICT()  # pragma: no cover
            elif isinstance(i, (list, tuple, set)):
                _DEPRECATION_ERROR_SEQUENCE(i)  # pragma: no cover
            elif isinstance(i, str) and ':' in i:
                error = True
                if '=' in i:
                    index0 = i.index('=')
                    index1 = i.index(':')
                    error = index0 > index1

                if error:
                    _DEPRECATION_ERROR(
                        "The identity format {!r} has been deprecated at "
                        "version 3.0.0. Try {!r} instead.".format(
                            i, i.replace(':', '=', 1))
                    )  # pragma: no cover
        # --- End: for

        return self.auxiliary_coordinates(*identities)

    def axis(self, identity, key=False, default=ValueError(), **kwargs):
        '''Alias of `cf.{{class}}.domain_axis`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'axis', kwargs,
                "Use methods of the 'domain_axes' attribute instead."
            )  # pragma: no cover

        return self.domain_axis(identity, key=key, default=default)

    def coord(self, identity, default=ValueError(), key=False,
              **kwargs):
        '''Alias for `cf.{{class}}.coordinate`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'coord', kwargs,
                "Use methods of the 'coordinates' attribute instead."
            )  # pragma: no cover

        if identity in self.domain_axes:
            # Allow an identity to be the domain axis construct key
            # spanned by a dimension coordinate construct
            return self.dimension_coordinate(
                identity, key=key, default=default)

        return self.coordinate(identity, key=key, default=default)

    def coords(self, *identities, **kwargs):
        '''Alias for `cf.{{class}}.coordinates`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'coords', kwargs,
                "Use methods of the 'coordinates' attribute instead."
            )  # pragma: no cover

        for i in identities:
            if isinstance(i, dict):
                _DEPRECATION_ERROR_DICT()  # pragma: no cover
            elif isinstance(i, (list, tuple, set)):
                _DEPRECATION_ERROR_SEQUENCE(i)  # pragma: no cover
            elif isinstance(i, str) and ':' in i:
                error = True
                if '=' in i:
                    index0 = i.index('=')
                    index1 = i.index(':')
                    error = index0 > index1

                if error:
                    _DEPRECATION_ERROR(
                        "The identity format {!r} has been deprecated at "
                        "version 3.0.0. Try {!r} instead.".format(
                            i, i.replace(':', '=', 1))
                    )  # pragma: no cover
        # --- End: for

        return self.coordinates.filter_by_identity(*identities)

    def dim(self, identity, default=ValueError(), key=False, **kwargs):
        '''Alias for `cf.{{class}}.dimension_coordinate`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'dim', kwargs,
                "Use methods of the 'dimension_coordinates' attribute "
                "instead."
            )  # pragma: no cover

        return self.dimension_coordinate(identity, key=key, default=default)

    def dims(self, *identities, **kwargs):
        '''Alias for `cf.{{class}}.dimension_coordinates`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'dims', kwargs,
                "Use methods of the 'dimension_coordinates' attribute "
                "instead."
            )  # pragma: no cover

        for i in identities:
            if isinstance(i, dict):
                _DEPRECATION_ERROR_DICT()  # pragma: no cover
            elif isinstance(i, (list, tuple, set)):
                _DEPRECATION_ERROR_SEQUENCE(i)  # pragma: no cover
            elif isinstance(i, str) and ':' in i:
                error = True
                if '=' in i:
                    index0 = i.index('=')
                    index1 = i.index(':')
                    error = index0 > index1

                if error:
                    _DEPRECATION_ERROR(
                        "The identity format {!r} has been deprecated at "
                        "version 3.0.0. Try {!r} instead.".format(
                            i, i.replace(':', '=', 1))
                    )  # pragma: no cover
        # --- End: for

        return self.dimension_coordinates.filter_by_identity(*identities)

    def domain_anc(self, identity, default=ValueError(), key=False,
                   **kwargs):
        '''Alias for `cf.{{class}}.domain_ancillary`.
        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'domain_anc', kwargs,
                "Use methods of the 'domain_ancillaries' attribute "
                "instead."
            )  # pragma: no cover

        return self.domain_ancillary(identity, key=key, default=default)

    def domain_ancs(self, *identities, **kwargs):
        '''Alias for `cf.{{class}}.domain_ancillaries`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'domain_ancs', kwargs,
                "Use methods of the 'domain_ancillaries' attribute "
                "instead."
            )  # pragma: no cover

        for i in identities:
            if isinstance(i, dict):
                _DEPRECATION_ERROR_DICT()  # pragma: no cover
            elif isinstance(i, (list, tuple, set)):
                _DEPRECATION_ERROR_SEQUENCE(i)  # pragma: no cover
            elif isinstance(i, str) and ':' in i:
                error = True
                if '=' in i:
                    index0 = i.index('=')
                    index1 = i.index(':')
                    error = index0 > index1

                if error:
                    _DEPRECATION_ERROR(
                        "The identity format {!r} has been deprecated at "
                        "version 3.0.0. Try {!r} instead.".format(
                            i, i.replace(':', '=', 1))
                    )  # pragma: no cover
        # --- End: for

        return self.domain_ancillaries.filter_by_identity(*identities)

    def key(self, identity, default=ValueError(), **kwargs):
        '''Alias for `cf.{{class}}.construct_key`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'key', kwargs,
                "Use 'construct' method or 'construct_key' method instead."
            )  # pragma: no cover

        greturn self.construct_key(identity, default=default)

    def measure(self, identity, default=ValueError(), key=False,
                **kwargs):
        '''Alias for `cf.{{class}}.cell_measure`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'measure', kwargs,
                "Use methods of the 'cell_measures' attribute instead"
            )  # pragma: no cover

        return self.cell_measure(identity, key=key, default=default)

    def measures(self, *identities, **kwargs):
        '''Alias for `cf.{{class}}.cell_measures`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'measures', kwargs,
                "Use methods of the 'cell_measures' attribute instead"
            )  # pragma: no cover

        for i in identities:
            if isinstance(i, dict):
                _DEPRECATION_ERROR_DICT()  # pragma: no cover
            elif isinstance(i, (list, tuple, set)):
                _DEPRECATION_ERROR_SEQUENCE(i)  # pragma: no cover
            elif isinstance(i, str) and ':' in i:
                error = True
                if '=' in i:
                    index0 = i.index('=')
                    index1 = i.index(':')
                    error = index0 > index1

                if error and i.startswith('measure:'):
                    error = False

                if error:
                    _DEPRECATION_ERROR(
                        "The identity format {!r} has been deprecated at "
                        "version 3.0.0. Try {!r} instead.".format(
                            i, i.replace(':', '=', 1))
                    )  # pragma: no cover
        # --- End: for

        return self.cell_measures(*identities)

    def ref(self, identity, default=ValueError(), key=False, **kwargs):
        '''Alias for `cf.{{class}}.coordinate_reference`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'ref', kwargs,
                "Use methods of the 'coordinate_references' attribute "
                "instead."
            )  # pragma: no cover

        return self.coordinate_reference(identity, key=key, default=default)

    def refs(self, *identities, **kwargs):
        '''Alias for `cf.{{class}}.coordinate_references`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'refs', kwargs,
                "Use methods of the 'coordinate_references' attribute "
                "instead."
            )  # pragma: no cover

        for i in identities:
            if isinstance(i, dict):
                _DEPRECATION_ERROR_DICT()  # pragma: no cover
            elif isinstance(i, (list, tuple, set)):
                _DEPRECATION_ERROR_SEQUENCE(i)  # pragma: no cover
        # --- End: for

        return self.coordinate_references(*identities)

# --- End: class
