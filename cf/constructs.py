import logging

import cfdm

from numpy import ndarray as numpy_ndarray
from numpy import ones as numpy_ones

from . import mixin

from .data import Data
from .query import Query

from .functions import parse_indices

logger = logging.getLogger(__name__)


class Constructs(mixin.ConstructsMixin,
                 cfdm.Constructs):
    '''A container for metadata constructs.

    The following metadata constructs can be included:

    * auxiliary coordinate constructs
    * coordinate reference constructs
    * cell measure constructs
    * dimension coordinate constructs
    * domain ancillary constructs
    * domain axis constructs
    * cell method constructs
    * field ancillary constructs

    The container may be used by `Field` and `Domain` instances. In
    the latter case cell method and field ancillary constructs must be
    flagged as "ignored" (see the *_ignore* parameter).

    The container is like a dictionary in many ways, in that it stores
    key/value pairs where the key is the unique construct key with
    correspondaing metadata construct value, and provides some of the
    usual dictionary methods.

    **Calling**

    Calling a `Constructs` instance selects metadata constructs by
    identity and is an alias for the `filter_by_identity` method. For
    example, to select constructs that have an identity of
    'air_temperature': ``d = c('air_temperature')``.

    .. versionadded:: 3.0.0

    '''

    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)

    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------
    def _create_auxiliary_mask_component(self, mask_shape, ind,
                                         compress):
        '''TODO

    :Parameters:

        ndim: `int`
            TODO

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

        return Data(auxiliary_mask)

    def _matching_values(self, value0, construct, value1):
        '''TODO
        '''
        if isinstance(value0, Query):
            return value0.evaluate(value1)

        return super()._matching_values(value0, construct, value1)

    def _indices(self, mode, **kwargs):
        '''Create indices that define a subspace of the field construct.

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

    .. seealso:: `subspace`, `where`, `__getitem__`, `__setitem__`

    :Parameters:

        mode: `str`, *optional*
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
        envelope = (mode == 'envelope')
        full = (mode == 'full')
        compress = (mode == 'compress')

        logger.debug(
            "indices:\n"
            "    envelope, full, compress = {} {} {}\n".format(
                envelope, full, compress
            )
        )  # pragma: no cover

        domain_axes = self.filter_by_type('domain_axis')
        constructs = self.filter_by_data()

#        domain_rank = self.rank
        
#        ordered_axes = self._parse_axes(axes)
        ordered_axes = sorted(domain_axes)
#        if ordered_axes is None or len(ordered_axes) != domain_rank:
#            raise ValueError(
#                "Must provide an ordered sequence of all domain axes "
#                "as the last positional argument. Got {!r}".format(axes)
#            )
        
        domain_shape = tuple(
            [domain_axes[axis].get_size(None) for axis in ordered_axes]
        )
        if None in domain_shape:
            raise ValueError(
                "Can't find indices when a domain axis has no size"
            )

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

                axes = self.data_axes()[key]

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
                        size = self[axis].get_size()
                        d = Data(list(range(size)))
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
                        d = Data(list(range(size)))
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
                if axis in ordered_axes:
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
                        n_items, constructs, len(points), points, domain_shape
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

                for i, (axis, start, stop) in enumerate(zip(
                        canonical_axes, ind.min(axis=1), ind.max(axis=1))
                ):
                    if axis not in ordered_axes:
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

                create_mask = (ind.shape[1] < masked_subspace_size)
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
            )

#        indices = parse_indices(domain_shape, tuple(indices))

        # Include the auxiliary mask
        indices = {
            'indices': indices,
            'axuiliary_mask': auxiliary_mask,
        }

        # Return the indices and the auxiliary mask
        return indices

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def filter_by_identity(self, *identities):
        '''Select metadata constructs by identity.

    .. versionadded:: 3.0.0

    .. seealso:: `filter_by_axis`, `filter_by_data`, `filter_by_key`,
                 `filter_by_measure`, `filter_by_method`,
                 `filter_by_naxes`, `filter_by_ncdim`,
                 `filter_by_ncvar`, `filter_by_property`,
                 `filter_by_size`, `filter_by_type`,
                 `filters_applied`, `inverse_filter`, `unfilter`

    :Parameters:

        identities: optional
            Select constructs that have any of the given identities or
            construct keys.

            An identity is specified by a string (e.g. ``'latitude'``,
            ``'long_name=time'``, etc.); or a compiled regular
            expression (e.g. ``re.compile('^atmosphere')``), for which
            all constructs whose identities match (via `re.search`)
            are selected.

            If no identities are provided then all constructs are selected.

            Each construct has a number of identities, and is selected
            if any of them match any of those provided. A construct's
            identities are those returned by its `!identities`
            method. In the following example, the construct ``x`` has
            five identities:

               >>> x.identities()
               ['time', 'long_name=Time', 'foo=bar', 'T', 'ncvar%t']

            A construct key may optionally have the ``'key%'``
            prefix. For example ``'dimensioncoordinate2'`` and
            ``'key%dimensioncoordinate2'`` are both acceptable keys.

            Note that the identifiers of a metadata construct in the
            output of a `print` or `!dump` call are always one of its
            identities, and so may always be used as an *identities*
            argument.

            Domain axis constructs may also be identified by their
            position in the field construct's data array. Positions
            are specified by either integers.

            .. note:: This is an extension to the functionality of
                      `cfdm.Constucts.filter_by_identity`.

    :Returns:

        `Constructs`
            The selected constructs and their construct keys.

    **Examples:**

    Select constructs that have a "standard_name" property of
    'latitude':

    >>> d = c.filter_by_identity('latitude')

    Select constructs that have a "long_name" property of 'Height':

    >>> d = c.filter_by_identity('long_name=Height')

    Select constructs that have a "standard_name" property of
    'latitude' or a "foo" property of 'bar':

    >>> d = c.filter_by_identity('latitude', 'foo=bar')

    Select constructs that have a netCDF variable name of 'time':

    >>> d = c.filter_by_identity('ncvar%time')

        '''
        # Allow keys without the 'key%' prefix
        identities = list(identities)
        for n, identity in enumerate(identities):
            if identity in self:
                identities[n] = 'key%'+identity
        # --- End: for

        return super().filter_by_identity(*identities)

# --- End: class
