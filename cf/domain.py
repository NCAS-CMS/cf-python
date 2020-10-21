import cfdm

from . import mixin

from .data import Data

from .functions import parse_indices

from .decorators import (_inplace_enabled,
                         _inplace_enabled_define_and_cleanup,
                         _manage_log_level_via_verbosity)


class Domain(mixin.FieldDomainMixin,
             mixin.ConstructsMixin,
             mixin.Properties,
             cfdm.Domain):
    '''A domain construct of the CF data model.

    The domain represents a set of discrete "locations" in what
    generally would be a multi-dimensional space, either in the real
    world or in a model's simulated world. The data array elements of
    a field construct correspond to individual location of a domain.

    The domain construct is defined collectively by the following
    constructs of the CF data model: domain axis, dimension
    coordinate, auxiliary coordinate, cell measure, coordinate
    reference, and domain ancillary constructs; as well as properties
    to describe the domain.

    '''
    # ----------------------------------------------------------------
    # Private attributes
    # ----------------------------------------------------------------
    @property
    def _cyclic(self):
        '''Storage for axis cyclicity

        '''
        return self._custom.get('_cyclic', set())

    @_cyclic.setter
    def _cyclic(self, value): self._custom['_cyclic'] = value

    @_cyclic.deleter
    def _cyclic(self): del self._custom['_cyclic']

    def subspace(self):
        '''TODO

        '''
        logger.debug(
            "{}.__getitem__\n"
            "    input indices  = {}".format(
                self.__class__.__name__, indices
            )
        )  # pragma: no cover
        
        if indices is Ellipsis:
            return self.copy()

        data = self.data
        shape = data.shape

        # Parse the index
        if not isinstance(indices, tuple):
            indices = (indices,)

        if isinstance(indices[0], str) and indices[0] == 'mask':
            auxiliary_mask = indices[:2]
            indices2 = indices[2:]
        else:
            auxiliary_mask = None
            indices2 = indices

        indices, roll = parse_indices(shape, indices2, cyclic=True)

        logger.debug(
            "    parsed indices = {}\n"
            "    roll           = {}".format(
                indices, roll
            )
        )  # pragma: no cover
        
        if roll:
            new = self
            axes = self.get_data_axes()
            cyclic_axes = self.cyclic()
            for iaxis, shift in roll.items():                
                axis = axes[iaxis]
                if axis not in cyclic_axes:
                    raise IndexError(
                        "Can't take a cyclic slice from non-cyclic {!r} "
                        "axis".format(
                            self.constructs.domain_axis_identity(axis))
                    )

                new = new.roll(iaxis, shift)
        else:
            new = self.copy()

        # ------------------------------------------------------------
        # Subspace the field construct's data
        # ------------------------------------------------------------
        if auxiliary_mask:
            auxiliary_mask = list(auxiliary_mask)
            findices = auxiliary_mask + indices
        else:
            findices = indices

        logger.debug(
            "    shape          = {}\n"
            "    indices        = {}\n"
            "    indices2       = {}\n"
            "    findices       = {}".format(
                shape, indices, indices2, findices
            )
        )  # pragma: no cover

        new_data = new.data[tuple(findices)]

        # Set sizes of domain axes
        data_axes = new.get_data_axes()
        domain_axes = new.domain_axes
        for axis, size in zip(data_axes, new_data.shape):
            domain_axes[axis].set_size(size)

        # ------------------------------------------------------------
        # Subspace constructs with data
        # ------------------------------------------------------------
        if data_axes:
            construct_data_axes = new.constructs.data_axes()

            for key, construct in new.constructs.filter_by_axis(
                    'or', *data_axes).items():                
#                construct_axes = construct_data_axes[key]
                dice = [
                    indices[data_axes.index(axis)]
                    for axis in construct_data_axes[key]
                ]
#                dice = []
#                needs_slicing = False
#                for axis in construct_axes:
#                    if axis in data_axes:
#                        needs_slicing = True
#                    dice.append(indices[data_axes.index(axis)])
#                    else:
#                        dice.append(slice(None))
                # --- End: for

                # Generally we do not apply an auxiliary mask to the
                # metadata items, but for DSGs we do.
                if auxiliary_mask and new.DSG:
                    item_mask = []
                    for mask in auxiliary_mask[1]:
                        iaxes = [data_axes.index(axis) for axis in
                                 construct_axes if axis in data_axes]
                        for i, (axis, size) in enumerate(zip(
                                data_axes, mask.shape)):
                            if axis not in construct_axes:
                                if size > 1:
                                    iaxes = None
                                    break

                                mask = mask.squeeze(i)
                        # --- End: for

                        if iaxes is None:
                            item_mask = None
                            break
                        else:
                            mask1 = mask.transpose(iaxes)
                            for i, axis in enumerate(construct_axes):
                                if axis not in data_axes:
                                    mask1.inset_dimension(i)
                            # --- End: for

                            item_mask.append(mask1)
                    # --- End: for

                    if item_mask:
#                        needs_slicing = True
                        dice = [auxiliary_mask[0], item_mask] + dice
                # --- End: if

                logger.debug(
                    '    dice = {}'.format(dice))  # pragma: no cover

                # Replace existing construct with its subspace
 #               if needs_slicing:
                new.set_construct(construct[tuple(dice)], key=key,
                                  axes=construct_axes, copy=False)
        # --- End: for

#        new.set_data(new_data, axes=data_axes, copy=False)

        return new

    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)

    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------
    def _parse_axes(self, axes):
        '''Conform axes.

    :Parameters:

        axes: (sequence of) `str`
            TODO

    :Returns:

        `list`
            The conformed axes.

        '''
        if axes is None:
            return axes

        if isinstance(axes, str):
            axes = (axes,)

        domain_axes_keys = self.domain_axes.keys()

        out = []
        for axis in axes:
            if axis not in domain_axes_keys:
                raise ValueError("Invalid axis: {!r}".format(axis))
            
            out.append(axis)

        return out

#    # ----------------------------------------------------------------
#    # Attributes
#    # ----------------------------------------------------------------
#    def size(self):
#        '''TODO
#
#        '''
#        size = 1
#        for domain_axis in self.constructs.filter_by_type('domain_axis'):
#            n = domain_axis.get_size(None)
#            if n is None:
#                raise ValueError(
#                    "Can't get domain size when domain axis "
#                    "has no size: {!r}".format(domain_axis)
#                )
#
#            size *= n
#
#        return size

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def close(self):
        '''Close all files referenced by the domain construct.

    Note that a closed file will be automatically reopened if its
    contents are subsequently required.

    :Returns:

        `None`

    **Examples:**

    >>> d.close()

        '''
        self.constructs.close()

    def cyclic(self, identity=None, iscyclic=True, period=None):
        '''Set the cyclicity of an axis.

    .. versionadded:: 3.TODO.0

    .. seealso:: `autocyclic`, `domain_axis`, `iscyclic`, `period`

    :Parameters:

        identity:
           Select the domain axis construct by one of:

              * An identity or key of a 1-d coordinate construct that
                whose data spans the domain axis construct.

              * A domain axis construct identity or key.

            The *identity* parameter selects the domain axis as
            returned by this call of the `domain_axis` method:
            ``d.domain_axis(identity)``.

        iscyclic: `bool`, optional
            If False then the axis is set to be non-cyclic. By
            default the selected axis is set to be cyclic.

        period: optional
            The period for a dimension coordinate construct which
            spans the selected axis. May be any numeric scalar object
            that can be converted to a `Data` object (which includes
            numpy array and `Data` objects). The absolute value of
            *period* is used. If *period* has units then they must be
            compatible with those of the dimension coordinates,
            otherwise it is assumed to have the same units as the
            dimension coordinates.

    :Returns:

        `set`
            The construct keys of the domain axes which were cyclic
            prior to the new setting, or the current cyclic domain
            axes if no axis was specified.

    **Examples:**

    >>> f.cyclic()
    set()
    >>> f.cyclic('X', period=360)
    set()
    >>> f.cyclic()
    {'domainaxis2'}
    >>> f.cyclic('X', iscyclic=False)
    {'domainaxis2'}
    >>> f.cyclic()
    set()

        '''
        cyclic = self._cyclic
        old = cyclic.copy()
        
        if identity is None:
            return old

        axis = self.domain_axis(identity, key=True)

        if iscyclic:
            dim = self.dimension_coordinate(axis, default=None)
            if dim is not None:
                if period is not None:
                    dim.period(period)
                elif dim.period() is None:
                    raise ValueError(
                        "A cyclic dimension coordinate must have a period")
        # --- End: if

        self._cyclic = cyclic.union((axis,))

        return old

    def domain_axis(self, identity, key=False, default=ValueError()):
        '''Return a domain axis construct, or its key.

    .. versionadded:: 3.TODO.0

    .. seealso:: `construct`, `auxiliary_coordinate`, `cell_measure`,
                 `cell_method`, `coordinate`, `coordinate_reference`,
                 `dimension_coordinate`, `domain_ancillary`,
                 `domain_axes`, `field_ancillary`

    :Parameters:

        identity:
           Select the domain axis construct by one of:

              * An identity or key of a 1-d coordinate construct that
                whose data spans the domain axis construct.

              * A domain axis construct identity or key.


            A construct identity is specified by a string
            (e.g. ``'latitude'``, ``'long_name=time'``,
            ``'ncvar%lat'``, etc.); or a compiled regular expression
            (e.g. ``re.compile('^atmosphere')``) that selects the
            relevant constructs whose identities match via
            `re.search`.

            Each construct has a number of identities, and is selected
            if any of them match any of those provided. A construct's
            identities are those returned by its `!identities`
            method. In the following example, the construct ``x`` has
            six identities:

               >>> x.identities()
               ['time', 'long_name=Time', 'foo=bar', 'standard_name=time', 'ncvar%t', 'T']

            A construct key may optionally have the ``'key%'``
            prefix. For example ``'dimensioncoordinate2'`` and
            ``'key%dimensioncoordinate2'`` are both acceptable keys.

            A position of a domain axis construct in the field
            construct's data is specified by an integer index.

            Note that in the output of a `print` call or `!dump`
            method, a construct is always described by one of its
            identities, and so this description may always be used as
            an *identity* argument.

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

        key: `bool`, optional
            If True then return the selected construct key. By
            default the construct itself is returned.

        default: optional
            Return the value of the *default* parameter if a construct
            can not be found. If set to an `Exception` instance then
            it will be raised instead.

    :Returns:

        `DomainAxis` or `str`
            The selected domain axis construct, or its key.

    **Examples:**

    TODO

        '''
        domain_axes = self.domain_axes(identity)
        if len(domain_axes) == 1:
            # identity is a unique domain axis construct identity
            da_key = domain_axes.key()
        else:
            # identity is not a unique domain axis construct identity
            da_key = self.domain_axis_key(identity, default=None)

        if da_key is None:
            return self._default(
                default,
                "No unique domain axis construct is identifable from "
                "{!r}".format(identity)
            )

        if key:
            return da_key

        return domain_axes.value()

    def _indices(self, mode, axes, **kwargs):
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
            "    envelope, full, compress = {} {} {}\n"
            "    axes = {}".format(
                envelope, full, compress, axes
            )
        )  # pragma: no cover

        domain_axes = self.constructs.filter_by_type('domain_axis')
        constructs = self.constructs.filter_by_data()

        domain_rank = self.rank
        
        ordered_axes = self._parse_axes(axes)
        if ordered_axes is None or len(ordered_axes) != domain_rank:
            raise ValueError(
                "Must provide an ordered sequence of all domain axes "
                "as the last positional argument. Got {!r}".format(axes)
            )
        
        domain_shape = tuple(
            [domain_axes[axis].get_size(None) for axis in ordered_axes]
        )
        if None in domain_shape:
            raise ValueError(
                "Can't find indices when a domain axis has no size"
            )

        # Initialize indices
        indices = [slice(None)] * domain_rank

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

            sorted_axes = tuple(sorted(axes))
            if sorted_axes not in parsed:
                n_axes += len(sorted_axes)

            parsed.setdefault(sorted_axes, []).append(
                (axes, key, construct, value))

            unique_axes.update(sorted_axes)
        # --- End: for

        if len(unique_axes) < n_axes:
            raise ValueError(
                "Can't find indices: Multiple constructs with incompatible "
                "domain axes"
            )

        auxiliary_mask = []

        for sorted_axes, axes_key_construct_value in parsed.items():
            axes, keys, constructs, points = list(
                zip(*axes_key_construct_value)
            )
            n_items = len(constructs)
            n_axes = len(sorted_axes)

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
                    # -------------------------------------------------
                    logger.debug('    1-d CASE 1: ')  # pragma: no cover

                    index = value

                    if envelope or full:
                        size = self.constructs[axis].get_size()
                        d = Data(list(range(size)))
                        ind = (d[value].array,)
                        index = slice(None)

                elif (item is not None
                      and isinstance(value, Query)
                      and value.operator in ('wi', 'wo')
                      and item.construct_type == 'dimension_coordinate'
                      and self.iscyclic(axis)):
                    # self.iscyclic(sorted_axes)):
                    # ------------------------------------------------
                    # 1-dimensional CASE 2: Axis is cyclic and
                    #                       subspace criterion is a
                    #                       'within' or 'without'
                    #                       Query instance
                    # -------------------------------------------------
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
                    # -------------------------------------------------
                    # 1-dimensional CASE 3: All other 1-d cases
                    # -------------------------------------------------
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
                    indices[ordered_axes.index(axis)] = index

            else:
                # -----------------------------------------------------
                # N-dimensional constructs
                # -----------------------------------------------------
                logger.debug(
                    "    {} N-d constructs: {!r}\n"
                    "    {} points        : {!r}\n"
                    "    shape          : {}".format(
                        n_items, constructs, len(points), points, domain_shape
                    )
                )  # pragma: no cover

                # Make sure that each N-d item has the same relative
                # axis order as the field's data array.
                #
                # For example, if the data array of the field is
                # ordered T Z Y X and the item is ordered Y T then the
                # item is transposed so that it is ordered T Y. For
                # example, if the field's data array is ordered Z Y X
                # and the item is ordered X Y T (T is size 1) then
                # transpose the item so that it is ordered Y X T.
                g = self.transpose(ordered_axes, constructs=True)

                item_axes = g.get_data_axes(keys[0])

                constructs = [g.constructs[key] for key in keys]
                logger.debug(
                    "    transposed N-d constructs: {!r}".format(constructs)
                )  # pragma: no cover

                item_matches = [(value == construct).data for
                                value, construct in zip(points, constructs)]

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

                bounds = [item.bounds.array[ind] for item in constructs
                          if item.has_bounds()]

                contains = False
                if bounds:
                    points2 = []
                    for v, construct in zip(points, constructs):
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
                            "object".format(constructs[0].ndim)
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
            # --- End: if

            if ind is not None:
                mask_shape = [None] * domain_rank
                masked_subspace_size = 1
                ind = numpy_array(ind)
                logger.debug('    ind = {}'.format(ind))  # pragma: no cover

                for i, (axis, start, stop) in enumerate(zip(
                        item_axes, ind.min(axis=1), ind.max(axis=1))):
                    if axis not in ordered_axes:
                        continue

                    position = ordered_axes.index(axis)

                    if indices[position] == slice(None):
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
#                            stop = self.axis_size(axis)
                            stop = self.domain_axes[axis].get_size()
                            size = stop - start
                            index = slice(start, stop)
                        else:
                            raise ValueError(
                                "Must have full, envelope or compress"
                            )  # pragma: no cover

                        indices[position] = index

                    mask_shape[position] = size
                    masked_subspace_size *= size
                    ind[i] -= start
                # --- End: for

                create_mask = ind.shape[1] < masked_subspace_size
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

                mask = self.data._create_auxiliary_mask_component(
                    mask_shape, ind, compress)
                auxiliary_mask.append(mask)
                logger.debug(
                    "    mask_shape  = {}\n"
                    "    mask.shape  = {}".format(mask_shape, mask.shape)
                )  # pragma: no cover
        # --- End: for

        indices = tuple(parse_indices(domain_shape, tuple(indices)))

        # Convert indices to a dictionary
        indices = {axis: index
                   for axis, index in zip(ordered_axes, indices)}
        
        if auxiliary_mask:
#            indices = ('mask', auxiliary_mask) + indices
            indices['axuiliary_mask'] = auxiliary_mask

        # Return the indices and the auxiliary mask
        return indices
    
    def get_data_axes(self, identity, default=ValueError()):
        '''Return the keys of the domain axis constructs spanned by the data
    of a metadata construct.

    .. versionadded:: 3.TODO.0

    .. seealso:: `del_data_axes`, `has_data_axes`, `set_data_axes`

    :Parameters:

        identity:
           Select the construct for which to return the domain axis
           constructs spanned by its data. By default the field
           construct is selected. May be:

              * The identity or key of a metadata construct.

            A construct identity is specified by a string
            (e.g. ``'latitude'``, ``'long_name=time'``,
            ``'ncvar%lat'``, etc.); or a compiled regular expression
            (e.g. ``re.compile('^atmosphere')``) that selects the
            relevant constructs whose identities match via
            `re.search`.

            Each construct has a number of identities, and is selected
            if any of them match any of those provided. A construct's
            identities are those returned by its `!identities`
            method. In the following example, the construct ``x`` has
            six identities:

               >>> x.identities()
               ['time', 'long_name=Time', 'foo=bar', 'standard_name=time', 'ncvar%t', 'T']

            A construct key may optionally have the ``'key%'``
            prefix. For example ``'dimensioncoordinate2'`` and
            ``'key%dimensioncoordinate2'`` are both acceptable keys.

            Note that in the output of a `print` call or `!dump`
            method, a construct is always described by one of its
            identities, and so this description may always be used as
            an *identity* argument.

        default: optional
            Return the value of the *default* parameter if the data
            axes have not been set.

            {{default Exception}}

    :Returns:

        `tuple`
            The keys of the domain axis constructs spanned by the data.

    **Examples:**

    TODO

        '''
        key = self.construct(identity, key=True, default=None)
        if key is None:
            return self.construct_key(identity, default=default)
        
        return super().get_data_axes(key=key, default=default)

    @_inplace_enabled(default=False)
    def transpose(self, axes, inplace=False):
        '''Permute the axes of the data array.

TODO    By default the order of the axes is reversed, but any ordering may
    be specified by selecting the axes of the output in the required
    order.

    .. versionadded:: 3.TODO.0
        
    .. seealso:: `domain_axis`, `flatten`, `insert_dimension`, `flip`,
                 `squeeze`, `unsqueeze`

    :Parameters:

        axes: sequence of `str`
            Select the domain axis order, defined by the domain axes
            that would be selected by passing each given axis
            description to a call of the field construct's
            `domain_axis` method. For example, for a value of ``'X'``,
            the domain axis construct returned by
            ``f.domain_axis('X')`` is selected.

            Each domain axis of the domain construct data must be
            provided.

        constructs: `bool`, optional
            If True then metadata constructs are also transposed so
            that their axes are in the same relative order as in the
            transposed data array of the field. By default metadata
            constructs are not altered.

        {{inplace: `bool`, optional}}

    :Returns:

        `Domain` or `None`
            The domain construct with transposed constructs, or `None`
            if the operation was in-place.

    **Examples:**

    >>> f.ndim
    3
    >>> g = f.transpose()
    >>> g = f.transpose(['time', 1, 'dim2'])
    >>> f.transpose(['time', -2, 'dim2'], inplace=True)

        '''
        f = _inplace_enabled_define_and_cleanup(self)

        try:
            axes = f._parse_axes(axes)
        except ValueError as error:
            raise ValueError("Can't transpose domain: {}".format(error))

        if axes is None or len(axes) != len(f.domain_axes):
            raise ValueError(
                "Can't transpose domain: Must provide order for "
                "all domain axes. Got: {}".format(axes)
            )
        
        for key, construct in f.constructs.filter_by_data().items():
            construct_axes = f.get_data_axes(key)
           
            iaxes = [
                construct_axes.index(axis) for axis in axes
                if axis in construct_axes
            ]
            
            # Transpose the construct
            construct.transpose(iaxes, inplace=True)

        return f

# --- End: class
