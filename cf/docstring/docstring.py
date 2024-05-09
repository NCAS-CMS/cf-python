"""Define docstring substitutions.

Text to be replaced is specified as a key in the returned dictionary,
with the replacement text defined by the corresponding value.

Special docstring substitutions, as defined by a class's
`_docstring_special_substitutions` method, may be used in the
replacement text, and will be substituted as usual.

Replacement text may not contain other non-special substitutions.

Keys must be `str` or `re.Pattern` objects:

* If a key is a `str` then the corresponding value must be a string.

* If a key is a `re.Pattern` object then the corresponding value must
  be a string or a callable, as accepted by the `re.Pattern.sub`
  method.

.. versionadded:: 3.7.0

"""

_docstring_substitution_definitions = {
    # ----------------------------------------------------------------
    # General substitutions (not indent-dependent)
    # ----------------------------------------------------------------
    "{{repr}}": "CF ",
    # ----------------------------------------------------------------
    # Class description substitutions (1 level of indentation)
    # ----------------------------------------------------------------
    "{{formula terms links}}": """See CF section 4.3.3 "Parametric Vertical Coordinate" and CF
    Appendix D "Parametric Vertical Coordinates" for details.""",
    # ----------------------------------------------------------------
    # Class description substitutions (1 level of indentation)
    # ----------------------------------------------------------------
    #
    # ----------------------------------------------------------------
    # Method description substitutions (2 levels of indentation)
    # ----------------------------------------------------------------
    # List comparison
    "{{List comparison}}": """Each construct in the list is compared with its `!equals`
        method, rather than the ``==`` operator.""",
    # regridding overview
    "{{regridding overview}}": """Regridding is the process of interpolating the field data
        values while preserving the qualities of the original data,
        and the metadata of the unaffected axes. The metadata for the
        regridded axes are taken from the *dst* parameter.""",
    # regrid Masked cells
    "{{regrid Masked cells}}": """**Masked cells**

        By default, the data mask of the source data is taken into
        account during the regridding process, but the destination
        grid mask is not. This behaviour may be changed with the
        *use_src_mask* and *use_dst_mask* parameters.

        In general the source data may be arbitrarily masked, meaning
        that the mask for the regridding axes may vary along the
        non-regridding axes. The exceptions to this are for
        second-order conservative, patch recovery regridding, and
        nearest source to destination methods, for which the mask of
        the regridding axes must be the same across all non-regridding
        axes. In these special cases an exception will be raised if
        the source data mask does not meet this requirement.""",
    # regrid Implementation
    "{{regrid Implementation}}": """**Implementation**

        The interpolation is carried out using regridding weights
        calculated by the `esmpy` package, a Python interface to the
        Earth System Modeling Framework regridding utility (ESMF,
        https://earthsystemmodeling.org/regrid). Outside of `esmpy`,
        these weights are then modified for masked cells (if required)
        and the regridded data are created as the dot product of the
        weights with the source data. (Note that whilst the `esmpy`
        package is also able to create the regridded data from its
        weights, this feature can't be integrated with the `dask`
        framework that underpins the field's data.)""",
    # regrid Logging
    "{{regrid Logging}}": """**Logging**

        Whether `esmpy` logging is enabled or not is determined by
        `cf.regrid_logging`. If it is enabled then logging takes place
        after every call. By default logging is disabled.""",
    # subspace halos
    "{{subspace halos}}": """If a halo is defined via a positional argument, then each
        subspaced axis will be extended to include that many extra
        elements at each "side" of the axis. The number of extra
        elements will be automatically reduced if including the full
        amount defined by the halo would extend the subspace beyond
        the axis limits.""",
    # ----------------------------------------------------------------
    # Method description substitutions (3 levels of indentation)
    # ----------------------------------------------------------------
    # i: deprecated at version 3.0.0
    "{{i: deprecated at version 3.0.0}}": """i: deprecated at version 3.0.0
                Use the *inplace* parameter instead.""",
    # default_to_zero: `bool`, optional
    "{{default_to_zero: `bool`, optional}}": """default_to_zero: `bool`, optional
                If False then do not assume that missing terms have a
                value of zero. By default a missing term is assumed to
                be zero.""",
    # key: `bool`, optional
    "{{key: `bool`, optional}}": """key: `bool`, optional
                If True then return the selected construct
                identifier. By default the construct itself is
                returned.""",
    # item: `bool`, optional
    "{{item: `bool`, optional}}": """item: `bool`, optional
                If True then return the selected construct identifier
                and the construct itself. By default the construct
                itself is returned. If *key* is True then *item* is
                ignored.""",
    # method: `str`, optional
    "{{method: `str` or `None`, optional}}": """method: `str` or `None`, optional
                Specify the regridding interpolation method. This
                parameter must be set unless *dst* is a
                `RegridOperator`, when the *method* is ignored.

                The *method* parameter may be one of the following:

                * ``'linear'``: Bilinear interpolation.

                * ``'bilinear'``: Deprecated alias for ``'linear'``.

                * ``'conservative_1st'``: First order conservative
                  interpolation. Preserves the integral of the source
                  field across the regridding. Weight calculation is
                  based on the ratio of source cell area overlapped
                  with the corresponding destination cell area.

                * ``'conservative'``: Alias for ``'conservative_1st'``

                * ``'conservative_2nd'``: Second-order conservative
                  interpolation. Preserves the integral of the source
                  field across the regridding. Weight calculation is
                  based on the ratio of source cell area overlapped
                  with the corresponding destination cell area. The
                  second-order conservative calculation also includes
                  the gradient across the source cell, so in general
                  it gives a smoother, more accurate representation of
                  the source field. This is particularly true when
                  going from a coarse to finer grid.

                * ``'patch'`` Patch recovery interpolation. Patch
                  rendezvous method of taking the least squares fit of
                  the surrounding surface patches. This is a higher
                  order method that may produce interpolation weights
                  that may be slightly less than 0 or slightly greater
                  than 1. This method typically results in better
                  approximations to values and derivatives when
                  compared to bilinear interpolation.

                * ``'nearest_stod'``: Nearest neighbour source to
                  destination interpolation for which each destination
                  point is mapped to the closest source point. A
                  source point can be mapped to multiple destination
                  points. Useful for regridding categorical data.

                * ``'nearest_dtos'``: Nearest neighbour destination to
                  source interpolation for which each source point is
                  mapped to the closest destination point. A
                  destination point can be mapped to multiple source
                  points. Some destination points may not be
                  mapped. Useful for regridding of categorical data.

                * `None`: This is the default and can only be used
                  when *dst* is a `RegridOperator`.""",
    # radius: optional
    "{{radius: optional}}": """radius: optional
                Specify the radius of the latitude-longitude plane
                defined in spherical polar coordinates. The radius is
                that which would be returned by this call of the field
                construct's `radius` method:
                ``f.radius(default=radius)``. The radius is defined by
                the datum of a coordinate reference construct, and if
                and only if no such radius is found then the default
                value given by the *radius* parameter is used
                instead. A value of ``'earth'`` is equivalent to a
                default value of 6371229 metres.""",
    # chunks
    "{{chunks: `int`, `tuple`, `dict` or `str`, optional}}": """chunks: `int`, `tuple`, `dict` or `str`, optional
                Specify the chunking of the underlying dask array.

                Any value accepted by the *chunks* parameter of the
                `dask.array.from_array` function is allowed.

                By default, ``"auto"`` is used to specify the array
                chunking, which uses a chunk size in bytes defined by
                the `cf.chunksize` function, preferring square-like
                chunk shapes.

                *Parameter example:*
                  A blocksize like ``1000``.

                *Parameter example:*
                  A blockshape like ``(1000, 1000)``.

                *Parameter example:*
                  Explicit sizes of all blocks along all dimensions
                  like ``((1000, 1000, 500), (400, 400))``.

                *Parameter example:*
                  A size in bytes, like ``"100MiB"`` which will choose
                  a uniform block-like shape, preferring square-like
                  chunk shapes.

                *Parameter example:*
                  A blocksize of ``-1`` or `None` in a tuple or
                  dictionary indicates the size of the corresponding
                  dimension.

                *Parameter example:*
                  Blocksizes of some or all dimensions mapped to
                  dimension positions, like ``{1: 200}``, or ``{0: -1,
                  1: (400, 400)}``.""",
    # Returns formula
    "{{Returns formula}}": """5-`tuple`
                * The standard name of the parametric coordinates.

                * The standard name of the computed non-parametric
                  coordinates.

                * The computed non-parametric coordinates in a
                  `DomainAncillary` construct.

                * A tuple of the domain axis construct keys for the
                  dimensions of the computed non-parametric
                  coordinates.

                * A tuple containing the construct key of the vertical
                  domain axis. If the vertical axis does not appear in
                  the computed non-parametric coordinates then this an
                  empty tuple.""",
    # collapse axes
    "{{collapse axes: (sequence of) `int`, optional}}": """axes: (sequence of) `int`, optional
                The axes to be collapsed. By default all axes are
                collapsed, resulting in output with size 1. Each axis
                is identified by its integer position. If *axes* is an
                empty sequence then the collapse is applied to each
                scalar element and the result has the same shape as
                the input data.""",
    # collapse squeeze
    "{{collapse squeeze: `bool`, optional}}": """squeeze: `bool`, optional
                By default, the axes which are collapsed are left in
                the result as dimensions with size one, so that the
                result will broadcast correctly against the input
                array. If set to True then collapsed axes are removed
                from the data.""",
    # collapse keepdims
    "{{collapse keepdims: `bool`, optional}}": """keepdims: `bool`, optional
                By default, the axes which are collapsed are left in
                the result as dimensions with size one, so that the
                result will broadcast correctly against the input
                array. If set to False then collapsed axes are removed
                from the data.""",
    # weights
    "{{weights: data_like, `dict`, or `None`, optional}}": """weights: data_like, `dict`, or `None`, optional
                Weights associated with values of the data. By default
                *weights* is `None`, meaning that all non-missing
                elements of the data have a weight of 1 and all
                missing elements have a weight of 0.

                If *weights* is a data_like object then it must be
                broadcastable to the array.

                If *weights* is a dictionary then each key specifies
                axes of the data (an `int` or `tuple` of `int`), with
                a corresponding value of data_like weights for those
                axes. The dimensions of a weights value must
                correspond to its key axes in the same order. Not all
                of the axes need weights assigned to them. The weights
                that will be used will be an outer product of the
                dictionary's values.

                However they are specified, the weights are internally
                broadcast to the shape of the data, and those weights
                that are missing data, or that correspond to the
                missing elements of the data, are assigned a weight of
                0.""",
    # collapse mtol
    "{{mtol: number, optional}}": """mtol: number, optional
                The sample size threshold below which collapsed values
                are set to missing data. It is defined as a fraction
                (between 0 and 1 inclusive) of the contributing input
                data values.

                The default of *mtol* is 1, meaning that a missing
                datum in the output array occurs whenever all of its
                contributing input array elements are missing data.

                For other values, a missing datum in the output array
                occurs whenever more than ``100*mtol%`` of its
                contributing input array elements are missing data.

                Note that for non-zero values of *mtol*, different
                collapsed elements may have different sample sizes,
                depending on the distribution of missing data in the
                input data.""",
    # ddof
    "{{ddof: number}}": """ddof: number
                The delta degrees of freedom, a non-negative
                number. The number of degrees of freedom used in the
                calculation is (N-*ddof*) where N represents the
                number of non-missing elements. A value of 1 applies
                Bessel's correction. If the calculation is weighted
                then *ddof* can only be 0 or 1.""",
    # split_every
    "{{split_every: `int` or `dict`, optional}}": """split_every: `int` or `dict`, optional
                Determines the depth of the recursive aggregation. If
                set to or more than the number of input chunks, the
                aggregation will be performed in two steps, one
                partial collapse per input chunk and a single
                aggregation at the end. If set to less than that, an
                intermediate aggregation step will be used, so that
                any of the intermediate or final aggregation steps
                operates on no more than ``split_every`` inputs. The
                depth of the aggregation graph will be
                :math:`log_{split_every}(input chunks along reduced
                axes)`. Setting to a low value can reduce cache size
                and network transfers, at the cost of more CPU and a
                larger dask graph.

                By default, `dask` heuristically decides on a good
                value. A default can also be set globally with the
                ``split_every`` key in `dask.config`. See
                `dask.array.reduction` for details.""",
    # Collapse chunk_function
    "{{chunk_function: callable, optional}}": """{{chunk_function: callable, optional}}
                Provides the ``chunk`` parameter to
                `dask.array.reduction`. If unset then an approriate
                default function will be used.""",
    # Collapse weights
    "{{Collapse weights: data_like or `None`, optional}}": """weights: data_like or `None`, optional
                Weights associated with values of the array. By
                default *weights* is `None`, meaning that all
                non-missing elements of the array are assumed to have
                a weight equal to one.

                When *weights* is a data_like object then it must have
                the same shape as the array.""",
    # percentile method
    "{{percentile method: `str`, optional}}": """method: `str`, optional
                Specify the interpolation method to use when the
                percentile lies between two data values. The methods
                are listed here, but their definitions must be
                referenced from the documentation for
                `numpy.percentile`.

                For the default ``'linear'`` method, if the percentile
                lies between two adjacent data values ``i < j`` then
                the percentile is calculated as ``i+(j-i)*fraction``,
                where ``fraction`` is the fractional part of the index
                surrounded by ``i`` and ``j``.

                ===============================
                *method*
                ===============================
                ``'inverted_cdf'``
                ``'averaged_inverted_cdf'``
                ``'closest_observation'``
                ``'interpolated_inverted_cdf'``
                ``'hazen'``
                ``'weibull'``
                ``'linear'`` (default)
                ``'median_unbiased'``
                ``'normal_unbiased'``
                ``'lower'``
                ``'higher'``
                ``'nearest'``
                ``'midpoint'``
                ===============================""",
    # use_src_mask
    "{{use_src_mask: `bool`, optional}}": """use_src_mask: `bool`, optional
                By default the mask of the source field is taken into
                account during the regridding process. The only
                possible exception to this is when the nearest source
                to destination regridding method (``'nearest_stod'``)
                is being used. In this case, if *use_src_mask* is
                False then each destination point is mapped to the
                closest source point, whether or not it is masked (see
                the *method* parameter for details).

                Ignored if *dst* is a `RegridOperator`.""",
    # use_dst_mask
    "{{use_dst_mask: `bool`, optional}}": """use_dst_mask: `bool`, optional
                If *dst* is a `Field` and *use_dst_mask* is False (the
                default) then the mask of data on the destination grid
                is **not** taken into account when performing
                regridding. If *use_dst_mask* is True then any masked
                cells in the *dst* field construct are transferred to
                the result. If *dst* has more dimensions than are
                being regridded, then the mask of the destination grid
                is taken as the subspace defined by index ``0`` of all
                the non-regridding dimensions.

                Ignored if *dst* is not a `Field`.""",
    # ignore_degenerate
    "{{ignore_degenerate: `bool`, optional}}": """ignore_degenerate: `bool`, optional
                For conservative regridding methods, if True (the
                default) then degenerate cells (those for which enough
                vertices collapse to leave a cell as either a line or
                a point) are skipped, not producing a result.
                Otherwise an error will be produced if degenerate
                cells are found, that will be present in the `esmpy`
                log files.

                For all other regridding methods, degenerate cells are
                always skipped, regardless of the value of
                *ignore_degenerate*.

                Ignored if *dst* is a `RegridOperator`.""",
    # return_operator
    "{{return_operator: `bool`, optional}}": """return_operator: `bool`, optional
                If True then do not perform the regridding, rather
                return the `RegridOperator` instance that defines the
                regridding operation, and which can be used in
                subsequent calls. See the *dst* parameter for
                details.""",
    # check_coordinates
    "{{check_coordinates: `bool`, optional}}": """check_coordinates: `bool`, optional
                If True and *dst* is a `RegridOperator`then the source
                grid coordinates defined by the operator are checked
                for compatibility against those of the source field. By
                default this check is not carried out. See the *dst*
                parameter for details.

                Ignored unless *dst* is a `RegridOperator`.""",
    # min_weight
    "{{min_weight: float, optional}}": """min_weight: float, optional
                A very small non-negative number. By default
                *min_weight* is ``2.5 * np.finfo("float64").eps``,
                i.e. ``5.551115123125783e-16``. It is used during
                linear and first-order conservative regridding when
                adjusting the weights matrix to account for the data
                mask. It is ignored for all other regrid methods, or
                if data being regridded has no missing values.

                In some cases (described below) for which weights
                might only be non-zero as a result of rounding errors,
                the *min_weight* parameter controls whether or a not
                cell in the regridded field is masked.

                The default value has been chosen empirically as the
                smallest value that produces the same masks as `esmpy`
                for the use cases defined in the cf test suite.

                Define ``w_ji`` as the multiplicative weight that
                defines how much of ``Vs_i`` (the value in source grid
                cell ``i``) contributes to ``Vd_j`` (the value in
                destination grid cell ``j``).

                **Linear regridding**

                Destination grid cell ``j`` will only be masked if a)
                it is masked in the destination grid definition; or b)
                ``w_ji >= min_weight`` for those masked source grid
                cells ``i`` for which ``w_ji > 0``.

                **Conservative first-order regridding**

                Destination grid cell ``j`` will only be masked if a)
                it is masked in the destination grid definition; or b)
                the sum of ``w_ji`` for all non-masked source grid
                cells ``i`` is strictly less than *min_weight*.""",
    # weights_file
    "{{weights_file: `str` or `None`, optional}}": """weights_file: `str` or `None`, optional
                Provide a netCDF file that contains, or will contain,
                the regridding weights. If `None` (the default) then
                the weights are computed in memory for regridding
                between the source and destination grids, and no file
                is created.

                If set to a file path that does not exist then the
                weights will be computed and also written to that
                file.

                If set to a file path that already exists then the
                weights will be read from this file, instead of being
                computed.

                .. note:: No checks are performed on an existing file
                          to ensure that the weights are appropriate
                          for the source field and the values of the
                          keyword parameters. Inappropriate weights
                          will produce incorrect results.

                          However, when regridding using weights from
                          a file, ensuring that the source field has
                          the same shape over the regridding axes, and
                          the parameter settings are the same as those
                          used when the weights file was created, will
                          ensure correct results.

                          A netCDF regridding weights file created
                          directly by ESMF has the same structure and
                          variable names (``S``, ``row``, and ``col``
                          for the weights, destination/row indices,
                          and source/col indices respectively), so may
                          be provided as a *weights_file*, noting that
                          no checks will be applied to it.

                **Performance**

                The computation of the weights can be much more costly
                than the regridding itself, in which case reading
                pre-calculated weights can improve performance.

                Ignored if *dst* is a `RegridOperator`.""",
    # aggregated_units
    "{{aggregated_units: `str` or `None`, optional}}": """aggregated_units: `str` or `None`, optional
                The units of the aggregated array. Set to `None` to
                indicate that there are no units.""",
    # aggregated_calendar
    "{{aggregated_calendar: `str` or `None`, optional}}": """aggregated_calendar: `str` or `None`, optional
                The calendar of the aggregated array. Set to `None` to
                indicate the CF default calendar, if applicable.""",
    # threshold
    "{{threshold: `int`, optional}}": """threshold: `int`, optional
                The graph growth factor under which we don't bother
                introducing an intermediate step. See
                `dask.array.rechunk` for details.""",
    # block_size_limit
    "{{block_size_limit: `int`, optional}}": """block_size_limit: `int`, optional
                The maximum block size (in bytes) we want to produce,
                as defined by the `cf.chunksize` function.""",
    # balance
    "{{balance: `bool`, optional}}": """balance: `bool`, optional
                If True, try to make each chunk the same size. By
                default this is not attempted.

                This means ``balance=True`` will remove any small
                leftover chunks, so using ``d.rechunk(chunks=len(d) //
                N, balance=True)`` will almost certainly result in
                ``N`` chunks.""",
    # bounds
    "{{bounds: `bool`, optional}}": """bounds: `bool`, optional
                If True (the default) then alter any bounds.""",
    # cull
    "{{cull_graph: `bool`, optional}}": """cull_graph: `bool`, optional
                If True then unnecessary tasks are removed (culled)
                from each array's dask graph before
                concatenation. This process can have a considerable
                overhead but can sometimes improve the overall
                performance of a workflow. If False (the default) then
                dask graphs are not culled. See
                `dask.optimization.cull` for details.""",
    # relaxed_units
    "{{relaxed_units: `bool`, optional}}": """relaxed_units: `bool`, optional
                If True then allow the concatenation of data with
                invalid but otherwise equal units. By default, if any
                data array has invalid units then the concatenation
                will fail. A `Units` object is considered to be
                invalid if its `!isvalid` attribute is `False`.""",
    # cfa substitutions
    "{{cfa substitutions: `dict`}}": """substitutions: `dict`
                The substitution definitions in a dictionary whose
                key/value pairs are the file name parts to be
                substituted and their corresponding substitution text.

                Each substitution definition may be specified with or
                without the ``${...}`` syntax. For instance, the
                following are equivalent: ``{'base': 'sub'}``,
                ``{'${base}': 'sub'}``.""",
    # cfa base
    "{{cfa base: `str`}}": """base: `str`
                The substitution definition to be removed. May be
                specified with or without the ``${...}`` syntax. For
                instance, the following are equivalent: ``'base'`` and
                ``'${base}'``.""",
    # regular args
    "{{regular args}}": """A sequence of three numeric values. The first two values in
                the sequence represent the coordinate range (see the bounds
                parameter for details), and the third value represents the
                cellsize.

                .. note:: The cellsize does not have to explicitly divide into
                          the range of the given dimension. But as it follows
                          `numpy.arange` while creating the points, one should
                          verify that that the number of grid points are
                          returned as expected.""",
    # weights weights
    "{{weights weights: `dict`}}": """weights: `dict`
                A dictionary that will get updated in place with any
                created weights.""",
    # weights weights_axes
    "{{weights weights_axes: `set`}}": """weights_axes: `set`
                A `set` that will get updated in place with the domain
                axis identifiers of the weights axes.""",
    # weights methods
    "{{weights methods: `bool`, optional}}": """methods: `bool`, optional
                If True then add a description of the method used to
                create the weights to the *weights* dictionary, as
                opposed to the actual weights.""",
    # weights measure
    "{{weights measure: `bool`, optional}}": """measure: `bool`, optional
                If True then create weights that are actual cell sizes
                with appropriate units.""",
    # weights auto
    "{{weights auto: `bool`, optional}}": """auto: `bool`, optional
                If True then return `False` if weights can't be found,
                rather than raising an exception.""",
    # ln_z
    "{{ln_z: `bool` or `None`, optional}}": """ln_z: `bool` or `None`, optional
                If True when *z*, *src_z* or *dst_z* are also set,
                calculate the vertical component of the regridding
                weights using the natural logarithm of the vertical
                coordinate values. This option should be used if the
                quantity being regridded varies approximately linearly
                with logarithm of the vertical coordinates. If False,
                then the weights are calculated using unaltered
                vertical values. If `None`, the default, then an
                exception is raised if any of *z*, *src_z* or *dst_z*
                have also been set.

                Ignored if *dst* is a `RegridOperator`.""",
    # pad_width
    "{{pad_width: sequence of `int`, optional}}": """pad_width: sequence of `int`, optional
                Number of values to pad before and after the edges of
                the axis.""",
    # to_size
    "{{to_size: `int`, optional}}": """to_size: `int`, optional
                Pad the axis after so that the new axis has the given
                size.""",
    # subspace config options
    "{{config: optional}}": """config: optional
                Configure the subspace by specifying the mode of
                operation (``mode``) and any halo to be added to the
                subspaced axes (``halo``), with positional arguments
                in the format ``mode``, or ``halo``, or ``mode,
                halo``, or with no positional arguments at all.

                A mode of operation is given as a `str`, and a halo as
                a non-negative `int` (or any object that can be
                converted to one):

                ==============  ======================================
                *mode*          Description
                ==============  ======================================
                Not provided    If no positional arguments are
                                provided then assume the
                                ``'compress'`` mode of operation with
                                no halo added to the subspaced axes.

                ``mode``        Define the mode of operation with no
                                halo added to the subspaced axes.

                ``mode, halo``  Define a mode of operation, as well as
                                a halo to be added to the subspaced
                                axes.

                ``halo``        Assume the ``'compress'`` mode of
                                operation and define a halo to be
                                added to the subspaced axes.
                ==============  ======================================""",
    # return_esmpy_regrid_operator
    "{{return_esmpy_regrid_operator: `bool`, optional}}": """return_esmpy_regrid_operator: `bool`, optional
                If True then do not perform the regridding, rather
                return the `esmpy.Regrid` instance that defines the
                regridding operation.""",
    # ----------------------------------------------------------------
    # Method description substitutions (4 levels of indentation)
    # ----------------------------------------------------------------
    # Returns construct
    "{{Returns construct}}": """The selected construct, or its identifier if *key* is
                True, or a tuple of both if *item* is True.""",
    # regrid RegridOperator
    "{{regrid RegridOperator}}": """* `RegridOperator`: The grid is defined by a regrid
                  operator that has been returned by a previous call
                  with the *return_operator* parameter set to True.

                  Unlike the other options, for which the regrid
                  weights need to be calculated, the regrid operator
                  already contains the weights. Therefore, for cases
                  where multiple fields with the same source grids
                  need to be regridded to the same destination grid,
                  using a regrid operator can give performance
                  improvements by avoiding having to calculate the
                  weights for each source field. Note that for the
                  other types of *dst* parameter, the calculation of
                  the regrid weights is not a lazy operation.

                   .. note:: When *dst* is a `RegridOperator`, the
                            source grid of the regrid operator is
                            immediately checked for compatibility with
                            the grid of the source field. By default
                            only the computationally cheap tests are
                            performed (checking that the coordinate
                            system, cyclicity, grid shape, regridding
                            dimensionality, mesh location, and feature
                            type are the same), with the grid
                            coordinates not being checked. The
                            coordinates check will be carried out,
                            however, if the *check_coordinates*
                            parameter is True.""",
    # Returns cfa_file_substitutions
    "{{Returns cfa_file_substitutions}}": """The CFA-netCDF file name substitutions in a dictionary
                whose key/value pairs are the file name parts to be
                substituted and their corresponding substitution
                text.""",
    # Returns cfa_clear_file_substitutions
    "{{Returns cfa_clear_file_substitutions}}": """The removed CFA-netCDF file name substitutions in a
                dictionary whose key/value pairs are the file name
                parts to be substituted and their corresponding
                substitution text.""",
    # Returns cfa_clear_file_substitutions
    "{{Returns cfa_del_file_substitution}}": """
                The removed CFA-netCDF file name substitution. If the
                substitution was not defined then an empty dictionary
                is returned.""",
    # subspace valid modes Field
    "{{subspace valid modes Field}}": """Valid modes are:

                * ``'compress'`` This is the default.
                     Unselected locations are removed to create the
                     subspace. If the result is not hyperrectangular
                     then the minimum amount of unselected locations
                     required to make it so will also be specially
                     selected. Missing data is inserted at the
                     specially selected locations, unless a halo has
                     been defined (of any size, including 0).

                * ``'envelope'``
                     The subspace is the smallest hyperrectangular
                     subspace that contains all of the selected
                     locations. Missing data is inserted at unselected
                     locations within the envelope, unless a halo has
                     been defined (of any size, including 0).

                * ``'full'``
                     The subspace has the same domain as the original
                     construct. Missing data is inserted at unselected
                     locations, unless a halo has been defined (of any
                     size, including 0).

                .. note:: Setting a halo size of `0` differs from not
                          not defining a halo at all. The shape of the
                          returned field will always be the same, but
                          in the former case missing data will not be
                          inserted at unselected locations (if any)
                          within the output domain.""",
    # subspace valid modes Domain
    "{{subspace valid modes Domain}}": """Valid modes are:

                * ``'compress'`` This is the default.
                     Unselected locations are removed to create the
                     subspace. If the result is not hyperrectangular
                     then the minimum amount of unselected locations
                     required to make it so will also be specially
                     selected.

                * ``'envelope'``
                     The subspace is the smallest hyperrectangular
                     subspace that contains all of the selected
                     locations.""",
}
