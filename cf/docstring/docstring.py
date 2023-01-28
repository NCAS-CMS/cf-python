"""Define docstring substitutions.

Text to be replaced is specified as a key in the returned dictionary,
with the replacement text defined by the corresponding value.

Special docstring subtitutions, as defined by a class's
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
    # General susbstitutions (not indent-dependent)
    # ----------------------------------------------------------------
    "{{repr}}": "CF ",
    # ----------------------------------------------------------------
    # Class description susbstitutions (1 level of indentation)
    # ----------------------------------------------------------------
    "{{formula terms links}}": """See CF section 4.3.3 "Parametric Vertical Coordinate" and CF
    Appendix D "Parametric Vertical Coordinates" for details.""",
    # ----------------------------------------------------------------
    # Class description susbstitutions (1 level of indentation)
    # ----------------------------------------------------------------
    #
    # ----------------------------------------------------------------
    # Method description susbstitutions (2 levels of indentation)
    # ----------------------------------------------------------------
    # List comparison
    "{{List comparison}}": """Each construct in the list is compared with its `!equals`
        method, rather than the ``==`` operator.""",
    # ----------------------------------------------------------------
    # Method description susbstitutions (3 levels of indentataion)
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
    "{{method: `str`, optional}}": """method: `str`, optional
                Specify the regridding method. This parameter must be
                set unless the new grid is specified by a regridding
                operator, which stores its own method. See the *dst*
                parameter.

                The *method* parameter may be one of the following:

                ======================  ==============================
                Method                  Description
                ======================  ==============================
                ``'linear'``            Bilinear interpolation.

                ``'bilinear'``          Deprecated alias for
                                        ``'linear'``.

                ``'conservative_1st'``  First order conservative
                                        interpolation.

                                        Preserve the area integral of
                                        the data across the
                                        interpolation from source to
                                        destination. It uses the
                                        proportion of the area of the
                                        overlapping source and
                                        destination cells to determine
                                        appropriate weights.

                                        In particular, the weight of a
                                        source cell is the ratio of
                                        the area of intersection of
                                        the source and destination
                                        cells to the area of the whole
                                        destination cell.

                                        It does not account for the
                                        field gradient across the
                                        source cell, unlike the
                                        second-order conservative
                                        method (see below).

                ``'conservative_2nd'``  Second-order conservative
                                        interpolation.

                                        As with first order (see
                                        above), preserves the area
                                        integral of the field between
                                        source and destination using a
                                        weighted sum, with weights
                                        based on the proportionate
                                        area of intersection.

                                        Unlike first-order, the
                                        second-order method
                                        incorporates further terms to
                                        take into consideration the
                                        gradient of the field across
                                        the source cell, thereby
                                        typically producing a smoother
                                        result of higher accuracy.

                ``'conservative'``      Alias for
                                        ``'conservative_1st'``

                ``'patch'``             Higher-order patch recovery
                                        interpolation.

                                        A second degree polynomial
                                        regridding method, which uses
                                        a least squares algorithm to
                                        calculate the polynomial.

                                        This method gives better
                                        derivatives in the resulting
                                        destination data than the
                                        linear method.

                ``'nearest_stod'``      Nearest neighbour
                                        interpolation for which each
                                        destination point is mapped to
                                        the closest source point.

                                        Useful for extrapolation of
                                        categorical data.

                ``'nearest_dtos'``      Nearest neighbour
                                        interpolation for which each
                                        source point is mapped to the
                                        destination point.

                                        Useful for extrapolation of
                                        categorical data.

                                        A given destination point may
                                        receive input from multiple
                                        source points, but no source
                                        point will map to more than
                                        one destination point.

                `None`                  This is the default and can
                                        only be used the new grid is
                                        specified by a regridding
                                        operator, which stores its own
                                        method.
                ======================  ==============================

                .. note:: When *dst* is a regrid operator then the
                          *method* may still be set, but must have the
                          value `None` or else agree with the
                          regridding operator's method.""",
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
                  the computed non-parametric coodinates then this an
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
    # aggregated_units
    "{{aggregated_units: `str` or `None`, optional}}": """aggregated_units: `str` or `None`, optional
                The units of the aggregated array. Set to `None` to
                indicate that there are no units.""",
    # aggregated_calendar
    "{{aggregated_calendar: `str` or `None`, optional}}": """aggregated_calendar: `str` or `None`, optional
                The calendar of the aggregated array. Set to `None` to
                indicate the CF default calendar, if applicable.""",
    # cull
    "{{cull_graph: `bool`, optional}}": """cull_graph: `bool`, optional
                By default *cull_graph* is True, meaning that
                unnecessary tasks are removed (culled) from each
                array's dask graph before concatenation. This process
                has a small overhead but can improve performance
                overall. If set to False then dask graphs are not
                culled. See `dask.optimization.cull` for details.

                .. versionadded:: TODODASKVER""",
    # ----------------------------------------------------------------
    # Method description susbstitutions (4 levels of indentataion)
    # ----------------------------------------------------------------
    # Returns construct
    "{{Returns construct}}": """The selected construct, or its identifier if *key* is
                True, or a tuple of both if *item* is True.""",
}
