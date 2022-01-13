import cfdm


class SubsampledArray(cfdm.SubsampledArray):
    """An underlying subsampled array.

    For some structured coordinate data (e.g. coordinates describing
    remote sensing products) space may be saved by storing a subsample
    of the data, called tie points. The uncompressed data can be
    reconstituted by interpolation, from the subsampled values. This
    process will likely result in a loss in accuracy (as opposed to
    precision) in the uncompressed variables, due to rounding and
    approximation errors in the interpolation calculations, but it is
    assumed that these errors will be small enough to not be of
    concern to users of the uncompressed dataset. The creator of the
    compressed dataset can control the accuracy of the reconstituted
    data through the degree of subsampling and the choice of
    interpolation method.

    See CF section 8.3 "Lossy Compression by Coordinate Subsampling"
    and Appendix J "Coordinate Interpolation Methods".

    >>> tie_point_indices={{package}}.TiePointIndex(data=[0, 4, 7, 8, 11])
    >>> w = {{package}}.InterpolationParameter(data=[5, 10, 5])
    >>> coords = {{package}}.SubsampledArray(
    ...     interpolation_name='quadratic',
    ...     compressed_array={{package}}.Data([15, 135, 225, 255, 345]),
    ...     shape=(12,),
    ...     tie_point_indices={0: tie_point_indices},
    ...     parameters={"w": w},
    ...     parameter_dimensions={"w": (0,)},
    ... )
    >>> print(coords[...])
    [ 15.          48.75        80.         108.75       135.
     173.88888889 203.88888889 225.         255.         289.44444444
     319.44444444 345.        ]

    **Cell boundaries**

    When the tie points array represents bounds tie points then the
    *shape* parameter describes the uncompressed bounds shape. See CF
    section 8.3.9 "Interpolation of Cell Boundaries".

    >>> bounds = {{package}}.SubsampledArray(
    ...     interpolation_name='quadratic',
    ...     compressed_array={{package}}.Data([0, 150, 240, 240, 360]),
    ...     shape=(12, 2),
    ...     tie_point_indices={0: tie_point_indices},
    ...     parameters={"w": w},
    ...     parameter_dimensions={"w": (0,)},
    ... )
    >>> print(bounds[...])
    [[0.0 33.2]
     [33.2 64.8]
     [64.8 94.80000000000001]
     [94.80000000000001 123.2]
     [123.2 150.0]
     [150.0 188.88888888888889]
     [188.88888888888889 218.88888888888889]
     [218.88888888888889 240.0]
     [240.0 273.75]
     [273.75 305.0]
     [305.0 333.75]
     [333.75 360.0]]

    .. versionadded:: TODODASK

    """

    def __array_function__(self, func, types, args, kwargs):
        return NotImplemented
