import cfdm

from ...mixin_container import Container
from .mixin import ArrayMixin, CompressedArrayMixin


class SubsampledArray(
    CompressedArrayMixin, ArrayMixin, Container, cfdm.SubsampledArray
):
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

    .. versionadded:: 3.14.0

    """

    def to_dask_array(self, chunks="auto"):
        """Convert the data to a `dask` array.

        .. versionadded:: 3.14.0

        :Parameters:

            chunks: `int`, `tuple`, `dict` or `str`, optional
                Specify the chunking of the returned dask array.

                Any value accepted by the *chunks* parameter of the
                `dask.array.from_array` function is allowed.

                The chunk sizes implied by *chunks* for a dimension that
                has been fragmented are ignored and replaced with values
                that are implied by that dimensions fragment sizes.

        :Returns:

            `dask.array.Array`
                The `dask` array representation.

        """
        from functools import partial

        import dask.array as da
        from dask import config
        from dask.array.core import getter, normalize_chunks
        from dask.base import tokenize

        name = (f"{self.__class__.__name__}-{tokenize(self)}",)

        dtype = self.dtype

        context = partial(config.set, scheduler="synchronous")

        compressed_dimensions = self.compressed_dimensions()
        conformed_data = self.conformed_data()
        compressed_data = conformed_data["data"]
        parameters = conformed_data["parameters"]
        dependent_tie_points = conformed_data["dependent_tie_points"]

        # If possible, convert the compressed data, parameters and
        # dependent tie points to dask arrays that don't support
        # concurrent reads. This prevents "compute called by compute"
        # failures problems at compute time.
        #
        # TODO: This won't be necessary if this is refactored so that
        #       arrays are part of the same dask graph as the
        #       compressed subarrays.
        compressed_data = self._lock_file_read(compressed_data)
        parameters = {
            k: self._lock_file_read(v) for k, v in parameters.items()
        }
        dependent_tie_points = {
            k: self._lock_file_read(v) for k, v in dependent_tie_points.items()
        }

        # Get the (cfdm) subarray class
        Subarray = self.get_Subarray()
        subarray_name = Subarray().__class__.__name__

        # Set the chunk sizes for the dask array
        #
        # Note: The chunks created here are incorrect for the
        #       compressed dimensions, since these chunk sizes are a
        #       function of the tie point indices which haven't yet
        #       been accessed. Therefore, the chunks for the
        #       compressed dimensons need to be redefined later.
        chunks = normalize_chunks(
            self.subarray_shapes(chunks),
            shape=self.shape,
            dtype=dtype,
        )

        # Re-initialise the chunks
        u_dims = list(compressed_dimensions)
        chunks = [[] if i in u_dims else c for i, c in enumerate(chunks)]

        # For each dimension, initialise the index of the chunk
        # previously created (prior to the chunk currently being
        # created). The value -1 is an arbitrary negative value that is
        # always less than any chunk index, which is always a natural
        # number.
        previous_chunk_location = [-1] * len(chunks)

        dsk = {}
        for (
            u_indices,
            u_shape,
            c_indices,
            subarea_indices,
            first,
            chunk_location,
        ) in zip(*self.subarrays(shapes=chunks)):
            subarray = Subarray(
                data=compressed_data,
                indices=c_indices,
                shape=u_shape,
                compressed_dimensions=compressed_dimensions,
                first=first,
                subarea_indices=subarea_indices,
                parameters=parameters,
                dependent_tie_points=dependent_tie_points,
                context_manager=context,
            )

            key = f"{subarray_name}-{tokenize(subarray)}"
            dsk[key] = subarray
            dsk[name + chunk_location] = (
                getter,
                key,
                Ellipsis,
                False,
                False,
            )

            # Add correct chunk sizes for compressed dimensions
            for d in u_dims[:]:
                previous = previous_chunk_location[d]
                new = chunk_location[d]
                if new > previous:
                    chunks[d].append(u_shape[d])
                    previous_chunk_location[d] = new
                elif new < previous:
                    # No more chunk sizes required for this compressed
                    # dimension
                    u_dims.remove(d)

        chunks = [tuple(c) for c in chunks]

        # Return the dask array
        return da.Array(dsk, name[0], chunks=chunks, dtype=dtype)
