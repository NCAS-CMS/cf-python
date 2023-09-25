from copy import deepcopy
from functools import partial
from itertools import accumulate, product

import numpy as np

from ..fragment import FullFragmentArray, NetCDFFragmentArray, UMFragmentArray
from ..utils import chunk_locations, chunk_positions
from .netcdfarray import NetCDFArray

# Store fragment array classes.
_FragmentArray = {
    "nc": NetCDFFragmentArray,
    "um": UMFragmentArray,
    "full": FullFragmentArray,
}


class CFANetCDFArray(NetCDFArray):
    """A CFA aggregated array stored in a netCDF file.

    .. versionadded:: 3.14.0

    """

    def __init__(
        self,
        filename=None,
        address=None,
        dtype=None,
        mask=True,
        units=False,
        calendar=False,
        instructions=None,
        substitutions=None,
        term=None,
        source=None,
        copy=True,
        x=None,
    ):
        """**Initialisation**

        :Parameters:

            filename: (sequence of) `str`, optional
                The name of the CFA-netCDF file containing the
                array. If a sequence then it must contain one element.

            address: (sequence of) `str`, optional
                The name of the CFA-netCDF aggregation variable for the
                array. If a sequence then it must contain one element.

            dtype: `numpy.dtype`
                The data type of the aggregated data array. May be
                `None` if the numpy data-type is not known (which can
                be the case for netCDF string types, for example).

            mask: `bool`
                If True (the default) then mask by convention when
                reading data from disk.

                A netCDF array is masked depending on the values of any of
                the netCDF variable attributes ``valid_min``,
                ``valid_max``, ``valid_range``, ``_FillValue`` and
                ``missing_value``.

            units: `str` or `None`, optional
                The units of the aggregated data. Set to `None` to
                indicate that there are no units.

            calendar: `str` or `None`, optional
                The calendar of the aggregated data. Set to `None` to
                indicate the CF default calendar, if applicable.

            instructions: `str`, optional
                The ``aggregated_data`` attribute value as found on
                the CFA netCDF variable. If set then this will be used
                to improve the performance of `__dask_tokenize__`.

            substitutions: `dict`, optional
                A dictionary whose key/value pairs define text
                substitutions to be applied to the fragment file
                names. Each key must be specified with the ``${...}``
                syntax, for instance ``{'${base}': 'sub'}``.

                .. versionadded:: 3.15.0

            term: `str`, optional
                The name of a non-standard aggregation instruction
                term from which the array is to be created, instead of
                creating the aggregated data in the standard
                terms. If set then *address* must be the name of the
                term's CFA-netCDF aggregation instruction variable,
                which must be defined on the fragment dimensions and
                no others. Each value of the aggregation instruction
                variable will be broadcast across the shape of the
                corresponding fragment.

                *Parameter example:*
                  ``address='cfa_tracking_id', term='tracking_id'``

                .. versionadded:: 3.15.0

            {{init source: optional}}

            {{init copy: `bool`, optional}}

        """
        if source is not None:
            super().__init__(source=source, copy=copy)

            try:
                fragment_shape = source.get_fragment_shape()
            except AttributeError:
                fragment_shape = None

            try:
                instructions = source._get_component("instructions")
            except AttributeError:
                instructions = None

            try:
                aggregated_data = source.get_aggregated_data(copy=False)
            except AttributeError:
                aggregated_data = {}

            try:
                substitutions = source.get_substitutions()
            except AttributeError:
                substitutions = None

            try:
                term = source.get_term()
            except AttributeError:
                term = None

        elif filename is not None:
            aggregated_data = {}

            location = x["location"]
            ndim = location.shape[0]

            chunks = [i.compressed().tolist() for i in location]
            shape = [sum(c) for c in chunks]
            positions = chunk_positions(chunks)
            locations = chunk_locations(chunks)

            if term is not None:
                # --------------------------------------------------------
                # This fragment contains a constant value, not file
                # locations.
                # --------------------------------------------------------
                term = x[term]
                fragment_shape = term.shape
                aggregated_data = {
                    frag_loc: {
                        "location": loc,
                        "fill_value": term[frag_loc].item(),
                        "format": "full",
                    }
                    for frag_loc, loc in zip(positions, locations)
                }
            else:
                a = x["address"]
                f = x["file"]
                fmt = x["format"]

                extra_dimension = f.ndim > ndim
                if extra_dimension:
                    # There is an extra non-fragment dimension
                    fragment_shape = f.shape[:-1]
                else:
                    fragment_shape = f.shape

                if not a.ndim:
                    a = np.full(f.shape, a, dtype=a.dtype)

                if not fmt.ndim:
                    fmt = np.full(fragment_shape, fmt, dtype=fmt.dtype)

                if extra_dimension:
                    aggregated_data = {
                        frag_loc: {
                            "location": loc,
                            "filename": f[frag_loc].tolist(),
                            "address": a[frag_loc].tolist(),
                            "format": fmt[frag_loc].item(),
                        }
                        for frag_loc, loc in zip(positions, locations)
                    }
                else:
                    aggregated_data = {
                        frag_loc: {
                            "location": loc,
                            "filename": (f[frag_loc].item(),),
                            "address": (a[frag_loc].item(),),
                            "format": fmt[frag_loc].item(),
                        }
                        for frag_loc, loc in zip(positions, locations)
                    }

                # Apply string substitutions to the fragment filenames
                if substitutions:
                    for value in aggregated_data.values():
                        filenames2 = []
                        for filename in value["filename"]:
                            for base, sub in substitutions.items():
                                filename = filename.replace(base, sub)

                            filenames2.append(filename)

                        value["filename"] = filenames2

            super().__init__(
                filename=filename,
                address=address,
                shape=shape,
                dtype=dtype,
                mask=mask,
                units=units,
                calendar=calendar,
                copy=copy,
            )
        else:
            super().__init__(
                filename=filename,
                address=address,
                dtype=dtype,
                mask=mask,
                units=units,
                calendar=calendar,
                copy=copy,
            )

            fragment_shape = None
            aggregated_data = None
            instructions = None
            term = None

        self._set_component("fragment_shape", fragment_shape, copy=False)
        self._set_component("aggregated_data", aggregated_data, copy=False)
        self._set_component("instructions", instructions, copy=False)
        self._set_component("term", term, copy=False)

        if substitutions is not None:
            self._set_component(
                "substitutions", substitutions.copy(), copy=False
            )

    def __dask_tokenize__(self):
        """Used by `dask.base.tokenize`.

        .. versionadded:: 3.14.0

        """
        out = super().__dask_tokenize__()
        aggregated_data = self._get_component("instructions", None)
        if aggregated_data is None:
            aggregated_data = self.get_aggregated_data(copy=False)

        return out + (aggregated_data,)

    def __getitem__(self, indices):
        """x.__getitem__(indices) <==> x[indices]"""
        return NotImplemented  # pragma: no cover

    def get_aggregated_data(self, copy=True):
        """Get the aggregation data dictionary.

        The aggregation data dictionary contains the definitions of
        the fragments and the instructions on how to aggregate them.
        The keys are indices of the CFA fragment dimensions,
        e.g. ``(1, 0, 0 ,0)``.

        .. versionadded:: 3.14.0

        :Parameters:

            copy: `bool`, optional
                Whether or not to return a copy of the aggregation
                dictionary. By default a deep copy is returned.

                .. warning:: If False then changing the returned
                             dictionary in-place will change the
                             aggregation dictionary stored in the
                             {{class}} instance, **as well as in any
                             copies of it**.

        :Returns:

            `dict`
                The aggregation data dictionary.

        **Examples**

        >>> a.shape
        (12, 1, 73, 144)
        >>> a.get_fragment_shape()
        (2, 1, 1, 1)
        >>> a.get_aggregated_data()
        {(0, 0, 0, 0): {'file': 'January-June.nc',
          'address': 'temp',
          'format': 'nc',
          'location': [(0, 6), (0, 1), (0, 73), (0, 144)]},
         (1, 0, 0, 0): {'file': 'July-December.nc',
          'address': 'temp',
          'format': 'nc',
          'location': [(6, 12), (0, 1), (0, 73), (0, 144)]}}

        """
        aggregated_data = self._get_component("aggregated_data")
        if copy:
            aggregated_data = deepcopy(aggregated_data)

        return aggregated_data

    def get_fragmented_dimensions(self):
        """Get the positions of dimensions that have two or more fragments.

        .. versionadded:: 3.14.0

        :Returns:

            `list`
                The dimension positions.

        **Examples**

        >>> a.get_fragment_shape()
        (20, 1, 40, 1)
        >>> a.get_fragmented_dimensions()
        [0, 2]

        >>> a.get_fragment_shape()
        (1, 1, 1)
        >>> a.get_fragmented_dimensions()
        []

        """
        return [
            i for i, size in enumerate(self.get_fragment_shape()) if size > 1
        ]

    def get_fragment_shape(self):
        """Get the sizes of the fragment dimensions.

        The fragment dimension sizes are given in the same order as
        the aggregated dimension sizes given by `shape`.

        .. versionadded:: 3.14.0

        :Returns:

            `tuple`
                The shape of the fragment dimensions.

        """
        return self._get_component("fragment_shape")

    def get_term(self, default=ValueError()):
        """The CFA aggregation instruction term for the data, if set.

        .. versionadded:: 3.15.0

        :Parameters:

            default: optional
                Return the value of the *default* parameter if the
                term has not been set. If set to an `Exception`
                instance then it will be raised instead.

        :Returns:

            `str`
                The CFA aggregation instruction term name.

        """
        return self._get_component("term", default=default)

    def subarray_shapes(self, shapes):
        """Create the subarray shapes.

        .. versionadded:: 3.14.0

        .. seealso:: `subarrays`

        :Parameters:

           shapes: `int`, sequence, `dict` or `str`, optional
                Define the subarray shapes.

                Any value accepted by the *chunks* parameter of the
                `dask.array.from_array` function is allowed.

                The subarray sizes implied by *chunks* for a dimension
                that has been fragmented are ignored, so their
                specification is arbitrary.

        :Returns:

            `tuple`
                The subarray sizes along each dimension.

        **Examples**

        >>> a.shape
        (12, 1, 73, 144)
        >>> a.get_fragment_shape()
        (2, 1, 1, 1)
        >>> a.fragmented_dimensions()
        [0]
        >>> a.subarray_shapes(-1)
        ((6, 6), (1,), (73,), (144,))
        >>> a.subarray_shapes(None)
        ((6, 6), (1,), (73,), (144,))
        >>> a.subarray_shapes("auto")
        ((6, 6), (1,), (73,), (144,))
        >>> a.subarray_shapes((None, 1, 40, 50))
        ((6, 6), (1,), (40, 33), (50, 50, 44))
        >>>  a.subarray_shapes((None, None, "auto", 50))
        ((6, 6), (1,), (73,), (50, 50, 44))
        >>>  a.subarray_shapes({2: 40})
        ((6, 6), (1,), (40, 33), (144,))

        """
        from numbers import Number

        from dask.array.core import normalize_chunks

        # Indices of fragmented dimensions
        f_dims = self.get_fragmented_dimensions()

        shape = self.shape
        aggregated_data = self.get_aggregated_data(copy=False)

        # Create the base chunks.
        chunks = []
        ndim = self.ndim
        for dim, (n_fragments, size) in enumerate(
            zip(self.get_fragment_shape(), self.shape)
        ):
            if dim in f_dims:
                # This aggregated dimension is spanned by more than
                # one fragment.
                c = []
                index = [0] * ndim
                for j in range(n_fragments):
                    index[dim] = j
                    loc = aggregated_data[tuple(index)]["location"][dim]
                    chunk_size = loc[1] - loc[0]
                    c.append(chunk_size)

                chunks.append(tuple(c))
            else:
                # This aggregated dimension is spanned by exactly one
                # fragment. Store None, for now, in the expectation
                # that it will get overwrittten.
                chunks.append(None)

        if isinstance(shapes, (str, Number)) or shapes is None:
            chunks = [
                c if i in f_dims else shapes for i, c in enumerate(chunks)
            ]
        elif isinstance(shapes, dict):
            chunks = [
                chunks[i] if i in f_dims else shapes.get(i, "auto")
                for i, c in enumerate(chunks)
            ]
        else:
            # chunks is a sequence
            if len(shapes) != ndim:
                raise ValueError(
                    f"Wrong number of 'shapes' elements in {shapes}: "
                    f"Got {len(shapes)}, expected {self.ndim}"
                )

            chunks = [
                c if i in f_dims else shapes[i] for i, c in enumerate(chunks)
            ]

        return normalize_chunks(chunks, shape=shape, dtype=self.dtype)

    def subarrays(self, subarray_shapes):
        """Return descriptors for every subarray.

        .. versionadded:: 3.14.0

        .. seealso:: `subarray_shapes`

        :Parameters:

            subarray_shapes: `tuple`
                The subarray sizes along each dimension, as returned
                by a prior call to `subarray_shapes`.

        :Returns:

            6-`tuple` of iterators
               Each iterator iterates over a particular descriptor
               from each subarray.

               1. The indices of the aggregated array that correspond
                  to each subarray.

               2. The shape of each subarray.

               3. The indices of the fragment that corresponds to each
                  subarray (some subarrays may be represented by a
                  part of a fragment).

               4. The location of each subarray.

               5. The location on the fragment dimensions of the
                  fragment that corresponds to each subarray.

               6. The shape of each fragment that overlaps each chunk.

        **Examples**

        An aggregated array with shape (12, 73, 144) has two
        fragments, both with with shape (6, 73, 144).

        >>> a.shape
        (12, 73, 144)
        >>> a.get_fragment_shape()
        (2, 1, 1)
        >>> a.fragmented_dimensions()
        [0]
        >>> subarray_shapes = a.subarray_shapes({1: 40})
        >>> print(subarray_shapes)
        ((6, 6), (40, 33), (144,))
        >>> (
        ...  u_indices,
        ...  u_shapes,
        ...  f_indices,
        ...  s_locations,
        ...  f_locations,
        ...  f_shapes,
        ... ) = a.subarrays(subarray_shapes)
        >>> for i in u_indices:
        ...    print(i)
        ...
        (slice(0, 6, None), slice(0, 40, None), slice(0, 144, None))
        (slice(0, 6, None), slice(40, 73, None), slice(0, 144, None))
        (slice(6, 12, None), slice(0, 40, None), slice(0, 144, None))
        (slice(6, 12, None), slice(40, 73, None), slice(0, 144, None))

        >>> for i in u_shapes
        ...    print(i)
        ...
        (6, 40, 144)
        (6, 33, 144)
        (6, 40, 144)
        (6, 33, 144)
        >>> for i in f_indices:
        ...    print(i)
        ...
        (slice(None, None, None), slice(0, 40, None), slice(0, 144, None))
        (slice(None, None, None), slice(40, 73, None), slice(0, 144, None))
        (slice(None, None, None), slice(0, 40, None), slice(0, 144, None))
        (slice(None, None, None), slice(40, 73, None), slice(0, 144, None))
        >>> for i in s_locations:
        ...    print(i)
        ...
        (0, 0, 0)
        (0, 1, 0)
        (1, 0, 0)
        (1, 1, 0)
        >>> for i in f_locations:
        ...    print(i)
        ...
        (0, 0, 0)
        (0, 0, 0)
        (1, 0, 0)
        (1, 0, 0)
        >>> for i in f_shapes:
        ...    print(i)
        ...
        (6, 73, 144)
        (6, 73, 144)
        (6, 73, 144)
        (6, 73, 144)

        """
        f_dims = self.get_fragmented_dimensions()

        # The indices of the uncompressed array that correspond to
        # each subarray, the shape of each uncompressed subarray, and
        # the location of each subarray
        s_locations = []
        u_shapes = []
        u_indices = []
        f_locations = []
        for dim, c in enumerate(subarray_shapes):
            nc = len(c)
            s_locations.append(tuple(range(nc)))
            u_shapes.append(c)

            if dim in f_dims:
                f_locations.append(tuple(range(nc)))
            else:
                # No fragmentation along this dimension
                f_locations.append((0,) * nc)

            c = tuple(accumulate((0,) + c))
            u_indices.append([slice(i, j) for i, j in zip(c[:-1], c[1:])])

        # For each subarray, the part of the fragment that corresponds
        # to it.
        f_indices = [
            (slice(None),) * len(u) if dim in f_dims else u
            for dim, u in enumerate(u_indices)
        ]

        # For each subarray, the shape of the fragment that
        # corresponds to it.
        f_shapes = [
            u_shape if dim in f_dims else (size,) * len(u_shape)
            for dim, (u_shape, size) in enumerate(zip(u_shapes, self.shape))
        ]

        return (
            product(*u_indices),
            product(*u_shapes),
            product(*f_indices),
            product(*s_locations),
            product(*f_locations),
            product(*f_shapes),
        )

    def to_dask_array(self, chunks="auto"):
        """Create a dask array with `FragmentArray` chunks.

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

        """
        import dask.array as da
        from dask.array.core import getter
        from dask.base import tokenize

        name = (f"{self.__class__.__name__}-{tokenize(self)}",)

        dtype = self.dtype
        units = self.get_units()
        calendar = self.get_calendar(None)
        aggregated_data = self.get_aggregated_data(copy=False)

        # Set the chunk sizes for the dask array
        chunks = self.subarray_shapes(chunks)

        if self.get_mask():
            fragment_arrays = _FragmentArray
        else:
            fragment_arrays = _FragmentArray.copy()
            fragment_arrays["nc"] = partial(_FragmentArray["nc"], mask=False)

        dsk = {}
        for (
            u_indices,
            u_shape,
            f_indices,
            chunk_location,
            fragment_location,
            fragment_shape,
        ) in zip(*self.subarrays(chunks)):
            kwargs = aggregated_data[fragment_location].copy()
            kwargs.pop("location", None)

            fragment_format = kwargs.pop("format", None)
            try:
                FragmentArray = fragment_arrays[fragment_format]
            except KeyError:
                raise ValueError(
                    "Can't get FragmentArray class for unknown "
                    f"fragment dataset format: {fragment_format!r}"
                )

            fragment = FragmentArray(
                dtype=dtype,
                shape=fragment_shape,
                aggregated_units=units,
                aggregated_calendar=calendar,
                **kwargs,
            )

            key = f"{fragment.__class__.__name__}-{tokenize(fragment)}"
            dsk[key] = fragment
            dsk[name + chunk_location] = (
                getter,
                key,
                f_indices,
                False,
                getattr(fragment, "_lock", False),
            )

        # Return the dask array
        return da.Array(dsk, name[0], chunks=chunks, dtype=dtype)
