from copy import deepcopy
from itertools import accumulate, product

from ...functions import abspath
from ..fragment import (
    MissingFragmentArray,
    NetCDFFragmentArray,
    UMFragmentArray,
)
from .netcdfarray import NetCDFArray


class CFANetCDFArray(NetCDFArray):
    """A CFA aggregated array stored in a netCDF file.

    .. versionadded:: 3.14.0

    """

    def __new__(cls, *args, **kwargs):
        """Store fragment array classes.

        .. versionadded:: 3.14.0

        """
        instance = super().__new__(cls)
        instance._FragmentArray = {
            "nc": NetCDFFragmentArray,
            "um": UMFragmentArray,
            None: MissingFragmentArray,
        }
        return instance

    def __init__(
        self,
        filename=None,
        ncvar=None,
        varid=None,
        group=None,
        dtype=None,
        mask=True,
        units=False,
        calendar=False,
        instructions=None,
        source=None,
        copy=True,
    ):
        """**Initialisation**

        :Parameters:

            filename: `str`
                The name of the netCDF file containing the array.

            ncvar: `str`, optional
                The name of the netCDF variable containing the
                array. Required unless *varid* is set.

            varid: `int`, optional
                The UNIDATA netCDF interface ID of the variable
                containing the array. Required if *ncvar* is not set,
                ignored if *ncvar* is set.

            group: `None` or sequence of `str`, optional
                Specify the netCDF4 group to which the netCDF variable
                belongs. By default, or if *group* is `None` or an
                empty sequence, it assumed to be in the root
                group. The last element in the sequence is the name of
                the group in which the variable lies, with other
                elements naming any parent groups (excluding the root
                group).

                *Parameter example:*
                  To specify that a variable is in the root group:
                  ``group=()`` or ``group=None``

                *Parameter example:*
                  To specify that a variable is in the group '/forecasts':
                  ``group=['forecasts']``

                *Parameter example:*
                  To specify that a variable is in the group
                  '/forecasts/model2': ``group=['forecasts', 'model2']``

            dtype: `numpy.dtype`
                The data type of the array in the netCDF file. May be
                `None` if the numpy data-type is not known (which can be
                the case for netCDF string types, for example).

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

            {{init source: optional}}

            {{init copy: `bool`, optional}}

            instructions: `str`, optional
                The ``aggregated_data`` attribute value found on the
                CFA netCDF variable. If set then this will be used by
                `__dask_tokenize__` to improve performance.

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
        elif filename is not None:
            from CFAPython import CFAFileFormat
            from CFAPython.CFADataset import CFADataset
            from CFAPython.CFAExceptions import CFAException
            from dask import compute, delayed

            cfa = CFADataset(filename, CFAFileFormat.CFANetCDF, "r")
            try:
                var = cfa.getVar(ncvar)
            except CFAException:
                raise ValueError(
                    f"CFA variable {ncvar} not found in file {filename}"
                )

            shape = tuple([d.len for d in var.getDims()])

            super().__init__(
                filename=filename,
                ncvar=ncvar,
                varid=varid,
                group=group,
                shape=shape,
                dtype=dtype,
                mask=mask,
                units=units,
                calendar=calendar,
                copy=copy,
            )

            fragment_shape = tuple(var.getFragDef())

            # Note: It is an as-yet-untested hypothesis that creating
            #       the 'aggregated_data' dictionary for massive
            #       aggretations (e.g. with O(10e6) fragments) will be
            #       slow, hence the parallelisation of the process
            #       with delayed + compute, and that the
            #       parallelisation overheads won't be noticeable for
            #       small aggregations (e.g. O(10) fragments).
            aggregated_data = {}
            set_fragment = self._set_fragment
            compute(
                *[
                    delayed(set_fragment(var, loc, aggregated_data, filename))
                    for loc in product(*[range(i) for i in fragment_shape])
                ]
            )

            del cfa
        else:
            super().__init__(
                filename=filename,
                ncvar=ncvar,
                varid=varid,
                group=group,
                dtype=dtype,
                mask=mask,
                units=units,
                calendar=calendar,
                copy=copy,
            )

            fragment_shape = None
            aggregated_data = None
            instructions = None

        self._set_component("fragment_shape", fragment_shape, copy=False)
        self._set_component("aggregated_data", aggregated_data, copy=False)
        self._set_component("instructions", instructions, copy=False)

    def __dask_tokenize__(self):
        """Used by `dask.base.tokenize`.

        .. versionadded:: 3.14.0

        """
        aggregated_data = self._get_component("instructions", None)
        if aggregated_data is None:
            aggregated_data = self.get_aggregated_data(copy=False)

        return (
            self.__class__.__name__,
            abspath(self.get_filename()),
            self.get_ncvar(),
            self.get_group(),
            aggregated_data,
        )

    def __getitem__(self, indices):
        """x.__getitem__(indices) <==> x[indices]"""
        return NotImplemented  # pragma: no cover

    def _set_fragment(self, var, frag_loc, aggregated_data, cfa_filename):
        """Create a new key/value pair in the *aggregated_data*
        dictionary.

        The *aggregated_data* dictionary contains the definitions of
        the fragments and the instructions on how to aggregate them,
        and is updated in-place.

        .. versionadded:: 3.14.0

        :Parameters:

            var: `CFAPython.CFAVariable.CFAVariable`
                The CFA aggregation variable.

            frag_loc: `tuple` of `int`
                The new key, that must be index of the CFA fragment
                dimensions, e.g. ``(1, 0, 0, 0)``.

            aggregated_data: `dict`
                The aggregated data dictionary to be updated in-place.

        :Returns:

            `None`

        """
        fragment = var.getFrag(frag_loc=frag_loc)

        filename = fragment.file
        fmt = fragment.format
        address = fragment.address

        if address is not None:
            if filename is None:
                # This fragment is in the CFA-netCDF file
                filename = cfa_filename
                fmt = "nc"
            else:
                # This fragment is in its own file
                filename = abspath(fragment.file)

        aggregated_data[frag_loc] = {
            "file": filename,
            "address": address,
            "format": fmt,
            "location": fragment.location,
        }

    def _subarray_shapes(self, shapes):
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

    def _subarrays(self, subarray_shapes):
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

    def get_FragmentArray(self, fragment_format):
        """Return a Fragment class.

        .. versionadded:: 3.14.0

        :Parameters:

            fragment_format: `str`
                The dataset format of the fragment. Either ``'nc'``,
                ``'um'``, or `None`.

        :Returns:

            `FragmentArray`
                The class for representing a fragment array of the
                given format.

        """
        try:
            return self._FragmentArray[fragment_format]
        except KeyError:
            raise ValueError(
                "Can't get FragmentArray class for unknown "
                f"fragment dataset format: {fragment_format!r}"
            )

    def get_fragmented_dimensions(self):
        """Get the positions dimension that have two or more fragments.

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
        the aggregated dimension sizes given by `shape`

        .. versionadded:: 3.14.0

        :Returns:

            `tuple`
                The shape of the fragment dimensions.

        """
        return self._get_component("fragment_shape")

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
        chunks = self._subarray_shapes(chunks)

        # Create a FragmentArray for each chunk
        get_FragmentArray = self.get_FragmentArray

        dsk = {}
        for (
            u_indices,
            u_shape,
            f_indices,
            chunk_location,
            fragment_location,
            fragment_shape,
        ) in zip(*self._subarrays(chunks)):
            d = aggregated_data[fragment_location]

            FragmentArray = get_FragmentArray(d["format"])

            fragment_array = FragmentArray(
                filename=d["file"],
                address=d["address"],
                dtype=dtype,
                shape=fragment_shape,
                aggregated_units=units,
                aggregated_calendar=calendar,
            )

            key = f"{fragment_array.__class__.__name__}-{tokenize(fragment_array)}"
            dsk[key] = fragment_array

            dsk[name + chunk_location] = (
                getter,
                key,
                f_indices,
                False,
                False,
            )

        # Return the dask array
        return da.Array(dsk, name[0], chunks=chunks, dtype=dtype)
