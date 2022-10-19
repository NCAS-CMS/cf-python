from copy import deepcopy
from itertools import accumulate, product
from numbers import Number

from CFAPython import CFAFileFormat
from CFAPython.CFADataset import CFADataset
from dask import compute, delayed

from .fragment import NetCDFFragmentArray
from .netcdfarray import NetCDFArray


class CFANetCDFArray(NetCDFArray):
    """A CFA aggregated array stored in a netCDF file.

    .. versionadded:: TODODASKVER

    """

    def __new__(cls, *args, **kwargs):
        """Store fragment array classes.

        .. versionadded:: TODODASKVER

        """
        instance = super().__new__(cls)
        instance._FragmentArray = {"nc": NetCDFFragmentArray, "um": None}
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
                The units of the netCDF variable. Set to `None` to
                indicate that there are no units.

            calendar: `str`, optional
                The calendar of the netCDF variable.  Set to `None` to
                indicate that there is no calendar, or the CF default
                calendar if applicable.

            source: optional
                Initialise the array from the given object.

                {{init source}}

            {{deep copy}}

        """
        if source is not None:
            super().__init__(source=source, copy=copy)

            try:
                fragment_shape = source.get_fragment_shape()
            except AttributeError:
                fragment_shape = None

            try:
                aggregated_data = source.get_aggregated_data(copy=False)
            except AttributeError:
                aggregated_data = {}
        else:
            cfa = CFADataset(filename, CFAFileFormat.CFANetCDF, "r")

            found_var = False
            for var in cfa.getVars():
                # groups??
                if var.name == ncvar:
                    found_var = True
                    break

            if not found_var:
                raise ValueError(
                    f"Can't find CFA-netCDF variable {ncvar} in file "
                    f"{filename}"
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

            # Note: It is an as-yet-untested assertion that creating
            #       aggregated_data for massive aggretations
            #       (e.g. with O(10e6) fragments) will be slow, hence
            #       the parallelisation of the process with delayed +
            #       compute. What about small aggregations (e.g. with
            #       O(10) fragments? Will overheads associated with
            #       delayed + compute be too much? Doing "if
            #       <condition> use dask else don't" could work, but
            #       how would you choose <condition>? Would it be
            #       configurable?
            aggregated_data = {}
            compute(
                *[
                    delayed(self._set_fragment(var, loc, aggregated_data))
                    for loc in product(*[range(i) for i in fragment_shape])
                ]
            )

            del cfa

        self._set_component("fragment_shape", fragment_shape, copy=False)
        self._set_component("aggregated_data", aggregated_data, copy=False)

    def _set_fragment(self, var, frag_loc, aggregated_data):
        """TODODASKDOCS.

        .. versionadded:: TODODASKVER

        :Parameters:

            var: `CFAPython.CFAVariable.CFAVariable`

            frag_loc: `tuple` of `int`

            aggregated_data: `dict`

        :Returns:

            `None`

        """
        fragment = var.getFrag(frag_loc=frag_loc)
        aggregated_data[frag_loc] = {
            "file": fragment.file,
            "address": fragment.address,
            "format": fragment.format,
            "location": fragment.location,
        }

    def get_FragmentArray(self, fragment_format):
        """Return the Fragment class.

        .. versionadded:: TODODASKVER

        :Returns:

            `FragmentArray`
                The class for representing fragment arrays.

        """
        FragmentArray = self._FragmentArray.get(fragment_format)
        if FragmentArray is None:
            raise ValueError(
                "Can't get FragmentArray class for unknown "
                f"fragment dataset format: {fragment_format!r}"
            )

        return FragmentArray

    def get_aggregated_data(self, copy=True):
        aggregated_data = self._get_component("aggregated_data")
        if copy:
            aggregated_data = deepcopy(aggregated_data)

        return aggregated_data

    def get_fragmented_dimensions(self):
        return [i for i, size in enumerate(self.fragment_shape) if size > 1]

    def is_cfa(self):
        return True

    def subarray_shapes(self, shapes):
        """Create the subarray shapes.

        .. versionadded:: TODODASKVER

        .. seealso:: `subarrays`

        :Parameters:

            {{subarray_shapes chunks: `int`, sequence, `dict`, or `str`, optional}}

        :Returns:

            `list`
                The subarray sizes along each uncompressed dimension.

        >>> a.shape
        (4, 20, 30)
        >>> a.compressed_dimensions()
        {1: (1,), 2: (2,)}
        >>> a.subarray_shapes(-1)
        [(4,), None, None]
        >>> a.subarray_shapes("auto")
        ["auto", None, None]
        >>> a.subarray_shapes(2)
        [2, None, None]
        >>> a.subarray_shapes("60B")
        ["60B", None, None]
        >>> a.subarray_shapes((2, None, None))
        [2, None, None]
        >>> a.subarray_shapes(((1, 3), None, None))
        [(1, 3), None, None]
        >>> a.subarray_shapes(("auto", None, None))
        ["auto", None, None]
        >>> a.subarray_shapes(("60B", None, None))
        ["60B", None, None]

        """
        # Indices of fragmented dimensions
        f_dims = self.get_fragmented_dimensions()

        aggregated_data = self.get_aggregated_data(copy=False)

        # Create chunks assuming that all non-fragmented dimensions
        # have one chunk
        chunks = []
        ndim = self.ndim
        for dim, (n_fragments, size) in enumerate(
            zip(self.fragment_shape, self.shape)
        ):
            if dim in f_dims:
                # This aggregated dimension is spanned by more than
                # one fragment.
                c = []
                index = [0] * ndim
                for j in range(n_fragments):
                    index[dim] = j
                    chunk_size = tuple(
                        loc[1] - loc[0]
                        for loc in aggregated_data[tuple(index)]["location"]
                    )
                    c.append(chunk_size)

                chunks.append(tuple(c))
            else:
                # This aggregated dimension is spanned by exactly one
                # fragment
                chunks.append((size,))

        if shapes == -1 or shapes is None:
            return chunks

        if isinstance(shapes, (str, Number)):
            return [
                chunks[i] if i in f_dims else shapes for i in range(self.ndim)
            ]

        if isinstance(shapes, dict):
            return [
                chunks[i] if i in f_dims else shapes.get(i, "auto")
                for i in range(self.ndim)
            ]

        if len(shapes) != self.ndim:
            # chunks is a sequence
            raise ValueError(
                f"Wrong number of 'shapes' elements in {shapes}: "
                f"Got {len(shapes)}, expected {self.ndim}"
            )

        # chunks is a sequence
        return [
            chunks[i] if i in f_dims else shapes[i] for i in range(self.ndim)
        ]

    def subarrays(self, *shapes):
        """Return descriptors for every subarray.

        .. versionadded:: TODODASKVER

        :Parameters:

            {{subarrays chunks: ``-1`` or sequence, optional}}

        :Returns:

            5-`tuple` of iterators
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

        """
        if shapes:
            subarray_shapes = shapes[0]
        else:
            subarray_shapes = self.subarray_shapes(shapes)

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
                # no fragmentation along this dimension
                f_locations.append(tuple(range(nc)))
            else:
                f_locations.append((0,) * nc)

            c = tuple(accumulate((0,) + c))
            u_indices.append([slice(i, j) for i, j in zip(c[:-1], c[1:])])

        # For each subarray, the part of the fragment that corresponds
        # to it.
        f_indices = [
            u if dim in f_dims else (slice(None),) * len(u)
            for dim, u in enumerate(u_indices)
        ]

        return (
            product(*u_indices),
            product(*u_shapes),
            product(*f_indices),
            product(*s_locations),
            product(*f_locations),
        )
