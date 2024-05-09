import logging
from numbers import Integral

import numpy as np
from cfdm import is_log_level_debug, is_log_level_info
from dask.array.slicing import normalize_index
from dask.base import is_dask_collection

from ..data import Data
from ..decorators import (
    _deprecated_kwarg_check,
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
    _manage_log_level_via_verbosity,
)
from ..functions import (
    _DEPRECATION_ERROR_KWARGS,
    bounds_combination_mode,
    normalize_slice,
)
from ..query import Query
from ..units import Units

logger = logging.getLogger(__name__)


_units_degrees = Units("degrees")

_empty_set = set()


class FieldDomain:
    """Mixin class for methods common to both field and domain
    constructs.

    .. versionadded:: 3.9.0

    """

    @property
    def _cyclic(self):
        """Storage for axis cyclicity.

        Do not change in-place.

        """
        return self._custom.get("_cyclic", _empty_set)

    @_cyclic.setter
    def _cyclic(self, value):
        """value must be a set.

        Do not change in-place.

        """
        self._custom["_cyclic"] = value

    @_cyclic.deleter
    def _cyclic(self):
        self._custom["_cyclic"] = _empty_set

    def _coordinate_reference_axes(self, key):
        """Returns the set of coordinate reference axes for a key.

        :Parameters:

            key: `str`
                Coordinate reference construct key.

        :Returns:

            `set`

        **Examples**

        >>> f._coordinate_reference_axes('coordinatereference0')

        """
        ref = self.constructs[key]

        axes = []

        for c_key in ref.coordinates():
            axes.extend(self.get_data_axes(c_key))

        for da_key in ref.coordinate_conversion.domain_ancillaries().values():
            axes.extend(self.get_data_axes(da_key))

        return set(axes)

    def _conform_coordinate_references(self, key, coordref=None):
        """Where possible replace the content of coordinate reference
        construct coordinates with coordinate construct keys.

        .. versionadded:: 3.0.0

        :Parameters:

            key: `str`
                Coordinate construct key.

            coordref: `CoordinateReference`, optional

                .. versionadded:: 3.6.0

        :Returns:

            `None`

        **Examples**

        >>> f._conform_coordinate_references('auxiliarycoordinate1')
        >>> f._conform_coordinate_references('auxiliarycoordinate1',
        ...                                  coordref=cr)

        """
        identity = self.constructs[key].identity(strict=True)

        if coordref is None:
            refs = self.coordinate_references(todict=True).values()
        else:
            refs = (coordref,)

        for ref in refs:
            if identity in ref.coordinates():
                ref.del_coordinate(identity, None)
                ref.set_coordinate(key)

    @_manage_log_level_via_verbosity
    def _equivalent_coordinate_references(
        self,
        field1,
        key0,
        key1,
        atol=None,
        rtol=None,
        s=None,
        t=None,
        verbose=None,
        axis_map=None,
    ):
        """True if coordinate reference constructs are equivalent.

        Two real numbers ``x`` and ``y`` are considered equal if
        ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on absolute
        differences) and ``rtol`` (the tolerance on relative differences)
        are positive, typically very small numbers. See the *atol* and
        *rtol* parameters.

        :Parameters:

            ref0: `CoordinateReference`

            ref1: `CoordinateReference`

            field1: `Field`
                The field which contains *ref1*.

        :Returns:

            `bool`

        """
        ref0 = self.coordinate_references(todict=True)[key0]
        ref1 = field1.coordinate_references(todict=True)[key1]

        if not ref0.equivalent(ref1, rtol=rtol, atol=atol, verbose=verbose):
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Non-equivalent coordinate "
                    f"references ({ref0!r}, {ref1!r})"
                )  # pragma: no cover

            return False

        # Compare the domain ancillaries
        # TODO consider case of None key ?
        for (
            term,
            identifier0,
        ) in ref0.coordinate_conversion.domain_ancillaries().items():
            if identifier0 is None:
                continue

            identifier1 = ref1.coordinate_conversion.domain_ancillaries()[term]

            #            key0 = domain_ancillaries.filter_by_key(identifier0).key()
            #            key1 = field1_domain_ancillaries.filter_by_key(identifier1).key()

            if not self._equivalent_construct_data(
                field1,
                key0=identifier0,  # key0,
                key1=identifier1,  # key1,
                rtol=rtol,
                atol=atol,
                s=s,
                t=t,
                verbose=verbose,
                axis_map=axis_map,
            ):
                # add traceback TODO
                return False

        return True

    def _indices(self, config, data_axes, ancillary_mask, kwargs):
        """Create indices that define a subspace of the field or domain
        construct.

        This method is intended to be called by the `indices` method.

        See the `indices` method for more details.

        .. versionadded:: 3.9.0

        .. seealso:: `_create_ancillary_mask_component`

        :Parameters:

            config: `tuple`
                The mode of operation and the halo size. See the
                *config* parameter of `indices` for details.

            data_axes: sequence of `str`, or `None`
                The domain axis identifiers of the data axes, or
                `None` if there is no data array.

            ancillary_mask: `bool`
                Whether or not to create ancillary masks. See
                `cf.Field.indices` for details.

            kwargs: `dict`, *optional*
                See the **kwargs** parameters of `indices` for
                details.

        :Returns:

            `dict`
                 The dictionary has two keys: ``'indices'`` and
                 ``'mask'``.

                 The ``'indices'`` key stores a dictionary keyed by
                 domain axis identifiers, each of which has a value of
                 the index for that domain axis.

                 The ``'mask'`` key stores a dictionary keyed by
                 tuples of domain axis identifier combinations, each
                 of which has of a `Data` object containing the
                 ancillary mask to apply to those domain axes
                 immediately after the the subspace has been created
                 by the ``'indices'``. This dictionary will always be
                 empty if the *ancillary_mask* parameter is False.

        """
        debug = is_log_level_debug(logger)

        # Parse mode and halo
        n_config = len(config)
        if not n_config:
            mode = None
            halo = None
        elif n_config == 1:
            try:
                halo = int(config[0])
            except ValueError:
                mode = config[0]
                halo = None
            else:
                mode = None
        elif n_config == 2:
            mode, halo = config
        else:
            raise ValueError(
                "Can't provide more than two positional arguments. "
                f"Got: {', '.join(repr(x) for x in config)}"
            )

        compress = mode is None or mode == "compress"
        envelope = mode == "envelope"
        full = mode == "full"
        if not (compress or envelope or full):
            raise ValueError(f"Invalid mode of operation: {mode!r}")

        if halo is not None:
            try:
                halo = int(halo)
            except ValueError:
                ok = False
            else:
                ok = halo >= 0

            if not ok:
                raise ValueError(
                    "halo positional argument must be convertible to a "
                    f"non-negative integer. Got {halo!r}"
                )

        domain_axes = self.domain_axes(todict=True)

        # Initialise the index for each axis
        indices = {axis: slice(None) for axis in domain_axes}

        parsed = {}
        unique_axes = set()
        n_axes = 0

        for identity, value in kwargs.items():
            key, construct = self.construct(
                identity, filter_by_data=True, item=True, default=(None, None)
            )
            if construct is not None:
                axes = self.get_data_axes(key)
            else:
                da_key = self.domain_axis(identity, key=True, default=None)
                if da_key is not None:
                    axes = (da_key,)
                    key = None
                    construct = None
                else:
                    raise ValueError(
                        f"Can't find indices. Ambiguous axis or axes "
                        f"defined by {identity!r}"
                    )

            if axes in parsed:
                # The axes are the same as an existing key
                parsed[axes].append((axes, key, construct, value, identity))
            else:
                new_key = True
                y = set(axes)
                for x in parsed:
                    if set(x) == set(y):
                        # The axes are the same but in a different
                        # order, so we don't need a new key.
                        parsed[x].append(
                            (axes, key, construct, value, identity)
                        )
                        new_key = False
                        break

                if new_key:
                    # The axes, taken in any order, are not the same
                    # as any keys, so create an new key.
                    n_axes += len(axes)
                    parsed[axes] = [(axes, key, construct, value, identity)]

            unique_axes.update(axes)

        if debug:
            logger.debug(
                f"  parsed       = {parsed!r}\n"
                f"  unique_axes  = {unique_axes!r}\n"
                f"  n_axes       = {n_axes!r}"
            )  # pragma: no cover

        if len(unique_axes) < n_axes:
            raise ValueError(
                "Can't find indices: Multiple constructs with incompatible "
                "domain axes"
            )

        mask = {}

        for canonical_axes, axes_key_construct_value_id in parsed.items():
            axes, keys, constructs, points, identities = tuple(
                zip(*axes_key_construct_value_id)
            )

            n_items = len(constructs)
            n_axes = len(canonical_axes)

            if n_items > n_axes:
                if n_axes == 1:
                    a = "axis"
                else:
                    a = "axes"

                raise ValueError(
                    f"Error: Can't specify {n_items} conditions for "
                    f"{n_axes} {a}: {points}. Consider applying the "
                    "conditions separately."
                )

            create_mask = False

            item_axes = axes[0]

            if debug:
                logger.debug(
                    f"  item_axes    = {item_axes!r}\n"
                    f"  keys         = {keys!r}"
                )  # pragma: no cover

            if n_axes == 1:
                # ----------------------------------------------------
                # 1-d construct
                # ----------------------------------------------------
                ind = None

                axis = item_axes[0]
                item = constructs[0]
                value = points[0]
                identity = identities[0]

                if debug:
                    logger.debug(
                        f"  {n_items} 1-d constructs: {constructs!r}\n"
                        f"  axis         = {axis!r}\n"
                        f"  value        = {value!r}\n"
                        f"  identity     = {identity!r}"
                    )  # pragma: no cover

                if isinstance(value, (list, slice, tuple, np.ndarray)):
                    # 1-d CASE 1: Value is already an index, e.g. [0],
                    #             [7,4,2], slice(0,4,2),
                    #             numpy.array([2,4,7]),
                    #             [True,False,True]
                    if debug:
                        logger.debug("  1-d CASE 1:")  # pragma: no cover

                    index = value

                    if envelope or full:
                        # Set ind
                        size = domain_axes[axis].get_size()
                        ind = (np.arange(size)[value],)
                        # Placeholder which will be overwritten later
                        index = None

                elif (
                    item is not None
                    and isinstance(value, Query)
                    and value.operator in ("wi", "wo")
                    and item.construct_type == "dimension_coordinate"
                    and self.iscyclic(axis)
                ):
                    # 1-d CASE 2: Axis is cyclic and subspace
                    #             criterion is a 'within' or 'without'
                    #             Query instance
                    if debug:
                        logger.debug("  1-d CASE 2:")  # pragma: no cover

                    if item.increasing:
                        anchor0 = value.value[0]
                        anchor1 = value.value[1]
                    else:
                        anchor0 = value.value[1]
                        anchor1 = value.value[0]

                    a = self.anchor(axis, anchor0, dry_run=True)["roll"]
                    b = self.flip(axis).anchor(axis, anchor1, dry_run=True)[
                        "roll"
                    ]

                    size = item.size
                    if abs(anchor1 - anchor0) >= item.period():
                        if value.operator == "wo":
                            set_start_stop = 0
                        else:
                            set_start_stop = -a

                        start = set_start_stop
                        stop = set_start_stop
                    elif a + b == size:
                        b = self.anchor(axis, anchor1, dry_run=True)["roll"]
                        if (b == a and value.operator == "wo") or not (
                            b == a or value.operator == "wo"
                        ):
                            set_start_stop = -a
                        else:
                            set_start_stop = 0

                        start = set_start_stop
                        stop = set_start_stop
                    else:
                        if value.operator == "wo":
                            start = b - size
                            stop = -a + size
                        else:
                            start = -a
                            stop = b - size

                    if start == stop == 0:
                        raise ValueError(
                            f"No indices found from: {identity}={value!r}"
                        )

                    index = slice(start, stop, 1)

                    if full:
                        # Set ind
                        try:
                            index = normalize_slice(index, size, cyclic=True)
                        except IndexError:
                            # Index is not a cyclic slice
                            ind = (np.arange(size)[index],)
                        else:
                            # Index is a cyclic slice
                            ind = (
                                np.arange(size)[
                                    np.arange(
                                        index.start, index.stop, index.step
                                    )
                                ],
                            )

                        # Placeholder which will be overwritten later
                        index = None

                elif item is not None:
                    # 1-d CASE 3: All other 1-d cases
                    if debug:
                        logger.debug("  1-d CASE 3:")  # pragma: no cover

                    index = item == value

                    # Performance: Convert the 1-d 'index' to a numpy
                    #              array of bool.
                    #
                    # This is beacuse Dask can be *very* slow at
                    # instantiation time when the 'index' is a Dask
                    # array, in which case contents of 'index' are
                    # unknown.
                    index = np.asanyarray(index)

                    if envelope or full:
                        # Set ind
                        index = np.asanyarray(index)
                        if np.ma.isMA(index):
                            ind = np.ma.where(index)
                        else:
                            ind = np.where(index)

                        # Placeholder which will be overwritten later
                        index = None
                    else:
                        # Convert bool to int, to save memory.
                        size = domain_axes[axis].get_size()
                        index = normalize_index(index, (size,))[0]
                else:
                    raise ValueError(
                        "Must specify a domain axis construct or a "
                        "construct with data for which to create indices"
                    )

                if debug:
                    logger.debug(
                        f"    index      = {index}\n    ind        = {ind}"
                    )  # pragma: no cover

                # Put the index into the correct place in the list of
                # indices.
                #
                # Note that we might overwrite it later if there's an
                # ancillary mask for this axis.
                indices[axis] = index

            else:
                # ----------------------------------------------------
                # N-d constructs
                # ----------------------------------------------------
                if debug:
                    logger.debug(
                        f"  {n_items} N-d constructs: {constructs!r}\n"
                        f"  {len(points)} points        : {points!r}\n"
                    )  # pragma: no cover

                # Make sure that each N-d item has the same axis order
                transposed_constructs = []

                for construct, construct_axes in zip(constructs, axes):
                    if construct_axes != canonical_axes:
                        iaxes = [
                            construct_axes.index(axis)
                            for axis in canonical_axes
                        ]
                        construct = construct.transpose(iaxes)

                    transposed_constructs.append(construct)

                if debug:
                    logger.debug(
                        "  transposed N-d constructs: "
                        f"{transposed_constructs!r}"
                    )  # pragma: no cover

                # Find where each construct matches its value
                item_matches = [
                    (construct == value).data
                    for value, construct in zip(points, transposed_constructs)
                ]

                # Find loctions that are True in all of the
                # constructs' matches
                item_match = item_matches.pop()
                for m in item_matches:
                    item_match &= m

                # Set ind
                item_match = np.asanyarray(item_match)
                if np.ma.isMA(item_match):
                    ind = np.ma.where(item_match)
                else:
                    ind = np.where(item_match)

                # Placeholders which will be overwritten later
                for axis in canonical_axes:
                    indices[axis] = None

                if debug:
                    logger.debug(
                        f"  item_match  = {item_match}\n"
                        f"  ind         = {ind}"
                    )  # pragma: no cover

                for i in ind:
                    if not i.size:
                        raise ValueError(
                            f"No {canonical_axes!r} axis indices found "
                            f"from: {value!r}"
                        )

                bounds = [
                    item.bounds
                    for item in transposed_constructs
                    if item.has_bounds()
                ]

                # If there are exactly two 2-d constructs, both with
                # cell bounds and both with 'cf.contains' values, then
                # do an extra check to remove any cells already
                # selected for which the given value is in fact
                # outside of the cell. This could happen if the cells
                # are not rectilinear (e.g. for curvilinear latitudes
                # and longitudes arrays).
                if n_items == constructs[0].ndim == len(bounds) == 2:
                    point2 = []
                    for v, construct in zip(points, transposed_constructs):
                        if isinstance(v, Query) and v.iscontains():
                            v = self._Data.asdata(v.value)
                            if v.Units:
                                v.Units = construct.Units

                            point2.append(v.datum())
                        else:
                            point2 = None
                            break

                    if point2:
                        from dask import compute, delayed

                        try:
                            from matplotlib.path import Path
                        except ModuleNotFoundError:
                            x = ", ".join(
                                [
                                    f"{i}={p!r}"
                                    for i, p in zip(identities, points)
                                ]
                            )
                            raise ImportError(
                                "Must install matplotlib to create indices "
                                f"for {self!r} from: {x}"
                            )

                        def _point_not_in_cell(nodes_x, nodes_y, point):
                            """Return True if a point is not in a 2-d
                            cell.

                            :Parameters:

                                nodes_x: array-like
                                    The cell x nodes

                                nodes_y: array-like
                                    The cell y nodes

                                point: (number, number)
                                    The (x, y) point to check.

                            :Returns:

                                `bool`

                            """
                            vertices = tuple(zip(nodes_x, nodes_y))
                            return not Path(vertices).contains_point(point)

                        bounds = [b.array[ind] for b in bounds]
                        delete = compute(
                            *[
                                delayed(_point_not_in_cell(x, y, point2))
                                for x, y in zip(*bounds)
                            ]
                        )
                        if any(delete):
                            ind = [np.delete(ind_1d, delete) for ind_1d in ind]

            if ind is not None:
                mask_component_shape = []
                masked_subspace_size = 1
                # TODONUMPY2: https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
                ind = np.array(ind, copy=False)

                for i, (axis, start, stop) in enumerate(
                    zip(canonical_axes, ind.min(axis=1), ind.max(axis=1))
                ):
                    if data_axes and axis not in data_axes:
                        continue

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
                        stop = domain_axes[axis].get_size()
                        size = stop - start
                        index = slice(None)
                    else:
                        raise ValueError(
                            "Must have mode full, envelope, or compress"
                        )  # pragma: no cover

                    # Overwrite the placeholder value of None
                    if indices[axis] is not None:
                        raise ValueError(
                            "This error means that there is a bug: The "
                            "'indices' dictionary should contain None for "
                            "each axis with an 'ind'."
                        )

                    indices[axis] = index

                    mask_component_shape.append(size)
                    masked_subspace_size *= size
                    ind[i] -= start

                create_mask = (
                    ancillary_mask
                    and halo is None
                    and data_axes
                    and ind.shape[1] < masked_subspace_size
                )
            else:
                create_mask = False

            # --------------------------------------------------------
            # Add a halo to the subspaced axes
            # --------------------------------------------------------
            if halo:
                # Note: We're about to make 'indices' inconsistent
                #       with 'ind', but that's OK because we're not
                #       going to use 'ind' again as 'create_mask' is
                #       False.
                reduced_halo = False
                for axis in item_axes:
                    index = indices[axis]
                    size = domain_axes[axis].get_size()

                    try:
                        # Is index a cyclic slice?
                        index = normalize_slice(index, size, cyclic=True)
                    except IndexError:
                        try:
                            index = normalize_slice(index, size)
                        except IndexError:
                            # Index is not a slice
                            cyclic = False
                        else:
                            # Index is a non-cyclic slice, but if the
                            # axis is cyclic then it could become a
                            # cyclic slice once the halo is added.
                            cyclic = self.iscyclic(axis)
                    else:
                        # Index is a cyclic slice
                        cyclic = self.iscyclic(axis)
                        if not cyclic:
                            raise IndexError(
                                "Can't take a cyclic slice of a non-cyclic "
                                "axis"
                            )

                    if cyclic:
                        # Cyclic slice, or potentially cyclic slice.
                        #
                        # E.g. for halo=1 and size=5:
                        #   slice(0, 2)        -> slice(-1, 3)
                        #   slice(-1, 2)       -> slice(-2, 3)
                        #   slice(1, None, -1) -> slice(2, -2, -1)
                        #   slice(1, -2, -1)   -> slice(2, -3, -1)
                        start = index.start
                        stop = index.stop
                        step = index.step
                        if step not in (1, -1):
                            # This restriction is due to the fact that
                            # the extended index is a slice (rather
                            # than a list of integers), and so we
                            # can't represent the uneven spacing that
                            # would be required if abs(step) != 1.
                            # Note that cyclic slices created by this
                            # method will always have a step of 1.
                            raise IndexError(
                                "A cyclic slice index can only have halos if "
                                f"it has step 1 or -1. Got {index!r}"
                            )

                        if step < 0 and stop is None:
                            stop = -1

                        if step > 0:
                            # Increasing cyclic slice
                            start = start - halo
                            if start < stop - size:
                                start = stop - size
                                reduced_halo = True

                            stop = stop + halo
                            if stop > size + start:
                                stop = size + start
                                reduced_halo = True

                        else:
                            # Decreasing cyclic slice
                            start = start + halo
                            if start > size + stop:
                                start = size + stop
                                reduced_halo = True

                            stop = stop - halo
                            if stop < start - size:
                                stop = start - size
                                reduced_halo = True

                        index = slice(start, stop, step)
                    else:
                        # A list/1-d array of int/bool, or a
                        # non-cyclic slice that can't become cyclic.
                        #
                        # E.g. for halo=1 and size=5:
                        #   slice(1, 3)                       -> [0, 1, 2, 3]
                        #   slice(1, 4, 2)                    -> [0, 1, 3, 4]
                        #   slice(2, 0, -1)                   -> [3, 2, 1, 0]
                        #   slice(2, 0, -1)                   -> [3, 2, 1, 0]
                        #   [1, 2]                            -> [0, 1, 2, 3]
                        #   [1, 3]                            -> [0, 1, 3, 4]
                        #   [2, 1]                            -> [3, 2, 1, 0]
                        #   [3, 1]                            -> [4, 3, 1, 0]
                        #   [1, 3, 2]                         -> [0, 1, 3, 2, 1]
                        #   [False, True, False, True, False] -> [0, 1, 3, 4]
                        if isinstance(index, slice):
                            index = np.arange(size)[index]
                        else:
                            if is_dask_collection(index):
                                index = np.asanyarray(index)

                            index = normalize_index(index, (size,))[0]

                        # Find the left-most and right-most elements
                        # ('iL' and iR') of the sequence of positive
                        # integers, and whether the sequence is
                        # increasing or decreasing at each end
                        # ('increasing_L' and 'increasing_R')
                        #
                        # For instance:
                        #
                        # ------------ -- -- ------------ ------------
                        # index        iL iR increasing_L increasing_R
                        # ------------ -- -- ------------ ------------
                        # [1, 2, 3, 4]  1  4 True         True
                        # [4, 3, 2, 1]  4  1 False        False
                        # [2, 1, 3, 4]  2  4 False        True
                        # [1, 2, 4, 3]  1  3 True         False
                        # [2, 2, 3, 4]  2  4 True         True
                        # [2, 2, 4, 3]  2  3 True         False
                        # [3, 3, 4, 4]  3  4 True         True
                        # [4, 4, 3, 3]  4  3 False        False
                        # [10]         10 10 True         True
                        # ------------ -- -- ------------ ------------
                        n_index = index.size
                        if n_index == 1:
                            iL = index[0]
                            iR = iL
                            increasing_L = True
                            increasing_R = True
                        elif n_index > 1:
                            iL = index[0]
                            iR = index[-1]
                            increasing_L = iL <= index[np.argmax(index != iL)]
                            increasing_R = (
                                iR >= index[-1 - np.argmax(index[::-1] != iR)]
                            )
                        else:
                            raise IndexError(
                                "Can't add a halo to a zero-sized index: "
                                f"{index}"
                            )

                        # Extend the list at each end, but not
                        # exceeding the axis limits.
                        if increasing_L:
                            start = iL - halo
                            if start < 0:
                                start = 0
                                reduced_halo = True

                            left = range(start, iL)
                        else:
                            start = iL + halo
                            if start > size - 1:
                                start = size - 1
                                reduced_halo = True

                            left = range(start, iL, -1)

                        if increasing_R:
                            stop = iR + 1 + halo
                            if stop > size:
                                stop = size
                                reduced_halo = True

                            right = range(iR + 1, stop)
                        else:
                            stop = iR - 1 - halo
                            if stop < -1:
                                stop = -1
                                reduced_halo = True

                            right = range(iR - 1, stop, -1)

                        index = index.tolist()
                        index[:0] = left
                        index.extend(right)

                    # Reset the returned index
                    indices[axis] = index

                if reduced_halo:
                    logger.warning(
                        "Halo reduced to keep subspace within axis limits"
                    )

            # Create an ancillary mask for these axes
            if debug:
                logger.debug(
                    f"  create_mask  = {create_mask}"
                )  # pragma: no cover

            if create_mask:
                mask[canonical_axes] = _create_ancillary_mask_component(
                    mask_component_shape, ind, compress
                )

        indices = {"indices": indices, "mask": mask}

        if debug:
            logger.debug(f"  indices      = {indices!r}")  # pragma: no cover

        # Return the indices and ancillary masks
        return indices

    def _roll_constructs(self, axis, shift):
        """Roll the metadata constructs in-place along axes.

        If a roll axis is spanned by a dimension coordinate construct
        then it must be a periodic dimension coordinate construct.

        .. versionadded:: 3.9.0

        :Parameters:

            axis: sequence of `str`
                The axis or axes along which elements are to be
                shifted, defined by their domain axis identifiers.

            shift: (sequence of) `int`
                The number of places by which elements are shifted.
                If a sequence, then *axis* must be a sequence of the
                same size, and each of the given axes is shifted by
                the corresponding number.  If an `int` while *axis* is
                a sequence, then the same value is used for all given
                axes.

        :Returns:

            `list`

                The shifts corresponding to each rolled axis.

        **Examples**

        """
        if isinstance(shift, Integral):
            if axis:
                shift = [shift] * len(axis)
            else:
                shift = [shift]
        else:
            shift = list(shift)

        if len(shift) != len(axis):
            raise ValueError(
                f"Can't roll {self.__class__.__name__}: "
                f"Must have the same number of shifts ({len(shift)}) "
                f"as axes ({len(axis)})."
            )

        for a in axis:
            dim = self.dimension_coordinate(filter_by_axis=(a,), default=None)
            if dim is not None and dim.period() is None:
                raise ValueError(
                    f"Can't roll {self.__class__.__name__}. "
                    f"{dim.identity()!r} axis has a non-periodic "
                    "dimension coordinate construct"
                )

        data_axes = self.constructs.data_axes()
        for key, construct in self.constructs.filter_by_data(
            todict=True
        ).items():
            construct_axes = data_axes.get(key, ())

            c_axes = []
            c_shifts = []
            for a, s in zip(axis, shift):
                if a in construct_axes:
                    c_axes.append(construct_axes.index(a))
                    c_shifts.append(s)

            if not c_axes:
                # This construct does not span the roll axes
                continue

            # TODODASK: Consider removing these two lines, now that
            #           multiaxis rolls are allowed on Data objects.
            c_axes = c_axes[0]
            c_shifts = c_shifts[0]

            construct.roll(c_axes, shift=c_shifts, inplace=True)

        return shift

    def _set_construct_parse_axes(self, item, axes=None, allow_scalar=True):
        """Parse axes for the set_construct method.

        :Parameters:

            item: metadata construct

            axes: (sequence of) `str or `int`, optional

            allow_scalar: `bool`, optional

        :Returns:

            `list`

        """
        data = item.get_data(None, _fill_value=False)

        if axes is None:
            # --------------------------------------------------------
            # The axes have not been set => infer the axes.
            # --------------------------------------------------------
            if data is not None:
                shape = item.shape
                if allow_scalar and shape == ():
                    axes = []
                else:
                    if not allow_scalar and not shape:
                        shape = (1,)

                    if not shape or len(shape) != len(set(shape)):
                        raise ValueError(
                            f"Can't insert {item!r}: Ambiguous shape: "
                            f"{shape}. Consider setting the 'axes' parameter."
                        )

                    domain_axes = self.domain_axes(todict=True)
                    axes = []
                    axes_sizes = [
                        domain_axis.get_size(None)
                        for domain_axis in domain_axes.values()
                    ]

                    for n in shape:
                        if not axes_sizes.count(n):
                            raise ValueError(
                                f"Can't insert {item!r}: There is no "
                                f"domain axis construct with size {n}."
                            )

                        if axes_sizes.count(n) != 1:
                            raise ValueError(
                                f"Can't insert {item!r}: Ambiguous shape: "
                                "f{shape}. Consider setting the 'axes' "
                                "parameter."
                            )

                        da_key = self.domain_axis(
                            filter_by_size=(n,), key=True
                        )
                        axes.append(da_key)
        else:
            # --------------------------------------------------------
            # Axes have been provided
            # --------------------------------------------------------
            if isinstance(axes, (str, int)):
                axes = (axes,)

            if axes and data is not None:
                ndim = item.ndim
                if not ndim and not allow_scalar:
                    ndim = 1

                if isinstance(axes, (str, int)):
                    axes = (axes,)

                if len(axes) != ndim or len(set(axes)) != ndim:
                    raise ValueError(
                        f"Can't insert {item!r}: Incorrect number of given "
                        f"axes (got {len(set(axes))}, expected {ndim})"
                    )

                axes2 = []
                for axis, size in zip(axes, item.data.shape):
                    da_key, domain_axis = self.domain_axis(
                        axis,
                        item=True,
                        default=ValueError(f"Unknown axis: {axis!r}"),
                    )

                    axis_size = domain_axis.get_size(None)
                    if size != axis_size:
                        raise ValueError(
                            f"Can't insert {item!r}: Mismatched axis size "
                            f"({size} != {axis_size})"
                        )

                    axes2.append(da_key)

                axes = axes2

                if ndim != len(set(axes)):
                    raise ValueError(
                        f"Can't insert {item!r}: Mismatched number of axes "
                        f"({len(set(axes))} != {ndim})"
                    )

        return axes

    @property
    def rank(self):
        """The number of axes in the domain.

        **Examples**

        TODO

        """
        return len(self.domain_axes(todict=True))

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def anchor(
        self, axis, value, inplace=False, dry_run=False, i=False, **kwargs
    ):
        """Roll a cyclic axis so that the given value lies in the first
        coordinate cell.

        A unique axis is selected with the *axes* and *kwargs*
        parameters.

        .. versionadded:: 3.9.0

        .. seealso:: `axis`, `cyclic`, `iscyclic`, `roll`

        :Parameters:

            axis:
                The cyclic axis to be anchored.

                domain axis selection TODO.

            value:
                Anchor the dimension coordinate values for the
                selected cyclic axis to the *value*. May be any
                numeric scalar object that can be converted to a
                `Data` object (which includes `numpy` and `Data`
                objects). If *value* has units then they must be
                compatible with those of the dimension coordinates,
                otherwise it is assumed to have the same units as the
                dimension coordinates. The coordinate values are
                transformed so that *value* is "equal to or just
                before" the new first coordinate value. More
                specifically:

                  * Increasing dimension coordinates with positive
                    period, P, are transformed so that *value* lies in
                    the half-open range (L-P, F], where F and L are
                    the transformed first and last coordinate values,
                    respectively.

            ..

                  * Decreasing dimension coordinates with positive
                    period, P, are transformed so that *value* lies in
                    the half-open range (L+P, F], where F and L are
                    the transformed first and last coordinate values,
                    respectively.

                *Parameter example:*
                  If the original dimension coordinates are ``0, 5,
                  ..., 355`` (evenly spaced) and the period is ``360``
                  then ``value=0`` implies transformed coordinates of
                  ``0, 5, ..., 355``; ``value=-12`` implies
                  transformed coordinates of ``-10, -5, ..., 345``;
                  ``value=380`` implies transformed coordinates of
                  ``380, 385, ..., 715``.

                *Parameter example:*
                  If the original dimension coordinates are ``355,
                  350, ..., 0`` (evenly spaced) and the period is
                  ``360`` then ``value=355`` implies transformed
                  coordinates of ``355, 350, ..., 0``; ``value=0``
                  implies transformed coordinates of ``0, -5, ...,
                  -355``; ``value=392`` implies transformed
                  coordinates of ``390, 385, ..., 30``.

            {{inplace: `bool`, optional}}

            dry_run: `bool`, optional
                Return a dictionary of parameters which describe the
                anchoring process. The construct is not changed, even
                if *inplace* is True.

            {{i: deprecated at version 3.0.0}}

            kwargs: deprecated at version 3.0.0

        :Returns:

            `dict`

        **Examples**

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

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "anchor", kwargs
            )  # pragma: no cover

        da_key, axis = self.domain_axis(axis, item=True)

        if dry_run:
            f = self
        else:
            f = _inplace_enabled_define_and_cleanup(self)

        dim = f.dimension_coordinate(filter_by_axis=(da_key,), default=None)
        if dim is None:
            raise ValueError(
                "Can't shift non-cyclic "
                f"{f.constructs.domain_axis_identity(da_key)!r} axis"
            )

        period = dim.period()
        if period is None:
            raise ValueError(f"Cyclic {dim.identity()!r} axis has no period")

        value = f._Data.asdata(value)
        if not value.Units:
            value = value.override_units(dim.Units)
        elif not value.Units.equivalent(dim.Units):
            raise ValueError(
                f"Anchor value has incompatible units: {value.Units!r}"
            )

        axis_size = axis.get_size()

        if axis_size <= 1:
            # Don't need to roll a size one axis
            if dry_run:
                return {"axis": da_key, "roll": 0, "nperiod": 0}

            return f

        c = dim.get_data(_fill_value=False)

        if dim.increasing:
            # Adjust value so it's in the range [c[0], c[0]+period)
            n = ((c[0] - value) / period).ceil()
            value1 = value + n * period

            shift = axis_size - np.argmax((c - value1 >= 0).array)
            if not dry_run:
                f.roll(da_key, shift, inplace=True)

            # Re-get dim
            dim = f.dimension_coordinate(filter_by_axis=(da_key,))
            # TODO CHECK n for dry run or not
            n = ((value - dim.data[0]) / period).ceil()
        else:
            # Adjust value so it's in the range (c[0]-period, c[0]]
            n = ((c[0] - value) / period).floor()
            value1 = value + n * period

            shift = axis_size - np.argmax((value1 - c >= 0).array)

            if not dry_run:
                f.roll(da_key, shift, inplace=True)

            # Re-get dim
            dim = f.dimension_coordinate(filter_by_axis=(da_key,))
            # TODO CHECK n for dry run or not
            n = ((value - dim.data[0]) / period).floor()

        if dry_run:
            return {"axis": da_key, "roll": shift, "nperiod": n * period}

        if n:
            with bounds_combination_mode("OR"):
                dim += n * period

        return f

    @_manage_log_level_via_verbosity
    def autocyclic(self, key=None, coord=None, verbose=None, config={}):
        """Set dimensions to be cyclic.

        A dimension is set to be cyclic if it has a unique longitude
        (or grid longitude) dimension coordinate construct with bounds
        and the first and last bounds values differ by 360 degrees (or
        an equivalent amount in other units).

        .. versionadded:: 1.0

        .. seealso:: `cyclic`, `iscyclic`, `period`

        :Parameters:

            {{verbose: `int` or `str` or `None`, optional}}

            config: `dict`
                Additional parameters for optimising the
                operation. See the code for details.

                .. versionadded:: 3.9.0

        :Returns:

           `bool` or `None`
               `True` if the dimension is cyclic, `False` if it isn't,
               or `None` if no checks were done.

        """
        noop = config.get("no-op")
        if noop:
            # Don't do anything
            return

        if "cyclic" in config:
            if not config["cyclic"]:
                if not noop:
                    self.cyclic(key, iscyclic=False, config=config)
                return False
            else:
                period = coord.period()
                if period is not None:
                    period = None
                else:
                    period = config.get("period")

                self.cyclic(key, iscyclic=True, period=period, config=config)
                return True

        if coord is None:
            key, coord = self.dimension_coordinate(
                "X", item=True, default=(None, None)
            )
            if coord is None:
                return False
        elif "X" in config:
            if not config["X"]:
                if not noop:
                    self.cyclic(key, iscyclic=False, config=config)
                return False
        elif not coord.X:
            if not noop:
                self.cyclic(key, iscyclic=False, config=config)
            return False

        bounds_range = config.get("bounds_range")
        if bounds_range is not None:
            bounds_units = config["bounds_units"]
        else:
            bounds = coord.get_bounds(None)
            if bounds is None:
                if not noop:
                    self.cyclic(key, iscyclic=False, config=config)
                return False

            data = bounds.get_data(None, _fill_value=False)
            if data is None:
                if not noop:
                    self.cyclic(key, iscyclic=False, config=config)
                return False

            bounds_units = bounds.Units

        period = coord.period()

        if period is not None:
            has_period = True
        else:
            period = config.get("period")
            if period is None:
                has_period = False
            else:
                has_period = True

        if not has_period:
            if bounds_units.islongitude:
                period = Data(360.0, units="degrees_east")
            elif bounds_units.equivalent(_units_degrees):
                period = Data(360.0, units="degrees")
            else:
                self.cyclic(key, iscyclic=False, config=config)
                return False

            period.Units = bounds_units

        if bounds_range is None:
            bounds_range = abs(data.last_element() - data.first_element())
            if bounds_range is np.ma.masked:
                bounds_range = None

        if bounds_range is None or bounds_range != period:
            if not noop:
                self.cyclic(key, iscyclic=False, config=config)
            return False

        config = config.copy()
        config["axis"] = self.get_data_axes(key, default=(None,))[0]

        self.cyclic(key, iscyclic=True, period=period, config=config)

        return True

    @_inplace_enabled(default=False)
    def auxiliary_to_dimension(
        self, *identity, inplace=False, **filter_kwargs
    ):
        """Move auxiliary coordinates to a dimension coordinate construct.

        A new dimension coordinate construct is derived
        from the selected auxiliary coordinate construct, and the
        auxiliary coordinate construct is removed.

        .. versionadded:: 3.14.1

        .. seealso:: `dimension_to_auxiliary`

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique 1-d auxiliary coordinate construct
                returned by ``f.auxiliary_coordinate(*identity,
                filter_by_naxes=(1,), **filter_kwargs)``. See
                `auxiliary_coordinate` for details.

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The {{class}} with the new dimension coordinate construct,
                or `None` if the operation was in-place.

        **Examples**

        >>> f = cf.example_field(0)
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> g = f.dimension_to_auxiliary('latitude')
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        Auxiliary coords: latitude(latitude(5)) = [-75.0, ..., 75.0] degrees_north
        >>> h = g.auxiliary_to_dimension('latitude')
        >>> print(h)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> h.equals(f)
        True

        """
        f = _inplace_enabled_define_and_cleanup(self)

        filter_kwargs["filter_by_naxes"] = (1,)

        key, aux = f.auxiliary_coordinate(
            *identity, item=True, **filter_kwargs
        )

        if aux.dtype.kind in "SU":
            raise ValueError(
                f"Can't create a dimension coordinate construct from {aux!r} "
                f"with datatype {aux.dtype}. Only numerical auxiliary "
                "coordinate constructs can be converted."
            )

        if aux.has_geometry():
            raise ValueError(
                f"Can't create a dimension coordinate construct from {aux!r} "
                "with geometry cells"
            )

        axis = f.get_data_axes(key)
        dim = f._DimensionCoordinate(source=aux)
        f.set_construct(dim, axes=axis)
        f.del_construct(key)
        return f

    def del_coordinate_reference(
        self, identity=None, construct=None, default=ValueError()
    ):
        """Remove a coordinate reference construct and all of its domain
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
                          coordinate reference construct when there is only
                          one of them.

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

                **Examples**

                >>> f.del_coordinate_reference('rotated_latitude_longitude')
                <CF CoordinateReference: rotated_latitude_longitude>

        """
        if construct is None:
            if identity is None:
                raise ValueError(
                    "An identity or construct must be provided in order to "
                    "determine the coordinate reference to select."
                )

            key = self.coordinate_reference(identity, key=True, default=None)
            if key is None:
                if default is None:
                    return

                return self._default(
                    default, f"Can't identify construct from {identity!r}"
                )

            ref = self.del_construct(key)

            for (
                da_key
            ) in ref.coordinate_conversion.domain_ancillaries().values():
                self.del_construct(da_key, default=None)

            return ref
        elif identity is not None:
            raise ValueError(
                "Provide only one of the identity and construct parameters "
                "to select a coordinate reference."
            )

        out = []

        c_key = self.construct(construct, key=True, default=None)
        if c_key is None:
            if default is None:
                return

            return self._default(
                default, f"Can't identify construct from {construct!r}"
            )

        for key, ref in tuple(self.coordinate_references(todict=True).items()):
            if c_key in ref.coordinates():
                self.del_coordinate_reference(
                    key, construct=None, default=default
                )
                out.append(ref)
                continue

            if (
                c_key
                in ref.coordinate_conversion.domain_ancillaries().values()
            ):
                self.del_coordinate_reference(
                    key, construct=None, default=default
                )
                out.append(ref)
                continue

        return out

    def del_domain_axis(
        self, identity=None, squeeze=False, default=ValueError()
    ):
        """Remove a domain axis construct.

        In general, a domain axis construct can only be removed if it
        is not spanned by any construct's data. However, a size 1
        domain axis construct can be removed in any case if the
        *squeeze* parameter is set to `True`. In this case, a metadata
        construct whose data spans only the removed domain axis
        construct will also be removed.

        .. versionadded:: 3.6.0

        .. seealso:: `del_construct`

        :Parameters:

            identity: optional
                Select the domain axis construct by one of:

                * An identity or key of a 1-d dimension or auxiliary
                  coordinate construct that whose data spans the
                  domain axis construct.

                  {{construct selection identity}}

                * A domain axis construct identity.

                  {{domain axis selection identity}}

                * The key of a domain axis construct.

                * `None`. This is the default, which selects the
                  domain axis construct when there is only one of
                  them.  ``'key%dimensioncoordinate2'`` are both
                  acceptable keys.

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
                construct that is spanned by any data array and
                squeeze the corresponding dimension from those arrays.

            default: optional
                Return the value of the *default* parameter if the
                construct can not be removed, or does not exist.

                {{default Exception}}

        :Returns:

            `DomainAxis`
                The removed domain axis construct.

        **Examples**

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

        """
        dakey, domain_axis = self.domain_axis(identity, item=True)

        if not squeeze:
            return self.del_construct(dakey)

        if dakey in self.get_data_axes(default=()):
            self.squeeze(dakey, inplace=True)

        for ckey, construct in self.constructs.filter_by_data(
            todict=True
        ).items():
            data = construct.get_data(None, _fill_value=False)
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

        return domain_axis

    def coordinate_reference_domain_axes(self, identity=None):
        """Return the domain axes that apply to a coordinate reference
        construct.

                :Parameters:

                    identity: optional
                        Select the coordinate reference construct by one of:

                        * The identity of a coordinate reference construct.

                          {{construct selection identity}}

                        * The key of a coordinate reference construct

                        * `None`. This is the default, which selects the
                          coordinate reference construct when there is only
                          one of them.

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
                        The identifiers of the domain axis constructs that span
                        the data of all coordinate and domain ancillary
                        constructs used by the selected coordinate reference
                        construct.

                **Examples**

                >>> f.coordinate_reference_domain_axes('coordinatereference0')
                {'domainaxis0', 'domainaxis1', 'domainaxis2'}

                >>> f.coordinate_reference_domain_axes(
                ...     'atmosphere_hybrid_height_coordinate')
                {'domainaxis0', 'domainaxis1', 'domainaxis2'}

        """
        cr = self.coordinate_reference(identity)

        data_axes = self.constructs.data_axes()

        axes = []
        for i in cr.coordinates() | set(
            cr.coordinate_conversion.domain_ancillaries().values()
        ):
            key = self.construct(i, key=True, default=None)
            axes.extend(data_axes.get(key, ()))

        return set(axes)

    def cyclic(
        self, *identity, iscyclic=True, period=None, config={}, **filter_kwargs
    ):
        """Get or set the cyclicity of an axis.

        .. versionadded:: 1.0

        .. seealso:: `autocyclic`, `iscyclic`, `period`, `domain_axis`

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique domain axis construct returned by
                ``f.domain_axis(*identity, **filter_kwargs)``. See
                `domain_axis` for details.

            iscyclic: `bool`, optional
                If False then the axis is set to be non-cyclic. By
                default the selected axis is set to be cyclic.

            period: optional
                The period for a dimension coordinate construct which
                spans the selected axis. May be any numeric scalar
                object that can be converted to a `Data` object (which
                includes numpy array and `Data` objects). The absolute
                value of *period* is used. If *period* has units then
                they must be compatible with those of the dimension
                coordinates, otherwise it is assumed to have the same
                units as the dimension coordinates.

            config: `dict`, optional
                Additional parameters for optimising the
                operation. See the code for details.

                .. versionadded:: 3.9.0

            axes: deprecated at version 3.0.0
                Use the *identity* and *filter_kwargs* parameters
                instead.

        :Returns:

            `set`
                The construct keys of the domain axes which were
                cyclic prior to the new setting, or the current cyclic
                domain axes if no axis was specified.

        **Examples**

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

        """
        if not iscyclic and config.get("no-op"):
            return self._cyclic.copy()

        old = None
        cyclic = self._cyclic

        if not identity and not filter_kwargs:
            return cyclic.copy()

        axis = config.get("axis")
        if axis is None:
            axis = self.domain_axis(*identity, key=True, **filter_kwargs)

        data = self.get_data(None, _fill_value=False)
        if data is not None:
            try:
                data_axes = self.get_data_axes()
                data.cyclic(data_axes.index(axis), iscyclic)
            except ValueError:
                pass

        if iscyclic:
            dim = config.get("coord")
            if dim is None:
                dim = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )

            if dim is not None:
                if config.get("period") is not None:
                    dim.period(**config)
                elif period is not None:
                    dim.period(period, **config)
                elif dim.period() is None:
                    raise ValueError(
                        "A cyclic dimension coordinate must have a period"
                    )

            if axis not in cyclic:
                # Never change _cyclic in-place
                old = cyclic.copy()
                cyclic = cyclic.copy()
                cyclic.add(axis)
                self._cyclic = cyclic

        elif axis in cyclic:
            # Never change _cyclic in-place
            old = cyclic.copy()
            cyclic = cyclic.copy()
            cyclic.discard(axis)
            self._cyclic = cyclic

        if old is None:
            old = cyclic.copy()

        return old

    @_inplace_enabled(default=False)
    def dimension_to_auxiliary(
        self, *identity, inplace=False, **filter_kwargs
    ):
        """Move dimension coordinates to an auxiliary coordinate construct.

        A new auxiliary coordinate construct is derived
        from the selected dimension coordinate construct, and the
        dimension coordinate construct is removed.

        .. versionadded:: 3.14.1

        .. seealso:: `auxiliary_to_dimension`

        :Parameters:

           identity, filter_kwargs: optional
               Select the unique dimension coordinate construct
               returned by ``f.dimension_coordinate(*identity, **filter_kwargs)``.
               See `dimension_coordinate` for details.

            {{inplace: `bool`, optional}}

        :Returns:

            `{{class}}` or `None`
                The {{class}} with the new auxiliary coordinate construct,
                or `None` if the operation was in-place.

        **Examples**

        >>> f = cf.example_field(0)
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> g = f.dimension_to_auxiliary('latitude')
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        Auxiliary coords: latitude(latitude(5)) = [-75.0, ..., 75.0] degrees_north
        >>> h = g.auxiliary_to_dimension('latitude')
        >>> print(h)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> h.equals(f)
        True

        """
        f = _inplace_enabled_define_and_cleanup(self)

        key, dim = f.dimension_coordinate(
            *identity, item=True, **filter_kwargs
        )

        axis = f.get_data_axes(key)
        aux = f._AuxiliaryCoordinate(source=dim)
        f.set_construct(aux, axes=axis)
        f.del_construct(key)
        return f

    @_deprecated_kwarg_check("axes", version="3.0.0", removed_at="4.0.0")
    def direction(self, identity, axes=None, **kwargs):
        """Whether or not a domain axis is increasing.

        An domain axis is considered to be increasing if its dimension
        coordinate values are increasing in index space or if it has
        no dimension coordinate.

        .. seealso:: `directions`

        :Parameters:

            identity: optional
                Select the domain axis construct by one of:

                * An identity or key of a 1-d dimension or auxiliary
                  coordinate construct that whose data spans the
                  domain axis construct.

                  {{construct selection identity}}

                * A domain axis construct identity

                  The domain axis is that which would be selected by
                  passing the given axis description to a call of the
                  construct's `domain_axis` method. For example, for a
                  value of ``'X'``, the domain axis construct returned
                  by ``f.domain_axis('X')`` is selected.

                * `None`. This is the default, which selects the
                   domain construct when there is only one of them.

            axes: deprecated at version 3.0.0
                Use the *identity* parameter instead.

            size: deprecated at version 3.0.0

            kwargs: deprecated at version 3.0.0

        :Returns:

            `bool`
                Whether or not the domain axis is increasing.

        **Examples**

        >>> print(f.dimension_coordinate('X').array)
        array([  0  30  60])
        >>> f.direction('X')
        True
        >>> g = f.flip('X')
        >>> g.direction('X')
        False

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "direction", kwargs
            )  # pragma: no cover

        #        axis = self.domain_axis(identity, key=True, default=None)
        #       if axis is None:
        #            return True

        for coord in self.dimension_coordinates(
            filter_by_axis=(identity,), todict=True
        ).values():
            return coord.direction()

        return True

    def directions(self):
        """Return a dictionary mapping all domain axes to their
        directions.

        .. seealso:: `direction`

        :Returns:

            `dict`
                A dictionary whose key/value pairs are domain axis
                keys and their directions.

        **Examples**

        >>> d.directions()
        {'domainaxis1': True, 'domainaxis1': False}

        """
        out = {key: True for key in self.domain_axes(todict=True)}

        data_axes = self.constructs.data_axes()

        for key, coord in self.dimension_coordinates(todict=True).items():
            axis = data_axes[key][0]
            out[axis] = coord.direction()

        return out

    def get_coordinate_reference(
        self, *identity, key=False, construct=None, default=ValueError()
    ):
        """Return a coordinate reference construct.

        .. versionadded:: 3.0.2

        .. seealso:: `construct`

        :Parameters:

            identity: optional
                Select the coordinate reference construct by one of:

                * The identity of a coordinate reference construct.

                  {{construct selection identity}}

                * The key of a coordinate reference construct

                * `None`. This is the default, which selects the
                  coordinate reference construct when there is only
                  one of them.

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
                Return the value of the *default* parameter if a
                construct can not be found.

                {{default Exception}}

        :Returns:

            `CoordinateReference` or `str`
                The selected coordinate reference construct, or its
                key.

        **Examples**

        """
        if construct is None:
            return self.coordinate_reference(
                *identity, key=key, default=default
            )

        out = []

        c_key = self.construct(construct, key=True, default=None)
        if c_key is None:
            if default is None:
                return
            return self._default(
                default, f"Can't identify construct from {construct!r}"
            )

        for cr_key, ref in tuple(
            self.coordinate_references(todict=True).items()
        ):
            if c_key in [
                ref.coordinates(),
                ref.coordinate_conversion.domain_ancillaries().values(),
            ]:
                if key:
                    if cr_key not in out:
                        out.append(cr_key)
                elif ref not in out:
                    out.append(ref)

                continue

        return out

    def iscyclic(self, *identity, **filter_kwargs):
        """Returns True if the given axis is cyclic.

        {{unique construct}}

        .. versionadded:: 1.0

        .. seealso:: `cyclic`, `period`, `domain_axis`

        :Parameters:

            identity: optional
                Select the unique domain axis construct returned by
                ``f.domain_axis(*identity, **filter_kwargs)``. See
                `domain_axis` for details.

            {{filter_kwargs: optional}}

                .. versionadded:: (cfdm) 3.9.0

        :Returns:

            `bool`
                True if the selected axis is cyclic, otherwise False.

        **Examples**

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

        """
        axis = self.domain_axis(
            *identity, key=True, default=None, **filter_kwargs
        )
        if axis is None:
            raise ValueError("Can't identify unique axis")

        return axis in self.cyclic()

    def match_by_rank(self, *ranks):
        """Whether or not the number of domain axis constructs satisfies
        conditions.

        .. versionadded:: 3.0.0

        .. seealso:: `match`, `match_by_property`,
                     `match_by_identity`, `match_by_ncvar`,
                     `match_by_construct`

        :Parameters:

            ranks: optional
                Define conditions on the number of domain axis
                constructs.

                A condition is one of:

                * `int`
                * a `Query` object

                The condition is satisfied if the number of domain
                axis constructs equals the condition value.

                *Parameter example:*
                  To see if the field construct has 4 domain axis
                  constructs: ``4``

                *Parameter example:*
                  To see if the field construct has at least 3 domain
                  axis constructs: ``cf.ge(3)``

        :Returns:

            `bool`
                Whether or not at least one of the conditions are met.

        **Examples**

        >>> f.match_by_rank(3, 4)

        >>> f.match_by_rank(cf.wi(2, 4))

        >>> f.match_by_rank(1, cf.gt(3))

        """
        if not ranks:
            return True

        n_domain_axes = len(self.domain_axes(todict=True))
        for rank in ranks:
            ok = rank == n_domain_axes
            if ok:
                return True

        return False

    def _parse_axes(self, axes):
        """Convert the given axes to their domain axis identifiers.

        .. versionadded:: 3.9.0

        :Parameters:

            axes:
                One or more axis specifications.

                If *axes* is a sequence then the returned identifiers
                are in the same order.

        :Returns:

            `list`
                The domain axis identifiers.

        """
        if isinstance(axes, str):
            axes = (axes,)
        else:
            try:
                len(axes)
            except TypeError:
                axes = (axes,)

        return [self.domain_axis(x, key=True) for x in axes]

    def replace_construct(
        self, *identity, new=None, copy=True, **filter_kwargs
    ):
        """Replace a metadata construct.

        Replacement assigns the same construct key and, if applicable, the
        domain axes of the original construct to the new, replacing
        construct.

        .. versionadded:: 3.0.0

        .. seealso:: `set_construct`, `construct`

        :Parameters:

            identity: optional
                Select the unique construct returned by
                ``f.construct(*identity, **filter_kwargs)``. See
                `construct` for details.

            new:
               The new construct to replace that selected by the
               *identity* parameter.

            copy: `bool`, optional
                If True then set a copy of the new construct. By default
                the construct is copied.

            {{filter_kwargs: optional}}

                .. versionadded:: 3.9.0

            construct:
                Deprecated at version 3.9.0

        :Returns:

                The construct that was replaced.

        **Examples**

        >>> f.replace_construct('X', new=X_construct)

        """
        key, c = self.construct(*identity, item=True, **filter_kwargs)

        if not isinstance(new, c.__class__):
            raise ValueError(
                f"Can't replace a {c.__class__.__name__} construct "
                f"with a {new.__class__.__name__} object"
            )

        axes = self.get_data_axes(key, default=None)
        if axes is not None:
            shape0 = getattr(c, "shape", None)
            shape1 = getattr(new, "shape", None)
            if shape0 != shape1:
                raise ValueError(
                    f"Can't replace {c.__class__.__name__} construct "
                    f"with a {new.__class__.__name__} object of different "
                    "shape."
                )

        self.set_construct(new, key=key, axes=axes, copy=copy)

        return c

    def set_construct(
        self,
        construct,
        key=None,
        axes=None,
        set_axes=True,
        copy=True,
        autocyclic={},
        conform=True,
    ):
        """Set a metadata construct.

        When inserting a construct with data, the domain axes constructs
        spanned by the data are either inferred, or specified with the
        *axes* parameter.

        For a dimension coordinate construct, an existing dimension
        coordinate construct is discarded if it spans the same domain axis
        construct (since only one dimension coordinate construct can be
        associated with a given domain axis construct).

        .. versionadded:: 3.0.0

        .. seealso:: `constructs`, `creation_commands`, `del_construct`,
                     `get_construct`, `set_coordinate_reference`,
                     `set_data_axes`

        :Parameters:

            construct:
                The metadata construct to be inserted.

            key: `str`, optional
                The construct identifier to be used for the construct. If
                not set then a new, unique identifier is created
                automatically. If the identifier already exists then the
                existing construct will be replaced.

                *Parameter example:*
                  ``key='cellmeasure0'``

            axes: (sequence of) `str` or `int`, optional
                Set the domain axes constructs that are spanned by the
                construct's data. If unset, and the *set_axes* parameter
                is True, then an attempt will be made to assign existing
                domain axis constructs to the data.

                The contents of the *axes* parameter is mapped to domain
                axis constructs by translating each element into a domain
                axis construct key via the `domain_axis` method.

                *Parameter example:*
                  ``axes='domainaxis1'``

                *Parameter example:*
                  ``axes='X'``

                *Parameter example:*
                  ``axes=['latitude']``

                *Parameter example:*
                  ``axes=['X', 'longitude']``

                *Parameter example:*
                  ``axes=[1, 0]``

            set_axes: `bool`, optional
                If False then do not set the domain axes constructs that
                are spanned by the data, even if the *axes* parameter has
                been set. By default the axes are set either according to
                the *axes* parameter, or an attempt will be made to assign
                existing domain axis constructs to the data.

            copy: `bool`, optional
                If True then set a copy of the construct. By default the
                construct is copied.

            autocyclic: `dict`, optional
                Additional parameters for optimising the operation,
                relating to coordinate periodicity and cyclicity. See
                the code for details.

                .. versionadded:: 3.9.0

            conform: `bool`, optional
                If True (the default), then attempt to replace
                placeholder identities in *construct* with existing
                construct identifiers. Specifically, cell method
                construct axis specifiers (such as ``'T'``) are mapped
                to domain axis construct identifiers, and coordinate
                reference construct coordinate specifiers (such as
                ``'latitude'``) are mapped to their corresponding
                dimension or auxiliary coordinate construct
                identifiers.

                .. versionadded:: 3.14.0

        :Returns:

            `str`
                The construct identifier for the construct.

        **Examples**

        >>> key = f.set_construct(c)
        >>> key = f.set_construct(c, copy=False)
        >>> key = f.set_construct(c, axes='domainaxis2')
        >>> key = f.set_construct(c, key='cellmeasure0')

        """
        construct_type = construct.construct_type

        if not set_axes:
            axes = None

        if construct_type in (
            "dimension_coordinate",
            "auxiliary_coordinate",
            "cell_measure",
        ):
            if construct.isscalar:
                # Turn a scalar object into 1-d
                if copy:
                    construct = construct.insert_dimension(0)
                    copy = False
                else:
                    construct.insert_dimension(0, inplace=True)

            if set_axes:
                axes = self._set_construct_parse_axes(
                    construct, axes, allow_scalar=False
                )

            if construct_type == "dimension_coordinate":
                data_axes = self.constructs.data_axes()
                for dim in self.dimension_coordinates(todict=True):
                    if dim == key:
                        continue

                    if data_axes.get(dim) == tuple(axes):
                        self.del_construct(dim, default=None)

        elif construct_type in ("domain_ancillary", "field_ancillary"):
            if set_axes:
                axes = self._set_construct_parse_axes(
                    construct, axes, allow_scalar=True
                )

        out = super().set_construct(construct, key=key, axes=axes, copy=copy)

        if construct_type == "dimension_coordinate":
            construct.autoperiod(inplace=True, config=autocyclic)
            if conform:
                self._conform_coordinate_references(out)

            self.autocyclic(key=out, coord=construct, config=autocyclic)
            if conform:
                try:
                    self._conform_cell_methods()
                except AttributeError:
                    pass

        elif construct_type == "auxiliary_coordinate":
            construct.autoperiod(inplace=True, config=autocyclic)
            if conform:
                self._conform_coordinate_references(out)
                try:
                    self._conform_cell_methods()
                except AttributeError:
                    pass

        elif construct_type == "cell_method":
            if conform:
                self._conform_cell_methods()

        elif construct_type == "coordinate_reference":
            if conform:
                for ckey in self.coordinates(todict=True):
                    self._conform_coordinate_references(
                        ckey, coordref=construct
                    )

        # Return the construct key
        return out

    def del_construct(self, *identity, default=ValueError(), **filter_kwargs):
        """Remove a metadata construct.

        If a domain axis construct is selected for removal then it
        can't be spanned by any data arrays of the field nor metadata
        constructs, nor be referenced by any cell method
        constructs. However, a domain ancillary construct may be
        removed even if it is referenced by coordinate reference
        construct.

        .. versionadded:: 3.16.2

        .. seealso:: `get_construct`, `constructs`, `has_construct`,
                     `set_construct`

        :Parameters:

            identity:
                Select the unique construct that has the identity,
                defined by its `!identities` method, that matches the
                given values.

                Additionally, the values are matched against construct
                identifiers, with or without the ``'key%'`` prefix.

                {{value match}}

                {{displayed identity}}

            default: optional
                Return the value of the *default* parameter if the
                data axes have not been set.

                {{default Exception}}

            {{filter_kwargs: optional}}

                .. versionadded:: 3.16.2

        :Returns:

                The removed metadata construct.

        **Examples**

        >>> f = {{package}}.example_field(0)
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> f.del_construct('time')
        <{{repr}}DimensionCoordinate: time(1) days since 2018-12-01 >
        >>> f.del_construct('time')
        Traceback (most recent call last):
            ...
        ValueError: Can't find unique construct to remove
        >>> f.del_construct('time', default='No time')
        'No time'
        >>> f.del_construct('dimensioncoordinate1')
        <{{repr}}DimensionCoordinate: longitude(8) degrees_east>
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), ncdim%lon(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north

        """
        # Need to re-define to overload this method since cfdm doesn't
        # have the concept of cyclic axes, so have to update the
        # register of cyclic axes when we delete a construct in cf.

        # Get the relevant key first because it will be lost upon deletion
        key = self.construct_key(*identity, default=None, **filter_kwargs)
        cyclic_axes = self._cyclic

        deld_construct = super().del_construct(
            *identity, default=None, **filter_kwargs
        )
        if deld_construct is None:
            if default is None:
                return

            return self._default(
                default, "Can't find unique construct to remove"
            )

        # If the construct deleted was a cyclic axes, remove it from the set
        # of stored cyclic axes, to sync that. This is safe now, since given
        # the block above we can be sure the deletion was successful.
        if key in cyclic_axes:
            # Never change value of _cyclic attribute in-place. Only copy now
            # when the copy is known to be required.
            cyclic_axes = cyclic_axes.copy()
            cyclic_axes.remove(key)
            self._cyclic = cyclic_axes

        return deld_construct

    def set_coordinate_reference(
        self, coordinate_reference, key=None, parent=None, strict=True
    ):
        """Set a coordinate reference construct.

        By default, this is equivalent to using the `set_construct`
        method. If, however, the *parent* parameter has been set to be
        a field or domain construct that contains the new coordinate
        reference construct then copies of its coordinate and domain
        ancillary constructs will be referenced by the inserted
        coordinate reference construct.

        .. versionadded:: 3.0.0

        .. seealso:: `set_construct`

        :Parameters:

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct to be inserted.

            key: `str`, optional
                The construct identifier to be used for the
                construct. If not set then a new, unique identifier is
                created automatically. If the identifier already
                exists then the existing construct will be replaced.

                *Parameter example:*
                  ``key='coordinatereference1'``

            parent: `Field` or `Domain`, optional
                A field or domain construct that contains the new
                coordinate reference construct.

            strict: `bool`, optional
                If False then allow non-strict identities for
                identifying coordinate and domain ancillary metadata
                constructs.

        :Returns:

            `str`
                The construct identifier for the coordinate reference
                construct.

        """
        if parent is None:
            return self.set_construct(coordinate_reference, key=key, copy=True)

        # Still here?
        ref = coordinate_reference.copy()

        coordinates = parent.coordinates(todict=True)
        domain_ancillaries = parent.domain_ancillaries(todict=True)

        ckeys = []
        for value in coordinate_reference.coordinates():
            if value in coordinates:
                identity = coordinates[value].identity(strict=strict)
                ckeys.append(self.coordinate(identity, key=True, default=None))

        ref.clear_coordinates()
        ref.set_coordinates(ckeys)

        coordinate_conversion = coordinate_reference.coordinate_conversion

        dakeys = {}
        for term, value in coordinate_conversion.domain_ancillaries().items():
            if value in domain_ancillaries:
                identity = domain_ancillaries[value].identity(strict=strict)
                dakeys[term] = self.domain_ancillary(
                    identity, key=True, default=None
                )
            else:
                dakeys[term] = None

        ref.coordinate_conversion.clear_domain_ancillaries()
        ref.coordinate_conversion.set_domain_ancillaries(dakeys)

        return self.set_construct(ref, key=key, copy=False)

    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    def aux(
        self,
        *identity,
        key=False,
        default=ValueError(),
        item=False,
        **filter_kwargs,
    ):
        """Alias for `auxiliary_coordinate`."""
        return self.auxiliary_coordinate(
            *identity, key=key, default=default, item=item, **filter_kwargs
        )

    def auxs(self, *identities, **filter_kwargs):
        """Alias for `coordinates`."""
        return self.auxiliary_coordinates(*identities, **filter_kwargs)

    def axes(self, *identities, **filter_kwargs):
        """Alias for `domain_axes`."""
        return self.domain_axes(*identities, **filter_kwargs)

    def axis(
        self,
        *identity,
        key=False,
        default=ValueError(),
        item=False,
        **filter_kwargs,
    ):
        """Alias for `domain_axis`."""
        return self.domain_axis(
            *identity, key=key, default=default, item=item, **filter_kwargs
        )

    def coord(
        self,
        *identity,
        key=False,
        default=ValueError(),
        item=False,
        **filter_kwargs,
    ):
        """Alias for `coordinate`."""
        return self.coordinate(
            *identity, key=key, default=default, item=item, **filter_kwargs
        )

    def coords(self, *identities, **filter_kwargs):
        """Alias for `coordinates`."""
        return self.coordinates(*identities, **filter_kwargs)

    def dim(
        self,
        *identity,
        key=False,
        default=ValueError(),
        item=False,
        **filter_kwargs,
    ):
        """Alias for `dimension_coordinate`."""
        return self.dimension_coordinate(
            *identity, key=key, default=default, item=item, **filter_kwargs
        )

    def dims(self, *identities, **filter_kwargs):
        """Alias for `dimension_coordinates`."""
        return self.dimension_coordinates(*identities, **filter_kwargs)

    def domain_anc(
        self,
        *identity,
        key=False,
        default=ValueError(),
        item=False,
        **filter_kwargs,
    ):
        """Alias for `domain_ancillary`."""
        return self.domain_ancillary(
            *identity, key=key, default=default, item=item, **filter_kwargs
        )

    def domain_ancs(self, *identities, **filter_kwargs):
        """Alias for `domain_ancillaries`."""
        return self.domain_ancillaries(*identities, **filter_kwargs)

    def key(self, *identity, default=ValueError(), **filter_kwargs):
        """Alias for `construct_key`."""
        return self.construct(
            *identity, default=default, key=True, **filter_kwargs
        )

    def measure(
        self,
        *identity,
        key=False,
        default=ValueError(),
        item=False,
        **filter_kwargs,
    ):
        """Alias for `cell_measure`."""
        return self.cell_measure(
            *identity, key=key, default=default, item=item, **filter_kwargs
        )

    def measures(self, *identities, **filter_kwargs):
        """Alias for `cell_measures`."""
        return self.cell_measures(*identities, **filter_kwargs)

    def ref(
        self,
        *identity,
        default=ValueError(),
        key=False,
        item=False,
        **filter_kwargs,
    ):
        """Alias for `coordinate_reference`."""
        return self.coordinate_reference(
            *identity, key=key, default=default, item=item, **filter_kwargs
        )

    def refs(self, *identities, **filter_kwargs):
        """Alias for `coordinate_references`."""
        return self.coordinate_references(*identities, **filter_kwargs)


def _create_ancillary_mask_component(mask_shape, ind, compress):
    """Create an ancillary mask component.

    .. versionadded:: 3.9.0

    .. seealso:: `_indices`

    :Parameters:

        mask_shape: `tuple`
            The shape of the mask component to be created.

              *Parameter example*
                ``mask_shape=(3,)``

              *Parameter example*
                ``mask_shape=(9, 10)``

        ind: sequence of `list`
            Integer indices with the same shape as *mask_shape*,
            previously created by a single argument call of
            ``np[.ma].where``, that define where the returned mask is
            False.

        compress: `bool`
            If True then remove whole slices which only contain masked
            points.

    :Returns:

        `Data`
            The mask array.

    **Examples**

    >>> f = cf.{{class}}()
    >>> d = _create_ancillary_mask_component((4,), ([0, 3, 1],))
    >>> print(d.array)
    [False False  True False]
    >>> d = f._create_ancillary_mask_component(
    ...     (4, 6), ([0, 3, 1], [5, 3, 2])
    ... )
    >>> print(d.array)
    [[ True  True  True  True  True False]
     [ True  True False  True  True  True]
     [ True  True  True  True  True  True]
     [ True  True  True False  True  True]]

    """
    mask = np.ones(mask_shape, dtype=bool)
    mask[tuple(ind)] = False

    # For compressed indices, remove slices which only contain masked
    # points.
    if compress:
        for i, (index, n) in enumerate(zip(ind, mask_shape)):
            index = np.unique(index)
            if index.size == n:
                continue

            mask = mask.take(index, axis=i)

    return Data(mask)
