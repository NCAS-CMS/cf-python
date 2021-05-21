import logging

from numbers import Integral

import numpy as np

try:
    from matplotlib.path import Path
except ImportError:
    pass

from ..query import Query
from ..data import Data
from ..units import Units

from ..functions import (
    parse_indices,
    bounds_combination_mode,
    _DEPRECATION_ERROR_KWARGS,
)

from ..decorators import (
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
    _manage_log_level_via_verbosity,
    _deprecated_kwarg_check,
)

logger = logging.getLogger(__name__)


_units_degrees = Units("degrees")


class FieldDomain:
    """Mixin class for methods common to both field and domain
    constructs.

    .. versionadded:: 3.9.0

    """

    def _coordinate_reference_axes(self, key):
        """Returns the set of coordinate reference axes for a key.

        :Parameters:

            key: `str`
                Coordinate reference construct key.

        :Returns:

            `set`

        **Examples:**

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

        **Examples:**

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

    def _construct(
        self,
        _method,
        _constructs_method,
        identities,
        key=False,
        item=False,
        default=ValueError(),
        **filter_kwargs,
    ):
        """An interface to `Constructs.filter`.

        {{unique construct}}

        .. versionadded:: 3.9.0

        :Parameters:

            _method: `str`
                The name of the calling method.

            _constructs_method: `str`
                The name of the corresponding method that can return
                any number of constructs.

            identities: sequence
                As for the *identities* parameter of the calling
                method.

            {{key: `bool`, optional}}

            {{item: `bool`, optional}}

            default: optional
                Return the value of the *default* parameter if there
                is no unique construct.

                {{default Exception}}

            {{filter_kwargs: optional}}

        :Returns:

                {{Returns construct}}

        """
        cached = filter_kwargs.get("cached")
        if cached is not None:
            return cached

        filter_kwargs["todict"] = True

        c = getattr(self, _constructs_method)(*identities, **filter_kwargs)

        # Return construct, or key, or both, or default
        n = len(c)
        if n == 1:
            k, construct = c.popitem()
            if key:
                return k

            if item:
                return k, construct

            return construct

        if default is None:
            return default

        return self._default(
            default,
            f"{self.__class__.__name__}.{_method}() can't return {n} "
            "constructs",
        )

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

    def _indices(self, mode, data_axes, auxiliary_mask, **kwargs):
        """Create indices that define a subspace of the field or domain
        construct.

        This method is intended to be called by the `indices` method.

        See the `indices` method for more details.

        .. versionadded:: 3.9.0

        :Parameters:

            mode: `str`
                The mode of operation. See the *mode* parameter of
                `indices` for details.

            data_axes: sequence of `str`, or `None`
                The domain axis identifiers of the data axes, or
                `None` if there is no data array.

            auxiliary_mask: `bool`
                Whether or not to create an auxiliary mask. See
                `indices` for details.

            kwargs: *optional*
                See the **kwargs** parameters of `indices` for
                details.

        :Returns:

            `dict`
                 The dictionary has two keys: ``'indices'`` and
                 ``'mask'``.

                 The ``'indices'`` key stores a dictionary keyed by
                 domain axis identifiers, each of which has a value of
                 the index for that domain axis.

                 The ``'mask'`` key stores a dictionary in keyed by
                 tuples of domain axis identifier combinations, each
                 of which has of a `Data` object containing the
                 auxiliary mask to apply to those domain axes at time
                 of the indices being used to create a subspace. This
                 dictionary will always be empty if *auxiliary_mask*
                 is False.

        """
        compress = mode == "compress"
        envelope = mode == "envelope"
        full = mode == "full"

        #        logger.debug(
        #            f"{self.__class__.__name__}._indices:\n"
        #            f"  mode         = {mode!r}\n"
        #            f"  input kwargs = {kwargs!r}"
        #        )  # pragma: no cover

        domain_axes = self.domain_axes(todict=True)
        #        constructs = self.constructs.filter_by_data()

        # Initialize indices
        indices = {axis: slice(None) for axis in domain_axes}

        parsed = {}
        unique_axes = set()
        n_axes = 0

        for identity, value in kwargs.items():
            key, construct = self.construct(
                identity,
                filter_by_data=True,
                item=True,
                default=(None, None),
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
                parsed[axes].append((axes, key, construct, value))
            else:
                new_key = True
                y = set(axes)
                for x in parsed:
                    if set(x) == set(y):
                        # The axes are the same but in a different
                        # order, so we don't need a new key.
                        parsed[x].append((axes, key, construct, value))
                        new_key = False
                        break

                if new_key:
                    # The axes, taken in any order, are not the same
                    # as any keys, so create an new key.
                    n_axes += len(axes)
                    parsed[axes] = [(axes, key, construct, value)]

            unique_axes.update(axes)

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

        auxiliary_mask = {}

        for canonical_axes, axes_key_construct_value in parsed.items():
            axes, keys, constructs, points = list(
                zip(*axes_key_construct_value)
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
                    f"{n_axes} {a}: {points}"
                )

            create_mask = False

            item_axes = axes[0]

            logger.debug(
                f"  item_axes    = {item_axes!r}\n  keys         = {keys!r}"
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
                    f"  {n_items} 1-d constructs: {constructs!r}\n"
                    f"  axis         = {axis!r}\n"
                    f"  value        = {value!r}"
                )  # pragma: no cover

                if isinstance(value, (list, slice, tuple, np.ndarray)):
                    # ------------------------------------------------
                    # 1-dimensional CASE 1: Value is already an index,
                    #                       e.g. [0], [7,4,2],
                    #                       slice(0,4,2),
                    #                       numpy.array([2,4,7]),
                    #                       [True, False, True]
                    # ------------------------------------------------
                    logger.debug("  1-d CASE 1:")  # pragma: no cover

                    index = value

                    if envelope or full:
                        size = domain_axes[axis].get_size()
                        # TODODASK - consider using dask.arange here
                        d = np.arange(size)  # self._Data(range(size))
                        ind = (d[value],)  # .array,)
                        index = slice(None)

                elif (
                    item is not None
                    and isinstance(value, Query)
                    and value.operator in ("wi", "wo")
                    and item.construct_type == "dimension_coordinate"
                    and self.iscyclic(axis)
                ):
                    # ------------------------------------------------
                    # 1-dimensional CASE 2: Axis is cyclic and
                    #                       subspace criterion is a
                    #                       'within' or 'without'
                    #                       Query instance
                    # ------------------------------------------------
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

                    index = slice(start, stop, 1)

                    if full:
                        # TODODASK - consider using some sort of
                        #            dask.arange here
                        d = self._Data(list(range(size)))
                        d.cyclic(0)
                        ind = (d[index].array,)
                        index = slice(None)

                elif item is not None:
                    # ------------------------------------------------
                    # 1-dimensional CASE 3: All other 1-d cases
                    # ------------------------------------------------
                    logger.debug("  1-d CASE 3:")  # pragma: no cover

                    item_match = value == item

                    if not item_match.any():
                        raise ValueError(
                            f"No {identity!r} axis indices found "
                            f"from: {value}"
                        )

                    index = np.asanyarray(item_match)

                    if envelope or full:
                        if np.ma.isMA(index):
                            ind = np.ma.where(index)
                        else:
                            ind = np.where(index)

                        index = slice(None)

                else:
                    raise ValueError(
                        "Must specify a domain axis construct or a "
                        "construct with data for which to create indices"
                    )

                logger.debug(
                    f"  index        = {index}\n" f"  ind          = {ind}"
                )  # pragma: no cover

                # Put the index into the correct place in the list of
                # indices.
                #
                # Note that we might overwrite it later if there's an
                # auxiliary mask for this axis.
                indices[axis] = index

            else:
                # ----------------------------------------------------
                # N-dimensional constructs
                # ----------------------------------------------------
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

                logger.debug(
                    f"  transposed N-d constructs: {transposed_constructs!r}"
                )  # pragma: no cover

                item_matches = [
                    (value == construct).data
                    for value, construct in zip(points, transposed_constructs)
                ]

                item_match = item_matches.pop()

                for m in item_matches:
                    item_match &= m

                item_match = item_match.array  # LAMA alert

                if np.ma.isMA:
                    ind = np.ma.where(item_match)
                else:
                    ind = np.where(item_match)

                logger.debug(
                    f"  item_match  = {item_match}\n" f"  ind         = {ind}"
                )  # pragma: no cover

                for i in ind:
                    if not i.size:
                        raise ValueError(
                            f"No {canonical_axes!r} axis indices found "
                            f"from: {value!r}"
                        )

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
                            if v.operator == "contains":
                                contains = True
                                v = v.value
                            elif v.operator == "eq":
                                v = v.value
                            else:
                                contains = False
                                break

                        v = self._Data.asdata(v)
                        if v.Units:
                            v.Units = construct.Units

                        points2.append(v.datum())

                if contains:
                    # The coordinates have bounds and the condition is
                    # a 'contains' Query object. Check each
                    # potentially matching cell for actually including
                    # the point.
                    try:
                        Path
                    except NameError:
                        raise ImportError(
                            "Need to install matplotlib to create indices "
                            f"based on {transposed_constructs[0].ndim}-d "
                            "constructs and a 'contains' Query object"
                        )

                    if n_items != 2:
                        raise ValueError(
                            f"Can't index for cell from {n_axes}-d "
                            "coordinate objects"
                        )

                    if 0 < len(bounds) < n_items:
                        raise ValueError("bounds alskdaskds TODO")

                    # Remove grid cells if, upon closer inspection,
                    # they do actually contain the point.
                    delete = [
                        n
                        for n, vertices in enumerate(zip(*zip(*bounds)))
                        if not Path(zip(*vertices)).contains_point(points2)
                    ]

                    if delete:
                        ind = [np.delete(ind_1d, delete) for ind_1d in ind]

            if ind is not None:
                mask_shape = []
                masked_subspace_size = 1
                ind = np.array(ind)

                for i, (axis, start, stop) in enumerate(
                    zip(canonical_axes, ind.min(axis=1), ind.max(axis=1))
                ):
                    if data_axes and axis not in data_axes:
                        continue

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
                            stop = domain_axes[axis].get_size()
                            size = stop - start
                            index = slice(start, stop)
                        else:
                            raise ValueError(
                                "Must have full, envelope or compress"
                            )  # pragma: no cover

                        indices[axis] = index

                    mask_shape.append(size)
                    masked_subspace_size *= size
                    ind[i] -= start

                create_mask = data_axes and ind.shape[1] < masked_subspace_size
            else:
                create_mask = False

            # TODODASK - if we have 2 list of integers then we need to
            #            apply different auxiliary masks (if any)
            #            after different __getitems__. SCRUB THAT! if
            #            we have an auxiliary mask, then by definition
            #            we do _not_ have a list(s) of integers

            # --------------------------------------------------------
            # Create an auxiliary mask for these axes
            # --------------------------------------------------------
            logger.debug(f"  create_mask  = {create_mask}")  # pragma: no cover

            if create_mask:
                mask = _create_auxiliary_mask_component(
                    mask_shape, ind, compress
                )
                auxiliary_mask[canonical_axes] = mask
                logger.debug(
                    f"  mask_shape   = {mask_shape}\n"
                    f"  mask.shape   = {mask.shape}"
                )  # pragma: no cover

        for axis, index in tuple(indices.items()):
            indices[axis] = parse_indices(
                (domain_axes[axis].get_size(),), (index,)
            )[0]

        # Include the auxiliary mask
        indices = {
            "indices": indices,
            "mask": auxiliary_mask,
        }

        logger.debug(f"  indices      = {indices!r}")  # pragma: no cover

        # Return the indices and the auxiliary mask
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

        **Examples:**

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

            # TODODASK - remove these two lines when multiaxis rolls
            #            are allowed at v4.0.0
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

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    @_deprecated_kwarg_check("i")
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

        **Examples:**

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
                Additional parameters for optimizing the
                operation. See the code for details.

                .. versionadded:: 3.9.0

        :Returns:

           `bool`

        """
        noop = config.get("no-op")

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
            has_period = False

        if period is None:
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

        if bounds_range != period:
            if not noop:
                self.cyclic(key, iscyclic=False, config=config)
            return False

        if has_period:
            period = None

        config = config.copy()
        config["axis"] = self.get_data_axes(key, default=(None,))[0]

        self.cyclic(key, iscyclic=True, period=period, config=config)

        return True

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

                **Examples:**

                >>> f.del_coordinate_reference('rotated_latitude_longitude')
                <CF CoordinateReference: rotated_latitude_longitude>

        """
        if construct is None:
            if identity is None:
                raise ValueError("TODO")

            key = self.coordinate_reference(identity, key=True, default=None)
            if key is None:
                if default is None:
                    return

                return self._default(
                    default,
                    f"Can't identify construct from {identity!r}",
                )

            ref = self.del_construct(key)

            for (
                da_key
            ) in ref.coordinate_conversion.domain_ancillaries().values():
                self.del_construct(da_key, default=None)

            return ref
        elif identity is not None:
            raise ValueError("TODO")

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

        **Examples:**

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

    def auxiliary_coordinate(
        self,
        *identity,
        default=ValueError(),
        key=False,
        item=False,
        **filter_kwargs,
    ):
        """Select an auxiliary coordinate construct.

        {{unique construct}}

        .. versionadded:: 3.0.0

        .. seealso:: `construct`, `auxiliary_coordinates`

        :Parameters:

            identity: optional
                Select auxiliary coordinate constructs that have an
                identity, defined by their `!identities` methods, that
                matches any of the given values.

                Additionally, the values are matched against construct
                identifiers, with or without the ``'key%'`` prefix.

                If no values are provided then all auxiliary
                coordinate constructs are selected.

                {{value match}}

                {{displayed identity}}

            {{key: `bool`, optional}}

            {{item: `bool`, optional}}

                .. versionadded:: (cfdm) 3.9.0

            default: optional
                Return the value of the *default* parameter if there
                is no unique construct.

                {{default Exception}}

            {{filter_kwargs: optional}}

                .. versionadded:: (cfdm) 3.9.0

        :Returns:

                {{Returns construct}}

        **Examples:**

        """
        return self._construct(
            "auxiliary_coordinate",
            "auxiliary_coordinates",
            identity,
            key=key,
            item=item,
            default=default,
            **filter_kwargs,
        )

    def construct(
        self,
        *identity,
        default=ValueError(),
        key=False,
        item=False,
        **filter_kwargs,
    ):
        """Select a metadata construct by its identity.

        .. seealso:: `del_construct`, `get_construct`, `has_construct`,
                     `set_construct`

        :Parameters:

            identity: optional
                Select constructs that have an identity, defined by
                their `!identities` methods, that matches any of the
                given values.

                Additionally, the values are matched against construct
                identifiers, with or without the ``'key%'`` prefix.

                If no values are provided then all constructs are
                selected.

                {{value match}}

                {{displayed identity}}

            {{key: `bool`, optional}}

            {{item: `bool`, optional}}

                .. versionadded:: (cfdm) 3.9.0

            default: optional
                Return the value of the *default* parameter if there
                is no unique construct.

                {{default Exception}}

            {{filter_kwargs: optional}}

                .. versionadded:: (cfdm) 3.9.0

        :Returns:

                {{Returns construct}}

        **Examples:**

        """
        return self._construct(
            "construct",
            "constructs",
            identity,
            key=key,
            item=item,
            default=default,
            **filter_kwargs,
        )

    def cell_measure(
        self,
        *identity,
        default=ValueError(),
        key=False,
        item=False,
        **filter_kwargs,
    ):
        """Select a cell measure construct.

        {{unique construct}}

        .. versionadded:: 3.0.0

        .. seealso:: `construct`, `cell_measures`

        :Parameters:

            identity: optional
                Select dimension coordinate constructs that have an
                identity, defined by their `!identities` methods, that
                matches any of the given values.

                Additionally, the values are matched against construct
                identifiers, with or without the ``'key%'`` prefix.

                If no values are provided then all dimension
                coordinate constructs are selected.

                {{value match}}

                {{displayed identity}}

            {{key: `bool`, optional}}

            {{item: `bool`, optional}}

                .. versionadded:: (cfdm) 3.9.0

            default: optional
                Return the value of the *default* parameter if there
                is no unique construct.

                {{default Exception}}

            {{filter_kwargs: optional}}

                .. versionadded:: (cfdm) 3.9.0

        :Returns:

                {{Returns construct}}

        **Examples:**

        """
        return self._construct(
            "cell_measure",
            "cell_measures",
            identity,
            key=key,
            item=item,
            default=default,
            **filter_kwargs,
        )

    def coordinate(
        self,
        *identity,
        default=ValueError(),
        key=False,
        item=False,
        **filter_kwargs,
    ):
        """Select a dimension or auxiliary coordinate construct.

        {{unique construct}}

        .. versionadded:: 3.0.0

        .. seealso:: `construct`, `coordinates`

        :Parameters:

            identity: optional
                Select dimension or auxiliary coordinate constructs
                that have an identity, defined by their `!identities`
                methods, that matches any of the given values.

                Additionally, the values are matched against construct
                identifiers, with or without the ``'key%'`` prefix.

                If no values are provided then all dimension or
                auxiliary coordinate constructs are selected.

                {{value match}}

                {{displayed identity}}

            {{key: `bool`, optional}}

            {{item: `bool`, optional}}

                .. versionadded:: (cfdm) 3.9.0

            default: optional
                Return the value of the *default* parameter if there
                is no unique construct.

                {{default Exception}}

            {{filter_kwargs: optional}}

                .. versionadded:: (cfdm) 3.9.0

        :Returns:

                {{Returns construct}}

        **Examples:**

        """
        return self._construct(
            "coordinate",
            "coordinates",
            identity,
            key=key,
            item=item,
            default=default,
            **filter_kwargs,
        )

    def coordinate_reference(
        self,
        *identity,
        default=ValueError(),
        key=False,
        item=False,
        **filter_kwargs,
    ):
        """Return a coordinate reference construct, or its key.

        .. versionadded:: 3.0.0

        .. seealso:: `construct`, `auxiliary_coordinate`, `cell_measure`,
                     `cell_method`, `coordinate`, `coordinate_references`,
                     `dimension_coordinate`, `domain_ancillary`,
                     `domain_axis`, `field_ancillary`

        :Parameters:

            identities: optional
                Select coordinate reference constructs that have an
                identity, defined by their `!identities` methods, that
                matches any of the given values.

                Additionally, the values are matched against construct
                identifiers, with or without the ``'key%'`` prefix.

                If no identities are provided then all coordinate
                reference constructs are selected.

                {{value match}}

                {{displayed identity}}

            {{key: `bool`, optional}}

            {{item: `bool`, optional}}

                .. versionadded:: (cfdm) 3.9.0

            default: optional
                Return the value of the *default* parameter if there
                is no unique construct.

                {{default Exception}}

            {{filter_kwargs: optional}}

                .. versionadded:: (cfdm) 3.9.0

        :Returns:

                {{Returns construct}}

        **Examples:**

        """
        return self._construct(
            "coordinate_reference",
            "coordinate_references",
            identity,
            key=key,
            item=item,
            default=default,
            **filter_kwargs,
        )

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

                **Examples:**

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

    def dimension_coordinate(
        self,
        *identity,
        key=False,
        default=ValueError(),
        item=False,
        **filter_kwargs,
    ):
        """Select a dimension coordinate construct.

        {{unique construct}}

        .. versionadded:: 3.0.0

        .. seealso:: `construct`, `dimension_coordinates`

        :Parameters:

            identity: optional
                Select dimension coordinate constructs that have an
                identity, defined by their `!identities` methods, that
                matches any of the given values.

                Additionally, the values are matched against construct
                identifiers, with or without the ``'key%'`` prefix.

                If no values are provided then all dimension
                coordinate constructs are selected.

                {{value match}}

                {{displayed identity}}

            {{key: `bool`, optional}}

            {{item: `bool`, optional}}

                .. versionadded:: (cfdm) 3.9.0

            default: optional
                Return the value of the *default* parameter if there
                is no unique construct.

                {{default Exception}}

            {{filter_kwargs: optional}}

                .. versionadded:: (cfdm) 3.9.0

        :Returns:

                {{Returns construct}}

        **Examples:**

        """
        return self._construct(
            "dimension_coordinate",
            "dimension_coordinates",
            identity,
            key=key,
            item=item,
            default=default,
            **filter_kwargs,
        )

    @_deprecated_kwarg_check("axes")
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

        **Examples:**

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

        **Examples:**

        >>> d.directions()
        {'domainaxis1': True, 'domainaxis1': False}

        """
        out = {key: True for key in self.domain_axes(todict=True)}

        data_axes = self.constructs.data_axes()

        for key, coord in self.dimension_coordinates(todict=True).items():
            axis = data_axes[key][0]
            out[axis] = coord.direction()

        return out

    def domain_ancillary(
        self,
        *identity,
        default=ValueError(),
        key=False,
        item=False,
        **filter_kwargs,
    ):
        """Select a domain ancillary construct.

        {{unique construct}}

        .. versionadded:: 3.0.0

        .. seealso:: `construct`, `domain_ancillaries`

        :Parameters:

            identity: optional
                Select domain ancillary constructs that have an
                identity, defined by their `!identities` methods, that
                matches any of the given values.

                Additionally, the values are matched against construct
                identifiers, with or without the ``'key%'`` prefix.

                If no values are provided then all domain ancillary
                constructs are selected.

                {{value match}}

                {{displayed identity}}

            {{key: `bool`, optional}}

            {{item: `bool`, optional}}

                .. versionadded:: (cfdm) 3.9.0

            default: optional
                Return the value of the *default* parameter if there
                is no unique construct.

                {{default Exception}}

            {{filter_kwargs: optional}}

                .. versionadded:: (cfdm) 3.9.0

        :Returns:

                {{Returns construct}}

        **Examples:**

        """
        return self._construct(
            "domain_ancillary",
            "domain_ancillaries",
            identity,
            key=key,
            item=item,
            default=default,
            **filter_kwargs,
        )

    def domain_axis(
        self,
        *identity,
        key=False,
        default=ValueError(),
        item=False,
        **filter_kwargs,
    ):
        """Select a domain axis construct.

        {{unique construct}}

        .. versionadded:: 3.0.0

        .. seealso:: `construct`, `domain_axes`

        :Parameters:

            identities: `tuple`, optional
                Select domain axis constructs that have an identity,
                defined by their `!identities` methods, that matches
                any of the given values.

                Additionally, the values are matched against construct
                identifiers, with or without the ``'key%'`` prefix.

                Additionally, if for a given `value``,
                ``f.coordinates(value, filter_by_naxes=(1,))`` returns
                1-d coordinate constructs that all span the same
                domain axis construct then that domain axis construct
                is selected. See `coordinates` for details.

                Additionally, if there is a `Field` data array and a
                value matches the integer position of an array
                dimension, then the corresponding domain axis
                construct is selected.

                If no values are provided then all domain axis
                constructs are selected.

                {{value match}}

                {{displayed identity}}

            {{key: `bool`, optional}}

            {{item: `bool`, optional}}

                .. versionadded:: (cfdm) 3.9.0

            default: optional
                Return the value of the *default* parameter if there
                is no unique construct.

                {{default Exception}}

            {{filter_kwargs: optional}}

                .. versionadded:: (cfdm) 3.9.0

        :Returns:

                {{Returns construct}}


        **Examples:**

        """
        return self._construct(
            "domain_axis",
            "domain_axes",
            identity,
            key=key,
            item=item,
            default=default,
            **filter_kwargs,
        )

    def get_coordinate_reference(
        self, identity=None, key=False, construct=None, default=ValueError()
    ):
        """TODO.

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

        **Examples:**

        """
        if construct is None:
            return self.coordinate_reference(
                identity=identity, key=key, default=default
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

        **Examples:**

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

        **Examples:**

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

        **Examples:**

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
                raise ValueError("TODO bb")

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
                Additional parameters for optimizing the operation,
                relating to coordinate periodicity and cyclicity. See
                the code for details.

                .. versionadded:: 3.9.0

        :Returns:

            `str`
                The construct identifier for the construct.

        **Examples:**

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
            self._conform_coordinate_references(out)
            self.autocyclic(key=out, coord=construct, config=autocyclic)
            try:
                self._conform_cell_methods()
            except AttributeError:
                pass

        elif construct_type == "auxiliary_coordinate":
            construct.autoperiod(inplace=True, config=autocyclic)
            self._conform_coordinate_references(out)
            try:
                self._conform_cell_methods()
            except AttributeError:
                pass

        elif construct_type == "cell_method":
            self._conform_cell_methods()

        elif construct_type == "coordinate_reference":
            for ckey in self.coordinates(todict=True):
                self._conform_coordinate_references(ckey, coordref=construct)

        # Return the construct key
        return out

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
            *identity,
            key=key,
            default=default,
            item=item,
            **filter_kwargs,
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
            *identity,
            key=key,
            default=default,
            item=item,
            **filter_kwargs,
        )

    def refs(self, *identities, **filter_kwargs):
        """Alias for `coordinate_references`."""
        return self.coordinate_references(*identities, **filter_kwargs)


def _create_auxiliary_mask_component(mask_shape, ind, compress):
    """Create an auxiliary mask component.

    .. versionadded:: 3.9.0

    :Parameters:

        mask_shape: `tuple`
            The shape of the mask component to be created.

              *Parameter example*
                ``mask_shape=(3,)``

              *Parameter example*
                ``mask_shape=(9, 10)``

        ind: sequnce of `list`
            As returned by a single argument call of
            ``np[.ma].where(....)``.

        compress: `bool`
            If True then remove whole slices which only contain masked
            points.

    :Returns:

        `Data`
            The mask array.

    **Examples:**

    >>> f = cf.{{class}}()
    >>> d = _create_auxiliary_mask_component(
    ...     (4,), ([0, 3, 1],)
    ... )
    >>> print(d.array)
    [False False  True False]
    >>> d = f._create_auxiliary_mask_component(
    ...     (4, 6), ([0, 3, 1], [5, 3, 2])
    ... )
    >>> print(d.array)
    [[ True  True  True  True  True False]
     [ True  True False  True  True  True]
     [ True  True  True  True  True  True]
     [ True  True  True False  True  True]]

    """
    # Note that, for now, auxiliary_mask has to be numpy array (rather
    # than a cf.Data object) because we're going to index it with
    # fancy indexing which a cf.Data object might not support - namely
    # a non-monotonic list of integers.
    auxiliary_mask = np.ones(mask_shape, dtype=bool)

    auxiliary_mask[tuple(ind)] = False

    # For compressed indices, remove slices which only contain masked
    # points.
    if compress:
        for i, (index, n) in enumerate(zip(ind, mask_shape)):
            index = set(index)
            if len(index) == n:
                continue

            auxiliary_mask = auxiliary_mask.take(sorted(index), axis=i)

    return Data(auxiliary_mask)
