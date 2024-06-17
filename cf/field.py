import logging
from collections import namedtuple
from functools import reduce
from operator import mul as operator_mul
from os import sep

import cfdm
import numpy as np
from cfdm import is_log_level_debug, is_log_level_detail, is_log_level_info

from . import (
    AuxiliaryCoordinate,
    Bounds,
    CellMeasure,
    CellMethod,
    Constructs,
    Count,
    DimensionCoordinate,
    Domain,
    DomainAncillary,
    DomainAxis,
    FieldList,
    Flags,
    Index,
    List,
    mixin,
)
from .constants import masked as cf_masked
from .data import Data
from .data.array import (
    GatheredArray,
    RaggedContiguousArray,
    RaggedIndexedArray,
    RaggedIndexedContiguousArray,
)
from .decorators import (
    _deprecated_kwarg_check,
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
    _manage_log_level_via_verbosity,
)
from .formula_terms import FormulaTerms
from .functions import (
    _DEPRECATION_ERROR,
    _DEPRECATION_ERROR_ARG,
    _DEPRECATION_ERROR_ATTRIBUTE,
    _DEPRECATION_ERROR_KWARG_VALUE,
    _DEPRECATION_ERROR_KWARGS,
    _DEPRECATION_ERROR_METHOD,
    DeprecationError,
    _section,
    abspath,
    flat,
    parse_indices,
)
from .functions import relaxed_identities as cf_relaxed_identities
from .functions import size as cf_size
from .query import Query, eq, ge, gt, le, lt
from .subspacefield import SubspaceField
from .timeduration import TimeDuration
from .units import Units

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Commonly used units
# --------------------------------------------------------------------
# _units_degrees = Units("degrees")
_units_radians = Units("radians")
_units_metres = Units("m")
_units_1 = Units("1")

# --------------------------------------------------------------------
# Map each allowed input collapse method name to its corresponding
# Data method. Input collapse methods not in this sictionary are
# assumed to have a corresponding Data method with the same name.
# --------------------------------------------------------------------
_collapse_methods = {
    **{
        name: name
        for name in [
            "mean",  # results in 'mean': 'mean' entry, etc.
            "mean_absolute_value",
            "mean_of_upper_decile",
            "max",
            "maximum_absolute_value",
            "min",
            "max",
            "minimum_absolute_value",
            "mid_range",
            "range",
            "median",
            "sd",
            "sum",
            "sum_of_squares",
            "integral",
            "root_mean_square",
            "var",
            "sample_size",
            "sum_of_weights",
            "sum_of_weights2",
        ]
    },
    **{  # non-identical mapped names:
        "avg": "mean",
        "average": "mean",
        "maximum": "max",
        "minimum": "min",
        "standard_deviation": "sd",
        "variance": "var",
    },
}

# --------------------------------------------------------------------
# Map each allowed input collapse method name to its corresponding CF
# cell method.
# --------------------------------------------------------------------
_collapse_cell_methods = {
    **{
        name: name
        for name in [
            "point",
            "mean",
            "mean_absolute_value",
            "mean_of_upper_decile",
            "maximum",
            "maximum_absolute_value",
            "minimum",
            "minimum_absolute_value",
            "mid_range",
            "range",
            "median",
            "standard_deviation",
            "sum",
            "root_mean_square",
            "sum_of_squares",
            "variance",
        ]
    },
    **{  # non-identical mapped names:
        "avg": "mean",
        "average": "mean",
        "max": "maximum",
        "min": "minimum",
        "sd": "standard_deviation",
        "integral": "sum",
        "var": "variance",
        "sample_size": "point",
        "sum_of_weights": "sum",
        "sum_of_weights2": "sum",
    },
}

# --------------------------------------------------------------------
# These Data methods may be weighted
# --------------------------------------------------------------------
_collapse_weighted_methods = set(
    (
        "mean",
        "mean_absolute_value",
        "mean_of_upper_decile",
        "avg",
        "average",
        "sd",
        "standard_deviation",
        "sum",
        "var",
        "variance",
        "sum_of_weights",
        "sum_of_weights2",
        "integral",
        "root_mean_square",
    )
)

# --------------------------------------------------------------------
# These Data methods may specify a number of degrees of freedom
# --------------------------------------------------------------------
_collapse_ddof_methods = set(("sd", "var"))

_earth_radius = Data(6371229.0, "m")

_relational_methods = (
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
)

_xxx = namedtuple(
    "data_dimension", ["size", "axis", "key", "coord", "coord_type", "scalar"]
)

# _empty_set = set()


class Field(mixin.FieldDomain, mixin.PropertiesData, cfdm.Field):
    """A field construct of the CF data model.

    The field construct is central to the CF data model, and includes
    all the other constructs. A field corresponds to a CF-netCDF data
    variable with all of its metadata. All CF-netCDF elements are
    mapped to a field construct or some element of the CF field
    construct. The field construct contains all the data and metadata
    which can be extracted from the file using the CF conventions.

    The field construct consists of a data array and the definition of
    its domain (that describes the locations of each cell of the data
    array), field ancillary constructs containing metadata defined
    over the same domain, and cell method constructs to describe how
    the cell values represent the variation of the physical quantity
    within the cells of the domain. The domain is defined collectively
    by the following constructs of the CF data model: domain axis,
    dimension coordinate, auxiliary coordinate, cell measure,
    coordinate reference and domain ancillary constructs.

    The field construct also has optional properties to describe
    aspects of the data that are independent of the domain. These
    correspond to some netCDF attributes of variables (e.g. units,
    long_name and standard_name), and some netCDF global file
    attributes (e.g. history and institution).

    **NetCDF interface**

    {{netCDF variable}}

    {{netCDF global attributes}}

    {{netCDF group attributes}}

    {{netCDF geometry group}}

    Some components exist within multiple constructs, but when written
    to a netCDF dataset the netCDF names associated with such
    components will be arbitrarily taken from one of them. The netCDF
    variable, dimension and sample dimension names and group
    structures for such components may be set or removed consistently
    across all such components with the `nc_del_component_variable`,
    `nc_set_component_variable`, `nc_set_component_variable_groups`,
    `nc_clear_component_variable_groups`,
    `nc_del_component_dimension`, `nc_set_component_dimension`,
    `nc_set_component_dimension_groups`,
    `nc_clear_component_dimension_groups`,
    `nc_del_component_sample_dimension`,
    `nc_set_component_sample_dimension`,
    `nc_set_component_sample_dimension_groups`,
    `nc_clear_component_sample_dimension_groups` methods.

    CF-compliance issues for field constructs read from a netCDF
    dataset may be accessed with the `dataset_compliance` method.

    """

    def __new__(cls, *args, **kwargs):
        """Store component classes."""
        instance = super().__new__(cls)
        instance._AuxiliaryCoordinate = AuxiliaryCoordinate
        instance._DimensionCoordinate = DimensionCoordinate
        instance._Bounds = Bounds
        instance._Constructs = Constructs
        instance._Domain = Domain
        instance._DomainAncillary = DomainAncillary
        instance._DomainAxis = DomainAxis
        #        instance._Data = Data
        instance._RaggedContiguousArray = RaggedContiguousArray
        instance._RaggedIndexedArray = RaggedIndexedArray
        instance._RaggedIndexedContiguousArray = RaggedIndexedContiguousArray
        instance._GatheredArray = GatheredArray
        instance._Count = Count
        instance._Index = Index
        instance._List = List
        return instance

    _special_properties = mixin.PropertiesData._special_properties
    _special_properties += ("flag_values", "flag_masks", "flag_meanings")

    def __init__(
        self, properties=None, source=None, copy=True, _use_data=True
    ):
        """**Initialisation**

        :Parameters:

            properties: `dict`, optional
                Set descriptive properties. The dictionary keys are
                property names, with corresponding values. Ignored if the
                *source* parameter is set.

                *Parameter example:*
                  ``properties={'standard_name': 'air_temperature'}``

                Properties may also be set after initialisation with the
                `set_properties` and `set_property` methods.

            {{init source: optional}}

            {{init copy: `bool`, optional}}

        """
        super().__init__(
            properties=properties,
            source=source,
            copy=copy,
            _use_data=_use_data,
        )

        if source:
            flags = getattr(source, "Flags", None)
            if flags is not None:
                self.Flags = flags.copy()

    def __getitem__(self, indices):
        """Return a subspace of the field construct defined by indices.

        f.__getitem__(indices) <==> f[indices]

        Subspacing by indexing uses rules that are very similar to the
        numpy indexing rules, the only differences being:

        * An integer index i specified for a dimension reduces the size of
          this dimension to unity, taking just the i-th element, but keeps
          the dimension itself, so that the rank of the array is not
          reduced.

        * When two or more dimensions’ indices are sequences of integers
          then these indices work independently along each dimension
          (similar to the way vector subscripts work in Fortran). This is
          the same indexing behaviour as on a Variable object of the
          netCDF4 package.

        * For a dimension that is cyclic, a range of indices specified by
          a `slice` that spans the edges of the data (such as ``-2:3`` or
          ``3:-2:-1``) is assumed to "wrap" around, rather then producing
          a null result.

        .. seealso:: `indices`, `squeeze`, `subspace`, `where`

        **Examples**

        >>> f.shape
        (12, 73, 96)
        >>> f[0].shape
        (1, 73, 96)
        >>> f[3, slice(10, 0, -2), 95:93:-1].shape
        (1, 5, 2)

        >>> f.shape
        (12, 73, 96)
        >>> f[:, [0, 72], [5, 4, 3]].shape
        (12, 2, 3)

        >>> f.shape
        (12, 73, 96)
        >>> f[...].shape
        (12, 73, 96)
        >>> f[slice(0, 12), :, 10:0:-2].shape
        (12, 73, 5)
        >>> f[[True, True, False, True, True, False, False, True, True, True,
        ...    True, True]].shape
        (9, 64, 128)
        >>> f[..., :6, 9:1:-2, [1, 3, 4]].shape
        (6, 4, 3)

        """
        debug = is_log_level_debug(logger)

        if debug:
            logger.debug(
                self.__class__.__name__ + ".__getitem__"
            )  # pragma: no cover
            logger.debug(f"    input indices = {indices}")  # pragma: no cover

        if indices is Ellipsis:
            return self.copy()

        data = self.data
        shape = data.shape

        # Parse the index
        if not isinstance(indices, tuple):
            indices = (indices,)

        if isinstance(indices[0], str) and indices[0] == "mask":
            ancillary_mask = indices[:2]
            indices2 = indices[2:]
        else:
            ancillary_mask = None
            indices2 = indices

        indices, roll = parse_indices(shape, indices2, cyclic=True)

        if roll:
            new = self
            axes = data._axes
            cyclic_axes = data._cyclic
            for iaxis, shift in roll.items():
                axis = axes[iaxis]
                if axis not in cyclic_axes:
                    _ = self.get_data_axes()[iaxis]
                    raise IndexError(
                        "Can't take a cyclic slice from non-cyclic "
                        f"{self.constructs.domain_axis_identity(_)!r} axis"
                    )

                new = new.roll(axis=iaxis, shift=shift)
        else:
            new = self.copy()

        data = new.data

        # ------------------------------------------------------------
        # Subspace the field construct's data
        # ------------------------------------------------------------
        if ancillary_mask:
            ancillary_mask = list(ancillary_mask)
            findices = ancillary_mask + indices
        else:
            findices = indices

        if debug:
            logger.debug(f"    shape    = {shape}")  # pragma: no cover
            logger.debug(f"    indices  = {indices}")  # pragma: no cover
            logger.debug(f"    indices2 = {indices2}")  # pragma: no cover
            logger.debug(f"    findices = {findices}")  # pragma: no cover

        new_data = data[tuple(findices)]

        if 0 in new_data.shape:
            raise IndexError(
                f"Indices {findices!r} result in a subspaced shape of "
                f"{new_data.shape}, but can't create a subspace of "
                f"{self.__class__.__name__} that has a size 0 axis"
            )

        # Set sizes of domain axes
        data_axes = new.get_data_axes()
        domain_axes = new.domain_axes(todict=True)
        for axis, size in zip(data_axes, new_data.shape):
            domain_axes[axis].set_size(size)

        # Record which axes were cyclic before the subspace
        org_cyclic = [data_axes.index(axis) for axis in new.cyclic()]

        # Set the subspaced data
        new.set_data(new_data, axes=data_axes, copy=False)

        # Update axis cylcicity. Note that this can only entail
        # setting an originally cyclic axis to be non-cyclic. Doing
        # this now enables us to disable the (possibly very slow)
        # automatic check for cyclicity on the 'set_construct' calls
        # below.
        if org_cyclic:
            new_cyclic = new_data.cyclic()
            [
                new.cyclic(i, iscyclic=False)
                for i in org_cyclic
                if i not in new_cyclic
            ]

        # ------------------------------------------------------------
        # Subspace constructs with data
        # ------------------------------------------------------------
        if data_axes:
            construct_data_axes = new.constructs.data_axes()

            for key, construct in new.constructs.filter_by_axis(
                *data_axes, axis_mode="or", todict=True
            ).items():
                construct_axes = construct_data_axes[key]
                dice = []
                needs_slicing = False
                for axis in construct_axes:
                    if axis in data_axes:
                        needs_slicing = True
                        dice.append(indices[data_axes.index(axis)])
                    else:
                        dice.append(slice(None))

                if debug:
                    logger.debug(
                        f"    dice = {tuple(dice)}"
                    )  # pragma: no cover

                # Generally we do not apply an ancillary mask to the
                # metadata items, but for DSGs we do.
                if ancillary_mask and new.DSG:
                    item_mask = []
                    for mask in ancillary_mask[1]:
                        iaxes = [
                            data_axes.index(axis)
                            for axis in construct_axes
                            if axis in data_axes
                        ]
                        for i, (axis, size) in enumerate(
                            zip(data_axes, mask.shape)
                        ):
                            if axis not in construct_axes:
                                if size > 1:
                                    iaxes = None
                                    break

                                mask = mask.squeeze(i)

                        if iaxes is None:
                            item_mask = None
                            break
                        else:
                            mask1 = mask.transpose(iaxes)
                            for i, axis in enumerate(construct_axes):
                                if axis not in data_axes:
                                    mask1.inset_dimension(i)

                            item_mask.append(mask1)

                    if item_mask:
                        needs_slicing = True
                        dice = [ancillary_mask[0], item_mask] + dice

                # Replace existing construct with its subspace
                if needs_slicing:
                    new.set_construct(
                        construct[tuple(dice)],
                        key=key,
                        axes=construct_axes,
                        copy=False,
                        autocyclic={"no-op": True},
                    )

        new.set_data(new_data, axes=data_axes, copy=False)

        return new

    def __setitem__(self, indices, value):
        """Called to implement assignment to x[indices]=value.

        x.__setitem__(indices, value) <==> x[indices]=value

        .. versionadded:: 2.0

        """
        if isinstance(value, self.__class__):
            value = self._conform_for_assignment(value)

        try:
            data = value.get_data(None, _fill_value=False)
        except AttributeError:
            pass
        else:
            if data is None:
                raise ValueError(
                    f"Can't assign to a {self.__class__.__name__} from a "
                    f"{value.__class__.__name__} with no data"
                )

            value = data

        data = self.get_data(_fill_value=False)
        data[indices] = value

    #    @property
    #    def _cyclic(self):
    #        """Storage for axis cyclicity.
    #
    #        Do not change the value in-place.
    #
    #        """
    #        return self._custom.get("_cyclic", _empty_set)
    #
    #    @_cyclic.setter
    #    def _cyclic(self, value):
    #        """value must be a set.
    #
    #        Do not change the value in-place.
    #
    #        """
    #        self._custom["_cyclic"] = value
    #
    #    @_cyclic.deleter
    #    def _cyclic(self):
    #        self._custom["_cyclic"] = _empty_set

    def analyse_items(self, relaxed_identities=None):
        """Analyse a domain.

        :Returns:

            `dict`
                A description of the domain.

        **Examples**

        >>> print(f)
        Axes           : time(3) = [1979-05-01 12:00:00, ..., 1979-05-03 12:00:00] gregorian
                       : air_pressure(5) = [850.000061035, ..., 50.0000038147] hPa
                       : grid_longitude(106) = [-20.5400109887, ..., 25.6599887609] degrees
                       : grid_latitude(110) = [23.3200002313, ..., -24.6399995089] degrees
        Aux coords     : latitude(grid_latitude(110), grid_longitude(106)) = [[67.1246607722, ..., 22.8886948065]] degrees_N
                       : longitude(grid_latitude(110), grid_longitude(106)) = [[-45.98136251, ..., 35.2925499052]] degrees_E
        Coord refs     : <CF CoordinateReference: rotated_latitude_longitude>

        >>> f.analyse_items()
        {
         'dim_coords': {'dim0': <CF Dim ....>,
         'aux_coords': {'N-d': {'aux0': <CF AuxiliaryCoordinate: latitude(110, 106) degrees_N>,
                                'aux1': <CF AuxiliaryCoordinate: longitude(110, 106) degrees_E>},
                        'dim0': {'1-d': {},
                                 'N-d': {},},
                        'dim1': {'1-d': {},
                                 'N-d': {},},
                        'dim2': {'1-d': {},
                                 'N-d': {'aux0': <CF AuxiliaryCoordinate: latitude(110, 106) degrees_N>,
                                         'aux1': <CF AuxiliaryCoordinate: longitude(110, 106) degrees_E>},},
                        'dim3': {'1-d': {},
                                 'N-d': {'aux0': <CF AuxiliaryCoordinate: latitude(110, 106) degrees_N>,
                                         'aux1': <CF AuxiliaryCoordinate: longitude(110, 106) degrees_E>},},},
         'axis_to_coord': {'dim0': <CF DimensionCoordinate: time(3) gregorian>,
                           'dim1': <CF DimensionCoordinate: air_pressure(5) hPa>,
                           'dim2': <CF DimensionCoordinate: grid_latitude(110) degrees>,
                           'dim3': <CF DimensionCoordinate: grid_longitude(106) degrees>},
         'axis_to_id': {'dim0': 'time',
                        'dim1': 'air_pressure',
                        'dim2': 'grid_latitude',
                        'dim3': 'grid_longitude'},
         'cell_measures': {'N-d': {},
                           'dim0': {'1-d': {},
                                    'N-d': {},},
                           'dim1': {'1-d': {},
                                    'N-d': {},},
                           'dim2': {'1-d': {},
                                    'N-d': {},},
                           'dim3': {'1-d': {},
                                    'N-d': {},},
            },
         'id_to_aux': {},
         'id_to_axis': {'air_pressure': 'dim1',
                        'grid_latitude': 'dim2',
                        'grid_longitude': 'dim3',
                        'time': 'dim0'},
         'id_to_coord': {'air_pressure': <CF DimensionCoordinate: air_pressure(5) hPa>,
                         'grid_latitude': <CF DimensionCoordinate: grid_latitude(110) degrees>,
                         'grid_longitude': <CF DimensionCoordinate: grid_longitude(106) degrees>,
                         'time': <CF DimensionCoordinate: time(3) gregorian>},
         'id_to_key': {'air_pressure': 'dim1',
                       'grid_latitude': 'dim2',
                       'grid_longitude': 'dim3',
                       'time': 'dim0'},
         'undefined_axes': [],
         'warnings': [],
        }

        """
        # ------------------------------------------------------------
        # Map each axis identity to its identifier, if such a mapping
        # exists.
        #
        # For example:
        # >>> id_to_axis
        # {'time': 'dim0', 'height': dim1'}
        # ------------------------------------------------------------
        id_to_axis = {}

        # ------------------------------------------------------------
        # For each dimension that is identified by a 1-d auxiliary
        # coordinate, map its dimension's its identifier.
        #
        # For example:
        # >>> id_to_aux
        # {'region': 'aux0'}
        # ------------------------------------------------------------
        id_to_aux = {}

        # ------------------------------------------------------------
        # The keys of the coordinate items which provide axis
        # identities
        #
        # For example:
        # >>> id_to_key
        # {'region': 'aux0'}
        # ------------------------------------------------------------
        #        id_to_key = {}

        axis_to_id = {}

        # ------------------------------------------------------------
        # Map each dimension's identity to the coordinate which
        # provides that identity.
        #
        # For example:
        # >>> id_to_coord
        # {'time': <CF Coordinate: time(12)>}
        # ------------------------------------------------------------
        id_to_coord = {}

        axis_to_coord = {}

        # ------------------------------------------------------------
        # List the dimensions which are undefined, in that no unique
        # identity can be assigned to them.
        #
        # For example:
        # >>> undefined_axes
        # ['dim2']
        # ------------------------------------------------------------
        undefined_axes = []

        # ------------------------------------------------------------
        #
        # ------------------------------------------------------------
        warnings = []
        id_to_dim = {}
        axis_to_aux = {}
        axis_to_dim = {}

        if relaxed_identities is None:
            relaxed_identities = cf_relaxed_identities()

        #        dimension_coordinates = self.dimension_coordinates(view=True)
        #        auxiliary_coordinates = self.auxiliary_coordinates(view=True)

        for axis in self.domain_axes(todict=True):
            #            dims = self.constructs.chain(
            #                "filter_by_type",
            #                ("dimension_coordinate",), "filter_by_axis", (axis,)
            #                mode="and", todict=True
            #            )
            key, dim = self.dimension_coordinate(
                item=True, default=(None, None), filter_by_axis=(axis,)
            )

            if dim is not None:
                # This axis of the domain has a dimension coordinate
                identity = dim.identity(strict=True, default=None)
                if identity is None:
                    # Dimension coordinate has no identity, but it may
                    # have a recognised axis.
                    for ctype in ("T", "X", "Y", "Z"):
                        if getattr(dim, ctype, False):
                            identity = ctype
                            break

                if identity is None and relaxed_identities:
                    identity = dim.identity(relaxed=True, default=None)

                if identity:
                    if identity in id_to_axis:
                        warnings.append("Field has multiple {identity!r} axes")

                    axis_to_id[axis] = identity
                    id_to_axis[identity] = axis
                    axis_to_coord[axis] = key
                    id_to_coord[identity] = key
                    axis_to_dim[axis] = key
                    id_to_dim[identity] = key
                    continue

            else:
                key, aux = self.auxiliary_coordinate(
                    filter_by_axis=(axis,),
                    axis_mode="and",  # TODO check this "and"
                    item=True,
                    default=(None, None),
                )
                if aux is not None:
                    # This axis of the domain does not have a
                    # dimension coordinate but it does have exactly
                    # one 1-d auxiliary coordinate, so that will do.
                    identity = aux.identity(strict=True, default=None)

                    if identity is None and relaxed_identities:
                        identity = aux.identity(relaxed=True, default=None)

                    if identity and aux.has_data():
                        if identity in id_to_axis:
                            warnings.append(
                                f"Field has multiple {identity!r} axes"
                            )

                        axis_to_id[axis] = identity
                        id_to_axis[identity] = axis
                        axis_to_coord[axis] = key
                        id_to_coord[identity] = key
                        axis_to_aux[axis] = key
                        id_to_aux[identity] = key
                        continue

            # Still here? Then this axis is undefined
            undefined_axes.append(axis)

        return {
            "axis_to_id": axis_to_id,
            "id_to_axis": id_to_axis,
            "axis_to_coord": axis_to_coord,
            "axis_to_dim": axis_to_dim,
            "axis_to_aux": axis_to_aux,
            "id_to_coord": id_to_coord,
            "id_to_dim": id_to_dim,
            "id_to_aux": id_to_aux,
            "undefined_axes": undefined_axes,
            "warnings": warnings,
        }

    def _is_broadcastable(self, shape):
        """Checks the field's data array is broadcastable to a shape.

        :Parameters:

            shape: sequence of `int`

        :Returns:

            `bool`

        """
        shape0 = getattr(self, "shape", None)
        if shape is None:
            return False

        shape1 = shape

        if tuple(shape1) == tuple(shape0):
            # Same shape
            return True

        ndim0 = len(shape0)
        ndim1 = len(shape1)
        if not ndim0 or not ndim1:
            # Either or both is scalar
            return True

        for setN in set(shape0), set(shape1):
            if setN == {1}:
                return True

        if ndim1 > ndim0:
            return False

        for n, m in zip(shape1[::-1], shape0[::-1]):
            if n != m and n != 1:
                return False

        return True

    def _axis_positions(self, axes, parse=True, return_axes=False):
        """Convert the given axes to their positions in the data.

        Any domain axes that are not spanned by the data are ignored.

        If there is no data then an empty list is returned.

        .. versionadded:: 3.9.0

        :Parameters:
            axes: (sequence of) `str` or `int`
                The axes to be converted. TODO domain axis selection

            parse: `bool`, optional

                If False then do not parse the *axes*. Parsing should
                always occur unless the given *axes* are the output of
                a previous call to `parse_axes`. By default *axes* is
                parsed by `_parse_axes`.

            return_axes: `bool`, optional

                If True then also return the domain axis identifiers
                corresponding to the positions.

        :Returns:

            `list` [, `list`]
                The domain axis identifiers. If *return_axes* is True
                then also return the corresponding domain axis
                identifiers.

        """
        data_axes = self.get_data_axes(default=None)
        if data_axes is None:
            return []

        if parse:
            axes = self._parse_axes(axes)

        axes = [a for a in axes if a in data_axes]
        positions = [data_axes.index(a) for a in axes]

        if return_axes:
            return positions, axes

        return positions

    def _binary_operation(self, other, method):
        """Implement binary arithmetic and comparison operations on the
        master data array with metadata-aware broadcasting.

        It is intended to be called by the binary arithmetic and
        comparison methods, such as `__sub__`, `__imul__`, `__rdiv__`,
        `__lt__`, etc.

        :Parameters:

            other: `Field` or `Query` or any object that broadcasts to the field construct's data

            method: `str`
                The binary arithmetic or comparison method name (such as
                ``'__idiv__'`` or ``'__ge__'``).

        :Returns:

            `Field`
                The new field, or the same field if the operation was an
                in place augmented arithmetic assignment.

        **Examples**

        >>> h = f._binary_operation(g, '__add__')
        >>> h = f._binary_operation(g, '__ge__')
        >>> f._binary_operation(g, '__isub__')
        >>> f._binary_operation(g, '__rdiv__')

        """
        debug = is_log_level_debug(logger)

        if isinstance(other, Query):
            # --------------------------------------------------------
            # Combine the field with a Query object
            # --------------------------------------------------------
            return NotImplemented

        if not isinstance(other, self.__class__):
            # --------------------------------------------------------
            # Combine the field with anything other than a Query
            # object or another field construct
            # --------------------------------------------------------
            if cf_size(other) == 1:
                # ----------------------------------------------------
                # No changes to the field metadata constructs are
                # required so can use the metadata-unaware parent
                # method
                # ----------------------------------------------------
                if other is None:
                    other = np.array(None, dtype=object)

                other = Data(other)
                if other.ndim > 0:
                    other.squeeze(inplace=True)

                return super()._binary_operation(other, method)

            if self._is_broadcastable(np.shape(other)):
                return super()._binary_operation(other, method)

            raise ValueError(
                f"Can't combine {self.__class__.__name__!r} with "
                f"{other.__class__.__name__!r} due to incompatible data "
                f"shapes: {self.shape}, {np.shape(other)})"
            )

        # ============================================================
        # Still here? Then combine the field with another field
        # ============================================================
        relaxed_identities = cf_relaxed_identities()

        units = self.Units
        sn = self.get_property("standard_name", None)
        ln = self.get_property("long_name", None)

        other_sn = other.get_property("standard_name", None)
        other_ln = other.get_property("long_name", None)

        field1 = other.copy()

        inplace = method[2] == "i"
        if not inplace:
            field0 = self.copy()
        else:
            field0 = self

        # Analyse the two fields' data array dimensions
        out0 = {}
        out1 = {}
        for i, (f, out) in enumerate(zip((field0, field1), (out0, out1))):
            data_axes = f.get_data_axes()
            for axis in f.domain_axes(todict=True):
                identity = None
                key = None
                coord = None
                coord_type = None

                key, coord = f.dimension_coordinate(
                    item=True, default=(None, None), filter_by_axis=(axis,)
                )
                if coord is not None:
                    # This axis of the domain has a dimension
                    # coordinate
                    identity = coord.identity(strict=True, default=None)
                    if identity is None:
                        # Dimension coordinate has no identity, but it
                        # may have a recognised axis.
                        for ctype in ("T", "X", "Y", "Z"):
                            if getattr(coord, ctype, False):
                                identity = ctype
                                break

                    if identity is None and relaxed_identities:
                        identity = coord.identity(relaxed=True, default=None)
                else:
                    key, coord = f.auxiliary_coordinate(
                        item=True,
                        default=(None, None),
                        filter_by_axis=(axis,),
                        axis_mode="exact",
                    )
                    if coord is not None:
                        # This axis of the domain does not have a
                        # dimension coordinate but it does have
                        # exactly one 1-d auxiliary coordinate, so
                        # that will do.
                        identity = coord.identity(strict=True, default=None)

                        if identity is None and relaxed_identities:
                            identity = coord.identity(
                                relaxed=True, default=None
                            )

                if identity is None:
                    identity = i
                else:
                    coord_type = coord.construct_type

                out[identity] = _xxx(
                    size=f.domain_axis(axis).get_size(),
                    axis=axis,
                    key=key,
                    coord=coord,
                    coord_type=coord_type,
                    scalar=(axis not in data_axes),
                )

        for identity, y in tuple(out1.items()):
            asdas = True
            if y.scalar and identity in out0 and isinstance(identity, str):
                a = out0[identity]
                if a.size > 1:
                    field1.insert_dimension(y.axis, position=0, inplace=True)
                    asdas = False

            if y.scalar and asdas:
                del out1[identity]

        for identity, a in tuple(out0.items()):
            asdas = True
            if a.scalar and identity in out1 and isinstance(identity, str):
                y = out1[identity]
                if y.size > 1:
                    field0.insert_dimension(a.axis, position=0, inplace=True)
                    asdas = False

            if a.scalar and asdas:
                del out0[identity]

        squeeze1 = []
        insert0 = []

        # List of axes that will have been added to field0 as new
        # trailing dimensions. E.g. ['domainaxis1']
        axes_added_from_field1 = []

        # Dictionary of size > 1 axes from field1 which will replace
        # matching size 1 axes in field0. E.g. {'domainaxis1':
        #     data_dimension(
        #         size=8,
        #         axis='domainaxis1',
        #         key='dimensioncoordinate1',
        #         coord=<CF DimensionCoordinate: longitude(8) degrees_east>,
        #         coord_type='dimension_coordinate',
        #         scalar=False
        #     )
        # }
        axes_to_replace_from_field1 = {}

        # List of field1 coordinate reference constructs which will be
        # added to field0
        refs_to_add_from_field1 = []

        # Check that the two fields are combinable
        for i, (identity, y) in enumerate(tuple(out1.items())):
            if isinstance(identity, int):
                if y.size == 1:
                    del out1[identity]
                    squeeze1.append(i)
                else:
                    insert0.append(y.axis)
            elif identity not in out0:
                insert0.append(y.axis)

        # Make sure that both data arrays have the same number of
        # dimensions
        if squeeze1:
            field1.squeeze(squeeze1, inplace=True)

        for axis1 in insert0:
            new_axis0 = field0.set_construct(self._DomainAxis(1))
            field0.insert_dimension(
                new_axis0, position=field0.ndim, inplace=True
            )
            axes_added_from_field1.append(axis1)

        while field1.ndim < field0.ndim:
            new_axis = field1.set_construct(self._DomainAxis(1))
            field1.insert_dimension(new_axis, position=0, inplace=True)

        while field0.ndim < field1.ndim:
            new_axis = field0.set_construct(self._DomainAxis(1))
            field0.insert_dimension(
                new_axis, position=field0.ndim, inplace=True
            )

        # Make sure that the dimensions in data1 are in the same order
        # as the dimensions in data0
        for identity, y in out1.items():
            if debug:
                logger.debug(f"{identity} {y}")

            if isinstance(identity, int) or identity not in out0:
                field1.swapaxes(
                    field1.get_data_axes().index(y.axis), -1, inplace=True
                )
            else:
                # This identity is also in out0
                a = out0[identity]
                if debug:
                    logger.debug(f"{identity} {y.axis} {a.axis}")
                    logger.debug(
                        f"{a} {field0.get_data_axes()} "
                        f"{field1.get_data_axes()} "
                        f"{field1.get_data_axes().index(y.axis)} "
                        f"{field0.get_data_axes().index(a.axis)}"
                    )

                field1.swapaxes(
                    field1.get_data_axes().index(y.axis),
                    field0.get_data_axes().index(a.axis),
                    inplace=True,
                )

        axis_map = {
            axis1: axis0
            for axis1, axis0 in zip(
                field1.get_data_axes(), field0.get_data_axes()
            )
        }

        if debug:
            logger.debug(f"\naxis_map= {axis_map}\n")

        # ------------------------------------------------------------
        # Check that the two fields have compatible metadata
        # ------------------------------------------------------------
        for i, (identity, y) in enumerate(tuple(out1.items())):
            if isinstance(identity, int) or identity not in out0:
                continue

            a = out0[identity]

            if y.size == 1:
                continue

            if y.size > 1 and a.size == 1:
                axes_to_replace_from_field1[y.axis] = y
                continue

            if y.size != a.size:
                raise ValueError(
                    f"Can't broadcast size {y.size} {identity!r} axis to size "
                    f"{a.size} {identity!r} axis"
                )

            # Ensure matching axis directions
            if y.coord.direction() != a.coord.direction():
                other.flip(y.axis, inplace=True)

            # Check for matching coordinate values
            if not y.coord._equivalent_data(a.coord):
                raise ValueError(
                    f"Can't combine size {y.size} {identity!r} axes with "
                    f"non-matching coordinate values"
                )

            # Check coord refs
            refs1 = field1.get_coordinate_reference(construct=y.key, key=True)
            refs0 = field0.get_coordinate_reference(construct=a.key, key=True)

            n_refs = len(refs1)
            n0_refs = len(refs0)

            if n_refs != n0_refs:
                raise ValueError(
                    f"Can't combine {self.__class__.__name__!r} with "
                    f"{other.__class__.__name__!r} because the coordinate "
                    f"references have different lengths: {n_refs} and "
                    f"{n0_refs}."
                )

            n_equivalent_refs = 0
            for ref1 in refs1:
                for ref0 in refs0[:]:
                    if field1._equivalent_coordinate_references(
                        field0, key0=ref1, key1=ref0, axis_map=axis_map
                    ):
                        n_equivalent_refs += 1
                        refs0.remove(ref0)
                        break

            if n_equivalent_refs != n_refs:
                raise ValueError(
                    f"Can't combine {self.__class__.__name__!r} with "
                    f"{other.__class__.__name__!r} because the fields have "
                    "incompatible coordinate references."
                )

        # Change the domain axis sizes in field0 so that they match
        # the broadcasted result data
        for identity, y in out1.items():
            if identity in out0 and isinstance(identity, str):
                a = out0[identity]
                if y.size > 1 and a.size == 1:
                    for key0, c in field0.constructs.filter_by_axis(
                        a.axis, axis_mode="or", todict=True
                    ).items():
                        removed_refs0 = field0.del_coordinate_reference(
                            construct=key0, default=None
                        )
                        if removed_refs0 and c.construct_type in (
                            "dimension_coordinate",
                            "auxiliary_coordinate",
                        ):
                            for ref in removed_refs0:
                                for key0 in ref.coordinates():
                                    field0.del_construct(key0, default=None)

                        field0.del_construct(key0, default=None)

                    field0.domain_axis(a.axis).set_size(y.size)
            elif y.size > 1:
                axis0 = axis_map[y.axis]
                field0.domain_axis(axis0).set_size(y.size)

        # ------------------------------------------------------------
        # Operate on the data
        # ------------------------------------------------------------
        new_data = field0.data._binary_operation(field1.data, method)

        field0.set_data(new_data, set_axes=False, copy=False)

        if debug:
            logger.debug(
                f"\naxes_added_from_field1= {axes_added_from_field1}\n"
            )
            logger.debug(
                f"axes_to_replace_from_field1= {axes_to_replace_from_field1}"
            )

        already_copied = {}

        # ------------------------------------------------------------
        # Copy over coordinate and cell meausure constructs from
        # field1
        # ------------------------------------------------------------
        #       if axes_added_from_field1:
        #           constructs = field1.constructs.filter_by_type(
        #               'dimension_coordinate', 'auxiliary_coordinate',
        #               'cell_measure')
        # #            constructs = constructs.filter_by_axis(
        #                  'subset', *axes_added_from_field1)
        #
        #            for key1, c in constructs.items():
        #                axes = [axis_map[axis1] for axis1 in
        #                        field1.get_data_axes(key1)]
        #                key0 = field0.set_construct(c, axes=axes, copy=False)
        #                already_copied[key1] = key0

        #        for axis1, y in axes_to_replace_from_field1.items():
        #            axis0 = axis_map[axis1]
        new_axes = set(axes_added_from_field1).union(
            axes_to_replace_from_field1
        )

        if debug:
            logger.debug(f"\nnew_axes ={new_axes}")

        if new_axes:
            constructs = field1.constructs.filter(
                filter_by_type=(
                    "dimension_coordinate",
                    "auxiliary_coordinate",
                    "cell_measure",
                ),
                filter_by_axis=new_axes,
                axis_mode="subset",
                todict=True,
            )
            #            constructs = field1.constructs.filter_by_type(
            #                "dimension_coordinate",
            #                "auxiliary_coordinate",
            #                "cell_measure",
            #                view=True,
            #            )
            #            constructs = constructs.filter_by_axis(
            #                *new_axes, mode="subset", view=True
            #            )
            for key, c in constructs.items():
                c_axes = field1.get_data_axes(key)
                axes = [axis_map[axis1] for axis1 in c_axes]
                key0 = field0.set_construct(c, axes=axes, copy=False)
                already_copied[key] = key0

        # ------------------------------------------------------------
        # Copy over coordinate reference constructs from field1,
        # including their domain ancillary constructs.
        # ------------------------------------------------------------
        for key, ref in field1.coordinate_references(todict=True).items():
            axes = field1._coordinate_reference_axes(key)
            if axes.issubset(new_axes):
                refs_to_add_from_field1.append(ref)
            elif axes.intersection(axes_to_replace_from_field1):
                refs_to_add_from_field1.append(ref)

        if debug:
            logger.debug(
                f"\nrefs_to_add_from_field1={refs_to_add_from_field1}"
            )  # pragma: no cover

        for ref in refs_to_add_from_field1:
            # Copy coordinates
            coords = []
            for key1 in ref.coordinates():
                if key1 not in already_copied:
                    c = field1.constructs.get(key1, None)
                    if c is None:
                        already_copied[key1] = None
                    else:
                        axes = [
                            axis_map[axis]
                            for axis in field1.get_data_axes(key1)
                        ]
                        key0 = field0.set_construct(c, axes=axes, copy=False)
                        already_copied[key1] = key0

                key0 = already_copied[key1]
                if key0 is not None:
                    coords.append(key0)

            ref.clear_coordinates()
            ref.set_coordinates(coords)

            # Copy domain ancillaries to field0
            for (
                term,
                key1,
            ) in ref.coordinate_conversion.domain_ancillaries().items():
                if key1 not in already_copied:
                    c = field1.constructs.get(key1, None)
                    if c is None:
                        already_copied[key1] = None
                    else:
                        axes = [
                            axis_map[axis]
                            for axis in field1.get_data_axes(key1)
                        ]
                        key0 = field0.set_construct(c, axes=axes, copy=False)
                        already_copied[key1] = key0

                key0 = already_copied[key1]
                ref.coordinate_conversion.set_domain_ancillary(term, key0)

            # Copy coordinate reference to field0
            field0.set_construct(ref, copy=False)

        # ------------------------------------------------------------
        # Remove misleading identities
        # ------------------------------------------------------------
        # Warning: This block of code is replicated in PropertiesData
        if sn != other_sn:
            if sn is not None and other_sn is not None:
                field0.del_property("standard_name", None)
                field0.del_property("long_name", None)
            elif other_sn is not None:
                field0.set_property("standard_name", other_sn, copy=False)
                if other_ln is None:
                    field0.del_property("long_name", None)
                else:
                    field0.set_property("long_name", other_ln, copy=False)
        elif ln is None and other_ln is not None:
            field0.set_property("long_name", other_ln, copy=False)

        # Warning: This block of code is replicated in PropertiesData
        new_units = field0.Units
        if (
            method in _relational_methods
            or not units.equivalent(new_units)
            and not (units.isreftime and new_units.isreftime)
        ):
            field0.del_property("standard_name", None)
            field0.del_property("long_name", None)

        if method in _relational_methods:
            field0.override_units(Units(), inplace=True)

        # ------------------------------------------------------------
        # Return the result field
        # ------------------------------------------------------------
        return field0

    def _conform_cell_methods(self):
        """Changes the axes of the field's cell methods so they conform.

        :Returns:

            `None`

        """
        axis_map = {}

        for cm in self.cell_methods(todict=True).values():
            for axis in cm.get_axes(()):
                if axis in axis_map:
                    continue

                if axis == "area":
                    axis_map[axis] = axis
                    continue

                axis_map[axis] = self.domain_axis(axis, key=True, default=axis)

            cm.change_axes(axis_map, inplace=True)

    def _conform_for_assignment(self, other, check_coordinates=False):
        """Conform *other* so that it is ready for metadata-unaware
        assignment broadcasting across *self*.

        Note that *other* is not changed.

        :Parameters:

            other: `Field`
                The field to conform.

        :Returns:

            `Field`
                The conformed version of *other*.

        **Examples**

        >>> h = f._conform_for_assignment(g)

        """
        # Analyse each domain
        s = self.analyse_items()
        v = other.analyse_items()

        if s["warnings"] or v["warnings"]:
            raise ValueError(
                f"Can't setitem: {s['warnings'] or v['warnings']}"
            )

        # Find the set of matching axes
        matching_ids = set(s["id_to_axis"]).intersection(v["id_to_axis"])
        if not matching_ids:
            raise ValueError("Can't assign: No matching axes")

        # ------------------------------------------------------------
        # Check that any matching axes defined by auxiliary
        # coordinates are done so in both fields.
        # ------------------------------------------------------------
        for identity in matching_ids:
            if (identity in s["id_to_aux"]) + (
                identity in v["id_to_aux"]
            ) == 1:
                raise ValueError(
                    f"Can't assign: {identity!r} axis defined by auxiliary in "
                    "only 1 field"
                )

        copied = False

        # ------------------------------------------------------------
        # Check that 1) all undefined axes in other have size 1 and 2)
        # that all of other's unmatched but defined axes have size 1
        # and squeeze any such axes out of its data array.
        #
        # For example, if   self.data is        P T     Z Y   X   A
        #              and  other.data is     1     B C   Y 1 X T
        #              then other.data becomes            Y   X T
        # ------------------------------------------------------------
        squeeze_axes1 = []
        other_domain_axes = other.domain_axes(todict=True)

        for axis1 in v["undefined_axes"]:
            axis_size = other_domain_axes[axis1].get_size()
            if axis_size != 1:
                raise ValueError(
                    "Can't assign: Can't broadcast undefined axis with "
                    f"size {axis_size}"
                )

            squeeze_axes1.append(axis1)

        for identity in set(v["id_to_axis"]).difference(matching_ids):
            axis1 = v["id_to_axis"][identity]
            axis_size = other_domain_axes[axis1].get_size()
            if axis_size != 1:
                raise ValueError(
                    "Can't assign: Can't broadcast size "
                    f"{axis_size} {identity!r} axis"
                )

            squeeze_axes1.append(axis1)

        if squeeze_axes1:
            if not copied:
                other = other.copy()
                copied = True

            other.squeeze(squeeze_axes1, inplace=True)

        # ------------------------------------------------------------
        # Permute the axes of other.data so that they are in the same
        # order as their matching counterparts in self.data
        #
        # For example, if   self.data is       P T Z Y X   A
        #              and  other.data is            Y X T
        #              then other.data becomes   T   Y X
        # ------------------------------------------------------------
        data_axes0 = self.get_data_axes()
        data_axes1 = other.get_data_axes()

        transpose_axes1 = []
        for axis0 in data_axes0:
            identity = s["axis_to_id"][axis0]
            if identity in matching_ids:
                axis1 = v["id_to_axis"][identity]
                if axis1 in data_axes1:
                    transpose_axes1.append(axis1)

        if transpose_axes1 != data_axes1:
            if not copied:
                other = other.copy()
                copied = True

            other.transpose(transpose_axes1, inplace=True)

        # ------------------------------------------------------------
        # Insert size 1 axes into other.data to match axes in
        # self.data which other.data doesn't have.
        #
        # For example, if   self.data is       P T Z Y X A
        #              and  other.data is        T   Y X
        #              then other.data becomes 1 T 1 Y X 1
        # ------------------------------------------------------------
        expand_positions1 = []
        for i, axis0 in enumerate(data_axes0):
            identity = s["axis_to_id"][axis0]
            if identity in matching_ids:
                axis1 = v["id_to_axis"][identity]
                if axis1 not in data_axes1:
                    expand_positions1.append(i)
            else:
                expand_positions1.append(i)

        if expand_positions1:
            if not copied:
                other = other.copy()
                copied = True

            for i in expand_positions1:
                new_axis = other.set_construct(other._DomainAxis(1))
                other.insert_dimension(new_axis, position=i, inplace=True)

        # ----------------------------------------------------------------
        # Make sure that each pair of matching axes has the same
        # direction
        # ----------------------------------------------------------------
        flip_axes1 = []
        for identity in matching_ids:
            axis1 = v["id_to_axis"][identity]
            axis0 = s["id_to_axis"][identity]
            if other.direction(axis1) != self.direction(axis0):
                flip_axes1.append(axis1)

        if flip_axes1:
            if not copied:
                other = other.copy()
                copied = True

            other.flip(flip_axes1, inplace=True)

        # Find the axis names which are present in both fields
        if not check_coordinates:
            return other

        # Still here?
        matching_ids = set(s["id_to_axis"]).intersection(v["id_to_axis"])

        for identity in matching_ids:
            key0 = s["id_to_coord"][identity]
            key1 = v["id_to_coord"][identity]

            coord0 = self.constructs[key0]
            coord1 = other.constructs[key1]

            # Check the sizes of the defining coordinates
            size0 = coord0.size
            size1 = coord1.size
            if size0 != size1:
                if size0 == 1 or size1 == 1:
                    continue

                raise ValueError(
                    f"Can't broadcast {identity!r} axes with sizes {size0} "
                    f"and {size1}"
                )

            # Check that equally sized defining coordinate data arrays
            # are compatible
            if not coord0._equivalent_data(coord1):
                raise ValueError(
                    f"Matching {identity!r} coordinate constructs have "
                    "different data"
                )

            # If the defining coordinates are attached to
            # coordinate references then check that those
            # coordinate references are equivalent

            # For each field, find the coordinate references which
            # contain the defining coordinate.
            refs0 = [
                key
                for key, ref in self.coordinate_references(todict=True).items()
                if key0 in ref.coordinates()
            ]
            refs1 = [
                key
                for key, ref in other.coordinate_references(
                    todict=True
                ).items()
                if key1 in ref.coordinates()
            ]

            nrefs = len(refs0)
            error_msg = (
                f"Can't combine {self.__class__.__name__!r} with "
                f"{other.__class__.__name__!r} because the defining "
                "coordinates are attached to incompatible coordinate "
                "references."
            )
            if nrefs > 1 or nrefs != len(refs1):
                raise ValueError(error_msg)

            if nrefs and not self._equivalent_coordinate_references(
                other, key0=refs0[0], key1=refs1[0], s=s, t=v
            ):
                raise ValueError(error_msg)

        return other

    def _conform_for_data_broadcasting(self, other):
        """Conforms the field with another, ready for data broadcasting.

        Note that the other field, *other*, is not changed in-place.

        :Parameters:

            other: `Field`
                The field to conform.

        :Returns:

            `Field`
                The conformed version of *other*.

        **Examples**

        >>> h = f._conform_for_data_broadcasting(g)

        """

        other = self._conform_for_assignment(other, check_coordinates=True)

        # Remove leading size one dimensions
        ndiff = other.ndim - self.ndim
        if ndiff > 0 and set(other.shape[:ndiff]) == set((1,)):
            for i in range(ndiff):
                other = other.squeeze(0)

        return other

    @_manage_log_level_via_verbosity
    def _equivalent_construct_data(
        self,
        field1,
        key0=None,
        key1=None,
        s=None,
        t=None,
        atol=None,
        rtol=None,
        verbose=None,
        axis_map=None,
    ):
        """True if the field has equivalent construct data to another.

        Two real numbers ``x`` and ``y`` are considered equal if
        ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on absolute
        differences) and ``rtol`` (the tolerance on relative differences)
        are positive, typically very small numbers. See the *atol* and
        *rtol* parameters.

        :Parameters:

            key0: `str`

            key1: `str`

            field1: `Field`

            s: `dict`, optional

            t: `dict`, optional

            atol: `float`, optional
                The tolerance on absolute differences between real
                numbers. The default value is set by the `atol` function.

            rtol: `float`, optional
                The tolerance on relative differences between real
                numbers. The default value is set by the `rtol` function.

            {{verbose: `int` or `str` or `None`, optional}}

        """
        item0 = self.constructs[key0]
        item1 = field1.constructs[key1]

        if item0.has_data() != item1.has_data():
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Only one item has data"
                )  # pragma: no cover

            return False

        if not item0.has_data():
            # Neither field has a data array
            return True

        if item0.size != item1.size:
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Different metadata construct "
                    f"data array size: {item0.size} != {item1.size}"
                )  # pragma: no cover

            return False

        if item0.ndim != item1.ndim:
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Different data array ranks "
                    f"({item0.ndim}, {item1.ndim})"
                )  # pragma: no cover

            return False

        axes0 = self.get_data_axes(key0, default=())
        axes1 = field1.get_data_axes(key1, default=())

        if s is None:
            s = self.analyse_items()
        if t is None:
            t = field1.analyse_items()

        transpose_axes = []
        if axis_map is None:
            for axis0 in axes0:
                axis1 = t["id_to_axis"].get(s["axis_to_id"][axis0], None)
                if axis1 is None:
                    if is_log_level_info(logger):
                        # TODO: improve message here (make user friendly):
                        logger.info(
                            "t['id_to_axis'] does not have a key "
                            f"s['axis_to_id'][axis0] for "
                            f"{self.__class__.__name__}"
                        )  # pragma: no cover

                    return False

                transpose_axes.append(axes1.index(axis1))
        else:
            for axis0 in axes0:
                axis1 = axis_map.get(axis0)
                if axis1 is None:
                    if is_log_level_info(logger):
                        # TODO: improve message here (make user friendly):
                        logger.info(
                            f"axis_map[axis0] is None for {self.__class__.__name__}"
                        )  # pragma: no cover

                    return False

                transpose_axes.append(axes1.index(axis1))

        copy1 = True

        if transpose_axes != list(range(item1.ndim)):
            if copy1:
                item1 = item1.copy()
                copy1 = False

            item1.transpose(transpose_axes, inplace=True)

        if item0.shape != item1.shape:
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Different shapes: "
                    f"{item0.shape} != {item1.shape}"
                )  # pragma: no cover

            return False

        flip_axes = [
            i
            for i, (axis1, axis0) in enumerate(zip(axes1, axes0))
            if field1.direction(axis1) != self.direction(axis0)
        ]

        if flip_axes:
            if copy1:
                item1 = item1.copy()
                copy1 = False

            item1.flip(flip_axes, inplace=True)

        if not item0._equivalent_data(
            item1, rtol=rtol, atol=atol, verbose=verbose
        ):
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Non-equivalent data"
                )  # pragma: no cover

            return False

        return True

    @property
    def DSG(self):
        """True if the field contains a collection of discrete sampling
        geometries.

        .. versionadded:: 2.0

        .. seealso:: `featureType`

        **Examples**

        >>> f.featureType
        'timeSeries'
        >>> f.DSG
        True

        >>> f.get_property('featureType', 'NOT SET')
        NOT SET
        >>> f.DSG
        False

        """
        return self.has_property("featureType")

    @property
    def Flags(self):
        """A `Flags` object containing self-describing CF flag values.

        Stores the `flag_values`, `flag_meanings` and `flag_masks` CF
        properties in an internally consistent manner.

        **Examples**

        >>> f.Flags
        <CF Flags: flag_values=[0 1 2], flag_masks=[0 2 2], flag_meanings=['low' 'medium' 'high']>

        """
        try:
            return self._custom["Flags"]
        except KeyError:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute 'Flags'"
            )

    @Flags.setter
    def Flags(self, value):
        self._custom["Flags"] = value

    @Flags.deleter
    def Flags(self):
        try:
            return self._custom.pop("Flags")
        except KeyError:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute 'Flags'"
            )

    @property
    def varray(self):
        """A numpy array view of the data array.

        Deprecated at version 3.14.0. Data are now stored as
        `dask` arrays for which, in general, a numpy array view is not
        robust.

        Changing the elements of the returned view changes the data array.

        .. seealso:: `array`, `data`, `datetime_array`

        **Examples**

        >>> f.data
        <CF Data(5): [0, ... 4] kg m-1 s-2>
        >>> a = f.array
        >>> type(a)
        <type 'numpy.ndarray'>
        >>> print(a)
        [0 1 2 3 4]
        >>> a[0] = 999
        >>> print(a)
        [999 1 2 3 4]
        >>> print(f.array)
        [999 1 2 3 4]
        >>> f.data
        <CF Data(5): [999, ... 4] kg m-1 s-2>

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "varray",
            message="Data are now stored as `dask` arrays for which, "
            "in general, a numpy array view is not robust.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    # ----------------------------------------------------------------
    # CF properties
    # ----------------------------------------------------------------
    @property
    def flag_values(self):
        """The flag_values CF property.

        Provides a list of the flag values. Use in conjunction with
        `flag_meanings`. See http://cfconventions.org/latest.html for
        details.

        Stored as a 1-d numpy array but may be set as any array-like
        object.

        **Examples**

        >>> f.flag_values = ['a', 'b', 'c']
        >>> f.flag_values
        array(['a', 'b', 'c'], dtype='|S1')
        >>> f.flag_values = numpy.arange(4)
        >>> f.flag_values
        array([1, 2, 3, 4])
        >>> del f.flag_values

        >>> f.set_property('flag_values', 1)
        >>> f.get_property('flag_values')
        array([1])
        >>> f.del_property('flag_values')

        """
        try:
            return self.Flags.flag_values
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_values'"
            )

    @flag_values.setter
    def flag_values(self, value):
        try:
            flags = self.Flags
        except AttributeError:
            self.Flags = Flags(flag_values=value)
        else:
            flags.flag_values = value

    @flag_values.deleter
    def flag_values(self):
        try:
            del self.Flags.flag_values
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_values'"
            )
        else:
            if not self.Flags:
                del self.Flags

    @property
    def flag_masks(self):
        """The flag_masks CF property.

        Provides a list of bit fields expressing Boolean or enumerated
        flags. See http://cfconventions.org/latest.html for details.

        Stored as a 1-d numpy array but may be set as array-like object.

        **Examples**

        >>> f.flag_masks = numpy.array([1, 2, 4], dtype='int8')
        >>> f.flag_masks
        array([1, 2, 4], dtype=int8)
        >>> f.flag_masks = (1, 2, 4, 8)
        >>> f.flag_masks
        array([1, 2, 4, 8], dtype=int8)
        >>> del f.flag_masks

        >>> f.set_property('flag_masks', 1)
        >>> f.get_property('flag_masks')
        array([1])
        >>> f.del_property('flag_masks')

        """
        try:
            return self.Flags.flag_masks
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_masks'"
            )

    @flag_masks.setter
    def flag_masks(self, value):
        try:
            flags = self.Flags
        except AttributeError:
            self.Flags = Flags(flag_masks=value)
        else:
            flags.flag_masks = value

    @flag_masks.deleter
    def flag_masks(self):
        try:
            del self.Flags.flag_masks
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_masks'"
            )
        else:
            if not self.Flags:
                del self.Flags

    @property
    def flag_meanings(self):
        """The flag_meanings CF property.

        Use in conjunction with `flag_values` to provide descriptive words
        or phrases for each flag value. If multi-word phrases are used to
        describe the flag values, then the words within a phrase should be
        connected with underscores. See
        http://cfconventions.org/latest.html for details.

        Stored as a 1-d numpy string array but may be set as a space
        delimited string or any array-like object.

        **Examples**

        >>> f.flag_meanings = 'low medium      high'
        >>> f.flag_meanings
        array(['low', 'medium', 'high'],
              dtype='|S6')
        >>> del flag_meanings

        >>> f.flag_meanings = ['left', 'right']
        >>> f.flag_meanings
        array(['left', 'right'],
              dtype='|S5')

        >>> f.flag_meanings = 'ok'
        >>> f.flag_meanings
        array(['ok'],
              dtype='|S2')

        >>> f.set_property('flag_meanings', numpy.array(['a', 'b']))
        >>> f.get_property('flag_meanings')
        array(['a', 'b'],
              dtype='|S1')
        >>> f.del_property('flag_meanings')

        """
        try:
            return " ".join(self.Flags.flag_meanings)
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_meanings'"
            )

    @flag_meanings.setter
    def flag_meanings(self, value):
        try:  # TODO deal with space-delimited strings
            flags = self.Flags
        except AttributeError:
            self.Flags = Flags(flag_meanings=value)
        else:
            flags.flag_meanings = value

    @flag_meanings.deleter
    def flag_meanings(self):
        try:
            del self.Flags.flag_meanings
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_meanings'"
            )
        else:
            if not self.Flags:
                del self.Flags

    @property
    def Conventions(self):
        """The Conventions CF property.

        The name of the conventions followed by the field. See
        http://cfconventions.org/latest.html for details.

        **Examples**

        >>> f.Conventions = 'CF-1.6'
        >>> f.Conventions
        'CF-1.6'
        >>> del f.Conventions

        >>> f.set_property('Conventions', 'CF-1.6')
        >>> f.get_property('Conventions')
        'CF-1.6'
        >>> f.del_property('Conventions')

        """
        return self.get_property("Conventions")

    @Conventions.setter
    def Conventions(self, value):
        self.set_property("Conventions", value, copy=False)

    @Conventions.deleter
    def Conventions(self):
        self.del_property("Conventions")

    @property
    def featureType(self):
        """The featureType CF property.

        The type of discrete sampling geometry, such as ``point`` or
        ``timeSeriesProfile``. See http://cfconventions.org/latest.html
        for details.

        .. versionadded:: 2.0

        **Examples**

        >>> f.featureType = 'trajectoryProfile'
        >>> f.featureType
        'trajectoryProfile'
        >>> del f.featureType

        >>> f.set_property('featureType', 'profile')
        >>> f.get_property('featureType')
        'profile'
        >>> f.del_property('featureType')

        """
        return self.get_property("featureType", default=AttributeError())

    @featureType.setter
    def featureType(self, value):
        self.set_property("featureType", value, copy=False)

    @featureType.deleter
    def featureType(self):
        self.del_property("featureType")

    @property
    def institution(self):
        """The institution CF property.

        Specifies where the original data was produced. See
        http://cfconventions.org/latest.html for details.

        **Examples**

        >>> f.institution = 'University of Reading'
        >>> f.institution
        'University of Reading'
        >>> del f.institution

        >>> f.set_property('institution', 'University of Reading')
        >>> f.get_property('institution')
        'University of Reading'
        >>> f.del_property('institution')

        """
        return self.get_property("institution")

    @institution.setter
    def institution(self, value):
        self.set_property("institution", value, copy=False)

    @institution.deleter
    def institution(self):
        self.del_property("institution")

    @property
    def references(self):
        """The references CF property.

        Published or web-based references that describe the data or
        methods used to produce it. See
        http://cfconventions.org/latest.html for details.

        **Examples**

        >>> f.references = 'some references'
        >>> f.references
        'some references'
        >>> del f.references

        >>> f.set_property('references', 'some references')
        >>> f.get_property('references')
        'some references'
        >>> f.del_property('references')

        """
        return self.get_property("references")

    @references.setter
    def references(self, value):
        self.set_property("references", value, copy=False)

    @references.deleter
    def references(self):
        self.del_property("references")

    @property
    def standard_error_multiplier(self):
        """The standard_error_multiplier CF property.

        If a data variable with a `standard_name` modifier of
        ``'standard_error'`` has this attribute, it indicates that the
        values are the stated multiple of one standard error. See
        http://cfconventions.org/latest.html for details.

        **Examples**

        >>> f.standard_error_multiplier = 2.0
        >>> f.standard_error_multiplier
        2.0
        >>> del f.standard_error_multiplier

        >>> f.set_property('standard_error_multiplier', 2.0)
        >>> f.get_property('standard_error_multiplier')
        2.0
        >>> f.del_property('standard_error_multiplier')

        """
        return self.get_property("standard_error_multiplier")

    @standard_error_multiplier.setter
    def standard_error_multiplier(self, value):
        self.set_property("standard_error_multiplier", value)

    @standard_error_multiplier.deleter
    def standard_error_multiplier(self):
        self.del_property("standard_error_multiplier")

    @property
    def source(self):
        """The source CF property.

        The method of production of the original data. If it was
        model-generated, `source` should name the model and its version,
        as specifically as could be useful. If it is observational,
        `source` should characterise it (for example, ``'surface
        observation'`` or ``'radiosonde'``). See
        http://cfconventions.org/latest.html for details.

        **Examples**

        >>> f.source = 'radiosonde'
        >>> f.source
        'radiosonde'
        >>> del f.source

        >>> f.set_property('source', 'surface observation')
        >>> f.get_property('source')
        'surface observation'
        >>> f.del_property('source')

        """
        return self.get_property("source")

    @source.setter
    def source(self, value):
        self.set_property("source", value, copy=False)

    @source.deleter
    def source(self):
        self.del_property("source")

    @property
    def title(self):
        """The title CF property.

        A short description of the file contents from which this field was
        read, or is to be written to. See
        http://cfconventions.org/latest.html for details.

        **Examples**

        >>> f.title = 'model data'
        >>> f.title
        'model data'
        >>> del f.title

        >>> f.set_property('title', 'model data')
        >>> f.get_property('title')
        'model data'
        >>> f.del_property('title')

        """
        return self.get_property("title")

    @title.setter
    def title(self, value):
        self.set_property("title", value, copy=False)

    @title.deleter
    def title(self):
        self.del_property("title")

    def cell_area(
        self,
        radius="earth",
        great_circle=False,
        cell_measures=True,
        coordinates=True,
        methods=False,
        return_cell_measure=False,
        insert=False,
        force=False,
    ):
        """Return the horizontal cell areas.

        .. versionadded:: 1.0

        .. seealso:: `bin`, `collapse`, `radius`, `weights`

        :Parameters:

            radius: optional
                Specify the radius used for calculating the areas of
                cells defined in spherical polar coordinates. The
                radius is that which would be returned by this call of
                the field construct's `~cf.Field.radius` method:
                ``f.radius(radius)``. See `cf.Field.radius` for
                details.

                By default *radius* is ``'earth'`` which means that if
                and only if the radius can not found from the datums
                of any coordinate reference constructs, then the
                default radius taken as 6371229 metres.

            great_circle: `bool`, optional
                If True then allow, if required, the derivation of i)
                area weights from polygon geometry cells by assuming
                that each cell part is a spherical polygon composed of
                great circle segments; and ii) and the derivation of
                line-length weights from line geometry cells by
                assuming that each line part is composed of great
                circle segments.

                .. versionadded:: 3.2.0

            cell_measures: `bool`, optional
                If True, the default, then area cell measure
                constructs are considered for cell area
                creation. Otherwise they are ignored.

                .. versionadded:: 3.16.1

            coordinates: `bool`, optional
                If True, the default, then coordinate constructs are
                considered for cell area creation. Otherwise they are
                ignored.

                .. versionadded:: 3.16.1

            methods: `bool`, optional
                If True, then return a dictionary describing the method
                used to create the cell areas instead of the default,
                a field construct.

                .. versionadded:: 3.16.1

            return_cell_measure: `bool`, optional
                If True, then return a cell measure construct instead
                of the default, a field construct.

                .. versionadded:: 3.16.1

            insert: deprecated at version 3.0.0

            force: deprecated at version 3.0.0

        :Returns:

            `Field`, `CellMeasure`, or `dict`
                A field construct, or cell measure construct containing
                the horizontal cell areas if *return_cell_measure* is True,
                or a dictionary describing the method used to create the
                cell areas if *methods* is True.

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

        >>> a = f.cell_area()
        >>> a
        <CF Field: cell_area(latitude(5), longitude(8)) m2>
        >>> print(a.array)
        [[4.27128714e+12 4.27128714e+12 4.27128714e+12 4.27128714e+12
          4.27128714e+12 4.27128714e+12 4.27128714e+12 4.27128714e+12]
         [1.16693735e+13 1.16693735e+13 1.16693735e+13 1.16693735e+13
          1.16693735e+13 1.16693735e+13 1.16693735e+13 1.16693735e+13]
         [3.18813213e+13 3.18813213e+13 3.18813213e+13 3.18813213e+13
          3.18813213e+13 3.18813213e+13 3.18813213e+13 3.18813213e+13]
         [1.16693735e+13 1.16693735e+13 1.16693735e+13 1.16693735e+13
          1.16693735e+13 1.16693735e+13 1.16693735e+13 1.16693735e+13]
         [4.27128714e+12 4.27128714e+12 4.27128714e+12 4.27128714e+12
          4.27128714e+12 4.27128714e+12 4.27128714e+12 4.27128714e+12]]
        >>> f.cell_area(methods=True)
        {(1,): 'linear longitude', (0,): 'linear sine latitude'}

        >>> a = f.cell_area(radius=cf.Data(3389.5, 'km'))

        >>> c = f.cell_area(return_cell_measure=True)
        >>> c
        <CF CellMeasure: measure:area(5, 8) m2>
        >>> f.set_construct(c)
        'cellmeasure0'
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        Cell measures   : measure:area(latitude(5), longitude(8)) = [[4271287143027.272, ..., 4271287143027.272]] m2
        >>> f.cell_area(methods=True)
        {(0, 1): 'area cell measure'}

        """
        if insert:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "cell_area",
                {"insert": insert},
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        if force:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "cell_area",
                {"force": force},
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        w = self.weights(
            "area",
            radius=radius,
            measure=True,
            scale=None,
            great_circle=great_circle,
            cell_measures=cell_measures,
            coordinates=coordinates,
            methods=methods,
        )

        if methods:
            if return_cell_measure:
                raise ValueError(
                    "Can't set both the 'methods' and 'return_cell_measure'"
                    "parameters."
                )
            return w

        if return_cell_measure:
            w = CellMeasure(source=w, copy=False)
            w.set_measure("area")
        else:
            w.set_property("standard_name", "cell_area", copy=False)

        return w

    def cfa_clear_file_substitutions(
        self,
    ):
        """Remove all of the CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Returns:

            `dict`
                {{Returns cfa_clear_file_substitutions}}

        **Examples**

        >>> f.cfa_clear_file_substitutions()
        {}

        """
        out = super().cfa_clear_file_substitution()

        for c in self.constructs.filter_by_data(todict=True).values():
            out.update(c.cfa_clear_file_substitutions())

        return out

    def cfa_del_file_substitution(
        self,
        base,
        constructs=True,
    ):
        """Remove a CFA-netCDF file name substitution.

        .. versionadded:: 3.15.0

        :Parameters:

            {{cfa base: `str`}}

            constructs: `bool`, optional
                If True (the default) then metadata constructs also
                have the file substitutions removed from them.

        :Returns:

            `dict`
                {{Returns cfa_del_file_substitution}}

        **Examples**

        >>> f.cfa_del_file_substitution('base')

        """
        super().cfa_del_file_substitution(base)

        if constructs:
            for c in self.constructs.filter_by_data(todict=True).values():
                c.cfa_del_file_substitution(base)

    def cfa_file_substitutions(self, constructs=True):
        """Return the CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Returns:

            `dict`
                {{Returns cfa_file_substitutions}}

        **Examples**

        >>> f.cfa_file_substitutions()
        {}

        """
        out = super().cfa_file_substitutions()

        if constructs:
            for c in self.constructs.filter_by_data(todict=True).values():
                out.update(c.cfa_file_substitutions())

        return out

    def del_file_location(
        self,
        location,
        constructs=True,
    ):
        """Remove a file location in-place.

        All data definitions that reference files will have references
        to files in the given location removed from them.

        .. versionadded:: 3.15.0

        .. seealso:: `add_file_location`, `file_locations`

        :Parameters:

            location: `str`
                 The file location to remove.

            constructs: `bool`, optional
                If True (the default) then metadata constructs also
                have the new file location removed from them.

        :Returns:

            `str`
                The removed location as an absolute path with no
                trailing path name component separator.

        **Examples**

        >>> d.del_file_location('/data/model/')
        '/data/model'

        """
        location = abspath(location).rstrip(sep)
        super().del_file_location(location)

        if constructs:
            for c in self.constructs.filter_by_data(todict=True).values():
                c.del_file_location(location)

        return location

    def cfa_update_file_substitutions(
        self,
        substitutions,
        constructs=True,
    ):
        """Set CFA-netCDF file name substitutions.

        .. versionadded:: 3.15.0

        :Parameters:

            {{cfa substitutions: `dict`}}

            constructs: `bool`, optional
                If True (the default) then metadata constructs also
                have the file substitutions set on them.

        :Returns:

            `None`

        **Examples**

        >>> f.cfa_update_file_substitutions({'base': '/data/model'})

        """
        super().cfa_update_file_substitutions(substitutions)

        if constructs:
            for c in self.constructs.filter_by_data(todict=True).values():
                c.cfa_update_file_substitutions(substitutions)

    def get_domain(self):
        """Return the domain.

        .. versionadded:: 3.16.2

        .. seealso:: `domain`

        :Returns:

            `Domain`
                 The domain.

        **Examples**

        >>> d = f.get_domain()

        """
        domain = super().get_domain()

        # Set axis cyclicity for the domain
        domain._cyclic = self._cyclic

        return domain

    def radius(self, default=None):
        """Return the radius of a latitude-longitude plane defined in
        spherical polar coordinates.

        The radius is taken from the datums of any coordinate
        reference constructs, but if and only if this is not possible
        then a default value may be used instead.

        .. versionadded:: 3.0.2

        .. seealso:: `bin`, `cell_area`, `collapse`, `weights`

        :Parameters:

            default: optional
                The radius is taken from the datums of any coordinate
                reference constructs, but if and only if this is not
                possible then the value set by the *default* parameter
                is used. May be set to any numeric scalar object,
                including `numpy` and `Data` objects. The units of the
                radius are assumed to be metres, unless specified by a
                `Data` object. If the special value ``'earth'`` is
                given then the default radius taken as 6371229
                metres. If *default* is `None` an exception will be
                raised if no unique datum can be found in the
                coordinate reference constructs.

                *Parameter example:*
                  Five equivalent ways to set a default radius of
                  6371200 metres: ``6371200``,
                  ``numpy.array(6371200)``, ``cf.Data(6371200)``,
                  ``cf.Data(6371200, 'm')``, ``cf.Data(6371.2,
                  'km')``.

        :Returns:

            `Data`
                The radius of the sphere, in units of metres.

        **Examples**

        >>> f.radius()
        <CF Data(): 6371178.98 m>

        >>> g.radius()
        ValueError: No radius found in coordinate reference constructs and no default provided
        >>> g.radius('earth')
        <CF Data(): 6371229.0 m>
        >>> g.radius(1234)
        <CF Data(): 1234.0 m>

        """
        radii = []
        for cr in self.coordinate_references(todict=True).values():
            r = cr.datum.get_parameter("earth_radius", None)
            if r is not None:
                r = Data.asdata(r)
                if not r.Units:
                    r.override_units("m", inplace=True)

                if r.size != 1:
                    radii.append(r)
                    continue

                got = False
                for _ in radii:
                    if r == _:
                        got = True
                        break

                if not got:
                    radii.append(r)

        if len(radii) > 1:
            raise ValueError(
                "Multiple radii found from coordinate reference "
                f"constructs: {radii!r}"
            )

        if not radii:
            if default is None:
                raise ValueError(
                    "No radius found from coordinate reference constructs "
                    "and no default provided"
                )

            if isinstance(default, str):
                if default != "earth":
                    raise ValueError(
                        "The default radius must be numeric, 'earth', "
                        "or None"
                    )

                return _earth_radius.copy()

            r = Data.asdata(default).squeeze()
        else:
            r = Data.asdata(radii[0]).squeeze()

        if r.size != 1:
            raise ValueError(f"Multiple radii: {r!r}")

        r.Units = Units("m")
        r.dtype = float
        return r

    def laplacian_xy(
        self, x_wrap=None, one_sided_at_boundary=False, radius=None
    ):
        r"""Calculate the Laplacian in X-Y coordinates.

        The horizontal Laplacian of a scalar function is calculated
        from a field that has dimension coordinates of X and Y, in
        either Cartesian (e.g. plane projection) or spherical polar
        coordinate systems.

        The horizontal Laplacian in Cartesian coordinates is given by:

        .. math:: \nabla^2 f(x, y) = \frac{\partial^2 f}{\partial x^2}
                                     +
                                     \frac{\partial^2 f}{\partial y^2}

        The horizontal Laplacian in spherical polar coordinates is
        given by:

        .. math:: \nabla^2 f(\theta, \phi) =
                    \frac{1}{r^2 \sin\theta}
                    \frac{\partial}{\partial \theta}
                    \left(
                    \sin\theta
                    \frac{\partial f}{\partial \theta}
                    \right)
                    +
                    \frac{1}{r^2 \sin^2\theta}
                    \frac{\partial^2 f}{\partial \phi^2}

        where *r* is radial distance to the origin, :math:`\theta` is
        the polar angle with respect to polar axis, and :math:`\phi`
        is the azimuthal angle.

        The Laplacian is calculated using centred finite differences
        apart from at the boundaries (see the *x_wrap* and
        *one_sided_at_boundary* parameters). If missing values are
        present then missing values will be returned at all points
        where a centred finite difference could not be calculated.

        .. versionadded:: 3.12.0

        .. seealso:: `derivative`, `grad_xy`, `iscyclic`,
                     `cf.curl_xy`, `cf.div_xy`

        :Parameters:

            x_wrap: `bool`, optional
                Whether the X axis is cyclic or not. By default
                *x_wrap* is set to the result of this call to the
                field construct's `iscyclic` method:
                ``f.iscyclic('X')``. If the X axis is cyclic then
                centred differences at one boundary will always use
                values from the other boundary, regardless of the
                setting of *one_sided_at_boundary*.

                The cyclicity of the Y axis is always set to the
                result of ``f.iscyclic('Y')``.

            one_sided_at_boundary: `bool`, optional
                If True then one-sided finite differences are
                calculated at the non-cyclic boundaries. By default
                missing values are set at non-cyclic boundaries.

            {{radius: optional}}

        :Returns:

            `Field` or `None`
                The horizontal Laplacian of the scalar field, or
                `None` if the operation was in-place.

        >>> f = cf.example_field(0)
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> f[...] = 0.1
        >>> print(f.array)
        [[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
         [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
         [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
         [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
         [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]]
        >>> lp = f.laplacian_xy(radius='earth')
        >>> lp
        <CF Field: long_name=X-Y Laplacian of specific_humidity(latitude(5), longitude(8)) m-2.rad-2>
        >>> print(lp.array)
        [[-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]]
        >>> lp = f.laplacian_xy(radius='earth', one_sided_at_boundary=True)
        >>> print(lp.array)
        [[0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]]

        """
        f = self.copy()
        identity = f.identity()

        x_key, x_coord = f.dimension_coordinate(
            "X", item=True, default=(None, None)
        )
        y_key, y_coord = f.dimension_coordinate(
            "Y", item=True, default=(None, None)
        )

        if x_coord is None:
            raise ValueError("Field has no unique 'X' dimension coordinate")

        if y_coord is None:
            raise ValueError("Field has no unique 'Y' dimension coordinate")

        if x_wrap is None:
            x_wrap = f.iscyclic(x_key)

        x_units = x_coord.Units
        y_units = y_coord.Units

        # Check for spherical polar coordinates
        latlon = (x_units.islongitude and y_units.islatitude) or (
            x_units.units == "degrees" and y_units.units == "degrees"
        )

        if latlon:
            # --------------------------------------------------------
            # Spherical polar coordinates
            # --------------------------------------------------------
            # Convert latitude and longitude units to radians, so that
            # the units of the result are nice.
            x_coord.Units = _units_radians
            y_coord.Units = _units_radians

            # Ensure that the lat and lon dimension coordinates have
            # standard names, so that metadata-aware broadcasting
            # works as expected when all of their units are radians.
            x_coord.standard_name = "longitude"
            y_coord.standard_name = "latitude"

            # Get theta as a field that will broadcast to f, and
            # adjust its values so that theta=0 is at the north pole.
            theta = np.pi / 2 - f.convert(y_key, full_domain=True)

            sin_theta = theta.sin()

            r = f.radius(default=radius)
            r2_sin_theta = sin_theta * r**2

            d2f_dphi2 = f.derivative(
                x_key, wrap=x_wrap, one_sided_at_boundary=one_sided_at_boundary
            ).derivative(
                x_key, wrap=x_wrap, one_sided_at_boundary=one_sided_at_boundary
            )

            term1 = d2f_dphi2 / (r2_sin_theta * sin_theta)

            df_dtheta = f.derivative(
                y_key, wrap=None, one_sided_at_boundary=one_sided_at_boundary
            )

            term2 = (df_dtheta * sin_theta).derivative(
                y_key, wrap=None, one_sided_at_boundary=one_sided_at_boundary
            ) / r2_sin_theta

            f = term1 + term2

            # Reset latitude and longitude coordinate units
            f.dimension_coordinate("longitude").Units = x_units
            f.dimension_coordinate("latitude").Units = y_units
        else:
            # --------------------------------------------------------
            # Cartesian coordinates
            # --------------------------------------------------------
            d2f_dx2 = f.derivative(
                x_key, wrap=x_wrap, one_sided_at_boundary=one_sided_at_boundary
            ).derivative(
                x_key, wrap=x_wrap, one_sided_at_boundary=one_sided_at_boundary
            )

            d2f_dy2 = f.derivative(
                y_key, wrap=None, one_sided_at_boundary=one_sided_at_boundary
            ).derivative(
                y_key, wrap=None, one_sided_at_boundary=one_sided_at_boundary
            )

            f = d2f_dx2 + d2f_dy2

        # Set the standard name and long name
        f.set_property("long_name", f"Horizontal Laplacian of {identity}")
        f.del_property("standard_name", None)

        return f

    def map_axes(self, other):
        """Map the axis identifiers of the field to their equivalent
        axis identifiers of another.

        :Parameters:

            other: `Field`

        :Returns:

            `dict`
                A dictionary whose keys are the axis identifiers of the
                field with corresponding values of axis identifiers of the
                of other field.

        **Examples**

        >>> f.map_axes(g)
        {'dim0': 'dim1',
         'dim1': 'dim0',
         'dim2': 'dim2'}

        """
        s = self.analyse_items()
        t = other.analyse_items()
        id_to_axis1 = t["id_to_axis"]

        out = {}
        for axis, identity in s["axis_to_id"].items():
            if identity in id_to_axis1:
                out[axis] = id_to_axis1[identity]

        return out

    def close(self):
        """Close all files referenced by the field construct.

        Deprecated at version 3.14.0. All files are now
        automatically closed when not being accessed.

        Note that a closed file will be automatically reopened if its
        contents are subsequently required.

        :Returns:

            `None`

        **Examples**

        >>> f.close()

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "close",
            "All files are now automatically closed when not being accessed.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def iscyclic(self, *identity, **filter_kwargs):
        """Returns True if the specified axis is cyclic.

        .. versionadded:: 1.0

        .. seealso:: `axis`, `cyclic`, `period`, `domain_axis`

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique domain axis construct returned by
                ``f.domain_axis(*identity, **filter_kwargs)``. See
                `domain_axis` for details.

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
            raise ValueError("Can't identify unique domain axis")

        return axis in self.cyclic()

    @classmethod
    def concatenate(
        cls, fields, axis=0, cull_graph=False, relaxed_units=False, copy=True
    ):
        """Join a sequence of fields together.

        This is different to `cf.aggregate` because it does not account
        for all metadata. For example, it assumes that the axis order is
        the same in each field.

        .. versionadded:: 1.0

        .. seealso:: `cf.aggregate`, `Data.concatenate`,
                     `Data.cull_graph`

        :Parameters:

            fields: (sequence of) `Field`
                The fields to concatenate.

            axis: `int`, optional
                The axis along which the arrays will be joined. The
                default is 0. Note that scalar arrays are treated as
                if they were one dimensional.

            {{cull_graph: `bool`, optional}}

                .. versionadded:: 3.14.0

            {{relaxed_units: `bool`, optional}}

                .. versionadded:: 3.15.1

            copy: `bool`, optional
                If True (the default) then make copies of the
                {{class}} constructs, prior to the concatenation,
                thereby ensuring that the input constructs are not
                changed by the concatenation process. If False then
                some or all input constructs might be changed
                in-place, but the concatenation process will be
                faster.

                .. versionadded:: 3.15.1

        :Returns:

            `Field`
                The field generated from the concatenation of input
                fields.

        """
        if isinstance(fields, cls):
            return fields.copy()

        field0 = fields[0]
        if copy:
            out = field0.copy()

        if len(fields) == 1:
            return out

        new_data = Data.concatenate(
            [f.get_data(_fill_value=False) for f in fields],
            axis=axis,
            cull_graph=cull_graph,
            relaxed_units=relaxed_units,
            copy=copy,
        )

        # Change the domain axis size
        dim = out.get_data_axes()[axis]
        out.set_construct(DomainAxis(size=new_data.shape[axis]), key=dim)

        # Insert the concatenated data
        out.set_data(new_data, set_axes=False, copy=False)

        # ------------------------------------------------------------
        # Concatenate constructs with data
        # ------------------------------------------------------------
        for key, construct in field0.constructs.filter_by_data(
            todict=True
        ).items():
            construct_axes = field0.get_data_axes(key)

            if dim not in construct_axes:
                # This construct does not span the concatenating axis
                # in the first field
                continue

            constructs = [construct]
            for f in fields[1:]:
                c = f.constructs.get(key)
                if c is None:
                    # This field does not have this construct
                    constructs = None
                    break

                constructs.append(c)

            if not constructs:
                # Not every field has this construct, so remove it
                # from the output field.
                out.del_construct(key)
                continue

            # Still here? Then try concatenating the constructs from
            # each field.
            try:
                construct = construct.concatenate(
                    constructs,
                    axis=construct_axes.index(dim),
                    cull_graph=cull_graph,
                    relaxed_units=relaxed_units,
                    copy=copy,
                )
            except ValueError:
                # Couldn't concatenate this construct, so remove it from
                # the output field.
                out.del_construct(key)
            else:
                # Successfully concatenated this construct, so insert
                # it into the output field.
                out.set_construct(
                    construct, key=key, axes=construct_axes, copy=False
                )

        return out

    def weights(
        self,
        weights=True,
        scale=None,
        measure=False,
        components=False,
        methods=False,
        radius="earth",
        data=False,
        great_circle=False,
        axes=None,
        cell_measures=True,
        coordinates=True,
        **kwargs,
    ):
        """Return weights for the data array values.

        The weights are those used during a statistical collapse of
        the data. For example when computing an area weight average.

        Weights for any combination of axes may be returned.

        Weights are either derived from the field construct's metadata
        (such as coordinate cell sizes) or provided explicitly in the
        form of other `Field` constructs. In any case, the outer
        product of these weights components is returned in a field
        which is broadcastable to the original field (see the
        *components* parameter for returning the components
        individually).

        By default null, equal weights are returned.

        .. versionadded:: 1.0

        .. seealso:: `bin`, `cell_area`, `collapse`, `moving_window`,
                     `radius`

        :Parameters:

            weights: *optional*
                Specify the weights to be created. There are three
                distinct methods:

                * **Type 1** will create weights for all axes of size
                  greater than 1, raising an exception if this is not
                  possible (this is the default).;

                * **Type 2** will always succeed in creating weights
                  for all axes of the field, even if some of those
                  weights are null.

                * **Type 3** allows particular types of weights to be
                  defined for particular axes, and an exception will
                  be raised if it is not possible to create the
                  weights.

            ..

                **Type 1** and **Type 2** come at the expense of not
                always being able to control exactly how the weights
                are created (although which methods were used can be
                inspected with use of the *methods* parameter).

                * **Type 1**: *weights* may be:

                  ==========  ========================================
                  *weights*   Description
                  ==========  ========================================
                  `True`      This is the default. Weights are created
                              for all axes (or a subset of them, see
                              the *axes* parameter). Set the *methods*
                              parameter to find out how the weights
                              were actually created.

                              The weights components are created for
                              axes of the field by one or more of the
                              following methods, in order of
                              preference,

                              If the *cell_measures* parameter is
                              True:

                                1. Volume cell measures (see the note
                                   on the *measure* parameter)

                                2. Area cell measures

                              If the *coordinates* parameter is True:

                                3. Area calculated from X and Y
                                   dimension coordinate constructs
                                   with bounds

                                4. Area calculated from 1-d auxiliary
                                   coordinate constructs for
                                   geometries or a UGRID mesh
                                   topology.

                                5. Length calculated from 1-d
                                   auxiliary coordinate constructs for
                                   geometries or a UGRID mesh
                                   topology.

                                6. Cell sizes of dimension coordinate
                                   constructs with bounds

                                7. Equal weights

                              and the outer product of these weights
                              components is returned in a field
                              constructs which is broadcastable to the
                              original field construct (see the
                              *components* parameter).
                  ==========  ========================================

                * **Type 2**: *weights* may be one of:

                  ==========  ========================================
                  *weights*   Description
                  ==========  ========================================
                  `None`      Equal weights for all axes.

                  `False`     Equal weights for all axes.

                  `Data`      Explicit weights in a `Data` object that
                              must be broadcastable to the field
                              construct's data, unless the *axes*
                              parameter is also set.

                  `dict`      Explicit weights in a dictionary of the
                              form that is returned from a call to the
                              `weights` method with ``component=True``
                  ==========  ========================================

                * **Type 3**: *weights* may be one, or a sequence, of:

                  ============  ==========================================
                  *weights*     Description
                  ============  ==========================================
                  ``'area'``    Cell area weights. The weights
                                components are created for axes of the
                                field by the following methods, in
                                order of preference,

                                If the *cell_measures* parameter is
                                True:

                                  1. Area cell measures.

                                If the *coordinates* parameter is
                                True:

                                  2. X and Y dimension coordinate
                                     constructs with bounds.

                                  3. X and Y 1-d auxiliary coordinate
                                     constructs for polygon cells
                                     defined by geometries or a UGRID
                                     mesh topology.

                                Set the *methods* parameter to find
                                out how the weights were actually
                                created.

                  ``'volume'``  Cell volume weights from the field
                                construct's volume cell measure
                                construct (see the note on the
                                *measure* parameter). Requires the
                                *cell_measures* parameter to be True.

                  `str`         Weights from the cell sizes of the
                                dimension coordinate construct with this
                                identity.

                  `Field`       Explicit weights from the data of another
                                field construct, which must be
                                broadcastable to this field construct.
                  ============  ==========================================

                If *weights* is a sequence of any combination of the
                above then the returned field contains the outer
                product of the weights defined by each element of the
                sequence. The ordering of the sequence is irrelevant.

                *Parameter example:*
                  To create to 2-dimensional weights based on cell
                  areas: ``f.weights('area')``. To create to
                  3-dimensional weights based on cell areas and linear
                  height: ``f.weights(['area', 'Z'])``.

            scale: number, optional
                If set to a positive number then scale the weights so
                that they are less than or equal to that number. If
                weights components have been requested (see the
                *components* parameter) then each component is scaled
                independently of the others.

                *Parameter example:*
                  To scale all weights so that they lie between 0 and
                  1: ``scale=1``.

            measure: `bool`, optional
                Create weights that are cell measures, i.e. which
                describe actual cell sizes (e.g. cell areas) with
                appropriate units (e.g. metres squared).

                Cell measures can be created for any combination of
                axes. For example, cell measures for a time axis are
                the time span for each cell with canonical units of
                seconds; cell measures for the combination of four
                axes representing time and three dimensional space
                could have canonical units of metres cubed seconds.

                .. note:: Specifying cell volume weights via
                          ``weights=['X', 'Y', 'Z']`` or
                          ``weights=['area', 'Z']`` (or other
                          equivalents) will produce **an incorrect
                          result if the vertical dimension coordinates
                          do not define the actual height or depth
                          thickness of every cell in the domain**. In
                          this case, ``weights='volume'`` should be
                          used instead, which requires the field
                          construct to have a "volume" cell measure
                          construct.

                          If ``weights=True`` then care also needs to
                          be taken, as a "volume" cell measure
                          construct will be used if present, otherwise
                          the cell volumes will be calculated using
                          the size of the vertical coordinate cells.

            {{radius: optional}}

            components: `bool`, optional
                If True then a dictionary of orthogonal weights
                components is returned instead of a field. Each key is
                a tuple of integers representing axis positions in the
                field construct's data, with corresponding values of
                weights in `Data` objects. The axes of weights match
                the axes of the field construct's data array in the
                order given by their dictionary keys.

            methods: `bool`, optional
                If True, then return a dictionary describing methods
                used to create the weights.

            data: `bool`, optional
                If True then return the weights in a `Data` instance
                that is broadcastable to the original data.

                .. versionadded:: 3.1.0

            great_circle: `bool`, optional
                If True then allow, if required, the derivation of i)
                area weights from polygon cells by assuming that each
                cell part is a spherical polygon composed of great
                circle segments; and ii) the derivation of
                line-length weights line cells by assuming that each
                line part is composed of great circle segments. Only
                applies to geometry and UGRID cells.

                .. versionadded:: 3.2.0

            axes: (sequence of) `int` or `str`, optional
                Modify the behaviour when *weights* is `True` or a
                `Data` instance. Ignored for any other value the
                *weights* parameter.

                If *weights* is `True` then weights are created only
                for the specified axes (as opposed to all
                axes). I.e. ``weight=True, axes=axes`` is identical to
                ``weights=axes``.

                If *weights* is a `Data` instance then the specified
                axes identify each dimension of the given weights. If
                the weights do not broadcast to the field construct's
                data then setting the *axes* parameter is required so
                that the broadcasting can be inferred, otherwise
                setting the *axes* is not required.

                *Parameter example:*
                  ``axes='T'``

                *Parameter example:*
                  ``axes=['longitude']``

                *Parameter example:*
                  ``axes=[3, 1]``

                .. versionadded:: 3.3.0

            cell_measures: `bool`, optional
                If True, the default, then area and volume cell
                measure constructs are considered for weights creation
                when *weights* is `True`, ``'area'``, or
                ``'volume'``. If False then cell measure constructs
                are ignored for these *weights*.

                .. versionadded:: 3.16.1

            coordinates: `bool`, optional
                If True, the default, then coordinate constructs are
                considered for weights creation for *weights* of
                `True` or ``'area'``. If False then coordinate
                constructs are ignored for these *weights*.

                .. versionadded:: 3.16.1

            kwargs: deprecated at version 3.0.0.

        :Returns:

            `Field` or `Data` or `dict`
                The weights field; or if *data* is True, weights data
                in broadcastable form; or if *components* is True,
                orthogonal weights in a dictionary.

        **Examples**

        >>> f
        <CF Field: air_temperature(time(12), latitude(145), longitude(192)) K>
        >>> f.weights()
        <CF Field: long_name:weight(time(12), latitude(145), longitude(192)) 86400 s.rad>
        >>> f.weights(scale=1.0)
        <CF Field: long_name:weight(time(12), latitude(145), longitude(192)) 1>
        >>> f.weights(components=True)
        {(0,): <CF Data(12): [30.0, ..., 31.0] d>,
         (1,): <CF Data(145): [5.94949998503e-05, ..., 5.94949998503e-05]>,
         (2,): <CF Data(192): [0.0327249234749, ..., 0.0327249234749] radians>}
        >>> f.weights(components=True, scale=1.0)
        {(0,): <CF Data(12): [0.967741935483871, ..., 1.0] 1>,
         (1,): <CF Data(145): [0.00272710399807, ..., 0.00272710399807]>,
         (2,): <CF Data(192): [1.0, ..., 1.0]>}
        >>> f.weights(components=True, scale=2.0)
        {(0,): <CF Data(12): [1.935483870967742, ..., 2.0] 1>,
         (1,): <CF Data(145): [0.00545420799614, ..., 0.00545420799614]>,
         (2,): <CF Data(192): [2.0, ..., 2.0]>}
        >>> f.weights(methods=True)
        {(0,): 'linear time',
         (1,): 'linear sine latitude',
         (2,): 'linear longitude'}

        """
        from .weights import Weights

        if isinstance(weights, str) and weights == "auto":
            _DEPRECATION_ERROR_KWARG_VALUE(
                self,
                "weights",
                "weights",
                "auto",
                message="Use value True instead.",
                version="3.0.7",
                removed_at="4.0.0",
            )  # pragma: no cover

        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "weights", kwargs, version="3.0.0", removed_at="4.0.0"
            )  # pragma: no cover

        if measure and scale is not None:
            raise ValueError("Can't set measure=True and scale")

        if components and data:
            raise ValueError("Can't set components=True and data=True")

        if weights is None or weights is False:
            # --------------------------------------------------------
            # All equal weights
            # --------------------------------------------------------
            if components or methods:
                # Return an empty components dictionary
                return {}

            if data:
                # Return an scalar Data instance
                return Data(1.0, "1")

            # Return a field containing a single weight of 1
            return Weights.field_scalar(self)

        # Still here?
        if methods:
            components = True

        comp = {}
        field_data_axes = self.get_data_axes()

        # All axes which have weights
        weights_axes = set()

        if weights is True and axes is not None:
            # --------------------------------------------------------
            # Restrict weights to the specified axes
            # --------------------------------------------------------
            weights = axes

        if weights is True:
            # --------------------------------------------------------
            # Auto-detect all weights
            # --------------------------------------------------------
            # Volume weights
            if cell_measures and Weights.cell_measure(
                self, "volume", comp, weights_axes, methods=methods, auto=True
            ):
                # Found volume weights from cell measures
                pass

            elif cell_measures and Weights.cell_measure(
                self,
                "area",
                comp,
                weights_axes,
                methods=methods,
                auto=True,
            ):
                # Found area weights from cell measures
                pass
            elif coordinates and Weights.area_XY(
                self,
                comp,
                weights_axes,
                measure=measure,
                radius=radius,
                methods=methods,
                auto=True,
            ):
                # Found area weights from X and Y dimension
                # coordinates
                pass

            domain_axes = self.domain_axes(todict=True)

            if coordinates:
                for da_key in domain_axes:
                    if Weights.polygon_area(
                        self,
                        da_key,
                        comp,
                        weights_axes,
                        measure=measure,
                        radius=radius,
                        great_circle=great_circle,
                        methods=methods,
                        auto=True,
                    ):
                        # Found area weights from polygon geometries
                        pass
                    elif Weights.line_length(
                        self,
                        da_key,
                        comp,
                        weights_axes,
                        measure=measure,
                        radius=radius,
                        great_circle=great_circle,
                        methods=methods,
                        auto=True,
                    ):
                        # Found linear weights from line geometries
                        pass
                    elif Weights.linear(
                        self,
                        da_key,
                        comp,
                        weights_axes,
                        measure=measure,
                        methods=methods,
                        auto=True,
                    ):
                        # Found linear weights from dimension coordinates
                        pass

            weights_axes = []
            for key in comp:
                weights_axes.extend(key)

            size_N_axes = []
            for key, domain_axis in domain_axes.items():
                if domain_axis.get_size(0) > 1:
                    size_N_axes.append(key)

            missing_axes = set(size_N_axes).difference(weights_axes)
            if missing_axes:
                missing_axes_id = self.constructs.domain_axis_identity(
                    missing_axes.pop()
                )
                raise ValueError(
                    f"Can't find weights for {missing_axes_id!r} axis."
                )

        elif isinstance(weights, dict):
            # --------------------------------------------------------
            # Dictionary
            # --------------------------------------------------------
            for key, value in weights.items():
                key = [self.domain_axis(i, key=True) for i in key]
                for k in key:
                    if k not in field_data_axes:
                        raise ValueError(
                            f"Can't find weights: {k!r} domain axis does "
                            "not correspond to one of the data axes."
                        )

                multiple_weights = weights_axes.intersection(key)
                if multiple_weights:
                    multiple_weights_id = self.constructs.domain_axis_identity(
                        multiple_weights.pop()
                    )
                    raise ValueError(
                        f"Can't find weights: Multiple specifications for "
                        f"{multiple_weights_id!r} domain axis"
                    )

                weights_axes.update(key)

                if methods:
                    comp[tuple(key)] = "custom data"
                else:
                    comp[tuple(key)] = value.copy()

        elif isinstance(weights, self.__class__):
            # --------------------------------------------------------
            # Field
            # --------------------------------------------------------
            Weights.field(self, [weights], comp, weights_axes)

        elif isinstance(weights, Data):
            # --------------------------------------------------------
            # Data
            # --------------------------------------------------------
            Weights.data(
                self,
                weights,
                comp,
                weights_axes,
                axes=axes,
                data=data,
                components=components,
                methods=methods,
            )
        else:
            # --------------------------------------------------------
            # String or sequence
            # --------------------------------------------------------
            fields = []
            axes = []
            measures = []

            if isinstance(weights, str):
                if weights in ("area", "volume"):
                    measures = (weights,)
                else:
                    axes.append(weights)
            else:
                # In rare edge cases (e.g. if a user sets
                # `weights=f[0].cell_area` when they really meant
                # `weights=f[0].cell_area()`) we reach this code but
                # find that weights is not iterable. So check it is.
                try:
                    weights = iter(weights)
                except TypeError:
                    raise TypeError(
                        f"Invalid type of 'weights' parameter: {weights!r}"
                    )

                for w in tuple(weights):
                    if isinstance(w, self.__class__):
                        fields.append(w)
                    elif isinstance(w, Data):
                        raise ValueError(
                            f"Invalid weight {w!r} in sequence of weights."
                        )
                    elif w in ("area", "volume"):
                        measures.append(w)
                    else:
                        axes.append(w)

            # Field weights
            Weights.field(self, fields, comp, weights_axes)

            # Volume weights
            if "volume" in measures:
                if not cell_measures:
                    raise ValueError(
                        "Can't create weights: Unable to use  "
                        "volume cell measures when cell_meaures=False"
                    )

                Weights.cell_measure(
                    self,
                    "volume",
                    comp,
                    weights_axes,
                    methods=methods,
                    auto=False,
                )

            # Area weights
            if "area" in measures:
                area_weights = False
                if cell_measures:
                    if Weights.cell_measure(
                        self,
                        "area",
                        comp,
                        weights_axes,
                        methods=methods,
                        auto=True,
                    ):
                        # Found area weights from cell measures
                        area_weights = True

                if not area_weights and coordinates:
                    if Weights.area_XY(
                        self,
                        comp,
                        weights_axes,
                        measure=measure,
                        radius=radius,
                        methods=methods,
                        auto=True,
                    ):
                        # Found area weights from X and Y dimension
                        # coordinates
                        area_weights = True
                    elif Weights.polygon_area(
                        self,
                        None,
                        comp,
                        weights_axes,
                        measure=measure,
                        radius=radius,
                        great_circle=great_circle,
                        methods=methods,
                        auto=True,
                    ):
                        # Found area weights from UGRID/geometry cells
                        area_weights = True

                if not area_weights:
                    raise ValueError(
                        "Can't create weights: Unable to find cell areas"
                    )

            for axis in axes:
                da_key = self.domain_axis(axis, key=True, default=None)
                if da_key is None:
                    raise ValueError(
                        "Can't create weights: can't find axis matching "
                        f"{axis!r}"
                    )

                if Weights.polygon_area(
                    self,
                    da_key,
                    comp,
                    weights_axes,
                    measure=measure,
                    radius=radius,
                    great_circle=great_circle,
                    methods=methods,
                    auto=True,
                ):
                    # Found area weights from polygon geometries
                    pass
                elif Weights.line_length(
                    self,
                    da_key,
                    comp,
                    weights_axes,
                    measure=measure,
                    radius=radius,
                    great_circle=great_circle,
                    methods=methods,
                    auto=True,
                ):
                    # Found linear weights from line geometries
                    pass
                else:
                    Weights.linear(
                        self,
                        da_key,
                        comp,
                        weights_axes,
                        measure=measure,
                        methods=methods,
                        auto=False,
                    )

                # Check for area weights specified by X and Y axes
                # separately and replace them with area weights
                xaxis = self.domain_axis("X", key=True, default=None)
                yaxis = self.domain_axis("Y", key=True, default=None)
                if xaxis != yaxis and (xaxis,) in comp and (yaxis,) in comp:
                    del comp[(xaxis,)]
                    del comp[(yaxis,)]
                    weights_axes.discard(xaxis)
                    weights_axes.discard(yaxis)
                    if not Weights.cell_measure(
                        self, "area", comp, weights_axes, methods=methods
                    ):
                        Weights.area_XY(
                            self,
                            comp,
                            weights_axes,
                            measure=measure,
                            radius=radius,
                            methods=methods,
                        )

        if not methods:
            if scale is not None:
                # ----------------------------------------------------
                # Scale the weights so that they are <= scale
                # ----------------------------------------------------
                for key, w in comp.items():
                    comp[key] = Weights.scale(w, scale)

            for w in comp.values():
                if not measure:
                    w.override_units("1", inplace=True)

        if components or methods:
            # --------------------------------------------------------
            # Return a dictionary of component weights, which may be
            # empty.
            # --------------------------------------------------------
            components = {}
            for key, v in comp.items():
                key = [field_data_axes.index(axis) for axis in key]
                if not key:
                    continue

                components[tuple(key)] = v

            return components

        # Still here?
        if not comp:
            # --------------------------------------------------------
            # No component weights have been defined so return an
            # equal weights field
            # --------------------------------------------------------
            f = Weights.field_scalar(self)
            if data:
                return f.data

            return f

        # ------------------------------------------------------------
        # Still here? Return a weights field which is the outer
        # product of the component weights
        # ------------------------------------------------------------
        pp = sorted(comp.items())
        waxes, wdata = pp.pop(0)
        while pp:
            a, y = pp.pop(0)
            wdata.outerproduct(y, inplace=True)
            waxes += a

        if scale is not None:
            # --------------------------------------------------------
            # Scale the weights so that they are <= scale
            # --------------------------------------------------------
            wdata = Weights.scale(wdata, scale)

        # ------------------------------------------------------------
        # Reorder the data so that its dimensions are in the same
        # relative order as self
        # ------------------------------------------------------------
        transpose = [
            waxes.index(axis) for axis in self.get_data_axes() if axis in waxes
        ]
        wdata = wdata.transpose(transpose)
        waxes = [waxes[i] for i in transpose]

        # Set cyclicity
        for axis in self.get_data_axes():
            if axis in waxes and self.iscyclic(axis):
                wdata.cyclic(waxes.index(axis), iscyclic=True)

        if data:
            # Insert missing size one dimensions for broadcasting
            for i, axis in enumerate(self.get_data_axes()):
                if axis not in waxes:
                    waxes.insert(i, axis)
                    wdata.insert_dimension(i, inplace=True)

            return wdata

        field = self.copy()
        field.nc_del_variable(None)
        field.del_data()
        field.del_data_axes()

        not_needed_axes = set(field.domain_axes(todict=True)).difference(
            weights_axes
        )

        for key in self.cell_methods(todict=True).copy():
            field.del_construct(key)

        for key in self.field_ancillaries(todict=True).copy():
            field.del_construct(key)

        for key in field.coordinate_references(todict=True).copy():
            if field.coordinate_reference_domain_axes(key).intersection(
                not_needed_axes
            ):
                field.del_coordinate_reference(key)

        for key in field.constructs.filter_by_axis(
            *not_needed_axes, axis_mode="or", todict=True
        ):
            field.del_construct(key)

        for key in not_needed_axes:
            field.del_construct(key)

        field.set_data(wdata, axes=waxes, copy=False)
        field.clear_properties()
        field.long_name = "weights"

        return field

    @_inplace_enabled(default=False)
    def digitize(
        self,
        bins,
        upper=False,
        open_ends=False,
        closed_ends=None,
        return_bins=False,
        inplace=False,
    ):
        """Return the indices of the bins to which each value belongs.

        Values (including masked values) that do not belong to any bin
        result in masked values in the output field construct of indices.

        Bins defined by percentiles are easily created with the
        `percentile` method

        *Example*:
          Find the indices for bins defined by the 10th, 50th and 90th
          percentiles:

          >>> bins = f.percentile([0, 10, 50, 90, 100], squeeze=True)
          >>> i = f.digitize(bins, closed_ends=True)

        The output field construct is given a ``long_name`` property, and
        some or all of the following properties that define the bins:

        =====================  ===========================================
        Property               Description
        =====================  ===========================================
        ``bin_count``          An integer giving the number of bins

        ``bin_bounds``         A 1-d vector giving the bin bounds. The
                               first two numbers describe the lower and
                               upper boundaries of the first bin, the
                               second two numbers describe the lower and
                               upper boundaries of the second bin, and so
                               on. The presence of left-unbounded and
                               right-unbounded bins (see the *bins* and
                               *open_ends* parameters) is deduced from the
                               ``bin_count`` property. If the
                               ``bin_bounds`` vector has 2N elements then
                               the ``bin_count`` property will be N+2 if
                               there are left-unbounded and
                               right-unbounded bins, or N if no such bins
                               are present.

        ``bin_interval_type``  A string that specifies the nature of the
                               bin boundaries, i.e. if they are closed or
                               open. For example, if the lower boundary is
                               closed and the upper boundary is open
                               (which is the case when the *upper*
                               parameter is False) then
                               ``bin_interval_type`` will have the value
                               ``'lower: closed upper: open'``.

        ``bin_units``          A string giving the units of the bin
                               boundary values (e.g. ``'Kelvin'``). If the
                               *bins* parameter is a `Data` object with
                               units then these are used to set this
                               property, otherwise the field construct's
                               units are used.

        ``bin_calendar``       A string giving the calendar of reference
                               date-time units for the bin boundary values
                               (e.g. ``'noleap'``). If the units are not
                               reference date-time units this property
                               will be omitted. If the calendar is the CF
                               default calendar, then this property may be
                               omitted. If the *bins* parameter is a
                               `Data` object with a calendar then this is
                               used to set this property, otherwise the
                               field construct's calendar is used.

        ``bin_standard_name``  A string giving the standard name of the
                               bin boundaries
                               (e.g. ``'air_temperature'``). If there is
                               no standard name then this property will be
                               omitted.

        ``bin_long_name``      A string giving the long name of the bin
                               boundaries (e.g. ``'Air Temperature'``). If
                               there is no long name, or the
                               ``bin_standard_name`` is present, then this
                               property will be omitted.
        =====================  ===========================================

        Of these properties, the ``bin_count`` and ``bin_bounds`` are
        guaranteed to be output, with the others being dependent on the
        available metadata.

        .. versionadded:: 3.0.2

        .. seealso:: `bin`, `histogram`, `percentile`

        :Parameters:

            bins: array_like
                The bin boundaries. One of:

                * An integer

                  Create this many equally sized, contiguous bins spanning
                  the range of the data. I.e. the smallest bin boundary is
                  the minimum of the data and the largest bin boundary is
                  the maximum of the data. In order to guarantee that each
                  data value lies inside a bin, the *closed_ends*
                  parameter is assumed to be True.

                * A 1-d array

                  When sorted into a monotonically increasing sequence,
                  each boundary, with the exception of the two end
                  boundaries, counts as the upper boundary of one bin and
                  the lower boundary of next. If the *open_ends* parameter
                  is True then the lowest lower bin boundary also defines
                  a left-unbounded (i.e. not bounded below) bin, and the
                  largest upper bin boundary also defines a
                  right-unbounded (i.e. not bounded above) bin.

                * A 2-d array

                  The second dimension, that must have size 2, contains
                  the lower and upper boundaries of each bin. The bins to
                  not have to be contiguous, but must not overlap. If the
                  *open_ends* parameter is True then the lowest lower bin
                  boundary also defines a left-unbounded (i.e. not bounded
                  below) bin, and the largest upper bin boundary also
                  defines a right-unbounded (i.e. not bounded above) bin.

            upper: `bool`, optional
                If True then each bin includes its upper bound but not its
                lower bound. By default the opposite is applied, i.e. each
                bin includes its lower bound but not its upper bound.

            open_ends: `bool`, optional
                If True then create left-unbounded (i.e. not bounded
                below) and right-unbounded (i.e. not bounded above) bins
                from the lowest lower bin boundary and largest upper bin
                boundary respectively. By default these bins are not
                created

            closed_ends: `bool`, optional
                If True then extend the most extreme open boundary by a
                small amount so that its bin includes values that are
                equal to the unadjusted boundary value. This is done by
                multiplying it by ``1.0 - epsilon`` or ``1.0 + epsilon``,
                whichever extends the boundary in the appropriate
                direction, where ``epsilon`` is the smallest positive
                64-bit float such that ``1.0 + epsilson != 1.0``. I.e. if
                *upper* is False then the largest upper bin boundary is
                made slightly larger and if *upper* is True then the
                lowest lower bin boundary is made slightly lower.

                By default *closed_ends* is assumed to be True if *bins*
                is a scalar and False otherwise.

            return_bins: `bool`, optional
                If True then also return the bins in their 2-d form.

            {{inplace: `bool`, optional}}

        :Returns:

            `Field` or `None`, [`Data`]
                The field construct containing indices of the bins to
                which each value belongs, or `None` if the operation was
                in-place.

                If *return_bins* is True then also return the bins in
                their 2-d form.

        **Examples**

        >>> f = cf.example_field(0)
        >>> f
        <CF Field: specific_humidity(latitude(5), longitude(8)) 0.001 1>
        >>> f.properties()
        {'Conventions': 'CF-1.7',
         'standard_name': 'specific_humidity',
         'units': '0.001 1'}
        >>> print(f.array)
        [[  7.  34.   3.  14.  18.  37.  24.  29.]
         [ 23.  36.  45.  62.  46.  73.   6.  66.]
         [110. 131. 124. 146.  87. 103.  57.  11.]
         [ 29.  59.  39.  70.  58.  72.   9.  17.]
         [  6.  36.  19.  35.  18.  37.  34.  13.]]
        >>> g = f.digitize([0, 50, 100, 150])
        >>> g
        <CF Field: long_name=Bin index to which each 'specific_humidity' value belongs(latitude(5), longitude(8))>
        >>> print(g.array)
        [[0 0 0 0 0 0 0 0]
         [0 0 0 1 0 1 0 1]
         [2 2 2 2 1 2 1 0]
         [0 1 0 1 1 1 0 0]
         [0 0 0 0 0 0 0 0]]
        >>> g.properties()
        {'Conventions': 'CF-1.7',
         'long_name': "Bin index to which each 'specific_humidity' value belongs",
         'bin_bounds': array([  0,  50,  50, 100, 100, 150]),
         'bin_count': 3,
         'bin_interval_type': 'lower: closed upper: open',
         'bin_standard_name': 'specific_humidity',
         'bin_units': '0.001 1'}

        >>> g = f.digitize([[10, 20], [40, 60], [100, 140]])
        >>> print(g.array)
        [[-- -- --  0  0 -- -- --]
         [-- --  1 --  1 -- -- --]
         [ 2  2  2 -- --  2  1  0]
         [--  1 -- --  1 -- --  0]
         [-- --  0 --  0 -- --  0]]
        >>> g.properties()
        {'Conventions': 'CF-1.7',
         'long_name': "Bin index to which each 'specific_humidity' value belongs",
         'bin_bounds': array([ 10,  20,  40,  60, 100, 140]),
         'bin_count': 3,
         'bin_interval_type': 'lower: closed upper: open',
         'bin_standard_name': 'specific_humidity',
         'bin_units': '0.001 1'}

        >>> g = f.digitize([[10, 20], [40, 60], [100, 140]], open_ends=True)
        >>> print(g.array)
        [[ 0 --  0  1  1 -- -- --]
         [-- --  2 --  2 --  0 --]
         [ 3  3  3 -- --  3  2  1]
         [--  2 -- --  2 --  0  1]
         [ 0 --  1 --  1 -- --  1]]
        >>> g.properties()
        {'Conventions': 'CF-1.7',
         'long_name': "Bin index to which each 'specific_humidity' value belongs",
         'bin_bounds': array([ 10,  20,  40,  60, 100, 140]),
         'bin_count': 5,
         'bin_interval_type': 'lower: closed upper: open',
         'bin_standard_name': 'specific_humidity',
         'bin_units': '0.001 1'}

        >>> g = f.digitize([2, 6, 45, 100], upper=True)
        >>> g
        <CF Field: long_name=Bin index to which each 'specific_humidity' value belongs(latitude(5), longitude(8))>
        >>> print(g.array)
        [[ 1  1  0  1  1  1  1  1]
         [ 1  1  1  2  2  2  0  2]
         [-- -- -- --  2 --  2  1]
         [ 1  2  1  2  2  2  1  1]
         [ 0  1  1  1  1  1  1  1]]
        >>> g.properties()
        {'Conventions': 'CF-1.7',
         'long_name': "Bin index to which each 'specific_humidity' value belongs",
         'bin_bounds': array([  2,   6,   6,  45,  45, 100]),
         'bin_count': 3,
         'bin_interval_type': 'lower: open upper: closed',
         'bin_standard_name': 'specific_humidity',
         'bin_units': '0.001 1'}

        >>> g, bins = f.digitize(10, return_bins=True)
        >>> bins
        <CF Data(10, 2): [[3.0, ..., 146.00000000000003]] 0.001 1>
        >>> g, bins = f.digitize(10, upper=True, return_bins=True)
        <CF Data(10, 2): [[2.999999999999999, ..., 146.0]] 0.001 1>
        >>> print(g.array)
        [[0 2 0 0 1 2 1 1]
         [1 2 2 4 3 4 0 4]
         [7 8 8 9 5 6 3 0]
         [1 3 2 4 3 4 0 0]
         [0 2 1 2 1 2 2 0]]

        >>> f[1, [2, 5]] = cf.masked
        >>> print(f.array)
        [[  7.  34.   3.  14.  18.  37.  24.  29.]
         [ 23.  36.   --  62.  46.   --   6.  66.]
         [110. 131. 124. 146.  87. 103.  57.  11.]
         [ 29.  59.  39.  70.  58.  72.   9.  17.]
         [  6.  36.  19.  35.  18.  37.  34.  13.]]
        >>> g = f.digitize(10)
        >>> print(g.array)
        [[ 0  2  0  0  1  2  1  1]
         [ 1  2 --  4  3 --  0  4]
         [ 7  8  8  9  5  6  3  0]
         [ 1  3  2  4  3  4  0  0]
         [ 0  2  1  2  1  2  2  0]]
        >>> g.properties()
        {'Conventions': 'CF-1.7',
         'long_name': "Bin index to which each 'specific_humidity' value belongs",
         'bin_bounds': array([  3. ,  17.3,  17.3,  31.6,  31.6,  45.9,  45.9,  60.2,
                60.2,  74.5,  74.5,  88.8,  88.8, 103.1, 103.1, 117.4, 117.4, 131.7,
                131.7, 146. ]),
         'bin_count': 10,
         'bin_interval_type': 'lower: closed upper: open',
         'bin_standard_name': 'specific_humidity',
         'bin_units': '0.001 1'}

        """
        f = _inplace_enabled_define_and_cleanup(self)

        new_data, bins = self.data.digitize(
            bins,
            upper=upper,
            open_ends=open_ends,
            closed_ends=closed_ends,
            return_bins=True,
        )
        units = new_data.Units

        f.set_data(new_data, set_axes=False, copy=False)
        f.override_units(units, inplace=True)

        # ------------------------------------------------------------
        # Set properties
        # ------------------------------------------------------------
        f.set_property(
            "long_name",
            f"Bin index to which each {self.identity()!r} value belongs",
            copy=False,
        )

        f.set_property("bin_bounds", bins.array.flatten(), copy=False)

        bin_count = bins.shape[0]
        if open_ends:
            bin_count += 2

        f.set_property("bin_count", bin_count, copy=False)

        if upper:
            bin_interval_type = "lower: open upper: closed"
        else:
            bin_interval_type = "lower: closed upper: open"

        f.set_property("bin_interval_type", bin_interval_type, copy=False)

        standard_name = f.del_property("standard_name", None)
        if standard_name is not None:
            f.set_property("bin_standard_name", standard_name, copy=False)
        else:
            long_name = f.del_property("long_name", None)
            if long_name is not None:
                f.set_property("bin_long_name", long_name, copy=False)

        bin_units = bins.Units
        units = getattr(bin_units, "units", None)
        if units is not None:
            f.set_property("bin_units", units, copy=False)

        calendar = getattr(bin_units, "calendar", None)
        if calendar is not None:
            f.set_property("bin_calendar", calendar, copy=False)

        if return_bins:
            return f, bins

        return f

    @_manage_log_level_via_verbosity
    def bin(
        self,
        method,
        digitized,
        weights=None,
        measure=False,
        scale=None,
        mtol=1,
        ddof=1,
        radius="earth",
        great_circle=False,
        return_indices=False,
        verbose=None,
    ):
        """Collapse the data values that lie in N-dimensional bins.

        The data values of the field construct are binned according to how
        they correspond to the N-dimensional histogram bins of another set
        of variables (see `cf.histogram` for details), and each bin of
        values is collapsed with one of the collapse methods allowed by
        the *method* parameter.

        The number of dimensions of the output binned data is equal to the
        number of field constructs provided by the *digitized*
        argument. Each such field construct defines a sequence of bins and
        provides indices to the bins that each value of another field
        construct belongs. There is no upper limit to the number of
        dimensions of the output binned data.

        The output bins are defined by the exterior product of the
        one-dimensional bins of each digitized field construct. For
        example, if only one digitized field construct is provided then
        the output bins simply comprise its one-dimensional bins; if there
        are two digitized field constructs then the output bins comprise
        the two-dimensional matrix formed by all possible combinations of
        the two sets of one-dimensional bins; etc.

        An output value for a bin is formed by collapsing (using the
        method given by the *method* parameter) the elements of the data
        for which the corresponding locations in the digitized field
        constructs, taken together, index that bin. Note that it may be
        the case that not all output bins are indexed by the digitized
        field constructs, and for these bins missing data is returned.

        The returned field construct will have a domain axis construct for
        each dimension of the output bins, with a corresponding dimension
        coordinate construct that defines the bin boundaries.

        .. versionadded:: 3.0.2

        .. seealso:: `collapse`, `digitize`, `weights`, `cf.histogram`

        :Parameters:

            method: `str`
                The collapse method used to combine values that map to
                each cell of the output field construct. The following
                methods are available (see
                https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
                for precise definitions):

                ============================  ============================  ========
                *method*                      Description                   Weighted
                ============================  ============================  ========
                ``'maximum'``                 The maximum of the values.    Never

                ``'minimum'``                 The minimum of the values.    Never

                ``'maximum_absolute_value'``  The maximum of the absolute   Never
                                              values.

                ``'minimum_absolute_value'``  The minimum of the absolute   Never
                                              values.

                ``'mid_range'``               The average of the maximum    Never
                                              and the minimum of the
                                              values.

                ``'range'``                   The absolute difference       Never
                                              between the maximum and the
                                              minimum of the values.

                ``'median'``                  The median of the values.     Never

                ``'sum'``                     The sum of the values.        Never

                ``'sum_of_squares'``          The sum of the squares of     Never
                                              values.

                ``'sample_size'``             The sample size, i.e. the     Never
                                              number of non-missing
                                              values.

                ``'sum_of_weights'``          The sum of weights, as        Never
                                              would be used for other
                                              calculations.

                ``'sum_of_weights2'``         The sum of squares of         Never
                                              weights, as would be used
                                              for other calculations.

                ``'mean'``                    The weighted or unweighted    May be
                                              mean of the values.

                ``'mean_absolute_value'``     The mean of the absolute      May be
                                              values.

                ``'mean_of_upper_decile'``    The mean of the upper group   May be
                                              of data values defined by
                                              the upper tenth of their
                                              distribution.

                ``'variance'``                The weighted or unweighted    May be
                                              variance of the values, with
                                              a given number of degrees of
                                              freedom.

                ``'standard_deviation'``      The square root of the        May be
                                              weighted or unweighted
                                              variance.

                ``'root_mean_square'``        The square root of the        May be
                                              weighted or unweighted mean
                                              of the squares of the
                                              values.

                ``'integral'``                The integral of values.       Always
                ============================  ============================  ========

                * Collapse methods that are "Never" weighted ignore the
                  *weights* parameter, even if it is set.

                * Collapse methods that "May be" weighted will only be
                  weighted if the *weights* parameter is set.

                * Collapse methods that are "Always" weighted require the
                  *weights* parameter to be set.

            digitized: (sequence of) `Field`
                One or more field constructs that contain digitized data
                with corresponding metadata, as would be output by
                `cf.Field.digitize`. Each field construct contains indices
                to the one-dimensional bins to which each value of an
                original field construct belongs; and there must be
                ``bin_count`` and ``bin_bounds`` properties as defined by
                the `digitize` method (and any of the extra properties
                defined by that method are also recommended).

                The bins defined by the ``bin_count`` and ``bin_bounds``
                properties are used to create a dimension coordinate
                construct for the output field construct.

                Each digitized field construct must be transformable so
                that it is broadcastable to the input field construct's
                data. This is done by using the metadata constructs of the
                to create a mapping of physically compatible dimensions
                between the fields, and then manipulating the dimensions
                of the digitized field construct's data to ensure that
                broadcasting can occur.

            weights: optional
                Specify the weights for the collapse calculations. The
                weights are those that would be returned by this call of
                the field construct's `~cf.Field.weights` method:
                ``f.weights(weights, measure=measure, scale=scale,
                radius=radius, great_circle=great_circle,
                components=True)``. See the *measure, *scale*, *radius*
                and *great_circle* parameters and `cf.Field.weights` for
                details.

                .. note:: By default *weights* is `None`, resulting in
                          **unweighted calculations**.

                .. note:: Setting *weights* to `True` is generally a good
                          way to ensure that all collapses are
                          appropriately weighted according to the field
                          construct's metadata. In this case, if it is not
                          possible to create weights for any axis then an
                          exception will be raised.

                          However, care needs to be taken if *weights* is
                          `True` when cell volume weights are desired. The
                          volume weights will be taken from a "volume"
                          cell measure construct if one exists, otherwise
                          the cell volumes will be calculated as being
                          proportional to the sizes of one-dimensional
                          vertical coordinate cells. In the latter case
                          **if the vertical dimension coordinates do not
                          define the actual height or depth thickness of
                          every cell in the domain then the weights will
                          be incorrect**.

                If *weights* is the boolean `True` then weights are
                calculated for all of the domain axis constructs.

                *Parameter example:*
                  To specify weights based on the field construct's
                  metadata for all axes use ``weights=True``.

                *Parameter example:*
                  To specify weights based on cell areas, leaving all
                  other axes unweighted, use ``weights='area'``.

                *Parameter example:*
                  To specify weights based on cell areas and linearly in
                  time, leaving all other axes unweighted, you could set
                  ``weights=('area', 'T')``.

            measure: `bool`, optional
                Create weights, as defined by the *weights* parameter,
                which are cell measures, i.e. which describe actual cell
                sizes (e.g. cell areas) with appropriate units
                (e.g. metres squared). By default the weights are scaled
                to lie between 0 and 1 and have arbitrary units (see the
                *scale* parameter).

                Cell measures can be created for any combination of
                axes. For example, cell measures for a time axis are the
                time span for each cell with canonical units of seconds;
                cell measures for the combination of four axes
                representing time and three dimensional space could have
                canonical units of metres cubed seconds.

                When collapsing with the ``'integral'`` method, *measure*
                must be True, and the units of the weights are
                incorporated into the units of the returned field
                construct.

                .. note:: Specifying cell volume weights via
                          ``weights=['X', 'Y', 'Z']`` or
                          ``weights=['area', 'Z']`` (or other equivalents)
                          will produce **an incorrect result if the
                          vertical dimension coordinates do not define the
                          actual height or depth thickness of every cell
                          in the domain**. In this case,
                          ``weights='volume'`` should be used instead,
                          which requires the field construct to have a
                          "volume" cell measure construct.

                          If ``weights=True`` then care also needs to be
                          taken, as a "volume" cell measure construct will
                          be used if present, otherwise the cell volumes
                          will be calculated using the size of the
                          vertical coordinate cells.

            scale: number, optional
                If set to a positive number then scale the weights, as
                defined by the *weights* parameter, so that they are less
                than or equal to that number. By default the weights are
                scaled to lie between 0 and 1 (i.e.  *scale* is 1).

                *Parameter example:*
                  To scale all weights so that they lie between 0 and 0.5:
                  ``scale=0.5``.

            mtol: number, optional
                Set the fraction of input data elements which is allowed
                to contain missing data when contributing to an individual
                output data element. Where this fraction exceeds *mtol*,
                missing data is returned. The default is 1, meaning that a
                missing datum in the output array occurs when its
                contributing input array elements are all missing data. A
                value of 0 means that a missing datum in the output array
                occurs whenever any of its contributing input array
                elements are missing data. Any intermediate value is
                permitted.

                *Parameter example:*
                  To ensure that an output array element is a missing
                  datum if more than 25% of its input array elements are
                  missing data: ``mtol=0.25``.

            ddof: number, optional
                The delta degrees of freedom in the calculation of a
                standard deviation or variance. The number of degrees of
                freedom used in the calculation is (N-*ddof*) where N
                represents the number of non-missing elements contributing
                to the calculation. By default *ddof* is 1, meaning the
                standard deviation and variance of the population is
                estimated according to the usual formula with (N-1) in the
                denominator to avoid the bias caused by the use of the
                sample mean (Bessel's correction).

            radius: optional
                Specify the radius used for calculating the areas of
                cells defined in spherical polar coordinates. The
                radius is that which would be returned by this call of
                the field construct's `~cf.Field.radius` method:
                ``f.radius(radius)``. See the `cf.Field.radius` for
                details.

                By default *radius* is ``'earth'`` which means that if
                and only if the radius can not found from the datums
                of any coordinate reference constructs, then the
                default radius taken as 6371229 metres.

            great_circle: `bool`, optional
                If True then allow, if required, the derivation of i) area
                weights from polygon geometry cells by assuming that each
                cell part is a spherical polygon composed of great circle
                segments; and ii) and the derivation of line-length
                weights from line geometry cells by assuming that each
                line part is composed of great circle segments.

                .. versionadded:: 3.2.0

            {{verbose: `int` or `str` or `None`, optional}}

        :Returns:

            `Field`
                The field construct containing the binned values.

        **Examples**

        Find the range of values that lie in each bin:

        >>> print(q)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 0.001 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(q.array)
        [[  7.  34.   3.  14.  18.  37.  24.  29.]
         [ 23.  36.  45.  62.  46.  73.   6.  66.]
         [110. 131. 124. 146.  87. 103.  57.  11.]
         [ 29.  59.  39.  70.  58.  72.   9.  17.]
         [  6.  36.  19.  35.  18.  37.  34.  13.]]
        >>> indices = q.digitize(10)
        >>> b = q.bin('range', digitized=indices)
        >>> print(b)
        Field: specific_humidity
        ------------------------
        Data            : specific_humidity(specific_humidity(10)) 0.001 1
        Cell methods    : latitude: longitude: range
        Dimension coords: specific_humidity(10) = [10.15, ..., 138.85000000000002] 0.001 1
        >>> print(b.array)
        [14. 11. 11. 13. 11.  0.  0.  0.  7.  0.]

        Find various metrics describing how
        ``tendency_of_sea_water_potential_temperature_expressed_as_heat_content``
        data varies with ``sea_water_potential_temperature`` and
        ``sea_water_salinity``:

        >>> t
        Field: sea_water_potential_temperature (ncvar%sea_water_potential_temperature)
        ------------------------------------------------------------------------------
        Data            : sea_water_potential_temperature(time(1), depth(1), latitude(5), longitude(8)) K
        Cell methods    : area: mean time(1): mean
        Dimension coords: time(1) = [2290-06-01 00:00:00] 360_day
                        : depth(1) = [3961.89990234375] m
                        : latitude(5) = [-1.875, ..., 3.125] degrees_north
                        : longitude(8) = [75.0, ..., 83.75] degrees_east
        Auxiliary coords: model_level_number(depth(1)) = [18]
        >>> s
        Field: sea_water_salinity (ncvar%sea_water_salinity)
        ----------------------------------------------------
        Data            : sea_water_salinity(time(1), depth(1), latitude(5), longitude(8)) psu
        Cell methods    : area: mean time(1): mean
        Dimension coords: time(1) = [2290-06-01 00:00:00] 360_day
                        : depth(1) = [3961.89990234375] m
                        : latitude(5) = [-1.875, ..., 3.125] degrees_north
                        : longitude(8) = [75.0, ..., 83.75] degrees_east
        Auxiliary coords: model_level_number(depth(1)) = [18]
        >>> x
        Field: tendency_of_sea_water_potential_temperature_expressed_as_heat_content (ncvar%tend)
        -----------------------------------------------------------------------------------------
        Data            : tendency_of_sea_water_potential_temperature_expressed_as_heat_content(time(1), depth(1), latitude(5), longitude(8)) W m-2
        Cell methods    : area: mean time(1): mean
        Dimension coords: time(1) = [2290-06-01 00:00:00] 360_day
                        : depth(1) = [3961.89990234375] m
                        : latitude(5) = [-1.875, ..., 3.125] degrees_north
                        : longitude(8) = [75.0, ..., 83.75] degrees_east
        Auxiliary coords: model_level_number(depth(1)) = [18]
        >>> print(x.array)
        [[[[-209.72  340.86   94.75  154.21   38.54 -262.75  158.22  154.58]
           [ 311.67  245.91 -168.16   47.61 -219.66 -270.33  226.1    52.0 ]
           [     -- -112.34  271.67  189.22    9.92  232.39  221.17  206.0 ]
           [     --      --  -92.31 -285.57  161.55  195.89 -258.29    8.35]
           [     --      --   -7.82 -299.79  342.32 -169.38  254.5   -75.4 ]]]]

        >>> t_indices = t.digitize(6)
        >>> s_indices = s.digitize(4)

        >>> n = x.bin('sample_size', [t_indices, s_indices])
        >>> print(n)
        Field: number_of_observations
        -----------------------------
        Data            : number_of_observations(sea_water_salinity(4), sea_water_potential_temperature(6)) 1
        Cell methods    : latitude: longitude: point
        Dimension coords: sea_water_salinity(4) = [6.3054151982069016, ..., 39.09366758167744] psu
                        : sea_water_potential_temperature(6) = [278.1569468180338, ..., 303.18466695149743] K
        >>> print(n.array)
        [[ 1  2 2  2 --  2]
         [ 2  1 3  3  3  2]
         [-- -- 3 --  1 --]
         [ 1 -- 1  3  2  1]]

        >>> m = x.bin('mean', [t_indices, s_indices], weights=['X', 'Y', 'Z', 'T'])
        >>> print(m)
        Field: tendency_of_sea_water_potential_temperature_expressed_as_heat_content
        ----------------------------------------------------------------------------
        Data            : tendency_of_sea_water_potential_temperature_expressed_as_heat_content(sea_water_salinity(4), sea_water_potential_temperature(6)) W m-2
        Cell methods    : latitude: longitude: mean
        Dimension coords: sea_water_salinity(4) = [6.3054151982069016, ..., 39.09366758167744] psu
                        : sea_water_potential_temperature(6) = [278.1569468180338, ..., 303.18466695149743] K
        >>> print(m.array)
        [[ 189.22 131.36    6.75 -41.61     --  100.04]
         [-116.73 232.38   -4.82 180.47 134.25 -189.55]
         [     --     --  180.69     --  47.61      --]
         [158.22      -- -262.75  64.12 -51.83 -219.66]]

        >>> i = x.bin(
        ...         'integral', [t_indices, s_indices],
        ...         weights=['X', 'Y', 'Z', 'T'], measure=True
        ...     )
        >>> print(i)
        Field: long_name=integral of tendency_of_sea_water_potential_temperature_expressed_as_heat_content
        --------------------------------------------------------------------------------------------------
        Data            : long_name=integral of tendency_of_sea_water_potential_temperature_expressed_as_heat_content(sea_water_salinity(4), sea_water_potential_temperature(6)) 86400 m3.kg.s-2
        Cell methods    : latitude: longitude: sum
        Dimension coords: sea_water_salinity(4) = [6.3054151982069016, ..., 39.09366758167744] psu
                        : sea_water_potential_temperature(6) = [278.1569468180338, ..., 303.18466695149743] K
        >>> print(i.array)
        [[ 3655558758400.0 5070927691776.0   260864491520.0 -1605439586304.0               --  3863717609472.0]
         [-4509735059456.0 4489564127232.0  -280126521344.0 10454746267648.0  7777254113280.0 -7317268463616.0]
         [              --              -- 10470463373312.0               --   919782031360.0               --]
         [ 3055211773952.0              -- -5073676009472.0  3715958833152.0 -2000787079168.0 -4243632160768.0]]

        >>> w = x.bin('sum_of_weights', [t_indices, s_indices], weights=['X', 'Y', 'Z', 'T'], measure=True)
        Field: long_name=sum_of_weights of tendency_of_sea_water_potential_temperature_expressed_as_heat_content
        --------------------------------------------------------------------------------------------------------
        Data            : long_name=sum_of_weights of tendency_of_sea_water_potential_temperature_expressed_as_heat_content(sea_water_salinity(4), sea_water_potential_temperature(6)) 86400 m3.s
        Cell methods    : latitude: longitude: sum
        Dimension coords: sea_water_salinity(4) = [7.789749830961227, ..., 36.9842486679554] psu
                        : sea_water_potential_temperature(6) = [274.50717671712243, ..., 302.0188242594401] K
        >>> print(w.array)
        [[19319093248.0 38601412608.0 38628990976.0 38583025664.0            --  38619795456.0]
         [38628990976.0 19319093248.0 57957281792.0 57929699328.0 57929695232.0  38601412608.0]
         [         --              -- 57948086272.0            -- 19319093248.0             --]
         [19309897728.0            -- 19309897728.0 57948086272.0 38601412608.0  19319093248.0]]

        Demonstrate that the integral divided by the sum of the cell
        measures is equal to the mean:

        >>> print(i/w)
        Field:
        -------
        Data            : (sea_water_salinity(4), sea_water_potential_temperature(6)) kg.s-3
        Cell methods    : latitude: longitude: sum
        Dimension coords: sea_water_salinity(4) = [7.789749830961227, ..., 36.9842486679554] psu
                        : sea_water_potential_temperature(6) = [274.50717671712243, ..., 302.0188242594401] K
        >>> (i/w == m).all()
        True

        """
        debug = is_log_level_debug(logger)

        if debug:
            logger.debug(f"    Method: {method}")  # pragma: no cover

        if method == "integral":
            if weights is None:
                raise ValueError(
                    "Must specify weights for 'integral' calculations."
                )

            if not measure:
                raise ValueError(
                    "Must set measure=True for 'integral' calculations."
                )

            if scale is not None:
                raise ValueError(
                    "Can't set scale for 'integral' calculations."
                )

        axes = []
        bin_indices = []
        shape = []
        dims = []
        names = []

        # Initialise the output binned field
        out = type(self)(properties=self.properties())

        # Sort out its identity
        if method == "sample_size":
            out.standard_name = "number_of_observations"
        elif method in (
            "integral",
            "sum_of_squares",
            "sum_of_weights",
            "sum_of_weights2",
        ):
            out.del_property("standard_name", None)

        long_name = self.get_property("long_name", None)
        if long_name is None:
            out.long_name = (
                method + " of " + self.get_property("standard_name", "")
            )
        else:
            out.long_name = method + " of " + long_name

        # ------------------------------------------------------------
        # Create domain axes and dimension coordinates for the output
        # binned field
        # ------------------------------------------------------------
        if isinstance(digitized, self.__class__):
            digitized = (digitized,)

        for f in digitized[::-1]:
            f = self._conform_for_data_broadcasting(f)

            if not self._is_broadcastable(f.shape):
                raise ValueError(
                    f"Conformed digitized field {f!r} construct must have "
                    f"shape broadcastable to {self.shape}."
                )

            bin_bounds = f.get_property("bin_bounds", None)
            bin_count = f.get_property("bin_count", None)
            bin_interval_type = f.get_property("bin_interval_type", None)
            bin_units = f.get_property("bin_units", None)
            bin_calendar = f.get_property("bin_calendar", None)
            bin_standard_name = f.get_property("bin_standard_name", None)
            bin_long_name = f.get_property("bin_long_name", None)

            if bin_count is None:
                raise ValueError(
                    f"Digitized field construct {f!r} must have a 'bin_count' "
                    "property."
                )

            if bin_bounds is None:
                raise ValueError(
                    f"Digitized field construct {f!r} must have a "
                    "'bin_bounds' property."
                )

            if bin_count != len(bin_bounds) / 2:
                raise ValueError(
                    f"Digitized field construct {f!r} bin_count must equal "
                    f"len(bin_bounds)/2. Got bin_count={bin_count}, "
                    f"len(bin_bounds)/2={len(bin_bounds) / 2}"
                )

            # Create dimension coordinate for bins
            dim = DimensionCoordinate()
            if bin_standard_name is not None:
                dim.standard_name = bin_standard_name
            elif bin_long_name is not None:
                dim.long_name = bin_long_name

            if bin_interval_type is not None:
                dim.set_property(
                    "bin_interval_type", bin_interval_type, copy=False
                )

            # Create units for the bins
            units = Units(bin_units, bin_calendar)

            data = Data(
                0.5 * (bin_bounds[1::2] + bin_bounds[0::2]), units=units
            )
            dim.set_data(data=data, copy=False)

            bounds_data = Data(
                np.reshape(bin_bounds, (bin_count, 2)), units=units
            )
            dim.set_bounds(self._Bounds(data=bounds_data))

            if debug:
                logger.debug(
                    "                    bins     : "
                    f"{dim.identity()} {bounds_data!r}"  # DCH
                )  # pragma: no cover

            # Set domain axis and dimension coordinate for bins
            axis = out.set_construct(self._DomainAxis(dim.size))
            out.set_construct(dim, axes=[axis], copy=False)

            axes.append(axis)
            bin_indices.append(f.data)
            shape.append(dim.size)
            dims.append(dim)
            names.append(dim.identity())

        # ------------------------------------------------------------
        # Initialise the ouput data as a totally masked array
        # ------------------------------------------------------------
        if method == "sample_size":
            dtype = int
        else:
            dtype = self.dtype

        data = Data.masked_all(shape=tuple(shape), dtype=dtype, units=None)
        out.set_data(data, axes=axes, copy=False)
        out.hardmask = False

        c = self.copy()

        # ------------------------------------------------------------
        # Parse the weights
        # ------------------------------------------------------------
        if weights is not None:
            if not measure and scale is None:
                scale = 1.0

            weights = self.weights(
                weights,
                scale=scale,
                measure=measure,
                radius=radius,
                great_circle=great_circle,
                components=True,
            )

        # ------------------------------------------------------------
        # Find the unique multi-dimensional bin indices (TODO: can I
        # LAMA this?)
        # ------------------------------------------------------------
        y = np.empty((len(bin_indices), bin_indices[0].size), dtype=int)
        for i, f in enumerate(bin_indices):
            y[i, :] = f.array.flatten()

        unique_indices = np.unique(y, axis=1)
        del f
        del y

        # DCH
        if debug:
            logger.debug(
                f"    Weights: {weights}\n",
                f"    Number of indexed ({', '.join(names)}) bins: "
                f"{unique_indices.shape[1]}\n"
                f"    ({', '.join(names)}) bin indices:",  # DCH
            )  # pragma: no cover

        # Loop round unique collections of bin indices
        for i in zip(*unique_indices):
            if debug:
                logger.debug(f"{' '.join(str(i))}")  # pragma: no cover

            b = bin_indices[0] == i[0]
            for a, n in zip(bin_indices[1:], i[1:]):
                b &= a == n

            b.filled(False, inplace=True)

            c.set_data(
                self.data.where(b, None, cf_masked), set_axes=False, copy=False
            )

            result = c.collapse(
                method=method, weights=weights, measure=measure
            ).data
            out.data[i] = result.datum()

        # Set correct units (note: takes them from the last processed
        # "result" variable in the above loop)
        out.override_units(result.Units, inplace=True)
        out.hardmask = True

        # ------------------------------------------------------------
        # Create a cell method (if possible)
        # ------------------------------------------------------------
        standard_names = []
        domain_axes = self.domain_axes(filter_by_size=(ge(2),), todict=True)

        for da_key in domain_axes:
            dim = self.dimension_coordinate(
                filter_by_axis=(da_key,), default=None
            )
            if dim is None:
                continue

            standard_name = dim.get_property("standard_name", None)
            if standard_name is None:
                continue

            standard_names.append(standard_name)

        if len(standard_names) == len(domain_axes):
            cell_method = CellMethod(
                axes=sorted(standard_names),
                method=_collapse_cell_methods[method],
            )
            out.set_construct(cell_method, copy=False)

        return out

    def histogram(self, digitized):
        """Return a multi-dimensional histogram of the data.

        **This has moved to** `cf.histogram`

        """
        raise RuntimeError("Use cf.histogram instead.")

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_manage_log_level_via_verbosity
    def collapse(
        self,
        method,
        axes=None,
        squeeze=False,
        mtol=1,
        weights=None,
        ddof=1,
        a=None,
        inplace=False,
        group=None,
        regroup=False,
        within_days=None,
        within_years=None,
        over_days=None,
        over_years=None,
        coordinate=None,
        group_by=None,
        group_span=None,
        group_contiguous=1,
        measure=False,
        scale=None,
        radius="earth",
        great_circle=False,
        verbose=None,
        remove_vertical_crs=True,
        _create_zero_size_cell_bounds=False,
        _update_cell_methods=True,
        i=False,
        _debug=False,
        **kwargs,
    ):
        """Collapse axes of the field.

        Collapsing one or more dimensions reduces their size and replaces
        the data along those axes with representative statistical
        values. The result is a new field construct with consistent
        metadata for the collapsed values.

        By default all axes with size greater than 1 are collapsed
        completely (i.e. to size 1) with a given collapse method.

        *Example:*
          Find the minimum of the entire data:

          >>> b = a.collapse('minimum')

        The collapse can also be applied to any subset of the field
        construct's dimensions. In this case, the domain axis and
        coordinate constructs for the non-collapsed dimensions remain the
        same. This is implemented either with the axes keyword, or with a
        CF-netCDF cell methods-like syntax for describing both the
        collapse dimensions and the collapse method in a single
        string. The latter syntax uses construct identities instead of
        netCDF dimension names to identify the collapse axes.

        Statistics may be created to represent variation over one
        dimension or a combination of dimensions.

        *Example:*
           Two equivalent techniques for creating a field construct of
           temporal maxima at each horizontal location:

           >>> b = a.collapse('maximum', axes='T')
           >>> b = a.collapse('T: maximum')

        *Example:*
          Find the horizontal maximum, with two equivalent techniques.

          >>> b = a.collapse('maximum', axes=['X', 'Y'])
          >>> b = a.collapse('X: Y: maximum')

        Variation over horizontal area may also be specified by the
        special identity 'area'. This may be used for any horizontal
        coordinate reference system.

        *Example:*
          Find the horizontal maximum using the special identity 'area':

          >>> b = a.collapse('area: maximum')


        **Collapse methods**

        The following collapse methods are available (see
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for precise definitions):

        ============================  ============================
        Method                        Description
        ============================  ============================
        ``'maximum'``                 The maximum of the values.

        ``'minimum'``                 The minimum of the values.

        ``'maximum_absolute_value'``  The maximum of the absolute
                                      values.

        ``'minimum_absolute_value'``  The minimum of the absolute
                                      values.

        ``'mid_range'``               The average of the maximum
                                      and the minimum of the
                                      values.

        ``'median'``                  The median of the values.

        ``'range'``                   The absolute difference
                                      between the maximum and the
                                      minimum of the values.

        ``'sum'``                     The sum of the values.

        ``'sum_of_squares'``          The sum of the squares of
                                      values.

        ``'sample_size'``             The sample size, i.e. the
                                      number of non-missing
                                      values.

        ``'sum_of_weights'``          The sum of weights, as
                                      would be used for other
                                      calculations.

        ``'sum_of_weights2'``         The sum of squares of
                                      weights, as would be used
                                      for other calculations.

        ``'mean'``                    The weighted or unweighted
                                      mean of the values.

        ``'mean_absolute_value'``     The mean of the absolute
                                      values.

        ``'mean_of_upper_decile'``    The mean of the upper group
                                      of data values defined by
                                      the upper tenth of their
                                      distribution.

        ``'variance'``                The weighted or unweighted
                                      variance of the values, with
                                      a given number of degrees of
                                      freedom.

        ``'standard_deviation'``      The weighted or unweighted
                                      standard deviation of the
                                      values, with a given number
                                      of degrees of freedom.

        ``'root_mean_square'``        The square root of the
                                      weighted or unweighted mean
                                      of the squares of the
                                      values.

        ``'integral'``                The integral of values.
        ============================  ============================


        **Data type and missing data**

        In all collapses, missing data array elements are accounted for in
        the calculation.

        Any collapse method that involves a calculation (such as
        calculating a mean), as opposed to just selecting a value (such as
        finding a maximum), will return a field containing double
        precision floating point numbers. If this is not desired then the
        data type can be reset after the collapse with the `dtype`
        attribute of the field construct.


        **Collapse weights**

        The calculations of means, standard deviations and variances are,
        by default, **not weighted**. For weights to be incorporated in
        the collapse, the axes to be weighted must be identified with the
        *weights* keyword.

        Weights are either derived from the field construct's metadata
        (such as cell sizes), or may be provided explicitly in the form of
        other field constructs containing data of weights values. In
        either case, the weights actually used are those derived by the
        `weights` method of the field construct with the same weights
        keyword value. Collapsed axes that are not identified by the
        *weights* keyword are unweighted during the collapse operation.

        *Example:*
          Create a weighted time average:

          >>> b = a.collapse('T: mean', weights=True)

        *Example:*
          Calculate the mean over the time and latitude axes, with
          weights only applied to the latitude axis:

          >>> b = a.collapse('T: Y: mean', weights='Y')

        *Example*
          Alternative syntax for specifying area weights:

          >>> b = a.collapse('area: mean', weights=True)

        An alternative technique for specifying weights is to set the
        *weights* keyword to the output of a call to the `weights` method.

        *Example*
          Alternative syntax for specifying weights:

          >>> b = a.collapse('area: mean', weights=a.weights('area'))

        **Multiple collapses**

        Multiple collapses normally require multiple calls to `collapse`:
        one on the original field construct and then one on each interim
        field construct.

        *Example:*
          Calculate the temporal maximum of the weighted areal means
          using two independent calls:

          >>> b = a.collapse('area: mean', weights=True).collapse('T: maximum')

        If preferred, multiple collapses may be carried out in a single
        call by using the CF-netCDF cell methods-like syntax (note that
        the colon (:) is only used after the construct identity that
        specifies each axis, and a space delimits the separate collapses).

        *Example:*
          Calculate the temporal maximum of the weighted areal means in
          a single call, using the cf-netCDF cell methods-like syntax:

          >>> b =a.collapse('area: mean T: maximum', weights=True)


        **Grouped collapses**

        A grouped collapse is one for which as axis is not collapsed
        completely to size 1. Instead the collapse axis is partitioned
        into non-overlapping groups and each group is collapsed to size
        1. The resulting axis will generally have more than one
        element. For example, creating 12 annual means from a timeseries
        of 120 months would be a grouped collapse.

        Selected statistics for overlapping groups can be calculated with
        the `moving_window` method.

        The *group* keyword defines the size of the groups. Groups can be
        defined in a variety of ways, including with `Query`,
        `TimeDuration` and `Data` instances.

        An element of the collapse axis can not be a member of more than
        one group, and may be a member of no groups. Elements that are not
        selected by the *group* keyword are excluded from the result.

        *Example:*
          Create annual maxima from a time series, defining a year to
          start on 1st December.

          >>> b = a.collapse('T: maximum', group=cf.Y(month=12))

        *Example:*
          Find the maximum of each group of 6 elements along an axis.

          >>> b = a.collapse('T: maximum', group=6)

        *Example:*
          Create December, January, February maxima from a time series.

          >>> b = a.collapse('T: maximum', group=cf.djf())

        *Example:*
          Create maxima for each 3-month season of a timeseries (DJF, MAM,
          JJA, SON).

          >>> b = a.collapse('T: maximum', group=cf.seasons())

        *Example:*
          Calculate zonal means for the western and eastern hemispheres.

          >>> b = a.collapse('X: mean', group=cf.Data(180, 'degrees'))

        Groups can be further described with the *group_span* parameter
        (to include groups whose actual span is not equal to a given
        value) and the *group_contiguous* parameter (to include
        non-contiguous groups, or any contiguous group containing
        overlapping cells).


        **Climatological statistics**

        Climatological statistics may be derived from corresponding
        portions of the annual cycle in a set of years (e.g. the average
        January temperatures in the climatology of 1961-1990, where the
        values are derived by averaging the 30 Januarys from the separate
        years); or from corresponding portions of the diurnal cycle in a
        set of days (e.g. the average temperatures for each hour in the
        day for May 1997). A diurnal climatology may also be combined with
        a multiannual climatology (e.g. the minimum temperature for each
        hour of the average day in May from a 1961-1990 climatology).

        Calculation requires two or three collapses, depending on the
        quantity being created, all of which are grouped collapses. Each
        collapse method needs to indicate its climatological nature with
        one of the following qualifiers,

        ================  =======================
        Method qualifier  Associated keyword
        ================  =======================
        ``within years``  *within_years*
        ``within days``   *within_days*
        ``over years``    *over_years* (optional)
        ``over days``     *over_days* (optional)
        ================  =======================

        and the associated keyword specifies how the method is to be
        applied.

        *Example*
          Calculate the multiannual average of the seasonal means:

          >>> b = a.collapse('T: mean within years T: mean over years',
          ...                within_years=cf.seasons(), weights=True)

        *Example:*
          Calculate the multiannual variance of the seasonal
          minima. Note that the units of the result have been changed
          from 'K' to 'K2':

          >>> b = a.collapse('T: minimum within years T: variance over years',
          ...                within_years=cf.seasons(), weights=True)

        When collapsing over years, it is assumed by default that each
        portion of the annual cycle is collapsed over all years that are
        present. This is the case in the above two examples. It is
        possible, however, to restrict the years to be included, or group
        them into chunks, with the *over_years* keyword.

        *Example:*
          Calculate the multiannual average of the seasonal means in 5
          year chunks:

          >>> b = a.collapse(
          ...     'T: mean within years T: mean over years', weights=True,
          ...     within_years=cf.seasons(), over_years=cf.Y(5)
          ... )

        *Example:*
          Calculate the multiannual average of the seasonal means,
          restricting the years from 1963 to 1968:

          >>> b = a.collapse(
          ...     'T: mean within years T: mean over years', weights=True,
          ...     within_years=cf.seasons(),
          ...     over_years=cf.year(cf.wi(1963, 1968))
          ... )

        Similarly for collapses over days, it is assumed by default that
        each portion of the diurnal cycle is collapsed over all days that
        are present, But it is possible to restrict the days to be
        included, or group them into chunks, with the *over_days* keyword.

        The calculation can be done with multiple collapse calls, which
        can be useful if the interim stages are needed independently, but
        be aware that the interim field constructs will have
        non-CF-compliant cell method constructs.

        *Example:*
          Calculate the multiannual maximum of the seasonal standard
          deviations with two separate collapse calls:

          >>> b = a.collapse('T: standard_deviation within years',
          ...                within_years=cf.seasons(), weights=True)


        .. versionadded:: 1.0

        .. seealso:: `bin`, `cell_area`, `convolution_filter`,
                     `moving_window`, `radius`, `weights`

        :Parameters:

            method: `str`
                Define the collapse method. All of the axes specified by
                the *axes* parameter are collapsed simultaneously by this
                method. The method is given by one of the following
                strings (see
                https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
                for precise definitions):

                ============================  ============================  ========
                *method*                      Description                   Weighted
                ============================  ============================  ========
                ``'maximum'``                 The maximum of the values.    Never

                ``'minimum'``                 The minimum of the values.    Never

                ``'maximum_absolute_value'``  The maximum of the absolute   Never
                                              values.

                ``'minimum_absolute_value'``  The minimum of the absolute   Never
                                              values.

                ``'mid_range'``               The average of the maximum    Never
                                              and the minimum of the
                                              values.

                ``'median'``                  The median of the values.     Never

                ``'range'``                   The absolute difference       Never
                                              between the maximum and the
                                              minimum of the values.

                ``'sum'``                     The sum of the values.        Never

                ``'sum_of_squares'``          The sum of the squares of     Never
                                              values.

                ``'sample_size'``             The sample size, i.e. the     Never
                                              number of non-missing
                                              values.

                ``'sum_of_weights'``          The sum of weights, as        Never
                                              would be used for other
                                              calculations.

                ``'sum_of_weights2'``         The sum of squares of         Never
                                              weights, as would be used
                                              for other calculations.

                ``'mean'``                    The weighted or unweighted    May be
                                              mean of the values.

                ``'mean_absolute_value'``     The mean of the absolute      May be
                                              values.

                ``'mean_of_upper_decile'``    The mean of the upper group   May be
                                              of data values defined by
                                              the upper tenth of their
                                              distribution.

                ``'variance'``                The weighted or unweighted    May be
                                              variance of the values, with
                                              a given number of degrees of
                                              freedom.

                ``'standard_deviation'``      The weighted or unweighted    May be
                                              standard deviation of the
                                              values, with a given number
                                              of degrees of freedom.

                ``'root_mean_square'``        The square root of the        May be
                                              weighted or unweighted mean
                                              of the squares of the
                                              values.

                ``'integral'``                The integral of values.       Always
                ============================  ============================  ========

                * Collapse methods that are "Never" weighted ignore the
                  *weights* parameter, even if it is set.

                * Collapse methods that "May be" weighted will only be
                  weighted if the *weights* parameter is set.

                * Collapse methods that are "Always" weighted require the
                  *weights* parameter to be set.

                An alternative form of providing the collapse method is to
                provide a CF cell methods-like string. In this case an
                ordered sequence of collapses may be defined and both the
                collapse methods and their axes are provided. The axes are
                interpreted as for the *axes* parameter, which must not
                also be set. For example:

                >>> g = f.collapse(
                ...     'time: max (interval 1 hr) X: Y: mean dim3: sd')

                is equivalent to:

                >>> g = f.collapse('max', axes='time')
                >>> g = g.collapse('mean', axes=['X', 'Y'])
                >>> g = g.collapse('sd', axes='dim3')

                Climatological collapses are carried out if a *method*
                string contains any of the modifiers ``'within days'``,
                ``'within years'``, ``'over days'`` or ``'over
                years'``. For example, to collapse a time axis into
                multiannual means of calendar monthly minima:

                >>> g = f.collapse(
                ...     'time: minimum within years T: mean over years',
                ...     within_years=cf.M()
                ... )

                which is equivalent to:

                >>> g = f.collapse(
                ...     'time: minimum within years', within_years=cf.M())
                >>> g = g.collapse('mean over years', axes='T')

            axes: (sequence of) `str`, optional
                The axes to be collapsed, defined by those which would be
                selected by passing each given axis description to a call
                of the field construct's `domain_axis` method. For
                example, for a value of ``'X'``, the domain axis construct
                returned by ``f.domain_axis('X')`` is selected. If a
                selected axis has size 1 then it is ignored. By default
                all axes with size greater than 1 are collapsed.

                *Parameter example:*
                  ``axes='X'``

                *Parameter example:*
                  ``axes=['X']``

                *Parameter example:*
                  ``axes=['X', 'Y']``

                *Parameter example:*
                  ``axes=['Z', 'time']``

                If the *axes* parameter has the special value ``'area'``
                then it is assumed that the X and Y axes are intended.

                *Parameter example:*
                  ``axes='area'`` is equivalent to ``axes=['X', 'Y']``.

                *Parameter example:*
                  ``axes=['area', Z']`` is equivalent to ``axes=['X', 'Y',
                  'Z']``.

            weights: optional
                Specify the weights for the collapse axes. The weights
                are, in general, those that would be returned by this
                call of the field construct's `weights` method:
                ``f.weights(weights, axes=axes, measure=measure,
                scale=scale, radius=radius, great_circle=great_circle,
                components=True)``. See the *axes*, *measure*,
                *scale*, *radius* and *great_circle* parameters and
                `cf.Field.weights` for details, and note that the
                value of *scale* may be modified depending on the
                value of *measure*.

                .. note:: By default *weights* is `None`, resulting in
                          **unweighted calculations**.

                .. note:: Unless the *method* is ``'integral'``, the
                          units of the weights are not combined with
                          the field's units in the collapsed field.

                If the alternative form of providing the collapse method
                and axes combined as a CF cell methods-like string via the
                *method* parameter has been used, then the *axes*
                parameter is ignored and the axes are derived from the
                *method* parameter. For example, if *method* is ``'T:
                area: minimum'`` then this defines axes of ``['T',
                'area']``. If *method* specifies multiple collapses,
                e.g. ``'T: minimum area: mean'`` then this implies axes of
                ``'T'`` for the first collapse, and axes of ``'area'`` for
                the second collapse.

                .. note:: Setting *weights* to `True` is generally a good
                          way to ensure that all collapses are
                          appropriately weighted according to the field
                          construct's metadata. In this case, if it is not
                          possible to create weights for any axis then an
                          exception will be raised.

                          However, care needs to be taken if *weights* is
                          `True` when cell volume weights are desired. The
                          volume weights will be taken from a "volume"
                          cell measure construct if one exists, otherwise
                          the cell volumes will be calculated as being
                          proportional to the sizes of one-dimensional
                          vertical coordinate cells. In the latter case
                          **if the vertical dimension coordinates do not
                          define the actual height or depth thickness of
                          every cell in the domain then the weights will
                          be incorrect**.

                *Parameter example:*
                  To specify weights based on the field construct's
                  metadata for all collapse axes use ``weights=True``.

                *Parameter example:*
                  To specify weights based on cell areas use
                  ``weights='area'``.

                *Parameter example:*
                  To specify weights based on cell areas and linearly in
                  time you could set ``weights=('area', 'T')``.

            measure: `bool`, optional
                If True, and *weights* is not `None`, create weights
                which are cell measures, i.e. which describe actual
                cell sizes (e.g. cell area) with appropriate units
                (e.g. metres squared). By default the weights units
                are ignored.

                Cell measures can be created for any combination of
                axes. For example, cell measures for a time axis are
                the time span for each cell with canonical units of
                seconds; cell measures for the combination of four
                axes representing time and three dimensional space
                could have canonical units of metres cubed seconds.

                When collapsing with the ``'integral'`` method,
                *measure* must be True, and the units of the weights
                are incorporated into the units of the returned field
                construct.

                .. note:: Specifying cell volume weights via
                          ``weights=['X', 'Y', 'Z']`` or
                          ``weights=['area', 'Z']`` (or other
                          equivalents) will produce **an incorrect
                          result if the vertical dimension coordinates
                          do not define the actual height or depth
                          thickness of every cell in the domain**. In
                          this case, ``weights='volume'`` should be
                          used instead, which requires the field
                          construct to have a "volume" cell measure
                          construct.

                          If ``weights=True`` then care also needs to
                          be taken, as a "volume" cell measure
                          construct will be used if present, otherwise
                          the cell volumes will be calculated using
                          the size of the vertical coordinate cells.

                .. versionadded:: 3.0.2

            scale: number or `None`, optional
                If set to a positive number then scale the weights so
                that they are less than or equal to that number. If
                set to `None`, the default, then the weights are not
                scaled.

                *Parameter example:*
                  To scale all weights so that they lie between 0 and
                  1 ``scale=1``.

                .. versionadded:: 3.0.2

                .. versionchanged:: 3.16.0 Default changed to `None`

            radius: optional
                Specify the radius used for calculating the areas of
                cells defined in spherical polar coordinates. The
                radius is that which would be returned by this call of
                the field construct's `~cf.Field.radius` method:
                ``f.radius(radius)``. See the `cf.Field.radius` for
                details.

                By default *radius* is ``'earth'`` which means that if
                and only if the radius can not found from the datums
                of any coordinate reference constructs, then the
                default radius taken as 6371229 metres.

                .. versionadded:: 3.0.2

            great_circle: `bool`, optional
                If True then allow, if required, the derivation of i) area
                weights from polygon geometry cells by assuming that each
                cell part is a spherical polygon composed of great circle
                segments; and ii) and the derivation of line-length
                weights from line geometry cells by assuming that each
                line part is composed of great circle segments.

                .. versionadded:: 3.2.0

            squeeze: `bool`, optional
                If True then size 1 collapsed axes are removed from the
                output data array. By default the axes which are collapsed
                are retained in the result's data array.

            mtol: number, optional
                Set the fraction of input data elements which is allowed
                to contain missing data when contributing to an individual
                output data element. Where this fraction exceeds *mtol*,
                missing data is returned. The default is 1, meaning that a
                missing datum in the output array occurs when its
                contributing input array elements are all missing data. A
                value of 0 means that a missing datum in the output array
                occurs whenever any of its contributing input array
                elements are missing data. Any intermediate value is
                permitted.

                *Parameter example:*
                  To ensure that an output array element is a missing
                  datum if more than 25% of its input array elements are
                  missing data: ``mtol=0.25``.

            ddof: number, optional
                The delta degrees of freedom in the calculation of a
                standard deviation or variance. The number of degrees of
                freedom used in the calculation is (N-*ddof*) where N
                represents the number of non-missing elements. By default
                *ddof* is 1, meaning the standard deviation and variance
                of the population is estimated according to the usual
                formula with (N-1) in the denominator to avoid the bias
                caused by the use of the sample mean (Bessel's
                correction).

            coordinate: optional
                Specify how the cell coordinate values for collapsed axes
                are placed. This has no effect on the cell bounds for the
                collapsed axes, which always represent the extrema of the
                input coordinates.

                The *coordinate* parameter may be one of:

                ===============  =========================================
                *coordinate*     Description
                ===============  =========================================
                `None`           This is the default.

                                 If the collapse is a climatological time
                                 collapse over years or over days then
                                 assume a value of ``'min'``, otherwise
                                 assume value of ``'mid_range'``.

                ``'mid_range'``  An output coordinate is the mean of
                                 first and last input coordinate bounds
                                 (or the first and last coordinates if
                                 there are no bounds). This is the
                                 default.

                ``'minimum'``    An output coordinate is the minimum of
                                 the input coordinates.

                ``'maximum'``    An output coordinate is the maximum of
                                 the input coordinates.
                ===============  =========================================

                *Parameter example:*
                  ``coordinate='minimum'``

            group: optional
                A grouped collapse is one for which an axis is not
                collapsed completely to size 1. Instead, the collapse axis
                is partitioned into non-overlapping groups and each group
                is collapsed to size 1, independently of the other
                groups. The results of the collapses are concatenated so
                that the output axis has a size equal to the number of
                groups.

                An element of the collapse axis can not be a member of
                more than one group, and may be a member of no
                groups. Elements that are not selected by the *group*
                parameter are excluded from the result.

                The *group* parameter defines how the axis elements are
                partitioned into groups, and may be one of:

                ===============  =========================================
                *group*          Description
                ===============  =========================================
                `Data`           Define groups by coordinate values that
                                 span the given range. The first group
                                 starts at the first coordinate bound of
                                 the first axis element (or its coordinate
                                 if there are no bounds) and spans the
                                 defined group size. Each subsequent
                                 group immediately follows the preceding
                                 one. By default each group contains the
                                 consecutive run of elements whose
                                 coordinate values lie within the group
                                 limits (see the *group_by* parameter).

                                 * By default each element will be in
                                   exactly one group (see the *group_by*,
                                   *group_span* and *group_contiguous*
                                   parameters).

                                 * By default groups may contain different
                                   numbers of elements.

                                 * If no units are specified then the
                                   units of the coordinates are assumed.

                `TimeDuration`   Define groups by a time interval spanned
                                 by the coordinates. The first group
                                 starts at or before the first coordinate
                                 bound of the first axis element (or its
                                 coordinate if there are no bounds) and
                                 spans the defined group size. Each
                                 subsequent group immediately follows the
                                 preceding one. By default each group
                                 contains the consecutive run of elements
                                 whose coordinate values lie within the
                                 group limits (see the *group_by*
                                 parameter).

                                 * By default each element will be in
                                   exactly one group (see the *group_by*,
                                   *group_span* and *group_contiguous*
                                   parameters).

                                 * By default groups may contain different
                                   numbers of elements.

                                 * The start of the first group may be
                                   before the first first axis element,
                                   depending on the offset defined by the
                                   time duration. For example, if
                                   ``group=cf.Y(month=12)`` then the first
                                   group will start on the closest 1st
                                   December to the first axis element.

                `Query`          Define groups from elements whose
                                 coordinates satisfy the query
                                 condition. Multiple groups are created:
                                 one for each maximally consecutive run
                                 within the selected elements.

                                 If a sequence of `Query` is provided then
                                 groups are defined for each query.

                                 * If a coordinate does not satisfy any of
                                   the query conditions then its element
                                   will not be in a group.

                                 * By default groups may contain different
                                   numbers of elements.

                                 * If no units are specified then the
                                   units of the coordinates are assumed.

                                 * If an element is selected by two or
                                   more queries then the latest one in the
                                   sequence defines which group it will be
                                   in.

                `int`            Define groups that contain the given
                                 number of elements. The first group
                                 starts with the first axis element and
                                 spans the defined number of consecutive
                                 elements. Each subsequent group
                                 immediately follows the preceding one.

                                 * By default each group has the defined
                                   number of elements, apart from the last
                                   group which may contain fewer elements
                                   (see the *group_span* parameter).

                `numpy.ndarray`  Define groups by selecting elements that
                                 map to the same value in the `numpy`
                                 array. The array must contain integers
                                 and have the same length as the axis to
                                 be collapsed and its sequence of values
                                 correspond to the axis elements. Each
                                 group contains the elements which
                                 correspond to a common non-negative
                                 integer value in the numpy array. Upon
                                 output, the collapsed axis is arranged in
                                 order of increasing group number. See the
                                 *regroup* parameter, which allows the
                                 creation of such a `numpy.array` for a
                                 given grouped collapse.

                                 * The groups do not have to be in runs of
                                   consecutive elements; they may be
                                   scattered throughout the axis.

                                 * An element which corresponds to a
                                   negative integer in the array will not
                                   be in any group.
                ===============  =========================================

                *Parameter example:*
                  To define groups of 10 kilometres: ``group=cf.Data(10,
                  'km')``.

                *Parameter example:*
                  To define groups of 5 days, starting and ending at
                  midnight on each day: ``group=cf.D(5)`` (see `cf.D`).

                *Parameter example:*
                  To define groups of 1 calendar month, starting and
                  ending at day 16 of each month: ``group=cf.M(day=16)``
                  (see `cf.M`).

                *Parameter example:*
                  To define groups of the season MAM in each year:
                  ``group=cf.mam()`` (see `cf.mam`).

                *Parameter example:*
                  To define groups of the seasons DJF and JJA in each
                  year: ``group=[cf.jja(), cf.djf()]``. To define groups
                  for seasons DJF, MAM, JJA and SON in each year:
                  ``group=cf.seasons()`` (see `cf.djf`, `cf.jja` and
                  `cf.season`).

                *Parameter example:*
                  To define groups for longitude elements less than or
                  equal to 90 degrees and greater than 90 degrees:
                  ``group=[cf.le(90, 'degrees'), cf.gt(90, 'degrees')]``
                  (see `cf.le` and `cf.gt`).

                *Parameter example:*
                  To define groups of 5 elements: ``group=5``.

                *Parameter example:*
                  For an axis of size 8, create two groups, the first
                  containing the first and last elements and the second
                  containing the 3rd, 4th and 5th elements, whilst
                  ignoring the 2nd, 6th and 7th elements:
                  ``group=numpy.array([0, -1, 4, 4, 4, -1, -2, 0])``.

            regroup: `bool`, optional
                If True then, for grouped collapses, do not collapse the
                field construct, but instead return a `numpy.array` of
                integers which identifies the groups defined by the
                *group* parameter. Each group contains the elements which
                correspond to a common non-negative integer value in the
                numpy array. Elements corresponding to negative integers
                are not in any group. The array may subsequently be used
                as the value of the *group* parameter in a separate
                collapse.

                For example:

                >>> groups = f.collapse('time: mean', group=10, regroup=True)
                >>> g = f.collapse('time: mean', group=groups)

                is equivalent to:

                >>> g = f.collapse('time: mean', group=10)

            group_by: optional
                Specify how coordinates are assigned to the groups defined
                by the *group*, *within_days* or *within_years*
                parameters. Ignored unless one of these parameters is set
                to a `Data` or `TimeDuration` object.

                The *group_by* parameter may be one of:

                ============  ============================================
                *group_by*    Description
                ============  ============================================
                `None`        This is the default.

                              If the groups are defined by the *group*
                              parameter (i.e. collapses other than
                              climatological time collapses) then assume a
                              value of ``'coords'``.

                              If the groups are defined by the
                              *within_days* or *within_years* parameter
                              (i.e. climatological time collapses) then
                              assume a value of ``'bounds'``.

                ``'coords'``  Each group contains the axis elements whose
                              coordinate values lie within the group
                              limits. Every element will be in a group.

                ``'bounds'``  Each group contains the axis elements whose
                              upper and lower coordinate bounds both lie
                              within the group limits. Some elements may
                              not be inside any group, either because the
                              group limits do not coincide with coordinate
                              bounds or because the group size is
                              sufficiently small.
                ============  ============================================

            group_span: optional
                Specify how to treat groups that may not span the desired
                range. For example, when creating 3-month means, the
                *group_span* parameter can be used to allow groups which
                only contain 1 or 2 months of data.

                By default, *group_span* is `None`. This means that only
                groups whose span equals the size specified by the
                definition of the groups are collapsed; unless the groups
                have been defined by one or more `Query` objects, in which
                case then the default behaviour is to collapse all groups,
                regardless of their size.

                In effect, the *group_span* parameter defaults to `True`
                unless the groups have been defined by one or more `Query`
                objects, in which case *group_span* defaults to `False`.

                The different behaviour when the groups have been defined
                by one or more `Query` objects is necessary because a
                `Query` object can only define the composition of a group,
                and not its size (see the parameter examples below for how
                to specify a group span in this case).

                .. note:: Prior to version 3.1.0, the default value of
                          *group_span* was effectively `False`.

                In general, the span of a group is the absolute difference
                between the lower bound of its first element and the upper
                bound of its last element. The only exception to this
                occurs if *group_span* is (by default or by explicit
                setting) an integer, in which case the span of a group is
                the number of elements in the group. See also the
                *group_contiguous* parameter for how to deal with groups
                that have gaps in their coverage.

                The *group_span* parameter is only applied to groups
                defined by the *group*, *within_days* or *within_years*
                parameters, and is otherwise ignored.

                The *group_span* parameter may be one of:

                ==============  ==========================================
                *group_span*    Description
                ==============  ==========================================
                `None`          This is the default. Apply a value of
                                `True` or `False` depending on how the
                                groups have been defined.

                `True`          Ignore groups whose span is not equal to
                                the size specified by the definition of
                                the groups. Only applicable if the groups
                                are defined by a `Data`, `TimeDuration` or
                                `int` object, and this is the default in
                                this case.

                `False`         Collapse all groups, regardless of their
                                size. This is the default if the groups
                                are defined by one to more `Query`
                                objects.

                `Data`          Ignore groups whose span is not equal to
                                the given size. If no units are specified
                                then the units of the coordinates are
                                assumed.

                `TimeDuration`  Ignore groups whose span is not equals to
                                the given time duration.

                `int`           Ignore groups that contain fewer than the
                                given number of elements
                ==============  ==========================================

                *Parameter example:*
                  To collapse into groups of 10km, ignoring any groups
                  that span less than that distance: ``group=cf.Data(10,
                  'km'), group_span=True``.

                *Parameter example:*
                  To collapse a daily timeseries into monthly groups,
                  ignoring any groups that span less than 1 calendar
                  month: monthly values: ``group=cf.M(), group_span=True``
                  (see `cf.M`).

                *Parameter example:*
                  To collapse a timeseries into seasonal groups, ignoring
                  any groups that span less than three months:
                  ``group=cf.seasons(), group_span=cf.M(3)`` (see
                  `cf.seasons` and `cf.M`).

            group_contiguous: `int`, optional
                Specify how to treat groups whose elements are not
                contiguous or have overlapping cells. For example, when
                creating a December to February means, the
                *group_contiguous* parameter can be used to allow groups
                which have no data for January.

                A group is considered to be contiguous unless it has
                coordinates with bounds that do not coincide for adjacent
                cells. The definition may be expanded to include groups
                whose coordinate bounds that overlap.

                By default *group_contiguous* is ``1``, meaning that
                non-contiguous groups, and those whose coordinate bounds
                overlap, are not collapsed

                .. note:: Prior to version 3.1.0, the default value of
                          *group_contiguous* was ``0``.

                The *group_contiguous* parameter is only applied to groups
                defined by the *group*, *within_days* or *within_years*
                parameters, and is otherwise ignored.

                The *group_contiguous* parameter may be one of:

                ===================  =====================================
                *group_contiguous*   Description
                ===================  =====================================
                ``0``                Allow non-contiguous groups, and
                                     those containing overlapping cells.

                ``1``                This is the default. Ignore
                                     non-contiguous groups, as well as
                                     contiguous groups containing
                                     overlapping cells.

                ``2``                Ignore non-contiguous groups,
                                     allowing contiguous groups containing
                                     overlapping cells.
                ===================  =====================================

                *Parameter example:*
                  To allow non-contiguous groups, and those containing
                  overlapping cells: ``group_contiguous=0``.

            within_days: optional
                Define the groups for creating CF "within days"
                climatological statistics.

                Each group contains elements whose coordinates span a time
                interval of up to one day. The results of the collapses
                are concatenated so that the output axis has a size equal
                to the number of groups.

                .. note:: For CF compliance, a "within days" collapse
                          should be followed by an "over days" collapse.

                The *within_days* parameter defines how the elements are
                partitioned into groups, and may be one of:

                ==============  ==========================================
                *within_days*   Description
                ==============  ==========================================
                `TimeDuration`  Defines the group size in terms of a time
                                interval of up to one day. The first group
                                starts at or before the first coordinate
                                bound of the first axis element (or its
                                coordinate if there are no bounds) and
                                spans the defined group size. Each
                                subsequent group immediately follows the
                                preceding one. By default each group
                                contains the consecutive run of elements
                                whose coordinate cells lie within the
                                group limits (see the *group_by*
                                parameter).

                                * Groups may contain different numbers of
                                  elements.

                                * The start of the first group may be
                                  before the first first axis element,
                                  depending on the offset defined by the
                                  time duration. For example, if
                                  ``group=cf.D(hour=12)`` then the first
                                  group will start on the closest midday
                                  to the first axis element.

                `Query`         Define groups from elements whose
                                coordinates satisfy the query
                                condition. Multiple groups are created:
                                one for each maximally consecutive run
                                within the selected elements.

                                If a sequence of `Query` is provided then
                                groups are defined for each query.

                                * Groups may contain different numbers of
                                  elements.

                                * If no units are specified then the units
                                  of the coordinates are assumed.

                                * If a coordinate does not satisfy any of
                                  the conditions then its element will not
                                  be in a group.

                                * If an element is selected by two or more
                                  queries then the latest one in the
                                  sequence defines which group it will be
                                  in.
                ==============  ==========================================

                *Parameter example:*
                  To define groups of 6 hours, starting at 00:00, 06:00,
                  12:00 and 18:00: ``within_days=cf.h(6)`` (see `cf.h`).

                *Parameter example:*
                  To define groups of 1 day, starting at 06:00:
                  ``within_days=cf.D(1, hour=6)`` (see `cf.D`).

                *Parameter example:*
                  To define groups of 00:00 to 06:00 within each day,
                  ignoring the rest of each day:
                  ``within_days=cf.hour(cf.le(6))`` (see `cf.hour` and
                  `cf.le`).

                *Parameter example:*
                  To define groups of 00:00 to 06:00 and 18:00 to 24:00
                  within each day, ignoring the rest of each day:
                  ``within_days=[cf.hour(cf.le(6)), cf.hour(cf.gt(18))]``
                  (see `cf.gt`, `cf.hour` and `cf.le`).

            within_years: optional
                Define the groups for creating CF "within years"
                climatological statistics.

                Each group contains elements whose coordinates span a time
                interval of up to one calendar year. The results of the
                collapses are concatenated so that the output axis has a
                size equal to the number of groups.

                .. note:: For CF compliance, a "within years" collapse
                          should be followed by an "over years" collapse.

                The *within_years* parameter defines how the elements are
                partitioned into groups, and may be one of:

                ==============  ==========================================
                *within_years*  Description
                ==============  ==========================================
                `TimeDuration`  Define the group size in terms of a time
                                interval of up to one calendar year. The
                                first group starts at or before the first
                                coordinate bound of the first axis element
                                (or its coordinate if there are no bounds)
                                and spans the defined group size. Each
                                subsequent group immediately follows the
                                preceding one. By default each group
                                contains the consecutive run of elements
                                whose coordinate cells lie within the
                                group limits (see the *group_by*
                                parameter).

                                * Groups may contain different numbers of
                                  elements.

                                * The start of the first group may be
                                  before the first first axis element,
                                  depending on the offset defined by the
                                  time duration. For example, if
                                  ``group=cf.Y(month=12)`` then the first
                                  group will start on the closest 1st
                                  December to the first axis element.

                 `Query`        Define groups from elements whose
                                coordinates satisfy the query
                                condition. Multiple groups are created:
                                one for each maximally consecutive run
                                within the selected elements.

                                If a sequence of `Query` is provided then
                                groups are defined for each query.

                                * The first group may start outside of the
                                  range of coordinates (the start of the
                                  first group is controlled by parameters
                                  of the `TimeDuration`).

                                * If group boundaries do not coincide with
                                  coordinate bounds then some elements may
                                  not be inside any group.

                                * If the group size is sufficiently small
                                  then some elements may not be inside any
                                  group.

                                * Groups may contain different numbers of
                                  elements.
                ==============  ==========================================

                *Parameter example:*
                  To define groups of 90 days: ``within_years=cf.D(90)``
                  (see `cf.D`).

                *Parameter example:*
                  To define groups of 3 calendar months, starting on the
                  15th of a month: ``within_years=cf.M(3, day=15)`` (see
                  `cf.M`).

                *Parameter example:*
                  To define groups for the season MAM within each year:
                  ``within_years=cf.mam()`` (see `cf.mam`).

                *Parameter example:*
                  To define groups for February and for November to
                  December within each year: ``within_years=[cf.month(2),
                  cf.month(cf.ge(11))]`` (see `cf.month` and `cf.ge`).

            over_days: optional
                Define the groups for creating CF "over days"
                climatological statistics.

                By default (or if *over_days* is `None`) each group
                contains all elements for which the time coordinate cell
                lower bounds have a common time of day but different
                dates, and for which the time coordinate cell upper bounds
                also have a common time of day but different dates. The
                collapsed dime axis will have a size equal to the number
                of groups that were found.

                For example, elements corresponding to the two time
                coordinate cells

                  | ``1999-12-31 06:00:00/1999-12-31 18:00:00``
                  | ``2000-01-01 06:00:00/2000-01-01 18:00:00``

                would be together in a group; and elements corresponding
                to the two time coordinate cells

                  | ``1999-12-31 00:00:00/2000-01-01 00:00:00``
                  | ``2000-01-01 00:00:00/2000-01-02 00:00:00``

                would also be together in a different group.

                .. note:: For CF compliance, an "over days" collapse
                          should be preceded by a "within days" collapse.

                The default groups may be split into smaller groups if the
                *over_days* parameter is one of:

                ==============  ==========================================
                *over_days*     Description
                ==============  ==========================================
                `TimeDuration`  Split each default group into smaller
                                groups which span the given time duration,
                                which must be at least one day.

                                * Groups may contain different numbers of
                                  elements.

                                * The start of the first group may be
                                  before the first first axis element,
                                  depending on the offset defined by the
                                  time duration. For example, if
                                  ``group=cf.M(day=15)`` then the first
                                  group will start on the closest 15th of
                                  a month to the first axis element.

                `Query`         Split each default group into smaller
                                groups whose coordinate cells satisfy the
                                query condition.

                                If a sequence of `Query` is provided then
                                groups are defined for each query.

                                * Groups may contain different numbers of
                                  elements.

                                * If a coordinate does not satisfy any of
                                  the conditions then its element will not
                                  be in a group.

                                * If an element is selected by two or more
                                  queries then the latest one in the
                                  sequence defines which group it will be
                                  in.
                ==============  ==========================================

                *Parameter example:*
                  To define groups for January and for June to December,
                  ignoring all other months: ``over_days=[cf.month(1),
                  cf.month(cf.wi(6, 12))]`` (see `cf.month` and `cf.wi`).

                *Parameter example:*
                  To define groups spanning 90 days:
                  ``over_days=cf.D(90)`` or ``over_days=cf.h(2160)``. (see
                  `cf.D` and `cf.h`).

                *Parameter example:*
                  To define groups that each span 3 calendar months,
                  starting and ending at 06:00 in the first day of each
                  month: ``over_days=cf.M(3, hour=6)`` (see `cf.M`).

                *Parameter example:*
                  To define groups that each span a calendar month
                  ``over_days=cf.M()`` (see `cf.M`).

                *Parameter example:*
                  To define groups for January and for June to December,
                  ignoring all other months: ``over_days=[cf.month(1),
                  cf.month(cf.wi(6, 12))]`` (see `cf.month` and `cf.wi`).

            over_years: optional
                Define the groups for creating CF "over years"
                climatological statistics.

                By default (or if *over_years* is `None`) each group
                contains all elements for which the time coordinate cell
                lower bounds have a common date of the year but different
                years, and for which the time coordinate cell upper bounds
                also have a common date of the year but different
                years. The collapsed dime axis will have a size equal to
                the number of groups that were found.

                For example, elements corresponding to the two time
                coordinate cells

                  | ``1999-12-01 00:00:00/2000-01-01 00:00:00``
                  | ``2000-12-01 00:00:00/2001-01-01 00:00:00``

                would be together in a group.

                .. note:: For CF compliance, an "over years" collapse
                          should be preceded by a "within years" or "over
                          days" collapse.

                The default groups may be split into smaller groups if the
                *over_years* parameter is one of:

                ==============  ==========================================
                *over_years*    Description
                ==============  ==========================================
                `TimeDuration`  Split each default group into smaller
                                groups which span the given time duration,
                                which must be at least one day.

                                * Groups may contain different numbers of
                                  elements.

                                * The start of the first group may be
                                  before the first first axis element,
                                  depending on the offset defined by the
                                  time duration. For example, if
                                  ``group=cf.Y(month=12)`` then the first
                                  group will start on the closest 1st
                                  December to the first axis element.

                `Query`         Split each default group into smaller
                                groups whose coordinate cells satisfy the
                                query condition.

                                If a sequence of `Query` is provided then
                                groups are defined for each query.

                                * Groups may contain different numbers of
                                  elements.

                                * If a coordinate does not satisfy any of
                                  the conditions then its element will not
                                  be in a group.

                                * If an element is selected by two or more
                                  queries then the latest one in the
                                  sequence defines which group it will be
                                  in.
                ==============  ==========================================

                *Parameter example:*
                  An element with coordinate bounds {1999-06-01 06:00:00,
                  1999-09-01 06:00:00} **matches** an element with
                  coordinate bounds {2000-06-01 06:00:00, 2000-09-01
                  06:00:00}.

                *Parameter example:*
                  An element with coordinate bounds {1999-12-01 00:00:00,
                  2000-12-01 00:00:00} **matches** an element with
                  coordinate bounds {2000-12-01 00:00:00, 2001-12-01
                  00:00:00}.

                *Parameter example:*
                  To define groups spanning 10 calendar years:
                  ``over_years=cf.Y(10)`` or ``over_years=cf.M(120)`` (see
                  `cf.M` and `cf.Y`).

                *Parameter example:*
                  To define groups spanning 5 calendar years, starting and
                  ending at 06:00 on 01 December of each year:
                  ``over_years=cf.Y(5, month=12, hour=6)`` (see `cf.Y`).

                *Parameter example:*
                  To define one group spanning 1981 to 1990 and another
                  spanning 2001 to 2005: ``over_years=[cf.year(cf.wi(1981,
                  1990), cf.year(cf.wi(2001, 2005)]`` (see `cf.year` and
                  `cf.wi`).

            remove_vertical_crs: `bool`, optional
                If True, the default, then remove a vertical
                coordinate reference construct and all of its domain
                ancillary constructs if any of its coordinate
                constructs or domain ancillary constructs span any
                collapse axes.

                If False then only the vertical coordinate reference
                construct's domain ancillary constructs that span any
                collapse axes are removed, but the vertical coordinate
                reference construct remains. This could result in
                `compute_vertical_coordinates` returning incorrect
                non-parametric vertical coordinate values.

                .. versionadded:: 3.14.1

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field` or `numpy.ndarray`
                 The collapsed field construct. Alternatively, if the
                 *regroup* parameter is True then a `numpy` array is
                 returned.

        **Examples**

        There are further worked examples in
        https://ncas-cms.github.io/cf-python/analysis.html#statistical-collapses

        """
        if _debug:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "collapse",
                {"_debug": _debug},
                "Use keyword 'verbose' instead.",
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "collapse", kwargs, version="3.0.0", removed_at="4.0.0"
            )  # pragma: no cover

        debug = is_log_level_debug(logger)

        if inplace:
            f = self
        else:
            f = self.copy()

        # Whether or not to create null bounds for null
        # collapses. I.e. if the collapse axis has size 1 and no
        # bounds, whether or not to create upper and lower bounds to
        # the coordinate value. If this occurs it's because the null
        # collapse is part of a grouped collapse and so will be
        # concatenated to other collapses for the final result: bounds
        # will be made for the grouped collapse, so all elements need
        # bounds.
        #        _create_zero_size_cell_bounds = kwargs.get(
        #            '_create_zero_size_cell_bounds', False)

        # ------------------------------------------------------------
        # Parse the methods and axes
        # ------------------------------------------------------------
        if measure and scale is not None:
            raise ValueError(
                "'scale' must be None when 'measure' is True. "
                f"Got: scale={scale!r}"
            )

        if ":" in method:
            # Convert a cell methods string (such as 'area: mean dim3:
            # dim2: max T: minimum height: variance') to a CellMethod
            # construct
            if axes is not None:
                raise ValueError(
                    "Can't collapse: Can't set 'axes' when 'method' is "
                    "CF-like cell methods string"
                )

            all_methods = []
            all_axes = []
            all_within = []
            all_over = []

            for cm in CellMethod.create(method):
                all_methods.append(cm.get_method(None))
                all_axes.append(cm.get_axes(()))
                all_within.append(cm.get_qualifier("within", None))
                all_over.append(cm.get_qualifier("over", None))
        else:
            x = method.split(" within ")
            if method == x[0]:
                within = None
                x = method.split(" over ")
                if method == x[0]:
                    over = None
                else:
                    method, over = x
            else:
                method, within = x

            if isinstance(axes, (str, int)):
                axes = (axes,)

            all_methods = (method,)
            all_within = (within,)
            all_over = (over,)
            all_axes = (axes,)

        # ------------------------------------------------------------
        # Convert axes into domain axis construct keys
        # ------------------------------------------------------------
        domain_axes = None

        input_axes = all_axes
        all_axes = []
        for axes in input_axes:
            if axes is None:
                domain_axes = self.domain_axes(
                    todict=False, cached=domain_axes
                )
                all_axes.append(list(domain_axes))
                continue

            axes2 = []
            for axis in axes:
                msg = (
                    "Must have '{}' axes for an '{}' collapse. Can't "
                    "find {{!r}} axis"
                )
                if axis == "area":
                    iterate_over = ("X", "Y")
                    msg = msg.format("', '".join(iterate_over), axis)
                elif axis == "volume":
                    iterate_over = ("X", "Y", "Z")
                    msg = msg.format("', '".join(iterate_over), axis)
                else:
                    iterate_over = (axis,)
                    msg = "Can't find the collapse axis identified by {!r}"

                for x in iterate_over:
                    a = self.domain_axis(x, key=True, default=None)
                    if a is None:
                        raise ValueError(msg.format(x))

                    axes2.append(a)

            all_axes.append(axes2)

        if debug:
            logger.debug(
                "    all_methods, all_axes, all_within, all_over = "
                f"{all_methods} {all_axes} {all_within} {all_over}"
            )  # pragma: no cover

        if group is not None and len(all_axes) > 1:
            raise ValueError(
                "Can't use the 'group' parameter for multiple collapses"
            )

        # ------------------------------------------------------------
        #
        # ------------------------------------------------------------
        domain_axes = f.domain_axes(todict=False, cached=domain_axes)

        for method, axes, within, over, axes_in in zip(
            all_methods, all_axes, all_within, all_over, input_axes
        ):
            method2 = _collapse_methods.get(method, None)
            if method2 is None:
                raise ValueError(f"Unknown collapse method: {method!r}")

            method = method2
            collapse_axes_all_sizes = f.domain_axes(
                filter_by_key=axes, todict=False
            )

            if debug:
                logger.debug(
                    f"    axes                    = {axes}"
                )  # pragma: no cover
                logger.debug(
                    f"    method                  = {method}"
                )  # pragma: no cover
                logger.debug(
                    f"    collapse_axes_all_sizes = {collapse_axes_all_sizes}"
                )  # pragma: no cover

            if not collapse_axes_all_sizes:
                raise ValueError(
                    "Can't collapse: Can not identify collapse axes"
                )

            if method in (
                "sum_of_weights",
                "sum_of_weights2",
                "sample_size",
                "integral",
                "maximum_absolute_value",
                "minimum_absolute_value",
                "mean_absolute_value",
                "range",
                "root_mean_square",
                "sum_of_squares",
            ):
                collapse_axes = collapse_axes_all_sizes.todict()  # copy()
            else:
                collapse_axes = collapse_axes_all_sizes.filter_by_size(
                    gt(1), todict=True
                )

            if debug:
                logger.debug(
                    f"    collapse_axes           = {collapse_axes}"
                )  # pragma: no cover

            if not collapse_axes:
                # Do nothing if there are no collapse axes
                if _create_zero_size_cell_bounds:
                    # Create null bounds if requested
                    for axis in axes:
                        #                        dc = f.dimension_coordinates(
                        #                            filter_by_axis=(axis,), axis_mode="and", todict=Tru#e
                        #                        ).value(None)
                        dc = f.dimension_coordinate(
                            filter_by_axis=(axis,), default=None
                        )
                        if dc is not None and not dc.has_bounds():
                            dc.set_bounds(dc.create_bounds(cellsize=0))

                continue

            # Check that there are enough elements to collapse
            collapse_axes_sizes = [
                da.get_size() for da in collapse_axes.values()
            ]
            size = reduce(operator_mul, collapse_axes_sizes, 1)

            if debug:
                logger.debug(
                    f"    collapse_axes_sizes     = {collapse_axes_sizes}"
                )  # pragma: no cover

            grouped_collapse = (
                within is not None or over is not None or group is not None
            )

            # --------------------------------------------------------
            # Set the group_by parameter
            # --------------------------------------------------------
            if group_by is None:
                if within is None and over is None:
                    group_by = "coords"
                else:
                    group_by = "bounds"
            elif (
                within is not None or over is not None
            ) and group_by == "coords":
                raise ValueError(
                    "Can't collapse: group_by parameter can't be "
                    "'coords' for a climatological time collapse."
                )

            # --------------------------------------------------------
            # Set the coordinate parameter
            # --------------------------------------------------------
            if coordinate is None and over is None:
                coordinate = "mid_range"

            if grouped_collapse:
                if len(collapse_axes) > 1:
                    raise ValueError(
                        "Can't do a grouped collapse on multiple axes "
                        "simultaneously"
                    )

                axis = [a for a in collapse_axes][0]

                # ------------------------------------------------------------
                # Grouped collapse: Calculate weights
                # ------------------------------------------------------------
                g_weights = weights
                if method not in _collapse_weighted_methods:
                    g_weights = None
                else:
                    if measure:
                        if method not in (
                            "integral",
                            "sum_of_weights",
                            "sum_of_weights2",
                        ):
                            raise ValueError(
                                f"Can't set measure=True for {method!r} "
                                "collapses"
                            )
                    elif method == "integral":
                        raise ValueError(
                            f"Must set measure=True for {method!r} collapses"
                        )

                    g_weights = f.weights(
                        weights,
                        components=True,
                        axes=list(collapse_axes),
                        scale=scale,
                        measure=measure,
                        radius=radius,
                        great_circle=great_circle,
                    )
                    if g_weights:
                        # For grouped collapses, bring the weights
                        # into memory. This is to prevent lazy
                        # operations being run on the entire weights
                        # array for every group.
                        iaxes = (self.get_data_axes().index(axis),)
                        if iaxes in g_weights:
                            g_weights[iaxes] = g_weights[iaxes].persist()
                    else:
                        g_weights = None

                f = f._collapse_grouped(
                    method,
                    axis,
                    within=within,
                    over=over,
                    within_days=within_days,
                    within_years=within_years,
                    over_days=over_days,
                    over_years=over_years,
                    group=group,
                    group_span=group_span,
                    group_contiguous=group_contiguous,
                    regroup=regroup,
                    mtol=mtol,
                    ddof=ddof,
                    measure=measure,
                    weights=g_weights,
                    squeeze=squeeze,
                    coordinate=coordinate,
                    group_by=group_by,
                    axis_in=axes_in[0],
                    verbose=verbose,
                )

                if regroup:
                    # Grouped collapse: Return the numpy array
                    return f

                # ----------------------------------------------------
                # Grouped collapse: Update the cell methods
                # ----------------------------------------------------
                f._update_cell_methods(
                    method=method,
                    domain_axes=collapse_axes,
                    input_axes=axes_in,
                    within=within,
                    over=over,
                    verbose=verbose,
                )
                continue

            elif regroup:
                raise ValueError(
                    "Can't return an array of groups for a non-grouped "
                    "collapse"
                )

            data_axes = f.get_data_axes()
            iaxes = [
                data_axes.index(axis)
                for axis in collapse_axes
                if axis in data_axes
            ]

            # ------------------------------------------------------------
            # Calculate weights
            # ------------------------------------------------------------
            if debug:
                logger.debug(
                    f"    Input weights           = {weights!r}"
                )  # pragma: no cover

            if method not in _collapse_weighted_methods:
                weights = None

            d_kwargs = {}
            if weights is not None:
                if measure:
                    if method not in (
                        "integral",
                        "sum_of_weights",
                        "sum_of_weights2",
                    ):
                        raise ValueError(
                            f"Can't set measure=True for {method!r} collapses"
                        )
                elif method == "integral":
                    raise ValueError(
                        f"Must set measure=True for {method!r} collapses"
                    )

                d_weights = f.weights(
                    weights,
                    components=True,
                    axes=list(collapse_axes.keys()),
                    scale=scale,
                    measure=measure,
                    radius=radius,
                    great_circle=great_circle,
                )
                if d_weights:
                    d_kwargs["weights"] = d_weights

                if debug:
                    logger.debug(
                        f"    Output weights          = {d_weights!r}"
                    )  # pragma: no cover

            elif method == "integral":
                raise ValueError(
                    f"Must set the 'weights' parameter for {method!r} "
                    "collapses"
                )

            if method in _collapse_ddof_methods:
                d_kwargs["ddof"] = ddof

            # ========================================================
            # Collapse the data array
            # ========================================================
            if debug:
                logger.debug(
                    "  Before collapse of data:\n"
                    f"    iaxes, d_kwargs = {iaxes} {d_kwargs}\n"
                    f"    f.shape = {f.shape}\n"
                    f"    f.dtype = {f.dtype}\n"
                )  # pragma: no cover

            getattr(f.data, method)(
                axes=iaxes,
                squeeze=squeeze,
                mtol=mtol,
                inplace=True,
                **d_kwargs,
            )

            if squeeze:
                # ----------------------------------------------------
                # Remove the collapsed axes from the field's list of
                # data array axes
                # ----------------------------------------------------
                f.set_data_axes(
                    [axis for axis in data_axes if axis not in collapse_axes]
                )

            if debug:
                logger.debug(
                    "  After collapse of data:\n"
                    f"    f.shape = {f.shape}\n"
                    f"    f.dtype = {f.dtype}\n"
                    f"collapse_axes = {collapse_axes}"
                )  # pragma: no cover

            # --------------------------------------------------------
            # Delete vertical coordinate references whose coordinates
            # and domain ancillaries span any of the collapse
            # axes. Also delete the corresponding domain ancillaries.
            #
            # This is because missing domain ancillaries in a
            # coordinate refernce are assumed to have the value zero,
            # which is most likely inapproriate.
            # --------------------------------------------------------
            if remove_vertical_crs:
                for ref_key, ref in f.coordinate_references(
                    todict=True
                ).items():
                    if (
                        "standard_name"
                        not in ref.coordinate_conversion.parameters()
                    ):
                        # This is not a vertical CRS
                        continue

                    ref_axes = []
                    axes = f.constructs.data_axes()
                    for c_key in ref.coordinates():
                        ref_axes.extend(axes[c_key])

                    for (
                        da_key
                    ) in (
                        ref.coordinate_conversion.domain_ancillaries().values()
                    ):
                        ref_axes.extend(axes.get(da_key, ()))

                    if set(ref_axes).intersection(flat(all_axes)):
                        f.del_coordinate_reference(ref_key)

            # ---------------------------------------------------------
            # Update dimension coordinates, auxiliary coordinates,
            # cell measures and domain ancillaries
            # ---------------------------------------------------------
            for axis, domain_axis in collapse_axes.items():
                # Ignore axes which are already size 1
                size = domain_axis.get_size()
                if size == 1:
                    continue

                # REMOVE all cell measures and domain ancillaries
                # which span this axis
                c = f.constructs.filter(
                    filter_by_type=("cell_measure", "domain_ancillary"),
                    filter_by_axis=(axis,),
                    axis_mode="or",
                    todict=True,
                )
                for key, value in c.items():
                    if debug:
                        logger.debug(
                            f"    Removing {value.construct_type}"
                        )  # pragma: no cover

                    f.del_construct(key)

                # REMOVE all 2+ dimensional auxiliary coordinates
                # which span this axis
                #                c = auxiliary_coordinates.filter_by_naxes(gt(1), view=True)
                c = f.auxiliary_coordinates(
                    filter_by_naxes=(gt(1),),
                    filter_by_axis=(axis,),
                    axis_mode="or",
                    todict=True,
                )
                for key, value in c.items():
                    if debug:
                        logger.debug(
                            f"    Removing {value.construct_type} {key!r}"
                        )  # pragma: no cover

                    f.del_construct(key)

                # REMOVE all 1 dimensional auxiliary coordinates which
                # span this axis and have different values in their
                # data array and bounds.
                #
                # KEEP, after changing their data arrays, all
                # one-dimensional auxiliary coordinates which span
                # this axis and have the same values in their data
                # array and bounds.
                c = f.auxiliary_coordinates(
                    filter_by_axis=(axis,), axis_mode="exact", todict=True
                )
                for key, aux in c.items():
                    if debug:
                        logger.debug(f"key = {key}")  # pragma: no cover

                    d = aux[0]

                    if aux.has_bounds() or (aux[:-1] != aux[1:]).any():
                        if debug:
                            logger.debug(
                                f"    Removing {aux.construct_type} {key!r}"
                            )  # pragma: no cover

                        f.del_construct(key)
                    else:
                        # Change the data array for this auxiliary
                        # coordinate
                        aux.set_data(d.data, copy=False)
                        if d.has_bounds():
                            aux.bounds.set_data(d.bounds.data, copy=False)

                # Reset the axis size
                f.domain_axes(todict=True)[axis].set_size(1)
                if debug:
                    logger.debug(
                        f"Changing axis size to 1: {axis}"
                    )  # pragma: no cover

                dim = f.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if dim is None:
                    continue

                # Create new dimension coordinate bounds
                if dim.has_bounds():
                    b = dim.bounds.data
                else:
                    b = dim.data

                cached_elements = b._get_cached_elements()
                try:
                    # Try to set the new bounds from cached values
                    bounds_data = Data(
                        [[cached_elements[0], cached_elements[-1]]],
                        dtype=b.dtype,
                        units=b.Units,
                    )
                except KeyError:
                    # Otherwise create the new bounds lazily
                    ndim = b.ndim
                    bounds_data = Data.concatenate(
                        [
                            b[(slice(0, 1, 1),) * ndim],
                            b[(slice(-1, None, 1),) * ndim],
                        ],
                        axis=-1,
                        copy=False,
                    )
                    if ndim == 1:
                        bounds_data.insert_dimension(0, inplace=True)

                bounds = self._Bounds(data=bounds_data)

                # Create a new dimension coordinate value
                if coordinate == "min":
                    coordinate = "minimum"
                    print(
                        "WARNING: coordinate='min' has been deprecated. "
                        "Use coordinate='minimum' instead."
                    )
                elif coordinate == "max":
                    coordinate = "maximum"
                    print(
                        "WARNING: coordinate='max' has been deprecated. "
                        "Use coordinate='maximum' instead."
                    )

                if coordinate == "mid_range":
                    data = bounds_data.mean(axes=1, weights=None, squeeze=True)
                elif coordinate == "minimum":
                    data = dim.data.min(squeeze=False)
                elif coordinate == "maximum":
                    data = dim.data.max(squeeze=False)
                else:
                    raise ValueError(
                        "Can't collapse: Bad parameter value: "
                        f"coordinate={coordinate!r}"
                    )

                dim.set_data(data, copy=False)
                dim.set_bounds(bounds, copy=False)

            # --------------------------------------------------------
            # Update the cell methods
            # --------------------------------------------------------
            if _update_cell_methods:
                f._update_cell_methods(
                    method,
                    domain_axes=collapse_axes,
                    input_axes=axes_in,
                    within=within,
                    over=over,
                    verbose=verbose,
                )

        # ------------------------------------------------------------
        # Return the collapsed field (or the classification array)
        # ------------------------------------------------------------
        return f

    @_manage_log_level_via_verbosity
    def _collapse_grouped(
        self,
        method,
        axis,
        within=None,
        over=None,
        within_days=None,
        within_years=None,
        over_days=None,
        over_years=None,
        group=None,
        group_span=None,
        group_contiguous=False,
        mtol=None,
        ddof=None,
        regroup=None,
        coordinate=None,
        measure=False,
        weights=None,
        squeeze=None,
        group_by=None,
        axis_in=None,
        verbose=None,
    ):
        """Implements a grouped collapse on a field.

        A grouped collapse is one for which an axis is not collapsed
        completely to size 1.

        :Parameters:

            method: `str`
                See `collapse` for details.

            measure: `bool`, optional
                See `collapse` for details.

            over: `str`
                See `collapse` for details.

            within: `str`
                See `collapse` for details.

        """

        def _ddddd(
            classification,
            n,
            lower,
            upper,
            increasing,
            coord,
            group_by_coords,
            extra_condition,
        ):
            """Returns configuration for a general collapse.

            :Parameter:

                extra_condition: `Query`

            :Returns:

                `numpy.ndarray`, `int`, date-time, date-time

            """
            if group_by_coords:
                q = ge(lower) & lt(upper)
            else:
                q = ge(lower, attr="lower_bounds") & le(
                    upper, attr="upper_bounds"
                )

            if extra_condition:
                q &= extra_condition

            index = q.evaluate(coord).array
            classification[index] = n

            if increasing:
                lower = upper
            else:
                upper = lower

            n += 1

            return classification, n, lower, upper

        def _time_interval(
            classification,
            n,
            coord,
            interval,
            lower,
            upper,
            lower_limit,
            upper_limit,
            group_by,
            extra_condition=None,
        ):
            """Prepares for a collapse where the group is a
            TimeDuration.

            :Parameters:

                classification: `numpy.ndarray`

                n: `int`

                coord: `DimensionCoordinate`

                interval: `TimeDuration`

                lower: date-time object

                upper: date-time object

                lower_limit: `datetime`

                upper_limit: `datetime`

                group_by: `str`

                extra_condition: `Query`, optional

            :Returns:

                (`numpy.ndarray`, `int`)

            """
            group_by_coords = group_by == "coords"

            if coord.increasing:
                # Increasing dimension coordinate
                lower, upper = interval.bounds(lower)
                while lower <= upper_limit:
                    lower, upper = interval.interval(lower)
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        True,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )
            else:
                # Decreasing dimension coordinate
                lower, upper = interval.bounds(upper)
                while upper >= lower_limit:
                    lower, upper = interval.interval(upper, end=True)
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        False,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )

            return classification, n

        def _time_interval_over(
            classification,
            n,
            coord,
            interval,
            lower,
            upper,
            lower_limit,
            upper_limit,
            group_by,
            extra_condition=None,
        ):
            """Prepares for a collapse over some TimeDuration.

            :Parameters:

                classification: `numpy.ndarray`

                n: `int`

                coord: `DimensionCoordinate`

                interval: `TimeDuration`

                lower: date-time

                upper: date-time

                lower_limit: date-time

                upper_limit: date-time

                group_by: `str`

                extra_condition: `Query`, optional

            :Returns:

                (`numpy.ndarray`, `int`)

            """
            group_by_coords = group_by == "coords"

            if coord.increasing:
                # Increasing dimension coordinate
                # lower, upper = interval.bounds(lower)
                upper = interval.interval(upper)[1]
                while lower <= upper_limit:
                    lower, upper = interval.interval(lower)
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        True,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )
            else:
                # Decreasing dimension coordinate
                # lower, upper = interval.bounds(upper)
                lower = interval.interval(upper, end=True)[0]
                while upper >= lower_limit:
                    lower, upper = interval.interval(upper, end=True)
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        False,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )

            return classification, n

        def _data_interval(
            classification,
            n,
            coord,
            interval,
            lower,
            upper,
            lower_limit,
            upper_limit,
            group_by,
            extra_condition=None,
        ):
            """Prepares for a collapse where the group is a data
            interval.

            :Returns:

                `numpy.ndarray`, `int`

            """
            group_by_coords = group_by == "coords"

            if coord.increasing:
                # Increasing dimension coordinate
                lower = lower.squeeze()
                while lower <= upper_limit:
                    upper = lower + interval
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        True,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )
            else:
                # Decreasing dimension coordinate
                upper = upper.squeeze()
                while upper >= lower_limit:
                    lower = upper - interval
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        False,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )

            return classification, n

        def _selection(
            classification,
            n,
            coord,
            selection,
            parameter,
            extra_condition=None,
            group_span=None,
            within=False,
        ):
            """Processes a group selection.

            :Parameters:

                classification: `numpy.ndarray`

                n: `int`

                coord: `DimensionCoordinate`

                selection: sequence of `Query`

                parameter: `str`
                    The name of the `cf.Field.collapse` parameter which
                    defined *selection*. This is used in error messages.

                    *Parameter example:*
                      ``parameter='within_years'``

                extra_condition: `Query`, optional

            :Returns:

                `numpy.ndarray`, `int`

            """
            # Create an iterator for stepping through each Query in
            # the selection sequence
            try:
                iterator = iter(selection)
            except TypeError:
                raise ValueError(
                    "Can't collapse: Bad parameter value: "
                    f"{parameter}={selection!r}"
                )

            for condition in iterator:
                if not isinstance(condition, Query):
                    raise ValueError(
                        f"Can't collapse: {parameter} sequence contains a "
                        f"non-{Query.__name__} object: {condition!r}"
                    )

                if extra_condition is not None:
                    condition &= extra_condition

                boolean_index = condition.evaluate(coord).array

                classification[boolean_index] = n
                n += 1

            #                if group_span is not None:
            #                    x = np.where(classification==n)[0]
            #                    for i in range(1, max(1, int(float(len(x))/group_span))):
            #                        n += 1
            #                        classification[x[i*group_span:(i + 1)*group_span]] = n
            #                n += 1

            return classification, n

        def _discern_runs(classification, within=False):
            """Processes a group classification.

            :Parameters:

                classification: `numpy.ndarray`

            :Returns:

                `numpy.ndarray`

            """
            x = np.where(np.diff(classification))[0] + 1
            if not x.size:
                if classification[0] >= 0:
                    classification[:] = 0

                return classification

            if classification[0] >= 0:
                classification[0 : x[0]] = 0

            n = 1
            for i, j in zip(x[:-1], x[1:]):
                if classification[i] >= 0:
                    classification[i:j] = n
                    n += 1

            if classification[x[-1]] >= 0:
                classification[x[-1] :] = n
                n += 1

            return classification

        def _discern_runs_within(classification, coord):
            """Processes group classification for a 'within'
            collapse."""
            size = classification.size
            if size < 2:
                return classification

            n = classification.max() + 1

            start = 0
            for i, c in enumerate(classification[: size - 1]):
                if c < 0:
                    continue

                if not coord[i : i + 2].contiguous(overlap=False):
                    classification[start : i + 1] = n
                    start = i + 1
                    n += 1

            return classification

        def _tyu(coord, group_by, time_interval):
            """Returns bounding values and limits for a general
            collapse.

            :Parameters:

                coord: `DimensionCoordinate`
                    The dimension coordinate construct associated with
                    the collapse.

                group_by: `str`
                    As for the *group_by* parameter of the `collapse` method.

                time_interval: `bool`
                    If True then then return a tuple of date-time
                    objects. If False return a tuple of `Data` objects.

            :Returns:

                `tuple`
                    A tuple of 4 `Data` object or, if *time_interval* is
                    True, a tuple of 4 date-time objects.

            """
            bounds = coord.get_bounds(None)
            if bounds is not None:
                lower_bounds = coord.lower_bounds
                upper_bounds = coord.upper_bounds
                lower = lower_bounds[0]
                upper = upper_bounds[0]
                lower_limit = lower_bounds[-1]
                upper_limit = upper_bounds[-1]
            elif group_by == "coords":
                if coord.increasing:
                    lower = coord.data[0]
                    upper = coord.data[-1]
                else:
                    lower = coord.data[-1]
                    upper = coord.data[0]

                lower_limit = lower
                upper_limit = upper
            else:
                raise ValueError(
                    f"Can't collapse: {coord.identity()!r} coordinate bounds "
                    f"are required with group_by={group_by!r}"
                )

            if time_interval:
                units = coord.Units
                if units.isreftime:
                    lower = lower.datetime_array[0]
                    upper = upper.datetime_array[0]
                    lower_limit = lower_limit.datetime_array[0]
                    upper_limit = upper_limit.datetime_array[0]
                elif not units.istime:
                    raise ValueError(
                        f"Can't group by {TimeDuration.__class__.__name__} "
                        f"when coordinates have units {coord.Units!r}"
                    )

            return (lower, upper, lower_limit, upper_limit)

        def _group_weights(weights, iaxis, index):
            """Subspaces weights components.

            :Parameters:

                weights: `dict` or `None`

                iaxis: `int`

                index: `list`

            :Returns:

                `dict` or `None`

            **Examples**

            >>> print(weights)
            None
            >>> print(_group_weights(weights, 2, [2, 3, 40]))
            None
            >>> print(_group_weights(weights, 1, slice(2, 56)))
            None

            >>> weights

            >>> _group_weights(weights, 2, [2, 3, 40])

            >>> _group_weights(weights, 1, slice(2, 56))

            """
            if not isinstance(weights, dict):
                return weights

            weights = weights.copy()
            for iaxes, value in weights.items():
                if iaxis in iaxes:
                    indices = [slice(None)] * len(iaxes)
                    indices[iaxes.index(iaxis)] = index
                    weights[iaxes] = value[tuple(indices)]
                    break

            return weights

        # START OF MAIN CODE

        debug = is_log_level_debug(logger)
        if debug:
            logger.debug(
                "    Grouped collapse:"
                f"        method            = {method!r}"
                f"        axis_in           = {axis_in!r}"
                f"        axis              = {axis!r}"
                f"        over              = {over!r}"
                f"        over_days         = {over_days!r}"
                f"        over_years        = {over_years!r}"
                f"        within            = {within!r}"
                f"        within_days       = {within_days!r}"
                f"        within_years      = {within_years!r}"
                f"        regroup           = {regroup!r}"
                f"        group             = {group!r}"
                f"        group_span        = {group_span!r}"
                f"        group_contiguous  = {group_contiguous!r}"
            )  # pragma: no cover

        # Size of uncollapsed axis
        axis_size = self.domain_axes(todict=True)[axis].get_size()
        # Integer position of collapse axis
        iaxis = self.get_data_axes().index(axis)

        fl = []

        # If group, rolling window, classification, etc, do something
        # special for size one axes - either return unchanged
        # (possibly mofiying cell methods with , e.g, within_days', or
        # raising an exception for 'can't match', I suppose.

        classification = None

        if group is not None:
            if within is not None or over is not None:
                raise ValueError(
                    "Can't set 'group' parameter for a climatological "
                    "collapse"
                )

            if isinstance(group, np.ndarray):
                classification = np.squeeze(group.copy())

                if classification.dtype.kind != "i":
                    raise ValueError(
                        "Can't group by numpy array of type "
                        f"{classification.dtype.name}"
                    )
                elif classification.shape != (axis_size,):
                    raise ValueError(
                        "Can't group by numpy array with incorrect "
                        f"shape: {classification.shape}"
                    )

                # Set group to None
                group = None

        if group is not None:
            if isinstance(group, Query):
                group = (group,)

            if isinstance(group, int):
                # ----------------------------------------------------
                # E.g. group=3
                # ----------------------------------------------------
                coord = None
                classification = np.empty((axis_size,), int)

                start = 0
                end = group
                n = 0
                while start < axis_size:
                    classification[start:end] = n
                    start = end
                    end += group
                    n += 1

                if group_span is True or group_span is None:
                    # Use the group definition as the group span
                    group_span = group

            elif isinstance(group, TimeDuration):
                # ----------------------------------------------------
                # E.g. group=cf.M()
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None:
                    raise ValueError(
                        "Dimension coordinates are required for a "
                        "grouped collapse with TimeDuration groups."
                    )

                classification = np.empty((axis_size,), int)
                classification.fill(-1)

                lower, upper, lower_limit, upper_limit = _tyu(
                    coord, group_by, True
                )

                classification, n = _time_interval(
                    classification,
                    0,
                    coord=coord,
                    interval=group,
                    lower=lower,
                    upper=upper,
                    lower_limit=lower_limit,
                    upper_limit=upper_limit,
                    group_by=group_by,
                )

                if group_span is True or group_span is None:
                    # Use the group definition as the group span
                    group_span = group

            elif isinstance(group, Data):
                # ----------------------------------------------------
                # Chunks of
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None:
                    axis_id = self.constructs.domain_axis_identity(axis)
                    raise ValueError(
                        f"Dimension coordinates for the {axis_id!r} axis are "
                        f"required for a collapse with group={group!r}"
                    )

                if coord.Units.isreftime:
                    axis_id = self.constructs.domain_axis_identity(axis)
                    raise ValueError(
                        f"Can't collapse reference-time axis {axis_id!r} "
                        f"with group={group!r}. In this case groups should "
                        "be defined with a TimeDuration instance."
                    )

                if group.size != 1:
                    raise ValueError(
                        "A Data instance 'group' parameter must have exactly "
                        f"one element: Got group={group!r}"
                    )

                if group.Units and not group.Units.equivalent(coord.Units):
                    axis_id = self.constructs.domain_axis_identity(axis)
                    raise ValueError(
                        f"Group units {group.Units!r} are not eqivalent to "
                        f"{axis_id!r} axis units {coord.Units!r}"
                    )

                classification = np.empty((axis_size,), int)
                classification.fill(-1)

                group = group.squeeze()

                lower, upper, lower_limit, upper_limit = _tyu(
                    coord, group_by, False
                )

                classification, n = _data_interval(
                    classification,
                    0,
                    coord=coord,
                    interval=group,
                    lower=lower,
                    upper=upper,
                    lower_limit=lower_limit,
                    upper_limit=upper_limit,
                    group_by=group_by,
                )

                if group_span is True or group_span is None:
                    # Use the group definition as the group span
                    group_span = group

            else:
                # ----------------------------------------------------
                # E.g. group=[cf.month(4), cf.month(cf.wi(9, 11))]
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None:
                    coord = self.auxiliary_coordinate(
                        filter_by_axis=(axis,), axis_mode="exact", default=None
                    )
                    if coord is None:
                        raise ValueError(
                            "Dimension and/or auxiliary coordinates are "
                            "required for a grouped collapse with a "
                            "sequence of groups."
                        )

                classification = np.empty((axis_size,), int)
                classification.fill(-1)

                classification, n = _selection(
                    classification,
                    0,
                    coord=coord,
                    selection=group,
                    parameter="group",
                )

                classification = _discern_runs(classification)

                if group_span is None:
                    group_span = False
                elif group_span is True:
                    raise ValueError(
                        "Can't collapse: Can't set group_span=True when "
                        f"group={group!r}"
                    )

        if classification is None:
            if over == "days":
                # ----------------------------------------------------
                # Over days
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None or not coord.Units.isreftime:
                    raise ValueError(
                        "Reference-time dimension coordinates are required "
                        "for an 'over days' collapse"
                    )

                if not coord.has_bounds():
                    raise ValueError(
                        "Reference-time dimension coordinate bounds are "
                        "required for an 'over days' collapse"
                    )

                cell_methods = self.cell_methods(todict=True)
                w = [
                    cm.get_qualifier("within", None)
                    for cm in cell_methods.values()
                ]
                if "days" not in w:
                    raise ValueError(
                        "An 'over days' collapse must come after a "
                        "'within days' cell method"
                    )

                # Parse the over_days parameter
                if isinstance(over_days, Query):
                    over_days = (over_days,)
                elif isinstance(over_days, TimeDuration):
                    if over_days.Units.istime and over_days < Data(1, "day"):
                        raise ValueError(
                            f"Bad parameter value: over_days={over_days!r}"
                        )

                coordinate = "minimum"

                classification = np.empty((axis_size,), int)
                classification.fill(-1)

                if isinstance(over_days, TimeDuration):
                    _, _, lower_limit, upper_limit = _tyu(
                        coord, "bounds", True
                    )

                bounds = coord.bounds
                lower_bounds = coord.lower_bounds.datetime_array
                upper_bounds = coord.upper_bounds.datetime_array

                HMS0 = None

                n = 0
                for lower, upper in zip(lower_bounds, upper_bounds):
                    HMS_l = (
                        eq(lower.hour, attr="hour")
                        & eq(lower.minute, attr="minute")
                        & eq(lower.second, attr="second")
                    ).addattr("lower_bounds")
                    HMS_u = (
                        eq(upper.hour, attr="hour")
                        & eq(upper.minute, attr="minute")
                        & eq(upper.second, attr="second")
                    ).addattr("upper_bounds")
                    HMS = HMS_l & HMS_u

                    if not HMS0:
                        HMS0 = HMS
                    elif HMS.equals(HMS0):
                        # We've got repeat of the first cell, which
                        # means that we must have now classified all
                        # cells. Therefore we can stop.
                        break

                    if debug:
                        logger.debug(
                            f"          HMS  = {HMS!r}"
                        )  # pragma: no cover

                    if over_days is None:
                        # --------------------------------------------
                        # over_days=None
                        # --------------------------------------------
                        # Over all days
                        index = HMS.evaluate(coord).array
                        classification[index] = n
                        n += 1
                    elif isinstance(over_days, TimeDuration):
                        # --------------------------------------------
                        # E.g. over_days=cf.M()
                        # --------------------------------------------
                        classification, n = _time_interval_over(
                            classification,
                            n,
                            coord=coord,
                            interval=over_days,
                            lower=lower,
                            upper=upper,
                            lower_limit=lower_limit,
                            upper_limit=upper_limit,
                            group_by="bounds",
                            extra_condition=HMS,
                        )
                    else:
                        # --------------------------------------------
                        # E.g. over_days=[cf.month(cf.wi(4, 9))]
                        # --------------------------------------------
                        classification, n = _selection(
                            classification,
                            n,
                            coord=coord,
                            selection=over_days,
                            parameter="over_days",
                            extra_condition=HMS,
                        )

            elif over == "years":
                # ----------------------------------------------------
                # Over years
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None or not coord.Units.isreftime:
                    raise ValueError(
                        "Reference-time dimension coordinates are required "
                        "for an 'over years' collapse"
                    )

                bounds = coord.get_bounds(None)
                if bounds is None:
                    raise ValueError(
                        "Reference-time dimension coordinate bounds are "
                        "required for an 'over years' collapse"
                    )

                cell_methods = self.cell_methods(todict=True)
                w = [
                    cm.get_qualifier("within", None)
                    for cm in cell_methods.values()
                ]
                o = [
                    cm.get_qualifier("over", None)
                    for cm in cell_methods.values()
                ]
                if "years" not in w and "days" not in o:
                    raise ValueError(
                        "An 'over years' collapse must come after a "
                        "'within years' or 'over days' cell method"
                    )

                # Parse the over_years parameter
                if isinstance(over_years, Query):
                    over_years = (over_years,)
                elif isinstance(over_years, TimeDuration):
                    if over_years.Units.iscalendartime:
                        over_years.Units = Units("calendar_years")
                        if not over_years.isint or over_years < 1:
                            raise ValueError(
                                "over_years is not a whole number of "
                                f"calendar years: {over_years!r}"
                            )
                    else:
                        raise ValueError(
                            "over_years is not a whole number of calendar "
                            f"years: {over_years!r}"
                        )

                coordinate = "minimum"

                classification = np.empty((axis_size,), int)
                classification.fill(-1)

                if isinstance(over_years, TimeDuration):
                    _, _, lower_limit, upper_limit = _tyu(
                        coord, "bounds", True
                    )

                lower_bounds = coord.lower_bounds.datetime_array
                upper_bounds = coord.upper_bounds.datetime_array
                mdHMS0 = None

                n = 0
                for lower, upper in zip(lower_bounds, upper_bounds):
                    mdHMS_l = (
                        eq(lower.month, attr="month")
                        & eq(lower.day, attr="day")
                        & eq(lower.hour, attr="hour")
                        & eq(lower.minute, attr="minute")
                        & eq(lower.second, attr="second")
                    ).addattr("lower_bounds")
                    mdHMS_u = (
                        eq(upper.month, attr="month")
                        & eq(upper.day, attr="day")
                        & eq(upper.hour, attr="hour")
                        & eq(upper.minute, attr="minute")
                        & eq(upper.second, attr="second")
                    ).addattr("upper_bounds")
                    mdHMS = mdHMS_l & mdHMS_u

                    if not mdHMS0:
                        # Keep a record of the first cell
                        mdHMS0 = mdHMS
                        if debug:
                            logger.debug(
                                f"        mdHMS0 = {mdHMS0!r}"
                            )  # pragma: no cover
                    elif mdHMS.equals(mdHMS0):
                        # We've got repeat of the first cell, which
                        # means that we must have now classified all
                        # cells. Therefore we can stop.
                        break

                    if debug:
                        logger.debug(
                            f"        mdHMS  = {mdHMS!r}"
                        )  # pragma: no cover

                    if over_years is None:
                        # --------------------------------------------
                        # over_years=None
                        # --------------------------------------------
                        # Over all years
                        index = mdHMS.evaluate(coord).array
                        classification[index] = n
                        n += 1
                    elif isinstance(over_years, TimeDuration):
                        # --------------------------------------------
                        # E.g. over_years=cf.Y(2)
                        # --------------------------------------------
                        classification, n = _time_interval_over(
                            classification,
                            n,
                            coord=coord,
                            interval=over_years,
                            lower=lower,
                            upper=upper,
                            lower_limit=lower_limit,
                            upper_limit=upper_limit,
                            group_by="bounds",
                            extra_condition=mdHMS,
                        )
                    else:
                        # --------------------------------------------
                        # E.g. over_years=cf.year(cf.lt(2000))
                        # --------------------------------------------
                        classification, n = _selection(
                            classification,
                            n,
                            coord=coord,
                            selection=over_years,
                            parameter="over_years",
                            extra_condition=mdHMS,
                        )

            elif within == "days":
                # ----------------------------------------------------
                # Within days
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None or not coord.Units.isreftime:
                    raise ValueError(
                        "Reference-time dimension coordinates are required "
                        "for an 'over years' collapse"
                    )

                bounds = coord.get_bounds(None)
                if bounds is None:
                    raise ValueError(
                        "Reference-time dimension coordinate bounds are "
                        "required for a 'within days' collapse"
                    )

                classification = np.empty((axis_size,), int)
                classification.fill(-1)

                # Parse the within_days parameter
                if isinstance(within_days, Query):
                    within_days = (within_days,)
                elif isinstance(within_days, TimeDuration):
                    if (
                        within_days.Units.istime
                        and TimeDuration(24, "hours") % within_days
                    ):
                        # % Data(1, 'day'): # % within_days:
                        raise ValueError(
                            f"Can't collapse: within_days={within_days!r} "
                            "is not an exact factor of 1 day"
                        )

                if isinstance(within_days, TimeDuration):
                    # ------------------------------------------------
                    # E.g. within_days=cf.h(6)
                    # ------------------------------------------------
                    lower, upper, lower_limit, upper_limit = _tyu(
                        coord, "bounds", True
                    )

                    classification, n = _time_interval(
                        classification,
                        0,
                        coord=coord,
                        interval=within_days,
                        lower=lower,
                        upper=upper,
                        lower_limit=lower_limit,
                        upper_limit=upper_limit,
                        group_by=group_by,
                    )

                    if group_span is True or group_span is None:
                        # Use the within_days definition as the group
                        # span
                        group_span = within_days

                else:
                    # ------------------------------------------------
                    # E.g. within_days=cf.hour(cf.lt(12))
                    # ------------------------------------------------
                    classification, n = _selection(
                        classification,
                        0,
                        coord=coord,
                        selection=within_days,
                        parameter="within_days",
                    )

                    classification = _discern_runs(classification)

                    classification = _discern_runs_within(
                        classification, coord
                    )

                    if group_span is None:
                        group_span = False
                    elif group_span is True:
                        raise ValueError(
                            "Can't collapse: Can't set group_span=True when "
                            f"within_days={within_days!r}"
                        )

            elif within == "years":
                # ----------------------------------------------------
                # Within years
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None or not coord.Units.isreftime:
                    raise ValueError(
                        "Can't collapse: Reference-time dimension "
                        'coordinates are required for a "within years" '
                        "collapse"
                    )

                if not coord.has_bounds():
                    raise ValueError(
                        "Can't collapse: Reference-time dimension coordinate "
                        'bounds are required for a "within years" collapse'
                    )

                classification = np.empty((axis_size,), int)
                classification.fill(-1)

                # Parse within_years
                if isinstance(within_years, Query):
                    within_years = (within_years,)
                elif within_years is None:
                    raise ValueError(
                        "Must set the within_years parameter for a "
                        '"within years" climatalogical time collapse'
                    )

                if isinstance(within_years, TimeDuration):
                    # ------------------------------------------------
                    # E.g. within_years=cf.M()
                    # ------------------------------------------------
                    lower, upper, lower_limit, upper_limit = _tyu(
                        coord, "bounds", True
                    )

                    classification, n = _time_interval(
                        classification,
                        0,
                        coord=coord,
                        interval=within_years,
                        lower=lower,
                        upper=upper,
                        lower_limit=lower_limit,
                        upper_limit=upper_limit,
                        group_by=group_by,
                    )

                    if group_span is True or group_span is None:
                        # Use the within_years definition as the group
                        # span
                        group_span = within_years

                else:
                    # ------------------------------------------------
                    # E.g. within_years=cf.season()
                    # ------------------------------------------------
                    classification, n = _selection(
                        classification,
                        0,
                        coord=coord,
                        selection=within_years,
                        parameter="within_years",
                        within=True,
                    )

                    classification = _discern_runs(classification, within=True)

                    classification = _discern_runs_within(
                        classification, coord
                    )

                    if group_span is None:
                        group_span = False
                    elif group_span is True:
                        raise ValueError(
                            "Can't collapse: Can't set group_span=True when "
                            f"within_years={within_years!r}"
                        )

            elif over is not None:
                raise ValueError(
                    f"Can't collapse: Bad 'over' syntax: {over!r}"
                )

            elif within is not None:
                raise ValueError(
                    f"Can't collapse: Bad 'within' syntax: {within!r}"
                )

        if classification is not None:
            # ---------------------------------------------------------
            # Collapse each group
            # ---------------------------------------------------------
            if debug:
                logger.debug(
                    f"        classification    = {classification}"
                )  # pragma: no cover

            unique = np.unique(classification)
            unique = unique[np.where(unique >= 0)[0]]
            unique.sort()

            ignore_n = -1
            for u in unique:
                index = np.where(classification == u)[0].tolist()

                pc = self.subspace(**{axis: index})

                # ----------------------------------------------------
                # Ignore groups that don't meet the specified criteria
                # ----------------------------------------------------
                if over is None:
                    coord = pc.coordinate(axis_in, default=None)

                    if group_span is not False:
                        if isinstance(group_span, int):
                            if (
                                pc.domain_axes(todict=True)[axis].get_size()
                                != group_span
                            ):
                                classification[index] = ignore_n
                                ignore_n -= 1
                                continue
                        else:
                            if coord is None:
                                axis_id = pc.constructs.domain_axis_identity(
                                    axis_in
                                )
                                raise ValueError(
                                    f"Can't collapse: No coordinates for "
                                    f"{axis_id!r} axis with group={group!r} "
                                    f"and group_span={group_span!r}"
                                )

                            bounds = coord.get_bounds(None)
                            if bounds is None:
                                raise ValueError(
                                    f"Can't collapse: No bounds on {coord!r} "
                                    f"with group={group!r} and "
                                    f"group_span={group_span!r}"
                                )

                            lb = bounds[0, 0].get_data(_fill_value=False)
                            ub = bounds[-1, 1].get_data(_fill_value=False)
                            if coord.T:
                                lb = lb.datetime_array.item()
                                ub = ub.datetime_array.item()

                            if not coord.increasing:
                                lb, ub = ub, lb

                            if group_span + lb != ub:
                                # The span of this group is not the
                                # same as group_span, so don't
                                # collapse it.
                                classification[index] = ignore_n
                                ignore_n -= 1
                                continue

                    if (
                        group_contiguous
                        and coord is not None
                        and coord.has_bounds()
                        and not coord.bounds.contiguous(
                            overlap=(group_contiguous == 2)
                        )
                    ):
                        # This group is not contiguous, so don't
                        # collapse it.
                        classification[index] = ignore_n
                        ignore_n -= 1
                        continue

                if regroup:
                    continue

                # ----------------------------------------------------
                # Still here? Then collapse the group
                # ----------------------------------------------------
                w = _group_weights(weights, iaxis, index)
                if debug:
                    logger.debug(
                        f"        Collapsing group {u}:"
                    )  # pragma: no cover

                fl.append(
                    pc.collapse(
                        method,
                        axis,
                        weights=w,
                        measure=measure,
                        mtol=mtol,
                        ddof=ddof,
                        coordinate=coordinate,
                        squeeze=False,
                        inplace=True,
                        _create_zero_size_cell_bounds=True,
                        _update_cell_methods=False,
                    )
                )

            if regroup:
                # return the numpy array
                return classification

        elif regroup:
            raise ValueError("There is no group classification to return.")

        # Still here?
        if not fl:
            c = "contiguous " if group_contiguous else ""
            s = f" spanning {group_span}" if group_span is not False else ""
            if within is not None:
                s = f" within {within}{s}"

            raise ValueError(
                f"Can't collapse: No {c}groups{s} were identified"
            )

        if len(fl) == 1:
            f = fl[0]
        else:
            # Hack to fix missing bounds!
            for g in fl:
                try:
                    c = g.dimension_coordinate(
                        filter_by_axis=(axis,), default=None
                    )
                    if not c.has_bounds():
                        c.set_bounds(c.create_bounds())
                except Exception:
                    pass

            # --------------------------------------------------------
            # Sort the list of collapsed fields
            # --------------------------------------------------------
            if (
                coord is not None
                and coord.construct_type == "dimension_coordinate"
            ):
                fl.sort(
                    key=lambda g: g.dimension_coordinate(
                        filter_by_axis=(axis,)
                    ).datum(0),
                    reverse=coord.decreasing,
                )

            # --------------------------------------------------------
            # Concatenate the partial collapses.
            #
            # Use cull_graph=True to prevent dask failures arising
            # from concatenating graphs with lots of unused nodes.
            # --------------------------------------------------------
            try:
                f = self.concatenate(fl, axis=iaxis, cull_graph=True)
            except ValueError as error:
                raise ValueError(f"Can't collapse: {error}")

        if squeeze and f.domain_axes(todict=True)[axis].get_size() == 1:
            # Remove a totally collapsed axis from the field's
            # data array
            f.squeeze(axis, inplace=True)

        # ------------------------------------------------------------
        # Return the collapsed field
        # ------------------------------------------------------------
        self.__dict__ = f.__dict__
        if debug:
            logger.debug("    End of grouped collapse")  # pragma: no cover

        return self

    def _update_cell_methods(
        self,
        method=None,
        domain_axes=None,
        input_axes=None,
        within=None,
        over=None,
        verbose=None,
    ):
        """Update the cell methods.

        :Parameters:

            method: `str`

            domain_axes: `Constructs` or `dict`

            {{verbose: `int` or `str` or `None`, optional}}

        :Returns:

            `None`

        """
        debug = is_log_level_debug(logger)

        original_cell_methods = self.cell_methods(todict=True)
        if debug:
            logger.debug(
                "  Update cell methods:"
                f"    Original cell methods = {original_cell_methods}"
                f"    method        = {method!r}"
                f"    within        = {within!r}"
                f"    over          = {over!r}"
            )  # pragma: no cover

        if input_axes and tuple(input_axes) == ("area",):
            axes = ("area",)
        else:
            axes = tuple(domain_axes)

        comment = None

        method = _collapse_cell_methods.get(method, method)

        cell_method = CellMethod(axes=axes, method=method)
        if within:
            cell_method.set_qualifier("within", within)
        elif over:
            cell_method.set_qualifier("over", over)

        if comment:
            cell_method.set_qualifier("comment", comment)

        if original_cell_methods:
            # There are already some cell methods
            if len(domain_axes) == 1:
                # Only one axis has been collapsed
                key, original_domain_axis = tuple(domain_axes.items())[0]

                lastcm = tuple(original_cell_methods.values())[-1]
                lastcm_method = _collapse_cell_methods.get(
                    lastcm.get_method(None), lastcm.get_method(None)
                )

                if (
                    original_domain_axis.get_size()
                    == self.domain_axes(todict=True)[key].get_size()
                ):
                    if (
                        lastcm.get_axes(None) == axes
                        and lastcm_method == method
                        and lastcm_method
                        in (
                            "mean",
                            "maximum",
                            "minimum",
                            "point",
                            "sum",
                            "median",
                            "mode",
                            "minimum_absolute_value",
                            "maximum_absolute_value",
                        )
                        and not lastcm.get_qualifier("within", None)
                        and not lastcm.get_qualifier("over", None)
                    ):
                        # It was a null collapse (i.e. the method is
                        # the same as the last one and the size of the
                        # collapsed axis hasn't changed).
                        if within:
                            lastcm.within = within
                        elif over:
                            lastcm.over = over

                        cell_method = None

        if cell_method is not None:
            self.set_construct(cell_method)

        if debug:
            logger.debug(
                f"    Modified cell methods = {self.cell_methods()}"
            )  # pragma: no cover

    @_inplace_enabled(default=False)
    def insert_dimension(
        self, axis, position=0, constructs=False, inplace=False
    ):
        """Insert a size 1 axis into the data array.

        .. versionadded:: 3.0.0

        .. seealso:: `domain_axis`, `flatten`, `flip`, `squeeze`,
                     `transpose`, `unsqueeze`

        :Parameters:

            axis:
                Select the domain axis to insert, generally defined by that
                which would be selected by passing the given axis description
                to a call of the field construct's `domain_axis` method. For
                example, for a value of ``'X'``, the domain axis construct
                returned by ``f.domain_axis('X')`` is selected.

                If *axis* is `None` then a new domain axis construct will
                created for the inserted dimension.

            position: `int`, optional
                Specify the position that the new axis will have in the
                data array. By default the new axis has position 0, the
                slowest varying position.

            constructs: `bool`, optional
                If True then also insert the new axis into all
                metadata constructs that don't already include it. By
                default, metadata constructs are not changed.

                .. versionadded:: 3.16.1

            {{inplace: `bool`, optional}}

        :Returns:

            `Field` or `None`
                The field construct with expanded data, or `None` if the
                operation was in-place.

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
        >>> g = f.insert_dimension('T', 0)
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(time(1), latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        A previously non-existent size 1 axis must be created prior to
        insertion:

        >>> f.insert_dimension(None, 1, inplace=True)
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(time(1), key%domainaxis3(1), latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        """
        return super().insert_dimension(
            axis=axis,
            position=position,
            constructs=constructs,
            inplace=inplace,
        )

    def indices(self, *config, **kwargs):
        """Create indices that define a subspace of the field construct.

        The subspace is defined by identifying indices based on the
        metadata constructs.

        Metadata constructs are selected by conditions specified on
        their data. Indices for subspacing are then automatically
        inferred from where the conditions are met.

        The returned tuple of indices may be used to created a
        subspace by indexing the original field construct with them.

        Metadata constructs and the conditions on their data are
        defined by keyword parameters.

        * Any domain axes that have not been identified remain
          unchanged.

        * Multiple domain axes may be subspaced simultaneously, and it
          doesn't matter which order they are specified in.

        * Subspace criteria may be provided for size 1 domain axes
          that are not spanned by the field construct's data.

        * Explicit indices may also be assigned to a domain axis
          identified by a metadata construct, with either a Python
          `slice` object, or a sequence of integers or booleans.

        * For a dimension that is cyclic, a subspace defined by a
          slice or by a `Query` instance is assumed to "wrap" around
          the edges of the data.

        * Conditions may also be applied to multi-dimensional metadata
          constructs. The "compress" mode is still the default mode
          (see the positional arguments), but because the indices may
          not be acting along orthogonal dimensions, some missing data
          may still need to be inserted into the field construct's
          data.

        **Ancillary masks**

        When creating an actual subspace with the indices, if the
        first element of the tuple of indices is ``'mask'`` then the
        second element is a tuple of auxiliary masks, and the
        remaining elements contain the usual indexing information that
        defines the extent of the subspace. Each auxiliary mask
        broadcasts to the subspaced data, and when the subspace is
        actually created, these masks are all automatically applied to
        the result.

        **Halos**

        {{subspace halos}}

        For instance, ``f.indices(X=slice(10, 20))`` will give
        identical results to each of ``f.indices(0, X=slice(10,
        20))``, ``f.indices(1, X=slice(11, 19))``, ``f.indices(2,
        X=slice(12, 18))``, etc.

        If a halo has been defined (of any size, including 0), then no
        ancillary masks will be created.

        .. versionadded:: 1.0

        .. seealso:: `subspace`, `where`, `__getitem__`,
                     `__setitem__`, `cf.Domain.indices`

        :Parameters:

            {{config: optional}}

                {{subspace valid modes Field}}

            kwargs: optional
                A keyword name is an identity of a metadata construct,
                and the keyword value provides a condition for
                inferring indices that apply to the dimension (or
                dimensions) spanned by the metadata construct's
                data. Indices are created that select every location
                for which the metadata construct's data satisfies the
                condition.

        :Returns:

            `tuple`
                The indices meeting the conditions.

        **Examples**

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
        >>> q[indices]
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

        >>> f = cf.example_field(2)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(120), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> f.indices(T=410.5)
        (dask.array<isclose, shape=(36,), dtype=bool, chunksize=(36,), chunktype=numpy.ndarray>,
         slice(None, None, None),
         slice(None, None, None))
        >>> f.indices(T=cf.dt('1961-11-16'))
        (dask.array<isclose, shape=(36,), dtype=bool, chunksize=(36,), chunktype=numpy.ndarray>,
         slice(0, 5, 1),
         slice(0, 8, 1))
        >>> indices = f.indices(T=cf.wi(cf.dt('1960-03-01'),
        ...                             cf.dt('1961-12-17 07:30')))
        >>> indices
        (dask.array<and_, shape=(36,), dtype=bool, chunksize=(36,), chunktype=numpy.ndarray>,
        slice(None, None, None),
        slice(None, None, None))
        >>> print(indices[0].compute())
        [False False False  True  True  True  True  True  True  True  True  True
          True  True  True  True  True  True  True  True  True  True  True  True
          True False False False False False False False False False False False]
        >>> print(f[indices])
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(22), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(22) = [1960-03-16 12:00:00, ..., 1961-12-16 12:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa

        >>> f = cf.example_field(1)
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
                        : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., kappa]
        Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m
        >>> indices = f.indices(latitude=cf.wi(51.5, 52.4))
        >>> indices
        ('mask',
         (<CF Data(1, 5, 9): [[[False, ..., False]]]>,),
         slice(None, None, None),
         [4, 5, 6],
         [0, 1, 2, 3, 4, 5, 6, 7, 8])
        >>> f[indices]
        <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(3), grid_longitude(9)) K>
        >>> print(f[indices].array)
        [[[264.2 275.9 262.5 264.9 264.7 270.2 270.4 -- --]
         [263.9 263.8 272.1 263.7 272.2 264.2 260.0 263.5 270.2]
         [-- -- -- -- -- -- 270.6 273.0 270.6]]]

        """
        if "exact" in config:
            _DEPRECATION_ERROR_ARG(
                self,
                "indices",
                "exact",
                "Keywords are now never interpreted as regular expressions.",
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        data_axes = self.get_data_axes()

        # Get the indices for every domain axis in the domain,
        # including any ancillary masks
        domain_indices = self._indices(config, data_axes, True, kwargs)

        # Initialise the output indices with any ancillary masks.
        # Ensure that each ancillary mask is broadcastable to the
        # data, by adding any missing size 1 dimensions and reordering
        # to the dimensions to the order of the field's data.
        ancillary_mask = domain_indices["mask"]
        if ancillary_mask:
            masks = []
            for axes, mask in ancillary_mask.items():
                axes = list(axes)
                for i, axis in enumerate(data_axes):
                    if axis not in axes:
                        axes.insert(0, axis)
                        mask.insert_dimension(0, inplace=True)

                new_order = [axes.index(axis) for axis in data_axes]
                mask.transpose(new_order, inplace=True)
                masks.append(mask)

            indices = ["mask", tuple(masks)]
        else:
            indices = []

        # Add the indices that apply to the field's data dimensions
        axis_indices = domain_indices["indices"]
        indices.extend([axis_indices[axis] for axis in data_axes])

        # Check that there are no invalid indices for size 1 axes not
        # spanned by the data
        if len(axis_indices) > len(data_axes):
            for axis, index in axis_indices.items():
                if axis in data_axes or (
                    isinstance(index, slice) and index == slice(None)
                ):
                    continue

                import dask.array as da

                shape = da.from_array([0])[index].compute_chunk_sizes().shape
                if 0 in shape:
                    raise IndexError(
                        "Can't create size 0 indices for the size 1 "
                        f"{self.constructs.domain_axis_identity(axis)!r} axis"
                    )

        return tuple(indices)

    @_inplace_enabled(default=True)
    def set_data(
        self, data, axes=None, set_axes=True, copy=True, inplace=True
    ):
        """Set the field construct data.

        .. versionadded:: 3.0.0

        .. seealso:: `data`, `del_data`, `get_data`, `has_data`,
                     `set_construct`

        :Parameters:

            data: `Data`
                The data to be inserted.

                {{data_like}}

            axes: (sequence of) `str` or `int`, optional
                Set the domain axes constructs that are spanned by the
                data. If unset, and the *set_axes* parameter is True, then
                an attempt will be made to assign existing domain axis
                constructs to the data.

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
                the *axes* parameter, or if any domain axis constructs
                exist then an attempt will be made to assign existing
                domain axis constructs to the data.

                If the *axes* parameter is `None` and no domain axis
                constructs exist then no attempt is made to assign domain
                axes constructs to the data, regardless of the value of
                *set_axes*.

            copy: `bool`, optional
                If True then set a copy of the data. By default the data
                are copied.

            {{inplace: `bool`, optional (default True)}}

                .. versionadded:: 3.7.0

        :Returns:

            `None` or `Field`
                If the operation was in-place then `None` is returned,
                otherwise return a new `Field` instance containing the new
                data.

        **Examples**

        >>> f = cf.Field()
        >>> f.set_data([1, 2, 3])
        >>> f.has_data()
        True
        >>> f.get_data()
        <CF Data(3): [1, 2, 3]>
        >>> f.data
        <CF Data(3): [1, 2, 3]>
        >>> f.del_data()
        <CF Data(3): [1, 2, 3]>
        >>> g = f.set_data([4, 5, 6], inplace=False)
        >>> g.data
        <CF Data(3): [4, 5, 6]>
        >>> f.has_data()
        False
        >>> print(f.get_data(None))
        None
        >>> print(f.del_data(None))
        None

        """
        data = self._Data(data, copy=False)

        # Construct new field
        f = _inplace_enabled_define_and_cleanup(self)

        domain_axes = f.domain_axes(todict=True)
        if axes is None and not domain_axes:
            set_axes = False

        if not set_axes:
            if not data.Units:
                units = getattr(f, "Units", None)
                if units is not None:
                    if copy:
                        copy = False
                        data = data.override_units(units, inplace=False)
                    else:
                        data.override_units(units, inplace=True)

            super(cfdm.Field, f).set_data(
                data, axes=None, copy=copy, inplace=True
            )

            return f

        if not data.ndim:
            # --------------------------------------------------------
            # The data array is scalar
            # --------------------------------------------------------
            if axes or axes == 0:
                raise ValueError(
                    "Can't set data: Wrong number of axes for scalar data "
                    f"array: axes={axes}"
                )

            axes = []

        elif axes is not None:
            # --------------------------------------------------------
            # Axes have been set
            # --------------------------------------------------------

            # Check that the input data spans all of the size > 1 axes
            sizes = set([axis.get_size() for axis in domain_axes.values()])
            data_shape = set(data.shape)
            sizes.discard(1)
            data_shape.discard(1)
            if data_shape != sizes:
                raise ValueError(
                    "Can't set data: Input data must span all axes that "
                    "have size greater than 1, as well as optionally "
                    "spanning any size 1 axes."
                )

            if isinstance(axes, (str, int, slice)):
                axes = (axes,)

            axes = [f.domain_axis(axis, key=True) for axis in axes]

            if len(axes) != data.ndim:
                raise ValueError(
                    f"Can't set data: Input data spans {data.ndim} "
                    f"dimensions, but only {len(axes)} were specified by "
                    "the 'axes' parameter."
                )

            for axis, size in zip(axes, data.shape):
                axis_size = domain_axes[axis].get_size(None)
                if size != axis_size:
                    axes_shape = tuple(
                        domain_axes[axis].get_size(None) for axis in axes
                    )
                    raise ValueError(
                        f"Can't set data: Input data shape {data.shape} "
                        f"differs from shape implied by the given axes "
                        f"{axes}: {axes_shape}"
                    )

        elif f.get_data_axes(default=None) is None:
            # --------------------------------------------------------
            # The data is not scalar and axes have not been set and
            # the domain does not have data axes defined
            #
            # => infer the axes
            # --------------------------------------------------------
            data_shape = data.shape
            if len(data_shape) != len(set(data_shape)):
                raise ValueError(
                    f"Can't insert data: Ambiguous data shape: {data_shape}. "
                    "Consider setting the axes parameter."
                )

            if not domain_axes:
                raise ValueError("Can't set data: No domain axes exist")

            axes = []
            for n in data_shape:
                da_key = f.domain_axis(
                    filter_by_size=(n,), key=True, default=None
                )
                if da_key is None:
                    raise ValueError(
                        "Can't insert data: Ambiguous data shape: "
                        f"{data_shape}. Consider setting the axes parameter."
                    )

                axes.append(da_key)

        else:
            # --------------------------------------------------------
            # The data is not scalar and axes have not been set, but
            # there are data axes defined on the field.
            # --------------------------------------------------------
            axes = f.get_data_axes()
            if len(axes) != data.ndim:
                raise ValueError(
                    f"Wrong number of axes for data array: {axes!r}"
                )

            for axis, size in zip(axes, data.shape):
                if domain_axes[axis].get_size(None) != size:
                    raise ValueError(
                        "Can't insert data: Incompatible size for axis "
                        f"{axis!r}: {size}"
                    )

        if not data.Units:
            units = getattr(f, "Units", None)
            if units is not None:
                if copy:
                    copy = False
                    data = data.override_units(units, inplace=False)
                else:
                    data.override_units(units, inplace=True)

        super(cfdm.Field, f).set_data(data, axes=axes, copy=copy, inplace=True)

        # Apply cyclic axes
        if axes:
            cyclic = self._cyclic
            if cyclic:
                cyclic_axes = [
                    axes.index(axis) for axis in cyclic if axis in axes
                ]
                if cyclic_axes:
                    data.cyclic(cyclic_axes, True)

        return f

    def domain_mask(self, **kwargs):
        """Return a boolean field that is True where criteria are met.

        .. versionadded:: 1.1

        .. seealso:: `indices`, `mask`, `subspace`

        :Parameters:

            kwargs: optional
                A dictionary of keyword arguments to pass to the `indices`
                method to define the criteria to meet for a element to be
                set as `True`.

        :Returns:

            `Field`
                The domain mask.

        **Examples**

        Create a domain mask which is masked at all between between -30
        and 30 degrees of latitude:

        >>> m = f.domain_mask(latitude=cf.wi(-30, 30))

        """
        mask = self.copy()

        mask.clear_properties()
        mask.nc_del_variable(None)

        for key in self.constructs.filter_by_type(
            "cell_method", "field_ancillary", todict=True
        ):
            mask.del_construct(key)

        false_everywhere = Data.zeros(self.shape, dtype=bool)

        mask.set_data(false_everywhere, axes=self.get_data_axes(), copy=False)

        mask.subspace[mask.indices(**kwargs)] = True

        mask.long_name = "domain mask"

        return mask

    @_inplace_enabled(default=False)
    @_manage_log_level_via_verbosity
    def compute_vertical_coordinates(
        self, default_to_zero=True, strict=True, inplace=False, verbose=None
    ):
        """Compute non-parametric vertical coordinates.

        When vertical coordinates are a function of horizontal location as
        well as parameters which depend on vertical location, they cannot
        be stored in a vertical dimension coordinate construct. In such
        cases a parametric vertical dimension coordinate construct is
        stored and a coordinate reference construct contains the formula
        for computing the required non-parametric vertical coordinates.

        {{formula terms links}}

        For example, multi-dimensional non-parametric parametric ocean
        altitude coordinates can be computed from one-dimensional
        parametric ocean sigma coordinates.

        Coordinate reference systems based on parametric vertical
        coordinates are identified from the coordinate reference
        constructs and, if possible, the corresponding non-parametric
        vertical coordinates are computed and stored in a new auxiliary
        coordinate construct.

        If there are no appropriate coordinate reference constructs then
        the field construct is unchanged.

        .. versionadded:: 3.8.0

        .. seealso:: `CoordinateReference`

        :Parameters:

            {{default_to_zero: `bool`, optional}}

            strict: `bool`
                If False then allow the computation to occur when

                * A domain ancillary construct has no standard name, but
                  the corresponding term has a standard name that is
                  prescribed

                * When the computed standard name can not be found by
                  inference from the standard names of the domain
                  ancillary constructs, nor from the
                  ``computed_standard_name`` parameter of the relevant
                  coordinate reference construct.

                By default an exception is raised in these cases.

                If a domain ancillary construct does have a standard name,
                but one that is inconsistent with any prescribed standard
                names, then an exception is raised regardless of the value
                of *strict*.

            {{inplace: `bool`, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

        :Returns:

            `Field` or `None`
                The field construct with the new non-parametric vertical
                coordinates, or `None` if the operation was in-place.

        **Examples**

        >>> f = cf.example_field(1)
        >>> print(f)
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
                        : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., kappa]
        Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m
        >>> print(f.auxiliary_coordinate('altitude', default=None))
        None
        >>> g = f.compute_vertical_coordinates()
        >>> print(g.auxiliary_coordinates)
        Constructs:
        {'auxiliarycoordinate0': <CF AuxiliaryCoordinate: latitude(10, 9) degrees_N>,
         'auxiliarycoordinate1': <CF AuxiliaryCoordinate: longitude(9, 10) degrees_E>,
         'auxiliarycoordinate2': <CF AuxiliaryCoordinate: long_name=Grid latitude name(10) >,
         'auxiliarycoordinate3': <CF AuxiliaryCoordinate: altitude(1, 10, 9) m>}
        >>> g.auxiliary_coordinate('altitude').dump()
        Auxiliary coordinate: altitude
            long_name = 'Computed from parametric atmosphere_hybrid_height_coordinate
                         vertical coordinates'
            standard_name = 'altitude'
            units = 'm'
            Data(1, 10, 9) = [[[10.0, ..., 5410.0]]] m
            Bounds:units = 'm'
            Bounds:Data(1, 10, 9, 2) = [[[[5.0, ..., 5415.0]]]] m

        """
        f = _inplace_enabled_define_and_cleanup(self)

        detail = is_log_level_detail(logger)
        debug = is_log_level_debug(logger)

        for cr in f.coordinate_references(todict=True).values():
            # --------------------------------------------------------
            # Compute the non-parametric vertical coordinates, if
            # possible.
            # --------------------------------------------------------
            (
                standard_name,
                computed_standard_name,
                computed,
                computed_axes,
                k_axis,
            ) = FormulaTerms.formula(f, cr, default_to_zero, strict)

            if computed is None:
                # No non-parametric vertical coordinates were
                # computed
                continue

            # --------------------------------------------------------
            # Convert the computed domain ancillary construct to an
            # auxiliary coordinate construct, and insert it into the
            # field construct.
            # --------------------------------------------------------
            c = f._AuxiliaryCoordinate(source=computed, copy=False)
            c.clear_properties()
            c.long_name = (
                f"Computed from parametric {standard_name} "
                "vertical coordinates"
            )
            if computed_standard_name:
                c.standard_name = computed_standard_name

            if c.has_bounds():
                c.bounds.clear_properties()

            if detail:
                logger.detail(
                    "Non-parametric coordinates:\n"
                    f"{c.dump(display=False, _level=1)}"
                )  # pragma: no cover

            key = f.set_construct(c, axes=computed_axes, copy=False)

            # Reference the new coordinates from the coordinate
            # reference construct
            cr.set_coordinate(key)

            if debug:
                logger.debug(
                    f"Non-parametric coordinates construct key: {key!r}\n"
                    "Updated coordinate reference construct:\n"
                    f"{cr.dump(display=False, _level=1)}"
                )  # pragma: no cover

        return f

    def match_by_construct(self, *identities, OR=False, **conditions):
        """Whether or not there are particular metadata constructs.

        .. note:: The API changed at version 3.1.0

        .. versionadded:: 3.0.0

        .. seealso:: `match`, `match_by_property`, `match_by_rank`,
                     `match_by_identity`, `match_by_ncvar`,
                     `match_by_units`, `construct`

        :Parameters:

            identities: optional
                Select the unique construct returned by
                ``f.construct(*identities)``. See `construct` for
                details.

            conditions: optional
                Identify the metadata constructs that have any of the
                given identities or construct keys, and whose data satisfy
                conditions.

                A construct identity or construct key (as defined by the
                *identities* parameter) is given as a keyword name and a
                condition on its data is given as the keyword value.

                The condition is satisfied if any of its data values
                equals the value provided.

                *Parameter example:*
                  ``longitude=180.0``

                *Parameter example:*
                  ``time=cf.dt('1959-12-16')``

                *Parameter example:*
                  ``latitude=cf.ge(0)``

                *Parameter example:*
                  ``latitude=cf.ge(0), air_pressure=500``

                *Parameter example:*
                  ``**{'latitude': cf.ge(0), 'long_name=soil_level': 4}``

            OR: `bool`, optional
                If True then return `True` if at least one metadata
                construct matches at least one of the criteria given by
                the *identities* or *conditions* arguments. By default
                `True` is only returned if the field constructs matches
                each of the given criteria.

            mode: deprecated at version 3.1.0
                Use the *OR* parameter instead.

            constructs: deprecated at version 3.1.0

        :Returns:

            `bool`
                Whether or not the field construct contains the specified
                metadata constructs.

        **Examples**

            TODO

        """
        if identities:
            if identities[0] == "or":
                _DEPRECATION_ERROR_ARG(
                    self,
                    "match_by_construct",
                    "or",
                    message="Use 'OR=True' instead.",
                    version="3.1.0",
                    removed_at="4.0.0",
                )  # pragma: no cover

            if identities[0] == "and":
                _DEPRECATION_ERROR_ARG(
                    self,
                    "match_by_construct",
                    "and",
                    message="Use 'OR=False' instead.",
                    version="3.1.0",
                    removed_at="4.0.0",
                )  # pragma: no cover

        if not identities and not conditions:
            return True

        constructs = self.constructs

        if not constructs:
            return False

        n = 0

        self_cell_methods = self.cell_methods(todict=True)

        for identity in identities:
            cms = False
            try:
                cms = ": " in identity
            except TypeError:
                cms = False

            if cms:
                cms = CellMethod.create(identity)
                for cm in cms:
                    axes = [
                        self.domain_axis(axis, key=True, default=axis)
                        for axis in cm.get_axes(())
                    ]
                    if axes:
                        cm.set_axes(axes)

            if not cms:
                filtered = constructs(identity)
                if filtered:
                    # Check for cell methods
                    if set(filtered.construct_types().values()) == {
                        "cell_method"
                    }:
                        key = tuple(self_cell_methods)[-1]
                        filtered = self.cell_method(
                            identity, filter_by_key=(key,), default=None
                        )
                        if filtered is None:
                            if not OR:
                                return False

                            n -= 1

                    n += 1
                elif not OR:
                    return False
            else:
                cell_methods = tuple(self_cell_methods.values())[-len(cms) :]
                for cm0, cm1 in zip(cms, cell_methods):
                    if cm0.has_axes() and set(cm0.get_axes()) != set(
                        cm1.get_axes(())
                    ):
                        if not OR:
                            return False

                        n -= 1
                        break

                    if cm0.has_method() and (
                        cm0.get_method() != cm1.get_method(None)
                    ):
                        if not OR:
                            return False

                        n -= 1
                        break

                    ok = True
                    for key, value in cm0.qualifiers():
                        if value != cm1.get_qualifier(key, None):
                            if not OR:
                                return False

                            ok = False
                            break

                    if not ok:
                        n -= 1
                        break

                n += 1

        if conditions:
            for identity, value in conditions.items():
                if self.subspace("test", **{identity: value}):
                    n += 1
                elif not OR:
                    return False

        if OR:
            return bool(n)

        return True

    @_inplace_enabled(default=False)
    def moving_window(
        self,
        method,
        window_size=None,
        axis=None,
        weights=None,
        mode=None,
        cval=None,
        origin=0,
        scale=None,
        radius="earth",
        great_circle=False,
        inplace=False,
    ):
        """Perform moving window calculations along an axis.

        Moving mean, sum, and integral calculations are possible.

        By default moving means are unweighted, but weights based on
        the axis cell sizes, or custom weights, may applied to the
        calculation via the *weights* parameter.

        By default moving integrals must be weighted.

        When appropriate, a new cell method construct is created to
        describe the calculation.

        .. note:: The `moving_window` method can not, in general, be
                  emulated by the `convolution_filter` method, as the
                  latter i) can not change the window weights as the
                  filter passes through the axis; and ii) does not
                  update the cell method constructs.

        .. versionadded:: 3.3.0

        .. seealso:: `bin`, `collapse`, `convolution_filter`, `radius`,
                     `weights`

        :Parameters:

            method: `str`
                Define the moving window method. The method is given
                by one of the following strings (see
                https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
                for precise definitions):

                ==================  ============================  ========
                *method*            Description                   Weighted
                ==================  ============================  ========
                ``'sum'``           The sum of the values.        Never

                ``'mean'``          The weighted or unweighted    May be
                                    mean of the values.

                ``'integral'``      The integral of values.       Always
                ==================  ============================  ========

                * Methods that are "Never" weighted ignore the
                  *weights* parameter, even if it is set.

                * Methods that "May be" weighted will only be weighted
                  if the *weights* parameter is set.

                * Methods that are "Always" weighted require the
                  *weights* parameter to be set.

            window_size: `int`
                Specify the size of the window used to calculate the
                moving window.

                *Parameter example:*
                  A 5-point moving window is set with
                  ``window_size=5``.

            axis: `str` or `int`
                Select the domain axis over which the filter is to be
                applied, defined by that which would be selected by
                passing the given axis description to a call of the
                field construct's `domain_axis` method. For example,
                for a value of ``'X'``, the domain axis construct
                returned by ``f.domain_axis('X')`` is selected.

            weights: optional
                Specify the weights for the moving window. The weights
                are, those that would be returned by this call of the
                field construct's `weights` method:
                ``f.weights(weights, axes=axis, radius=radius,
                great_circle=great_circle, data=True)``. See the
                *axis*, *radius* and *great_circle* parameters and
                `cf.Field.weights` for details.

                .. note:: By default *weights* is `None`, resulting in
                          **unweighted calculations**.

                .. note:: Setting *weights* to `True` is generally a
                          good way to ensure that the moving window
                          calculations are appropriately weighted
                          according to the field construct's
                          metadata. In this case, if it is not
                          possible to create weights for the selected
                          *axis* then an exception will be raised.

                *Parameter example:*
                  To specify weights on the cell sizes of the selected
                  axis: ``weights=True``.

            mode: `str`, optional
                The *mode* parameter determines how the input array is
                extended when the filter overlaps an array border. The
                default value is ``'constant'`` or, if the dimension
                being convolved is cyclic (as ascertained by the
                `iscyclic` method), ``'wrap'``. The valid values and
                their behaviours are as follows:

                ==============  ==========================  ===========================
                *mode*          Description                 Behaviour
                ==============  ==========================  ===========================
                ``'reflect'``   The input is extended by    ``(c b a | a b c | c b a)``
                                reflecting about the edge

                ``'constant'``  The input is extended by    ``(k k k | a b c | k k k)``
                                filling all values beyond
                                the edge with the same
                                constant value (``k``),
                                defined by the *cval*
                                parameter.

                ``'nearest'``   The input is extended by    ``(a a a | a b c | c c c)``
                                replicating the last point

                ``'mirror'``    The input is extended by    ``(c b | a b c | b a)``
                                reflecting about the
                                centre of the last point.

                ``'wrap'``      The input is extended by    ``(a b c | a b c | a b c)``
                                wrapping around to the
                                opposite edge.
                ==============  ==========================  ===========================

                The position of the window relative to each value can
                be changed by using the *origin* parameter.

            cval: scalar, optional
                Value to fill past the edges of the array if *mode* is
                ``'constant'``. Ignored for other modes. Defaults to
                `None`, in which case the edges of the array will be
                filled with missing data. The only other valid value
                is ``0``.

                *Parameter example:*
                   To extend the input by filling all values beyond
                   the edge with zero: ``cval=0``

            origin: `int`, optional
                Controls the placement of the filter. Defaults to 0,
                which is the centre of the window. If the window size,
                defined by the *window_size* parameter, is even then
                then a value of 0 defines the index defined by
                ``window_size/2 -1``.

                *Parameter example:*
                  For a window size of 5, if ``origin=0`` then the
                  window is centred on each point. If ``origin=-2``
                  then the window is shifted to include the previous
                  four points. If ``origin=1`` then the window is
                  shifted to include the previous point and the and
                  the next three points.

            radius: optional
                Specify the radius used for calculating the areas of
                cells defined in spherical polar coordinates. The
                radius is that which would be returned by this call of
                the field construct's `~cf.Field.radius` method:
                ``f.radius(radius)``. See the `cf.Field.radius` for
                details.

                By default *radius* is ``'earth'`` which means that if
                and only if the radius can not found from the datums
                of any coordinate reference constructs, then the
                default radius taken as 6371229 metres.

            great_circle: `bool`, optional
                If True then allow, if required, the derivation of i)
                area weights from polygon geometry cells by assuming
                that each cell part is a spherical polygon composed of
                great circle segments; and ii) and the derivation of
                line-length weights from line geometry cells by
                assuming that each line part is composed of great
                circle segments.

            scale: number, optional
                If set to a positive number then scale the weights so
                that they are less than or equal to that number. By
                default the weights are scaled to lie between 0 and 1
                (i.e.  *scale* is 1).

                Ignored if the moving window method is not
                weighted. The *scale* parameter can not be set for
                moving integrals.

                *Parameter example:*
                  To scale all weights so that they lie between 0 and
                  0.5: ``scale=0.5``.

            {{inplace: `bool`, optional}}

        :Returns:

            `Field` or `None`
                The field construct of moving window values, or `None`
                if the operation was in-place.

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
        >>> print(f.array)
        [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
         [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
         [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
         [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
         [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
        >>> print(f.coordinate('X').bounds.array)
        [[  0.  45.]
         [ 45.  90.]
         [ 90. 135.]
         [135. 180.]
         [180. 225.]
         [225. 270.]
         [270. 315.]
         [315. 360.]]
        >>> f.iscyclic('X')
        True
        >>> f.iscyclic('Y')
        False

        Create a weighted 3-point running mean for the cyclic 'X'
        axis:

        >>> g = f.moving_window('mean', 3, axis='X', weights=True)
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean longitude(8): mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(g.array)
        [[0.02333 0.01467 0.017   0.01167 0.023   0.02633 0.03    0.02   ]
         [0.04167 0.03467 0.04767 0.051   0.06033 0.04167 0.04833 0.03167]
         [0.084   0.12167 0.13367 0.119   0.112   0.08233 0.057   0.05933]
         [0.035   0.04233 0.056   0.05567 0.06667 0.04633 0.03267 0.01833]
         [0.01833 0.02033 0.03    0.024   0.03    0.02967 0.028   0.01767]]
        >>> print(g.coordinate('X').bounds.array)
        [[-45.  90.]
         [  0. 135.]
         [ 45. 180.]
         [ 90. 225.]
         [135. 270.]
         [180. 315.]
         [225. 360.]
         [270. 405.]]

        Create an unweighted 3-point running mean for the cyclic 'X'
        axis:

        >>> g = f.moving_window('mean', 3, axis='X')

        Create an weighted 4-point running integral for the non-cyclic
        'Y' axis:

        >>> g = f.moving_window('integral', 4, axis='Y', weights=True)
        >>> g.Units
        <Units: 0.0174532925199433 rad>
        >>> print(g.array)
        [[   --    --    --    --   --    --   --   --]
         [ 8.37 11.73 10.05 13.14 8.88 11.64 4.59 4.02]
         [ 8.34 11.79 10.53 13.77 8.88 11.64 4.89 3.54]
         [   --    --    --    --   --    --   --   --]
         [   --    --    --    --   --    --   --   --]]
        >>> print(g.coordinate('Y').bounds.array)
        [[-90.  30.]
         [-90.  60.]
         [-60.  90.]
         [-30.  90.]
         [ 30.  90.]]
        >>> g = f.moving_window('integral', 4, axis='Y', weights=True, cval=0)
        >>> print(g.array)
        [[ 7.5   9.96  8.88 11.04  7.14  9.48  4.32  3.51]
         [ 8.37 11.73 10.05 13.14  8.88 11.64  4.59  4.02]
         [ 8.34 11.79 10.53 13.77  8.88 11.64  4.89  3.54]
         [ 7.65 10.71  9.18 11.91  7.5   9.45  4.71  1.56]
         [ 1.05  2.85  1.74  3.15  2.28  3.27  1.29  0.9 ]]

        """
        method_values = ("mean", "sum", "integral")
        if method not in method_values:
            raise ValueError(
                f"Non-valid 'method' parameter value: {method!r}. "
                f"Expected one of {method_values!r}"
            )

        if cval is not None and cval != 0:
            raise ValueError("The cval parameter must be None or 0")

        window_size = int(window_size)

        # Construct new field
        f = _inplace_enabled_define_and_cleanup(self)

        # Find the axis for the moving window
        axis = f.domain_axis(axis, key=True)
        iaxis = self.get_data_axes().index(axis)

        if method == "sum" or weights is False:
            weights = None

        if method == "integral":
            measure = True
            if weights is None:
                raise ValueError(
                    "Must set weights parameter for 'integral' method"
                )

            if scale is not None:
                raise ValueError(
                    "Can't set the 'scale' parameter for moving integrals"
                )
        else:
            if scale is None:
                scale = 1.0

            measure = False

        if weights is not None:
            if isinstance(weights, Data):
                if weights.ndim > 1:
                    raise ValueError(
                        f"The input weights (shape {weights.shape}) do not "
                        f"match the selected axis (size {f.shape[iaxis]})"
                    )

                if weights.ndim == 1:
                    if weights.shape[0] != f.shape[iaxis]:
                        raise ValueError(
                            f"The input weights (size {weights.size}) do not "
                            f"match the selected axis (size {f.shape[iaxis]})"
                        )

            # Get the data weights
            w = f.weights(
                weights,
                axes=axis,
                measure=measure,
                scale=scale,
                radius=radius,
                great_circle=great_circle,
                data=True,
            )

            # Multiply the field by the (possibly adjusted) weights
            f = f * w

        # Create the window weights
        window = np.full((window_size,), 1.0)
        if weights is None and method == "mean":
            # If there is no data weighting, make sure that the sum of
            # the window weights is 1.
            window /= window.size

        f.convolution_filter(
            window,
            axis=axis,
            mode=mode,
            cval=cval,
            origin=origin,
            update_bounds=True,
            inplace=True,
        )

        if weights is not None and method == "mean":
            # Divide the field by the running sum of the adjusted data
            # weights
            w.convolution_filter(
                window=window,
                axis=iaxis,
                mode=mode,
                cval=0,
                origin=origin,
                inplace=True,
            )
            f = f / w

        # Add a cell method
        if f.domain_axis(axis).get_size() > 1 or method == "integral":
            f._update_cell_methods(
                method=method, domain_axes=f.domain_axes(axis, todict=True)
            )

        return f

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def convolution_filter(
        self,
        window=None,
        axis=None,
        mode=None,
        cval=None,
        origin=0,
        update_bounds=True,
        inplace=False,
        weights=None,
        i=False,
    ):
        """Convolve the field construct along the given axis with the
        specified filter.

        The magnitude of the integral of the filter (i.e. the sum of
        the window weights defined by the *window* parameter) affects
        the convolved values. For example, window weights of ``[0.2,
        0.2 0.2, 0.2, 0.2]`` will produce a non-weighted 5-point
        running mean; and window weights of ``[1, 1, 1, 1, 1]`` will
        produce a 5-point running sum. Note that the window weights
        returned by functions of the `scipy.signal.windows` package do
        not necessarily sum to 1 (see the examples for details).

        .. note:: The `moving_window` method can not, in general, be
                  emulated by the `convolution_filter` method, as the
                  latter i) can not change the window weights as the
                  filter passes through the axis; and ii) does not
                  update the cell method constructs.

        .. seealso:: `collapse`, `derivative`, `moving_window`,
                     `cf.relative_vorticity`

        :Parameters:

            window: sequence of numbers
                Specify the window weights to use for the filter.

                *Parameter example:*
                  An unweighted 5-point moving average can be computed
                  with ``window=[0.2, 0.2, 0.2, 0.2, 0.2]``

                Note that the `scipy.signal.windows` package has suite
                of window functions for creating window weights for
                filtering (see the examples for details).

                .. versionadded:: 3.3.0 (replaces the old weights
                                  parameter)

            axis:
                Select the domain axis over which the filter is to be
                applied, defined by that which would be selected by
                passing the given axis description to a call of the field
                construct's `domain_axis` method. For example, for a value
                of ``'X'``, the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

            mode: `str`, optional
                The *mode* parameter determines how the input array is
                extended when the filter overlaps an array border. The
                default value is ``'constant'`` or, if the dimension being
                convolved is cyclic (as ascertained by the `iscyclic`
                method), ``'wrap'``. The valid values and their behaviours
                are as follows:

                ==============  ==========================  ===========================
                *mode*          Description                 Behaviour
                ==============  ==========================  ===========================
                ``'reflect'``   The input is extended by    ``(c b a | a b c | c b a)``
                                reflecting about the edge

                ``'constant'``  The input is extended by    ``(k k k | a b c | k k k)``
                                filling all values beyond
                                the edge with the same
                                constant value (``k``),
                                defined by the *cval*
                                parameter.

                ``'nearest'``   The input is extended by    ``(a a a | a b c | d d d)``
                                replicating the last point

                ``'mirror'``    The input is extended by    ``(c b | a b c | b a)``
                                reflecting about the
                                centre of the last point.

                ``'wrap'``      The input is extended by    ``(a b c | a b c | a b c)``
                                wrapping around to the
                                opposite edge.
                ==============  ==========================  ===========================

                The position of the window relative to each value can be
                changed by using the *origin* parameter.

            cval: scalar, optional
                Value to fill past the edges of the array if *mode* is
                ``'constant'``. Ignored for other modes. Defaults to
                `None`, in which case the edges of the array will be
                filled with missing data.

                *Parameter example:*
                   To extend the input by filling all values beyond the
                   edge with zero: ``cval=0``

            origin: `int`, optional
                Controls the placement of the filter. Defaults to 0, which
                is the centre of the window. If the window has an even
                number of weights then then a value of 0 defines the index
                defined by ``width/2 -1``.

                *Parameter example:*
                  For a weighted moving average computed with a weights
                  window of ``[0.1, 0.15, 0.5, 0.15, 0.1]``, if
                  ``origin=0`` then the average is centred on each
                  point. If ``origin=-2`` then the average is shifted to
                  include the previous four points. If ``origin=1`` then
                  the average is shifted to include the previous point and
                  the and the next three points.

            update_bounds: `bool`, optional
                If False then the bounds of a dimension coordinate
                construct that spans the convolved axis are not
                altered. By default, the bounds of a dimension coordinate
                construct that spans the convolved axis are updated to
                reflect the width and origin of the window.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            weights: deprecated at version 3.3.0
                Use the *window* parameter instead.

        :Returns:

            `Field` or `None`
                The convolved field construct, or `None` if the operation
                was in-place.

        **Examples**

        >>> f = cf.example_field(2)
        >>> print(f)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> print(f.array[:, 0, 0])
        [210.7 305.3 249.4 288.9 231.1 200.  234.4 289.2 204.3 203.6 261.8 256.2
         212.3 231.7 255.1 213.9 255.8 301.2 213.3 200.1 204.6 203.2 244.6 238.4
         304.5 269.8 267.9 282.4 215.  288.7 217.3 307.1 299.3 215.9 290.2 239.9]
        >>> print(f.coordinate('T').bounds.dtarray[0])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-01-01 00:00:00)]
        >>> print(f.coordinate('T').bounds.dtarray[2])
        [cftime.DatetimeGregorian(1960-02-01 00:00:00)
         cftime.DatetimeGregorian(1960-03-01 00:00:00)]

        Create a 5-point (non-weighted) running mean:

        >>> g = f.convolution_filter([0.2, 0.2, 0.2, 0.2, 0.2], 'T')
        >>> print(g)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> print(g.array[:, 0, 0])
        [ -- -- 257.08 254.94 240.76 248.72 231.8 226.3 238.66 243.02 227.64
         233.12 243.42 233.84 233.76 251.54 247.86 236.86 235.0 224.48 213.16
         218.18 239.06 252.1 265.04 272.6 267.92 264.76 254.26 262.1 265.48
         265.66 265.96 270.48 -- --]
        >>> print(g.coordinate('T').bounds.dtarray[0])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-03-01 00:00:00)]
        >>> print(g.coordinate('T').bounds.dtarray[2])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-05-01 00:00:00)]

        Create a 5-point running sum:

        >>> g = f.convolution_filter([1, 1, 1, 1, 1], 'T')
        >>> print(g)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> print(g.array[:, 0, 0])
        [ -- -- 1285.4 1274.7 1203.8 1243.6 1159.0 1131.5 1193.3 1215.1
         1138.2 1165.6 1217.1 1169.2 1168.8 1257.7 1239.3 1184.3 1175.0
         1122.4 1065.8 1090.9 1195.3 1260.5 1325.2 1363.0 1339.6 1323.8
         1271.3 1310.5 1327.4 1328.3 1329.8 1352.4 -- --]
        >>> print(g.coordinate('T').bounds.dtarray[0])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-03-01 00:00:00)]
        >>> print(g.coordinate('T').bounds.dtarray[2])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-05-01 00:00:00)]

        Calculate a convolution along the time axis with Gaussian window
        weights, using the "nearest" mode at the border of the edges of
        the time dimension (note that the window weights returned by
        `scipy.signal.windows` functions do not necessarily sum to 1):

        >>> import scipy.signal.windows
        >>> gaussian_window = scipy.signal.windows.gaussian(3, std=0.4)
        >>> print(gaussian_window)
        [0.04393693 1.         0.04393693]
        >>> g = f.convolution_filter(gaussian_window, 'T', mode='nearest')
        >>> print(g.array[:, 0, 0])
        [233.37145775 325.51538316 275.50732596 310.01169661 252.58076685
         220.4526426  255.89394793 308.47513278 225.95212089 224.07900476
         282.00220208 277.03050023 233.73682991 252.23612278 274.67829762
         236.34737939 278.43191451 321.81081556 235.32558483 218.46124456
         222.31976533 222.93647058 264.00254989 262.52577025 326.82874967
         294.94950081 292.16197475 303.61714525 240.09238279 307.69393641
         243.47762505 329.79781991 322.27901629 241.80082237 310.22645435
         263.19096851]
        >>> print(g.coordinate('T').bounds.dtarray[0])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-02-01 00:00:00)]
        >>> print(g.coordinate('T').bounds.dtarray[1])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-03-01 00:00:00)]

        """
        if weights is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "convolution_filter",
                {"weights": weights},
                message="Use keyword 'window' instead.",
                version="3.3.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        if isinstance(window, str):
            _DEPRECATION_ERROR(
                "A string-valued 'window' parameter has been deprecated "
                "at version 3.0.0 and is no longer available. Provide a "
                "sequence of numerical window weights instead. "
                "scipy.signal.windows may be used to generate particular "
                "window functions.",
                version="3.3.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        if isinstance(window[0], str):
            _DEPRECATION_ERROR(
                "A string-valued 'window' parameter element has been "
                "deprecated at version 3.0.0 and is no longer available. "
                "Provide a sequence of numerical window weights instead. "
                "scipy.signal.windows may be used to generate particular "
                "window functions.",
                version="3.3.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        # Retrieve the axis
        axis_key = self.domain_axis(axis, key=True)
        iaxis = self.get_data_axes().index(axis_key)

        # Default mode to 'wrap' if the axis is cyclic
        if mode is None:
            if self.iscyclic(axis_key):
                mode = "wrap"
            else:
                mode = "constant"

        # Construct new field
        f = _inplace_enabled_define_and_cleanup(self)

        f.data.convolution_filter(
            window=window,
            axis=iaxis,
            mode=mode,
            cval=cval,
            origin=origin,
            inplace=True,
        )

        # Update the bounds of the convolution axis if necessary
        if update_bounds:
            coord = f.dimension_coordinate(
                filter_by_axis=(axis_key,), default=None
            )
            if coord is not None and coord.has_bounds():
                old_bounds = coord.bounds.data
                new_bounds = self._Data.empty(
                    old_bounds.shape, units=coord.Units
                )
                len_weights = len(window)
                lower_offset = len_weights // 2 + origin
                upper_offset = len_weights - 1 - lower_offset
                if mode == "wrap":
                    if coord.direction():
                        new_bounds[:, 0:1] = coord.roll(
                            0, upper_offset
                        ).bounds[:, 0:1]
                        new_bounds[:, 1:] = (
                            coord.roll(0, -lower_offset).bounds[:, 1:]
                        ) + coord.period()
                    else:
                        new_bounds[:, 0:1] = (
                            coord.roll(0, upper_offset).bounds[:, 0:1]
                            + coord.period()
                        )
                        new_bounds[:, 1:] = coord.roll(
                            0, -lower_offset
                        ).bounds[:, 1:]
                else:
                    length = old_bounds.shape[0]
                    new_bounds[upper_offset:length, 0:1] = old_bounds[
                        0 : length - upper_offset, 0:1
                    ]
                    new_bounds[0:upper_offset, 0:1] = old_bounds[0, 0:1]
                    new_bounds[0 : length - lower_offset, 1:] = old_bounds[
                        lower_offset:length, 1:
                    ]
                    new_bounds[length - lower_offset : length, 1:] = (
                        old_bounds[length - 1, 1:]
                    )

                coord.set_bounds(self._Bounds(data=new_bounds))

        return f

    def convert(
        self, *identity, full_domain=True, cellsize=False, **filter_kwargs
    ):
        """Convert a metadata construct into a new field construct.

        The new field construct has the properties and data of the
        metadata construct, and domain axis constructs corresponding to
        the data. By default it also contains other metadata constructs
        (such as dimension coordinate and coordinate reference constructs)
        that define its domain.

        Only metadata constructs that can have data may be converted
        and they can be converted even if they do not actually have
        any data. Constructs such as cell methods which cannot have
        data cannot be converted.

        The `cf.read` function allows a field construct to be derived
        directly from a netCDF variable that corresponds to a metadata
        construct. In this case, the new field construct will have a
        domain limited to that which can be inferred from the
        corresponding netCDF variable - typically only domain axis and
        dimension coordinate constructs. This will usually result in a
        different field construct to that created with the convert method.

        .. versionadded:: 3.0.0

        .. seealso:: `cf.read`, `construct`

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique construct returned by
                ``f.construct(*identity, **filter_kwargs)``. See
                `construct` for details.

            full_domain: `bool`, optional
                If False then do not create a domain, other than domain
                axis constructs, for the new field construct. By default
                as much of the domain as possible is copied to the new
                field construct.

            cellsize: `bool`, optional
                If True then create a field construct from the selected
                metadata construct's cell sizes.

        :Returns:

            `Field`
                The new field construct.

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
        >>> x = f.convert('X')
        >>> print(x)
        Field: longitude (ncvar%lon)
        ----------------------------
        Data            : longitude(longitude(8)) degrees_east
        Dimension coords: longitude(8) = [22.5, ..., 337.5] degrees_east
        >>> print(x.array)
        [ 22.5  67.5 112.5 157.5 202.5 247.5 292.5 337.5]
        >>> cs = f.convert('X', cellsize=True)
        >>> print(cs)
        Field: longitude (ncvar%lon)
        ----------------------------
        Data            : longitude(longitude(8)) degrees_east
        Dimension coords: longitude(8) = [22.5, ..., 337.5] degrees_east
        >>> print(cs.array)
        [45. 45. 45. 45. 45. 45. 45. 45.]
        >>> print(f.convert('X', full_domain=False))
        Field: longitude (ncvar%lon)
        ----------------------------
        Data            : longitude(ncdim%lon(8)) degrees_east

        """
        key, c = self.construct(
            *identity, item=True, default=(None, None), **filter_kwargs
        )

        f = super().convert(full_domain=full_domain, filter_by_key=(key,))

        if cellsize:
            # Change the new field's data to cell sizes
            try:
                cs = c.cellsize
            except AttributeError as error:
                raise ValueError(error)

            f.set_data(cs.data, set_axes=False, copy=False)

        return f

    @_inplace_enabled(default=False)
    def cumsum(
        self, axis, masked_as_zero=False, coordinate=None, inplace=False
    ):
        """Return the field cumulatively summed along the given axis.

        The cell bounds of the axis are updated to describe the range
        over which the sums apply, and a new "sum" cell method
        construct is added to the resulting field construct.

        .. versionadded:: 3.0.0

        .. seealso:: `collapse`, `convolution_filter`,
                     `moving_window`, `sum`

        :Parameters:

            axis:
                Select the domain axis over which the cumulative sums
                are to be calculated, defined by that which would be
                selected by passing the given axis description to a
                call of the field construct's `domain_axis`
                method. For example, for a value of ``'X'``, the
                domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

            {{inplace: `bool`, optional}}

            coordinate: deprecated at version 3.14.0
                Set how the cell coordinate values for the summed axis
                are defined, relative to the new cell bounds.

            masked_as_zero: deprecated at version 3.14.0
                See `Data.cumsum` for examples of the new behaviour
                when there are masked values.

        :Returns:

            `Field` or `None`
                The field construct with the cumulatively summed axis,
                or `None` if the operation was in-place.

        **Examples**

        >>> f = cf.example_field(2)
        >>> print(f)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> print(f.dimension_coordinate('T').bounds[[0, -1]].datetime_array)
        [[cftime.DatetimeGregorian(1959-12-01 00:00:00)
          cftime.DatetimeGregorian(1960-01-01 00:00:00)]
         [cftime.DatetimeGregorian(1962-11-01 00:00:00)
          cftime.DatetimeGregorian(1962-12-01 00:00:00)]]
        >>> print(f.array[:, 0, 0])
        [210.7 305.3 249.4 288.9 231.1 200.  234.4 289.2 204.3 203.6 261.8 256.2
         212.3 231.7 255.1 213.9 255.8 301.2 213.3 200.1 204.6 203.2 244.6 238.4
         304.5 269.8 267.9 282.4 215.  288.7 217.3 307.1 299.3 215.9 290.2 239.9]
        >>> g = f.cumsum('T')
        >>> print(g)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean time(36): sum
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> print(g.dimension_coordinate('T').bounds[[0, -1]].datetime_array)
        [[cftime.DatetimeGregorian(1959-12-01 00:00:00)
          cftime.DatetimeGregorian(1960-01-01 00:00:00)]
         [cftime.DatetimeGregorian(1959-12-01 00:00:00)
          cftime.DatetimeGregorian(1962-12-01 00:00:00)]]
        >>> print(g.array[:, 0, 0])
        [ 210.7  516.   765.4 1054.3 1285.4 1485.4 1719.8 2009.  2213.3 2416.9
         2678.7 2934.9 3147.2 3378.9 3634.  3847.9 4103.7 4404.9 4618.2 4818.3
         5022.9 5226.1 5470.7 5709.1 6013.6 6283.4 6551.3 6833.7 7048.7 7337.4
         7554.7 7861.8 8161.1 8377.  8667.2 8907.1]

        """
        # TODODASKAPI
        if masked_as_zero:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "cumsum",
                {"masked_as_zero": None},
                message="",
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        # TODODASKAPI
        if coordinate is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "cumsum",
                {"coordinate": None},
                message="",
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        # Retrieve the axis
        axis_key, domain_axis = self.domain_axis(
            axis, item=True, default=(None, None)
        )
        if axis_key is None:
            raise ValueError(f"Invalid axis specifier: {axis!r}")

        # Construct new field
        f = _inplace_enabled_define_and_cleanup(self)

        # Get the axis index
        axis_index = f.get_data_axes().index(axis_key)

        f.data.cumsum(axis_index, inplace=True)

        if domain_axis.get_size() > 1:
            # Update the bounds of the summed axis
            coord = f.dimension_coordinate(
                filter_by_axis=(axis_key,), default=None
            )
            if coord is not None and coord.has_bounds():
                bounds = coord.get_bounds_data(None)
                if bounds is not None:
                    bounds[:, 0] = bounds[0, 0]

            # Add a cell method
            f._update_cell_methods(
                method="sum", domain_axes={axis_key: domain_axis}
            )

        return f

    def file_locations(self, constructs=True):
        """The locations of files containing parts of the data.

        Returns the locations of any files that may be required to
        deliver the computed data array.

        .. versionadded:: 3.15.0

        .. seealso:: `add_file_location`, `del_file_location`

        :Parameters:

            constructs: `bool`, optional
                If True (the default) then the file locations from
                metadata constructs are also returned.

        :Returns:

            `set`
                The unique file locations as absolute paths with no
                trailing path name component separator.

        **Examples**

        >>> f.file_locations()
        {'/home/data1', 'file:///data2'}

        """
        out = super().file_locations()
        if constructs:
            for c in self.constructs.filter_by_data(todict=True).values():
                out.update(c.file_locations())

        return out

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def flip(self, axes=None, inplace=False, i=False, **kwargs):
        """Flip (reverse the direction of) axes of the field.

        .. seealso:: `domain_axis`, `flatten`, `insert_dimension`,
                     `squeeze`, `transpose`, `unsqueeze`

        :Parameters:

            axes: (sequence of) `str` or `int`, optional
                Select the domain axes to flip, defined by the domain
                axes that would be selected by passing each given axis
                description to a call of the field construct's
                `domain_axis` method. For example, for a value of
                ``'X'``, the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

                If no axes are provided then all axes are flipped.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field` or `None`
                The field construct with flipped axes, or `None` if
                the operation was in-place.

        **Examples**

        >>> g = f.flip()
        >>> g = f.flip('time')
        >>> g = f.flip(1)
        >>> g = f.flip(['time', 1, 'dim2'])
        >>> f.flip(['dim2'], inplace=True)

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "flip", kwargs, version="3.0.0", removed_at="4.0.0"
            )  # pragma: no cover

        if axes is None and not kwargs:
            # Flip all the axes
            axes = set(self.get_data_axes(default=()))
            iaxes = list(range(self.ndim))
        else:
            if isinstance(axes, (str, int)):
                axes = (axes,)

            axes = set([self.domain_axis(axis, key=True) for axis in axes])

            data_axes = self.get_data_axes(default=())
            iaxes = [
                data_axes.index(axis)
                for axis in axes.intersection(self.get_data_axes())
            ]

        # Flip the requested axes in the field's data array
        f = _inplace_enabled_define_and_cleanup(self)
        super(Field, f).flip(iaxes, inplace=True)

        # Flip any constructs which span the flipped axes
        for key, construct in f.constructs.filter_by_data(todict=True).items():
            construct_axes = f.get_data_axes(key)
            construct_flip_axes = axes.intersection(construct_axes)
            if construct_flip_axes:
                iaxes = [
                    construct_axes.index(axis) for axis in construct_flip_axes
                ]
                construct.flip(iaxes, inplace=True)

        return f

    def argmax(self, axis=None, unravel=False):
        """Return the indices of the maximum values along an axis.

        If no axis is specified then the returned index locates the
        maximum of the whole data.

        In case of multiple occurrences of the maximum values, the
        indices corresponding to the first occurrence are returned.

        **Performance**

        If the data index is returned as a `tuple` (see the *unravel*
        parameter) then all delayed operations are computed.

        .. versionadded:: 3.0.0

        .. seealso:: `argmin`, `where`, `cf.Data.argmax`

        :Parameters:

            axis: optional
                Select the domain axis over which to locate the
                maximum values, defined by the domain axis that would
                be selected by passing the given *axis* to a call of
                the field construct's `domain_axis` method. For
                example, for a value of ``'X'``, the domain axis
                construct returned by ``f.domain_axis('X')`` is
                selected.

                By default the maximum over the flattened data is
                located.

            unravel: `bool`, optional
                If True then when locating the maximum over the whole
                data, return the location as an integer index for each
                axis as a `tuple`. By default an index to the
                flattened array is returned in this case. Ignored if
                locating the maxima over a subset of the axes.

        :Returns:

            `Data` or `tuple` of `int`
                The location of the maximum, or maxima.

        **Examples**

        >>> f = cf.example_field(2)
        >>> print(f)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa

        Find the T axis indices of the maximum at each X-Y location:

        >>> i = f.argmax('T')
        >>> print(i.array)
        [[31 10  2 27 16  7 21 17]
         [24 34  1 22 28 17 19 13]
         [20  7 15 21  6 20  8 18]
         [24  1  7 18 19  6 11 18]
         [16  8 12  9 12  2  6 17]]

        Find the coordinates of the global maximum value, showing that
        it occurs on 1960-11-16 at location 292.5 degrees east, 45.0
        degrees north:

        >>> g = f[f.argmax(unravel=True)]
        >>> print(g)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(1), latitude(1), longitude(1)) K
        Cell methods    : area: mean
        Dimension coords: time(1) = [1960-11-16 00:00:00]
                        : latitude(1) = [45.0] degrees_north
                        : longitude(1) = [292.5] degrees_east
                        : air_pressure(1) = [850.0] hPa

        See `cf.Data.argmax` for further examples.

        """
        if axis is not None:
            axis = self.domain_axis(axis, key=True)
            axis = self.get_data_axes().index(axis)

        return self.data.argmax(axis=axis, unravel=unravel)

    def argmin(self, axis=None, unravel=False):
        """Return the indices of the minimum values along an axis.

        If no axis is specified then the returned index locates the
        minimum of the whole data.

        In case of multiple occurrences of the minimum values, the
        indices corresponding to the first occurrence are returned.

        **Performance**

        If the data index is returned as a `tuple` (see the *unravel*
        parameter) then all delayed operations are computed.

        .. versionadded:: 3.15.1

        .. seealso:: `argmax`, `where`, `cf.Data.argmin`

        :Parameters:

            axis: optional
                Select the domain axis over which to locate the
                minimum values, defined by the domain axis that would
                be selected by passing the given *axis* to a call of
                the field construct's `domain_axis` method. For
                example, for a value of ``'X'``, the domain axis
                construct returned by ``f.domain_axis('X')`` is
                selected.

                By default the minimum over the flattened data is
                located.

            unravel: `bool`, optional
                If True then when locating the minimum over the whole
                data, return the location as an integer index for each
                axis as a `tuple`. By default an index to the
                flattened array is returned in this case. Ignored if
                locating the minima over a subset of the axes.

        :Returns:

            `Data` or `tuple` of `int`
                The location of the minimum, or minima.

        **Examples**

        >>> f = cf.example_field(2)
        >>> print(f)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa

        Find the T axis indices of the minimum at each X-Y location:

        >>> i = f.argmin('T')
        >>> print(i.array)
        [[ 5 14  4 32 11 34  9 27]
         [10  7  4 21 11 10  3 33]
         [21 33  1 30 26  8 33  4]
         [15 10 12 19 23 20 30 25]
         [28 33 31 11 15 12  9  7]]

        Find the coordinates of the global minimum value, showing that
        it occurs on 1960-03-16 12:00:00 at location 292.5 degrees
        east, -45.0 degrees north:

        >>> g = f[f.argmin(unravel=True)]
        >>> print(g)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(1), latitude(1), longitude(1)) K
        Cell methods    : area: mean
        Dimension coords: time(1) = [1960-03-16 12:00:00]
                        : latitude(1) = [-45.0] degrees_north
                        : longitude(1) = [292.5] degrees_east
                        : air_pressure(1) = [850.0] hPa

        See `cf.Data.argmin` for further examples.

        """
        if axis is not None:
            axis = self.domain_axis(axis, key=True)
            axis = self.get_data_axes().index(axis)

        return self.data.argmin(axis=axis, unravel=unravel)

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def squeeze(self, axes=None, inplace=False, i=False, **kwargs):
        """Remove size 1 axes from the data.

        By default all size 1 axes are removed, but particular size 1 axes
        may be selected for removal.

        Squeezed domain axis constructs are not removed from the metadata
        constructs, nor from the domain of the field construct.

        .. seealso:: `domain_axis`, `flatten`, `insert_dimension`, `flip`,
                     `remove_axes`, `transpose`, `unsqueeze`

        :Parameters:

            axes: (sequence of) `str` or `int`, optional
                Select the domain axes to squeeze, defined by the domain
                axes that would be selected by passing each given axis
                description to a call of the field construct's
                `domain_axis` method. For example, for a value of ``'X'``,
                the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

                If no axes are provided then all size 1 axes are squeezed.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field` or `None`
                The field construct with squeezed data, or `None` if the
                operation was in-place.

                **Examples**

        >>> g = f.squeeze()
        >>> g = f.squeeze('time')
        >>> g = f.squeeze(1)
        >>> g = f.squeeze(['time', 1, 'dim2'])
        >>> f.squeeze(['dim2'], inplace=True)

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "squeeze", kwargs, version="3.0.0", removed_at="4.0.0"
            )  # pragma: no cover

        data_axes = self.get_data_axes()

        if axes is None:
            domain_axes = self.domain_axes(todict=True)
            axes = [
                axis
                for axis in data_axes
                if domain_axes[axis].get_size(None) == 1
            ]
        else:
            if isinstance(axes, (str, int)):
                axes = (axes,)

            axes = [self.domain_axis(x, key=True) for x in axes]
            axes = set(axes).intersection(data_axes)

        iaxes = [data_axes.index(axis) for axis in axes]

        # Squeeze the field's data array
        return super().squeeze(iaxes, inplace=inplace)

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def swapaxes(self, axis0, axis1, inplace=False, i=False):
        """Interchange two axes of the data.

        .. seealso:: `flatten`, `flip`, `insert_dimension`, `squeeze`,
                     `transpose`

        :Parameters:

            axis0, axis1: TODO
                Select the axes to swap. Each axis is identified by its
                original integer position.

            {{inplace: `bool`, optional}}

        :Returns:

            `Field` or `None`
                The field construct with data with swapped axis
                positions. If the operation was in-place then `None` is
                returned.

        **Examples**

        >>> f.shape
        (1, 2, 3)
        >>> f.swapaxes(1, 0).shape
        (2, 1, 3)
        >>> f.swapaxes(0, -1).shape
        (3, 2, 1)
        >>> f.swapaxes(1, 1).shape
        (1, 2, 3)
        >>> f.swapaxes(-1, -1).shape
        (1, 2, 3)

        """
        data_axes = self.get_data_axes(default=None)

        da_key0 = self.domain_axis(axis0, key=True)
        da_key1 = self.domain_axis(axis1, key=True)

        if da_key0 not in data_axes:
            raise ValueError(
                f"Can't swapaxes: Bad axis specification: {axis0!r}"
            )

        if da_key1 not in data_axes:
            raise ValueError(
                f"Can't swapaxes: Bad axis specification: {axis1!r}"
            )

        axis0 = data_axes.index(da_key0)
        axis1 = data_axes.index(da_key1)

        f = _inplace_enabled_define_and_cleanup(self)
        super(Field, f).swapaxes(axis0, axis1, inplace=True)

        if data_axes is not None:
            data_axes = list(data_axes)
            data_axes[axis1], data_axes[axis0] = (
                data_axes[axis0],
                data_axes[axis1],
            )
            f.set_data_axes(data_axes)

        return f

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def transpose(
        self,
        axes=None,
        constructs=False,
        inplace=False,
        items=True,
        i=False,
        **kwargs,
    ):
        """Permute the axes of the data array.

        By default the order of the axes is reversed, but any ordering may
        be specified by selecting the axes of the output in the required
        order.

        By default metadata constructs are not transposed, but they may be
        if the *constructs* parameter is set.

        .. seealso:: `domain_axis`, `flatten`, `insert_dimension`, `flip`,
                     `squeeze`, `unsqueeze`

        :Parameters:

            axes: (sequence of) `str` or `int`, optional
                Select the domain axis order, defined by the domain axes
                that would be selected by passing each given axis
                description to a call of the field construct's
                `domain_axis` method. For example, for a value of ``'X'``,
                the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

                Each dimension of the field construct's data must be
                provided, or if no axes are specified then the axis order
                is reversed.

            constructs: `bool`, optional
                If True then metadata constructs are also transposed so
                that their axes are in the same relative order as in the
                transposed data array of the field. By default metadata
                constructs are not altered.

            {{inplace: `bool`, optional}}

            items: deprecated at version 3.0.0
                Use the *constructs* parameter instead.

            {{i: deprecated at version 3.0.0}}

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field` or `None`
                The field construct with transposed data, or `None` if the
                operation was in-place.

        **Examples**

        >>> f.ndim
        3
        >>> g = f.transpose()
        >>> g = f.transpose(['time', 1, 'dim2'])
        >>> f.transpose(['time', -2, 'dim2'], inplace=True)

        """
        if not items:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "transpose",
                {"items": items},
                "Use keyword 'constructs' instead.",
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "transpose", kwargs, version="3.0.0", removed_at="4.0.0"
            )  # pragma: no cover

        if axes is None:
            iaxes = list(range(self.ndim - 1, -1, -1))
        else:
            data_axes = self.get_data_axes(default=())
            if isinstance(axes, (str, int)):
                axes = (axes,)

            axes2 = [self.domain_axis(x, key=True) for x in axes]

            if sorted(axes2) != sorted(data_axes):
                raise ValueError(
                    f"Can't transpose {self.__class__.__name__}: "
                    f"Bad axis specification: {axes!r}"
                )

            iaxes = [data_axes.index(axis) for axis in axes2]

        # Transpose the field's data array
        return super().transpose(iaxes, constructs=constructs, inplace=inplace)

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def unsqueeze(self, inplace=False, i=False, axes=None, **kwargs):
        """Insert size 1 axes into the data array.

        All size 1 domain axes which are not spanned by the field
        construct's data are inserted.

        The axes are inserted into the slowest varying data array positions.

        .. seealso:: `flatten`, `flip`, `insert_dimension`, `squeeze`,
                     `transpose`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            axes: deprecated at version 3.0.0

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field` or `None`
                The field construct with size-1 axes inserted in its data,
                or `None` if the operation was in-place.

        **Examples**

        >>> g = f.unsqueeze()
        >>> f.unsqueeze(['dim2'], inplace=True)

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "unsqueeze", kwargs, version="3.0.0", removed_at="4.0.0"
            )  # pragma: no cover

        if axes is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "unsqueeze",
                {"axes": axes},
                "All size one domain axes missing from the data are "
                "inserted. Use method 'insert_dimension' to insert an "
                "individual size one domain axis.",
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        f = _inplace_enabled_define_and_cleanup(self)

        size_1_axes = self.domain_axes(filter_by_size=(1,), todict=True)
        for axis in set(size_1_axes).difference(self.get_data_axes()):
            f.insert_dimension(axis, position=0, inplace=True)

        return f

    def domain_axis_position(self, *identity, **filter_kwargs):
        """Return the position in the data of a domain axis construct.

        .. versionadded:: 3.0.0

        .. seealso:: `domain_axis`

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique domain axis construct returned by
                ``f.domain_axis(*identity, **filter_kwargs)``. See
                `domain_axis` for details.

        :Returns:

            `int`
                The position in the field construct's data of the
                selected domain axis construct.

        **Examples**

        >>> f
        <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
        >>> f.get_data_axes()
        ('domainaxis0', 'domainaxis1', 'domainaxis2')
        >>> f.domain_axis_position('T')
        0
        >>> f.domain_axis_position('latitude')
        1
        >>> f.domain_axis_position('domainaxis1')
        1
        >>> f.domain_axis_position(2)
        2
        >>> f.domain_axis_position(-2)
        1

        """
        key = self.domain_axis(*identity, key=True)
        return self.get_data_axes().index(key)

    def axes_names(self, *identities, **kwargs):
        """Return canonical identities for each domain axis construct.

        :Parameters:

            kwargs: deprecated at version 3.0.0

        :Returns:

            `dict`
                The canonical name for the domain axis construct.

        **Examples**

        >>> f.axis_names()
        {'domainaxis0': 'atmosphere_hybrid_height_coordinate',
         'domainaxis1': 'grid_latitude',
         'domainaxis2': 'grid_longitude'}

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "axes_names", kwargs, version="3.0.0", removed_at="4.0.0"
            )  # pragma: no cover

        out = self.domain_axes(todict=True).copy()

        for key in tuple(out):
            value = self.constructs.domain_axis_identity(key)
            if value is not None:
                out[key] = value
            else:
                del out[key]

        return out

    def axis_size(
        self, *identity, default=ValueError(), axes=None, **filter_kwargs
    ):
        """Return the size of a domain axis construct.

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique domain axis construct returned by
                ``f.domain_axis(*identity, **filter_kwargs)``. See
                `domain_axis` for details.

            default: optional
                Return the value of the *default* parameter if a domain
                axis construct can not be found. If set to an `Exception`
                instance then it will be raised instead.

            axes: deprecated at version 3.0.0

        :Returns:

            `int`
                The size of the selected domain axis

        **Examples**

        >>> f
        <CF Field: eastward_wind(time(3), air_pressure(5), latitude(110), longitude(106)) m s-1>
        >>> f.axis_size('longitude')
        106
        >>> f.axis_size('Z')
        5

        """
        if axes:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "axis_size",
                "Use keyword 'identity' instead.",
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        axis = self.domain_axis(*identity, default=None, **filter_kwargs)
        if axis is None:
            return self._default(default)

        return axis.get_size(default=default)

    def grad_xy(self, x_wrap=None, one_sided_at_boundary=False, radius=None):
        r"""Calculate the (X, Y) gradient vector.

        The horizontal gradient vector of a scalar function is
        calculated from a field that has dimension coordinates of X
        and Y, in either Cartesian (e.g. plane projection) or
        spherical polar coordinate systems.

        The horizontal gradient vector in Cartesian coordinates is
        given by:

        .. math:: \nabla f(x, y) = \left(
                                   \frac{\partial f}{\partial x},
                                   \frac{\partial f}{\partial y}
                                   \right)

        The horizontal gradient vector in spherical polar coordinates
        is given by:

        .. math:: \nabla f(\theta, \phi) = \left(
                                           \frac{1}{r}
                                           \frac{\partial f}{\partial \theta},
                                           \frac{1}{r \sin\theta}
                                           \frac{\partial f}{\partial \phi}
                                           \right)

        where *r* is radial distance to the origin, :math:`\theta` is
        the polar angle with respect to polar axis, and :math:`\phi`
        is the azimuthal angle.

        The gradient vector components are calculated using centred
        finite differences apart from at the boundaries (see the
        *x_wrap* and *one_sided_at_boundary* parameters). If missing
        values are present then missing values will be returned at all
        points where a centred finite difference could not be
        calculated.

        .. versionadded:: 3.12.0

        .. seealso:: `derivative`, `iscyclic`, `laplacian_xy`,
                     `cf.curl_xy`, `cf.div_xy`

        :Parameters:

            x_wrap: `bool`, optional
                Whether the X axis is cyclic or not. By default
                *x_wrap* is set to the result of this call to the
                field construct's `iscyclic` method:
                ``f.iscyclic('X')``. If the X axis is cyclic then
                centred differences at one boundary will always use
                values from the other boundary, regardless of the
                setting of *one_sided_at_boundary*.

                The cyclicity of the Y axis is always set to the
                result of ``f.iscyclic('Y')``.

            one_sided_at_boundary: `bool`, optional
                If True then one-sided finite differences are
                calculated at the non-cyclic boundaries. By default
                missing values are set at non-cyclic boundaries.

            {{radius: optional}}

        :Returns:

            `FieldList`
                The horizontal gradient vector of the scalar field.

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
        >>> f[...] = 0.1
        >>> print(f.array)
        [[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
         [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
         [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
         [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
         [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]]
        >>> fx, fy = f.grad_xy(radius='earth')
        >>> fx, fy
        (<CF Field: long_name=X gradient of specific_humidity(latitude(5), longitude(8)) m-1.rad-1>,
         <CF Field: long_name=Y gradient of specific_humidity(latitude(5), longitude(8)) m-1.rad-1>)
        >>> print(fx.array)
        [[0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]]
        >>> print(fy.array)
        [[-- -- -- -- -- -- -- --]
         [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
         [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
         [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
         [-- -- -- -- -- -- -- --]]
        >>> fx, fy = f.grad_xy(radius='earth', one_sided_at_boundary=True)
        >>> print(fy.array)
        [[0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]]

        """
        f = self.copy()
        identity = f.identity()

        x_key, x_coord = f.dimension_coordinate(
            "X", item=True, default=(None, None)
        )
        y_key, y_coord = f.dimension_coordinate(
            "Y", item=True, default=(None, None)
        )

        if x_coord is None:
            raise ValueError("Field has no unique 'X' dimension coordinate")

        if y_coord is None:
            raise ValueError("Field has no unique 'Y' dimension coordinate")

        if x_wrap is None:
            x_wrap = f.iscyclic(x_key)

        x_units = x_coord.Units
        y_units = y_coord.Units

        # Check for spherical polar coordinates
        latlon = (x_units.islongitude and y_units.islatitude) or (
            x_units.units == "degrees" and y_units.units == "degrees"
        )

        if latlon:
            # --------------------------------------------------------
            # Spherical polar coordinates
            # --------------------------------------------------------
            # Convert latitude and longitude units to radians, so that
            # the units of the result are nice.
            x_coord.Units = _units_radians
            y_coord.Units = _units_radians

            # Ensure that the lat and lon dimension coordinates have
            # standard names, so that metadata-aware broadcasting
            # works as expected when all of their units are radians.
            x_coord.standard_name = "longitude"
            y_coord.standard_name = "latitude"

            # Get theta as a field that will broadcast to f, and
            # adjust its values so that theta=0 is at the north pole.
            theta = np.pi / 2 - f.convert(y_key, full_domain=True)

            r = f.radius(default=radius)

            X = f.derivative(
                x_key, wrap=x_wrap, one_sided_at_boundary=one_sided_at_boundary
            ) / (theta.sin() * r)

            Y = (
                f.derivative(
                    y_key,
                    wrap=None,
                    one_sided_at_boundary=one_sided_at_boundary,
                )
                / r
            )

            # Reset latitude and longitude coordinate units
            X.dimension_coordinate("longitude").Units = x_units
            X.dimension_coordinate("latitude").Units = y_units

            Y.dimension_coordinate("longitude").Units = x_units
            Y.dimension_coordinate("latitude").Units = y_units
        else:
            # --------------------------------------------------------
            # Cartesian coordinates
            # --------------------------------------------------------
            X = f.derivative(
                x_key, wrap=x_wrap, one_sided_at_boundary=one_sided_at_boundary
            )

            Y = f.derivative(
                y_key, wrap=None, one_sided_at_boundary=one_sided_at_boundary
            )

        # Set the standard name and long name
        X.set_property("long_name", f"X gradient of {identity}")
        Y.set_property("long_name", f"Y gradient of {identity}")
        X.del_property("standard_name", None)
        Y.del_property("standard_name", None)

        return FieldList((X, Y))

    @_inplace_enabled(default=False)
    @_manage_log_level_via_verbosity
    def halo(
        self,
        depth,
        axes=None,
        tripolar=None,
        fold_index=-1,
        inplace=False,
        verbose=None,
        size=None,
    ):
        """Expand the field construct by adding a halo to its data.

        The halo may be applied over a subset of the data dimensions and
        each dimension may have a different halo size (including
        zero). The halo region is populated with a copy of the proximate
        values from the original data.

        The metadata constructs are similarly extended where appropriate.

        **Cyclic axes**

        A cyclic axis that is expanded with a halo of at least size 1 is
        no longer considered to be cyclic.

        **Tripolar domains**

        Global tripolar domains are a special case in that a halo added to
        the northern end of the "Y" axis must be filled with values that
        are flipped in "X" direction. Such domains can not be identified
        from the field construct's metadata, so need to be explicitly
        indicated with the *tripolar* parameter.

        .. versionadded:: 3.5.0

        :Parameters:

            depth: `int` or `dict`
                Specify the size of the halo for each axis.

                If *depth* is a non-negative `int` then this is the
                halo size that is applied to all of the axes defined
                by the *axes* parameter.

                Alternatively, halo sizes may be assigned to axes
                individually by providing a `dict` for which a key
                specifies an axis (by passing the axis description to
                a call of the field construct's `domain_axis`
                method. For example, for a value of ``'X'``, the
                domain axis construct returned by
                ``f.domain_axis('X')``) with a corresponding value of
                the halo size for that axis. Axes not specified by the
                dictionary are not expanded, and the *axes* parameter
                must not also be set.

                *Parameter example:*
                  Specify a halo size of 1 for all otherwise selected
                  axes: ``1``

                *Parameter example:*
                  Specify a halo size of zero: ``0``. This results in
                  no change to the data shape.

                *Parameter example:*
                  For data with three dimensions, specify a halo size
                  of 3 for the first dimension and 1 for the second
                  dimension: ``{0: 3, 1: 1}``. This is equivalent to
                  ``{0: 3, 1: 1, 2: 0}``

                *Parameter example:*
                  Specify a halo size of 2 for the "longitude" and
                  "latitude" axes: ``depth=2, axes=['latutude',
                  'longitude']``, or equivalently ``depth={'latitude':
                  2, 'longitude': 2}``.

            axes: (sequence of) `str` or `int`, optional
                Select the domain axes to be expanded, defined by the
                domain axes that would be selected by passing each given
                axis description to a call of the field construct's
                `domain_axis` method. For example, for a value of ``'X'``,
                the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

                By default, or if *axes* is `None`, all axes that span the
                data are selected. No axes are expanded if *axes* is an
                empty sequence.

                *Parameter example:*
                  ``axes='X'``

                *Parameter example:*
                  ``axes=['Y']``

                *Parameter example:*
                  ``axes=['X', 'Y']``

                *Parameter example:*
                  ``axes='longitude'``

                *Parameter example:*
                  ``axes=2``

                *Parameter example:*
                  ``axes='ncdim%i'``

            tripolar: `dict`, optional
                A dictionary defining the "X" and "Y" axes of a global
                tripolar domain. This is necessary because in the
                global tripolar case the "X" and "Y" axes need special
                treatment, as described above. It must have keys
                ``'X'`` and ``'Y'``, whose values identify the
                corresponding domain axis construct by passing the
                value to a call of the field construct's `domain_axis`
                method. For example, for a value of ``'ncdim%i'``, the
                domain axis construct returned by
                ``f.domain_axis('ncdim%i')``.

                The "X" and "Y" axes must be a subset of those
                identified by the *depth* or *axes* parameter.

                See the *fold_index* parameter.

                *Parameter example:*
                  Define the "X" and Y" axes by their netCDF dimension
                  names: ``{'X': 'ncdim%i', 'Y': 'ncdim%j'}``

                *Parameter example:*
                  Define the "X" and Y" axes by positions 2 and 1
                  respectively of the data: ``{'X': 2, 'Y': 1}``

            fold_index: `int`, optional
                Identify which index of the "Y" axis corresponds to the
                fold in "X" axis of a tripolar grid. The only valid values
                are ``-1`` for the last index, and ``0`` for the first
                index. By default it is assumed to be the last
                index. Ignored if *tripolar* is `None`.

            {{inplace: `bool`, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

            size: deprecated at version 3.14.0
                Use the *depth* parameter instead.

        :Returns:

            `Field` or `None`
                The expanded field construct, or `None` if the operation
                was in-place.

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
        >>> print(f.array)
        [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
         [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
         [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
         [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
         [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
        >>> print(f.coordinate('X').array)
        [ 22.5  67.5 112.5 157.5 202.5 247.5 292.5 337.5]

        >>> g = f.halo(1)
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(7), longitude(10)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(7) = [-75.0, ..., 75.0] degrees_north
                        : longitude(10) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(g.array)
        [[0.007 0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029 0.029]
         [0.007 0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029 0.029]
         [0.023 0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066 0.066]
         [0.11  0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011 0.011]
         [0.029 0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017 0.017]
         [0.006 0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013 0.013]
         [0.006 0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013 0.013]]
        >>> print(g.coordinate('X').array)
        [ 22.5  22.5  67.5 112.5 157.5 202.5 247.5 292.5 337.5 337.5]

        >>> g = f.halo(1, axes='Y')
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(7), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(7) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(g.array)
        [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
         [0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
         [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
         [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
         [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
         [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]
         [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
        >>> h = f.halo({'Y': 1})
        >>> h.equals(g)
        True

        >>> g = f.halo({'Y': 2, 'X': 1})
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(9), longitude(10)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(9) = [-75.0, ..., 75.0] degrees_north
                        : longitude(10) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(g.array)
        [[0.007 0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029 0.029]
         [0.023 0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066 0.066]
         [0.007 0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029 0.029]
         [0.023 0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066 0.066]
         [0.11  0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011 0.011]
         [0.029 0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017 0.017]
         [0.006 0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013 0.013]
         [0.029 0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017 0.017]
         [0.006 0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013 0.013]]

        """
        f = _inplace_enabled_define_and_cleanup(self)

        # TODODASKAPI
        if size is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "halo",
                {"size": None},
                message="Use the 'depth' parameter instead.",
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        # Set the halo depth for each axis.
        data_axes = f.get_data_axes(default=())
        if isinstance(depth, dict):
            if axes is not None:
                raise ValueError(
                    "Can't set existing axes when depth is a dict."
                )

            axis_halo = {
                self.domain_axis(k, key=True): v for k, v in depth.items()
            }

            if not set(data_axes).issuperset(axis_halo):
                raise ValueError(
                    f"Can't apply halo: Bad axis specification: {depth!r}"
                )
        else:
            if axes is None:
                axes = data_axes

            if isinstance(axes, (str, int)):
                axes = (axes,)

            axis_halo = {self.domain_axis(k, key=True): depth for k in axes}

        if tripolar:
            # Find the X and Y axes of a tripolar grid
            tripolar = tripolar.copy()
            X_axis = tripolar.pop("X", None)
            Y_axis = tripolar.pop("Y", None)

            if X_axis is None:
                raise ValueError("Must provide a tripolar 'X' axis.")

            if Y_axis is None:
                raise ValueError("Must provide a tripolar 'Y' axis.")

            X = self.domain_axis(X_axis, key=True)
            Y = self.domain_axis(Y_axis, key=True)

            try:
                i_X = data_axes.index(X)
            except ValueError:
                raise ValueError(f"Axis {X_axis!r} is not spanned by the data")

            try:
                i_Y = data_axes.index(Y)
            except ValueError:
                raise ValueError(f"Axis {Y_axis!r} is not spanned by the data")

            tripolar["X"] = i_X
            tripolar["Y"] = i_Y

            tripolar_axes = {X: "X", Y: "Y"}

        # Add halos to the field construct's data
        depth = {data_axes.index(axis): h for axis, h, in axis_halo.items()}

        f.data.halo(
            depth,
            tripolar=tripolar,
            fold_index=fold_index,
            inplace=True,
            verbose=verbose,
        )

        # Change domain axis sizes
        for axis, h in axis_halo.items():
            d = f.domain_axis(axis)
            d.set_size(d.get_size() + 2 * h)

        # Add halos to metadata constructs
        for key, c in f.constructs.filter_by_data(todict=True).items():
            construct_axes = f.get_data_axes(key)
            construct_size = {
                construct_axes.index(axis): h
                for axis, h in axis_halo.items()
                if axis in construct_axes
            }

            if not construct_size:
                # This construct does not span an expanded axis
                continue

            construct_tripolar = False
            if tripolar and set(construct_axes).issuperset(tripolar_axes):
                construct_tripolar = {
                    axis_type: construct_axes.index(axis)
                    for axis, axis_type in tripolar_axes.items()
                }

            c.halo(
                construct_size,
                tripolar=construct_tripolar,
                fold_index=fold_index,
                inplace=True,
                verbose=verbose,
            )

        return f

    @_inplace_enabled(default=False)
    def pad_missing(self, axis, pad_width=None, to_size=None, inplace=False):
        """Pad an axis with missing data.

         The field's data and all metadata constructs that span the
         axis are padded.

        .. versionadded:: 3.16.1

        :Parameters:

             axis: `str` or `int`
                 Select the domain axis which is to be padded, defined
                 by that which would be selected by passing the given
                 axis description to a call of the field construct's
                 `domain_axis` method. For example, for a value of
                 ``'X'``, the domain axis construct returned by
                 ``f.domain_axis('X')`` is selected.

             {{pad_width: sequence of `int`, optional}}

             {{to_size: `int`, optional}}

             {{inplace: `bool`, optional}}

        :Returns:

            `Field` or `None`
                The padded field construct, or `None` if the operation
                was in-place.

        **Examples**

        >>> f = cf.example_field(6)
        >>> print(f)
        Field: precipitation_amount (ncvar%pr)
        --------------------------------------
        Data            : precipitation_amount(cf_role=timeseries_id(2), time(4))
        Dimension coords: time(4) = [2000-01-16 12:00:00, ..., 2000-04-15 00:00:00] gregorian
        Auxiliary coords: latitude(cf_role=timeseries_id(2)) = [25.0, 7.0] degrees_north
                        : longitude(cf_role=timeseries_id(2)) = [10.0, 40.0] degrees_east
                        : cf_role=timeseries_id(cf_role=timeseries_id(2)) = [x1, y2]
                        : altitude(cf_role=timeseries_id(2), 3, 4) = [[[1.0, ..., --]]] m
        Coord references: grid_mapping_name:latitude_longitude
        >>> print(f.array)
        [[1. 2. 3. 4.]
         [5. 6. 7. 8.]]
        >>> g = f.pad_missing('T', (0, 5))
        >>> print(g)
        Field: precipitation_amount (ncvar%pr)
        --------------------------------------
        Data            : precipitation_amount(cf_role=timeseries_id(2), time(9))
        Dimension coords: time(9) = [2000-01-16 12:00:00, ..., --] gregorian
        Auxiliary coords: latitude(cf_role=timeseries_id(2)) = [25.0, 7.0] degrees_north
                        : longitude(cf_role=timeseries_id(2)) = [10.0, 40.0] degrees_east
                        : cf_role=timeseries_id(cf_role=timeseries_id(2)) = [x1, y2]
                        : altitude(cf_role=timeseries_id(2), 3, 4) = [[[1.0, ..., --]]] m
        Coord references: grid_mapping_name:latitude_longitude
        >>> print(g.array)
        [[1.0 2.0 3.0 4.0 -- -- -- -- --]
         [5.0 6.0 7.0 8.0 -- -- -- -- --]]
        >>> h = g.pad_missing('cf_role=timeseries_id', (0, 1))
        >>> print(h)
        Field: precipitation_amount (ncvar%pr)
        --------------------------------------
        Data            : precipitation_amount(cf_role=timeseries_id(3), time(9))
        Dimension coords: time(9) = [2000-01-16 12:00:00, ..., --] gregorian
        Auxiliary coords: latitude(cf_role=timeseries_id(3)) = [25.0, 7.0, --] degrees_north
                        : longitude(cf_role=timeseries_id(3)) = [10.0, 40.0, --] degrees_east
                        : cf_role=timeseries_id(cf_role=timeseries_id(3)) = [x1, y2, --]
                        : altitude(cf_role=timeseries_id(3), 3, 4) = [[[1.0, ..., --]]] m
        Coord references: grid_mapping_name:latitude_longitude
        >>> print(h.array)
        [[1.0 2.0 3.0 4.0 -- -- -- -- --]
         [5.0 6.0 7.0 8.0 -- -- -- -- --]
         [ --  --  --  -- -- -- -- -- --]]

        >>> print(f.pad_missing('time', to_size=6))
        Field: precipitation_amount (ncvar%pr)
        --------------------------------------
        Data            : precipitation_amount(cf_role=timeseries_id(2), time(6))
        Dimension coords: time(6) = [2000-01-16 12:00:00, ..., --] gregorian
        Auxiliary coords: latitude(cf_role=timeseries_id(2)) = [25.0, 7.0] degrees_north
                        : longitude(cf_role=timeseries_id(2)) = [10.0, 40.0] degrees_east
                        : cf_role=timeseries_id(cf_role=timeseries_id(2)) = [x1, y2]
                        : altitude(cf_role=timeseries_id(2), 3, 4) = [[[1.0, ..., --]]] m
        Coord references: grid_mapping_name:latitude_longitude

        """
        f = _inplace_enabled_define_and_cleanup(self)

        try:
            axis1 = f._parse_axes(axis)
        except ValueError:
            raise ValueError(
                f"Can't pad_missing: Bad axis specification: {axis!r}"
            )

        if len(axis1) != 1:
            raise ValueError(
                f"Can't pad_missing: Bad axis specification: {axis!r}"
            )

        data_axes = f.get_data_axes()
        axis = axis1[0]
        iaxis = data_axes.index(axis)

        # Pad the field
        super(Field, f).pad_missing(
            iaxis, pad_width=pad_width, to_size=to_size, inplace=True
        )

        # Set new domain axis size
        domain_axis = f.domain_axis(axis)
        domain_axis.set_size(f.shape[iaxis])

        data_axes = f.constructs.data_axes()
        for key, construct in f.constructs.filter_by_data(todict=True).items():
            construct_axes = data_axes[key]
            if axis not in construct_axes:
                continue

            # Pad the construct
            iaxis = construct_axes.index(axis)
            construct.pad_missing(
                iaxis, pad_width=pad_width, to_size=to_size, inplace=True
            )

        return f

    def percentile(
        self,
        ranks,
        axes=None,
        method="linear",
        squeeze=False,
        mtol=1,
        interpolation=None,
    ):
        """Compute percentiles of the data along the specified axes.

        The default is to compute the percentiles along a flattened
        version of the data.

        If the input data are integers, or floats smaller than float64, or
        the input data contains missing values, then output data type is
        float64. Otherwise, the output data type is the same as that of
        the input.

        If multiple percentile ranks are given then a new, leading data
        dimension is created so that percentiles can be stored for each
        percentile rank.

        The output field construct has a new dimension coordinate
        construct that records the percentile ranks represented by its
        data.

        **Accuracy**

        The `percentile` method returns results that are consistent
        with `numpy.percentile`, which may be different to those
        created by `dask.percentile`. The dask method uses an
        algorithm that calculates approximate percentiles which are
        likely to be different from the correct values when there are
        two or more dask chunks.

        >>> import numpy as np
        >>> import dask.array as da
        >>> import cf
        >>> a = np.arange(101)
        >>> dx = da.from_array(a, chunks=10)
        >>> da.percentile(dx, 40).compute()
        array([40.36])
        >>> np.percentile(a, 40)
        40.0
        >>> d = cf.Data(a, chunks=10)
        >>> d.percentile(40).array
        array([40.])

        .. versionadded:: 3.0.4

        .. seealso:: `bin`, `collapse`, `digitize`, `where`

        :Parameters:

            ranks: (sequence of) number
                Percentile ranks, or sequence of percentile ranks, to
                compute, which must be between 0 and 100 inclusive.

            axes: (sequence of) `str` or `int`, optional
                Select the domain axes over which to calculate the
                percentiles, defined by the domain axes that would be
                selected by passing each given axis description to a call
                of the field construct's `domain_axis` method. For
                example, for a value of ``'X'``, the domain axis construct
                returned by ``f.domain_axis('X')`` is selected.

                By default, or if *axes* is `None`, all axes are selected.

            {{percentile method: `str`, optional}}

                .. versionadded:: 3.14.0

            squeeze: `bool`, optional
                If True then all size 1 axes are removed from the returned
                percentiles data. By default axes over which percentiles
                have been calculated are left in the result as axes with
                size 1, meaning that the result is guaranteed to broadcast
                correctly against the original data.

            mtol: number, optional
                Set the fraction of input data elements which is allowed
                to contain missing data when contributing to an individual
                output data element. Where this fraction exceeds *mtol*,
                missing data is returned. The default is 1, meaning that a
                missing datum in the output array occurs when its
                contributing input array elements are all missing data. A
                value of 0 means that a missing datum in the output array
                occurs whenever any of its contributing input array
                elements are missing data. Any intermediate value is
                permitted.

                *Parameter example:*
                  To ensure that an output array element is a missing
                  datum if more than 25% of its input array elements are
                  missing data: ``mtol=0.25``.

            interpolation: deprecated at version 3.14.0
                Use the *method* parameter instead.

        :Returns:

            `Field`
                The percentiles of the original data.

        **Examples**

        >>> f = cf.example_field(0)
        >>> print(f)
        Field: specific_humidity
        ------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: time(1) = [2019-01-01 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
        >>> print(f.array)
        [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
         [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
         [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
         [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
         [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
        >>> p = f.percentile([20, 40, 50, 60, 80])
        >>> print(p)
        Field: specific_humidity
        ------------------------
        Data            : specific_humidity(long_name=Percentile ranks for latitude, longitude dimensions(5), latitude(1), longitude(1)) 1
        Dimension coords: time(1) = [2019-01-01 00:00:00]
                        : latitude(1) = [0.0] degrees_north
                        : longitude(1) = [180.0] degrees_east
                        : long_name=Percentile ranks for latitude, longitude dimensions(5) = [20, ..., 80]
        >>> print(p.array)
        [[[0.0164]]
         [[0.032 ]]
         [[0.036 ]]
         [[0.0414]]
         [[0.0704]]]

        Find the standard deviation of the values above the 80th percentile:

        >>> p80 = f.percentile(80)
        >>> print(p80)
        Field: specific_humidity
        ------------------------
        Data            : specific_humidity(latitude(1), longitude(1)) 1
        Dimension coords: time(1) = [2019-01-01 00:00:00]
                        : latitude(1) = [0.0] degrees_north
                        : longitude(1) = [180.0] degrees_east
                        : long_name=Percentile ranks for latitude, longitude dimensions(1) = [80]
        >>> g = f.where(f<=p80, cf.masked)
        >>> print(g.array)
        [[  --    --    --    --    --    -- -- --]
         [  --    --    --    --    -- 0.073 -- --]
         [0.11 0.131 0.124 0.146 0.087 0.103 -- --]
         [  --    --    --    --    -- 0.072 -- --]
         [  --    --    --    --    --    -- -- --]]
        >>> g.collapse('standard_deviation', weights=True).data
        <CF Data(1, 1): [[0.024609938742357642]] 1>

        Find the mean of the values above the 45th percentile along the
        X axis:

        >>> p45 = f.percentile(45, axes='X')
        >>> print(p45.array)
        [[0.0189 ]
         [0.04515]
         [0.10405]
         [0.04185]
         [0.02125]]
        >>> g = f.where(f<=p45, cf.masked)
        >>> print(g.array)
        [[  -- 0.034    --    --    -- 0.037 0.024 0.029]
         [  --    --    -- 0.062 0.046 0.073    -- 0.066]
         [0.11 0.131 0.124 0.146    --    --    --    --]
         [  -- 0.059    -- 0.07  0.058 0.072    --    --]
         [  -- 0.036    -- 0.035   --  0.037 0.034    --]]
        >>> print(g.collapse('X: mean', weights=True).array)
        [[0.031  ]
         [0.06175]
         [0.12775]
         [0.06475]
         [0.0355 ]]

        Find the histogram bin boundaries associated with given
        percentiles, and digitize the data based on these bins:

        >>> bins = f.percentile([0, 10, 50, 90, 100], squeeze=True)
        >>> print(bins.array)
        [0.003  0.0088 0.036  0.1037 0.146 ]
        >>> i = f.digitize(bins, closed_ends=True)
        >>> print(i.array)
        [[0 1 0 1 1 2 1 1]
         [1 2 2 2 2 2 0 2]
         [3 3 3 3 2 2 2 1]
         [1 2 2 2 2 2 1 1]
         [0 2 1 1 1 2 1 1]]

        """
        # TODODASKAPI: interpolation -> method
        if interpolation is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "percentile",
                {"interpolation": None},
                message="Use the 'method' parameter instead.",
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        data_axes = self.get_data_axes(default=())

        if axes is None:
            axes = data_axes[:]
            iaxes = list(range(self.ndim))
        else:
            if isinstance(axes, (str, int)):
                axes = (axes,)

            axes = set([self.domain_axis(axis, key=True) for axis in axes])
            iaxes = [
                data_axes.index(axis)
                for axis in axes.intersection(self.get_data_axes())
            ]

        data = self.data.percentile(
            ranks,
            axes=iaxes,
            method=method,
            squeeze=False,
            mtol=mtol,
        )

        # ------------------------------------------------------------
        # Initialise the output field with the percentile data
        # ------------------------------------------------------------

        # TODODASK: Make sure that this is OK when `ranks` is a scalar

        out = type(self)()
        out.set_properties(self.properties())

        for axis in [
            axis
            for axis in self.domain_axes(todict=True)
            if axis not in data_axes
        ]:
            out.set_construct(self._DomainAxis(1), key=axis)

        out_data_axes = []
        if data.ndim == self.ndim:
            for n, axis in zip(data.shape, data_axes):
                out_data_axes.append(
                    out.set_construct(self._DomainAxis(n), key=axis)
                )
        elif data.ndim == self.ndim + 1:
            for n, axis in zip(data.shape[1:], data_axes):
                out_data_axes.append(
                    out.set_construct(self._DomainAxis(n), key=axis)
                )

            out_data_axes.insert(
                0, out.set_construct(self._DomainAxis(data.shape[0]))
            )

        out.set_data(data, axes=out_data_axes, copy=False)

        # ------------------------------------------------------------
        # Create dimension coordinate constructs for the percentile
        # axes
        # ------------------------------------------------------------
        if axes:
            for key, c in self.dimension_coordinates(
                filter_by_axis=axes, axis_mode="subset", todict=True
            ).items():
                c_axes = self.get_data_axes(key)

                c = c.copy()

                bounds = c.get_bounds_data(
                    c.get_data(None, _fill_value=False), _fill_value=False
                )
                if bounds is not None and bounds.shape[0] > 1:
                    bounds = Data(
                        [bounds.min().datum(), bounds.max().datum()],
                        units=c.Units,
                    )

                    data = bounds.mean()
                    c.set_data(data, copy=False)

                    bounds.insert_dimension(inplace=True)
                    c.set_bounds(self._Bounds(data=bounds), copy=False)

                out.set_construct(c, axes=c_axes, key=key, copy=False)

        # TODO optimise constructs access?
        other_axes = set(
            [
                axis
                for axis in self.domain_axes(todict=True)
                if axis not in axes or self.domain_axis(axis).size == 1
            ]
        )

        # ------------------------------------------------------------
        # Copy constructs to the output field
        # ------------------------------------------------------------
        if other_axes:
            for key, c in self.constructs.filter_by_axis(
                *other_axes, axis_mode="subset", todict=True
            ).items():
                c_axes = self.get_data_axes(key)
                out.set_construct(c, axes=c_axes, key=key)

        # ------------------------------------------------------------
        # Copy coordinate reference constructs to the output field
        # ------------------------------------------------------------
        out_coordinates = out.coordinates(todict=True)
        out_domain_ancillaries = out.domain_ancillaries(todict=True)

        for cr_key, ref in self.coordinate_references(todict=True).items():
            ref = ref.copy()

            for c_key in ref.coordinates():
                if c_key not in out_coordinates:
                    ref.del_coordinate(c_key)

            for (
                term,
                da_key,
            ) in ref.coordinate_conversion.domain_ancillaries().items():
                if da_key not in out_domain_ancillaries:
                    ref.coordinate_conversion.set_domain_ancillary(term, None)

            out.set_construct(ref, key=cr_key, copy=False)

        # ------------------------------------------------------------
        # Create a dimension coordinate for the percentile ranks
        # ------------------------------------------------------------
        dim = DimensionCoordinate()
        data = Data(ranks).squeeze()
        data.override_units(Units(), inplace=True)
        if not data.shape:
            data.insert_dimension(inplace=True)
        dim.set_data(data, copy=False)

        if out.ndim == self.ndim:
            axis = out.set_construct(self._DomainAxis(1))
        else:
            axis = out_data_axes[0]

        axes = sorted(axes)
        if len(axes) == 1:
            dim.long_name = (
                "Percentile ranks for "
                + self.constructs.domain_axis_identity(axes[0])
                + " dimensions"
            )
        else:
            dim.long_name = (
                "Percentile ranks for "
                + ", ".join(map(self.constructs.domain_axis_identity, axes))
                + " dimensions"
            )

        out.set_construct(dim, axes=axis, copy=False)

        if squeeze:
            out.squeeze(inplace=True)

        return out

    @_inplace_enabled(default=False)
    def flatten(self, axes=None, return_axis=False, inplace=False):
        """Flatten axes of the field.

        Any subset of the domain axes may be flattened.

        The shape of the data may change, but the size will not.

        Metadata constructs whose data spans the flattened axes will
        either themselves be flattened, or else removed.

        Cell method constructs that apply to the flattened axes will
        be removed or, if possible, have their axis specifications
        changed to standard names.

        The flattening is executed in row-major (C-style) order. For
        example, the array ``[[1, 2], [3, 4]]`` would be flattened
        across both dimensions to ``[1 2 3 4]``.

        .. versionadded:: 3.0.2

        .. seealso:: `compress`, `insert_dimension`, `flip`, `swapaxes`,
                     `transpose`

        :Parameters:

            axes: (sequence of) `str` or `int`, optional
                Select the domain axes to be flattened, defined by the
                domain axes that would be selected by passing each
                given axis description to a call of the field
                construct's `domain_axis` method. For example, for a
                value of ``'X'``, the domain axis construct returned
                by ``f.domain_axis('X')`` is selected.

                If no axes are provided then all axes spanned by the
                field construct's data are flattened.

                No axes are flattened if *axes* is an empty sequence.

            return_axis: `bool`, optional
                If True then also return either the key of the
                flattened domain axis construct; or `None` if the axes
                to be flattened do not span the data.

            {{inplace: `bool`, optional}}

        :Returns:

            `Field` or `None`, [`str` or `None`]
                The new, flattened field construct, or `None` if the
                operation was in-place.

                If *return_axis* is True then also return either the
                key of the flattened domain axis construct; or `None`
                if the axes to be flattened do not span the data.

        **Examples**

        See `cf.Data.flatten` for more examples of how the data are
        flattened.

        >>> f.shape
        (1, 2, 3, 4)
        >>> f.flatten().shape
        (24,)
        >>> f.flatten([]).shape
        (1, 2, 3, 4)
        >>> f.flatten([1, 3]).shape
        (1, 8, 3)
        >>> f.flatten([0, -1], inplace=True)
        >>> f.shape
        (4, 2, 3)

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
                        : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., kappa]
        Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m
        >>> print(t.flatten())
        Field: air_temperature (ncvar%ta)
        ---------------------------------
        Data            : air_temperature(key%domainaxis4(90)) K
        Cell methods    : grid_latitude: grid_longitude: mean where land (interval: 0.1 degrees) time(1): maximum
        Field ancils    : air_temperature standard_error(key%domainaxis4(90)) = [0.76, ..., 0.32] K
        Dimension coords: time(1) = [2019-01-01 00:00:00]
        Auxiliary coords: latitude(key%domainaxis4(90)) = [53.941, ..., 50.225] degrees_N
                        : longitude(key%domainaxis4(90)) = [2.004, ..., 8.156] degrees_E
        Cell measures   : measure:area(key%domainaxis4(90)) = [2391.9657, ..., 2392.6009] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : surface_altitude(key%domainaxis4(90)) = [0.0, ..., 270.0] m
        >>> print(t.flatten(['grid_latitude', 'grid_longitude']))
        Field: air_temperature (ncvar%ta)
        ---------------------------------
        Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), key%domainaxis4(90)) K
        Cell methods    : grid_latitude: grid_longitude: mean where land (interval: 0.1 degrees) time(1): maximum
        Field ancils    : air_temperature standard_error(key%domainaxis4(90)) = [0.76, ..., 0.32] K
        Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
                        : time(1) = [2019-01-01 00:00:00]
        Auxiliary coords: latitude(key%domainaxis4(90)) = [53.941, ..., 50.225] degrees_N
                        : longitude(key%domainaxis4(90)) = [2.004, ..., 8.156] degrees_E
        Cell measures   : measure:area(key%domainaxis4(90)) = [2391.9657, ..., 2392.6009] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(key%domainaxis4(90)) = [0.0, ..., 270.0] m

        >>> t.domain_axes.keys()
        >>> dict_keys(
        ...     ['domainaxis0', 'domainaxis1', 'domainaxis2', 'domainaxis3'])
        >>> t.flatten(return_axis=True)
        (<CF Field: air_temperature(key%domainaxis4(90)) K>,
         'domainaxis4')
        >>> t.flatten('grid_longitude', return_axis=True)
        (<CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>,
         'domainaxis2')
        >>> t.flatten('time', return_axis=True)
        (<CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>,
         None)

        """
        f = _inplace_enabled_define_and_cleanup(self)

        data_axes = self.get_data_axes()

        if axes is None:
            axes = data_axes
        else:
            if isinstance(axes, (str, int)):
                axes = (axes,)

            axes = [self.domain_axis(x, key=True) for x in axes]
            axes = set(axes).intersection(data_axes)

        # Note that it is important to sort the iaxes, as we rely on
        # the first iaxis in the list being the left-most flattened
        # axis
        iaxes = sorted([data_axes.index(axis) for axis in axes])

        if not len(iaxes):
            if inplace:
                f = None
            if return_axis:
                return f, None
            return f

        if len(iaxes) == 1:
            if inplace:
                f = None
            if return_axis:
                return f, tuple(axes)[0]
            return f

        #        # Make sure that the metadata constructs have the same
        #        # relative axis order as the data (pre-flattening)
        #        f.transpose(f.get_data_axes(), constructs=True, inplace=True)

        # Create the new data axes
        shape = f.shape
        new_data_axes = [
            axis for i, axis in enumerate(data_axes) if i not in iaxes
        ]
        new_axis_size = np.prod([shape[i] for i in iaxes])
        new_axis = f.set_construct(self._DomainAxis(new_axis_size))
        new_data_axes.insert(iaxes[0], new_axis)

        # Flatten the field's data
        super(Field, f).flatten(iaxes, inplace=True)

        # Set the new data axes
        f.set_data_axes(new_data_axes)

        # Modify or remove cell methods that span the flatten axes
        for key, cm in f.cell_methods(todict=True).items():
            cm_axes = set(cm.get_axes(()))
            if not cm_axes or cm_axes.isdisjoint(axes):
                continue

            if cm_axes.difference(axes):
                f.del_construct(key)
                continue

            if cm_axes.issubset(axes):
                cm_axes = list(cm_axes)
                set_axes = True
                for i, a in enumerate(cm_axes):
                    sn = None
                    for c in f.coordinates(
                        filter_by_axis=(a,), axis_mode="exact", todict=True
                    ).values():
                        sn = c.get_property("standard_name", None)
                        if sn is not None:
                            break

                    #                    for ctype in (
                    #                        "dimension_coordinate",
                    #                        "auxiliary_coordinate",
                    #                    ):
                    #                        for c in (
                    #                            f.constructs.filter_by_type(ctype, view=True)
                    #                            .filter_by_axis(a, mode="exact", view=True)
                    #                            .values()
                    #                        ):
                    #                            sn = c.get_property("standard_name", None)
                    #                            if sn is not None:
                    #                                break
                    #
                    #                        if sn is not None:
                    #                            break

                    if sn is None:
                        f.del_construct(key)
                        set_axes = False
                        break
                    else:
                        cm_axes[i] = sn

                if set_axes:
                    cm.set_axes(cm_axes)

        # Flatten the constructs that span all of the flattened axes,
        # or all of the flattened axes all bar some which have size 1.
        #        d = dict(f.constructs.filter_by_axis('exact', *axes))
        #        axes2 = [axis for axis in axes
        #                 if f.domain_axes[axis].get_size() > 1]
        #        if axes2 != axes:
        #            d.update(f.constructs.filter_by_axis(
        #                'subset', *axes).filter_by_axis('and', *axes2))

        # Flatten the constructs that span all of the flattened axes,
        # and no others.
        for key, c in f.constructs.filter_by_axis(
            *axes, axis_mode="and", todict=True
        ).items():
            c_axes = f.get_data_axes(key)
            c_iaxes = sorted(
                [c_axes.index(axis) for axis in axes if axis in c_axes]
            )
            c.flatten(c_iaxes, inplace=True)
            new_data_axes = [
                axis for i, axis in enumerate(c_axes) if i not in c_iaxes
            ]
            new_data_axes.insert(c_iaxes[0], new_axis)
            f.set_data_axes(new_data_axes, key=key)

        # Remove constructs that span some, but not all, of the
        # flattened axes
        for key in f.constructs.filter_by_axis(
            *axes, axis_mode="or", todict=True
        ):
            f.del_construct(key)

        # Remove the domain axis constructs for the flattened axes
        for key in axes:
            f.del_construct(key)

        if return_axis:
            return f, new_axis

        return f

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def roll(self, axis, shift, inplace=False, i=False, **kwargs):
        """Roll the field along a cyclic axis.

        A unique axis is selected with the axes and kwargs parameters.

        .. versionadded:: 1.0

        .. seealso:: `anchor`, `axis`, `cyclic`, `iscyclic`, `period`

        :Parameters:

            axis:
                The cyclic axis to be rolled, defined by that which
                would be selected by passing the given axis
                description to a call of the field construct's
                `domain_axis` method. For example, for a value of
                ``'X'``, the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

            shift: `int`
                The number of places by which the selected cyclic axis is
                to be rolled.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field`
                The rolled field.

        **Examples**

        Roll the data of the "X" axis one elements to the right:

        >>> f.roll('X', 1)

        Roll the data of the "X" axis three elements to the left:

        >>> f.roll('X', -3)

        """
        # TODODASK: Consider allowing multiple roll axes, since Data
        #           now supports them.

        axis = self.domain_axis(
            axis,
            key=True,
            default=ValueError(
                f"Can't roll: Bad axis specification: {axis!r}"
            ),
        )

        f = _inplace_enabled_define_and_cleanup(self)

        axis = f._parse_axes(axis)

        # Roll the metadata constructs in-place
        shift = f._roll_constructs(axis, shift)

        iaxes = self._axis_positions(axis, parse=False)
        if iaxes:
            # TODODASK: Remove these two lines if multiaxis rolls are
            #           allowed

            iaxis = iaxes[0]
            shift = shift[0]

            super(Field, f).roll(iaxis, shift, inplace=True)

        return f

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_manage_log_level_via_verbosity
    def where(
        self,
        condition,
        x=None,
        y=None,
        inplace=False,
        construct=None,
        i=False,
        verbose=None,
        item=None,
        **item_options,
    ):
        """Assign to data elements depending on a condition.

        The elements to be changed are identified by a
        condition. Different values can be assigned according to where
        the condition is True (assignment from the *x* parameter) or
        False (assignment from the *y* parameter).

        **Missing data**

        Array elements may be set to missing values if either *x* or
        *y* are the `cf.masked` constant, or by assignment from any
        missing data elements in *x* or *y*.

        If the data mask is hard (see the `hardmask` attribute) then
        missing data values in the array will not be overwritten,
        regardless of the content of *x* and *y*.

        If the *condition* contains missing data then the
        corresponding elements in the array will not be assigned to,
        regardless of the contents of *x* and *y*.

        **Broadcasting**

        The array and the *condition*, *x* and *y* parameters must all
        be broadcastable across the original array, such that the size
        of the result is identical to the original size of the
        array. Leading size 1 dimensions of these parameters are
        ignored, thereby also ensuring that the shape of the result is
        identical to the original shape of the array.

        If *condition* is a `Query` object then for the purposes of
        broadcasting, the condition is considered to be that which is
        produced by applying the query to the field's array.

        **Performance**

        If any of the shapes of the *condition*, *x*, or *y*
        parameters, or the field, is unknown, then there is a
        possibility that an unknown shape will need to be calculated
        immediately by executing all delayed operations on that
        object.

        .. seealso:: `hardmask`, `indices`, `mask`, `subspace`,
                     `__setitem__`, `cf.masked`

        :Parameters:

            condition: array_like, `Field` or `Query`
                The condition which determines how to assign values to
                the field's data.

                Assignment from the *x* and *y* parameters will be
                done where elements of the condition evaluate to
                `True` and `False` respectively.

                If *condition* is a `Query` object then this implies a
                condition defined by applying the query to the data.

                *Parameter example:*
                  ``f.where(f.data<0, x=-999)`` will set all data
                  values that are less than zero to -999.

                *Parameter example:*
                  ``f.where(True, x=-999)`` will set all data values
                  to -999. This is equivalent to ``f[...] = -999``.

                *Parameter example:*
                  ``f.where(False, y=-999)`` will set all data values
                  to -999. This is equivalent to ``f[...] = -999``.

                *Parameter example:*
                  If field construct ``f`` has shape ``(5, 3)`` then
                  ``f.where([True, False, True], x=-999,
                  y=cf.masked)`` will set data values in columns 0 and
                  2 to -999, and data values in column 1 to missing
                  data. This works because the condition has shape
                  ``(3,)`` which broadcasts to the field construct's
                  shape.

                *Parameter example:*
                  ``f.where(cf.lt(0), x=-999)`` will set all data
                  values that are less than zero to -999. This is
                  equivalent to ``f.where(f.data<0, x=-999)``.

                If *condition* is a `Field` then it is first
                transformed so that it is broadcastable to the data
                being assigned to. This is done by using the metadata
                constructs of the two field constructs to create a
                mapping of physically identical dimensions between the
                fields, and then manipulating the dimensions of other
                field construct's data to ensure that they are
                broadcastable. If either of the field constructs does
                not have sufficient metadata to create such a mapping
                then an exception will be raised. In this case, any
                manipulation of the dimensions must be done manually,
                and the `Data` instance of *construct* (rather than
                the field construct itself) may be used for the
                condition.

                *Parameter example:*
                  If field construct ``f`` has shape ``(5, 3)`` and
                  ``g = f.transpose() < 0`` then ``f.where(g,
                  x=-999)`` will set all data values that are less
                  than zero to -999, provided there are sufficient
                  metadata for the data dimensions to be
                  mapped. However, ``f.where(g.data, x=-999)`` will
                  always fail in this example, because the shape of
                  the condition is ``(3, 5)``, which does not
                  broadcast to the shape of the ``f``.

            x, y:  array-like or `Field` or `None`
                Specify the assignment values. Where the condition is
                True assign to the data from *x*, and where the
                condition is False assign to the data from *y*.

                If *x* is `None` (the default) then no assignment is
                carried out where the condition is True.

                If *y* is `None` (the default) then no assignment is
                carried out where the condition is False.

                *Parameter example:*
                  ``d.where(condition)``, for any ``condition``,
                  returns data with identical data values.

                *Parameter example:*
                  ``d.where(cf.lt(0), x=-d, y=cf.masked)`` will change
                  the sign of all negative data values, and set all
                  other data values to missing data.

                *Parameter example:*
                  ``d.where(cf.lt(0), x=-d)`` will change the sign of
                  all negative data values, and leave all other data
                  values unchanged. This is equivalent to, but faster
                  than, ``d.where(cf.lt(0), x=-d, y=d)``

                *Parameter example:*
                  ``f.where(condition)``, for any ``condition``,
                  returns a field construct with identical data
                  values.

                *Parameter example:*
                  ``f.where(cf.lt(0), x=-f.data, y=cf.masked)`` will
                  change the sign of all negative data values, and set
                  all other data values to missing data.

                If *x* or *y* is a `Field` then it is first
                transformed so that its data is broadcastable to the
                data being assigned to. This is done by using the
                metadata constructs of the two field constructs to
                create a mapping of physically identical dimensions
                between the fields, and then manipulating the
                dimensions of other field construct's data to ensure
                that they are broadcastable. If either of the field
                constructs does not have sufficient metadata to create
                such a mapping then an exception will be raised. In
                this case, any manipulation of the dimensions must be
                done manually, and the `Data` instance of *x* or *y*
                (rather than the field construct itself) may be used
                for the condition.

                *Parameter example:*
                  If field construct ``f`` has shape ``(5, 3)`` and
                  ``g = f.transpose() * 10`` then ``f.where(cf.lt(0),
                  x=g)`` will set all data values that are less than
                  zero to the equivalent elements of field construct
                  ``g``, provided there are sufficient metadata for
                  the data dimensions to be mapped. However,
                  ``f.where(cf.lt(0), x=g.data)`` will always fail in
                  this example, because the shape of the condition is
                  ``(3, 5)``, which does not broadcast to the shape of
                  the ``f``.

            construct: `str`, optional
                Define the condition by applying the *construct*
                parameter to the given metadata construct's data,
                rather than the data of the field construct. Must be

                * The identity or key of a metadata coordinate
                  construct that has data.

            ..

                The *construct* parameter selects the metadata
                construct that is returned by this call of the field
                construct's `construct` method:
                ``f.construct(construct)``. See `cf.Field.construct`
                for details.

                *Parameter example:*
                  ``f.where(cf.wi(-30, 30), x=cf.masked,
                  construct='latitude')`` will set all data values
                  within 30 degrees of the equator to missing data.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            item: deprecated at version 3.0.0
                Use the *construct* parameter instead.

            item_options: deprecated at version 3.0.0

        :Returns:

            `Field` or `None`
                A new field construct with an updated data array, or
                `None` if the operation was in-place.

        **Examples**

        Set data array values to 15 everywhere:

        >>> f.where(True, 15)

        This example could also be done with subspace assignment:

        >>> f[...] = 15

        Set all negative data array values to zero and leave all other
        elements unchanged:

        >>> g = f.where(f<0, 0)

        Multiply all positive data array elements by -1 and set other data
        array elements to 3.14:

        >>> g = f.where(f>0, -f, 3.14)

        Set all values less than 280 and greater than 290 to missing data:

        >>> g = f.where((f < 280) | (f > 290), cf.masked)

        This example could also be done with a `Query` object:

        >>> g = f.where(cf.wo(280, 290), cf.masked)

        or equivalently:

        >>> g = f.where(f==cf.wo(280, 290), cf.masked)

        Set data array elements in the northern hemisphere to missing data
        in-place:

        >>> condition = f.domain_mask(latitude=cf.ge(0))
        >>> f.where(condition, cf.masked, inplace=True)

        Missing data can only be changed if the mask is "soft":

        >>> f[0] = cf.masked
        >>> g = f.where(True, 99)
        >>> print(g[0],array)
        [--]
        >>> f.hardmask = False
        >>> g = f.where(True, 99)
        >>> print(g[0],array)
        [99]

        This in-place example could also be done with subspace assignment
        by indices:

        >>> northern_hemisphere = f.indices(latitude=cf.ge(0))
        >>> f.subspace[northern_hemisphere] = cf.masked

        Set a polar rows to their zonal-mean values:

        >>> condition = f.domain_mask(latitude=cf.set([-90, 90]))
        >>> g = f.where(condition, f.collapse('longitude: mean'))

        """
        if item is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "where",
                {"item": item},
                "Use keyword 'construct' instead.",
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        if item_options:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "where",
                {"item_options": item_options},
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        if x is None and y is None:
            if inplace:
                return
            return self.copy()

        self_class = self.__class__

        if isinstance(condition, self_class):
            # --------------------------------------------------------
            # Condition is another field construct
            # --------------------------------------------------------
            condition = self._conform_for_assignment(condition)

        elif construct is not None:
            if not isinstance(condition, Query):
                raise ValueError(
                    "A condition on a metadata construct must be a Query "
                    "object"
                )

            # Apply the Query to a metadata construct of the field,
            # making sure that the construct's data is broadcastable
            # to the field's data.
            g = self.transpose(self.get_data_axes(), constructs=True)

            key = g.construct_key(
                construct,
                default=ValueError(
                    f"Can't identify unique {construct!r} construct"
                ),
            )
            construct = g.constructs[key]

            construct_data_axes = g.get_data_axes(key, default=None)
            if construct_data_axes is None:
                raise ValueError(
                    f"Can't identify construct data axes for {construct!r}."
                )

            data_axes = g.get_data_axes()

            construct_data = construct.get_data(None, _fill_value=False)
            if construct_data is None:
                raise ValueError(f"{construct!r} has no data")

            if construct_data_axes != data_axes:
                s = [
                    i
                    for i, axis in enumerate(construct_data_axes)
                    if axis not in data_axes
                ]
                if s:
                    construct_data.squeeze(s, inplace=True)
                    construct_data_axes = [
                        axis
                        for axis in construct_data_axes
                        if axis not in data_axes
                    ]

                for i, axis in enumerate(data_axes):
                    if axis not in construct_data_axes:
                        construct_data.insert_dimension(i, inplace=True)

            condition = condition.evaluate(construct_data)

        if x is not None and isinstance(x, self_class):
            x = self._conform_for_assignment(x)

        if y is not None and isinstance(y, self_class):
            y = self._conform_for_assignment(y)

        return super().where(condition, x, y, inplace=inplace, verbose=verbose)

    @property
    def subspace(self):
        """Create a subspace of the field construct.

        Creation of a new field construct which spans a subspace of the
        domain of an existing field construct is achieved either by
        identifying indices based on the metadata constructs (subspacing
        by metadata) or by indexing the field construct directly
        (subspacing by index).

        The subspacing operation, in either case, also subspaces any
        metadata constructs of the field construct (e.g. coordinate
        metadata constructs) which span any of the domain axis constructs
        that are affected. The new field construct is created with the
        same properties as the original field construct.

        **Subspacing by metadata**

        Subspacing by metadata, signified by the use of round brackets,
        selects metadata constructs and specifies conditions on their
        data. Indices for subspacing are then automatically inferred from
        where the conditions are met.

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

        **Subspacing by index**

        Subspacing by indexing, signified by the use of square brackets,
        uses rules that are very similar to the numpy indexing rules, the
        only differences being:

        * An integer index i specified for a dimension reduces the size of
          this dimension to unity, taking just the i-th element, but keeps
          the dimension itself, so that the rank of the array is not
          reduced.

        * When two or more dimensions’ indices are sequences of integers
          then these indices work independently along each dimension
          (similar to the way vector subscripts work in Fortran). This is
          the same indexing behaviour as on a Variable object of the
          netCDF4 package.

        * For a dimension that is cyclic, a range of indices specified by
          a `slice` that spans the edges of the data (such as ``-2:3`` or
          ``3:-2:-1``) is assumed to "wrap" around, rather then producing
          a null result.

        **Halos**

        {{subspace halos}}

        For instance, ``f.subspace(X=slice(10, 20))`` will give
        identical results to each of ``f.subspace(0, X=slice(10,
        20))``, ``f.subspace(1, X=slice(11, 19))``, ``f.subspace(2,
        X=slice(12, 18))``, etc.

        .. versionadded:: 1.0

        .. seealso:: `indices`, `where`, `__getitem__`,
                     `__setitem__`, `cf.Domain.subspace`

        :Parameters:

            {{config: optional}}

                {{subspace valid modes Field}}

                In addition, an extra positional argument of ``'test'``
                is allowed. When provided, the subspace is not
                returned, instead `True` or `False` is returned
                depending on whether or not it is possible for the
                requested subspace to be created.

            keyword parameters: optional
                A keyword name is an identity of a metadata construct, and
                the keyword value provides a condition for inferring
                indices that apply to the dimension (or dimensions)
                spanned by the metadata construct's data. Indices are
                created that select every location for which the metadata
                construct's data satisfies the condition.

        :Returns:

            `Field` or `bool`
                An independent field construct containing the subspace of
                the original field. If the ``'test'`` positional argument
                has been set then return `True` or `False` depending on
                whether or not it is possible to create specified
                subspace.

        **Examples**

        There are further worked examples
        :ref:`in the tutorial <Subspacing-by-metadata>`.

        >>> g = f.subspace(X=112.5)
        >>> g = f.subspace(X=112.5, latitude=cf.gt(-60))
        >>> g = f.subspace(latitude=cf.eq(-45) | cf.ge(20))
        >>> g = f.subspace(X=[1, 2, 4], Y=slice(None, None, -1))
        >>> g = f.subspace(X=cf.wi(-100, 200))
        >>> g = f.subspace(X=slice(-2, 4))
        >>> g = f.subspace(Y=[True, False, True, True, False])
        >>> g = f.subspace(T=410.5)
        >>> g = f.subspace(T=cf.dt('1960-04-16'))
        >>> g = f.subspace(T=cf.wi(cf.dt('1962-11-01'),
        ...                        cf.dt('1967-03-17 07:30')))
        >>> g = f.subspace('compress', X=[1, 2, 4, 6])
        >>> g = f.subspace('envelope', X=[1, 2, 4, 6])
        >>> g = f.subspace('full', X=[1, 2, 4, 6])
        >>> g = f.subspace(latitude=cf.wi(51, 53))

        >>> g = f.subspace[::-1, 0]
        >>> g = f.subspace[:, :, 1]
        >>> g = f.subspace[:, 0]
        >>> g = f.subspace[..., 6:3:-1, 3:6]
        >>> g = f.subspace[0, [2, 3, 9], [4, 8]]
        >>> g = t.subspace[0, :, -2]
        >>> g = f.subspace[0, [2, 3, 9], [4, 8]]
        >>> g = f.subspace[:, -2:3]
        >>> g = f.subspace[:, 3:-2:-1]
        >>> g = f.subspace[..., [True, False, True, True, False]]

        """
        return SubspaceField(self)

    def add_file_location(
        self,
        location,
        constructs=True,
    ):
        """Add a new file location in-place.

        All data definitions that reference files are additionally
        referenced from the given location.

        .. versionadded:: 3.15.0

        .. seealso:: `del_file_location`, `file_locations`

        :Parameters:

            location: `str`
                The new location.

            constructs: `bool`, optional
                If True (the default) then metadata constructs also
                have the new file location added to them.

        :Returns:

            `str`
                The new location as an absolute path with no trailing
                path name component separator.

        **Examples**

        >>> f.add_file_location('/data/model/')
        '/data/model'

        """
        location = super().add_file_location(location)
        if constructs:
            for c in self.constructs.filter_by_data(todict=True).values():
                c.add_file_location(location)

        return location

    def section(self, axes=None, stop=None, min_step=1, **kwargs):
        """Return a FieldList of m dimensional sections of a Field of n
        dimensions, where M <= N.

        :Parameters:

            axes: optional
                A query for the m axes that define the sections of the
                Field as accepted by the Field object's axes method. The
                keyword arguments are also passed to this method. See
                TODO cf.Field.axes for details. If an axis is returned that is
                not a data axis it is ignored, since it is assumed to be a
                dimension coordinate of size 1.

            stop: `int`, optional
                Stop after taking this number of sections and return. If
                *stop* is `None` all sections are taken.

        :Returns:

            `FieldList`
                The sections of the field construct.

        **Examples**

        Section a field into 2-d longitude/time slices, checking the
        units:

        >>> f.section({None: 'longitude', units: 'radians'},
        ...           {None: 'time',
        ...            'units': 'days since 2006-01-01 00:00:00'})

        Section a field into 2-d longitude/latitude slices, requiring
        exact names:

        >>> f.section(['latitude', 'longitude'], exact=True)

        Section a field into 2-d longitude/latitude slices, showing
        the results:

        >>> f
        <CF Field: eastward_wind(model_level_number(6), latitude(145), longitude(192)) m s-1>
        >>> f.section(('X', 'Y'))
        [<CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>,
         <CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>,
         <CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>,
         <CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>,
         <CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>,
         <CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>]

        """
        # TODODASK: This still need some attention, keyword checking,
        #           testing, docs, etc., but has been partially
        #           already updated due to changes already happening
        #           in `cf.functions._section` that might be
        #           overlooked/obscured later. See the daskification
        #           of `cf.functions._section` and `cf.Data.section`
        #           for more details.

        axes = [self.domain_axis(axis, key=True) for axis in axes]
        axis_indices = []
        for key in axes:
            try:
                axis_indices.append(self.get_data_axes().index(key))
            except ValueError:
                pass

        axes = axis_indices

        return FieldList(
            tuple(_section(self, axes, min_step=min_step).values())
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    @_manage_log_level_via_verbosity
    def regrids(
        self,
        dst,
        method=None,
        src_cyclic=None,
        dst_cyclic=None,
        use_src_mask=True,
        use_dst_mask=False,
        fracfield=False,
        src_axes=None,
        dst_axes=None,
        axis_order=None,
        ignore_degenerate=True,
        return_operator=False,
        check_coordinates=False,
        min_weight=None,
        weights_file=None,
        src_z=None,
        dst_z=None,
        z=None,
        ln_z=None,
        verbose=None,
        return_esmpy_regrid_operator=False,
        inplace=False,
        i=False,
        _compute_field_mass=None,
    ):
        """Regrid the field to a new latitude and longitude grid.

        {{regridding overview}}

        The 2-d or 3-d regridding takes place on a sphere, with the
        grid being defined by latitude and longitude spherical polar
        coordinates, and any available vertical coordinates. In the
        3-d case, the regridding may be done assuming linear or log
        linear weights in the vertical.

        **Latitude and longitude coordinates**

        The source and destination grids of the regridding must both
        be defined by latitude and longitude coordinates, which may be
        1-d dimension coordinates or 2-d auxiliary coordinates. These
        are automatically detected from the field being regridded and
        the specification of the destination grid given by the *dst*
        parameter.

        When a grid is defined by 2-d latitude and longitude
        coordinates, it is necessary for their X and Y dimensions to
        be defined. This is either automatically inferred from the
        existence of 1-d dimension coordinates, or else must be
        specified with *src_axes* or *dst_axes* parameters.

        **Curvilinear Grids**

        Grids in projection coordinate systems can be regridded as
        long as two dimensional latitude and longitude coordinates are
        present.

        **Tripolar Grids**

        Connections across the bipole fold are not currently
        supported, but are not necessary in some cases, for example if
        the points on either side are together without a gap (as is
        the case for NEMO model outputs).

        **UGRID meshes**

        Data defined on UGRID face or node cells may be regridded to
        any other latitude-longitude grid, including other UGRID
        meshes and DSG feature types.

        **DSG feature types**

        Data on any latitude-longitude grid (including tripolar and
        UGRID meshes) may be regridded to any DSG feature type.

        **Cyclicity of the X axis**

        The cyclicity of the X (longitude) axes of the source and
        destination grids (i.e. whether or not the first and last
        cells of the axis are adjacent) are taken into account. By
        default, the cyclicity is inferred from the grids' defining
        coordinates, but may be also be provided with the *src_cyclic*
        and *dst_cyclic* parameters.

        {{regrid Masked cells}}

        {{regrid Implementation}}

        {{regrid Logging}}

        .. versionadded:: 1.0.4

        .. seealso:: `regridc`

        :Parameters:

            dst: `Field`, `Domain`, `RegridOperator` or sequence of `Coordinate`
                The definition of the destination grid on which to
                regrid the field's data. One of:

                * `Field`: The grid is defined by the latitude and
                  longitude coordinates of the field construct's
                  domain.

                * `Domain`: The grid is defined by the latitude and
                  longitude coordinates of the domain construct.

                * Sequence of `Coordinate`: The grid is defined by two
                  1-d dimension coordinate constructs, or two 2-d
                  auxiliary coordinate constructs, that define the
                  spherical latitude and longitude coordinates (in any
                  order) of the destination grid.

                  In the 2-d case, both coordinate constructs must
                  have their axes in the same order, which must be
                  specified by the *dst_axes* parameter.

                {{regrid RegridOperator}}

            {{method: `str` or `None`, optional}}

            src_cyclic: `None` or `bool`, optional
                Specifies whether or not the source grid longitude
                axis is cyclic (i.e. the first and last cells of the
                axis are adjacent). If `None` (the default) then the
                cyclicity will be inferred from the source grid
                coordinates, defaulting to `False` if it can not be
                determined.

            dst_cyclic: `None` or `bool`, optional
                Specifies whether or not the destination grid
                longitude axis is cyclic (i.e. the first and last
                cells of the axis are adjacent). If `None` (the
                default) then the cyclicity will be inferred from the
                destination grid coordinates, defaulting to `False` if
                it can not be determined.

                Ignored if *dst* is a `RegridOperator`.

            {{use_src_mask: `bool`, optional}}

            {{use_dst_mask: `bool`, optional}}

            src_axes: `dict`, optional
                When the source grid's X and Y dimensions can not be
                inferred from the existence of 1-d dimension
                coordinates, then they must be identified with the
                *src_axes* dictionary, with keys ``'X'`` and ``'Y'``.

                The dictionary values identify a unique domain axis by
                passing the given axis description to a call of the
                field construct's `domain_axis` method. For example,
                for a value of ``'ncdim%x'``, the domain axis
                construct returned by ``f.domain_axis('ncdim%x')`` is
                selected.

                *Parameter example:*
                  ``{'X': 'ncdim%x', 'Y': 'ncdim%y'}``

                *Parameter example:*
                  ``{'X': 1, 'Y': 0}``

            dst_axes: `dict`, optional
                When the destination grid's X and Y dimensions can not
                be inferred from the existence of 1-d dimension
                coordinates, then they must be identified with the
                *dst_axes* dictionary, with keys ``'X'`` and ``'Y'``.

                If *dst* is a `Field` or `Domain`, then the dictionary
                values identify a unique domain axis by passing the
                given axis description to a call of the destination
                field or domain construct's `domain_axis` method. For
                example, for a value of ``'ncdim%x'``, the domain axis
                construct returned by ``f.domain_axis('ncdim%x')`` is
                selected.

                If *dst* is a sequence of `Coordinate`, then the
                dictionary values identify a unique domain axis by its
                position in the 2-d coordinates' data arrays, i.e. the
                dictionary values must be ``0`` and ``1``:

                Ignored if *dst* is a `RegridOperator`.

                *Parameter example:*
                  ``{'X': 'ncdim%x', 'Y': 'ncdim%y'}``

                *Parameter example:*
                  ``{'X': 1, 'Y': 0}``

            {{ignore_degenerate: `bool`, optional}}

            {{return_operator: `bool`, optional}}

                .. versionadded:: 3.10.0

            {{check_coordinates: `bool`, optional}}

                .. versionadded:: 3.14.0

            {{min_weight: float, optional}}

                .. versionadded:: 3.14.0

            {{weights_file: `str` or `None`, optional}}

                Ignored if *dst* is a `RegridOperator`.

                .. versionadded:: 3.15.2

            src_z: optional
                If `None`, the default, then the regridding is 2-d in
                the latitude-longitude plane.

                If not `None` then 3-d spherical regridding is enabled
                by identifying the source grid vertical coordinates
                from which to derive the vertical component of the
                regridding weights. The vertical coordinate construct
                may be 1-d or 3-d and is defined by the unique
                construct returned by ``f.coordinate(src_z)``

                Ignored if *dst* is a `RegridOperator`.

                .. versionadded:: 3.16.2

            dst_z: optional
                If `None`, the default, then the regridding is 2-d in
                the latitude-longitude plane.

                If not `None` then 3-d spherical regridding is enabled
                by identifying the destination grid vertical
                coordinates from which to derive the vertical
                component of the regridding weights. The vertical
                coordinate construct may be 1-d or 3-d.

                Ignored if *dst* is a `RegridOperator`.

                .. versionadded:: 3.16.2

            z: optional
                The *z* parameter is a convenience that may be used to
                replace both *src_z* and *dst_z* when they would
                contain identical values. If not `None` then 3-d
                spherical regridding is enabled. See *src_z* and
                *dst_z* for details.

                Ignored if *dst* is a `RegridOperator`.

                *Example:*
                  ``z='Z'`` is equivalent to ``src_z='Z', dst_z='Z'``.

                .. versionadded:: 3.16.2

            {{ln_z: `bool` or `None`, optional}}

                .. versionadded:: 3.16.2

            {{verbose: `int` or `str` or `None`, optional}}

                .. versionadded:: 3.16.0

            {{inplace: `bool`, optional}}

            {{return_esmpy_regrid_operator: `bool`, optional}}

                .. versionadded:: 3.16.2

            axis_order: sequence, optional
                Deprecated at version 3.14.0.

            fracfield: `bool`, optional
                Deprecated at version 3.14.0.

            _compute_field_mass: `dict`, optional
                Deprecated at version 3.14.0.

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Field` or `None` or `RegridOperator`
                The regridded field construct; or `None` if the
                operation was in-place; or the regridding operator if
                *return_operator* is True; or the `esmpy.Regrid` operator
                object if *return_esmpy_regrid_operator* is True.

        **Examples**

        >>> src, dst = cf.example_fields(1, 0)
        >>> print(src)
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
                        : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., kappa]
        Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m
        >>> print(dst)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> x = src.regrids(dst, method='linear')
        >>> print(x)
        Field: air_temperature (ncvar%ta)
        ---------------------------------
        Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), latitude(5), longitude(8)) K
        Cell methods    : latitude(5): longitude(8): mean where land (interval: 0.1 degrees) time(1): maximum
        Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        Coord references: standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(latitude(5), longitude(8)) = [[--, ..., --]] m

        >>> r = src.regrids(dst, method='linear', return_operator=True)
        >>> y = src.regrids(r)
        >>> y.equals(x)
        True

        """
        from .regrid import regrid

        return regrid(
            "spherical",
            self,
            dst,
            method=method,
            src_cyclic=src_cyclic,
            dst_cyclic=dst_cyclic,
            use_src_mask=use_src_mask,
            use_dst_mask=use_dst_mask,
            src_axes=src_axes,
            dst_axes=dst_axes,
            ignore_degenerate=ignore_degenerate,
            return_operator=return_operator,
            check_coordinates=check_coordinates,
            min_weight=min_weight,
            weights_file=weights_file,
            src_z=src_z,
            dst_z=dst_z,
            z=z,
            ln_z=ln_z,
            return_esmpy_regrid_operator=return_esmpy_regrid_operator,
            inplace=inplace,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def regridc(
        self,
        dst,
        axes=None,
        method=None,
        use_src_mask=True,
        use_dst_mask=False,
        fracfield=False,
        axis_order=None,
        ignore_degenerate=True,
        return_operator=False,
        check_coordinates=False,
        min_weight=None,
        weights_file=None,
        src_z=None,
        dst_z=None,
        z=None,
        ln_z=None,
        return_esmpy_regrid_operator=False,
        inplace=False,
        i=False,
        _compute_field_mass=None,
    ):
        """Regrid the field to a new Cartesian grid.

        {{regridding overview}}

        Between one and three axes may be simultaneously regridded in
        Cartesian space.

        **Coordinates**

        The source and destination grids of the regridding must both
        be defined by equivalent coordinates, which must be 1-d
        dimension coordinates. These are automatically detected from
        the field being regridded and the specification of the
        destination grid given by the *dst* parameter.

        **UGRID meshes**

        At present, Cartesian regridding is only available when
        neither the source nor destination grid is a UGRID mesh.

        {{regrid Masked cells}}

        {{regrid Implementation}}

        {{regrid Logging}}

        .. seealso:: `regrids`

        :Parameters:

            dst: `Field`, `Domain`, `RegridOperator` or sequence of `DimensionCoordinate`
                The definition of the destination grid on which to
                regrid the field's data. One of:

                * `Field`: The grid is defined by the coordinates of
                  the field construct's domain.

                * `Domain`: The grid is defined by the coordinates of
                  the domain construct.

                * Sequence of `DimensionCoordinate`: The grid is
                  defined by between one and three 1-d dimension
                  coordinate constructs that define the coordinates of
                  the destination grid. The order of the coordinate
                  constructs **must** match the order of source field
                  regridding axes defined by the *src_axes* or *axes*
                  parameter.

                {{regrid RegridOperator}}

            {{method: `str` or `None`, optional}}

            {{use_src_mask: `bool`, optional}}

            {{use_dst_mask: `bool`, optional}}

            src_axes: sequence, optional
                Define the source grid axes to be regridded. The
                sequence of between one and three values identify
                unique domain axes by passing each axis description to
                a call of the source field construct's `domain_axis`
                method. For example, for a value of ``'ncdim%x'``, the
                domain axis construct returned by
                ``f.domain_axis('ncdim%x')`` is selected.

                Must have the same number of values as the *dst_axes*
                parameter, if set, and the source and destination
                regridding axes must be specified in the same
                order. See the *axes* parameter.

                Ignored if *dst* is a `RegridOperator`.

                *Parameter example:*
                  ``['T']``

                *Parameter example:*
                  ``[1, 0]``

                .. versionadded:: 3.14.0

            dst_axes: `sequence`, optional
                When the destination grid is defined by a `Field` or
                `Domain`, define the destination grid axes to be
                regridded. The sequence of between one and three
                values identify unique domain axes by passing each
                axis description to a call of the destination field or
                domain construct's `domain_axis` method. For example,
                for a value of ``'ncdim%x'``, the domain axis
                construct returned by ``g.domain_axis('ncdim%x')`` is
                selected.

                Must have the same number of values as the *src_axes*
                parameter, if set, and the source and destination
                regridding axes must be specified in the same
                order. See the *axes* parameter.

                Ignored if *dst* is a `RegridOperator`.

                *Parameter example:*
                  ``['T']``

                *Parameter example:*
                  ``[1, 0]``

                .. versionadded:: 3.14.0

            axes: optional
                Define the axes to be regridded for the source grid
                and, if *dst* is a `Field` or `Domain`, the
                destination grid. The *axes* parameter is a
                convenience that may be used to replace *src_axes* and
                *dst_axes* when they would contain identical
                sequences. It may also be used in place of *src_axes*
                if *dst_axes* is not required.

            {{ignore_degenerate: `bool`, optional}}

            {{return_operator: `bool`, optional}}

                .. versionadded:: 3.10.0

            {{check_coordinates: `bool`, optional}}

                .. versionadded:: 3.14.0

            {{min_weight: float, optional}}

                .. versionadded:: 3.14.0

            {{weights_file: `str` or `None`, optional}}

                .. versionadded:: 3.15.2

            src_z: optional
                If not `None` then *src_z* specifies the identity of a
                vertical coordinate construct of the source grid. On
                its own this make no difference to the result, but it
                allows the setting of *ln_z* to True.

                Ignored if *dst* is a `RegridOperator`.

                .. versionadded:: 3.16.2

            dst_z: optional
                If not `None` then *dst_z* specifies the identity of a
                vertical coordinate construct of the destination
                grid. On its own this make no difference to the
                result, but it allows the setting of *ln_z* to True.

                Ignored if *dst* is a `RegridOperator`.

                .. versionadded:: 3.16.2

            z: optional
                The *z* parameter is a convenience that may be used to
                replace both *src_z* and *dst_z* when they would
                contain identical values.

                Ignored if *dst* is a `RegridOperator`.

                *Example:*
                  ``z='Z'`` is equivalent to ``src_z='Z', dst_z='Z'``.

                .. versionadded:: 3.16.2

            {{ln_z: `bool` or `None`, optional}}

                .. versionadded:: 3.16.2

            {{inplace: `bool`, optional}}

            {{return_esmpy_regrid_operator: `bool`, optional}}

                .. versionadded:: 3.16.2

            axis_order: sequence, optional
                Deprecated at version 3.14.0.

            fracfield: `bool`, optional
                Deprecated at version 3.14.0.

            _compute_field_mass: `dict`, optional
                Deprecated at version 3.14.0.

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Field` or `None` or `RegridOperator`
                The regridded field construct; or `None` if the
                operation was in-place; or the regridding operator if
                *return_operator* is True; or the `esmpy.Regrid` operator
                object if *return_esmpy_regrid_operator* is True.

        **Examples**

        >>> src, dst = cf.example_fields(1, 0)
        >>> print(src)
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
                        : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., kappa]
        Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m
        >>> print(dst)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> x = src.regridc(dst, method='linear', axes=['Y'])
        >>> print(x)
        Field: air_temperature (ncvar%ta)
        ---------------------------------
        Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), latitude(5), grid_longitude(9)) K
        Cell methods    : latitude(5): grid_longitude(9): mean where land (interval: 0.1 degrees) time(1): maximum
        Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : grid_longitude(9) = [-4.7, ..., -1.18] degrees
                        : time(1) = [2019-01-01 00:00:00]
        Coord references: standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(latitude(5), grid_longitude(9)) = [[--, ..., --]] m

        >>> r = src.regridc(dst, method='linear', axes=['Y'], return_operator=True)
        >>> y = src.regridc(r)
        >>> y.equals(x)
        True

        """
        from .regrid import regrid

        return regrid(
            "Cartesian",
            self,
            dst,
            method=method,
            src_cyclic=False,
            dst_cyclic=False,
            use_src_mask=use_src_mask,
            use_dst_mask=use_dst_mask,
            axes=axes,
            ignore_degenerate=ignore_degenerate,
            return_operator=return_operator,
            check_coordinates=check_coordinates,
            min_weight=min_weight,
            weights_file=weights_file,
            src_z=src_z,
            dst_z=dst_z,
            z=z,
            ln_z=ln_z,
            return_esmpy_regrid_operator=return_esmpy_regrid_operator,
            inplace=inplace,
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def derivative(
        self,
        axis,
        wrap=None,
        one_sided_at_boundary=False,
        inplace=False,
        i=False,
        cyclic=None,
    ):
        """Calculate the derivative along the specified axis.

        The derivative is calculated using centred finite differences
        apart from at the boundaries (see the *one_sided_at_boundary*
        parameter). If missing values are present then missing values
        will be returned at all points where a centred finite
        difference could not be calculated.

        :Parameters:

            axis:
                The axis, defined by that which would be selected by
                passing the given axis description to a call of the
                field construct's `domain_axis` method. For example,
                for a value of ``'X'``, the domain axis construct
                returned by ``f.domain_axis('X')`` is selected.

            wrap: `bool`, optional
                Whether the axis is cyclic or not. By default *wrap*
                is set to the result of this call to the field
                construct's `iscyclic` method:
                ``f.iscyclic(axis)``. If the axis is cyclic then
                centred differences at one boundary will always use
                values from the other boundary, regardless of the
                setting of *one_sided_at_boundary*.

            one_sided_at_boundary: `bool`, optional
                If True, and the field is not cyclic or *wrap* is
                True, then one-sided finite differences are calculated
                at the non-cyclic boundaries. By default missing
                values are set at non-cyclic boundaries.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Field` or `None`
                The derivative of the field along the specified axis,
                or `None` if the operation was in-place.

        **Examples**

        >>> f = cf.example_field(0)
        >>> print(f.dimension_coordinate('X').array)
        [ 22.5  67.5 112.5 157.5 202.5 247.5 292.5 337.5]
        >>> print(f.dimension_coordinate('X').cellsize.array)
        [45. 45. 45. 45. 45. 45. 45. 45.]
        >>> f[...] = [45, 90, 135, 180, 225, 270, 315, 360]
        >>> f.iscyclic('X')
        True
        >>> print(f.array)
        [[ 45.  90. 135. 180. 225. 270. 315. 360.]
         [ 45.  90. 135. 180. 225. 270. 315. 360.]
         [ 45.  90. 135. 180. 225. 270. 315. 360.]
         [ 45.  90. 135. 180. 225. 270. 315. 360.]
         [ 45.  90. 135. 180. 225. 270. 315. 360.]]
        >>> print(f.derivative('X').array)
        [[-3.  1.  1.  1.  1.  1.  1. -3.]
         [-3.  1.  1.  1.  1.  1.  1. -3.]
         [-3.  1.  1.  1.  1.  1.  1. -3.]
         [-3.  1.  1.  1.  1.  1.  1. -3.]
         [-3.  1.  1.  1.  1.  1.  1. -3.]]
        >>> print(f.derivative('X', wrap=False).array)
        [[-- 1.0 1.0 1.0 1.0 1.0 1.0 --]
         [-- 1.0 1.0 1.0 1.0 1.0 1.0 --]
         [-- 1.0 1.0 1.0 1.0 1.0 1.0 --]
         [-- 1.0 1.0 1.0 1.0 1.0 1.0 --]
         [-- 1.0 1.0 1.0 1.0 1.0 1.0 --]]
        >>> print(
        ...   f.derivative('X', wrap=False, one_sided_at_boundary=True).array
        ... )
        [[1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]]

        """
        if cyclic:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "derivative",
                {"cyclic": cyclic},
                "Use the 'wrap' keyword instead",
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        # Retrieve the axis
        axis_in = axis
        axis = self.domain_axis(axis, key=True, default=None)
        if axis is None:
            raise ValueError(f"Invalid axis specifier: {axis_in}")

        coord = self.dimension_coordinate(filter_by_axis=(axis,), default=None)
        if coord is None:
            raise ValueError(
                f"No dimension coordinates for axis defined by {axis_in}"
            )

        # Get the axis index
        axis_index = self.get_data_axes().index(axis)

        # Automatically detect the cyclicity of the axis if cyclic is
        # None
        cyclic = self.iscyclic(axis)
        if wrap is None:
            wrap = cyclic

        # Set the boundary conditions
        if wrap:
            mode = "wrap"
        elif one_sided_at_boundary:
            mode = "nearest"
        else:
            mode = "constant"

        f = _inplace_enabled_define_and_cleanup(self)

        # Find the differences of the data
        f.convolution_filter(
            [1, 0, -1], axis=axis, mode=mode, update_bounds=False, inplace=True
        )

        # Find the differences of the coordinates
        d = None
        if wrap and cyclic:
            period = coord.period()
            if period is None:
                raise ValueError(
                    "Can't calculate derivative when cyclic dimension "
                    f"coordinate {coord!r} has no period"
                )

            # Fix the boundary differences for cyclic periodic
            # coordinates. Need to extend the coordinates to include a
            # dummy value at each end, grabbed from the other end,
            # that maintains strict monotonicity.
            c_data = coord.data
            d2 = self._Data.empty((c_data.size + 2,), units=c_data.Units)
            if not coord.direction():
                period = -period

            d2[1:-1] = c_data
            d2[0] = c_data[-1] - period
            d2[-1] = c_data[0] + period
            c_data = d2
            d = d2.convolution_filter(
                window=[1, 0, -1], axis=0, mode="constant"
            )[1:-1]

        if d is None:
            d = coord.data.convolution_filter(
                window=[1, 0, -1], axis=0, mode=mode, cval=np.nan
            )

        # Reshape the coordinate differences so that they broadcast to
        # the data
        for _ in range(self.ndim - 1 - axis_index):
            d.insert_dimension(position=1, inplace=True)

        # Find the derivative
        f.data /= d

        # Update the standard name and long name
        f.set_property("long_name", f"{axis_in} derivative of {f.identity()}")
        f.del_property("standard_name", None)

        return f

    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    def field_anc(
        self,
        *identity,
        key=False,
        default=ValueError(),
        item=False,
        **filter_kwargs,
    ):
        """Alias for `cf.Field.field_ancillary`."""
        return self.field_ancillary(
            *identity, key=key, default=default, item=item, **filter_kwargs
        )

    def field_ancs(self, *identities, **filter_kwargs):
        """Alias for `field_ancillaries`."""
        return self.field_ancillaries(*identities, **filter_kwargs)

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    @property
    def _Axes(self):
        """Return domain axis constructs.

        Deprecated at version 3.0.0. Use `domain_axes` method instead.

        """
        raise DeprecationError(
            f"{self.__class__.__name__} attribute '_Axes' has been deprecated "
            "at version 3.0.0 and is no longer available and will be removed"
            "at v4.0.0"
            "Use 'domain_axes' instead."
        )

    @property
    def CellMethods(self):
        """Return cell method constructs.

        Deprecated at version 3.0.0. Use `cell_methods` method instead.

        """
        raise DeprecationError(
            f"{self.__class__.__name__} attribute 'CellMethods' has been "
            "deprecated at version 3.0.0 and is no longer available "
            "and will be removed at v4.0.0. "
            "Use 'cell_methods' instead."
        )

    @property
    def Items(self):
        """Return domain items as (construct key, construct) pairs.

        Deprecated at version 3.0.0. Use `constructs` method instead.

        """
        raise DeprecationError(
            f"{self.__class__.__name__} attribute 'Items' has been deprecated "
            "at version 3.0.0 and is no longer available "
            "and will be removed at v4.0.0. "
            "Use 'constructs' instead."
        )

    def CM(self, xxx):
        """Return cell method constructs.

        Deprecated at version 3.0.0.

        """
        raise DeprecationError(
            f"{self.__class__.__name__} method 'CM' has been deprecated "
            "at version 3.0.0 and is no longer available "
            "and will be removed at v4.0.0. "
        )

    def axis_name(self, *args, **kwargs):
        """Return the canonical name for an axis.

        Deprecated at version 3.0.0. Use `domain_axis_identity` method
        instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "axis_name",
            "Use 'domain_axis_identity' method instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def data_axes(self):
        """Return the domain axes for the data array dimensions.

        Deprecated at version 3.0.0. Use `get_data_axes` method instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "data_axes",
            "Use 'get_data_axes' method instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @_manage_log_level_via_verbosity
    def equivalent(self, other, rtol=None, atol=None, verbose=None):
        """True if two fields are equivalent, False otherwise.

        Deprecated at version 3.0.0.

        """
        _DEPRECATION_ERROR_METHOD(self, "equivalent", version="3.0.0")

    @classmethod
    def example_field(cls, n):
        """Return an example field construct.

        Deprecated at version 3.0.5. Use function `cf.example_field` instead.

        .. versionadded:: 3.0.4

        """
        _DEPRECATION_ERROR_METHOD(
            cls,
            "example_field",
            "Use function 'cf.example_field' instead.",
            version="3.0.5",
            removed_at="4.0.0",
        )  # pragma: no cover

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def expand_dims(self, position=0, axes=None, i=False, **kwargs):
        """Insert a size 1 axis into the data array.

        Deprecated at version 3.0.0. Use `insert_dimension` method
        instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "expand_dims",
            "Use 'insert_dimension' method instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def field(
        self,
        description=None,
        role=None,
        axes=None,
        axes_all=None,
        axes_subset=None,
        axes_superset=None,
        exact=False,
        inverse=False,
        match_and=True,
        ndim=None,
        bounds=False,
    ):
        """Create an independent field from a domain item.

        Deprecated at version 3.0.0. Use 'convert' method instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "field",
            "Use 'convert' method instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def HDF_chunks(self, *chunksizes):
        """Deprecated at version 3.0.0.

        Use methods 'Data.nc_hdf5_chunksizes',
        'Data.nc_set_hdf5_chunksizes', 'Data.nc_clear_hdf5_chunksizes'
        instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "HDF_chunks",
            "Use methods 'Data.nc_hdf5_chunksizes', "
            "'Data.nc_set_hdf5_chunksizes', "
            "'Data.nc_clear_hdf5_chunksizes' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def insert_measure(
        self, item, key=None, axes=None, copy=True, replace=True
    ):
        """Insert a cell measure object into the field.

        Deprecated at version 3.0.0. Use method 'set_construct' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "insert_measure",
            "Use method 'set_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def insert_dim(self, item, key=None, axes=None, copy=True, replace=True):
        """Insert a dimension coordinate object into the field.

        Deprecated at version 3.0.0. Use method 'set_construct' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "insert_dim",
            "Use method 'set_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def insert_axis(self, axis, key=None, replace=True):
        """Insert a domain axis into the field.

        Deprecated at version 3.0.0. Use method 'set_construct' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "insert_axis",
            "Use method 'set_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def insert_item(
        self, role, item, key=None, axes=None, copy=True, replace=True
    ):
        """Insert an item into the field.

        Deprecated at version 3.0.0. Use method 'set_construct' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "insert_item",
            "Use method 'set_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def insert_aux(self, item, key=None, axes=None, copy=True, replace=True):
        """Insert an auxiliary coordinate object into the field.

        Deprecated at version 3.0.0. Use method 'set_construct' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "insert_aux",
            "Use method 'set_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def insert_cell_methods(self, item):
        """Insert one or more cell method objects into the field.

        Deprecated at version 3.0.0. Use method 'set_construct' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "insert_cell_methods",
            "Use method 'set_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def insert_domain_anc(
        self, item, key=None, axes=None, copy=True, replace=True
    ):
        """Insert a domain ancillary object into the field.

        Deprecated at version 3.0.0. Use method 'set_construct' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "insert_domain_anc",
            "Use method 'set_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def insert_data(self, data, axes=None, copy=True, replace=True):
        """Insert a data array into the field.

        Deprecated at version 3.0.0. Use method 'set_data' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "insert_data",
            "Use method 'set_data' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def insert_field_anc(
        self, item, key=None, axes=None, copy=True, replace=True
    ):
        """Insert a field ancillary object into the field.

        Deprecated at version 3.0.0. Use method 'set_construct'
        instead.g

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "insert_field_anc",
            "Use method 'set_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def insert_ref(self, item, key=None, axes=None, copy=True, replace=True):
        """Insert a coordinate reference object into the field.

        Deprecated at version 3.0.0. Use method 'set_construct' or
        'set_coordinate_reference' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "insert_ref",
            "Use method 'set_construct' or 'set_coordinate_reference' "
            "instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def item(
        self,
        *identity,
        key=False,
        default=ValueError(),
        item=False,
        **filter_kwargs,
    ):
        """Return a domain item of the field.

        Deprecated. Use `construct` method instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "item",
            "Use 'construct' method instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def items(self, *identities, **filter_kwargs):
        """Return domain items as (construct key, construct) pairs.

        Deprecated. Use `constructs` method instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "items",
            "Use 'constructs' method instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def item_axes(
        self,
        description=None,
        role=None,
        axes=None,
        axes_all=None,
        axes_subset=None,
        axes_superset=None,
        exact=False,
        inverse=False,
        match_and=True,
        ndim=None,
        default=None,
    ):
        """Return the axes of a domain item of the field.

        Deprecated at version 3.0.0. Use the 'get_data_axes' method
        instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "item_axes",
            "Use method 'get_data_axes' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def items_axes(
        self,
        description=None,
        role=None,
        axes=None,
        axes_all=None,
        axes_subset=None,
        axes_superset=None,
        exact=False,
        inverse=False,
        match_and=True,
        ndim=None,
    ):
        """Return the axes of items of the field.

        Deprecated at version 3.0.0. Use the 'data_axes' method of
        attribute 'constructs' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "items_axes",
            "Use the 'data_axes' method of attribute 'constructs' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def key_item(self, identity, default=ValueError(), **kwargs):
        """Return an item, or its identifier, from the field.

        Deprecated at version 3.0.0

        """
        _DEPRECATION_ERROR_METHOD(
            self, "key_item", version="3.0.0", removed_at="4.0.0"
        )

    def new_identifier(self, item_type):
        """Return a new, unused construct key.

        Deprecated at version 3.0.0. Use 'new_identifier' method of
        'constructs' attribute instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            " new_identifier",
            "Use 'new_identifier' method of 'constructs' attribute instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def remove_item(
        self,
        description=None,
        role=None,
        axes=None,
        axes_all=None,
        axes_subset=None,
        axes_superset=None,
        ndim=None,
        exact=False,
        inverse=False,
        match_and=True,
        key=False,
    ):
        """Remove and return an item from the field.

        Deprecated at version 3.0.0. Use `del_construct` method instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "remove_item",
            "Use method 'del_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def remove_items(
        self,
        description=None,
        role=None,
        axes=None,
        axes_all=None,
        axes_subset=None,
        axes_superset=None,
        ndim=None,
        exact=False,
        inverse=False,
        match_and=True,
    ):
        """Remove and return items from the field.

        Deprecated at version 3.0.0. Use `del_construct` method instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "remove_items",
            "Use method 'del_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def remove_axes(self, axes=None, **kwargs):
        """Remove and return axes from the field.

        Deprecated at version 3.0.0. Use method 'del_construct' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "remove_axes",
            "Use method 'del_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def remove_axis(self, axes=None, size=None, **kwargs):
        """Remove and return a unique axis from the field.

        Deprecated at version 3.0.0. Use method 'del_construct' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "remove_axis",
            "Use method 'del_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def remove_data(self, default=ValueError()):
        """Remove and return the data array.

        Deprecated at version 3.0.0. Use method 'del_data' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "remove_data",
            "Use method 'del_data' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def transpose_item(self, description=None, iaxes=None, **kwargs):
        """Permute the axes of a field item data array.

        Deprecated at version 3.0.0. Use method 'transpose_construct'
        instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "transpose_item",
            "Use method 'transpose_construct' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def unlimited(self, *args):
        """Deprecated at version 3.0.0.

        Use methods `DomainAxis.nc_is_unlimited`, and
        `DomainAxis.nc_set_unlimited` instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "unlimited",
            "Use methods 'DomainAxis.nc_is_unlimited', and "
            "'DomainAxis.nc_set_unlimited' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover
