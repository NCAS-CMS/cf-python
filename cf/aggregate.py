import logging
from collections import namedtuple
from operator import itemgetter

from cfdm import is_log_level_debug, is_log_level_detail, is_log_level_info
from numpy import argsort as numpy_argsort
from numpy import dtype as numpy_dtype
from numpy import sort as numpy_sort

from .auxiliarycoordinate import AuxiliaryCoordinate
from .data.data import Data
from .decorators import (
    _manage_log_level_via_verbose_attr,
    _manage_log_level_via_verbosity,
    _reset_log_emergence_level,
)
from .domainaxis import DomainAxis
from .fieldlist import FieldList
from .functions import _DEPRECATION_ERROR_FUNCTION_KWARGS, _numpy_allclose
from .functions import atol as cf_atol
from .functions import flat, hash_array
from .functions import rtol as cf_rtol
from .query import gt
from .units import Units

logger = logging.getLogger(__name__)


_dtype_float = numpy_dtype(float)

# # --------------------------------------------------------------------
# # Global properties, as defined in Appendix A of the CF conventions.
# # --------------------------------------------------------------------
# _global_properties = set(('comment',
#                           'Conventions',
#                           'history',
#                           'institution',
#                           'references',
#                           'source',
#                           'title',
#                           ))

# --------------------------------------------------------------------
# Data variable properties, as defined in Appendix A of the CF
# conventions, without those which are not simple. And less
# 'long_name'.
# --------------------------------------------------------------------
_signature_properties = set(
    (
        "add_offset",
        "calendar",
        "cell_methods",
        "_FillValue",
        "flag_masks",
        "flag_meanings",
        "flag_values",
        "missing_value",
        "scale_factor",
        "standard_error_multiplier",
        "standard_name",
        "units",
        "valid_max",
        "valid_min",
        "valid_range",
    )
)

# _standard_properties = _data_properties.union(_global_properties)

_no_units = Units()


class _HFLCache:
    """A cache for coordinate and cell measure hashes, first and last
    values and first and last cell bounds."""

    def __init__(self):
        self.hash = {}
        self.fl = {}
        self.flb = {}
        self.hash_to_array = {}


class _Meta:
    """A summary of a field.

    This object contains everything you need to know in order to
    aggregate the field.

    """

    #
    _canonical_units = {}

    #
    _canonical_cell_methods = []

    #
    _structural_signature = namedtuple(
        "signature",
        (
            "Type",
            "Identity",
            "Units",
            "Cell_methods",
            "Data",
            "Properties",
            "standard_error_multiplier",
            "valid_min",
            "valid_max",
            "valid_range",
            "Flags",
            "Coordinate_references",
            "Axes",
            "dim_coord_index",
            "Nd_coordinates",
            "Cell_measures",
            "Domain_ancillaries",
            "Field_ancillaries",
        ),
    )

    def __init__(
        self,
        f,
        rtol=None,
        atol=None,
        verbose=None,
        relaxed_units=False,
        allow_no_identity=False,
        respect_valid=False,
        equal_all=False,
        exist_all=False,
        equal=None,
        exist=None,
        ignore=None,
        dimension=(),
        relaxed_identities=False,
        ncvar_identities=False,
        field_identity=None,
        copy=True,
    ):
        """**initialisation**

        :Parameters:

            f: `Field` or `Domain`

            verbose: `int` or `str` or `None`, optional
                If an integer from ``-1`` to ``3``, or an equivalent
                string equal ignoring case to one of:

                * ``'DISABLE'`` (``0``)
                * ``'WARNING'`` (``1``)
                * ``'INFO'`` (``2``)
                * ``'DETAIL'`` (``3``)
                * ``'DEBUG'`` (``-1``)

                set for the duration of the method call only as the
                minimum cut-off for the verboseness level of displayed
                output (log) messages, regardless of the
                globally-configured `cf.log_level`. Note that
                increasing numerical value corresponds to increasing
                verbosity, with the exception of ``-1`` as a special
                case of maximal and extreme verbosity.

                Otherwise, if `None` (the default value), output
                messages will be shown according to the value of the
                `cf.log_level` setting.

                Overall, the higher a non-negative integer or
                equivalent string that is set (up to a maximum of
                ``3``/``'DETAIL'``) for increasing verbosity, the more
                description that is printed to convey information
                about the operation.

            relaxed_units: `bool`, optional
                If True then assume that field and metadata constructs
                with the same identity but missing units actually have
                equivalent (but unspecified) units, so that aggregation
                may occur. By default such field constructs are not
                aggregatable.

            allow_no_identity: `bool`, optional
                If True then assume that field and metadata constructs
                with no identity (see the *relaxed_identities* parameter)
                actually have the same (but unspecified) identity, so
                that aggregation may occur. By default such field
                constructs are not aggregatable.

            rtol: number, optional
                The tolerance on relative differences between real
                numbers. The default value is set by the
                `cf.rtol` function.

            atol: number, optional
                The tolerance on absolute differences between real
                numbers. The default value is set by the
                `cf.atol` function.

            dimension: (sequence of) `str`, optional
                Create new axes for each input field which has one or
                more of the given properties. For each CF property name
                specified, if an input field has the property then, prior
                to aggregation, a new axis is created with an auxiliary
                coordinate whose datum is the property's value and the
                property itself is deleted from that field.

            copy: `bool` optional
                If False then do not copy fields prior to aggregation.
                Setting this option to False may change input fields in
                place, and the output fields may not be independent of
                the inputs. However, if it is known that the input
                fields are never to accessed again (such as in this case:
                ``f = cf.aggregate(f)``) then setting *copy* to False can
                reduce the time taken for aggregation.

        """
        self._bool = False
        self.cell_values = False

        self.verbose = verbose

        self.sort_indices = {}
        self.sort_keys = {}
        self.key_to_identity = {}

        self.all_field_anc_identities = set()
        self.all_domain_anc_identities = set()

        self.message = ""

        strict_identities = not (
            relaxed_identities
            or ncvar_identities
            or field_identity is not None
        )

        self.relaxed_identities = relaxed_identities
        self.strict_identities = strict_identities
        self.field_identity = field_identity
        self.ncvar_identities = ncvar_identities

        # Initialise the flag which indicates whether or not this
        # field has already been aggregated
        self.aggregated_field = False

        # Map axis canonical identities to their identifiers
        #
        # For example: {'time': 'dim2'}
        self.id_to_axis = {}

        # Map axis identifiers to their canonical identities
        #
        # For example: {'dim2': 'time'}
        self.axis_to_id = {}

        # ------------------------------------------------------------
        # Parent field or domain
        # ------------------------------------------------------------
        self.field = f
        self.has_data = f.has_data()
        self.identity = f.identity(
            strict=strict_identities,
            relaxed=relaxed_identities and not ncvar_identities,
            nc_only=ncvar_identities,
            default=None,
        )

        if field_identity:
            self.identity = f.get_property(field_identity, None)

        construct_axes = f.constructs.data_axes()

        # ------------------------------------------------------------
        #
        # ------------------------------------------------------------
        signature_override = getattr(f, "aggregate", None)
        if signature_override is not None:
            self.signature = signature_override
            self._bool = True
            return

        if self.identity is None:
            if not allow_no_identity and self.has_data:
                self.message = (
                    "no identity; consider setting " "relaxed_identities"
                )
                return
        #        elif not self.has_data:
        #            self.message = "{} has no data".format(f.__class__.__name__)
        #            return

        # ------------------------------------------------------------
        # Promote selected properties to 1-d, size 1 auxiliary
        # coordinates
        # ------------------------------------------------------------
        _copy = copy
        for prop in dimension:
            value = f.get_property(prop, None)
            if value is None:
                continue

            aux_coord = AuxiliaryCoordinate(
                properties={"long_name": prop},
                data=Data([value], units=""),
                copy=False,
            )
            aux_coord.nc_set_variable(prop)
            aux_coord.id = prop

            if _copy:
                # Copy the field, as we're about to change it.
                f = f.copy()
                self.field = f
                _copy = False

            axis = f.set_construct(DomainAxis(1))
            f.set_construct(aux_coord, axes=[axis], copy=False)

            f.del_property(prop)

        if dimension:
            construct_axes = f.constructs.data_axes()

        self.units = self.canonical_units(
            f, self.identity, relaxed_units=relaxed_units
        )

        # ------------------------------------------------------------
        # Coordinate and cell measure arrays
        # ------------------------------------------------------------
        self.hash_values = {}
        self.first_values = {}
        self.last_values = {}
        self.first_bounds = {}
        self.last_bounds = {}

        # Dictionaries mapping auxiliary coordinate identifiers
        # to their auxiliary coordinate objects
        auxs_1d = f.auxiliary_coordinates(filter_by_naxes=(1,), todict=True)

        # A set containing the identity of each coordinate
        #
        # For example: set(['time', 'height', 'latitude',
        # 'longitude'])
        self.all_coord_identities = {None: set()}

        self.axis = {}

        # ------------------------------------------------------------
        # Coordinate references (formula_terms and grid mappings)
        # ------------------------------------------------------------
        refs = f.coordinate_references(todict=True)
        if not refs:
            self.coordrefs = ()
        else:
            self.coordrefs = list(refs.values())

        for axis, domain_axis in f.domain_axes(todict=True).items():

            # List some information about each 1-d coordinate which
            # spans this axis. The order of elements is arbitrary, as
            # ultimately it will get sorted by each element's 'name'
            # key values.
            #
            # For example: [{'name': 'time', 'key': 'dim0', 'units':
            # <CF Units: ...>}, {'name': 'forecast_ref_time', 'key':
            # 'aux0', 'units': <CF Units: ...>}]
            info_dim = []

            dim_coord_key, dim_coord = f.dimension_coordinate(
                filter_by_axis=(axis,), item=True, default=(None, None)
            )
            dim_identity = None

            if dim_coord is not None:
                # ----------------------------------------------------
                # 1-d dimension coordinate
                # ----------------------------------------------------
                dim_identity = self.coord_has_identity_and_data(dim_coord)

                if dim_identity is None:
                    return

                # Find the canonical units for this dimension
                # coordinate
                units = self.canonical_units(
                    dim_coord, dim_identity, relaxed_units=relaxed_units
                )

                info_dim.append(
                    {
                        "identity": dim_identity,
                        "key": dim_coord_key,
                        "units": units,
                        "hasdata": dim_coord.has_data(),
                        "hasbounds": dim_coord.has_bounds(),
                        "coordrefs": self.find_coordrefs(axis),
                    }
                )
            #                     'size'     : None})

            # Find the 1-d auxiliary coordinates which span this axis
            aux_coords = {
                aux: auxs_1d.pop(aux)
                for aux in tuple(auxs_1d)
                if axis in construct_axes[aux]
            }

            info_aux = []
            for key, aux_coord in aux_coords.items():
                # ----------------------------------------------------
                # 1-d auxiliary coordinate
                # ----------------------------------------------------
                if dim_identity is not None:
                    axes = (dim_identity,)
                else:
                    axes = None

                aux_identity = self.coord_has_identity_and_data(
                    aux_coord, axes=(dim_identity,)
                )
                if aux_identity is None:
                    return

                # Find the canonical units for this 1-d auxiliary
                # coordinate
                units = self.canonical_units(
                    aux_coord, aux_identity, relaxed_units=relaxed_units
                )

                info_aux.append(
                    {
                        "identity": aux_identity,
                        "key": key,
                        "units": units,
                        "hasdata": aux_coord.has_data(),
                        "hasbounds": aux_coord.has_bounds(),
                        "coordrefs": self.find_coordrefs(key),
                    }
                )

            # Sort the 1-d auxiliary coordinate information
            info_aux.sort(key=itemgetter("identity"))

            # Prepend the dimension coordinate information to the
            # auxiliary coordinate information
            info_1d_coord = info_dim + info_aux

            # Find the canonical identity for this axis
            identity = None
            if info_1d_coord:
                identity = info_1d_coord[0]["identity"]
            elif not self.relaxed_identities:
                self.message = (
                    "axis has no one-dimensional nor scalar coordinates"
                )
                return

            size = None
            if identity is None and self.relaxed_identities:
                # There are no 1-d coordinates and relaxed identities
                # are on, so see if we can identify the domain axis by
                # its netCDF dimension name.
                identity = domain_axis.nc_get_dimension(None)
                if identity is None:
                    self.message = (
                        f"axis {f.constructs.domain_axis_identity(axis)!r} "
                        "has no netCDF dimension name"
                    )  # TODO
                    return

                size = domain_axis.get_size()

            axis_identities = {
                "ids": "identity",
                "keys": "key",
                "units": "units",
                "hasdata": "hasdata",
                "hasbounds": "hasbounds",
                "coordrefs": "coordrefs",
            }
            self.axis[identity] = {
                name: tuple(i[idt] for i in info_1d_coord)
                for name, idt in axis_identities.items()
            }

            if info_dim:
                self.axis[identity]["dim_coord_index"] = 0
            else:
                self.axis[identity]["dim_coord_index"] = None

            # Store the axis size, which will be None unless we
            # identified the dimension solely by its netCDF dimension
            # name.
            self.axis[identity]["size"] = size

            self.id_to_axis[identity] = axis
            self.axis_to_id[axis] = identity

        # Create a sorted list of the axes' canonical identities
        #
        # For example: ['latitude', 'longitude', 'time']
        self.axis_ids = sorted(self.axis)

        # ------------------------------------------------------------
        # N-d auxiliary coordinates
        # ------------------------------------------------------------
        self.nd_aux = {}
        for key, nd_aux_coord in f.auxiliary_coordinates(
            filter_by_naxes=(gt(1),), todict=True
        ).items():
            # Find axes' canonical identities
            axes = [self.axis_to_id[axis] for axis in construct_axes[key]]
            axes = tuple(sorted(axes))

            # Find this N-d auxiliary coordinate's identity
            identity = self.coord_has_identity_and_data(
                nd_aux_coord, axes=axes
            )
            if identity is None:
                return

            # Find the canonical units
            units = self.canonical_units(
                nd_aux_coord, identity, relaxed_units=relaxed_units
            )

            self.nd_aux[identity] = {
                "key": key,
                "units": units,
                "axes": axes,
                "hasdata": nd_aux_coord.has_data(),
                "hasbounds": nd_aux_coord.has_bounds(),
                "coordrefs": self.find_coordrefs(key),
            }

        # ------------------------------------------------------------
        # Cell methods
        # ------------------------------------------------------------
        self.cell_methods = self.canonical_cell_methods(rtol=rtol, atol=atol)

        # ------------------------------------------------------------
        # Field ancillaries
        # ------------------------------------------------------------
        self.field_anc = {}
        field_ancillaries = f.constructs.filter_by_type(
            "field_ancillary", todict=True
        )
        for key, field_anc in field_ancillaries.items():
            # Find this field ancillary's identity
            identity = self.field_ancillary_has_identity_and_data(field_anc)
            if identity is None:
                return

            # Find the canonical units
            units = self.canonical_units(
                field_anc, identity, relaxed_units=relaxed_units
            )

            # Find axes' canonical identities
            axes = [
                self.axis_to_id[axis] for axis in construct_axes[key]
            ]  # f.get_data_axes(key)]
            axes = tuple(sorted(axes))

            self.field_anc[identity] = {
                "key": key,
                "units": units,
                "axes": axes,
            }

        # ------------------------------------------------------------
        # Coordinate reference structural signatures. (Do this after
        # self.key_to_identity has been populated with domain
        # ancillary keys.)
        # ------------------------------------------------------------
        self.coordref_signatures = self.coordinate_reference_signatures(
            self.coordrefs
        )

        # ------------------------------------------------------------
        # Domain ancillaries
        # ------------------------------------------------------------
        self.domain_anc = {}

        # List of keys of domain ancillaries which are used in
        # coordinate references
        ancs_in_refs = []

        # Firstly process domain ancillaries which are used in
        # coordinate references
        for ref in f.coordinate_references(todict=True).values():
            ref_identity = ref.identity()
            for (
                term,
                identifier,
            ) in ref.coordinate_conversion.domain_ancillaries().items():
                key, anc = f.domain_ancillary(
                    identifier, item=True, default=(None, None)
                )
                if anc is None:
                    continue

                # Set this domain ancillary's identity
                identity = (ref_identity, term)
                identity = self.domain_ancillary_has_identity_and_data(
                    anc, identity
                )

                # Find the canonical units
                units = self.canonical_units(
                    anc, identity, relaxed_units=relaxed_units
                )

                # Find the canonical identities of the axes
                axes = [self.axis_to_id[axis] for axis in construct_axes[key]]
                axes = tuple(sorted(axes))

                self.domain_anc[identity] = {
                    "key": key,
                    "units": units,
                    "axes": axes,
                }

                self.key_to_identity[key] = identity

                ancs_in_refs.append(key)

        # Secondly process domain ancillaries which are not being used
        # in coordinate references
        for key, anc in f.domain_ancillaries(todict=True).items():
            if key in ancs_in_refs:
                continue

            # Find this domain ancillary's identity
            identity = self.domain_ancillary_has_identity_and_data(anc)
            if identity is None:
                return

            # Find the canonical units
            units = self.canonical_units(
                anc, identity, relaxed_units=relaxed_units
            )

            # Find the canonical identities of the axes
            axes = [self.axis_to_id[axis] for axis in construct_axes[key]]
            axes = tuple(sorted(axes))

            self.domain_anc[identity] = {
                "key": key,
                "units": units,
                "axes": axes,
            }

            self.key_to_identity[key] = identity

        # ------------------------------------------------------------
        # Cell measures
        # ------------------------------------------------------------
        self.msr = {}
        info_msr = {}
        copied_field = False
        for key, msr in f.cell_measures(todict=True).items():
            # If the measure is an external variable, remove it because
            # the dimensions are not known so there is no way to tell if the
            # aggregation should have changed it. (This is sufficiently
            # sensible behaviour for now, but will be reviewed in future.)
            # Note: for CF <=1.8 only cell measures can be external variables.
            if msr.nc_get_external():
                # Only create one copy of field if there is >1 external measure
                if not copied_field:
                    self.field = self.field.copy()  # copy as will delete msr
                    f = self.field
                    copied_field = True

                f.del_construct(key)

                if is_log_level_info(logger):
                    logger.info(
                        f"Removed {msr.identity()!r} construct from a copy "
                        f"of input field {f.identity()!r} pre-aggregation "
                        "because it is an external variable so it "
                        "is not possible to determine the influence the "
                        "aggregation process should have on it."
                    )

                continue

            if not self.cell_measure_has_data_and_units(msr):
                return

            # Find the canonical units for this cell measure
            units = self.canonical_units(
                msr,
                msr.identity(
                    strict=strict_identities, nc_only=ncvar_identities
                ),
                relaxed_units=relaxed_units,
            )

            # Find axes' canonical identities
            axes = [self.axis_to_id[axis] for axis in construct_axes[key]]
            axes = tuple(sorted(axes))

            if units in info_msr:
                # Check for ambiguous cell measures, i.e. those which
                # have the same units and span the same axes.
                for value in info_msr[units]:
                    if axes == value["axes"]:
                        self.message = f"duplicate {msr!r}"
                        return
            else:
                info_msr[units] = []

            info_msr[units].append({"key": key, "axes": axes})

        # For each cell measure's canonical units, sort the
        # information by axis identities.
        for units, value in info_msr.items():
            value.sort(key=itemgetter("axes"))
            self.msr[units] = {
                "keys": tuple([v["key"] for v in value]),
                "axes": tuple([v["axes"] for v in value]),
            }

        # ------------------------------------------------------------
        # Properties and attributes
        # ------------------------------------------------------------
        if not (equal or exist or equal_all or exist_all):
            self.properties = ()
        else:
            properties = f.properties()
            if ignore:
                for p in ignore:
                    properties.pop(p, None)

            if equal:
                eq = dict(
                    [(p, properties[p]) for p in equal if p in properties]
                )
            else:
                eq = {}

            if exist:
                ex = [p for p in exist if p in properties]
            else:
                ex = []

            eq_all = {}
            ex_all = []

            if equal_all:
                if not equal and not exist:
                    eq_all = properties
                else:  # None is Falsy (evaluates to False) & "short-circuits"
                    eq_all = dict(
                        [
                            (p, properties[p])
                            for p in properties
                            if (equal and p not in eq)
                            or (exist and p not in ex)
                        ]
                    )
            elif exist_all:
                if not equal and not exist:
                    ex_all = list(properties)
                else:  # None is Falsy (evaluates to False) & "short-circuits"
                    ex_all = [
                        p
                        for p in properties
                        if (equal and p not in eq) or (exist and p not in ex)
                    ]

            self.properties = tuple(
                sorted(ex_all + ex + list(eq_all.items()) + list(eq.items()))
            )

        # Attributes
        self.attributes = set(("file",))

        # ------------------------------------------------------------
        # Still here? Then create the structural signature.
        # ------------------------------------------------------------
        self.respect_valid = respect_valid
        self.structural_signature()

        # Initialise the flag which indicates whether or not this
        # field has already been aggregated
        self.aggregated_field = False

        self.sort_indices = {}
        self.sort_keys = {}

        # Finally, set the object to True
        self._bool = True

    def __bool__(self):
        """x.__bool__() <==> bool(x)"""
        return self._bool

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        return (
            f"<CF {self.__class__.__name__}: {getattr(self, 'field', None)!r}>"
        )

    def __str__(self):
        """x.__str__() <==> str(x)"""
        strings = []
        for attr in sorted(self.__dict__):
            strings.append(
                f"{self.__class__.__name__}.{attr} = {getattr(self, attr)!r}"
            )

        return "\n".join(strings)

    def coordinate_values(self):
        """Create a report listing coordinate cell values and bounds."""
        string = ["First cell: " + str(self.first_values)]
        string.append("Last cell:  " + str(self.last_values))
        string.append("First cell bounds: " + str(self.first_bounds))
        string.append("Last cell bounds:  " + str(self.last_bounds))

        return "\n".join(string)

    def copy(self):
        """Replace the field associated with a summary class with a deep
        copy."""
        new = _Meta.__new__(_Meta)
        new.__dict__ = self.__dict__.copy()
        new.field = new.field.copy()
        return new

    def canonical_units(self, variable, identity, relaxed_units=False):
        """Updates the `_canonical_units` attribute.

        :Parameters:

            variable: Construct

            identity: `str`

            relaxed_units: `bool`
                See the `aggregate` function for details.

        :Returns:

            `Units` or `None`

        """
        if variable.has_data():
            var_units = variable.Units
        elif variable.has_bounds():
            var_units = variable.bounds.Units
        else:
            return _no_units

        _canonical_units = self._canonical_units

        if identity in _canonical_units:
            if var_units:
                for u in _canonical_units[identity]:
                    if var_units.equivalent(u):
                        return u

                # Still here?
                _canonical_units[identity].append(var_units)

            elif relaxed_units or variable.dtype.kind in ("S", "U"):
                var_units = _no_units
        else:
            if var_units:
                _canonical_units[identity] = [var_units]
            elif relaxed_units or variable.dtype.kind in ("S", "U"):
                var_units = _no_units

        # Still here?
        return var_units

    def canonical_cell_methods(self, rtol=None, atol=None):
        """Updates the `_canonical_cell_methods` attribute.

        :Parameters:

            atol: `float`

            rtol: `float`

        :Returns:

            `CellMethods` or `None`

        """
        _canonical_cell_methods = self._canonical_cell_methods

        cell_methods = self.field.constructs.filter_by_type(
            "cell_method", todict=True
        )

        #        cms = getattr(self.field, 'CellMethods', None) # TODO
        if not cell_methods:
            return

        cms = []
        for cm in cell_methods.values():
            # cm.set_axes([self.axis_to_id.get(axis, axis) for axis in
            #              cm.get_axes(())])
            cm = cm.change_axes(self.axis_to_id)
            cm = cm.sorted()
            cms.append(cm)

        for canonical_cms in _canonical_cell_methods:  # TODO
            if len(cms) != len(canonical_cms):
                continue

            equivalent = True
            for cm, canonical_cm in zip(cms, canonical_cms):
                if not cm.equivalent(canonical_cm, rtol=rtol, atol=atol):
                    equivalent = False
                    break

            if equivalent:
                return canonical_cms

        # Still here?
        cms = tuple(cms)

        _canonical_cell_methods.append(cms)

        return cms

    def cell_measure_has_data_and_units(self, msr):
        """True only if a cell measure has both data and units.

        :Parameters:

            msr: `CellMeasure`

        :Returns:

            `bool`

        """
        if not msr.Units:
            self.message = f"{msr.identity()!r} cell measure has no units"
            return

        if not msr.has_data():
            self.message = f"{msr.identity()!r} cell measure has no data"
            return

        return True

    def coord_has_identity_and_data(self, coord, axes=None):
        """Return a coordinate construct's identity if it has one and
        has data.

        :Parameters:

            coord: Coordinate construct

            axes: sequence of `str`, optional
                Specifiers for the axes the coordinate must span. By
                default, axes are not considered when making this check.

        :Returns:

            `str` or `None`
                The coordinate construct's identity, or `None` if there is
                no identity and/or no data.

        """
        identity = coord.identity(
            strict=self.strict_identities,
            relaxed=self.relaxed_identities and not self.ncvar_identities,
            nc_only=self.ncvar_identities,
            default=None,
        )

        if identity is not None:
            all_coord_identities = self.all_coord_identities.setdefault(
                axes, set()
            )

            if identity in all_coord_identities:
                self.message = f"multiple {identity!r} coordinates"
                return

            if coord.has_data() or (
                coord.has_bounds() and coord.bounds.has_data()
            ):
                all_coord_identities.add(identity)
                return identity

        # Still here?
        self.message = f"{coord!r} has no identity or no data"

    def field_ancillary_has_identity_and_data(self, anc):
        """Return a field ancillary's identity if it has one and has
        data.

        :Parameters:

            coord: `FieldAncillary`

        :Returns:

            `str` or `None`
                The coordinate construct's identity, or `None` if
                there is no identity and/or no data.

        """
        identity = anc.identity(
            strict=self.strict_identities,
            relaxed=self.relaxed_identities and not self.ncvar_identities,
            nc_only=self.ncvar_identities,
            default=None,
        )

        if identity is not None:
            all_field_anc_identities = self.all_field_anc_identities

            if identity in all_field_anc_identities:
                self.message = f"multiple {identity!r} field ancillaries"
                return

            if anc.has_data():
                all_field_anc_identities.add(identity)
                return identity

        # Still here?
        self.message = (
            f"{anc.identity()!r} field ancillary has no identity or " "no data"
        )

    def coordinate_reference_signatures(self, refs):
        """List the structural signatures of given coordinate
        references.

        :Parameters:

            refs: sequence of `CoordinateReference`

        :Returns:

            `list`
                A structural signature of each coordinate reference
                object.

        **Examples**

        >>> sig = coordinate_reference_signatures(refs)

        """
        signatures = []

        if not refs:
            return signatures

        signatures = [ref.structural_signature() for ref in refs]

        for signature in signatures:
            if signature[0] is None:
                self.messsage = (
                    f"{self.f.identity()!r} field can't be aggregated due "
                    "to it having an unidentifiable "
                    "coordinate reference"
                )
                return

        signatures.sort()

        return signatures

    def domain_ancillary_has_identity_and_data(self, anc, identity=None):
        """Return a domain ancillary's identity if it has one and has
        data.

        :Parameters:

            anc: cf.DomainAncillary

            identity: optional

        :Returns:

            `str` or `None`
                The domain ancillary identity, or None if there is no
                identity and/or no data.

        """
        if identity is not None:
            anc_identity = identity
        else:
            anc_identity = anc.identity(
                strict=self.strict_identities,
                relaxed=self.relaxed_identities and not self.ncvar_identities,
                nc_only=self.ncvar_identities,
                default=None,
            )

        if anc_identity is None:
            self.message = (
                f"{anc.identity()!r} domain ancillary has no identity"
            )
            return

        all_domain_anc_identities = self.all_domain_anc_identities

        if anc_identity in all_domain_anc_identities:
            self.message = f"multiple {anc.identity()!r} domain ancillaries"
            return

        if not anc.has_data():
            self.message = f"{anc.identity()!r} domain ancillary has no data"
            return

        all_domain_anc_identities.add(anc_identity)

        return anc_identity

    @_manage_log_level_via_verbose_attr
    def print_info(self, signature=True):
        """Log information on the structural signature and coordinate
        values.

        :Parameters:

            signature: `_Meta`

        :Returns:

            `None`

        """
        if not is_log_level_detail(logger):
            return

        if signature:
            logger.detail(
                "STRUCTURAL SIGNATURE:\n" + self.string_structural_signature()
            )
        if self.cell_values:
            logger.detail(
                "CANONICAL COORDINATES:\n" + self.coordinate_values()
            )

        logger.debug(f"COMPLETE AGGREGATION METADATA:\n{self}")

    def string_structural_signature(self):
        """Return a multi-line string giving a field's structual
        signature.

        :Returns:

            `str`

        """
        string = []

        for key, value in self.signature._asdict().items():
            string.append(f"-> {key}: {value!r}")

        return "\n".join(string)

    def structural_signature(self):
        """Build the structual signature of a field from its components.

        :Returns:

            `tuple`

        """
        f = self.field

        # Initialise the structual signature with:
        #
        # * the construct type (field or domain)
        # * the identity
        # * the canonical units
        # * the canonical cell methods
        # * whether or not there is a data array
        Type = f.construct_type
        Identity = self.identity
        Units = self.units.formatted(definition=True)
        Cell_methods = self.cell_methods
        Data = self.has_data
        #        signature_append = signature.append

        # Properties
        #        signature_append(('Properties', self.properties))
        Properties = self.properties

        # standard_error_multiplier
        #        signature_append(('standard_error_multiplier',
        #                          f.get_property('standard_error_multiplier', None)))
        standard_error_multiplier = f.get_property(
            "standard_error_multiplier", None
        )

        # valid_min, valid_max, valid_range
        if self.respect_valid:
            valid_min = f.get_property("valid_min", None)
            valid_max = f.get_property("valid_max", None)
            valid_range = f.get_property("valid_range", None)
        else:
            valid_min = None
            valid_max = None
            valid_range = None

        # Flags
        Flags = getattr(f, "Flags", None)

        # Coordinate references
        Coordinate_references = tuple(self.coordref_signatures)

        # 1-d coordinates for each axis. Note that self.axis_ids has
        # already been sorted.
        axis = self.axis
        x = [
            (
                identity,
                ("ids", axis[identity]["ids"]),
                (
                    "units",
                    tuple(
                        [
                            u.formatted(definition=True)
                            for u in axis[identity]["units"]
                        ]
                    ),
                ),
                ("hasdata", axis[identity]["hasdata"]),
                ("hasbounds", axis[identity]["hasbounds"]),
                ("coordrefs", axis[identity]["coordrefs"]),
                ("size", axis[identity]["size"]),
            )
            for identity in self.axis_ids
        ]
        Axes = tuple(x)

        # Whether or not each axis has a dimension coordinate
        x = [
            False if axis[identity]["dim_coord_index"] is None else True
            for identity in self.axis_ids
        ]

        dim_coord_index = tuple(x)

        # N-d auxiliary coordinates
        nd_aux = self.nd_aux
        x = [
            (
                identity,
                (
                    "units",
                    nd_aux[identity]["units"].formatted(definition=True),
                ),
                ("axes", nd_aux[identity]["axes"]),
                ("hasdata", nd_aux[identity]["hasdata"]),
                ("hasbounds", nd_aux[identity]["hasbounds"]),
                ("coordrefs", nd_aux[identity]["coordrefs"]),
            )
            for identity in sorted(nd_aux)
        ]
        Nd_coordinates = tuple(x)

        # Cell measures
        msr = self.msr
        x = [
            (
                ("units", units.formatted(definition=True)),
                ("axes", msr[units]["axes"]),
            )
            for units in sorted(msr)
        ]
        Cell_measures = tuple(x)

        # Domain ancillaries
        domain_anc = self.domain_anc
        x = [
            (
                identity,
                (
                    "units",
                    domain_anc[identity]["units"].formatted(definition=True),
                ),
                ("axes", domain_anc[identity]["axes"]),
            )
            for identity in sorted(domain_anc)
        ]
        Domain_ancillaries = tuple(x)

        # Field ancillaries
        field_anc = self.field_anc
        x = [
            (
                identity,
                (
                    "units",
                    field_anc[identity]["units"].formatted(definition=True),
                ),
                ("axes", field_anc[identity]["axes"]),
            )
            for identity in sorted(field_anc)
        ]
        Field_ancillaries = tuple(x)

        self.signature = self._structural_signature(
            Type=Type,
            Identity=Identity,
            Units=Units,
            Cell_methods=Cell_methods,
            Data=Data,
            Properties=Properties,
            standard_error_multiplier=standard_error_multiplier,
            valid_min=valid_min,
            valid_max=valid_max,
            valid_range=valid_range,
            Flags=Flags,
            Coordinate_references=Coordinate_references,
            Axes=Axes,
            dim_coord_index=dim_coord_index,
            Nd_coordinates=Nd_coordinates,
            Cell_measures=Cell_measures,
            Domain_ancillaries=Domain_ancillaries,
            Field_ancillaries=Field_ancillaries,
        )

    def find_coordrefs(self, key):
        """Return all the coordinate references that point to a
        coordinate.

        :Parameters:

            key: `str`
                The key of the coordinate consrtuct.

        :Returns:

            `tuple` or `None`

        **Examples**

        >>> m.find_coordrefs('dim0')
        >>> m.find_coordrefs('aux1')

        """
        coordrefs = self.coordrefs

        if not coordrefs:
            return

        # Select the coordinate references which contain a pointer to
        # this coordinate
        names = [
            ref.identity() for ref in coordrefs if key in ref.coordinates()
        ]

        if not names:
            return

        return tuple(sorted(names))


@_manage_log_level_via_verbosity
def aggregate(
    fields,
    verbose=None,
    relaxed_units=False,
    overlap=True,
    contiguous=False,
    relaxed_identities=False,
    ncvar_identities=False,
    respect_valid=False,
    equal_all=False,
    exist_all=False,
    equal=None,
    exist=None,
    ignore=None,
    exclude=False,
    dimension=(),
    concatenate=True,
    copy=True,
    axes=None,
    donotchecknonaggregatingaxes=False,
    allow_no_identity=False,
    atol=None,
    rtol=None,
    no_overlap=False,
    shared_nc_domain=False,
    field_identity=None,
    info=False,
):
    """Aggregate field constructs into as few field constructs as
    possible.

    Aggregation is the combination of field constructs to create a new
    field construct that occupies a "larger" domain. Using the
    :ref:`aggregation rules <Aggregation-rules>`,
    field constructs are separated into aggregatable groups and each
    group is then aggregated to a single field construct.

    **Identifying field and metadata constructs**

    In order to ascertain whether or not field constructs are
    aggregatable, the aggregation rules rely on field constructs (and
    their metadata constructs where applicable) being identified by
    standard name properties. However, it is sometimes the case that
    standard names are not available. In such cases the `id` attribute
    (which is not a CF property) may be set on any construct, which
    will be treated like a standard name if one doesn't
    exist.

    Alternatively the *relaxed_identities* parameter allows long name
    properties or netCDF variable names to be used when standard names
    are missing; the *field_identity* parameter forces the field
    construct identities to be taken from a particular property; and
    the *ncvar_identities* parameter forces field and metadata
    constructs to be identified by their netCDF file variable names.

    :Parameters:

        fields: `FieldList` or sequence of `Field`
            The field constructs to aggregate.

        verbose: `int` or `str` or `None`, optional
            If an integer from ``-1`` to ``3``, or an equivalent string
            equal ignoring case to one of:

            * ``'DISABLE'`` (``0``)
            * ``'WARNING'`` (``1``)
            * ``'INFO'`` (``2``)
            * ``'DETAIL'`` (``3``)
            * ``'DEBUG'`` (``-1``)

            set for the duration of the method call only as the minimum
            cut-off for the verboseness level of displayed output (log)
            messages, regardless of the globally-configured `cf.log_level`.
            Note that increasing numerical value corresponds to increasing
            verbosity, with the exception of ``-1`` as a special case of
            maximal and extreme verbosity.

            Otherwise, if `None` (the default value), output messages will
            be shown according to the value of the `cf.log_level` setting.

            Overall, the higher a non-negative integer or equivalent string
            that is set (up to a maximum of ``3``/``'DETAIL'``) for
            increasing verbosity, the more description that is printed to
            convey information about the aggregation process. Explicitly:

            =============  =================================================
            Value set      Result
            =============  =================================================
            ``0``          * No information is displayed.

            ``1``          * Display information on which fields are
                           unaggregatable, and why.

            ``2``          * As well as the above, display the structural
                           signatures of the fields and, when there is more
                           than one field construct with the same structural
                           signature, their canonical first and last
                           coordinate values.

            ``3``/``-1``   * As well as the above, display the field
                           construct's complete aggregation metadata.
            =============  =================================================

        overlap: `bool`, optional
            If False then require that aggregated field constructs
            have adjacent dimension coordinate construct cells which
            do not overlap (but they may share common boundary
            values). Ignored for a dimension coordinate construct that
            does not have bounds. See also the *contiguous* parameter.

        contiguous: `bool`, optional
            If True then require that aggregated field constructs
            have adjacent dimension coordinate construct cells which
            overlap or share common boundary values. Ignored for a
            dimension coordinate construct that does not have
            bounds. See also the *overlap* parameter.

        relaxed_units: `bool`, optional
            If True then assume that field and metadata constructs
            with the same identity but missing units actually have
            equivalent (but unspecified) units, so that aggregation
            may occur. By default such field constructs are not
            aggregatable.

        allow_no_identity: `bool`, optional
            If True then assume that field and metadata constructs with
            no identity (see the *relaxed_identities* parameter) actually
            have the same (but unspecified) identity, so that aggregation
            may occur. By default such field constructs are not
            aggregatable.

        relaxed_identities: `bool`, optional
            If True and there is no standard name property nor "id"
            attribute, then allow field and metadata constructs to be
            identifiable by long name properties or netCDF variable
            names. Also allows netCDF dimension names to be used when
            there are no spanning 1-d coordinates.

        field_identity: `str`, optional
            Specify a property with which to identify field constructs
            instead of any other technique. How metadata constructs
            are identified is not affected by this parameter. See the
            *relaxed_identities* and *ncvar_identities* parameters.

            *Parameter example:*
              Force field constructs to be identified by the values of
              their long_name properties:
              ``field_identity='long_name'``

            .. versionadded:: 3.1.0

        ncvar_identities: `bool`, optional
            If True then force field and metadata constructs to be
            identified by their netCDF file variable names See also the
            *relaxed_identies* parameter.

        equal_all: `bool`, optional
            If True then require that aggregated fields have the same set
            of non-standard CF properties (including
            `~cf.Field.long_name`), with the same values. See the
            *concatenate* parameter.

        equal: (sequence of) `str`, optional
            Specify CF properties for which it is required that aggregated
            fields all contain the properties, with the same values. See
            the *concatenate* parameter.

        exist_all: `bool`, optional
            If True then require that aggregated fields have the same set
            of non-standard CF properties (including, in this case,
            long_name), but not requiring the values to be the same. See
            the *concatenate* parameter.

        exist: (sequence of) `str`, optional
            Specify CF properties for which it is required that aggregated
            fields all contain the properties, but not requiring the
            values to be the same. See the *concatenate* parameter.

        ignore: (sequence of) `str`, optional
            Specify CF properties to omit from any properties
            specified by or implied by the *equal_all*, *exist_all*,
            *equal* and *exist* parameters.

        exclude: `bool`, optional
            If True then do not return unaggregatable field
            constructs. By default, all input field constructs are
            represent in the outputs.

        respect_valid: `bool`, optional
            If True then the CF properties `~cf.Field.valid_min`,
            `~cf.Field.valid_max` and `~cf.Field.valid_range` are taken
            into account during aggregation. I.e. a requirement for
            aggregation is that fields have identical values for each
            these attributes, if set. By default these CF properties are
            ignored and are not set in the output fields.

        dimension: (sequence of) `str`, optional
            Create new axes for each input field which has one or more of
            the given properties. For each CF property name specified, if
            an input field has the property then, prior to aggregation, a
            new axis is created with an auxiliary coordinate whose datum
            is the property's value and the property itself is deleted
            from that field.

        concatenate: `bool`, optional
            If False then a CF property is omitted from an aggregated
            field if the property has unequal values across constituent
            fields or is missing from at least one constituent field. By
            default a CF property in an aggregated field is the
            concatenated collection of the distinct values from the
            constituent fields, delimited with the string
            ``' :AGGREGATED: '``.

        copy: `bool`, optional
            If False then do not copy fields prior to aggregation.
            Setting this option to False may change input fields in place,
            and the output fields may not be independent of the
            inputs. However, if it is known that the input fields are
            never to accessed again (such as in this case: ``f =
            cf.aggregate(f)``) then setting *copy* to False can reduce the
            time taken for aggregation.

        axes: (sequence of) `str`, optional
            Select axes to aggregate over. Aggregation will only occur
            over as large a subset as possible of these axes. Each axis is
            identified by the exact identity of a one dimensional
            coordinate object, as returned by its `!identity`
            method. Aggregations over more than one axis will occur in the
            order given. By default, aggregation will be over as many axes
            as possible.

        donotchecknonaggregatingaxes: `bool`, optional
            If True, and *axes* is set, then checks for consistent data
            array values will only be made for one dimensional coordinate
            objects which span the any of the given aggregating axes. This
            can reduce the time taken for aggregation, but if any those
            checks would have failed then this clearly allows the
            possibility of an incorrect result. Therefore, this option
            should only be used in cases for which it is known that the
            non-aggregating axes are in fact already entirely consistent.

        atol: number, optional
            The tolerance on absolute differences between real
            numbers. The default value is set by the
            `cf.atol` function.

        rtol: number, optional
            The tolerance on relative differences between real
            numbers. The default value is set by the
            `cf.rtol` function.

        no_overlap:
            Use the *overlap* parameter instead.

        shared_nc_domain: deprecated at version 3.0.0
            No longer required due to updated CF data model.

        info: deprecated at version 3.5.0
            Use the *verbose* parameter instead.

    :Returns:

        `FieldList`
            The aggregated field constructs.

    **Examples**

    The following six fields comprise eastward wind at two different times
    and for three different atmospheric heights for each time:

    >>> f
    [<CF Field: eastward_wind(latitude(73), longitude(96)>,
     <CF Field: eastward_wind(latitude(73), longitude(96)>,
     <CF Field: eastward_wind(latitude(73), longitude(96)>,
     <CF Field: eastward_wind(latitude(73), longitude(96)>,
     <CF Field: eastward_wind(latitude(73), longitude(96)>,
     <CF Field: eastward_wind(latitude(73), longitude(96)>]
    >>> g = cf.aggregate(f)
    >>> g
    [<CF Field: eastward_wind(height(3), time(2), latitude(73), longitude(96)>]
    >>> g[0].source
    'Model A'
    >>> g = cf.aggregate(f, dimension=('source',))
    [<CF Field: eastward_wind(source(1), height(3), time(2), latitude(73), longitude(96)>]
    >>> g[0].source
    AttributeError: 'Field' object has no attribute 'source'

    """
    if no_overlap is not False:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "cf.aggregate",
            {"no_overlap": no_overlap},
            "Use keyword 'overlap' instead.",
        )  # pragma: no cover

    if info is not False:  # catch 'Falsy' entries e.g. standard info=0
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "cf.aggregate",
            {"info": info},
            "Use keyword 'verbose' instead."
            "\n\n"
            "Note the informational levels have been remapped: "
            "\ninfo=0 maps to verbose=1"
            "\ninfo=1 maps to verbose=2"
            "\ninfo=2 maps to verbose=3"
            "\ninfo=3 maps to verbose=-1",
            version="3.5.0",
        )  # pragma: no cover

    # Initialise the cache for coordinate and cell measure hashes,
    # first and last values and first and last cell bounds
    hfl_cache = _HFLCache()

    output_constructs = []

    output_constructs_append = output_constructs.append

    if exclude:
        exclude = " NOT"
    else:
        exclude = ""

    if atol is None:
        atol = cf_atol()

    if rtol is None:
        rtol = cf_rtol()

    atol = float(atol)
    rtol = float(rtol)

    if axes is not None and isinstance(axes, str):
        axes = (axes,)

    # Parse parameters
    strict_identities = not (relaxed_identities or ncvar_identities)

    if isinstance(dimension, str):
        dimension = (dimension,)

    if exist_all and equal_all:
        raise ValueError(
            "Only one of 'exist_all' and 'equal_all' can be True, since "
            "these options are conflicting. Run 'help(cf.aggregate)' to read "
            "descriptions of each option to see which is applicable."
        )

    if equal or exist or ignore:
        properties = {"equal": equal, "exist": exist, "ignore": ignore}

        for key, value in properties.items():
            if not value:
                continue

            if isinstance(value, str):
                # If it is a string then convert to a single element
                # sequence
                properties[key] = (value,)
            else:
                try:
                    value[0]
                except TypeError:
                    raise TypeError(
                        f"Bad type of {key!r} parameter: {type(value)!r}"
                    )

        equal = properties["equal"]
        exist = properties["exist"]
        ignore = properties["ignore"]

        if equal and exist and set(equal).intersection(exist):
            raise AttributeError(
                "Can't specify the same properties in both the 'equal' "
                f" and 'exist' parameters: {set(equal).intersection(exist)!r}"
            )

        if ignore:
            ignore = _signature_properties.union(ignore)
    elif not ignore:
        ignore = _signature_properties

    unaggregatable = False
    status = 0

    # ================================================================
    # 1. Group together fields with the same structural signature
    # ================================================================
    signatures = {}
    for f in flat(fields):
        # ------------------------------------------------------------
        # Create the metadata summary, including the structural
        # signature
        # ------------------------------------------------------------
        meta = _Meta(
            f,
            verbose=verbose,
            rtol=rtol,
            atol=atol,
            relaxed_units=relaxed_units,
            allow_no_identity=allow_no_identity,
            equal_all=equal_all,
            exist_all=exist_all,
            equal=equal,
            exist=exist,
            ignore=ignore,
            dimension=dimension,
            relaxed_identities=relaxed_identities,
            ncvar_identities=ncvar_identities,
            field_identity=field_identity,
            respect_valid=respect_valid,
            copy=copy,
        )

        if not meta:
            unaggregatable = True
            status = 1

            if is_log_level_info(logger):
                # Note: deliberately no gap between 'has' and '{exclude}'
                logger.info(
                    f"Unaggregatable {f!r} has{exclude} been output: "
                    f"{meta.message}"
                )

            if not exclude:
                # This field does not have a structural signature, so
                # it can't be aggregated. Put it straight into the
                # output list and move on to the next input construct.
                if not copy:
                    output_constructs_append(f)
                else:
                    output_constructs_append(f.copy())

            continue

        # ------------------------------------------------------------
        # This field has a structural signature, so append it to the
        # list of fields with the same structural signature.
        # ------------------------------------------------------------
        signatures.setdefault(meta.signature, []).append(meta)

    # ================================================================
    # 2. Within each group of fields with the same structural
    #    signature, aggregate as many fields as possible. Sort the
    #    signatures so that independent aggregations of the same set
    #    of input fields return fields in the same order.
    # ================================================================

    #    x = []
    #    for signature in signatures:
    #        x.append(signature)
    #
    #    if len(x) == 2:
    #        logger.info(hash(x[0]))
    #        logger.info(hash(x[1]))
    #        for key, value in x[0]._asdict().items():
    #            if hash(value) != hash(getattr(x[1], key)):
    #                logger.info('{} no equal!'.format(key))
    #            if key == 'Coordinate_references' and value:
    #                for q1, q2 in zip(value, x[1].Coordinate_references):
    #                    for w1, w2 in zip(q1, q2):
    #                        logger.info(w1)
    #                        logger.info(w2)
    #                        logger.info(hash(w1))
    #                        logger.info(hash(w2))

    for signature in signatures:  # sorted(signatures):
        meta = signatures[signature]

        # Print useful information
        meta[0].print_info()

        # Note (verbosity): the interface between cf.aggregate's use of:
        #    _manage_log_level_via_verbosity
        # and some (only print_info ATM) of _Meta's methods' use of:
        #    _manage_log_level_via_verbose_attr
        # breaks the verbosity management here. This is currently the
        # only case in the codebases cfdm and cf where both decorators are at
        # play. Logic to handle the interface between the two has not
        # yet been added, so the latter called with print_info resets the
        # log level prematurely w.r.t the intentions of the former. For now,
        # we can work around this by resetting the verbosity manually after
        # the small number of print_info calls in this function, like so:
        if verbose is not None:
            # We already know _is_valid_log_level_int(verbose) is True since
            # if not, decorator would have errored before cf.aggregate ran.
            _reset_log_emergence_level(verbose)

        logger.detail("")

        if len(meta) == 1:
            # --------------------------------------------------------
            # There's only one field with this signature, so we can
            # add it straight to the output list and move on to the
            # next signature.
            # --------------------------------------------------------
            if not copy:
                output_constructs_append(meta[0].field)
            else:
                output_constructs_append(meta[0].field.copy())

            continue

        if not meta[0].units.isvalid:
            if is_log_level_info(logger):
                x = ", ".join(set(repr(m.units) for m in meta))
                logger.info(
                    f"Unaggregatable {meta[0].field.identity()!r} fields "
                    f"have{exclude} been output: Non-valid units {x}"
                )

            if not exclude:
                if copy:
                    output_constructs.extend(m.field.copy() for m in meta)
                else:
                    output_constructs.extend(m.field for m in meta)

            continue

        # ------------------------------------------------------------
        # Still here? Then there are 2 or more fields with this
        # signature which may be aggregatable. These fields need to be
        # passed through until no more aggregations are possible. With
        # each pass, the number of fields in the group will reduce by
        # one for each aggregation that occurs. Each pass represents
        # an aggregation in another axis.
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # For each axis's 1-d coordinates, create the canonical hash
        # value and the first and last cell values.
        # ------------------------------------------------------------
        if axes is None:
            # Aggregation will be over as many axes as possible
            aggregating_axes = meta[0].axis_ids
            _create_hash_and_first_values(
                meta, None, False, hfl_cache, rtol, atol
            )

        else:
            # Specific aggregation axes have been selected
            aggregating_axes = []
            axis_items = meta[0].axis.items()
            for axis in axes:
                coord = meta[0].field.coordinate(axis, default=None)
                if coord is None:
                    continue

                coord_identity = coord.identity(
                    strict=strict_identities,
                    relaxed=relaxed_identities and not ncvar_identities,
                    nc_only=ncvar_identities,
                    default=None,
                )
                for identity, value in axis_items:
                    if (
                        identity not in aggregating_axes
                        and coord_identity in value["ids"]
                    ):
                        aggregating_axes.append(identity)
                        break

            _create_hash_and_first_values(
                meta,
                aggregating_axes,
                donotchecknonaggregatingaxes,
                hfl_cache,
                rtol,
                atol,
            )

        # Print useful information
        for m in meta:
            m.print_info(signature=False)

        # See 'Note (verbosity)' above
        if verbose is not None:
            _reset_log_emergence_level(verbose)

        logger.detail("")

        # Take a shallow copy in case we abandon and want to output
        # the original, unaggregated fields.
        meta0 = meta[:]

        unaggregatable = False

        for axis in aggregating_axes:

            number_of_fields = len(meta)
            if number_of_fields == 1:
                break

            # --------------------------------------------------------
            # Separate the fields with the same structural signature
            # into groups such that either within each group the
            # fields' domains differ only long the axis or each group
            # contains only one field.
            #
            # Note that the 'a_identity' attribute, that gives the
            # identity of the aggregating axis, is set in
            # _group_fields().
            # --------------------------------------------------------
            grouped_meta = _group_fields(meta, axis)

            if not grouped_meta:
                if is_log_level_info(logger):
                    logger.info(
                        f"Unaggregatable {meta[0].field.identity()!r} fields "
                        f"have{exclude} been output: {meta[0].message}"
                    )

                unaggregatable = True
                break

            if len(grouped_meta) == number_of_fields:
                if is_log_level_debug(logger):
                    logger.debug(
                        f"{meta[0].field.identity()!r} fields can't be "
                        f"aggregated along their {axis!r} axis"
                    )
                continue

            # --------------------------------------------------------
            # Within each group, aggregate as many fields as possible.
            # --------------------------------------------------------
            for m in grouped_meta:

                if len(m) == 1:
                    continue

                # ----------------------------------------------------
                # Still here? The sort the fields in place by the
                # canonical first values of their 1-d coordinates for
                # the aggregating axis.
                # ----------------------------------------------------
                _sorted_by_first_values(m, axis)

                # ----------------------------------------------------
                # Check that the aggregating axis's 1-d coordinates
                # don't overlap, and don't aggregate anything in this
                # group if any do.
                # ----------------------------------------------------
                if not _ok_coordinate_arrays(
                    m, axis, overlap, contiguous, verbose
                ):
                    if is_log_level_info(logger):
                        logger.info(
                            f"Unaggregatable {m[0].field.identity()!r} fields "
                            f"have{exclude} been output: {m[0].message}"
                        )

                    unaggregatable = True
                    break

                # ----------------------------------------------------
                # Still here? Then pass through the fields
                # ----------------------------------------------------
                m0 = m[0].copy()

                for m1 in m[1:]:
                    m0 = _aggregate_2_fields(
                        m0,
                        m1,
                        rtol=rtol,
                        atol=atol,
                        verbose=verbose,
                        concatenate=concatenate,
                        copy=(copy or not exclude),
                    )

                    if not m0:
                        # Couldn't aggregate these two fields, so
                        # abandon all aggregations on the fields with
                        # this structural signature, including those
                        # already done.
                        if is_log_level_info(logger):
                            logger.info(
                                f"Unaggregatable {m1.field.identity()!r} "
                                f"fields have{exclude} been output: "
                                f"{m1.message}"
                            )

                        unaggregatable = True
                        break

                m[:] = [m0]

            if unaggregatable:
                break

            # --------------------------------------------------------
            # Still here? Then the aggregation along this axis was
            # completely successful for each sub-group, so reassemble
            # the aggregated fields as a single list ready for
            # aggregation along the next axis.
            # --------------------------------------------------------
            meta = [m for gm in grouped_meta for m in gm]

        # Add fields to the output list
        if unaggregatable:
            status = 1
            if not exclude:
                if copy:
                    output_constructs.extend((m.field.copy() for m in meta0))
                else:
                    output_constructs.extend((m.field for m in meta0))
        else:
            output_constructs.extend((m.field for m in meta))

    aggregate.status = status

    if status:
        logger.info("")

    Type = "field"
    if output_constructs:
        Type = output_constructs[0].construct_type
        for x in output_constructs[1:]:
            if x.construct_type != Type:
                raise ValueError(
                    "Can't aggregate a mixture of field and domain constructs"
                )

    if Type == "field":
        output_constructs = FieldList(output_constructs)

    return output_constructs


# --------------------------------------------------------------------
# Initialise the status
# --------------------------------------------------------------------
aggregate.status = 0


def _create_hash_and_first_values(
    meta, axes, donotchecknonaggregatingaxes, hfl_cache, rtol, atol
):
    """Updates each field's _Meta object.

    :Parameters:

        meta: `list` of `_Meta`

        axes: `None` or `list`

        donotchecknonaggregatingaxes: `bool`

    :Returns:

        `None`

    """
    for m in meta:
        field = m.field

        item_axes = m.field.constructs.data_axes()

        m_sort_keys = m.sort_keys
        m_sort_indices = m.sort_indices

        m_hash_values = m.hash_values
        m_first_values = m.first_values
        m_last_values = m.last_values

        m_id_to_axis = m.id_to_axis

        # --------------------------------------------------------
        # Create a hash value for each metadata array
        # --------------------------------------------------------

        # --------------------------------------------------------
        # 1-d coordinates
        # --------------------------------------------------------
        for identity in m.axis_ids:
            if (
                axes is not None
                and donotchecknonaggregatingaxes
                and identity not in axes
            ):
                x = [None] * len(m.axis[identity]["keys"])
                m_hash_values[identity] = x
                m_first_values[identity] = x[:]
                m_last_values[identity] = x[:]
                continue

            # Still here?
            m_axis_identity = m.axis[identity]
            axis = m_id_to_axis[identity]

            # If this axis has no 1-d coordinates and is defined only
            # by its netCDF dimension name and its size, then hash the
            # domain axis object
            axis_size = m_axis_identity["size"]
            if axis_size is not None:
                m_hash_values[identity] = [hash(field.constructs[axis])]
                m_first_values[identity] = [None]
                m_last_values[identity] = [None]
                m_sort_indices[axis] = slice(None)
                continue

            # Still here?
            dim_coord = m.field.dimension_coordinate(
                filter_by_axis=(axis,), default=None
            )

            # Find the sort indices for this axis ...
            if dim_coord is not None:
                # ... which has a dimension coordinate
                m_sort_keys[axis] = axis
                if not field.direction(axis):
                    # Axis is decreasing
                    sort_indices = slice(None, None, -1)
                    null_sort = False
                else:
                    # Axis is increasing
                    sort_indices = slice(None)
                    null_sort = True
            else:
                # ... or which doesn't have a dimension coordinate but
                # does have one or more 1-d auxiliary coordinates
                aux = m_axis_identity["keys"][0]
                sort_indices = numpy_argsort(field.constructs[aux].array)
                m_sort_keys[axis] = aux
                null_sort = False

            m_sort_indices[axis] = sort_indices

            hash_values = []
            first_values = []
            last_values = []

            for key, canonical_units in zip(
                m_axis_identity["keys"], m_axis_identity["units"]
            ):
                coord = field.constructs[key]

                # Get the hash of the data array and its first and
                # last values
                h, first, last = _get_hfl(
                    coord,
                    canonical_units,
                    sort_indices,
                    null_sort,
                    True,
                    False,
                    hfl_cache,
                    rtol,
                    atol,
                )

                first_values.append(first)
                last_values.append(last)

                if coord.has_bounds():
                    if coord.construct_type == "dimension_coordinate":
                        # Get the hash of the dimension coordinate
                        # bounds data array and its first and last
                        # cell values
                        hb, fb, lb = _get_hfl(
                            coord.bounds,
                            canonical_units,
                            sort_indices,
                            null_sort,
                            False,
                            True,
                            hfl_cache,
                            rtol,
                            atol,
                        )
                        m.first_bounds[identity] = fb
                        m.last_bounds[identity] = lb
                    else:
                        # Get the hash of the auxiliary coordinate
                        # bounds data array
                        hb = _get_hfl(
                            coord.bounds,
                            canonical_units,
                            sort_indices,
                            null_sort,
                            False,
                            False,
                            hfl_cache,
                            rtol,
                            atol,
                        )

                    h = (h, hb)
                else:
                    h = (h,)

                hash_values.append(h)
            #                else:
            #                    coord_units = coord.Units
            #
            #                    # Change the coordinate data type if required
            #                    if coord.dtype.char not in ('d', 'S'):
            #                        coord = coord.copy(_only_Data=True)
            #                        coord.dtype = _dtype_float
            #
            #                    # Change the coordinate's units to the canonical ones
            #                    coord.Units = canonical_units
            #
            #                    # Get the coordinate's data array
            #                    if null_sort:
            #                        array = coord.Data.array
            #                    else:
            #                        array = coord.Data.array[sort_indices]
            #
            #                    hash_value = hash_array(array)
            #
            #                    first_values.append(array.item(0)) #[0])
            #                    last_values.append(array.item(-1)) #[-1])
            #
            #                    if coord._hasbounds:
            #                        if null_sort:
            #                            array = coord.bounds.Data.array
            #                        else:
            #                            array = coord.bounds.Data.array[sort_indices, ...]
            #
            #                        hash_value = (hash_value, hash_array(array))
            #
            #                        if key[:3] == 'dim':  # can do better than this! DCH
            #                            # Record the bounds of the first and last
            #                            # (sorted) cells of a dimension coordinate
            #                            # (don't need to do this for an auxiliary
            #                            # coordinate).
            #                            array0 = array[0, ...].copy()
            #                            array0.sort()
            #                            m.first_bounds[identity] = array0
            #
            #                            array0 = array[-1, ...].copy()
            #                            array0.sort()
            #                            m.last_bounds[identity] = array0
            #
            #                    hash_values.append(hash_value)
            #
            #                    # Reinstate the coordinate's original units
            #                    coord.Units = coord_units

            m_hash_values[identity] = hash_values
            m_first_values[identity] = first_values
            m_last_values[identity] = last_values

        # ------------------------------------------------------------
        # N-d auxiliary coordinates
        # ------------------------------------------------------------
        if donotchecknonaggregatingaxes:
            for aux in m.nd_aux.values():
                aux["hash_value"] = (None,)
        else:
            for aux in m.nd_aux.values():
                key = aux["key"]
                canonical_units = aux["units"]

                coord = field.construct(
                    key
                )  # TODO why not field.constructs[key]?

                axes = [m_id_to_axis[identity] for identity in aux["axes"]]
                domain_axes = item_axes[key]
                if axes != domain_axes:
                    coord = coord.copy()  # _only_Data=True)
                    iaxes = [domain_axes.index(axis) for axis in axes]
                    coord.transpose(iaxes, inplace=True)

                sort_indices = tuple([m_sort_indices[axis] for axis in axes])

                # Get the hash of the data array
                h = _get_hfl(
                    coord,
                    canonical_units,
                    sort_indices,
                    False,
                    False,
                    False,
                    hfl_cache,
                    rtol,
                    atol,
                )

                if coord.has_bounds():
                    # Get the hash of the bounds data array
                    hb = _get_hfl(
                        coord.bounds,
                        canonical_units,
                        sort_indices,
                        False,
                        False,
                        False,
                        hfl_cache,
                        rtol,
                        atol,
                    )
                    h = (h, hb)
                else:
                    h = (h,)

                aux["hash_value"] = h

        # ------------------------------------------------------------
        # Cell measures
        # ------------------------------------------------------------
        if donotchecknonaggregatingaxes:
            for msr in m.msr.values():
                msr["hash_values"] = [(None,) * len(msr["keys"])]
        else:
            for canonical_units, msr in m.msr.items():
                hash_values = []
                for key, axes in zip(msr["keys"], msr["axes"]):
                    coord = field.constructs[key]

                    axes = tuple([m_id_to_axis[identity] for identity in axes])

                    domain_axes = item_axes[key]
                    if axes != domain_axes:
                        coord = coord.copy()  # _only_Data=True)  # TODO
                        iaxes = [domain_axes.index(axis) for axis in axes]
                        coord.transpose(iaxes, inplace=True)

                    sort_indices = [m_sort_indices[axis] for axis in axes]

                    # Get the hash of the data array
                    h = _get_hfl(
                        coord,
                        canonical_units,
                        tuple(sort_indices),
                        False,
                        False,
                        False,
                        hfl_cache,
                        rtol,
                        atol,
                    )

                    hash_values.append((h,))

                msr["hash_values"] = hash_values

        # ------------------------------------------------------------
        # Field ancillaries
        # ------------------------------------------------------------
        if donotchecknonaggregatingaxes:
            for anc in m.field_anc.values():
                anc["hash_value"] = (None,)
        else:
            for anc in m.field_anc.values():
                key = anc["key"]
                canonical_units = anc["units"]

                field_anc = field.construct(key)

                axes = tuple(
                    [m_id_to_axis[identity] for identity in anc["axes"]]
                )
                domain_axes = item_axes[key]
                if axes != domain_axes:
                    field_anc = field_anc.copy()  # _only_Data=True)  # TODO
                    iaxes = [domain_axes.index(axis) for axis in axes]
                    field_anc.transpose(iaxes, inplace=True)

                sort_indices = tuple([m_sort_indices[axis] for axis in axes])

                # Get the hash of the data array
                h = _get_hfl(
                    field_anc,
                    canonical_units,
                    sort_indices,
                    False,
                    False,
                    False,
                    hfl_cache,
                    rtol,
                    atol,
                )

                anc["hash_value"] = (h,)

        # ------------------------------------------------------------
        # Domain ancillaries
        # ------------------------------------------------------------
        if donotchecknonaggregatingaxes:
            for anc in m.domain_anc.values():
                anc["hash_value"] = (None,)
        else:
            for anc in m.domain_anc.values():
                key = anc["key"]
                canonical_units = anc["units"]

                domain_anc = field.construct(key)

                axes = tuple(
                    [m_id_to_axis[identity] for identity in anc["axes"]]
                )
                domain_axes = item_axes[key]
                if axes != domain_axes:
                    domain_anc = domain_anc.copy()  # _only_Data=True)  # TODO
                    iaxes = [domain_axes.index(axis) for axis in axes]
                    domain_anc.transpose(iaxes, inplace=True)

                sort_indices = tuple([m_sort_indices[axis] for axis in axes])

                # Get the hash of the data array
                h = _get_hfl(
                    domain_anc,
                    canonical_units,
                    sort_indices,
                    null_sort=False,
                    first_and_last_values=False,
                    first_and_last_bounds=False,
                    hfl_cache=hfl_cache,
                    rtol=rtol,
                    atol=atol,
                )

                if domain_anc.has_bounds():
                    # Get the hash of the bounds data array
                    hb = _get_hfl(
                        domain_anc.bounds,
                        canonical_units,
                        sort_indices,
                        null_sort=False,
                        first_and_last_values=False,
                        first_and_last_bounds=False,
                        hfl_cache=hfl_cache,
                        rtol=rtol,
                        atol=atol,
                    )
                    h = (h, hb)
                else:
                    h = (h,)

                anc["hash_value"] = h

        #            for anc in m.domain_anc.values():
        #                key             = anc['key']
        #                canonical_units = anc['units']
        #
        #                field_anc = field.item(key)
        #
        #                axes = tuple(
        #                    [m_id_to_axis[identity] for identity in anc['axes']])
        #                domain_axes = item_axes[key]
        #                if axes != domain_axes:
        #                    field_anc = field_anc.copy()  #_only_Data=True)  # TODO
        #                    iaxes = [domain_axes.index(axis) for axis in axes]
        #                    field_anc.transpose(iaxes, inplace=True)
        #
        #                sort_indices = tuple([m_sort_indices[axis] for axis in axes])
        #
        #                # Get the hash of the data array
        #                h = _get_hfl(field_anc, canonical_units, sort_indices,
        #                             False, False, False, hfl_cache, rtol, atol)
        #
        #                anc['hash_value'] = h

        m.cell_values = True


def _get_hfl(
    v,
    canonical_units,
    sort_indices,
    null_sort,
    first_and_last_values,
    first_and_last_bounds,
    hfl_cache,
    rtol,
    atol,
):
    """Return the hash value, and optionally first and last values or
    bounds.

    :Parameters:

        v: Metadata construct

    :Returns:

        `int` or 3-`tuple`
            Hash value for the coordinates and cell measures, in a tuple with
            the first and last cell values or bounds if either is requested.

    """
    create_hash = True
    create_fl = first_and_last_values
    create_flb = first_and_last_bounds

    key = None

    d = v.get_data(None)
    if d is None:
        if create_fl or create_flb:
            return None, None, None

        return

    #    if d._pmsize == 1:
    #        partition = d.partitions.matrix.item()
    #        if not partition.part:
    #            key = getattr(partition.subarray, "file_address", None)
    #            if key is not None:
    #                hash_value = hfl_cache.hash.get(key, None)
    #                create_hash = hash_value is None
    #
    #                if first_and_last_values:
    #                    first, last = hfl_cache.fl.get(key, (None, None))
    #                    create_fl = first is None
    #
    #                if first_and_last_bounds:
    #                    first, last = hfl_cache.flb.get(key, (None, None))
    #                    create_flb = first is None

    if create_hash or create_fl or create_flb:
        # Change the data type if required
        if d.dtype.char not in ("d", "S", "U"):
            d = d.copy()
            d.dtype = _dtype_float

        # Change the units to the canonical ones
        units = d.Units
        d.Units = canonical_units

        # Get the data array
        if null_sort:
            array = d.array
        else:
            array = d.array[sort_indices]

        # Reinstate the original units
        d.Units = units

        if create_hash:
            hash_value = hash_array(array)

            if hash_value not in hfl_cache.hash_to_array:
                # Compare arrays, overriding hash value
                found_close = False
                for hash_value0, array0 in hfl_cache.hash_to_array.items():

                    if array0.shape != array.shape:
                        continue

                    if array0.shape != array.shape:
                        continue

                    if _numpy_allclose(array0, array, rtol=rtol, atol=atol):
                        hash_value = hash_value0
                        found_close = True
                        break

                if not found_close:
                    hfl_cache.hash_to_array[hash_value] = array
            else:
                pass

            hfl_cache.hash[key] = hash_value

        if create_fl:
            first = array.item(0)
            last = array.item(-1)
            hfl_cache.fl[key] = (first, last)

        if create_flb:
            # Record the bounds of the first and last (sorted) cells
            first = numpy_sort(array[0, ...])
            last = numpy_sort(array[-1, ...])
            hfl_cache.flb[key] = (first, last)

    if first_and_last_values or first_and_last_bounds:
        return hash_value, first, last
    else:
        return hash_value


def _group_fields(meta, axis):
    """Return a FieldList of the potentially aggregatable fields.

    :Parameters:

        meta: `list` of `_Meta`

        axis: `str`
            The name of the axis to group for aggregation.

    :Returns:

        `list` of cf.FieldList

    """
    axes = meta[0].axis_ids

    if axes:
        if axis in axes:
            # Move axis to the end of the axes list
            axes = axes[:]
            axes.remove(axis)
            axes.append(axis)

        sort_by_axis_ids = itemgetter(*axes)

        def _hash_values(m):
            return sort_by_axis_ids(m.hash_values)

        meta.sort(key=_hash_values)

    # Create a new group of potentially aggregatable fields (which
    # contains the first field in the sorted list)
    m0 = meta[0]
    groups_of_fields = [[m0]]

    hash0 = m0.hash_values

    for m0, m1 in zip(meta[:-1], meta[1:]):

        # -------------------------------------------------------------
        # Count the number of axes which are different between the two
        # fields
        # -------------------------------------------------------------
        count = 0
        hash1 = m1.hash_values
        for identity, value in hash0.items():
            if value != hash1[identity]:
                count += 1
                a_identity = identity

        hash0 = hash1

        if count == 1:
            # --------------------------------------------------------
            # Exactly one axis has different 1-d coordinate values
            # --------------------------------------------------------
            if a_identity != axis:
                # But it's not the axis that we're trying currently to
                # aggregate over
                groups_of_fields.append([m1])
                continue

            # Still here? Then it is the axis that we're trying
            # currently to aggregate over.
            ok = True

            # Check the N-d auxiliary coordinates
            for identity, aux0 in m0.nd_aux.items():
                if (
                    a_identity not in aux0["axes"]
                    and aux0["hash_value"] != m1.nd_aux[identity]["hash_value"]
                ):
                    # This matching pair of N-d auxiliary coordinates
                    # does not span the aggregating axis and they have
                    # different data array values
                    ok = False
                    break

            if not ok:
                groups_of_fields.append([m1])
                continue

            # Still here? Then check the cell measures
            msr0 = m0.msr
            for units in msr0:
                for axes, hash_value0, hash_value1 in zip(
                    msr0[units]["axes"],
                    msr0[units]["hash_values"],
                    m1.msr[units]["hash_values"],
                ):

                    if a_identity not in axes and hash_value0 != hash_value1:
                        # There is a matching pair of cell measures
                        # with these units which does not span the
                        # aggregating axis and they have different
                        # data array values
                        ok = False
                        break

            if not ok:
                groups_of_fields.append([m1])
                continue

            # Still here? Then set the identity of the aggregating
            # axis
            m0.a_identity = a_identity
            m1.a_identity = a_identity

            # Append parent1 to this group of potentially aggregatable
            # fields
            groups_of_fields[-1].append(m1)

        elif not count:
            # --------------------------------------------------------
            # Zero axes have different 1-d coordinate values, so don't
            # aggregate anything in this entire group.
            # --------------------------------------------------------
            meta[
                0
            ].message = "Some fields have identical sets of 1-d coordinates."
            return ()

        else:
            # --------------------------------------------------------
            # Two or more axes have different 1-d coordinate values,
            # so create a new sub-group of potentially aggregatable
            # fields which contains parent1.
            # --------------------------------------------------------
            groups_of_fields.append([m1])

    return groups_of_fields


def _sorted_by_first_values(meta, axis):
    """Sort fields inplace.

    :Parameters:

        meta: `list` of `_Meta`

        axis: `str`

    :Returns:

        `None`

    """
    sort_by_axis_ids = itemgetter(axis)

    def _first_values(m):
        return sort_by_axis_ids(m.first_values)

    meta.sort(key=_first_values)


@_manage_log_level_via_verbosity
def _ok_coordinate_arrays(meta, axis, overlap, contiguous, verbose=None):
    """Return True if the aggregating 1-d coordinates of the aggregating
    axis are all aggregatable.

    It is assumed that the input metadata objects have already been
    sorted by the canonical first values of their 1-d coordinates.

    :Parameters:

        meta: `list` of `_Meta`

        axis: `str`
            Find the canonical identity of the aggregating axis.

        overlap: `bool`
            See the `cf.aggregate` function for details.

        contiguous: `bool`
            See the `cf.aggregate` function for details.

        verbose: `int` or `str` or `None`, optional
            See the `cf.aggregate` function for details.

    :Returns:

        `bool`

    **Examples**

    >>> if not _ok_coordinate_arrays(meta, 'latitude', True, False):
    ...     print("Don't aggregate")

    """
    m = meta[0]

    dim_coord_index = m.axis[axis]["dim_coord_index"]

    if dim_coord_index is not None:
        # ------------------------------------------------------------
        # The aggregating axis has a dimension coordinate
        # ------------------------------------------------------------
        # Check for overlapping dimension coordinate cell centres
        dim_coord_index0 = dim_coord_index

        #  TODO 2019-06-21 check that the coords are all increasing, by now?
        for m0, m1 in zip(meta[:-1], meta[1:]):
            dim_coord_index1 = m1.axis[axis]["dim_coord_index"]
            if (
                m0.last_values[axis][dim_coord_index0]
                >= m1.first_values[axis][dim_coord_index1]
            ):
                # Found overlap
                meta[0].message = (
                    f"{m.axis[axis]['ids'][dim_coord_index]!r} "
                    "dimension coordinate ranges overlap: "
                    f"[{m0.first_values[axis][dim_coord_index0]}, "
                    f"{m0.last_values[axis][dim_coord_index0]}], "
                    f"[{m1.first_values[axis][dim_coord_index1]}, "
                    f"{m1.last_values[axis][dim_coord_index1]}]"
                )

                return False

            dim_coord_index0 = dim_coord_index1

        if axis in m.first_bounds:
            # --------------------------------------------------------
            # The dimension coordinates have bounds
            # --------------------------------------------------------
            if not overlap:
                for m0, m1 in zip(meta[:-1], meta[1:]):
                    if m1.first_bounds[axis][0] < m0.last_bounds[axis][1]:
                        # Do not aggregate anything in this group
                        # because overlapping has been disallowed and
                        # the first cell from field1 overlaps with the
                        # last cell from field0.
                        meta[0].message = (
                            f"overlap={m.axis[axis]['ids'][dim_coord_index]} "
                            f"and {overlap!r} dimension coordinate bounds "
                            f"values overlap ({m1.first_bounds[axis][0]} "
                            f"< {m0.last_bounds[axis][1]})"
                        )

                        return

            if contiguous:
                for m0, m1 in zip(meta[:-1], meta[1:]):
                    if m0.last_bounds[axis][1] < m1.first_bounds[axis][0]:
                        # Do not aggregate anything in this group
                        # because contiguous coordinates have been
                        # specified and the first cell from parent1 is
                        # not contiguous with the last cell from
                        # parent0.
                        meta[0].message = (
                            "contiguous="
                            f"{m.axis[axis]['ids'][dim_coord_index]} and "
                            f"{contiguous!r} dimension coordinate cells are "
                            f"not contiguous ({m0.last_bounds[axis][1]} < "
                            f"{m1.first_bounds[axis][0]})"
                        )
                        return

    else:
        # ------------------------------------------------------------
        # The aggregating axis does not have a dimension coordinate,
        # but it does have at least one 1-d auxiliary coordinate.
        # ------------------------------------------------------------
        # Check for duplicate auxiliary coordinate values
        for i, identity in enumerate(meta[0].axis[axis]["ids"]):
            set_of_1d_aux_coord_values = set()
            number_of_1d_aux_coord_values = 0
            for m in meta:
                aux = m.axis[axis]["keys"][i]
                array = m.field.constructs[aux].array
                set_of_1d_aux_coord_values.update(array)
                number_of_1d_aux_coord_values += array.size
                if (
                    len(set_of_1d_aux_coord_values)
                    != number_of_1d_aux_coord_values
                ):
                    meta[0].message = (
                        f"no {identity!r} dimension coordinates and "
                        f"{identity!r} auxiliary coordinates have duplicate "
                        "values"
                    )

                    return

    # ----------------------------------------------------------------
    # Still here? Then the aggregating axis does not overlap between
    # any of the fields.
    # ----------------------------------------------------------------
    return True


@_manage_log_level_via_verbosity
def _aggregate_2_fields(
    m0, m1, rtol=None, atol=None, verbose=None, concatenate=True, copy=True
):
    """Aggregate two fields, returning the _Meta object of the
    aggregated field.

    :Parameters:

        m0: `_Meta`

        m1: `_Meta`

        rtol: `float`, optional
            See the `cf.aggregate` function for details.

        atol: `float`, optional
            See the `cf.aggregate` function for details.

        verbose: `int` or `str` or `None`, optional
            See the `cf.aggregate` function for details.

    :Returns:

        out: `_Meta` or `bool`

    """
    a_identity = m0.a_identity

    parent0 = m0.field
    parent1 = m1.field
    if copy:
        parent1 = parent1.copy()

    # ----------------------------------------------------------------
    # Map the axes of parent1 to those of parent0
    # ----------------------------------------------------------------
    dim1_name_map = {
        m1.id_to_axis[identity]: m0.id_to_axis[identity]
        for identity in m0.axis_ids
    }

    dim0_name_map = {axis0: axis1 for axis1, axis0 in dim1_name_map.items()}

    # ----------------------------------------------------------------
    # In each field, find the identifier of the aggregating axis.
    # ----------------------------------------------------------------
    adim0 = m0.id_to_axis[a_identity]
    adim1 = m1.id_to_axis[a_identity]

    # ----------------------------------------------------------------
    # Make sure that, along the aggregating axis, parent1 runs in the
    # same direction as parent0
    # ----------------------------------------------------------------
    direction0 = parent0.direction(adim0)
    if parent1.direction(adim1) != direction0:
        parent1.flip(adim1, inplace=True)

    # ----------------------------------------------------------------
    # Find matching pairs of coordinates and cell measures which span
    # the aggregating axis
    # ----------------------------------------------------------------
    # 1-d coordinates
    spanning_variables = [
        (key0, key1, parent0.constructs[key0], parent1.constructs[key1])
        for key0, key1 in zip(
            m0.axis[a_identity]["keys"], m1.axis[a_identity]["keys"]
        )
    ]

    hash_values0 = m0.hash_values[a_identity]
    hash_values1 = m1.hash_values[a_identity]

    for i, (hash0, hash1) in enumerate(zip(hash_values0, hash_values1)):
        # try:
        #     hash_values0[i].append(hash_values1[i])
        # except AttributeError:
        #     hash_values0[i] = [hash_values0[i], hash_values1[i]]
        hash_values0[i] = hash_values0[i] + hash_values1[i]

    # N-d auxiliary coordinates
    for identity in m0.nd_aux:
        aux0 = m0.nd_aux[identity]
        aux1 = m1.nd_aux[identity]
        if a_identity in aux0["axes"]:
            key0 = aux0["key"]
            key1 = aux1["key"]
            spanning_variables.append(
                (
                    key0,
                    key1,
                    parent0.constructs[key0],
                    parent1.constructs[key1],
                )
            )

            hash_value0 = aux0["hash_value"]
            hash_value1 = aux1["hash_value"]
            aux0["hash_value"] = hash_value0 + hash_value1

    # Cell measures
    for units in m0.msr:
        hash_values0 = m0.msr[units]["hash_values"]
        hash_values1 = m1.msr[units]["hash_values"]
        for i, (axes, key0, key1) in enumerate(
            zip(
                m0.msr[units]["axes"],
                m0.msr[units]["keys"],
                m1.msr[units]["keys"],
            )
        ):
            if a_identity in axes:
                spanning_variables.append(
                    (
                        key0,
                        key1,
                        parent0.constructs[key0],
                        parent1.constructs[key1],
                    )
                )

                hash_values0[i] = hash_values0[i] + hash_values1[i]

    # Field ancillaries
    for identity in m0.field_anc:
        anc0 = m0.field_anc[identity]
        anc1 = m1.field_anc[identity]
        if a_identity in anc0["axes"]:
            key0 = anc0["key"]
            key1 = anc1["key"]
            spanning_variables.append(
                (
                    key0,
                    key1,
                    parent0.constructs[key0],
                    parent1.constructs[key1],
                )
            )

            hash_value0 = anc0["hash_value"]
            hash_value1 = anc1["hash_value"]
            anc0["hash_value"] = hash_value0 + hash_value1

    # Domain ancillaries
    for identity in m0.domain_anc:
        anc0 = m0.domain_anc[identity]
        anc1 = m1.domain_anc[identity]
        if a_identity in anc0["axes"]:
            key0 = anc0["key"]
            key1 = anc1["key"]
            spanning_variables.append(
                (
                    key0,
                    key1,
                    parent0.constructs[key0],
                    parent1.constructs[key1],
                )
            )

            hash_value0 = anc0["hash_value"]
            hash_value1 = anc1["hash_value"]
            anc0["hash_value"] = hash_value0 + hash_value1

    # ----------------------------------------------------------------
    # For each matching pair of coordinates, cell measures, field and
    # domain ancillaries which span the aggregating axis, insert the
    # one from parent1 into the one from parent0
    # ----------------------------------------------------------------
    for key0, key1, construct0, construct1 in spanning_variables:
        construct_axes0 = parent0.get_data_axes(key0)
        construct_axes1 = parent1.get_data_axes(key1)

        # Ensure that the axis orders are the same in both constructs
        iaxes = [
            construct_axes1.index(dim0_name_map[axis0])
            for axis0 in construct_axes0
        ]
        #        iaxes = [axes1.index(dim0_name_map[axis0]) for axis0 in axes0]
        construct1.transpose(iaxes, inplace=True)

        # Find the position of the concatenating axis
        # axis = axes0.index(adim0)
        axis = construct_axes0.index(adim0)

        if direction0:
            # The fields are increasing along the aggregating axis
            data = Data.concatenate(
                (construct0.get_data(), construct1.get_data()),
                axis,
            )
            construct0.set_data(data, copy=False)
            if construct0.has_bounds():
                data = Data.concatenate(
                    (
                        construct0.bounds.get_data(_fill_value=False),
                        construct1.bounds.get_data(_fill_value=False),
                    ),
                    axis,
                )
                construct0.bounds.set_data(data, copy=False)
        else:
            # The fields are decreasing along the aggregating axis
            data = Data.concatenate(
                (
                    construct1.get_data(_fill_value=False),
                    construct0.get_data(_fill_value=False),
                ),
                axis,
            )
            construct0.set_data(data)
            if construct0.has_bounds():
                data = Data.concatenate(
                    (
                        construct1.bounds.get_data(_fill_value=False),
                        construct0.bounds.get_data(_fill_value=False),
                    ),
                    axis,
                )
                construct0.bounds.set_data(data)

    # ----------------------------------------------------------------
    # Update the size of the aggregating axis in the output parent
    # construct
    # ----------------------------------------------------------------
    if m0.has_data:
        # ----------------------------------------------------------------
        # Insert the data array from parent1 into the data array of
        # parent0
        # ----------------------------------------------------------------
        data_axes0 = list(parent0.get_data_axes())
        data_axes1 = list(parent1.get_data_axes())

        # Ensure that both data arrays span the same axes, including
        # the aggregating axis.
        for axis1 in data_axes1:
            axis0 = dim1_name_map[axis1]
            if axis0 not in data_axes0:
                parent0.insert_dimension(axis0, position=0, inplace=True)
                data_axes0.insert(0, axis0)

        for axis0 in data_axes0:
            axis1 = dim0_name_map[axis0]
            if axis1 not in data_axes1:
                parent1.insert_dimension(axis1, position=0, inplace=True)

        # Find the position of the concatenating axis
        if adim0 not in data_axes0:
            # Insert the aggregating axis at position 0 because is not
            # already spanned by either data arrays
            parent0.insert_dimension(adim0, position=0, inplace=True)
            parent1.insert_dimension(adim1, position=0, inplace=True)
            axis = 0
        else:
            axis = data_axes0.index(adim0)

        # Get the data axes again, in case we've inserted new dimensions
        data_axes0 = parent0.get_data_axes()
        data_axes1 = parent1.get_data_axes()

        # Ensure that the axis orders are the same in both fields
        transpose_axes1 = [dim0_name_map[axis0] for axis0 in data_axes0]
        if transpose_axes1 != data_axes1:
            parent1.transpose(transpose_axes1, inplace=True)

        if direction0:
            # The fields are increasing along the aggregating axis
            data = Data.concatenate(
                (parent0.get_data(), parent1.get_data()), axis
            )
        else:
            # The fields are decreasing along the aggregating axis
            data = Data.concatenate(
                (parent1.get_data(), parent0.get_data()), axis
            )

        # Update the size of the aggregating axis in parent0
        domain_axis = parent0.constructs[adim0]
        domain_axis += parent1.constructs[adim1].get_size()

        # Insert the concatentated data into the field
        parent0.set_data(data, set_axes=False, copy=False)
    else:
        domain_axis = parent0.constructs[adim0]
        domain_axis += parent1.constructs[adim1].get_size()

    # Make sure that parent0 has a standard_name, if possible.
    if getattr(parent0, "id", None) is not None:
        standard_name = parent1.get_property("standard_name", None)
        if standard_name is not None:
            parent0.set_property("standard_name", standard_name)
            del parent0.id

    # -----------------------------------------------------------------
    # Update the properties in parent0
    # -----------------------------------------------------------------
    for prop in set(parent0.properties()).difference(
        parent0._special_properties
    ):
        value0 = parent0.get_property(prop, None)
        value1 = parent1.get_property(prop, None)

        if prop in ("valid_min", "valid_max", "valid_range"):
            if not m0.respect_valid:
                parent0.del_property(prop, None)

            continue

        if prop in ("_FillValue", "missing_value"):
            continue

        # Still here?
        if parent0._equals(value0, value1):
            continue

        if concatenate:
            if value1 is not None:
                if value0 is not None:
                    parent0.set_property(
                        prop, "%s :AGGREGATED: %s" % (value0, value1)
                    )
                else:
                    parent0.set_property(prop, " :AGGREGATED: %s" % value1)
        else:
            if value0 is not None:
                parent0.del_property(prop)

    # Make a note that the parent construct in this _Meta object has
    # already been aggregated

    m0.aggregated_field = True

    # ----------------------------------------------------------------
    # Return the _Meta object containing the aggregated parent
    # construct
    # ----------------------------------------------------------------
    return m0
