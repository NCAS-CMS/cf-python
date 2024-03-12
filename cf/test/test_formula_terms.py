import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


def _formula_terms(standard_name):
    """Return a field construct with a vertical CRS, its computed non-
    parametric coordinates, and the computed standard name."""
    # field: air_temperature
    field = cf.Field()
    field.set_properties({"standard_name": "air_temperature", "units": "K"})
    data = cf.Data([0, 1, 2], units="K", dtype="f8")

    # domain_axis: Z
    c = cf.DomainAxis()
    c.set_size(3)
    c.nc_set_dimension("z")
    axisZ = field.set_construct(c, key="domainaxis1", copy=False)

    field.set_data(data)

    # coordinate_reference:
    coordref = cf.CoordinateReference()
    coordref.coordinate_conversion.set_parameter(
        "standard_name", standard_name
    )

    aux = cf.AuxiliaryCoordinate()
    aux.long_name = (
        f"Computed from parametric {standard_name} vertical coordinates"
    )

    if standard_name == "atmosphere_ln_pressure_coordinate":
        computed_standard_name = "air_pressure"

        # Computed vertical corodinates
        aux.standard_name = computed_standard_name
        data = cf.Data([700, 500, 300], "hPa", dtype="f8")
        aux.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[800, 600], [600, 400], [400, 200]], "hPa", dtype="f8")
        bounds.set_data(data)
        aux.set_bounds(bounds)

        # domain_ancillary: p0
        p0 = cf.DomainAncillary()
        p0.standard_name = (
            "reference_air_pressure_for_atmosphere_vertical_coordinate"
        )
        data = cf.Data(1000.0, units="hPa", dtype="f8")
        p0.set_data(data)
        p0_key = field.set_construct(p0, axes=(), copy=False)

        # domain_ancillary: Z
        lev = cf.DomainAncillary()
        lev.standard_name = standard_name
        data = -(aux.data / p0.data).log()
        lev.set_data(data)
        bounds = cf.Bounds()
        data = -(aux.bounds.data / p0.data).log()
        bounds.set_data(data)
        lev.set_bounds(bounds)
        lev_key = field.set_construct(lev, axes=axisZ, copy=False)

        # dimension_coordinate: Z
        levc = cf.DimensionCoordinate(source=lev)
        levc_key = field.set_construct(levc, axes=axisZ, copy=False)

        # coordinate_reference:
        coordref.set_coordinates({levc_key})
        coordref.coordinate_conversion.set_domain_ancillaries(
            {"p0": p0_key, "lev": lev_key}
        )
        field.set_construct(coordref)

    elif standard_name == "atmosphere_sigma_coordinate":
        computed_standard_name = "air_pressure"

        # Computed vertical corodinates
        aux.standard_name = computed_standard_name
        data = cf.Data([700, 500, 300], "hPa", dtype="f8")
        aux.set_data(data)
        b = cf.Bounds()
        data = cf.Data([[800, 600], [600, 400], [400, 200]], "hPa", dtype="f8")
        b.set_data(data)
        aux.set_bounds(b)

        # domain_ancillary: ps
        ps = cf.DomainAncillary()
        ps.standard_name = "surface_air_pressure"
        data = cf.Data(1000, units="hPa", dtype="f8")
        ps.set_data(data)
        ps_key = field.set_construct(ps, axes=(), copy=False)

        # domain_ancillary: ptop
        ptop = cf.DomainAncillary()
        ptop.standard_name = "air_pressure_at_top_of_atmosphere_model"
        data = cf.Data(10, units="hPa", dtype="f8")
        ptop.set_data(data)
        ptop_key = field.set_construct(ptop, axes=(), copy=False)

        # domain_ancillary: sigma
        sigma = cf.DomainAncillary()
        sigma.standard_name = standard_name
        data = cf.Data([0.6969697, 0.49494949, 0.29292929])
        sigma.set_data(data)
        b = cf.Bounds()
        data = cf.Data(
            [
                [0.7979798, 0.5959596],
                [0.5959596, 0.39393939],
                [0.39393939, 0.19191919],
            ]
        )
        b.set_data(data)
        sigma.set_bounds(b)
        sigma_key = field.set_construct(sigma, axes=axisZ, copy=False)

        # dimension_coordinate: sigma
        sigmac = cf.DimensionCoordinate(source=sigma)
        sigmac_key = field.set_construct(sigmac, axes=axisZ, copy=False)

        # coordinate_reference:
        coordref.set_coordinates({sigmac_key})
        coordref.coordinate_conversion.set_domain_ancillaries(
            {"ptop": ptop_key, "ps": ps_key, "sigma": sigma_key}
        )
        field.set_construct(coordref)

    elif standard_name == "atmosphere_hybrid_sigma_pressure_coordinate":
        computed_standard_name = "air_pressure"

        # Computed vertical corodinates
        aux.standard_name = computed_standard_name
        data = cf.Data([700, 500, 300], "hPa", dtype="f8")
        aux.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[800, 600], [600, 400], [400, 200]], "hPa", dtype="f8")
        bounds.set_data(data)
        aux.set_bounds(bounds)

        # domain_ancillary: ps
        ps = cf.DomainAncillary()
        ps.standard_name = "surface_air_pressure"
        data = cf.Data(1000, units="hPa", dtype="f8")
        ps.set_data(data)
        ps_key = field.set_construct(ps, axes=(), copy=False)

        # domain_ancillary: p0
        p0 = cf.DomainAncillary()
        data = cf.Data(1000, units="hPa", dtype="f8")
        p0.set_data(data)
        p0_key = field.set_construct(p0, axes=(), copy=False)

        # domain_ancillary: a
        a = cf.DomainAncillary()
        data = cf.Data([0.6, 0.3, 0], dtype="f8")
        a.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[0.75, 0.45], [0.45, 0.15], [0.15, 0]])
        bounds.set_data(data)
        a.set_bounds(bounds)
        a_key = field.set_construct(a, axes=axisZ, copy=False)

        # domain_ancillary: b
        b = cf.DomainAncillary()
        data = cf.Data([0.1, 0.2, 0.3], dtype="f8")
        b.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[0.05, 0.15], [0.15, 0.25], [0.25, 0.2]])
        bounds.set_data(data)
        b.set_bounds(bounds)
        b_key = field.set_construct(b, axes=axisZ, copy=False)

        # dimension_coordinate: sigma
        sigma = cf.DimensionCoordinate()
        sigma.standard_name = standard_name
        data = cf.Data([0.6969697, 0.49494949, 0.29292929])
        sigma.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data(
            [
                [0.7979798, 0.5959596],
                [0.5959596, 0.39393939],
                [0.39393939, 0.19191919],
            ]
        )
        bounds.set_data(data)
        sigma.set_bounds(bounds)
        sigma_key = field.set_construct(sigma, axes=axisZ, copy=False)

        # coordinate_reference:
        coordref.set_coordinates({sigma_key})
        coordref.coordinate_conversion.set_domain_ancillaries(
            {"p0": p0_key, "a": a_key, "b": b_key, "ps": ps_key}
        )
        field.set_construct(coordref)

    elif standard_name == "atmosphere_sleve_coordinate":
        computed_standard_name = "altitude"

        # Computed vertical corodinates
        aux.standard_name = computed_standard_name
        data = cf.Data([100, 200, 300], "m", dtype="f8")
        aux.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[50, 150], [150, 250], [250, 350]], "m", dtype="f8")
        bounds.set_data(data)
        aux.set_bounds(bounds)

        # domain_ancillary: ztop
        ztop = cf.DomainAncillary()
        ztop.standard_name = "altitude_at_top_of_atmosphere_model"
        data = cf.Data(1000, units="m", dtype="f8")
        ztop.set_data(data)
        ztop_key = field.set_construct(ztop, axes=(), copy=False)

        # domain_ancillary: zsurf1
        zsurf1 = cf.DomainAncillary()
        data = cf.Data(90, units="m", dtype="f8")
        zsurf1.set_data(data)
        zsurf1_key = field.set_construct(zsurf1, axes=(), copy=False)

        # domain_ancillary: zsurf2
        zsurf2 = cf.DomainAncillary()
        data = cf.Data(0.1, units="m", dtype="f8")
        zsurf2.set_data(data)
        zsurf2_key = field.set_construct(zsurf2, axes=(), copy=False)

        # domain_ancillary: b1
        b1 = cf.DomainAncillary()
        data = cf.Data([0.05, 0.04, 0.03], dtype="f8")
        b1.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[0.055, 0.045], [0.045, 0.035], [0.035, 0.025]])
        bounds.set_data(data)
        b1.set_bounds(bounds)
        b1_key = field.set_construct(b1, axes=axisZ, copy=False)

        # domain_ancillary: b2
        b2 = cf.DomainAncillary()
        data = cf.Data([0.5, 0.4, 0.3])
        b2.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[0.55, 0.45], [0.45, 0.35], [0.35, 0.25]])
        bounds.set_data(data)
        b2.set_bounds(bounds)
        b2_key = field.set_construct(b2, axes=axisZ, copy=False)

        # domain_ancillary: a
        a = cf.DomainAncillary()
        data = cf.Data([0.09545, 0.19636, 0.29727])
        a.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data(
            [[0.044995, 0.145905], [0.145905, 0.246815], [0.246815, 0.347725]]
        )
        bounds.set_data(data)
        a.set_bounds(bounds)
        a_key = field.set_construct(a, axes=axisZ, copy=False)

        # coordinate_reference:
        coordref.coordinate_conversion.set_domain_ancillaries(
            {
                "zsurf1": zsurf1_key,
                "a": a_key,
                "b1": b1_key,
                "b2": b2_key,
                "zsurf2": zsurf2_key,
                "ztop": ztop_key,
            }
        )
        field.set_construct(coordref)

    elif standard_name == "ocean_sigma_coordinate":
        computed_standard_name = "altitude"

        # Computed vertical corodinates
        aux.standard_name = computed_standard_name
        data = cf.Data([10, 20, 30], "m", dtype="f8")
        aux.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[5, 15], [15, 25], [25, 35]], "m", dtype="f8")
        bounds.set_data(data)
        aux.set_bounds(bounds)

        # domain_ancillary: depth
        depth = cf.DomainAncillary()
        depth.standard_name = "sea_floor_depth_below_geoid"
        data = cf.Data(-1000.0, units="m")
        depth.set_data(data)
        depth_key = field.set_construct(depth, axes=(), copy=False)

        # domain_ancillary: eta
        eta = cf.DomainAncillary()
        eta.standard_name = "sea_surface_height_above_geoid"
        data = cf.Data(100.0, units="m")
        eta.set_data(data)
        eta_key = field.set_construct(eta, axes=(), copy=False)

        # domain_ancillary: sigma
        sigma = cf.DomainAncillary()
        sigma.standard_name = standard_name
        data = cf.Data([0.1, 0.08888888888888889, 0.07777777777777778])
        sigma.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data(
            [
                [0.10555556, 0.09444444],
                [0.09444444, 0.08333333],
                [0.08333333, 0.07222222],
            ]
        )
        bounds.set_data(data)
        sigma.set_bounds(bounds)
        sigma_key = field.set_construct(sigma, axes=axisZ, copy=False)

        # dimension_coordinate: sigma
        sigmac = cf.DimensionCoordinate(source=sigma)
        sigmac_key = field.set_construct(sigmac, axes=axisZ, copy=False)

        # coordinate_reference:
        coordref.set_coordinates({sigmac_key})
        coordref.coordinate_conversion.set_domain_ancillaries(
            {"depth": depth_key, "eta": eta_key, "sigma": sigma_key}
        )
        field.set_construct(coordref)

    elif standard_name == "ocean_s_coordinate":
        computed_standard_name = "altitude"

        # Computed vertical corodinates
        aux.standard_name = computed_standard_name
        data = cf.Data([15.01701191, 31.86034296, 40.31150319], units="m")
        aux.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data(
            [
                [15.01701191, 23.42877638],
                [23.42877638, 31.86034296],
                [31.86034296, 40.31150319],
            ],
            units="m",
        )
        bounds.set_data(data)
        aux.set_bounds(bounds)

        # domain_ancillary: depth
        depth = cf.DomainAncillary()
        depth.standard_name = "sea_floor_depth_below_geoid"
        data = cf.Data(-1000.0, units="m")
        depth.set_data(data)
        depth_key = field.set_construct(depth, axes=(), copy=False)

        # domain_ancillary: eta
        eta = cf.DomainAncillary()
        eta.standard_name = "sea_surface_height_above_geoid"
        data = cf.Data(100.0, units="m")
        eta.set_data(data)
        eta_key = field.set_construct(eta, axes=(), copy=False)

        # domain_ancillary: depth_c
        depth_c = cf.DomainAncillary()
        data = cf.Data(10.0, units="m")
        depth_c.set_data(data)
        depth_c_key = field.set_construct(depth_c, axes=(), copy=False)

        # domain_ancillary: a
        a = cf.DomainAncillary()
        data = cf.Data(0.5)
        a.set_data(data)
        a_key = field.set_construct(a, axes=(), copy=False)

        # domain_ancillary: b
        b = cf.DomainAncillary()
        data = cf.Data(0.75)
        b.set_data(data)
        b_key = field.set_construct(b, axes=(), copy=False)

        # domain_ancillary: s
        s = cf.DomainAncillary()
        s.standard_name = standard_name
        data = cf.Data([0.1, 0.08, 0.07])
        s.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[0.10, 0.09], [0.09, 0.08], [0.08, 0.07]])
        bounds.set_data(data)
        s.set_bounds(bounds)
        s_key = field.set_construct(s, axes=axisZ, copy=False)

        # dimension_coordinate: s
        sc = cf.DimensionCoordinate(source=s)
        sc_key = field.set_construct(sc, axes=axisZ, copy=False)

        # coordinate_reference:
        coordref.set_coordinates({sc_key})
        coordref.coordinate_conversion.set_domain_ancillaries(
            {
                "depth": depth_key,
                "eta": eta_key,
                "depth_c": depth_c_key,
                "a": a_key,
                "b": b_key,
                "s": s_key,
            }
        )
        field.set_construct(coordref)

    elif standard_name == "ocean_s_coordinate_g1":
        computed_standard_name = "altitude"

        # Computed vertical corodinates
        aux.standard_name = computed_standard_name
        data = cf.Data([555.4, 464.32, 373.33], units="m")
        aux.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data(
            [[600.85, 509.86], [509.86, 418.87], [418.87, 327.88]], units="m"
        )
        bounds.set_data(data)
        aux.set_bounds(bounds)

        # domain_ancillary: depth
        depth = cf.DomainAncillary()
        depth.standard_name = "sea_floor_depth_below_geoid"
        data = cf.Data(-1000.0, units="m")
        depth.set_data(data)
        depth_key = field.set_construct(depth, axes=(), copy=False)

        # domain_ancillary: eta
        eta = cf.DomainAncillary()
        eta.standard_name = "sea_surface_height_above_geoid"
        data = cf.Data(100.0, units="m")
        eta.set_data(data)
        eta_key = field.set_construct(eta, axes=(), copy=False)

        # domain_ancillary: depth_c
        depth_c = cf.DomainAncillary()
        data = cf.Data(10.0, units="m")
        depth_c.set_data(data)
        depth_c_key = field.set_construct(depth_c, axes=(), copy=False)

        # domain_ancillary: C
        C = cf.DomainAncillary()
        data = cf.Data([-0.5, -0.4, -0.3])
        C.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[-0.55, -0.45], [-0.45, -0.35], [-0.35, -0.25]])
        bounds.set_data(data)
        C.set_bounds(bounds)
        C_key = field.set_construct(C, axes=axisZ, copy=False)

        # domain_ancillary: s
        s = cf.DomainAncillary()
        s.standard_name = standard_name
        data = cf.Data([0.1, 0.08, 0.07])
        s.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[0.10, 0.09], [0.09, 0.08], [0.08, 0.07]])
        bounds.set_data(data)
        s.set_bounds(bounds)
        s_key = field.set_construct(s, axes=axisZ, copy=False)

        # dimension_coordinate: s
        sc = cf.DimensionCoordinate(source=s)
        sc_key = field.set_construct(sc, axes=axisZ, copy=False)

        # coordinate_reference:
        coordref.set_coordinates({sc_key})
        coordref.coordinate_conversion.set_domain_ancillaries(
            {
                "depth": depth_key,
                "eta": eta_key,
                "depth_c": depth_c_key,
                "C": C_key,
                "s": s_key,
            }
        )
        field.set_construct(coordref)

    elif standard_name == "ocean_s_coordinate_g2":
        computed_standard_name = "altitude"

        # Computed vertical corodinates
        aux.standard_name = computed_standard_name
        data = cf.Data([555.45454545, 464.36363636, 373.36363636], units="m")
        aux.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data(
            [
                [600.90909091, 509.90909091],
                [509.90909091, 418.90909091],
                [418.90909091, 327.90909091],
            ],
            units="m",
        )
        bounds.set_data(data)
        aux.set_bounds(bounds)

        # domain_ancillary: depth
        depth = cf.DomainAncillary()
        depth.standard_name = "sea_floor_depth_below_geoid"
        data = cf.Data(-1000.0, units="m")
        depth.set_data(data)
        depth_key = field.set_construct(depth, axes=(), copy=False)

        # domain_ancillary: eta
        eta = cf.DomainAncillary()
        eta.standard_name = "sea_surface_height_above_geoid"
        data = cf.Data(100.0, units="m")
        eta.set_data(data)
        eta_key = field.set_construct(eta, axes=(), copy=False)

        # domain_ancillary: depth_c
        depth_c = cf.DomainAncillary()
        data = cf.Data(10.0, units="m")
        depth_c.set_data(data)
        depth_c_key = field.set_construct(depth_c, axes=(), copy=False)

        # domain_ancillary: C
        C = cf.DomainAncillary()
        data = cf.Data([-0.5, -0.4, -0.3])
        C.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[-0.55, -0.45], [-0.45, -0.35], [-0.35, -0.25]])
        bounds.set_data(data)
        C.set_bounds(bounds)
        C_key = field.set_construct(C, axes=axisZ, copy=False)

        # domain_ancillary: s
        s = cf.DomainAncillary()
        s.standard_name = standard_name
        data = cf.Data([0.1, 0.08, 0.07])
        s.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[0.10, 0.09], [0.09, 0.08], [0.08, 0.07]])
        bounds.set_data(data)
        s.set_bounds(bounds)
        s_key = field.set_construct(s, axes=axisZ, copy=False)

        # dimension_coordinate: s
        sc = cf.DimensionCoordinate(source=s)
        sc_key = field.set_construct(sc, axes=axisZ, copy=False)

        # coordinat
        # coordinate_reference:
        coordref.set_coordinates({sc_key})
        coordref.coordinate_conversion.set_domain_ancillaries(
            {
                "depth": depth_key,
                "eta": eta_key,
                "depth_c": depth_c_key,
                "C": C_key,
                "s": s_key,
            }
        )
        field.set_construct(coordref)

    elif standard_name == "ocean_sigma_z_coordinate":
        computed_standard_name = "altitude"

        # Computed vertical corodinates
        aux.standard_name = computed_standard_name
        data = cf.Data([10.0, 30.0, 40.0], "m", dtype="f8")
        aux.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data(
            [[10.0, 19.0], [25.0, 35.0], [35.0, 45.0]], "m", dtype="f8"
        )
        bounds.set_data(data)
        aux.set_bounds(bounds)

        # domain_ancillary: depth
        depth = cf.DomainAncillary()
        depth.standard_name = "sea_floor_depth_below_geoid"
        data = cf.Data(-1000.0, units="m")
        depth.set_data(data)
        depth_key = field.set_construct(depth, axes=(), copy=False)

        # domain_ancillary: eta
        eta = cf.DomainAncillary()
        eta.standard_name = "sea_surface_height_above_geoid"
        data = cf.Data(100.0, units="m")
        eta.set_data(data)
        eta_key = field.set_construct(eta, axes=(), copy=False)

        # domain_ancillary: depth_c
        depth_c = cf.DomainAncillary()
        data = cf.Data(10.0, units="m")
        depth_c.set_data(data)
        depth_c_key = field.set_construct(depth_c, axes=(), copy=False)

        # domain_ancillary: nsigma
        nsigma = cf.DomainAncillary()
        data = cf.Data(1)
        nsigma.set_data(data)
        nsigma_key = field.set_construct(nsigma, axes=(), copy=False)

        # domain_ancillary: zlev
        zlev = cf.DomainAncillary()
        zlev.standard_name = "altitude"
        data = cf.Data([20, 30, 40], units="m", dtype="f8")
        zlev.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[15, 25], [25, 35], [35, 45]], units="m", dtype="f8")
        bounds.set_data(data)
        zlev.set_bounds(bounds)
        zlev_key = field.set_construct(zlev, axes=axisZ, copy=False)

        # domain_ancillary: sigma
        sigma = cf.DomainAncillary()
        sigma.standard_name = standard_name
        data = cf.Data([0.1, 0.08, 0.07])
        sigma.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[0.10, 0.09], [0.09, 0.08], [0.08, 0.07]])
        bounds.set_data(data)
        sigma.set_bounds(bounds)
        sigma_key = field.set_construct(sigma, axes=axisZ, copy=False)

        # dimension_coordinate: sigma
        sigmac = cf.DimensionCoordinate(source=sigma)
        sigmac_key = field.set_construct(sigmac, axes=axisZ, copy=False)

        # coordinate_reference:
        coordref.set_coordinates({sigmac_key})
        coordref.coordinate_conversion.set_domain_ancillaries(
            {
                "depth": depth_key,
                "eta": eta_key,
                "depth_c": depth_c_key,
                "nsigma": nsigma_key,
                "zlev": zlev_key,
                "sigma": sigma_key,
            }
        )
        field.set_construct(coordref)

    elif standard_name == "ocean_double_sigma_coordinate":
        computed_standard_name = "altitude"

        # Computed vertical corodinates
        aux.standard_name = computed_standard_name
        data = cf.Data(
            [0.15000000000000002, 0.12, 932.895], units="m", dtype="f8"
        )
        aux.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data(
            [
                [1.50000e-01, 1.35000e-01],
                [1.35000e-01, 1.20000e-01],
                [9.22880e02, 9.32895e02],
            ],
            units="m",
            dtype="f8",
        )
        bounds.set_data(data)
        aux.set_bounds(bounds)

        # domain_ancillary: depth
        depth = cf.DomainAncillary()
        depth.standard_name = "sea_floor_depth_below_geoid"
        data = cf.Data(-1000.0, units="m")
        depth.set_data(data)
        depth_key = field.set_construct(depth, axes=(), copy=False)

        # domain_ancillary: z1
        z1 = cf.DomainAncillary()
        data = cf.Data(2, units="m")
        z1.set_data(data)
        z1_key = field.set_construct(z1, axes=(), copy=False)

        # domain_ancillary: z2
        z2 = cf.DomainAncillary()
        data = cf.Data(1.5, units="m")
        z2.set_data(data)
        z2_key = field.set_construct(z2, axes=(), copy=False)

        # domain_ancillary: a
        a = cf.DomainAncillary()
        data = cf.Data(2.5, units="m")
        a.set_data(data)
        a_key = field.set_construct(a, axes=(), copy=False)

        # domain_ancillary: href
        href = cf.DomainAncillary()
        data = cf.Data(10.5, units="m")
        href.set_data(data)
        href_key = field.set_construct(href, axes=(), copy=False)

        # domain_ancillary: k_c
        k_c = cf.DomainAncillary()
        data = cf.Data(1)
        k_c.set_data(data)
        k_c_key = field.set_construct(k_c, axes=(), copy=False)

        # dimension_coordinate: sigma
        sigma = cf.DomainAncillary()
        sigma.standard_name = standard_name
        data = cf.Data([0.1, 0.08, 0.07])
        sigma.set_data(data)
        bounds = cf.Bounds()
        data = cf.Data([[0.10, 0.09], [0.09, 0.08], [0.08, 0.07]])
        bounds.set_data(data)
        sigma.set_bounds(bounds)
        sigma_key = field.set_construct(sigma, axes=axisZ, copy=False)

        # dimension_coordinate: sigma
        sigmac = cf.DimensionCoordinate(source=sigma)
        sigmac_key = field.set_construct(sigmac, axes=axisZ, copy=False)

        # coordinate_reference:
        coordref.set_coordinates({sigmac_key})
        coordref.coordinate_conversion.set_domain_ancillaries(
            {
                "depth": depth_key,
                "a": a_key,
                "k_c": k_c_key,
                "z1": z1_key,
                "z2": z2_key,
                "href": href_key,
                "sigma": sigma_key,
            }
        )
        field.set_construct(coordref)

    else:
        raise ValueError(
            f"Bad standard name: {standard_name}, "
            "not an element of FormulaTerms.standard_names"
        )

    return (field, aux, computed_standard_name)


class FormulaTermsTest(unittest.TestCase):
    def test_compute_vertical_coordinates(self):
        # ------------------------------------------------------------
        # atmosphere_hybrid_height_coordinate
        # ------------------------------------------------------------
        f = cf.example_field(1)
        self.assertIsNone(f.auxiliary_coordinate("altitude", default=None))

        orog = f.domain_ancillary("surface_altitude")
        a = f.domain_ancillary("ncvar%a")
        b = f.domain_ancillary("ncvar%b")

        # Set bounds properties on 'a' to check that they do not
        # appear on the bounds of the computed vertical coordinates
        a.bounds.set_property("foo", "bar")

        g = f.compute_vertical_coordinates(verbose=None)
        altitude = g.auxiliary_coordinate("altitude")

        self.assertTrue(altitude)
        self.assertTrue(altitude.has_bounds())
        self.assertEqual(altitude.shape, (1,) + orog.shape)
        self.assertEqual(altitude.bounds.shape, altitude.shape + (2,))

        # Check that bounds properties were cleared
        properties = altitude.bounds.properties()
        properties.pop("units", None)
        self.assertFalse(properties)

        # Check array values
        orog = orog.data.insert_dimension(-1)
        x = a.data + b.data * orog
        x.transpose([2, 0, 1], inplace=True)
        self.assertTrue(x.equals(altitude.data, verbose=3))

        # Check array bounds values
        orog = orog.insert_dimension(-1)
        bounds = a.bounds.data + b.bounds.data * orog
        bounds.transpose([2, 0, 1, 3], inplace=True)
        self.assertTrue(bounds.equals(altitude.bounds.data, verbose=3))

        # ------------------------------------------------------------
        # Missing 'a' bounds
        # ------------------------------------------------------------
        a.del_bounds()
        g = f.compute_vertical_coordinates(verbose=None)

        altitude = g.auxiliary_coordinate("altitude")
        orog = f.domain_ancillary("surface_altitude")
        self.assertTrue(altitude)
        self.assertEqual(altitude.shape, (1,) + orog.shape)
        self.assertFalse(altitude.has_bounds())

        # Check array values
        orog = orog.data.insert_dimension(-1)
        x = a.data + b.data * orog
        x.transpose([2, 0, 1], inplace=True)
        self.assertTrue(x.equals(altitude.data, verbose=3))

        # ------------------------------------------------------------
        # Missing 'a'
        # ------------------------------------------------------------
        f.del_construct("ncvar%a")
        g = f.compute_vertical_coordinates(verbose=None)

        altitude = g.auxiliary_coordinate("altitude")
        orog = f.domain_ancillary("surface_altitude")

        self.assertTrue(altitude)
        self.assertTrue(altitude.has_bounds())
        self.assertEqual(altitude.shape, (1,) + orog.shape)
        self.assertEqual(altitude.bounds.shape, altitude.shape + (2,))

        # Check array values
        orog = orog.data.insert_dimension(-1)
        x = b.data * orog
        x.transpose([2, 0, 1], inplace=True)
        self.assertTrue(x.equals(altitude.data, verbose=3))

        # Check array bounds values
        orog = orog.insert_dimension(-1)
        bounds = b.bounds.data * orog
        bounds.transpose([2, 0, 1, 3], inplace=True)
        self.assertTrue(bounds.equals(altitude.bounds.data, verbose=3))

        # ------------------------------------------------------------
        # Missing 'a' and no 'b' bounds
        # ------------------------------------------------------------
        b.del_bounds()
        g = f.compute_vertical_coordinates(verbose=None)

        altitude = g.auxiliary_coordinate("altitude")
        orog = f.domain_ancillary("surface_altitude")

        self.assertTrue(altitude)
        self.assertFalse(altitude.has_bounds())
        self.assertEqual(altitude.shape, (1,) + orog.shape)

        # Check array values
        orog = orog.data.insert_dimension(-1)
        x = b.data * orog
        x.transpose([2, 0, 1], inplace=True)
        self.assertTrue(x.equals(altitude.data, verbose=3))

        # ------------------------------------------------------------
        # Missing 'a' and missing 'b'
        # ------------------------------------------------------------
        f.del_construct("ncvar%b")

        g = f.compute_vertical_coordinates(verbose=None)

        altitude = g.auxiliary_coordinate("altitude")
        orog = f.domain_ancillary("surface_altitude")

        self.assertTrue(altitude)
        self.assertTrue(altitude.has_bounds())
        self.assertEqual(altitude.shape, orog.shape)
        self.assertEqual(altitude.bounds.shape, altitude.shape + (2,))

        # Check array values
        x = 0 * orog.data
        self.assertTrue(x.equals(altitude.data), repr(x))

        # Check array bounds values
        orog = orog.insert_dimension(-1)
        bounds = cf.Data([0, 0]) * orog.data
        self.assertTrue(bounds.equals(altitude.bounds.data), repr(x))

        # ------------------------------------------------------------
        # Check in-place
        # ------------------------------------------------------------
        self.assertIsNone(f.compute_vertical_coordinates(inplace=True))

        f.del_construct("surface_altitude")
        with self.assertRaises(ValueError):
            g = f.compute_vertical_coordinates()

        # ------------------------------------------------------------
        # Check with no vertical coordinates
        # ------------------------------------------------------------
        f = cf.example_field(0)
        g = f.compute_vertical_coordinates()
        self.assertTrue(g.equals(f))

        # ------------------------------------------------------------
        # Check other types
        # ------------------------------------------------------------
        for standard_name in cf.formula_terms.FormulaTerms.standard_names:
            if standard_name == "atmosphere_hybrid_height_coordinate":
                continue

            f, a, csn = _formula_terms(standard_name)

            g = f.compute_vertical_coordinates(verbose=None)

            x = g.auxiliary_coordinate(csn)

            self.assertTrue(
                x.equals(a, atol=1e-5, rtol=1e-05, verbose=-1),
                f"{standard_name}, {x.array}, {a.array}"
                f"\n{x.bounds.array}\n{a.bounds.array}",
            )


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
