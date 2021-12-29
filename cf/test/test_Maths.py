import datetime
import faulthandler
import os
import unittest

import numpy

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class MathTest(unittest.TestCase):
    filename1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_file1.nc"
    )

    def test_relative_vorticity_distance(self):
        x_min = 0.0
        x_max = 100.0
        dx = 1.0

        x_1d = numpy.arange(x_min, x_max, dx)
        size = x_1d.size

        data_1d = x_1d * 2.0 + 1.0
        data_2d = numpy.broadcast_to(data_1d[numpy.newaxis, :], (size, size))

        dim_x = cf.DimensionCoordinate(
            data=cf.Data(x_1d, "m"), properties={"axis": "X"}
        )
        dim_y = cf.DimensionCoordinate(
            data=cf.Data(x_1d, "m"), properties={"axis": "Y"}
        )

        u = cf.Field()
        X = u.set_construct(cf.DomainAxis(size=dim_x.data.size))
        Y = u.set_construct(cf.DomainAxis(size=dim_y.data.size))
        u.set_construct(dim_x, axes=[X])
        u.set_construct(dim_y, axes=[Y])
        u.set_data(cf.Data(data_2d, "m/s"), axes=("Y", "X"))

        v = cf.Field()
        v.set_construct(cf.DomainAxis(size=dim_x.data.size))
        v.set_construct(cf.DomainAxis(size=dim_y.data.size))
        v.set_construct(dim_x, axes=[X])
        v.set_construct(dim_y, axes=[Y])
        v.set_data(cf.Data(data_2d, "m/s"), axes=("X", "Y"))

        rv = cf.relative_vorticity(u, v, one_sided_at_boundary=True)
        self.assertTrue((rv.array == 0.0).all())

    def test_relative_vorticity_latlong(self):
        lat_min = -90.0
        lat_max = 90.0
        dlat = 1.0

        lat_1d = numpy.arange(lat_min, lat_max, dlat)
        lat_size = lat_1d.size

        lon_min = 0.0
        lon_max = 359.0
        dlon = 1.0

        lon_1d = numpy.arange(lon_min, lon_max, dlon)
        lon_size = lon_1d.size

        u_1d = lat_1d * 2.0 + 1.0
        u_2d = numpy.broadcast_to(u_1d[numpy.newaxis, :], (lon_size, lat_size))

        v_1d = lon_1d * 2.0 + 1.0
        v_2d = numpy.broadcast_to(v_1d[:, numpy.newaxis], (lon_size, lat_size))
        v_2d = v_2d * numpy.cos(lat_1d * numpy.pi / 180.0)[numpy.newaxis, :]

        rv_array = (
            u_2d
            / cf.Data(6371229.0, "meters")
            * numpy.tan(lat_1d * numpy.pi / 180.0)[numpy.newaxis, :]
        )

        dim_x = cf.DimensionCoordinate(
            data=cf.Data(lon_1d, "degrees_east"), properties={"axis": "X"}
        )
        dim_y = cf.DimensionCoordinate(
            data=cf.Data(lat_1d, "degrees_north"), properties={"axis": "Y"}
        )

        u = cf.Field()
        u.set_construct(cf.DomainAxis(size=lon_1d.size))
        u.set_construct(cf.DomainAxis(size=lat_1d.size))
        u.set_construct(dim_x)
        u.set_construct(dim_y)
        u.set_data(cf.Data(u_2d, "m/s"), axes=("X", "Y"))
        u.cyclic("X", period=360.0)

        v = cf.Field()
        v.set_construct(cf.DomainAxis(size=lon_1d.size))
        v.set_construct(cf.DomainAxis(size=lat_1d.size))
        v.set_construct(dim_x)
        v.set_construct(dim_y)
        v.set_data(cf.Data(v_2d, "m/s"), axes=("X", "Y"))
        v.cyclic("X", period=360.0)

        rv = cf.relative_vorticity(u, v, wrap=False)
        self.assertTrue(numpy.allclose(rv.array, rv_array))

    def test_div_xy(self):
        f = cf.example_field(0)

        # Spherical polar coordinates
        r = f.radius("earth")
        for wrap in (None, True, False):
            for one_sided in (True, False):
                x, y = f.grad_xy(x_wrap=wrap, one_sided_at_boundary=one_sided)

                d = cf.div_xy(
                    x, y, x_wrap=wrap, one_sided_at_boundary=one_sided
                )

                theta = 90 - f.convert("Y", full_domain=True)
                theta.Units = cf.Units("radians")

                term1 = x.derivative(
                    "X", wrap=wrap, one_sided_at_boundary=one_sided
                ) / (theta.sin() * r)
                term2 = (y * theta.sin()).derivative(
                    "Y", one_sided_at_boundary=one_sided
                ) / (theta.sin() * r)

                d0 = term1 + term2

                # Check the data
                self.assertTrue((d.data == d0.data).all())

                # Check the metadata
                del d.long_name
                d0.set_data(d.data)
                self.assertTrue(d.equals(d0))

        # Cartesian coordinates
        dim_x = f.dimension_coordinate("X")
        dim_y = f.dimension_coordinate("Y")
        dim_x.override_units("m", inplace=True)
        dim_y.override_units("m", inplace=True)
        dim_x.standard_name = "projection_x_coordinate"
        dim_y.standard_name = "projection_y_coordinate"
        f.cyclic("X", iscyclic=False)

        for wrap in (None, True, False):
            for one_sided in (True, False):
                x, y = f.grad_xy(x_wrap=wrap, one_sided_at_boundary=one_sided)

                d = cf.div_xy(
                    x, y, x_wrap=wrap, one_sided_at_boundary=one_sided
                )

                term1 = x.derivative(
                    "X", wrap=wrap, one_sided_at_boundary=one_sided
                )
                term2 = y.derivative("Y", one_sided_at_boundary=one_sided)

                d0 = term1 + term2

                del d.long_name
                del d0.long_name

                self.assertTrue(d.equals(d0))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
