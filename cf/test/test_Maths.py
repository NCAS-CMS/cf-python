import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import numpy as np

import cf


class MathTest(unittest.TestCase):
    def test_curl_xy(self):
        f = cf.example_field(0)

        # Spherical polar coordinates
        theta = 90 - f.convert("Y", full_domain=True)
        sin_theta = theta.sin()

        radius = 2
        r = f.radius(radius)

        for wrap in (False, True, None):
            for one_sided in (True, False):
                x, y = f.grad_xy(
                    radius=radius, x_wrap=wrap, one_sided_at_boundary=one_sided
                )

                c = cf.curl_xy(
                    x,
                    y,
                    radius=radius,
                    x_wrap=wrap,
                    one_sided_at_boundary=one_sided,
                )

                self.assertEqual(c.Units, cf.Units("m-2"))

                term1 = (x * sin_theta).derivative(
                    "Y", one_sided_at_boundary=one_sided
                )
                term2 = y.derivative(
                    "X", wrap=wrap, one_sided_at_boundary=one_sided
                )

                c0 = (term1 - term2) / (sin_theta * r)

                # Check the data
                with cf.rtol(1e-10):
                    self.assertTrue(c.data.allclose(c0.data))

                del c.long_name
                c0.set_data(c.data)
                self.assertTrue(c.equals(c0))

        # Cartesian coordinates
        dim_x = f.dimension_coordinate("X")
        dim_y = f.dimension_coordinate("Y")
        dim_x.override_units("m", inplace=True)
        dim_y.override_units("m", inplace=True)
        dim_x.standard_name = "projection_x_coordinate"
        dim_y.standard_name = "projection_y_coordinate"
        f.cyclic("X", iscyclic=False)

        for wrap in (False, True, None):
            for one_sided in (True, False):
                x, y = f.grad_xy(x_wrap=wrap, one_sided_at_boundary=one_sided)

                d = cf.div_xy(
                    x, y, x_wrap=wrap, one_sided_at_boundary=one_sided
                )

                self.assertEqual(d.Units, cf.Units("m-2"))

                term1 = x.derivative(
                    "X", wrap=wrap, one_sided_at_boundary=one_sided
                )
                term2 = y.derivative("Y", one_sided_at_boundary=one_sided)

                d0 = term1 + term2

                del d.long_name
                del d0.long_name

                self.assertTrue(d.equals(d0, rtol=1e-10))

        # Test case when spherical dimension coordinates have units
        # but no standard names
        f = cf.example_field(0)
        x, y = f.grad_xy(radius="earth")
        del x.dimension_coordinate("X").standard_name
        del x.dimension_coordinate("Y").standard_name
        del y.dimension_coordinate("X").standard_name
        del y.dimension_coordinate("Y").standard_name
        c = cf.curl_xy(x, y, radius="earth")
        self.assertEqual(c.shape, f.shape)
        self.assertEqual(c.dimension_coordinate("Y").standard_name, "latitude")
        self.assertEqual(
            c.dimension_coordinate("X").standard_name, "longitude"
        )

    def test_div_xy(self):
        f = cf.example_field(0)

        # Spherical polar coordinates
        theta = 90 - f.convert("Y", full_domain=True)
        sin_theta = theta.sin()

        radius = 2
        r = f.radius(radius)

        for wrap in (False, True, None):
            for one_sided in (False, True):
                x, y = f.grad_xy(
                    radius=radius, x_wrap=wrap, one_sided_at_boundary=one_sided
                )

                d = cf.div_xy(
                    x,
                    y,
                    radius=radius,
                    x_wrap=wrap,
                    one_sided_at_boundary=one_sided,
                )
                self.assertEqual(d.Units, cf.Units("m-2"))

                term1 = x.derivative(
                    "X",
                    wrap=wrap,
                    one_sided_at_boundary=one_sided,
                )
                term2 = (y * sin_theta).derivative(
                    "Y",
                    one_sided_at_boundary=one_sided,
                )

                d0 = (term1 + term2) / (sin_theta * r)

                # Check the data
                with cf.rtol(1e-10):
                    self.assertTrue(d.data.allclose(d0.data))

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

        for wrap in (False, True, None):
            for one_sided in (True, False):
                x, y = f.grad_xy(x_wrap=wrap, one_sided_at_boundary=one_sided)

                d = cf.div_xy(
                    x, y, x_wrap=wrap, one_sided_at_boundary=one_sided
                )

                self.assertEqual(d.Units, cf.Units("m-2"))

                term1 = x.derivative(
                    "X", wrap=wrap, one_sided_at_boundary=one_sided
                )
                term2 = y.derivative("Y", one_sided_at_boundary=one_sided)

                d0 = term1 + term2

                del d.long_name
                del d0.long_name
                self.assertTrue(d.equals(d0, rtol=1e-10))

        # Test case when spherical dimension coordinates have units
        # but no standard names
        f = cf.example_field(0)
        x, y = f.grad_xy(radius="earth")
        del x.dimension_coordinate("X").standard_name
        del x.dimension_coordinate("Y").standard_name
        del y.dimension_coordinate("X").standard_name
        del y.dimension_coordinate("Y").standard_name
        d = cf.div_xy(x, y, radius="earth")
        self.assertEqual(d.shape, f.shape)
        self.assertEqual(d.dimension_coordinate("Y").standard_name, "latitude")
        self.assertEqual(
            d.dimension_coordinate("X").standard_name, "longitude"
        )

    def test_differential_operators(self):
        f = cf.example_field(0)

        radius = 2

        fx, fy = f.grad_xy(radius=radius, one_sided_at_boundary=True)
        c = cf.curl_xy(fx, fy, radius=radius, one_sided_at_boundary=True)

        # Divergence of curl is zero
        dc = cf.div_xy(c, c, radius=radius, one_sided_at_boundary=True)

        zeros = dc.copy()
        zeros[...] = 0
        self.assertTrue(dc.data.equals(zeros.data, rtol=0, atol=1e-15))

        # Curl of gradient is zero
        cg = cf.curl_xy(fx, fy, radius=radius, one_sided_at_boundary=True)

        zeros = cg.copy()
        zeros[...] = 0
        self.assertTrue(cg.data.equals(zeros.data, rtol=0, atol=1e-15))

    def test_histogram(self):
        f = cf.example_field(0)
        g = f.copy()
        g.standard_name = "air_temperature"
        g[...] = np.arange(40).reshape(5, 8) + 253.15
        g.override_units("K", inplace=True)

        atol = 1e-15

        # 1-d histogram
        indices = f.digitize(10)
        h = cf.histogram(indices)
        self.assertTrue((h.array == [9, 7, 9, 4, 5, 1, 1, 1, 2, 1]).all)
        h = cf.histogram(indices, density=True)
        self.assertEqual(h.Units, cf.Units("1"))
        # Check that integral is 1
        bin_measures = h.dimension_coordinate().cellsize
        integral = (h * bin_measures).sum()
        self.assertTrue(np.allclose(integral.array, 1, rtol=0, atol=atol))

        # 2-d histogram
        indices_t = g.digitize(5)
        h = cf.histogram(indices, indices_t)
        self.assertEqual(h.Units, cf.Units())
        # The -1 values correspond to missing values in h
        self.assertTrue(
            (
                h.array
                == [
                    [3, 3, 2, -1, -1, -1, -1, -1, -1, -1],
                    [1, 1, 2, 1, 3, -1, -1, -1, -1, -1],
                    [1, -1, -1, 1, -1, 1, 1, 1, 2, 1],
                    [2, 1, 1, 2, 2, -1, -1, -1, -1, -1],
                    [2, 2, 4, -1, -1, -1, -1, -1, -1, -1],
                ]
            ).all()
        )

        h = cf.histogram(indices, indices_t, density=True)
        self.assertEqual(h.Units, cf.Units("K-1"))
        # Check that integral is 1
        bin_measures = h.dimension_coordinate("air_temperature").cellsize
        bin_measures.outerproduct(
            h.dimension_coordinate("specific_humidity").cellsize, inplace=True
        )
        integral = (h * bin_measures).sum()
        self.assertTrue(np.allclose(integral.array, 1, rtol=0, atol=atol))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
