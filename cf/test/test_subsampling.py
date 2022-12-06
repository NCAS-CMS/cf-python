import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf

n_tmpfiles = 1
tmpfiles = [
    tempfile.mkstemp("_test_subsampling.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
(tempfile,) = tmpfiles


def _remove_tmpfiles():
    """Remove temporary files created during tests."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class SubsampledTest(unittest.TestCase):
    """Test management of underlying subsampled arrays."""

    biquadratic = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "subsampled_2.nc"
    )

    linear = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "subsampled_1.nc"
    )
    bilinear = linear
    quadratic = linear

    def setUp(self):
        """Preparations called immediately before each test method."""
        # Disable log messages to silence expected warnings
        cf.log_level("DISABLE")
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows:
        #
        # cf.log_level('DEBUG')
        # < ... test code ... >
        # cf.log_level('DISABLE')

    def test_linear(self):
        """Test linear interpolation."""
        f = cf.read(self.linear)

        q = f[0]
        lat0 = f[5].data
        lon0 = f[9].data

        lat = q.construct("latitude").data
        lon = q.construct("longitude").data

        self.assertEqual(lat.dtype, float)
        self.assertEqual(lon.dtype, float)

        self.assertTrue(
            lat.equals(lat0, ignore_data_type=True, rtol=0, atol=1e-25)
        )
        self.assertTrue(
            lon.equals(lon0, ignore_data_type=True, rtol=0, atol=1e-25)
        )

        # Bounds
        lat0 = f[6].data
        lon0 = f[10].data

        lat = q.construct("latitude").bounds.data
        lon = q.construct("longitude").bounds.data

        self.assertEqual(lat.dtype, float)
        self.assertEqual(lon.dtype, float)

        self.assertTrue(
            lat.equals(lat0, ignore_data_type=True, rtol=0, atol=1e-13)
        )
        self.assertTrue(
            lon.equals(lon0, ignore_data_type=True, rtol=0, atol=1e-13)
        )

    def test_bi_linear(self):
        """Test bi-linear interpolation."""
        f = cf.read(self.bilinear)

        q = f[0]
        lat0 = f[1].data
        lon0 = f[3].data

        lat = q.construct("ncvar%a_2d").data
        lon = q.construct("ncvar%b_2d").data

        self.assertEqual(lat.dtype, float)
        self.assertEqual(lon.dtype, float)

        self.assertTrue(
            lat.equals(lat0, ignore_data_type=True, rtol=0, atol=1e-25)
        )
        self.assertTrue(
            lon.equals(lon0, ignore_data_type=True, rtol=0, atol=1e-25)
        )

        # Bounds
        lat0 = f[2].data
        lon0 = f[4].data

        lat = q.construct("ncvar%a_2d").bounds.data
        lon = q.construct("ncvar%b_2d").bounds.data

        self.assertEqual(lat.dtype, float)
        self.assertEqual(lon.dtype, float)

        self.assertTrue(
            lat.equals(
                lat0, ignore_data_type=True, rtol=0, atol=1e-13, verbose=-1
            )
        )
        self.assertTrue(
            lon.equals(
                lon0, ignore_data_type=True, rtol=0, atol=1e-13, verbose=-1
            )
        )

    def test_quadratic(self):
        """Test quadratic interpolation."""
        f = cf.read(self.quadratic)

        t = f[-2]
        lat0 = f[7].data
        lon0 = f[11].data

        lat = t.construct("latitude").data
        lon = t.construct("longitude").data

        self.assertEqual(lat.dtype, float)
        self.assertEqual(lon.dtype, float)

        self.assertTrue(
            lat.equals(lat0, ignore_data_type=True, rtol=0, atol=1e-25)
        )
        self.assertTrue(
            lon.equals(lon0, ignore_data_type=True, rtol=0, atol=2e-5)
        )

        # Bounds
        lat0 = f[8].data
        lon0 = f[12].data

        lat = t.construct("latitude").bounds.data
        lon = t.construct("longitude").bounds.data

        self.assertEqual(lat.dtype, float)
        self.assertEqual(lon.dtype, float)

        self.assertTrue(
            lat.equals(
                lat0, ignore_data_type=True, rtol=0, atol=9e-5, verbose=-1
            )
        )
        self.assertTrue(
            lon.equals(
                lon0, ignore_data_type=True, rtol=0, atol=9e-5, verbose=-1
            )
        )

    @unittest.skipIf(True, "TODO: awaiting test file")
    def test_quadratic_latitude_longitude(self):
        """Test quadratic latitude longitude interpolation."""
        f = cf.read(self.quadratic)

        i = f["change"]
        lat0 = f["change"].data
        lon0 = f["change"].data

        lat = i.construct("latitude").data
        lon = i.construct("longitude").data

        self.assertEqual(lat.dtype, float)
        self.assertEqual(lon.dtype, float)

        self.assertTrue(
            lat.equals(lat0, ignore_data_type=True, rtol=0, atol="change")
        )
        self.assertTrue(
            lon.equals(lon0, ignore_data_type=True, rtol=0, atol="change")
        )

    def test_bi_quadratic_latitude_longitude(self):
        """Test bi-quadratic latitude longitude interpolation."""
        f = cf.read(self.biquadratic)

        i = f[-3]
        lat0 = f[-2].data
        lon0 = f[-1].data

        lat = i.construct("latitude").data
        lon = i.construct("longitude").data

        self.assertEqual(lat.dtype, float)
        self.assertEqual(lon.dtype, float)

        self.assertTrue(
            lat.equals(lat0, ignore_data_type=True, rtol=0, atol=6e-6)
        )
        self.assertTrue(
            lon.equals(lon0, ignore_data_type=True, rtol=0, atol=3e-5)
        )

        # Check original filenames
        self.assertEqual(i.get_original_filenames(), set([self.biquadratic]))

    def test_non_standard(self):
        """Test non-standardised interpolation."""
        f = cf.read(self.linear)

        # Get a field with non-standardised coordinate interpolation
        t = f[15]
        self.assertEqual(t.nc_get_variable(), "t3")

        # Check that the we can inspect the compressed data as if it
        # were uncompressed
        a_2d = t.construct("ncvar%a_2d")
        self.assertEqual(a_2d.shape, (18, 12))
        self.assertEqual(a_2d.get_property("units"), "m")

        # Check that we can't uncompress the data
        with self.assertRaises(ValueError):
            a_2d.array


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
