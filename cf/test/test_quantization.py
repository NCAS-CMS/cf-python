import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

import netCDF4
import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import cf

# Set up temporary files
n_tmpfiles = 2
tmpfiles = [
    tempfile.mkstemp("_test_quantization.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
[tmpfile1, tmpfile2] = tmpfiles


def _remove_tmpfiles():
    """Remove temporary files created during tests."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class quantizationTest(unittest.TestCase):
    """Test the reading and writing with quantization."""

    f1 = cf.example_field(1)

    netcdf3_fmts = (
        "NETCDF3_CLASSIC",
        "NETCDF3_64BIT",
        "NETCDF3_64BIT_OFFSET",
        "NETCDF3_64BIT_DATA",
    )

    def test_quantization_read_write(self):
        """Test reading, writing, and storing quantization."""
        f = self.f1.copy()
        # Add some precision to the data
        f.data[...] = f.array + np.pi

        # Set a quantisation instruction
        nsd = 2
        q0 = cf.Quantization(
            {
                "algorithm": "granular_bitround",
                "implementation": "foobar",
                "quantization_nsd": nsd,
            }
        )
        f.set_quantize_on_write(q0)

        # Write the field and read it back in
        tmpfile1 = "delme1.nc"
        cf.write(f, tmpfile1)
        g = cf.read(tmpfile1)[0]

        # Check that f and g have different data (i.e. that
        # quantization occured on disk, and not in f in memory)
        self.assertFalse(np.allclose(f.data, g.data))

        # Check that g has the correct Quantisation component
        q = g.get_quantization()
        self.assertIsInstance(q, cf.Quantization)
        self.assertEqual(
            q.parameters(),
            {
                "_QuantizeGranularBitRoundNumberOfSignificantDigits": nsd,
                "algorithm": "granular_bitround",
                "implementation": (
                    f"libnetcdf version {netCDF4.__netcdf4libversion__}"
                ),
                "quantization_nsd": nsd,
            },
        )

        # Write the quantized field and read it back in
        cf.write(g, tmpfile2)
        h = cf.read(tmpfile2)[0]
        self.assertIsInstance(h.get_quantization(), cf.Quantization)

        # Check that h and g are equal
        self.assertTrue(h.equals(g))

        # Check that h and g are not equal when they only differ by
        # their quantization components
        h._set_quantization(q0)
        self.assertFalse(h.equals(g))

    def test_quantize_on_write(self):
        """Test del/get/set quantize-on-write methods."""
        f = self.f1.copy()
        q0 = cf.Quantization(
            {
                "algorithm": "digitround",
                "quantization_nsd": 9,
                "implementation": "foobar",
            }
        )

        self.assertIsNone(f.set_quantize_on_write(q0))

        q1 = q0.copy()
        q1.del_parameter("implementation")

        self.assertTrue(f.get_quantize_on_write().equals(q1))
        self.assertTrue(f.del_quantize_on_write().equals(q1))
        self.assertIsNone(f.get_quantize_on_write(None))
        self.assertIsNone(f.del_quantize_on_write(None))

        # Check consistency of arguments
        with self.assertRaises(ValueError):
            f.set_quantize_on_write()

        with self.assertRaises(ValueError):
            f.set_quantize_on_write(quantization_nsd=2)

        with self.assertRaises(ValueError):
            f.set_quantize_on_write(quantization_nsd=2, quantization_nsb=2)

        with self.assertRaises(ValueError):
            f.set_quantize_on_write(algorithm="digitround", quantization_nsb=2)

        with self.assertRaises(ValueError):
            f.set_quantize_on_write(
                algorithm="digitround", quantization_nsd=2, quantization_nsb=2
            )

        with self.assertRaises(ValueError):
            f.set_quantize_on_write(algorithm="bitround", quantization_nsd=2)

        # Can't set quantization metadata when there is a
        # quantize-on-write instruction
        f._set_quantization(q0)
        with self.assertRaises(ValueError):
            f.set_quantize_on_write(q0)

    def test_quantization(self):
        """Test _del/get/_set quantization methods."""
        f = self.f1.copy()
        q0 = cf.Quantization(
            {
                "algorithm": "digitround",
                "quantization_nsd": 9,
                "implementation": "foobar",
            }
        )

        self.assertIsNone(f._set_quantization(q0))

        with self.assertRaises(ValueError):
            f.set_quantize_on_write(q0)

        self.assertTrue(f.get_quantization().equals(q0))
        self.assertTrue(f._del_quantization().equals(q0))
        self.assertIsNone(f.get_quantization(None))
        self.assertIsNone(f._del_quantization(None))

    def test_quantization_write_exceptions(self):
        """Test writing quantization exceptions."""
        f = self.f1.copy()

        # digit_round
        f.set_quantize_on_write(algorithm="digitround", quantization_nsd=2)
        with self.assertRaises(ValueError):
            cf.write(f, tmpfile1)

        # NetCDF3 formats
        for fmt in self.netcdf3_fmts:
            with self.assertRaises(ValueError):
                cf.write(f, tmpfile1, fmt=fmt)

        # Integer data type
        f.data.dtype = int
        with self.assertRaises(ValueError):
            cf.write(f, tmpfile1)

        # Out-of-range quantization_nsd
        f.data.dtype = "float32"
        f.set_quantize_on_write(algorithm="bitgroom", quantization_nsd=8)
        with self.assertRaises(ValueError):
            cf.write(f, tmpfile1)

        f.data.dtype = "float64"
        f.set_quantize_on_write(algorithm="bitgroom", quantization_nsd=16)
        with self.assertRaises(ValueError):
            cf.write(f, tmpfile1)

        # Out-of-range quantization_nsb
        f.data.dtype = "float32"
        f.set_quantize_on_write(algorithm="bitround", quantization_nsb=24)
        with self.assertRaises(ValueError):
            cf.write(f, tmpfile1)

        f.data.dtype = "float64"
        f.set_quantize_on_write(algorithm="bitround", quantization_nsb=53)
        with self.assertRaises(ValueError):
            cf.write(f, tmpfile1)

    def test_quantization_copy(self):
        """Test that quantization information gets copied."""
        f = self.f1.copy()

        f.set_quantize_on_write(algorithm="bitround", quantization_nsb=2)
        q = f.get_quantize_on_write()
        g = f.copy()
        self.assertTrue(g.get_quantize_on_write().equals(q))

        f.del_quantize_on_write(q)
        f._set_quantization(q)
        g = f.copy()
        self.assertTrue(g.get_quantization().equals(q))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
