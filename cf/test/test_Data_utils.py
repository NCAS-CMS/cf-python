import datetime
import faulthandler
import unittest

import dask.array as da
import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class DataUtilsTest(unittest.TestCase):
    def test_Data_Utils__da_ma_allclose(self):
        """TODO."""
        # Create a range of inputs to test against.
        # Note that 'a' and 'a2' should be treated as 'allclose' for this
        # method, the same result as np.ma.allclose would give because all
        # of the *unmasked* elements are 'allclose', whereas in our
        # Data.equals method that builds on this method, we go even further
        # and insist on the mask being identical as well as the data
        # (separately, i.e. unmasked) all being 'allclose', so inside our
        # cf.Data objects 'a' and 'a2' would instead *not* be considered equal.
        a = np.ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
        a2 = np.ma.array([10.0, 2.0, 3.0], mask=[1, 0, 0])
        b = np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0])
        c = np.ma.array([1.0, 2.0, 100.0], mask=[1, 0, 0])
        d = np.array([1.0, 2.0, 3.0])
        e = a + 5e-04  # outside of default tolerances
        f = a + 5e-06  # within default tolerances

        # Test the function with these inputs as both numpy and dask arrays...
        allclose = cf.data.dask_utils._da_ma_allclose
        da_ = da.from_array(a)

        self.assertTrue(allclose(a, a).compute())
        self.assertTrue(allclose(da_, da_).compute())

        self.assertTrue(allclose(a2, a).compute())
        self.assertTrue(allclose(da.from_array(a2), da_).compute())

        self.assertTrue(allclose(b, a).compute())
        self.assertTrue(allclose(da.from_array(b), da_).compute())
        # ...including testing the 'masked_equal' parameter
        self.assertFalse(allclose(b, a, masked_equal=False).compute())
        self.assertFalse(
            allclose(da.from_array(b), da_, masked_equal=False).compute()
        )

        self.assertFalse(allclose(c, a).compute())
        self.assertFalse(allclose(da.from_array(c), da_).compute())

        self.assertTrue(allclose(d, a).compute())
        self.assertTrue(allclose(da.from_array(d), da_).compute())

        self.assertFalse(allclose(e, a).compute())
        self.assertFalse(allclose(da.from_array(e), da_).compute())

        self.assertTrue(allclose(f, a).compute())
        self.assertTrue(allclose(da.from_array(f), da_).compute())

        # Test when array inputs have different chunk sizes
        da_ = da.from_array(a, chunks=(1, 2))
        self.assertTrue(allclose(da.from_array(b, chunks=(3,)), da_).compute())
        self.assertFalse(
            allclose(
                da.from_array(b, chunks=(3,)), da_, masked_equal=False
            ).compute()
        )
        self.assertFalse(
            allclose(da.from_array(c, chunks=(3,)), da_).compute()
        )

        # Test the 'rtol' and 'atol' parameters:
        self.assertFalse(allclose(e, a, rtol=1e-06).compute())
        self.assertFalse(allclose(da.from_array(e), da_, rtol=1e-06).compute())
        b1 = e / 10000
        b2 = a / 10000
        self.assertTrue(allclose(b1, b2, atol=1e-05).compute())
        self.assertTrue(
            allclose(
                da.from_array(b1), da.from_array(b2), atol=1e-05
            ).compute()
        )

    def test_Data_Utils__is_numeric_dtype(self):
        """TODO."""
        _is_numeric_dtype = cf.data.utils._is_numeric_dtype
        for a in [
            np.array([0, 1, 2]),
            np.array([False, True, True]),
            np.ma.array([10.0, 2.0, 3.0], mask=[1, 0, 0]),
            np.array(10),
        ]:
            self.assertTrue(_is_numeric_dtype(a))

        for b in [
            np.array(["a", "b", "c"], dtype="S1"),
            np.empty(1, dtype=object),
        ]:
            self.assertFalse(_is_numeric_dtype(b))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
