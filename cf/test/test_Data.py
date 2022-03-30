import datetime
import faulthandler
import inspect
import itertools
import os
import unittest
import warnings
from functools import reduce
from operator import mul

import dask.array as da
import numpy as np

SCIPY_AVAILABLE = False
try:
    from scipy.ndimage import convolve1d

    SCIPY_AVAILABLE = True
# not 'except ImportError' as that can hide nested errors, catch anything:
except Exception:
    pass  # test with this dependency will then be skipped by unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf

# To facilitate the testing of logging outputs (see comment tag 'Logging note')
logger = cf.logging.getLogger(__name__)


# Variables for _collapse
a = np.arange(-100, 200.0, dtype=float).reshape(3, 4, 5, 5)

w = np.arange(1, 301.0, dtype=float).reshape(a.shape)
w[-1, -1, ...] = w[-1, -1, ...] * 2
w /= w.min()

ones = np.ones(a.shape, dtype=float)

ma = np.ma.arange(-100, 200.0, dtype=float).reshape(3, 4, 5, 5)
ma[:, 1, 4, 4] = np.ma.masked
ma[0, :, 2, 3] = np.ma.masked
ma[0, 3, :, 3] = np.ma.masked
ma[1, 2, 3, :] = np.ma.masked


# If True, all tests that will not pass temporarily due to the LAMA-to-Dask
# migration will be skipped. These skips will be incrementally removed as the
# migration progresses. TODODASK: ensure all skips are removed once complete.
TEST_DASKIFIED_ONLY = True


def reshape_array(a, axes):
    new_order = [i for i in range(a.ndim) if i not in axes]
    new_order.extend(axes)
    b = np.transpose(a, new_order)
    new_shape = b.shape[: b.ndim - len(axes)]
    new_shape += (reduce(mul, b.shape[b.ndim - len(axes) :]),)
    b = b.reshape(new_shape)
    return b


class DataTest(unittest.TestCase):

    axes_combinations = [
        axes
        for n in range(1, a.ndim + 1)
        for axes in itertools.combinations(range(a.ndim), n)
    ]

    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file.nc"
    )

    tempdir = os.path.dirname(os.path.abspath(__file__))

    filename6 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file2.nc"
    )

    a = a
    w = w
    ma = ma
    ones = ones

    test_only = []

    def setUp(self):
        # Suppress the warning output for some specific warnings which are
        # expected due to the nature of the tests being performed.
        expexted_warning_msgs = [
            "divide by zero encountered in arctanh",
            "invalid value encountered in arctanh",
            "divide by zero encountered in log",
            "invalid value encountered in log",
            "invalid value encountered in arcsin",
        ]
        for expected_warning in expexted_warning_msgs:
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=expected_warning,
            )

    def test_Data_equals(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        shape = 3, 4
        chunksize = 2, 6
        a = np.arange(12).reshape(*shape)

        d = cf.Data(a, "m", chunks=chunksize)
        self.assertTrue(d.equals(d))  # check equal to self
        self.assertTrue(d.equals(d.copy()))  # also do self-equality checks!

        # Different but equivalent datatype, which should *fail* the equality
        # test (i.e. equals return False) because we want equals to check
        # for strict equality, including equality of data type.
        d2 = cf.Data(a.astype(np.float32), "m", chunks=chunksize)
        self.assertTrue(d2.equals(d2.copy()))
        with self.assertLogs(level=30) as catch:
            self.assertFalse(d2.equals(d, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different data types: float32 != int64" in log_msg
                    for log_msg in catch.output
                )
            )

        e = cf.Data(a, "s", chunks=chunksize)  # different units to d
        self.assertTrue(e.equals(e.copy()))
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(e.equals(d, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different Units (<Units: s>, <Units: m>)" in log_msg
                    for log_msg in catch.output
                )
            )

        f = cf.Data(np.arange(12), "m", chunks=(6,))  # different shape to d
        self.assertTrue(f.equals(f.copy()))
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(f.equals(d, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different shapes: (12,) != (3, 4)" in log_msg
                    for log_msg in catch.output
                )
            )

        g = cf.Data(
            np.ones(shape, dtype="int64"), "m", chunks=chunksize
        )  # different values
        self.assertTrue(g.equals(g.copy()))
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(g.equals(d, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different array values" in log_msg
                    for log_msg in catch.output
                )
            )

        # Test NaN values
        d3 = cf.Data(a.astype(np.float64), "m", chunks=chunksize)
        h = cf.Data(np.full(shape, np.nan), "m", chunks=chunksize)
        # TODODASK: implement and test equal_nan kwarg to configure NaN eq.
        self.assertFalse(h.equals(h.copy()))
        with self.assertLogs(level=cf.log_level().value) as catch:
            # Compare to d3 not d since np.nan has dtype float64 (IEEE 754)
            self.assertFalse(h.equals(d3, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different array values" in log_msg
                    for log_msg in catch.output
                )
            )

        # Test inf values
        i = cf.Data(np.full(shape, np.inf), "m", chunks=chunksize)
        self.assertTrue(i.equals(i.copy()))
        with self.assertLogs(level=cf.log_level().value) as catch:
            # np.inf is also of dtype float64 (see comment on NaN tests above)
            self.assertFalse(i.equals(d3, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different array values" in log_msg
                    for log_msg in catch.output
                )
            )
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(h.equals(i, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different array values" in log_msg
                    for log_msg in catch.output
                )
            )

        # Test masked arrays
        # 1. Example case where the masks differ only (data is identical)
        mask_test_chunksize = (2, 1)
        j1 = cf.Data(
            np.ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0]),
            "m",
            chunks=mask_test_chunksize,
        )
        self.assertTrue(j1.equals(j1.copy()))
        j2 = cf.Data(
            np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0]),
            "m",
            chunks=mask_test_chunksize,
        )
        self.assertTrue(j2.equals(j2.copy()))
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(j1.equals(j2, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different array values" in log_msg
                    for log_msg in catch.output
                )
            )
        # 2. Example case where the data differs only (masks are identical)
        j3 = cf.Data(
            np.ma.array([1.0, 2.0, 100.0], mask=[1, 0, 0]),
            "m",
            chunks=mask_test_chunksize,
        )
        self.assertTrue(j3.equals(j3.copy()))
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(j1.equals(j3, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different array values" in log_msg
                    for log_msg in catch.output
                )
            )

        # 3. Trivial case of data that is fully masked
        j4 = cf.Data(
            np.ma.masked_all(shape, dtype="int"), "m", chunks=chunksize
        )
        self.assertTrue(j4.equals(j4.copy()))
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(j4.equals(d, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different array values" in log_msg
                    for log_msg in catch.output
                )
            )
        # 4. Case where all the unmasked data is 'allclose' to other data but
        # the data is not 'allclose' to it where it is masked, i.e. the data
        # on its own (namely without considering the mask) is not equal to the
        # other data on its own (e.g. note the 0-th element in below examples).
        # This differs to case (2): there data differs *only where unmasked*.
        # Note these *should* be considered equal inside cf.Data, and indeed
        # np.ma.allclose and our own _da_ma_allclose methods also hold
        # these to be 'allclose'.
        j5 = cf.Data(
            np.ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0]),
            "m",
            chunks=mask_test_chunksize,
        )
        self.assertTrue(j5.equals(j5.copy()))
        j6 = cf.Data(
            np.ma.array([10.0, 2.0, 3.0], mask=[1, 0, 0]),
            "m",
            chunks=mask_test_chunksize,
        )
        self.assertTrue(j6.equals(j6.copy()))
        self.assertTrue(j5.equals(j6))

        # Test non-numeric dtype arrays
        sa1 = cf.Data(
            np.array(["one", "two", "three"], dtype="S5"), "m", chunks=(3,)
        )
        self.assertTrue(sa1.equals(sa1.copy()))
        sa2_data = np.array(["one", "two", "four"], dtype="S4")
        sa2 = cf.Data(sa2_data, "m", chunks=(3,))
        self.assertTrue(sa2.equals(sa2.copy()))
        # Unlike for numeric types, for string-like data as long as the data
        # is the same consider the arrays equal, even if the dtype differs.
        # TODO DASK: this behaviour will be added via cfdm, test fails for now
        # ## self.assertTrue(sa1.equals(sa2))
        sa3_data = sa2_data.astype("S5")
        sa3 = cf.Data(sa3_data, "m", chunks=mask_test_chunksize)
        self.assertTrue(sa3.equals(sa3.copy()))
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(sa1.equals(sa3, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different array values" in log_msg
                    for log_msg in catch.output
                )
            )
        # ...including masked string arrays
        sa4 = cf.Data(
            np.ma.array(
                ["one", "two", "three"],
                mask=[0, 0, 1],
                dtype="S5",
            ),
            "m",
            chunks=mask_test_chunksize,
        )
        self.assertTrue(sa4.equals(sa4.copy()))
        sa5 = cf.Data(
            np.ma.array(
                ["one", "two", "three"],
                mask=[0, 1, 0],
                dtype="S5",
            ),
            "m",
            chunks=mask_test_chunksize,
        )
        self.assertTrue(sa5.equals(sa5.copy()))
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(sa4.equals(sa5, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different array values" in log_msg
                    for log_msg in catch.output
                )
            )

        # Test where inputs are scalars
        scalar_test_chunksize = (10,)
        s1 = cf.Data(1, chunks=scalar_test_chunksize)
        self.assertTrue(s1.equals(s1.copy()))
        s2 = cf.Data(10, chunks=scalar_test_chunksize)
        self.assertTrue(s2.equals(s2.copy()))
        s3 = cf.Data("a_string", chunks=scalar_test_chunksize)
        self.assertTrue(s3.equals(s3.copy()))
        # 1. both are scalars
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(s1.equals(s2, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different array values" in log_msg
                    for log_msg in catch.output
                )
            )
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(s1.equals(s3, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different data types: int64 != <U8" in log_msg
                    for log_msg in catch.output
                )
            )
        # 2. only one is a scalar
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(s1.equals(d, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different shapes: () != (3, 4)" in log_msg
                    for log_msg in catch.output
                )
            )

        # Test rtol and atol parameters
        tol_check_chunksize = 1, 1
        k1 = cf.Data(np.array([10.0, 20.0]), chunks=tol_check_chunksize)
        self.assertTrue(k1.equals(k1.copy()))
        k2 = cf.Data(np.array([10.01, 20.01]), chunks=tol_check_chunksize)
        self.assertTrue(k2.equals(k2.copy()))
        # Only one log check is sufficient here
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(k1.equals(k2, atol=0.005, rtol=0, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different array values (atol=0.005, rtol=0)"
                    in log_msg
                    for log_msg in catch.output
                )
            )
        self.assertTrue(k1.equals(k2, atol=0.02, rtol=0))
        self.assertFalse(k1.equals(k2, atol=0, rtol=0.0005))
        self.assertTrue(k1.equals(k2, atol=0, rtol=0.002))

        # Test ignore_fill_value parameter
        m1 = cf.Data(1, fill_value=1000, chunks=scalar_test_chunksize)
        self.assertTrue(m1.equals(m1.copy()))
        m2 = cf.Data(1, fill_value=2000, chunks=scalar_test_chunksize)
        self.assertTrue(m2.equals(m2.copy()))
        with self.assertLogs(level=cf.log_level().value) as catch:
            self.assertFalse(m1.equals(m2, verbose=2))
            self.assertTrue(
                any(
                    "Data: Different fill value: 1000 != 2000" in log_msg
                    for log_msg in catch.output
                )
            )
            self.assertTrue(m1.equals(m2, ignore_fill_value=True))

        # Test verbose parameter: 1/'INFO' level is behaviour change boundary
        for checks in [(1, False), (2, True)]:
            verbosity_level, expect_to_see_msg = checks
            with self.assertLogs(level=cf.log_level().value) as catch:
                # Logging note: want to assert in the former case (verbosity=1)
                # that nothing is logged, but need to use workaround to prevent
                # AssertionError on fact that nothing is logged here. When at
                # Python =>3.10 this can be replaced by 'assertNoLogs' method.
                logger.warning(
                    "Log warning to prevent test error on empty log."
                )

                self.assertFalse(d2.equals(d, verbose=verbosity_level))
                self.assertIs(
                    any(
                        "Data: Different data types: float32 != int64"
                        in log_msg
                        for log_msg in catch.output
                    ),
                    expect_to_see_msg,
                )

        # Test ignore_data_type parameter
        self.assertTrue(d2.equals(d, ignore_data_type=True))

        # Test all possible chunk combinations
        for j, i in itertools.product([1, 2], [1, 2, 3]):
            d = cf.Data(np.arange(6).reshape(2, 3), "m", chunks=(j, i))
            for j, i in itertools.product([1, 2], [1, 2, 3]):
                e = cf.Data(np.arange(6).reshape(2, 3), "m", chunks=(j, i))
                self.assertTrue(d.equals(e))

    def test_Data_halo(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(np.arange(12).reshape(3, 4), "m", chunks=-1)
        d[-1, -1] = cf.masked
        d[1, 1] = cf.masked

        e = d.copy()
        self.assertIsNone(e.halo(1, inplace=True))

        e = d.halo(0)
        self.assertTrue(d.equals(e, verbose=2))

        shape = d.shape
        for i in (1, 2):
            e = d.halo(i)

            self.assertEqual(e.shape, (shape[0] + i * 2, shape[1] + i * 2))

            # Body
            self.assertTrue(d.equals(e[i:-i, i:-i]))

            # Corners
            self.assertTrue(e[:i, :i].equals(d[:i, :i], verbose=2))
            self.assertTrue(e[:i, -i:].equals(d[:i, -i:], verbose=2))
            self.assertTrue(e[-i:, :i].equals(d[-i:, :i], verbose=2))
            self.assertTrue(e[-i:, -i:].equals(d[-i:, -i:], verbose=2))

        for i in (1, 2):
            e = d.halo(i, axes=0)

            self.assertEqual(e.shape, (shape[0] + i * 2, shape[1]))
            self.assertTrue(d.equals(e[i:-i, :], verbose=2))

        for j, i in zip([1, 1, 2, 2], [1, 2, 1, 2]):
            e = d.halo({0: j, 1: i})

            self.assertEqual(e.shape, (shape[0] + j * 2, shape[1] + i * 2))

            # Body
            self.assertTrue(d.equals(e[j:-j, i:-i], verbose=2))

            # Corners
            self.assertTrue(e[:j, :i].equals(d[:j, :i], verbose=2))
            self.assertTrue(e[:j, -i:].equals(d[:j, -i:], verbose=2))
            self.assertTrue(e[-j:, :i].equals(d[-j:, :i], verbose=2))
            self.assertTrue(e[-j:, -i:].equals(d[-j:, -i:], verbose=2))

        # Tripolar
        for i in (1, 2):
            e = d.halo(i)

            t = d.halo(i, tripolar={"X": 1, "Y": 0})
            self.assertTrue(t[-i:].equals(e[-i:, ::-1], verbose=2))

            t = d.halo(i, tripolar={"X": 1, "Y": 0}, fold_index=0)
            self.assertTrue(t[:i].equals(e[:i, ::-1], verbose=2))

        # Depth too large for axis size
        with self.assertRaises(ValueError):
            d.halo(4)

    def test_Data_mask(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # TODODASK: once test_Data_apply_masking is passing after daskification
        # of apply_masking, might make sense to combine this test with that?

        # Test for a masked Data object (having some masked points)
        a = self.ma
        d = cf.Data(a, units="m")
        self.assertTrue((a == d.array).all())
        self.assertTrue((a.mask == d.mask.array).all())
        self.assertEqual(d.mask.shape, d.shape)
        self.assertEqual(d.mask.dtype, bool)
        self.assertEqual(d.mask.Units, cf.Units(None))
        self.assertTrue(d.mask.hardmask)
        self.assertIn(True, d.mask.array)

        # Test for a non-masked Data object
        a2 = np.arange(-100, 200.0, dtype=float).reshape(3, 4, 5, 5)
        d2 = cf.Data(a2, units="m")
        d2[...] = a2
        self.assertTrue((a2 == d2.array).all())
        self.assertEqual(d2.shape, d2.mask.shape)
        self.assertEqual(d2.mask.dtype, bool)
        self.assertEqual(d2.mask.Units, cf.Units(None))
        self.assertTrue(d2.mask.hardmask)
        self.assertNotIn(True, d2.mask.array)

        # Test for a masked Data object of string type, including chunking
        a3 = np.ma.array(["one", "two", "four"], dtype="S4")
        a3[1] = np.ma.masked
        d3 = cf.Data(a3, "m", chunks=(3,))
        self.assertTrue((a3 == d3.array).all())
        self.assertEqual(d3.shape, d3.mask.shape)
        self.assertEqual(d3.mask.dtype, bool)
        self.assertEqual(d3.mask.Units, cf.Units(None))
        self.assertTrue(d3.mask.hardmask)
        self.assertTrue(d3.mask.array[1], True)

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_apply_masking(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = self.ma
        d = cf.Data(a, units="m")

        b = a.copy()
        e = d.apply_masking()
        self.assertTrue((b == e.array).all())
        self.assertTrue((b.mask == e.mask.array).all())

        b = np.ma.where(a == 0, np.ma.masked, a)
        e = d.apply_masking(fill_values=[0])
        self.assertTrue((b == e.array).all())
        self.assertTrue((b.mask == e.mask.array).all())

        b = np.ma.where((a == 0) | (a == 11), np.ma.masked, a)
        e = d.apply_masking(fill_values=[0, 11])
        self.assertTrue((b == e.array).all())
        self.assertTrue((b.mask == e.mask.array).all())

        b = np.ma.where(a < 30, np.ma.masked, a)
        e = d.apply_masking(valid_min=30)
        self.assertTrue((b == e.array).all())
        self.assertTrue((b.mask == e.mask.array).all())

        b = np.ma.where(a > -60, np.ma.masked, a)
        e = d.apply_masking(valid_max=-60)
        self.assertTrue((b == e.array).all())
        self.assertTrue((b.mask == e.mask.array).all())

        b = np.ma.where((a < -20) | (a > 80), np.ma.masked, a)
        e = d.apply_masking(valid_range=[-20, 80])
        self.assertTrue((b == e.array).all())
        self.assertTrue((b.mask == e.mask.array).all())

        d.set_fill_value(70)

        b = np.ma.where(a == 70, np.ma.masked, a)
        e = d.apply_masking(fill_values=True)
        self.assertTrue((b == e.array).all())
        self.assertTrue((b.mask == e.mask.array).all())

        b = np.ma.where((a == 70) | (a < 20) | (a > 80), np.ma.masked, a)
        e = d.apply_masking(fill_values=True, valid_range=[20, 80])
        self.assertTrue((b == e.array).all())
        self.assertTrue((b.mask == e.mask.array).all())

    def test_Data_convolution_filter(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        #        raise unittest.SkipTest("GSASL has no PLAIN support")
        if not SCIPY_AVAILABLE:
            raise unittest.SkipTest("SciPy must be installed for this test.")

        d = cf.Data(self.ma, units="m", chunks=(2, 4, 5, 3))

        window = [0.1, 0.15, 0.5, 0.15, 0.1]

        e = d.convolution_filter(window=window, axis=-1, inplace=True)
        self.assertIsNone(e)

        d = cf.Data(self.ma, units="m")

        for axis in (0, 1):
            # Test  weights in different modes
            for mode in ("reflect", "constant", "nearest", "wrap"):
                b = convolve1d(self.ma, window, axis=axis, mode=mode)
                e = d.convolution_filter(
                    window, axis=axis, mode=mode, cval=0.0
                )
                self.assertTrue((e.array == b).all())

        for dtype in ("int", "int32", "float", "float32"):
            a = np.ma.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype)
            a[2] = np.ma.masked
            d = cf.Data(a, chunks=(4, 4, 1))
            a = a.astype(float).filled(np.nan)

            for window in ((1, 2, 1), (1, 2, 2, 1), (1, 2, 3, 2, 1)):
                for cval in (0, np.nan):
                    for origin in (-1, 0, 1):
                        b = convolve1d(
                            a,
                            window,
                            axis=0,
                            cval=cval,
                            origin=origin,
                            mode="constant",
                        )
                        e = d.convolution_filter(
                            window,
                            axis=0,
                            cval=cval,
                            origin=origin,
                            mode="constant",
                        )
                        self.assertTrue((e.array == b).all())

    def test_Data_diff(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = np.ma.arange(12.0).reshape(3, 4)
        a[1, 1] = 4.5
        a[2, 2] = 10.5
        a[1, 2] = np.ma.masked

        d = cf.Data(a)

        self.assertTrue((d.array == a).all())

        e = d.copy()
        self.assertIsNone(e.diff(inplace=True))
        self.assertTrue(e.equals(d.diff()))

        for n in (0, 1, 2):
            for axis in (0, 1, -1, -2):
                a_diff = np.diff(a, n=n, axis=axis)
                d_diff = d.diff(n=n, axis=axis)

                self.assertTrue((a_diff == d_diff).all())
                self.assertTrue((a_diff.mask == d_diff.mask).all())

                e = d.copy()
                x = e.diff(n=n, axis=axis, inplace=True)
                self.assertIsNone(x)
                self.assertTrue(e.equals(d_diff))

        d = cf.Data(self.ma, "km")
        for n in (0, 1, 2):
            for axis in (0, 1, 2, 3):
                a_diff = np.diff(self.ma, n=n, axis=axis)
                d_diff = d.diff(n=n, axis=axis)
                self.assertTrue((a_diff == d_diff).all())
                self.assertTrue((a_diff.mask == d_diff.mask).all())

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_ndim'")
    def test_Data_compressed(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = np.ma.arange(12).reshape(3, 4)

        d = cf.Data(a)
        self.assertTrue((d.array == a).all())
        self.assertTrue((a.compressed() == d.compressed()).all())

        e = d.copy()
        x = e.compressed(inplace=True)
        self.assertIsNone(x)
        self.assertTrue(e.equals(d.compressed()))

        a[1, 1] = np.ma.masked
        a[2, 3] = np.ma.masked

        d = cf.Data(a)
        self.assertTrue((d.array == a).all())
        self.assertTrue((d.mask.array == a.mask).all())
        self.assertTrue((a.compressed() == d.compressed()).all())

        e = d.copy()
        x = e.compressed(inplace=True)
        self.assertIsNone(x)
        self.assertTrue(e.equals(d.compressed()))

        d = cf.Data(self.a, "km")
        self.assertTrue((self.a.flatten() == d.compressed()).all())

        d = cf.Data(self.ma, "km")
        self.assertTrue((self.ma.compressed() == d.compressed()).all())

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_shape'")
    def test_Data_stats(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data([[0, 1, 2], [3, -99, 5]], mask=[[0, 0, 0], [0, 1, 0]])

        self.assertIsInstance(d.stats(), dict)
        _ = d.stats(all=True)
        _ = d.stats(mean_of_upper_decile=True, range=False)

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_shape'")
    def test_Data__init__dtype_mask(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for m in (1, 20, True):
            d = cf.Data([[1, 2, 3], [4, 5, 6]], mask=m)
            self.assertFalse(d.count())
            self.assertEqual(d.shape, (2, 3))

        for m in (0, False):
            d = cf.Data([[1, 2, 3], [4, 5, 6]], mask=m)
            self.assertEqual(d.count(), d.size)
            self.assertEqual(d.shape, (2, 3))

        d = cf.Data([[1, 2, 3], [4, 5, 6]], mask=[[0], [1]])
        self.assertEqual(d.count(), 3)
        self.assertEqual(d.shape, (2, 3))

        d = cf.Data([[1, 2, 3], [4, 5, 6]], mask=[0, 1, 1])
        self.assertEqual(d.count(), 2)
        self.assertEqual(d.shape, (2, 3))

        d = cf.Data([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [1, 0, 1]])
        self.assertEqual(d.count(), 3)
        self.assertEqual(d.shape, (2, 3))

        a = np.ma.array(
            [[280.0, -99, -99, -99], [281.0, 279.0, 278.0, 279.0]],
            dtype=float,
            mask=[[0, 1, 1, 1], [0, 0, 0, 0]],
        )

        d = cf.Data([[280, -99, -99, -99], [281, 279, 278, 279]])
        self.assertEqual(d.dtype, np.dtype(int))

        d = cf.Data(
            [[280, -99, -99, -99], [281, 279, 278, 279]],
            dtype=float,
            mask=[[0, 1, 1, 1], [0, 0, 0, 0]],
        )

        self.assertEqual(d.dtype, a.dtype)
        self.assertEqual(d.mask.shape, a.mask.shape)
        self.assertTrue((d.array == a).all())
        self.assertTrue((d.mask.array == np.ma.getmaskarray(a)).all())

        a = np.array(
            [[280.0, -99, -99, -99], [281.0, 279.0, 278.0, 279.0]], dtype=float
        )
        mask = np.ma.masked_all(a.shape).mask

        d = cf.Data([[280, -99, -99, -99], [281, 279, 278, 279]], dtype=float)

        self.assertEqual(d.dtype, a.dtype)
        self.assertEqual(d.mask.shape, mask.shape)
        self.assertTrue((d.array == a).all())
        self.assertTrue((d.mask.array == np.ma.getmaskarray(a)).all())

        # Mask broadcasting
        a = np.ma.array(
            [[280.0, -99, -99, -99], [281.0, 279.0, 278.0, 279.0]],
            dtype=float,
            mask=[[0, 1, 1, 0], [0, 1, 1, 0]],
        )

        d = cf.Data(
            [[280, -99, -99, -99], [281, 279, 278, 279]],
            dtype=float,
            mask=[0, 1, 1, 0],
        )

        self.assertEqual(d.dtype, a.dtype)
        self.assertEqual(d.mask.shape, a.mask.shape)
        self.assertTrue((d.array == a).all())
        self.assertTrue((d.mask.array == np.ma.getmaskarray(a)).all())

    def test_Data_digitize(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for a in [
            np.arange(120).reshape(3, 2, 20),
            np.ma.arange(120).reshape(3, 2, 20),
        ]:

            if np.ma.isMA(a):
                a[0, 1, [2, 5, 6, 7, 8]] = np.ma.masked
                a[2, 0, [12, 14, 17]] = np.ma.masked

            d = cf.Data(a, "km")

            for upper in (False, True):
                for bins in (
                    [2, 6, 10, 50, 100],
                    [[2, 6], [6, 10], [10, 50], [50, 100]],
                ):
                    e = d.digitize(bins, upper=upper, open_ends=True)
                    b = np.digitize(a, [2, 6, 10, 50, 100], right=upper)

                    self.assertTrue((e.array == b).all())
                    self.assertTrue(
                        (np.ma.getmask(e.array) == np.ma.getmask(b)).all()
                    )

                    # TODODASK: Reinstate the following test when
                    #           __sub__, minimum, and maximum have
                    #           been daskified

        #                    e.where(
        #                        cf.set([e.minimum(), e.maximum()]),
        #                        cf.masked,
        #                        e - 1,
        #                        inplace=True,
        #                    )
        #                    f = d.digitize(bins, upper=upper)
        #                    self.assertTrue(e.equals(f, verbose=2))

        # Check returned bins
        bins = [2, 6, 10, 50, 100]
        e, b = d.digitize(bins, return_bins=True)
        self.assertTrue(
            (b.array == [[2, 6], [6, 10], [10, 50], [50, 100]]).all()
        )
        self.assertTrue(b.Units == d.Units)

        # Check digitized units
        self.assertTrue(e.Units == cf.Units(None))

        # Check inplace
        self.assertIsNone(d.digitize(bins, inplace=True))
        self.assertTrue(d.equals(e))

    def test_Data_cumsum(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(self.a)
        e = d.copy()
        f = d.cumsum(axis=0)
        self.assertIsNone(e.cumsum(axis=0, inplace=True))
        self.assertTrue(e.equals(f, verbose=2))

        d = cf.Data(self.a, chunks=3)

        for i in [None] + list(range(d.ndim)):
            b = np.cumsum(self.a, axis=i)
            e = d.cumsum(axis=i)
            self.assertTrue((e.array == b).all())

        d = cf.Data(self.ma, chunks=3)

        for i in [None] + list(range(d.ndim)):
            b = np.cumsum(self.ma, axis=i)
            e = d.cumsum(axis=i)
            self.assertTrue(cf.functions._numpy_allclose(e.array, b))

    def test_Data_flatten(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(self.ma.copy())
        self.assertTrue(d.equals(d.flatten([]), verbose=2))
        self.assertIsNone(d.flatten(inplace=True))

        d = cf.Data(self.ma.copy())

        b = self.ma.flatten()
        for axes in (None, list(range(d.ndim))):
            e = d.flatten(axes)
            self.assertEqual(e.ndim, 1)
            self.assertEqual(e.shape, b.shape)
            self.assertTrue(cf.functions._numpy_allclose(e.array, b))

        for axes in self.axes_combinations:
            e = d.flatten(axes)

            if len(axes) <= 1:
                shape = d.shape
            else:
                shape = [n for i, n in enumerate(d.shape) if i not in axes]
                shape.insert(
                    sorted(axes)[0],
                    np.prod([n for i, n in enumerate(d.shape) if i in axes]),
                )

            self.assertEqual(e.shape, tuple(shape))
            self.assertEqual(e.ndim, d.ndim - len(axes) + 1)
            self.assertEqual(e.size, d.size)

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute 'partitions'")
    def test_Data_CachedArray(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        factor = 0.99999999999999

        cf.tempdir(self.tempdir)

        original_FMF = cf.free_memory_factor(1 - factor)
        d = cf.Data(np.arange(100))
        cf.free_memory_factor(factor)
        _ = d.array

        for partition in d.partitions.flat:
            self.assertTrue(partition.in_cached_file)

        _ = np.arange(1000000).reshape(100, 10000)

        cf.free_memory_factor(1 - factor)
        d = cf.Data(np.arange(10000).reshape(100, 100))
        cf.free_memory_factor(factor)

        _ = d.array

        for partition in d.partitions.flat:
            self.assertTrue(partition.in_cached_file)

        cf.free_memory_factor(original_FMF)

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_cached_arithmetic_units(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(self.a, "m")
        e = cf.Data(self.a, "s")

        f = d / e
        self.assertEqual(f.Units, cf.Units("m s-1"))

        d = cf.Data(self.a, "days since 2000-01-02")
        e = cf.Data(self.a, "days since 1999-01-02")

        f = d - e
        self.assertEqual(f.Units, cf.Units("days"))

        # Repeat with caching partitions to disk
        fmt = cf.constants.CONSTANTS["FM_THRESHOLD"]
        cf.constants.CONSTANTS["FM_THRESHOLD"] = cf.total_memory()

        d = cf.Data(self.a, "m")
        e = cf.Data(self.a, "s")

        f = d / e
        self.assertEqual(f.Units, cf.Units("m s-1"))

        d = cf.Data(self.a, "days since 2000-01-02")
        e = cf.Data(self.a, "days since 1999-01-02")

        f = d - e
        self.assertEqual(f.Units, cf.Units("days"))

        # Reset
        cf.constants.CONSTANTS["FM_THRESHOLD"] = fmt

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_auxiliary_mask'")
    def test_Data_AUXILIARY_MASK(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data()
        self.assertIsNone(d._auxiliary_mask)
        self.assertIsNone(d._auxiliary_mask_return())

        d = cf.Data.empty((90, 60))
        m = np.full(d.shape, fill_value=False, dtype=bool)

        self.assertIsNone(d._auxiliary_mask)
        self.assertEqual(d._auxiliary_mask_return().shape, m.shape)
        self.assertTrue((d._auxiliary_mask_return() == m).all())
        self.assertIsNone(d._auxiliary_mask)

        m[[0, 2, 80], [0, 40, 20]] = True

        d._auxiliary_mask_add_component(cf.Data(m))
        self.assertEqual(len(d._auxiliary_mask), 1)
        self.assertEqual(d._auxiliary_mask_return().shape, m.shape)
        self.assertTrue((d._auxiliary_mask_return() == m).all())

        d = cf.Data.empty((90, 60))
        m = np.full(d.shape, fill_value=False, dtype=bool)

        d = cf.Data.empty((90, 60))
        d._auxiliary_mask_add_component(cf.Data(m[0:1, :]))
        self.assertEqual(len(d._auxiliary_mask), 1)
        self.assertTrue((d._auxiliary_mask_return() == m).all())

        d = cf.Data.empty((90, 60))
        d._auxiliary_mask_add_component(cf.Data(m[:, 0:1]))
        self.assertEqual(len(d._auxiliary_mask), 1)
        self.assertTrue((d._auxiliary_mask_return() == m).all())

        d = cf.Data.empty((90, 60))
        d._auxiliary_mask_add_component(cf.Data(m[:, 0:1]))
        d._auxiliary_mask_add_component(cf.Data(m[0:1, :]))
        self.assertEqual(len(d._auxiliary_mask), 2)
        self.assertEqual(d._auxiliary_mask_return().shape, m.shape)
        self.assertTrue((d._auxiliary_mask_return() == m).all())

        # --------------------------------------------------------
        d = cf.Data(np.arange(120).reshape(30, 4))
        e = cf.Data(np.arange(120, 280).reshape(40, 4))

        fm = cf.Data.full((70, 4), fill_value=False, dtype=bool)

        fm[0, 0] = True
        fm[10, 2] = True
        fm[20, 1] = True

        dm = fm[:30]
        d._auxiliary_mask = [dm]

        f = cf.Data.concatenate([d, e], axis=0)
        self.assertEqual(f.shape, fm.shape)
        self.assertTrue((f._auxiliary_mask_return().array == fm).all())

        # --------------------------------------------------------
        d = cf.Data(np.arange(120).reshape(30, 4))
        e = cf.Data(np.arange(120, 280).reshape(40, 4))

        fm = cf.Data.full((70, 4), False, bool)
        fm[50, 0] = True
        fm[60, 2] = True
        fm[65, 1] = True

        em = fm[30:]
        e._auxiliary_mask = [em]

        f = cf.Data.concatenate([d, e], axis=0)
        self.assertEqual(f.shape, fm.shape)
        self.assertTrue((f._auxiliary_mask_return().array == fm).all())

        # --------------------------------------------------------
        d = cf.Data(np.arange(120).reshape(30, 4))
        e = cf.Data(np.arange(120, 280).reshape(40, 4))

        fm = cf.Data.full((70, 4), False, bool)
        fm[0, 0] = True
        fm[10, 2] = True
        fm[20, 1] = True
        fm[50, 0] = True
        fm[60, 2] = True
        fm[65, 1] = True

        dm = fm[:30]
        d._auxiliary_mask = [dm]
        em = fm[30:]
        e._auxiliary_mask = [em]

        f = cf.Data.concatenate([d, e], axis=0)
        self.assertEqual(f.shape, fm.shape)
        self.assertTrue((f._auxiliary_mask_return().array == fm).all())

    def test_Data__contains__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data([[0, 1, 2], [3, 4, 5]], units="m", chunks=2)

        for value in (
            4,
            4.0,
            cf.Data(3),
            cf.Data(0.005, "km"),
            np.array(2),
            da.from_array(2),
        ):
            self.assertIn(value, d)

        for value in (
            99,
            np.array(99),
            da.from_array(99),
            cf.Data(99, "km"),
            cf.Data(2, "seconds"),
        ):
            self.assertNotIn(value, d)

        for value in (
            [1],
            [[1]],
            [1, 2],
            [[1, 2]],
            np.array([1]),
            np.array([[1]]),
            np.array([1, 2]),
            np.array([[1, 2]]),
            da.from_array([1]),
            da.from_array([[1]]),
            da.from_array([1, 2]),
            da.from_array([[1, 2]]),
            cf.Data([1]),
            cf.Data([[1]]),
            cf.Data([1, 2]),
            cf.Data([[1, 2]]),
            cf.Data([0.005], "km"),
        ):
            with self.assertRaises(TypeError):
                value in d

        # Strings
        d = cf.Data(["foo", "bar"])
        self.assertIn("foo", d)
        self.assertNotIn("xyz", d)

        with self.assertRaises(TypeError):
            ["foo"] in d

    def test_Data_asdata(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(self.ma)

        self.assertIs(d.asdata(d), d)
        self.assertIs(cf.Data.asdata(d), d)
        self.assertIs(d.asdata(d, dtype=d.dtype), d)
        self.assertIs(cf.Data.asdata(d, dtype=d.dtype), d)

        self.assertIsNot(d.asdata(d, dtype="float32"), d)
        self.assertIsNot(cf.Data.asdata(d, dtype="float32"), d)
        self.assertIsNot(d.asdata(d, dtype=d.dtype, copy=True), d)
        self.assertIsNot(cf.Data.asdata(d, dtype=d.dtype, copy=True), d)

        self.assertTrue(
            cf.Data.asdata(cf.Data([1, 2, 3]), dtype=float, copy=True).equals(
                cf.Data([1.0, 2, 3]), verbose=2
            )
        )

        self.assertTrue(
            cf.Data.asdata([1, 2, 3]).equals(cf.Data([1, 2, 3]), verbose=2)
        )
        self.assertTrue(
            cf.Data.asdata([1, 2, 3], dtype=float).equals(
                cf.Data([1.0, 2, 3]), verbose=2
            )
        )

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_ndim'")
    def test_Data_squeeze_insert_dimension(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data([list(range(1000))])
        self.assertEqual(d.shape, (1, 1000))
        e = d.squeeze()
        self.assertEqual(e.shape, (1000,))
        self.assertIsNone(d.squeeze(inplace=True))
        self.assertEqual(d.shape, (1000,))

        d = cf.Data([list(range(1000))])
        d.transpose(inplace=True)
        self.assertEqual(d.shape, (1000, 1))
        e = d.squeeze()
        self.assertEqual(e.shape, (1000,))
        self.assertIsNone(d.squeeze(inplace=True))
        self.assertEqual(d.shape, (1000,))

        d.insert_dimension(0, inplace=True)
        d.insert_dimension(-1, inplace=True)
        self.assertEqual(d.shape, (1, 1000, 1))
        e = d.squeeze()
        self.assertEqual(e.shape, (1000,))
        e = d.squeeze(-1)
        self.assertEqual(e.shape, (1, 1000))
        self.assertIsNone(e.squeeze(0, inplace=True))
        self.assertEqual(e.shape, (1000,))

        d = e
        d.insert_dimension(0, inplace=True)
        d.insert_dimension(-1, inplace=True)
        d.insert_dimension(-1, inplace=True)
        self.assertEqual(d.shape, (1, 1000, 1, 1))
        e = d.squeeze([0, 2])
        self.assertEqual(e.shape, (1000, 1))

        array = np.arange(1000).reshape(1, 100, 10)
        d = cf.Data(array)
        e = d.squeeze()
        f = e.insert_dimension(0)
        a = f.array
        self.assertTrue(np.allclose(a, array))

    def test_Data__getitem__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(np.ma.arange(450).reshape(9, 10, 5), chunks=(4, 5, 1))

        for indices in (
            Ellipsis,
            (slice(None), slice(None)),
            (slice(None), Ellipsis),
            (Ellipsis, slice(None)),
            (Ellipsis, slice(None), Ellipsis),
        ):
            self.assertEqual(d[indices].shape, d.shape)

        for indices in (
            ([1, 3, 4], slice(None), [2, -1]),
            (slice(0, 6, 2), slice(None), [2, -1]),
            (slice(0, 6, 2), slice(None), slice(2, 5, 2)),
            (slice(0, 6, 2), list(range(10)), slice(2, 5, 2)),
        ):
            self.assertEqual(d[indices].shape, (3, 10, 2))

        for indices in (
            (slice(0, 6, 2), -2, [2, -1]),
            (slice(0, 6, 2), -2, slice(2, 5, 2)),
        ):
            self.assertEqual(d[indices].shape, (3, 1, 2))

        for indices in (
            ([1, 3, 4], -2, [2, -1]),
            ([4, 3, 1], -2, [2, -1]),
            ([1, 4, 3], -2, [2, -1]),
            ([4, 1, 4], -2, [2, -1]),
        ):
            e = d[indices]
            self.assertEqual(e.shape, (3, 1, 2))
            self.assertEqual(e._axes, d._axes)

        d.__keepdims_indexing__ = False
        self.assertFalse(d.__keepdims_indexing__)
        for indices in (
            ([1, 3, 4], -2, [2, -1]),
            (slice(0, 6, 2), -2, [2, -1]),
            (slice(0, 6, 2), -2, slice(2, 5, 2)),
            ([1, 4, 3], -2, [2, -1]),
            ([4, 3, 4], -2, [2, -1]),
            ([1, 4, 4], -2, [2, -1]),
        ):
            e = d[indices]
            self.assertFalse(e.__keepdims_indexing__)
            self.assertEqual(e.shape, (3, 2))
            self.assertEqual(e._axes, d._axes[0::2])

        self.assertFalse(d.__keepdims_indexing__)
        d.__keepdims_indexing__ = True
        self.assertTrue(d.__keepdims_indexing__)

        d = cf.Data(np.ma.arange(24).reshape(3, 8))
        e = d[0, 2:4]

        # Cyclic slices
        d = cf.Data(np.ma.arange(24).reshape(3, 8))
        d.cyclic(1)
        self.assertTrue((d[0, :6].array == [[0, 1, 2, 3, 4, 5]]).all())
        e = d[0, -2:4]
        self.assertEqual(e._axes, d._axes)
        self.assertEqual(e.shape, (1, 6))
        self.assertTrue((e[0].array == [[6, 7, 0, 1, 2, 3]]).all())
        self.assertFalse(e.cyclic())

        d.__keepdims_indexing__ = False
        e = d[:, 4]
        self.assertEqual(e.shape, (3,))
        self.assertFalse(e.cyclic())
        self.assertEqual(e._axes, d._axes[0:1])
        d.__keepdims_indexing__ = True

        e = d[0, -2:6]
        self.assertEqual(e.shape, (1, 8))
        self.assertTrue((e[0].array == [[6, 7, 0, 1, 2, 3, 4, 5]]).all())
        self.assertTrue(e.cyclic(), set([1]))

        with self.assertRaises(IndexError):
            # Cyclic slice of non-cyclic axis
            e = d[-1:1]

        d.cyclic(0)
        e = d[-1:1, -2:-4]
        self.assertEqual(e.shape, (2, 6))
        self.assertTrue((e[:, 0].array == [[22], [6]]).all())
        self.assertTrue((e[0].array == [[22, 23, 16, 17, 18, 19]]).all())
        self.assertFalse(e.cyclic())

        e = d[-1:2, -2:4]
        self.assertEqual(e.shape, (3, 6))
        self.assertEqual(e.cyclic(), set([0]))
        e = d[-1:1, -2:6]
        self.assertEqual(e.shape, (2, 8))
        self.assertEqual(e.cyclic(), set([1]))
        e = d[-1:2, -2:6]
        self.assertEqual(e.shape, (3, 8))
        self.assertEqual(e.cyclic(), set([0, 1]))

        d.cyclic(0, False)
        d.__keepdims_indexing__ = False
        e = d[0, :6]
        self.assertFalse(e.__keepdims_indexing__)
        self.assertEqual(e.shape, (6,))
        self.assertTrue((e.array == [0, 1, 2, 3, 4, 5]).all())
        e = d[0, -2:4]
        self.assertEqual(e.shape, (6,))
        self.assertTrue((e.array == [6, 7, 0, 1, 2, 3]).all())
        self.assertFalse(e.cyclic())
        d.__keepdims_indexing__ = True

        # Keepdims indexing
        d = cf.Data([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(d[0].shape, (1, 3))
        self.assertEqual(d[:, 1].shape, (2, 1))
        self.assertEqual(d[0, 1].shape, (1, 1))
        d.__keepdims_indexing__ = False
        self.assertEqual(d[0].shape, (3,))
        self.assertEqual(d[:, 1].shape, (2,))
        self.assertEqual(d[0, 1].shape, ())
        d.__keepdims_indexing__ = True

        # Orthogonal indexing
        self.assertEqual(d[[0], [0, 2]].shape, (1, 2))
        self.assertEqual(d[[0, 1], [0, 2]].shape, (2, 2))
        self.assertEqual(d[[0, 1], [2]].shape, (2, 1))

        # Ancillary masks
        #
        # TODODASK: Test __getitem__ with ancillary masks. Can only do
        #           this when cf.Data.where has been daskified

    def test_Data__setitem__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for hardmask in (False, True):
            a = np.ma.arange(90).reshape(9, 10)
            if hardmask:
                a.harden_mask()
            else:
                a.soften_mask()

            d = cf.Data(a.copy(), "metres", hardmask=hardmask, chunks=(3, 5))

            a[:, 1] = np.ma.masked
            d[:, 1] = cf.masked

            a[0, 2] = -6
            d[0, 2] = -6

            a[0:3, 1] = -1
            d[0:3, 1] = -1

            a[0:2, 3] = -1
            d[0:2, 3] = -1

            a[3, 4:6] = -2
            d[3, 4:6] = -2

            a[0:2, 1:4] = -3
            d[0:2, 1:4] = -3

            a[5:7, [3, 5, 6]] = -4
            d[5:7, [3, 5, 6]] = -4

            a[8, [8, 6, 5]] = -5
            d[8, [8, 6, 5]] = -5

            a[...] = -a
            d[...] = -d

            a[0] = a[2]
            d[0] = d[2]

            self.assertTrue((d.array == a).all())
            self.assertTrue((d.array.mask == a.mask).all())

        # Units
        a = np.ma.arange(90).reshape(9, 10)
        d = cf.Data(a, "metres")
        d[...] = cf.Data(a * 100, "cm")
        self.assertTrue((d.array == a).all())
        self.assertTrue((d.array.mask == a.mask).all())

        # Cyclic axes
        d.cyclic(1)
        self.assertTrue((d[0].array == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).all())
        d[0, -1:1] = [-99, -1]
        self.assertTrue(
            (d[0].array == [-1, 1, 2, 3, 4, 5, 6, 7, 8, -99]).all()
        )
        self.assertEqual(d.cyclic(), set([1]))

        # Multiple list/1-d array indices
        with self.assertRaises(NotImplementedError):
            d[[1, 2], [0, 4, 1]] = 9

        with self.assertRaises(NotImplementedError):
            d[[1], [0, 4, 1]] = 9

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_outerproduct(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(np.arange(1200).reshape(40, 30))

        e = cf.Data(np.arange(5))
        f = d.outerproduct(e)
        self.assertEqual(f.shape, (40, 30, 5))

        e = cf.Data(np.arange(5).reshape(5, 1))
        f = d.outerproduct(e)
        self.assertEqual(f.shape, (40, 30, 5, 1))

        e = cf.Data(np.arange(30).reshape(6, 5))
        f = d.outerproduct(e)
        self.assertEqual(f.shape, (40, 30, 6, 5))

        e = cf.Data(7)
        f = d.outerproduct(e)
        self.assertEqual(f.shape, (40, 30), f.shape)

        e = cf.Data(np.arange(5))
        self.assertIsNone(d.outerproduct(e, inplace=True))
        self.assertEqual(d.shape, (40, 30, 5), d.shape)

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_all(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(np.array([[0] * 1000]))
        self.assertTrue(not d.all())
        d[-1, -1] = 1
        self.assertFalse(d.all())
        d[...] = 1
        self.assertTrue(d.all())
        d[...] = cf.masked
        self.assertTrue(d.all())

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_any(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(np.array([[0] * 1000]))
        self.assertFalse(d.any())
        d[-1, -1] = 1
        self.assertTrue(d.any())
        d[...] = 1
        self.assertTrue(d.any())
        d[...] = cf.masked
        self.assertFalse(d.any())

    def test_Data_array(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Scalar numeric array
        d = cf.Data(9, "km")
        a = d.array
        self.assertEqual(a.shape, ())
        self.assertEqual(a, np.array(9))
        d[...] = cf.masked
        a = d.array
        self.assertEqual(a.shape, ())
        self.assertIs(a[()], np.ma.masked)

        # Non-scalar numeric array
        b = np.arange(24).reshape(2, 1, 3, 4)
        d = cf.Data(b, "km", fill_value=-123)
        a = d.array
        a[0, 0, 0, 0] = -999
        a2 = d.array
        self.assertTrue((a2 == b).all())
        self.assertFalse((a2 == a).all())

        # Fill value
        d[0, 0, 0, 0] = cf.masked
        self.assertEqual(d.array.fill_value, d.fill_value)

        # Date-time array
        d = cf.Data([["2000-12-3 12:00"]], "days since 2000-12-01", dt=True)
        self.assertEqual(d.array, 2.5)

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_binary_mask(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = np.ma.ones((1000,), dtype="int32")
        a[[1, 900]] = np.ma.masked
        a[[0, 10, 910]] = 0

        d = cf.Data(np.arange(1000.0), "radians")
        d[[1, 900]] = cf.masked
        d[[10, 910]] = 0

        b = d.binary_mask

        self.assertEqual(b.Units, cf.Units("1"))
        self.assertEqual(b.dtype, np.dtype("int32"))
        self.assertTrue((b.array == a).all())

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_clip(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        c0 = -53.234
        c1 = 34.345456567

        a = self.a + 0.34567
        ac = np.clip(a, c0, c1)

        d = cf.Data(a, "km")
        self.assertIsNotNone(d.clip(c0, c1))
        self.assertIsNone(d.clip(c0, c1, inplace=True))

        d = cf.Data(a, "km")
        e = d.clip(c0, c1)
        self.assertTrue((e.array == ac).all())

        e = d.clip(c0 * 1000, c1 * 1000, units="m")
        self.assertTrue((e.array == ac).all())

        d.clip(c0 * 100, c1 * 100, units="10m", inplace=True)
        self.assertTrue(d.allclose(ac, rtol=1e-05, atol=1e-08))

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_months_years(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        calendar = "360_day"
        d = cf.Data(
            [1.0, 2],
            units=cf.Units("months since 2000-1-1", calendar=calendar),
        )
        self.assertTrue((d.array == np.array([1.0, 2])).all())
        a = np.array(
            [
                cf.dt(2000, 2, 1, 10, 29, 3, 831223, calendar=calendar),
                cf.dt(2000, 3, 1, 20, 58, 7, 662446, calendar=calendar),
            ]
        )

        self.assertTrue(
            (d.datetime_array == a).all(), "{}, {}".format(d.datetime_array, a)
        )

        calendar = "standard"
        d = cf.Data(
            [1.0, 2],
            units=cf.Units("months since 2000-1-1", calendar=calendar),
        )
        self.assertTrue((d.array == np.array([1.0, 2])).all())
        a = np.array(
            [
                cf.dt(2000, 1, 31, 10, 29, 3, 831223, calendar=calendar),
                cf.dt(2000, 3, 1, 20, 58, 7, 662446, calendar=calendar),
            ]
        )
        self.assertTrue(
            (d.datetime_array == a).all(), "{}, {}".format(d.datetime_array, a)
        )

        calendar = "360_day"
        d = cf.Data(
            [1.0, 2], units=cf.Units("years since 2000-1-1", calendar=calendar)
        )
        self.assertTrue((d.array == np.array([1.0, 2])).all())
        a = np.array(
            [
                cf.dt(2001, 1, 6, 5, 48, 45, 974678, calendar=calendar),
                cf.dt(2002, 1, 11, 11, 37, 31, 949357, calendar=calendar),
            ]
        )
        self.assertTrue(
            (d.datetime_array == a).all(), "{}, {}".format(d.datetime_array, a)
        )

        calendar = "standard"
        d = cf.Data(
            [1.0, 2], units=cf.Units("years since 2000-1-1", calendar=calendar)
        )
        self.assertTrue((d.array == np.array([1.0, 2])).all())
        a = np.array(
            [
                cf.dt(2000, 12, 31, 5, 48, 45, 974678, calendar=calendar),
                cf.dt(2001, 12, 31, 11, 37, 31, 949357, calendar=calendar),
            ]
        )
        self.assertTrue(
            (d.datetime_array == a).all(), "{}, {}".format(d.datetime_array, a)
        )

        d = cf.Data(
            [1.0, 2],
            units=cf.Units("years since 2000-1-1", calendar="360_day"),
        )
        d *= 31

    def test_Data_datetime_array(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Scalar array
        for d, x in zip(
            [
                cf.Data(11292.5, "days since 1970-1-1"),
                cf.Data("2000-12-1 12:00", dt=True),
            ],
            [11292.5, 0],
        ):
            a = d.datetime_array
            self.assertEqual(a.shape, ())
            self.assertEqual(
                a, np.array(cf.dt("2000-12-1 12:00", calendar="standard"))
            )

            a = d.array
            self.assertEqual(a.shape, ())
            self.assertEqual(a, x)

        # Non-scalar array
        for d, x in zip(
            [
                cf.Data([[11292.5, 11293.5]], "days since 1970-1-1"),
                cf.Data([["2000-12-1 12:00", "2000-12-2 12:00"]], dt=True),
            ],
            ([[11292.5, 11293.5]], [[0, 1]]),
        ):
            a = d.datetime_array
            self.assertTrue(
                (
                    a
                    == np.array(
                        [
                            [
                                cf.dt("2000-12-1 12:00", calendar="standard"),
                                cf.dt("2000-12-2 12:00", calendar="standard"),
                            ]
                        ]
                    )
                ).all()
            )

            a = d.array
            self.assertTrue((a == x).all())

    def test_Data_asdatetime_asreftime_isdatetime(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data([[1.93, 5.17]], "days since 2000-12-29")
        self.assertFalse(d._isdatetime())
        self.assertIsNone(d._asreftime(inplace=True))
        self.assertFalse(d._isdatetime())

        e = d._asdatetime()
        self.assertTrue(e._isdatetime())
        self.assertEqual(e.dtype, np.dtype(object))
        self.assertIsNone(e._asdatetime(inplace=True))
        self.assertTrue(e._isdatetime())

        # Round trip
        f = e._asreftime()
        self.assertTrue(f.equals(d))

    def test_Data_ceil(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for x in (1, -1):
            a = 0.9 * x * self.a
            c = np.ceil(a)

            d = cf.Data(a)
            e = d.ceil()
            self.assertIsNone(d.ceil(inplace=True))
            self.assertTrue(d.equals(e, verbose=2))
            self.assertEqual(d.shape, c.shape)
            self.assertTrue((d.array == c).all())

    def test_Data_floor(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for x in (1, -1):
            a = 0.9 * x * self.a
            c = np.floor(a)

            d = cf.Data(a)
            e = d.floor()
            self.assertIsNone(d.floor(inplace=True))
            self.assertTrue(d.equals(e, verbose=2))
            self.assertEqual(d.shape, c.shape)
            self.assertTrue((d.array == c).all())

    def test_Data_trunc(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for x in (1, -1):
            a = 0.9 * x * self.a
            c = np.trunc(a)

            d = cf.Data(a)
            e = d.trunc()
            self.assertIsNone(d.trunc(inplace=True))
            self.assertTrue(d.equals(e, verbose=2))
            self.assertEqual(d.shape, c.shape)
            self.assertTrue((d.array == c).all())

    def test_Data_rint(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for x in (1, -1):
            a = 0.9 * x * self.a
            c = np.rint(a)

            d = cf.Data(a)
            d0 = d.copy()
            e = d.rint()
            x = e.array

            self.assertTrue((x == c).all())
            self.assertTrue(d.equals(d0, verbose=2))
            self.assertIsNone(d.rint(inplace=True))
            self.assertTrue(d.equals(e, verbose=2))
            self.assertEqual(d.shape, c.shape)
            self.assertTrue((d.array == c).all())

    def test_Data_round(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for decimals in range(-8, 8):
            a = self.a + 0.34567
            c = np.round(a, decimals=decimals)

            d = cf.Data(a)
            e = d.round(decimals=decimals)

            self.assertIsNone(d.round(decimals=decimals, inplace=True))

            self.assertTrue(d.equals(e, verbose=2))
            self.assertEqual(d.shape, c.shape)
            self.assertTrue((d.array == c).all())

    def test_Data_datum(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(5, "metre")
        self.assertEqual(d.datum(), 5)
        self.assertEqual(d.datum(0), 5)
        self.assertEqual(d.datum(-1), 5)

        for d in [
            cf.Data([4, 5, 6, 1, 2, 3], "metre"),
            cf.Data([[4, 5, 6], [1, 2, 3]], "metre"),
        ]:
            self.assertEqual(d.datum(0), 4)
            self.assertEqual(d.datum(-1), 3)
            for index in d.ndindex():
                self.assertEqual(d.datum(index), d.array[index].item())
                self.assertEqual(d.datum(*index), d.array[index].item())

        d = cf.Data(5, "metre")
        d[()] = cf.masked
        self.assertIs(d.datum(), cf.masked)
        self.assertIs(d.datum(0), cf.masked)
        self.assertIs(d.datum(-1), cf.masked)

        d = cf.Data([[5]], "metre")
        d[0, 0] = cf.masked
        self.assertIs(d.datum(), cf.masked)
        self.assertIs(d.datum(0), cf.masked)
        self.assertIs(d.datum(-1), cf.masked)
        self.assertIs(d.datum(0, 0), cf.masked)
        self.assertIs(d.datum(-1, 0), cf.masked)
        self.assertIs(d.datum((0, 0)), cf.masked)
        self.assertIs(d.datum([0, -1]), cf.masked)
        self.assertIs(d.datum(-1, -1), cf.masked)

        d = cf.Data([1, 2])
        with self.assertRaises(ValueError):
            d.datum()

        with self.assertRaises(ValueError):
            d.datum(3)

        with self.assertRaises(ValueError):
            d.datum(0, 0)

        d = cf.Data([[1, 2]])
        with self.assertRaises(ValueError):
            d.datum((0,))

    def test_Data_flip(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        array = np.arange(24000).reshape(120, 200)
        d = cf.Data(array.copy(), "metre")

        for axes, indices in zip(
            (0, 1, [0, 1]),
            (
                (slice(None, None, -1), slice(None)),
                (slice(None), slice(None, None, -1)),
                (slice(None, None, -1), slice(None, None, -1)),
            ),
        ):
            array = array[indices]
            d.flip(axes, inplace=True)

        self.assertTrue((d.array == array).all())

        array = np.arange(3 * 4 * 5).reshape(3, 4, 5) + 1
        d = cf.Data(array.copy(), "metre", chunks=-1)

        self.assertEqual(d[0].shape, (1, 4, 5))
        self.assertEqual(d[-1].shape, (1, 4, 5))
        self.assertEqual(d[0].maximum(), 4 * 5)
        self.assertEqual(d[-1].maximum(), 3 * 4 * 5)

        for i in (2, 1):
            e = d.flip(i)
            self.assertEqual(e[0].shape, (1, 4, 5))
            self.assertEqual(e[-1].shape, (1, 4, 5))
            self.assertEqual(e[0].maximum(), 4 * 5)
            self.assertEqual(e[-1].maximum(), 3 * 4 * 5)

        i = 0
        e = d.flip(i)
        self.assertEqual(e[0].shape, (1, 4, 5))
        self.assertEqual(e[-1].shape, (1, 4, 5))
        self.assertEqual(e[0].maximum(), 3 * 4 * 5)
        self.assertEqual(e[-1].maximum(), 4 * 5)

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute 'datum'")
    def test_Data_max(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for pp in (False, True):
            d = cf.Data([[4, 5, 6], [1, 2, 3]], "metre")
            self.assertEqual(
                d.maximum(_preserve_partitions=pp), cf.Data(6, "metre")
            )
            self.assertEqual(d.maximum(_preserve_partitions=pp).datum(), 6)
            d[0, 2] = cf.masked
            self.assertEqual(d.maximum(_preserve_partitions=pp), 5)
            self.assertEqual(d.maximum(_preserve_partitions=pp).datum(), 5)
            self.assertEqual(
                d.maximum(_preserve_partitions=pp), cf.Data(0.005, "km")
            )

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_ndim'")
    def test_Data_min(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for pp in (False, True):
            d = cf.Data([[4, 5, 6], [1, 2, 3]], "metre")
            self.assertEqual(
                d.minimum(_preserve_partitions=pp), cf.Data(1, "metre")
            )
            self.assertEqual(d.minimum(_preserve_partitions=pp).datum(), 1)
            d[1, 0] = cf.masked
            self.assertEqual(d.minimum(_preserve_partitions=pp), 2)
            self.assertEqual(d.minimum(_preserve_partitions=pp).datum(), 2)
            self.assertEqual(
                d.minimum(_preserve_partitions=pp), cf.Data(0.002, "km")
            )

    def test_Data_ndindex(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for d in (
            cf.Data(5, "metre"),
            cf.Data([4, 5, 6, 1, 2, 3], "metre"),
            cf.Data([[4, 5, 6], [1, 2, 3]], "metre"),
        ):
            for i, j in zip(d.ndindex(), np.ndindex(d.shape)):
                self.assertEqual(i, j)

    def test_Data_roll(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = np.arange(10 * 15 * 19).reshape(10, 1, 15, 19)

        d = cf.Data(a.copy())

        e = d.roll(0, 4)
        e.roll(2, 120, inplace=True)
        e.roll(3, -77, inplace=True)

        a = np.roll(a, 4, 0)
        a = np.roll(a, 120, 2)
        a = np.roll(a, -77, 3)

        self.assertEqual(e.shape, a.shape)
        self.assertTrue((a == e.array).all())

        f = e.roll(3, 77)
        f.roll(2, -120, inplace=True)
        f.roll(0, -4, inplace=True)

        self.assertEqual(f.shape, d.shape)
        self.assertTrue(f.equals(d, verbose=2))

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_ndim'")
    def test_Data_swapaxes(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = np.arange(10 * 15 * 19).reshape(10, 1, 15, 19)

        d = cf.Data(a.copy())

        for i in range(-a.ndim, a.ndim):
            for j in range(-a.ndim, a.ndim):
                b = np.swapaxes(a.copy(), i, j)
                e = d.swapaxes(i, j)
                message = "cf.Data.swapaxes({}, {}) failed".format(i, j)
                self.assertEqual(b.shape, e.shape, message)
                self.assertTrue((b == e.array).all(), message)

    def test_Data_transpose(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = np.arange(10 * 15 * 19).reshape(10, 1, 15, 19)

        d = cf.Data(a.copy())

        for indices in (range(a.ndim), range(-a.ndim, 0)):
            for axes in itertools.permutations(indices):
                a = np.transpose(a, axes)
                d.transpose(axes, inplace=True)
                message = (
                    "cf.Data.transpose({}) failed: "
                    "d.shape={}, a.shape={}".format(axes, d.shape, a.shape)
                )
                self.assertEqual(d.shape, a.shape, message)
                self.assertTrue((d.array == a).all(), message)

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_unique(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data([[4, 2, 1], [1, 2, 3]], "metre")
        self.assertTrue((d.unique() == cf.Data([1, 2, 3, 4], "metre")).all())
        d[1, -1] = cf.masked
        self.assertTrue((d.unique() == cf.Data([1, 2, 4], "metre")).all())

    def test_Data_year_month_day_hour_minute_second(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data([[1.901, 5.101]], "days since 2000-12-29")
        self.assertTrue(d.year.equals(cf.Data([[2000, 2001]])))
        self.assertTrue(d.month.equals(cf.Data([[12, 1]])))
        self.assertTrue(d.day.equals(cf.Data([[30, 3]])))
        self.assertTrue(d.hour.equals(cf.Data([[21, 2]])))
        self.assertTrue(d.minute.equals(cf.Data([[37, 25]])))
        self.assertTrue(d.second.equals(cf.Data([[26, 26]])))

        d = cf.Data(
            [[1.901, 5.101]], cf.Units("days since 2000-12-29", "360_day")
        )
        self.assertTrue(d.year.equals(cf.Data([[2000, 2001]])))
        self.assertTrue(d.month.equals(cf.Data([[12, 1]])))
        self.assertTrue(d.day.equals(cf.Data([[30, 4]])))
        self.assertTrue(d.hour.equals(cf.Data([[21, 2]])))
        self.assertTrue(d.minute.equals(cf.Data([[37, 25]])))
        self.assertTrue(d.second.equals(cf.Data([[26, 26]])))

        # Can't get year from data with non-reference time units
        with self.assertRaises(ValueError):
            cf.Data([[1, 2]], units="m").year

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "'NoneType' is not iterable")
    def test_Data_BINARY_AND_UNARY_OPERATORS(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        array = np.arange(3 * 4 * 5).reshape(3, 4, 5) + 1

        arrays = (
            np.arange(3 * 4 * 5).reshape(3, 4, 5) + 1.0,
            np.arange(3 * 4 * 5).reshape(3, 4, 5) + 1,
        )

        for a0 in arrays:
            for a1 in arrays[::-1]:
                d = cf.Data(a0[(slice(None, None, -1),) * a0.ndim], "metre")
                d.flip(inplace=True)
                x = cf.Data(a1, "metre")

                message = "Failed in {!r}+{!r}".format(d, x)
                self.assertTrue(
                    (d + x).equals(cf.Data(a0 + a1, "m"), verbose=1), message
                )
                message = "Failed in {!r}*{!r}".format(d, x)
                self.assertTrue(
                    (d * x).equals(cf.Data(a0 * a1, "m2"), verbose=1), message
                )
                message = "Failed in {!r}/{!r}".format(d, x)
                self.assertTrue(
                    (d / x).equals(cf.Data(a0 / a1, "1"), verbose=1), message
                )
                message = "Failed in {!r}-{!r}".format(d, x)
                self.assertTrue(
                    (d - x).equals(cf.Data(a0 - a1, "m"), verbose=1), message
                )
                message = "Failed in {!r}//{!r}".format(d, x)
                self.assertTrue(
                    (d // x).equals(cf.Data(a0 // a1, "1"), verbose=1), message
                )

                message = "Failed in {!r}.__truediv__//{!r}".format(d, x)
                self.assertTrue(
                    d.__truediv__(x).equals(
                        cf.Data(array.__truediv__(array), "1"), verbose=1
                    ),
                    message,
                )

                message = "Failed in {!r}__rtruediv__{!r}".format(d, x)
                self.assertTrue(
                    d.__rtruediv__(x).equals(
                        cf.Data(array.__rtruediv__(array), "1"), verbose=1
                    ),
                    message,
                )

                try:
                    d ** x
                except Exception:
                    pass
                else:
                    message = "Failed in {!r}**{!r}".format(d, x)
                    self.assertTrue((d ** x).all(), message)
        # --- End: for

        for a0 in arrays:
            d = cf.Data(a0, "metre")
            for x in (2, 2.0):
                message = "Failed in {!r}+{}".format(d, x)
                self.assertTrue(
                    (d + x).equals(cf.Data(a0 + x, "m"), verbose=1), message
                )
                message = "Failed in {!r}*{}".format(d, x)
                self.assertTrue(
                    (d * x).equals(cf.Data(a0 * x, "m"), verbose=1), message
                )
                message = "Failed in {!r}/{}".format(d, x)
                self.assertTrue(
                    (d / x).equals(cf.Data(a0 / x, "m"), verbose=1), message
                )
                message = "Failed in {!r}-{}".format(d, x)
                self.assertTrue(
                    (d - x).equals(cf.Data(a0 - x, "m"), verbose=1), message
                )
                message = "Failed in {!r}//{}".format(d, x)
                self.assertTrue(
                    (d // x).equals(cf.Data(a0 // x, "m"), verbose=1), message
                )
                message = "Failed in {!r}**{}".format(d, x)
                self.assertTrue(
                    (d ** x).equals(cf.Data(a0 ** x, "m2"), verbose=1), message
                )
                message = "Failed in {!r}.__truediv__{}".format(d, x)
                self.assertTrue(
                    d.__truediv__(x).equals(
                        cf.Data(a0.__truediv__(x), "m"), verbose=1
                    ),
                    message,
                )
                message = "Failed in {!r}.__rtruediv__{}".format(d, x)
                self.assertTrue(
                    d.__rtruediv__(x).equals(
                        cf.Data(a0.__rtruediv__(x), "m-1"), verbose=1
                    ),
                    message,
                )

                message = "Failed in {}+{!r}".format(x, d)
                self.assertTrue(
                    (x + d).equals(cf.Data(x + a0, "m"), verbose=1), message
                )
                message = "Failed in {}*{!r}".format(x, d)
                self.assertTrue(
                    (x * d).equals(cf.Data(x * a0, "m"), verbose=1), message
                )
                message = "Failed in {}/{!r}".format(x, d)
                self.assertTrue(
                    (x / d).equals(cf.Data(x / a0, "m-1"), verbose=1), message
                )
                message = "Failed in {}-{!r}".format(x, d)
                self.assertTrue(
                    (x - d).equals(cf.Data(x - a0, "m"), verbose=1), message
                )
                message = "Failed in {}//{!r}\n{!r}\n{!r}".format(
                    x, d, x // d, x // a0
                )
                self.assertTrue(
                    (x // d).equals(cf.Data(x // a0, "m-1"), verbose=1),
                    message,
                )

                try:
                    x ** d
                except Exception:
                    pass
                else:
                    message = "Failed in {}**{!r}".format(x, d)
                    self.assertTrue((x ** d).all(), message)

                a = a0.copy()
                try:
                    a += x
                except TypeError:
                    pass
                else:
                    e = d.copy()
                    e += x
                    message = "Failed in {!r}+={}".format(d, x)
                    self.assertTrue(
                        e.equals(cf.Data(a, "m"), verbose=1), message
                    )

                a = a0.copy()
                try:
                    a *= x
                except TypeError:
                    pass
                else:
                    e = d.copy()
                    e *= x
                    message = "Failed in {!r}*={}".format(d, x)
                    self.assertTrue(
                        e.equals(cf.Data(a, "m"), verbose=1), message
                    )

                a = a0.copy()
                try:
                    a /= x
                except TypeError:
                    pass
                else:
                    e = d.copy()
                    e /= x
                    message = "Failed in {!r}/={}".format(d, x)
                    self.assertTrue(
                        e.equals(cf.Data(a, "m"), verbose=1), message
                    )

                a = a0.copy()
                try:
                    a -= x
                except TypeError:
                    pass
                else:
                    e = d.copy()
                    e -= x
                    message = "Failed in {!r}-={}".format(d, x)
                    self.assertTrue(
                        e.equals(cf.Data(a, "m"), verbose=1), message
                    )

                a = a0.copy()
                try:
                    a //= x
                except TypeError:
                    pass
                else:
                    e = d.copy()
                    e //= x
                    message = "Failed in {!r}//={}".format(d, x)
                    self.assertTrue(
                        e.equals(cf.Data(a, "m"), verbose=1), message
                    )

                a = a0.copy()
                try:
                    a **= x
                except TypeError:
                    pass
                else:
                    e = d.copy()
                    e **= x
                    message = "Failed in {!r}**={}".format(d, x)
                    self.assertTrue(
                        e.equals(cf.Data(a, "m2"), verbose=1), message
                    )

                a = a0.copy()
                try:
                    a.__itruediv__(x)
                except TypeError:
                    pass
                else:
                    e = d.copy()
                    e.__itruediv__(x)
                    message = "Failed in {!r}.__itruediv__({})".format(d, x)
                    self.assertTrue(
                        e.equals(cf.Data(a, "m"), verbose=1), message
                    )
            # --- End: for

            for x in (cf.Data(2, "metre"), cf.Data(2.0, "metre")):
                self.assertTrue(
                    (d + x).equals(cf.Data(a0 + x.datum(), "m"), verbose=1)
                )
                self.assertTrue(
                    (d * x).equals(cf.Data(a0 * x.datum(), "m2"), verbose=1)
                )
                self.assertTrue(
                    (d / x).equals(cf.Data(a0 / x.datum(), "1"), verbose=1)
                )
                self.assertTrue(
                    (d - x).equals(cf.Data(a0 - x.datum(), "m"), verbose=1)
                )
                self.assertTrue(
                    (d // x).equals(cf.Data(a0 // x.datum(), "1"), verbose=1)
                )

                try:
                    d ** x
                except Exception:
                    pass
                else:
                    self.assertTrue(
                        (x ** d).all(), "{}**{}".format(x, repr(d))
                    )

                self.assertTrue(
                    d.__truediv__(x).equals(
                        cf.Data(a0.__truediv__(x.datum()), ""), verbose=1
                    )
                )

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_BROADCASTING(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        A = [
            np.array(3),
            np.array([3]),
            np.array([3]).reshape(1, 1),
            np.array([3]).reshape(1, 1, 1),
            np.arange(5).reshape(5, 1),
            np.arange(5).reshape(1, 5),
            np.arange(5).reshape(1, 5, 1),
            np.arange(5).reshape(5, 1, 1),
            np.arange(5).reshape(1, 1, 5),
            np.arange(25).reshape(1, 5, 5),
            np.arange(25).reshape(5, 1, 5),
            np.arange(25).reshape(5, 5, 1),
            np.arange(125).reshape(5, 5, 5),
        ]

        for a in A:
            for b in A:
                d = cf.Data(a)
                e = cf.Data(b)
                ab = a * b
                de = d * e
                self.assertEqual(de.shape, ab.shape)
                self.assertTrue((de.array == ab).all())

    def test_Data_ERROR(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        return  # !!!!!!

        d = cf.Data([0.0, 1])
        e = cf.Data([1.0, 2])

        oldm = cf.Data.mask_fpe(False)
        olds = cf.Data.seterr("raise")

        with self.assertRaises(FloatingPointError):
            _ = e / d

        with self.assertRaises(FloatingPointError):
            _ = e ** 123456

        cf.Data.mask_fpe(True)
        cf.Data.seterr(all="raise")

        g = cf.Data([-99, 2.0])
        g[0] = cf.masked
        f = e / d
        self.assertTrue(f.equals(g, verbose=2))

        g = cf.Data([1.0, -99])
        g[1] = cf.masked
        f = e ** 123456
        self.assertTrue(f.equals(g, verbose=2))

        cf.Data.mask_fpe(True)
        cf.Data.seterr(all="ignore")
        f = e / d
        f = e ** 123456

        cf.Data.mask_fpe(oldm)
        cf.Data.seterr(**olds)

    def test_Data__len__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        self.assertEqual(3, len(cf.Data([1, 2, 3])))
        self.assertEqual(2, len(cf.Data([[1, 2, 3], [4, 5, 6]])))
        self.assertEqual(1, len(cf.Data([[1, 2, 3]])))

        # len() of unsized object
        with self.assertRaises(TypeError):
            len(cf.Data(1))

    def test_Data__float__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for x in (-1.9, -1.5, -1.4, -1, 0, 1, 1.0, 1.4, 1.9):
            self.assertEqual(float(cf.Data(x)), float(x))
            self.assertEqual(float(cf.Data(x)), float(x))

        with self.assertRaises(TypeError):
            float(cf.Data([1, 2]))

    def test_Data__int__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for x in (-1.9, -1.5, -1.4, -1, 0, 1, 1.0, 1.4, 1.9):
            self.assertEqual(int(cf.Data(x)), int(x))
            self.assertEqual(int(cf.Data(x)), int(x))

        with self.assertRaises(Exception):
            _ = int(cf.Data([1, 2]))

    def test_Data__round__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for ndigits in ([], [0], [1], [2], [3]):
            for x in (
                -1.9123,
                -1.5789,
                -1.4123,
                -1.789,
                0,
                1.123,
                1.0234,
                1.412,
                1.9345,
            ):
                self.assertEqual(
                    round(cf.Data(x), *ndigits), round(x, *ndigits)
                )
                self.assertEqual(
                    round(cf.Data(x), *ndigits), round(x, *ndigits)
                )

        with self.assertRaises(Exception):
            _ = round(cf.Data([1, 2]))

    def test_Data_argmax(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(np.arange(120).reshape(4, 5, 6))

        self.assertEqual(d.argmax().array, 119)

        index = d.argmax(unravel=True)
        self.assertEqual(index, (3, 4, 5))
        self.assertEqual(d[index].array, 119)

        e = d.argmax(axis=1)
        self.assertEqual(e.shape, (4, 6))
        self.assertTrue(
            e.equals(cf.Data.full(shape=(4, 6), fill_value=4, dtype=int))
        )

        self.assertEqual(d[d.argmax(unravel=True)].array, 119)

        d = cf.Data([0, 4, 2, 3, 4])
        self.assertEqual(d.argmax().array, 1)

        # Bad axis
        with self.assertRaises(Exception):
            d.argmax(axis=d.ndim)

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "hits 'NoneType' is not iterable")
    def test_Data__collapse_SHAPE(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = np.arange(-100, 200.0, dtype=float).reshape(3, 4, 5, 5)

        for h in (
            "sample_size",
            "sum",
            "min",
            "max",
            "mean",
            "var",
            "sd",
            "mid_range",
            "range",
            "integral",
            "maximum_absolute_value",
            "minimum_absolute_value",
            "sum_of_squares",
            "root_mean_square",
            "mean_absolute_value",
            "median",
            "mean_of_upper_decile",
            "sum_of_weights",
            "sum_of_weights2",
        ):

            d = cf.Data(a[(slice(None, None, -1),) * a.ndim].copy())
            d.flip(inplace=True)
            _ = cf.Data(self.w.copy())

            shape = list(d.shape)

            for axes in self.axes_combinations:
                e = getattr(d, h)(
                    axes=axes, squeeze=False, _preserve_partitions=False
                )

                shape = list(d.shape)
                for i in axes:
                    shape[i] = 1

                shape = tuple(shape)
                self.assertEqual(
                    e.shape,
                    shape,
                    "{}, axes={}, not squeezed bad shape: {} != {}".format(
                        h, axes, e.shape, shape
                    ),
                )

            for axes in self.axes_combinations:
                e = getattr(d, h)(
                    axes=axes, squeeze=True, _preserve_partitions=False
                )
                shape = list(d.shape)
                for i in sorted(axes, reverse=True):
                    shape.pop(i)

                shape = tuple(shape)
                self.assertEqual(
                    e.shape,
                    shape,
                    "{}, axes={}, squeezed bad shape: {} != {}".format(
                        h, axes, e.shape, shape
                    ),
                )

            e = getattr(d, h)(squeeze=True, _preserve_partitions=False)
            shape = ()
            self.assertEqual(
                e.shape,
                shape,
                "{}, axes={}, squeezed bad shape: {} != {}".format(
                    h, None, e.shape, shape
                ),
            )

            e = getattr(d, h)(squeeze=False, _preserve_partitions=False)
            shape = (1,) * d.ndim
            self.assertEqual(
                e.shape,
                shape,
                "{}, axes={}, not squeezed bad shape: {} != {}".format(
                    h, None, e.shape, shape
                ),
            )
        # --- End: for

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_ndim'")
    def test_Data_max_min_sum_sum_of_squares(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for pp in (True, False):
            # unweighted, unmasked
            d = cf.Data(self.a)
            for _np, h in zip(
                (np.sum, np.amin, np.amax, np.sum),
                ("sum", "min", "max", "sum_of_squares"),
            ):
                for axes in self.axes_combinations:
                    b = reshape_array(self.a, axes)
                    if h == "sum_of_squares":
                        b = b ** 2

                    b = _np(b, axis=-1)
                    e = getattr(d, h)(
                        axes=axes, squeeze=True, _preserve_partitions=pp
                    )
                    self.assertTrue(
                        e.allclose(b, rtol=1e-05, atol=1e-08),
                        "{}, axis={}, unweighted, unmasked "
                        "\ne={}, \nb={}".format(h, axes, e.array, b),
                    )

            # unweighted, masked
            d = cf.Data(self.ma)
            for _np, h in zip(
                (np.ma.sum, np.ma.amin, np.ma.amax, np.ma.sum),
                ("sum", "min", "max", "sum_of_squares"),
            ):
                for axes in self.axes_combinations:
                    b = reshape_array(self.ma, axes)
                    if h == "sum_of_squares":
                        b = b ** 2

                    b = _np(b, axis=-1)
                    b = np.ma.asanyarray(b)
                    e = getattr(d, h)(
                        axes=axes, squeeze=True, _preserve_partitions=pp
                    )

                    self.assertTrue(
                        (e.mask.array == b.mask).all(),
                        "{}, axis={}, \ne.mask={}, \nb.mask={}".format(
                            h, axes, e.mask.array, b.mask
                        ),
                    )

                    self.assertTrue(
                        e.allclose(b, rtol=1e-05, atol=1e-08),
                        "{}, axis={}, unweighted, masked "
                        "\ne={}, \nb={}".format(h, axes, e.array, b),
                    )

    def test_Data_percentile_median(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # ranks: a sequence of percentile rank inputs. NOTE: must
        # include 50 as the last input so that cf.Data.median is also
        # tested correctly.
        ranks = ([30, 60, 90], [90, 30], [20])
        ranks = ranks + (50,)

        d = cf.Data(self.a, chunks=(2, 2, 3, 5))

        for axis in [None] + self.axes_combinations:
            for keepdims in (True, False):
                for q in ranks:
                    a1 = np.percentile(d, q, axis=axis, keepdims=keepdims)
                    b1 = d.percentile(q, axes=axis, squeeze=not keepdims)
                    self.assertEqual(b1.shape, a1.shape)
                    self.assertTrue((b1.array == a1).all())

        # Masked data
        a = self.ma
        filled = np.ma.filled(a, np.nan)
        d = cf.Data(self.ma, chunks=(2, 2, 3, 5))

        with np.testing.suppress_warnings() as sup:
            sup.filter(
                category=RuntimeWarning,
                message=".*All-NaN slice encountered.*",
            )
            sup.filter(
                category=UserWarning,
                message="Warning: 'partition' will ignore the 'mask' of the MaskedArray.*",
            )
            for axis in [None] + self.axes_combinations:
                for keepdims in (True, False):
                    for q in ranks:
                        a1 = np.nanpercentile(
                            filled, q, axis=axis, keepdims=keepdims
                        )
                        mask = np.isnan(a1)
                        if mask.any():
                            a1 = np.ma.masked_where(mask, a1, copy=False)

                        b1 = d.percentile(q, axes=axis, squeeze=not keepdims)
                        self.assertEqual(b1.shape, a1.shape)
                        self.assertTrue((b1.array == a1).all())

        # Test scalar input (not masked)
        a = np.array(9)
        d = cf.Data(a)
        for keepdims in (True, False):
            for q in ranks:
                a1 = np.nanpercentile(a, q, keepdims=keepdims)
                b1 = d.percentile(q, squeeze=not keepdims)
                self.assertEqual(b1.shape, a1.shape)
                self.assertTrue((b1.array == a1).all())

        # Test scalar input (masked)
        a = np.ma.array(9, mask=True)
        filled = np.ma.filled(a.astype(float), np.nan)
        d = cf.Data(a)

        with np.testing.suppress_warnings() as sup:
            sup.filter(
                category=RuntimeWarning,
                message=".*All-NaN slice encountered.*",
            )
            sup.filter(
                category=UserWarning,
                message="Warning: 'partition' will ignore the 'mask' of the MaskedArray.*",
            )
            for keepdims in (True, False):
                for q in ranks:
                    a1 = np.nanpercentile(filled, q, keepdims=keepdims)
                    mask = np.isnan(a1)
                    if mask.any():
                        a1 = np.ma.masked_where(mask, a1, copy=False)

                    b1 = d.percentile(q, squeeze=not keepdims)
                    self.assertEqual(b1.shape, a1.shape)
                    self.assertTrue(
                        (b1.array == a1).all() in (True, np.ma.masked)
                    )

        # Test mtol=1
        d = cf.Data(self.a)
        d[...] = cf.masked  # All masked
        for axis in [None] + self.axes_combinations:
            for q in ranks:
                e = d.percentile(q, axes=axis, mtol=1)
                self.assertFalse(np.ma.count(e.array, keepdims=True).any())

        a = np.ma.arange(12).reshape(3, 4)
        d = cf.Data(a)
        d[1, -1] = cf.masked  # 1 value masked
        for q in ranks:
            e = d.percentile(q, mtol=1)
            self.assertTrue(np.ma.count(e.array, keepdims=True).all())

        # Test mtol=0
        for q in ranks:
            e = d.percentile(q, mtol=0)
            self.assertFalse(np.ma.count(e.array, keepdims=True).any())

        # Test mtol=0.1
        for q in ranks:
            e = d.percentile(q, axes=0, mtol=0.1)
            self.assertEqual(np.ma.count(e.array), 3 * e.shape[0])

        for q in ranks[:-1]:  # axis=1: exclude the non-sequence rank
            e = d.percentile(q, axes=1, mtol=0.1)
            self.assertEqual(np.ma.count(e.array), 2 * e.shape[0])

        q = ranks[-1]  # axis=1: test the non-sequence rank
        e = d.percentile(q, axes=1, mtol=0.1)
        self.assertEqual(np.ma.count(e.array), e.shape[0] - 1)

        # Check invalid ranks (those not in [0, 100])
        for q in (-9, [999], [50, 999], [999, 50]):
            with self.assertRaises(ValueError):
                d.percentile(q).array

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_mean_of_upper_decile(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for pp in (True, False):
            # unweighted, unmasked
            d = cf.Data(self.a)
            for axes in self.axes_combinations:
                b = reshape_array(self.a, axes)
                p = np.percentile(b, 90, axis=-1, keepdims=True)
                b = np.ma.where(b < p, np.ma.masked, b)
                b = np.average(b, axis=-1)

                e = d.mean_of_upper_decile(
                    axes=axes, squeeze=True, _preserve_partitions=pp
                )

                self.assertTrue(
                    e.allclose(b, rtol=1e-05, atol=1e-08),
                    "mean_of_upper_decile, axis={}, unweighted, "
                    "unmasked \ne={}, \nb={}".format(axes, e.array, b),
                )

            # unweighted, masked
            d = cf.Data(self.ma)
            for axes in self.axes_combinations:
                b = reshape_array(self.ma, axes)
                b = np.ma.filled(b, np.nan)
                with np.testing.suppress_warnings() as sup:
                    sup.filter(
                        RuntimeWarning, message=".*All-NaN slice encountered"
                    )
                    p = np.nanpercentile(b, 90, axis=-1, keepdims=True)

                b = np.ma.masked_where(np.isnan(b), b, copy=False)

                p = np.where(np.isnan(p), b.max() + 1, p)

                with np.testing.suppress_warnings() as sup:
                    sup.filter(
                        RuntimeWarning,
                        message=".*invalid value encountered in less",
                    )
                    b = np.ma.where(b < p, np.ma.masked, b)

                b = np.ma.average(b, axis=-1)
                b = np.ma.asanyarray(b)

                e = d.mean_of_upper_decile(
                    axes=axes, squeeze=True, _preserve_partitions=pp
                )

                self.assertTrue(
                    (e.mask.array == b.mask).all(),
                    "mean_of_upper_decile, axis={}, \ne.mask={}, "
                    "\nb.mask={}".format(axes, e.mask.array, b.mask),
                )
                self.assertTrue(
                    e.allclose(b, rtol=1e-05, atol=1e-08),
                    "mean_of_upper_decile, axis={}, "
                    "unweighted, masked "
                    "\ne={}, \nb={}".format(axes, e.array, b),
                )

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_ndim'")
    def test_Data_range_mid_range(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for pp in (True, False):
            # unweighted, unmasked
            d = cf.Data(self.a)
            for h in ("range", "mid_range"):
                for axes in self.axes_combinations:
                    b = reshape_array(self.a, axes)
                    mn = np.amin(b, axis=-1)
                    mx = np.amax(b, axis=-1)
                    if h == "range":
                        b = mx - mn
                    elif h == "mid_range":
                        b = (mx + mn) * 0.5

                    e = getattr(d, h)(
                        axes=axes, squeeze=True, _preserve_partitions=pp
                    )
                    self.assertTrue(
                        e.allclose(b, rtol=1e-05, atol=1e-08),
                        "{}, axis={}, unweighted, unmasked "
                        "\ne={}, \nb={}".format(h, axes, e.array, b),
                    )

            # unweighted, masked
            d = cf.Data(self.ma)
            for h in ("range", "mid_range"):
                for axes in self.axes_combinations:
                    b = reshape_array(self.ma, axes)
                    mn = np.amin(b, axis=-1)
                    mx = np.amax(b, axis=-1)
                    if h == "range":
                        b = mx - mn
                    elif h == "mid_range":
                        b = (mx + mn) * 0.5

                    b = np.ma.asanyarray(b)

                    e = getattr(d, h)(
                        axes=axes, squeeze=True, _preserve_partitions=pp
                    )

                    self.assertTrue(
                        (e.mask.array == b.mask).all(),
                        "{}, axis={}, \ne.mask={}, "
                        "\nb.mask={}".format(h, axes, e.mask.array, b.mask),
                    )

                    self.assertTrue(
                        e.allclose(b, rtol=1e-05, atol=1e-08),
                        "{}, axis={}, unweighted, masked "
                        "\ne={}, \nb={}".format(h, axes, e.array, b),
                    )

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute 'w' for DataTest")
    def test_Data_integral(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for pp in (True, False):
            # unmasked
            d = cf.Data(self.a)
            x = cf.Data(self.w)
            for axes in self.axes_combinations:
                b = reshape_array(self.a, axes)
                v = reshape_array(self.w, axes)
                b = np.sum(b * v, axis=-1)

                e = d.integral(
                    axes=axes, squeeze=True, weights=x, _preserve_partitions=pp
                )

                self.assertTrue(
                    e.allclose(b, rtol=1e-05, atol=1e-08),
                    "axis={}, unmasked \ne={}, \nb={}".format(
                        axes, e.array, b
                    ),
                )

            # masked
            d = cf.Data(self.ma)
            for axes in self.axes_combinations:
                b = reshape_array(self.ma, axes)
                v = reshape_array(self.w, axes)
                b = np.sum(b * v, axis=-1)
                b = np.ma.asanyarray(b)

                e = d.integral(
                    axes=axes, squeeze=True, weights=x, _preserve_partitions=pp
                )

                self.assertTrue(
                    (e.mask.array == b.mask).all(),
                    "axis={} masked, \ne.mask={}, "
                    "\nb.mask={}".format(axes, e.mask.array, b.mask),
                )

                self.assertTrue(
                    e.allclose(b, rtol=1e-05, atol=1e-08),
                    "axis={}, masked \ne={}, \nb={}".format(axes, e.array, b),
                )

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_ndim'")
    def test_Data_sum_of_weights_sum_of_weights2(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for pp in (True, False):
            # unweighted, unmasked
            d = cf.Data(self.a)
            for h in ("sum_of_weights", "sum_of_weights2"):
                for axes in self.axes_combinations:
                    b = reshape_array(self.ones, axes)
                    b = b.sum(axis=-1)
                    e = getattr(d, h)(
                        axes=axes, squeeze=True, _preserve_partitions=pp
                    )

                    self.assertTrue(
                        e.allclose(b, rtol=1e-05, atol=1e-08),
                        "{}, axis={}, unweighted, unmasked, pp={}, "
                        "\ne={}, \nb={}".format(h, axes, pp, e.array, b),
                    )
            # --- End: for

            # unweighted, masked
            d = cf.Data(self.ma)
            for a, h in zip(
                (self.mones, self.mones), ("sum_of_weights", "sum_of_weights2")
            ):
                for axes in self.axes_combinations:
                    b = reshape_array(a, axes)
                    b = np.ma.asanyarray(b.sum(axis=-1))
                    e = getattr(d, h)(
                        axes=axes, squeeze=True, _preserve_partitions=pp
                    )

                    self.assertTrue(
                        (e.mask.array == b.mask).all(),
                        "{}, axis={}, unweighted, masked, pp={}, "
                        "\ne.mask={}, \nb.mask={}".format(
                            h, axes, pp, e.mask.array, b.mask
                        ),
                    )
                    self.assertTrue(
                        e.allclose(b, rtol=1e-05, atol=1e-08),
                        "{}, axis={}, unweighted, masked, pp={}, "
                        "\ne={}, \nb={}".format(h, axes, pp, e.array, b),
                    )
            # --- End: for

            # weighted, masked
            d = cf.Data(self.ma)
            x = cf.Data(self.w)
            for a, h in zip(
                (self.mw, self.mw * self.mw),
                ("sum_of_weights", "sum_of_weights2"),
            ):
                for axes in self.axes_combinations:
                    a = a.copy()
                    a.mask = self.ma.mask
                    b = reshape_array(a, axes)
                    b = np.ma.asanyarray(b.sum(axis=-1))
                    e = getattr(d, h)(
                        axes=axes,
                        weights=x,
                        squeeze=True,
                        _preserve_partitions=pp,
                    )
                    self.assertTrue(
                        (e.mask.array == b.mask).all(),
                        "{}, axis={}, \ne.mask={}, "
                        "\nb.mask={}".format(h, axes, e.mask.array, b.mask),
                    )

                    self.assertTrue(
                        e.allclose(b, rtol=1e-05, atol=1e-08),
                        "{}, axis={}, \ne={}, \nb={}".format(
                            h, axes, e.array, b
                        ),
                    )
            # --- End: for

            # weighted, unmasked
            d = cf.Data(self.a)
            for a, h in zip(
                (self.w, self.w * self.w),
                ("sum_of_weights", "sum_of_weights2"),
            ):
                for axes in self.axes_combinations:
                    b = reshape_array(a, axes)
                    b = b.sum(axis=-1)
                    e = getattr(d, h)(
                        axes=axes,
                        weights=x,
                        squeeze=True,
                        _preserve_partitions=pp,
                    )
                    self.assertTrue(
                        e.allclose(b, rtol=1e-05, atol=1e-08),
                        "{}, axis={}, \ne={}, \nb={}".format(
                            h, axes, e.array, b
                        ),
                    )

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_ndim'")
    def test_Data_mean_mean_absolute_value(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for absolute in (False, True):
            a = self.a
            ma = self.ma
            method = "mean"
            if absolute:
                a = np.absolute(a)
                ma = np.absolute(ma)
                method = "mean_absolute_value"

            # unweighted, unmasked
            d = cf.Data(self.a)
            for axes in self.axes_combinations:
                b = reshape_array(a, axes)
                b = np.mean(b, axis=-1)
                e = getattr(d, method)(axes=axes, squeeze=True)

                self.assertTrue(
                    e.allclose(b, rtol=1e-05, atol=1e-08),
                    "{} axis={}, unweighted, unmasked \ne={}, "
                    "\nb={}".format(method, axes, e.array, b),
                )
            # --- End: for

            # weighted, unmasked
            x = cf.Data(self.w)
            for axes in self.axes_combinations:
                b = reshape_array(a, axes)
                v = reshape_array(self.w, axes)
                b = np.average(b, axis=-1, weights=v)

                e = getattr(d, method)(axes=axes, weights=x, squeeze=True)

                self.assertTrue(
                    e.allclose(b, rtol=1e-05, atol=1e-08),
                    "{} weighted, unmasked axis={}, \ne={}, "
                    "\nb={}".format(method, axes, e.array, b),
                )
            # --- End: for

            # unweighted, masked
            d = cf.Data(self.ma)
            for axes in self.axes_combinations:
                b = reshape_array(ma, axes)
                b = np.ma.average(b, axis=-1)
                b = np.ma.asanyarray(b)

                e = getattr(d, method)(axes=axes, squeeze=True)

                self.assertTrue(
                    (e.mask.array == b.mask).all(),
                    "{} unweighted, masked axis={}, \ne.mask={}, "
                    "\nb.mask={}".format(method, axes, e.mask.array, b.mask),
                )
                self.assertTrue(
                    e.allclose(b, rtol=1e-05, atol=1e-08),
                    "{} unweighted, masked axis={}, \ne={}, "
                    "\nb={}, ".format(method, axes, e.array, b),
                )
            # --- End: for

            # weighted, masked
            for axes in self.axes_combinations:
                b = reshape_array(ma, axes)
                v = reshape_array(self.mw, axes)
                b = np.ma.average(b, axis=-1, weights=v)
                b = np.ma.asanyarray(b)

                e = getattr(d, method)(axes=axes, weights=x, squeeze=True)

                self.assertTrue(
                    (e.mask.array == b.mask).all(),
                    "{} weighted, masked axis={}, \ne.mask={}, "
                    "\nb.mask={}".format(method, axes, e.mask.array, b.mask),
                )

                self.assertTrue(
                    e.allclose(b, rtol=1e-05, atol=1e-08),
                    "{} weighted, masked axis={}, \ne={}, "
                    "\nb={}, ".format(method, axes, e.array, b),
                )
        # --- End: for

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_ndim'")
    def test_Data_root_mean_square(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # unweighted, unmasked
        d = cf.Data(self.a)
        for axes in self.axes_combinations:
            b = reshape_array(self.a, axes) ** 2
            b = np.mean(b, axis=-1) ** 0.5
            e = d.root_mean_square(axes=axes, squeeze=True)
            self.assertTrue(
                e.allclose(b, rtol=1e-05, atol=1e-08),
                "axis={}, unweighted, unmasked \ne={}, "
                "\nb={}".format(axes, e.array, b),
            )
        # --- End: for

        # weighted, unmasked
        x = cf.Data(self.w)
        for axes in self.axes_combinations:
            b = reshape_array(self.a, axes) ** 2
            v = reshape_array(self.w, axes)
            b = np.average(b, axis=-1, weights=v) ** 0.5

            e = d.root_mean_square(axes=axes, weights=x, squeeze=True)

            self.assertTrue(
                e.allclose(b, rtol=1e-05, atol=1e-08),
                "axis={}, weighted, unmasked \ne={}, "
                "\nb={}".format(axes, e.array, b),
            )
        # --- End: for

        # unweighted, masked
        d = cf.Data(self.ma)
        for axes in self.axes_combinations:
            b = reshape_array(self.ma, axes) ** 2
            b = np.ma.average(b, axis=-1)
            b = np.ma.asanyarray(b) ** 0.5

            e = d.root_mean_square(axes=axes, squeeze=True)

            self.assertTrue(
                (e.mask.array == b.mask).all(),
                "axis={}, unweighted, masked \ne.mask={}, "
                "\nb.mask={}, ".format(axes, e.mask.array, b.mask),
            )
            self.assertTrue(
                e.allclose(b, rtol=1e-05, atol=1e-08),
                "axis={}, unweighted, masked \ne={}, "
                "\nb={}, ".format(axes, e.array, b),
            )
        # --- End: for

        # weighted, masked
        for axes in self.axes_combinations:
            b = reshape_array(self.ma, axes) ** 2
            v = reshape_array(self.mw, axes)
            b = np.ma.average(b, axis=-1, weights=v)
            b = np.ma.asanyarray(b) ** 0.5

            e = d.root_mean_square(axes=axes, weights=x, squeeze=True)

            self.assertTrue(
                (e.mask.array == b.mask).all(),
                "axis={}, weighted, masked \ne.mask={}, "
                "\nb.mask={}, ".format(axes, e.mask.array, b.mask),
            )
            self.assertTrue(
                e.allclose(b, rtol=1e-05, atol=1e-08),
                "axis={}, weighted, masked \ne={}, \nb={}, ".format(
                    axes, e.array, b
                ),
            )

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attribute '_ndim'")
    def test_Data_sample_size(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # unmasked
        d = cf.Data(self.a)
        for axes in self.axes_combinations:
            b = reshape_array(self.ones, axes)
            b = b.sum(axis=-1)
            e = d.sample_size(axes=axes, squeeze=True)

            self.assertTrue(
                e.allclose(b, rtol=1e-05, atol=1e-08),
                "axis={}, \ne={}, \nb={}".format(axes, e.array, b),
            )
        # --- End: for

        # masked
        d = cf.Data(self.ma)
        for axes in self.axes_combinations:
            b = reshape_array(self.mones, axes)
            b = b.sum(axis=-1)
            e = d.sample_size(axes=axes, squeeze=True)

            self.assertTrue(
                e.allclose(b, rtol=1e-05, atol=1e-08),
                "axis={}, \ne={}, \nb={}".format(axes, e.array, b),
            )

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'axes_combinations'")
    def test_Data_sd_var(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        ddofs = (0, 1)

        for pp in (False, True):
            # unweighted, unmasked
            d = cf.Data(self.a, units="K")
            for _np, h in zip((np.var, np.std), ("var", "sd")):
                for ddof in ddofs:
                    for axes in self.axes_combinations:
                        b = reshape_array(self.a, axes)
                        b = _np(b, axis=-1, ddof=ddof)
                        e = getattr(d, h)(
                            axes=axes,
                            squeeze=True,
                            ddof=ddof,
                            _preserve_partitions=pp,
                        )
                        self.assertTrue(
                            e.allclose(b, rtol=1e-05, atol=1e-08),
                            "{}, axis={}, unweighted, unmasked pp={}, "
                            "\ne={}, \nb={}".format(h, axes, pp, e.array, b),
                        )
            # --- End: for

            # unweighted, masked
            d = cf.Data(self.ma, units="K")
            for _np, h in zip((np.ma.var, np.ma.std), ("var", "sd")):
                for ddof in ddofs:
                    for axes in self.axes_combinations:
                        b = reshape_array(self.ma, axes)
                        b = _np(b, axis=-1, ddof=ddof)
                        e = getattr(d, h)(
                            axes=axes,
                            squeeze=True,
                            ddof=ddof,
                            _preserve_partitions=pp,
                        )
                        self.assertTrue(
                            e.allclose(b, rtol=1e-05, atol=1e-08),
                            "{}, axis={}, unweighted, masked, pp={}, "
                            "\ne={}, \nb={}".format(h, axes, pp, e.array, b),
                        )
            # --- End: for

            # weighted, unmasked
            d = cf.Data(self.a, units="K")
            x = cf.Data(self.w)
            for h in ("var", "sd"):
                for axes in self.axes_combinations:
                    for ddof in (0, 1):
                        b = reshape_array(self.a, axes)
                        v = reshape_array(self.w, axes)

                        avg = np.average(b, axis=-1, weights=v)
                        if np.ndim(avg) < b.ndim:
                            avg = np.expand_dims(avg, -1)

                        b, V1 = np.average(
                            (b - avg) ** 2, axis=-1, weights=v, returned=True
                        )

                        if ddof == 1:
                            # Calculate the weighted unbiased
                            # variance. The unbiased variance
                            # weighted with _reliability_ weights
                            # is [V1**2/(V1**2-V2)]*var.
                            V2 = np.asanyarray((v * v).sum(axis=-1))
                            b *= V1 * V1 / (V1 * V1 - V2)
                        elif ddof == 0:
                            pass

                        if h == "sd":
                            b **= 0.5

                        b = np.ma.asanyarray(b)

                        e = getattr(d, h)(
                            axes=axes,
                            weights=x,
                            squeeze=True,
                            ddof=ddof,
                            _preserve_partitions=pp,
                        )

                        self.assertTrue(
                            e.allclose(b, rtol=1e-05, atol=1e-08),
                            "{}, axis={}, weighted, unmasked, pp={}, "
                            "ddof={}, \ne={}, \nb={}".format(
                                h, axes, pp, ddof, e.array, b
                            ),
                        )
            # --- End: for

            # weighted, masked
            d = cf.Data(self.ma, units="K")
            x = cf.Data(self.w)
            for h in ("var", "sd"):
                for axes in self.axes_combinations:
                    for ddof in (0, 1):
                        b = reshape_array(self.ma, axes)
                        v = reshape_array(self.mw, axes)

                        not_enough_data = np.ma.count(b, axis=-1) <= ddof

                        avg = np.ma.average(b, axis=-1, weights=v)
                        if np.ndim(avg) < b.ndim:
                            avg = np.expand_dims(avg, -1)

                        b, V1 = np.ma.average(
                            (b - avg) ** 2, axis=-1, weights=v, returned=True
                        )

                        b = np.ma.where(not_enough_data, np.ma.masked, b)

                        if ddof == 1:
                            # Calculate the weighted unbiased
                            # variance. The unbiased variance
                            # weighted with _reliability_ weights
                            # is [V1**2/(V1**2-V2)]*var.
                            V2 = np.asanyarray((v * v).sum(axis=-1))
                            b *= V1 * V1 / (V1 * V1 - V2)
                        elif ddof == 0:
                            pass

                        if h == "sd":
                            b **= 0.5

                        e = getattr(d, h)(
                            axes=axes,
                            weights=x,
                            squeeze=True,
                            ddof=ddof,
                            _preserve_partitions=pp,
                        )

                        if h == "sd":
                            self.assertEqual(e.Units, d.Units)
                        else:
                            self.assertEqual(e.Units, d.Units ** 2)

                        self.assertTrue(
                            (e.mask.array == b.mask).all(),
                            "{}, axis={}, \ne.mask={}, "
                            "\nb.mask={}, ".format(
                                h, axes, e.mask.array, b.mask
                            ),
                        )
                        self.assertTrue(
                            e.allclose(b, rtol=1e-05, atol=1e-08),
                            "{}, axis={}, weighted, masked, pp={}, "
                            "ddof={}, \ne={}, \nb={}".format(
                                h, axes, pp, ddof, e.array, b
                            ),
                        )
        # --- End: for

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "hits unexpected kwarg 'select'")
    def test_Data_dumpd_loadd_dumps(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.read(self.filename)[0].data

        dumpd = d.dumpd()
        self.assertTrue(d.equals(cf.Data(loadd=dumpd), verbose=2))
        self.assertTrue(d.equals(cf.Data(loadd=dumpd), verbose=2))

        d.to_disk()
        self.assertTrue(d.equals(cf.Data(loadd=dumpd), verbose=2))

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "hits unexpected kwarg 'select'")
    def test_Data_section(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename6)[0]
        self.assertEqual(
            list(sorted(f.data.section((1, 2)).keys())),
            [(x, None, None) for x in range(1800)],
        )
        d = cf.Data(np.arange(120).reshape(2, 3, 4, 5))
        x = d.section([1, 3])
        self.assertEqual(len(x), 8)
        e = cf.Data.reconstruct_sectioned_data(x)
        self.assertTrue(e.equals(d))

    @unittest.skipIf(TEST_DASKIFIED_ONLY, "no attr. 'partition_configuration'")
    def test_Data_count(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(ma)
        self.assertEqual(d.count(), 284, d.count())
        self.assertEqual(d.count_masked(), d.size - 284, d.count_masked())

        d = cf.Data(a)
        self.assertEqual(d.count(), d.size)
        self.assertEqual(d.count_masked(), 0)

    def test_Data_exp(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for x in (1, -1):
            a = 0.9 * x * self.ma
            c = np.ma.exp(a)

            d = cf.Data(a)
            e = d.exp()
            self.assertIsNone(d.exp(inplace=True))
            self.assertTrue(d.equals(e, verbose=2))
            self.assertEqual(d.shape, c.shape)
            # The CI at one point gave a failure due to
            # precision with:
            # self.assertTrue((d.array==c).all()) so need a
            # check which accounts for floating point calcs:
            np.testing.assert_allclose(d.array, c)
        # --- End: for

        d = cf.Data(a, "m")
        with self.assertRaises(Exception):
            _ = d.exp()

    def test_Data_func(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = np.array([[np.e, np.e ** 2, np.e ** 3.5], [0, 1, np.e ** -1]])

        # Using sine as an example function to apply
        b = np.sin(a)
        c = cf.Data(a, "s")
        d = c.func(np.sin)
        self.assertTrue((d.array == b).all())
        self.assertEqual(d.shape, b.shape)
        e = c.func(np.cos)
        self.assertFalse((e.array == b).all())

        # Using log2 as an example function to apply
        b = np.log2(a)
        c = cf.Data(a, "s")
        d = c.func(np.log2)
        self.assertTrue((d.array == b).all())
        self.assertEqual(d.shape, b.shape)
        e = c.func(np.log10)
        self.assertFalse((e.array == b).all())

        # Test in-place operation via inplace kwarg
        d = c.func(np.log2, inplace=True)
        self.assertIsNone(d)
        self.assertTrue((c.array == b).all())
        self.assertEqual(c.shape, b.shape)

        # Test the preserve_invalid keyword with function that has a
        # restricted domain and an input that lies outside of the domain.
        a = np.ma.array(
            [0, 0.5, 1, 1.5],  # note arcsin has domain [1, -1]
            mask=[1, 0, 0, 0],
        )
        b = np.arcsin(a)
        c = cf.Data(a, "s")
        d = c.func(np.arcsin)
        self.assertIs(d.array[3], np.ma.masked)
        self.assertTrue((d.array == b).all())
        self.assertEqual(d.shape, b.shape)
        e = c.func(np.arcsin, preserve_invalid=True)
        self.assertIsNot(e.array[3], np.ma.masked)
        self.assertTrue(np.isnan(e[3]))
        self.assertIs(e.array[0], np.ma.masked)

    def test_Data_log(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Test natural log, base e
        a = np.array([[np.e, np.e ** 2, np.e ** 3.5], [0, 1, np.e ** -1]])
        b = np.log(a)
        c = cf.Data(a, "s")
        d = c.log()
        self.assertTrue((d.array == b).all())
        self.assertEqual(d.shape, b.shape)

        # Test in-place operation via inplace kwarg
        d = c.log(inplace=True)
        self.assertIsNone(d)
        self.assertTrue((c.array == b).all())
        self.assertEqual(c.shape, b.shape)

        # Test another base, using 10 as an example (special managed case)
        a = np.array([[10, 100, 10 ** 3.5], [0, 1, 0.1]])
        b = np.log10(a)
        c = cf.Data(a, "s")
        d = c.log(base=10)
        self.assertTrue((d.array == b).all())
        self.assertEqual(d.shape, b.shape)

        # Test an arbitrary base, using 4 (not a special managed case like 10)
        a = np.array([[4, 16, 4 ** 3.5], [0, 1, 0.25]])
        b = np.log(a) / np.log(4)  # the numpy way, using log rules from school
        c = cf.Data(a, "s")
        d = c.log(base=4)
        self.assertTrue((d.array == b).all())
        self.assertEqual(d.shape, b.shape)

        # Check units for general case
        self.assertEqual(d.Units, cf.Units("1"))

        # Text values outside of the restricted domain for a logarithm
        a = np.array([0, -1, -2])
        b = np.log(a)
        c = cf.Data(a)
        d = c.log()
        # Requires assertion form below to test on expected NaN and inf's
        np.testing.assert_equal(d.array, b)
        self.assertEqual(d.shape, b.shape)

    def test_Data_trigonometric_hyperbolic(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Construct all trig. and hyperbolic method names from the 3 roots:
        trig_methods_root = ["sin", "cos", "tan"]
        trig_methods = trig_methods_root + [
            "arc" + method for method in trig_methods_root
        ]
        trig_and_hyperbolic_methods = trig_methods + [
            method + "h" for method in trig_methods
        ]

        for method in trig_and_hyperbolic_methods:
            for x in (1, -1):
                a = 0.9 * x * self.ma

                # Use more appropriate data for testing for inverse methods;
                # apply some trig operation to convert it to valid range:
                if method.startswith("arc"):
                    if method == "arccosh":  # has unusual domain (x >= 1)
                        a = np.cosh(a.data)  # convert non-masked x to >= 1
                    else:  # convert non-masked values x to range |x| < 1
                        a = np.sin(a.data)

                c = getattr(np.ma, method)(a)
                for units in (None, "", "1", "radians", "K"):
                    d = cf.Data(a, units=units)
                    # Suppress warnings that some values are
                    # invalid (NaN, +/- inf) or there is
                    # attempted division by zero, as this is
                    # expected with inverse trig:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        e = getattr(d, method)()
                        self.assertIsNone(getattr(d, method)(inplace=True))

                    self.assertTrue(
                        d.equals(e, verbose=2), "{}".format(method)
                    )
                    self.assertEqual(d.shape, c.shape)
                    self.assertTrue(
                        (d.array == c).all(),
                        "{}, {}, {}, {}".format(method, units, d.array, c),
                    )
                    self.assertTrue(
                        (d.mask.array == c.mask).all(),
                        "{}, {}, {}, {}".format(method, units, d.array, c),
                    )
        # --- End: for

        # Also test masking behaviour: masking of invalid data occurs for
        # np.ma module by default but we don't want that so there is logic
        # to workaround it. So check that invalid values do emerge.
        inverse_methods = [
            method
            for method in trig_and_hyperbolic_methods
            if method.startswith("arc")
        ]

        d = cf.Data([2, 1.5, 1, 0.5, 0], mask=[1, 0, 0, 0, 1])
        for method in inverse_methods:
            with np.errstate(invalid="ignore", divide="ignore"):
                e = getattr(d, method)()
            self.assertTrue(
                (e.mask.array == d.mask.array).all(),
                "{}, {}, {}".format(method, e.array, d),
            )

        # In addition, test that 'nan', inf' and '-inf' emerge distinctly
        f = cf.Data([-2, -1, 1, 2], mask=[0, 0, 0, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            g = f.arctanh().array  # expect [ nan, -inf,  inf,  --]

        self.assertTrue(np.isnan(g[0]))
        self.assertTrue(np.isneginf(g[1]))
        self.assertTrue(np.isposinf(g[2]))
        self.assertIs(g[3], cf.masked)

        # AT2
        #
        # # Treat arctan2 separately (as is a class method & takes two inputs)
        # for x in (1, -1):
        #     a1 = 0.9 * x * self.ma
        #     a2 = 0.5 * x * self.a
        #     # Transform data for 'a' into range more appropriate for inverse:
        #     a1 = np.sin(a1.data)
        #     a2 = np.cos(a2.data)

        #     c = np.ma.arctan2(a1, a2)
        #     for units in (None, '', '1', 'radians', 'K'):
        #         d1 = cf.Data(a1, units=units)
        #         d2 = cf.Data(a2, units=units)
        #         e = cf.Data.arctan2(d1, d2)
        #         # Note: no inplace arg for arctan2 (operates on 2 arrays)
        #         self.assertEqual(d1.shape, c.shape)
        #         self.assertTrue((e.array == c).all())
        #         self.assertTrue((d1.mask.array == c.mask).all())

    def test_Data_filled(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data([[1, 2, 3]])
        self.assertTrue((d.filled().array == [[1, 2, 3]]).all())

        d[0, 0] = cf.masked
        self.assertTrue(
            (d.filled().array == [[-9223372036854775806, 2, 3]]).all()
        )

        d.set_fill_value(-99)
        self.assertTrue((d.filled().array == [[-99, 2, 3]]).all())

        self.assertTrue((d.filled(1e10).array == [[1e10, 2, 3]]).all())

        d = cf.Data(["a", "b", "c"], mask=[1, 0, 0])
        self.assertTrue((d.filled().array == ["", "b", "c"]).all())

    def test_Data_del_units(self):
        d = cf.Data(1)
        with self.assertRaises(ValueError):
            d.del_units()

        d = cf.Data(1, "m")
        self.assertEqual(d.del_units(), "m")
        with self.assertRaises(ValueError):
            d.del_units()

        d = cf.Data(1, "days since 2000-1-1")
        self.assertEqual(d.del_units(), "days since 2000-1-1")
        with self.assertRaises(ValueError):
            d.del_units()

        d = cf.Data(1, "days since 2000-1-1", calendar="noleap")
        self.assertEqual(d.del_units(), "days since 2000-1-1")
        self.assertEqual(d.Units, cf.Units(None, "noleap"))
        with self.assertRaises(ValueError):
            d.del_units()

    def test_Data_del_calendar(self):
        for units in (None, "", "m", "days since 2000-1-1"):
            d = cf.Data(1, units)
            with self.assertRaises(ValueError):
                d.del_calendar()

        d = cf.Data(1, "days since 2000-1-1", calendar="noleap")
        self.assertEqual(d.del_calendar(), "noleap")
        with self.assertRaises(ValueError):
            d.del_calendar()

    def test_Data_get_calendar(self):
        for units in (None, "", "m", "days since 2000-1-1"):
            d = cf.Data(1, units)
            with self.assertRaises(ValueError):
                d.get_calendar()

        d = cf.Data(1, "days since 2000-1-1", calendar="noleap")
        self.assertTrue(d.get_calendar(), "noleap")

    def test_Data_has_units(self):
        d = cf.Data(1, "")
        self.assertTrue(d.has_units())
        d = cf.Data(1, "m")
        self.assertTrue(d.has_units())

        d = cf.Data(1)
        self.assertFalse(d.has_units())
        d = cf.Data(1, calendar="noleap")
        self.assertFalse(d.has_units())

    def test_Data_has_calendar(self):
        d = cf.Data(1, "days since 2000-1-1", calendar="noleap")
        self.assertTrue(d.has_calendar())

        for units in (None, "", "m", "days since 2000-1-1"):
            d = cf.Data(1, units)
            self.assertFalse(d.has_calendar())

    def test_Data_where(self):
        a = np.arange(10)
        d = cf.Data(a)
        b = np.where(a < 5, a, 10 * a)
        e = d.where(a < 5, d, 10 * a)
        self.assertTrue(e.shape == b.shape)
        self.assertTrue((e.array == b).all())

        d = cf.Data(a, "km")
        b = np.where(a < 5, 10 * a, a)
        e = d.where(a < 5, cf.Data(10000 * a, "metre"))
        self.assertTrue(e.shape == b.shape)
        self.assertTrue((e.array == b).all())

        a = np.array([[1, 2], [3, 4]])
        d = cf.Data(a)
        b = np.where([[True, False], [True, True]], a, [[9, 8], [7, 6]])
        e = d.where([[True, False], [True, True]], d, [[9, 8], [7, 6]])
        self.assertTrue(e.shape == b.shape)
        self.assertTrue((e.array == b).all())

        b = np.where([[True, False], [True, True]], [[9, 8], [7, 6]], a)
        e = d.where([[True, False], [True, True]], [[9, 8], [7, 6]])
        self.assertTrue(e.shape == b.shape)
        self.assertTrue((e.array == b).all())

        b = np.where([True, False], [9, 8], a)
        e = d.where([True, False], [9, 8])
        self.assertTrue(e.shape == b.shape)
        self.assertTrue((e.array == b).all())

        a = np.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
        d = cf.Data(a)
        b = np.where(a < 4, a, -1)
        e = d.where(a < 4, d, -1)
        self.assertTrue(e.shape == b.shape)
        self.assertTrue((e.array == b).all())

        x, y = np.ogrid[:3, :4]
        d = cf.Data(x)
        with self.assertRaises(ValueError):
            # Can't change shape
            d.where(x < y, d, 10 + y)

        with self.assertRaises(ValueError):
            # Can't change shape
            d.where(False, d, 10 + y)

        a = np.ma.arange(9, dtype=int).reshape(3, 3)
        d = cf.Data(a, mask=[[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        e = d.where(a > 5, None, -999)
        self.assertTrue(e.shape == d.shape)
        self.assertTrue((e.array.mask == d.array.mask).all())
        self.assertTrue(
            (e.array == [[-999, -999, -999], [5, -999, -999], [6, 7, 8]]).all()
        )

        d.soften_mask()
        e = d.where(a > 5, None, -999)
        self.assertTrue(e.shape == d.shape)
        self.assertTrue((e.array.mask == False).all())
        self.assertTrue(
            (
                e.array == [[-999, -999, -999], [-999, -999, -999], [6, 7, 8]]
            ).all()
        )

        a = np.arange(10)
        d = cf.Data(a)
        e = d.where(a < 5, cf.masked)
        self.assertTrue((e.array.mask == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]).all())
        self.assertTrue((e.array == a).all())

    def test_Data_empty(self):
        for shape, dtype_in, dtype_out in zip(
            [(), (3,), (4, 5)], [None, int, bool], [float, int, bool]
        ):
            d = cf.Data.empty(shape, dtype=dtype_in, chunks=-1)
            self.assertEqual(d.shape, shape)
            self.assertEqual(d.dtype, dtype_out)

    def test_Data_full(self):
        fill_value = 999
        for shape, dtype_in, dtype_out in zip(
            [(), (2,), (4, 5)], [None, float, bool], [int, float, bool]
        ):
            d = cf.Data.full(shape, fill_value, dtype=dtype_in, chunks=-1)
            self.assertEqual(d.shape, shape)
            self.assertEqual(d.dtype, dtype_out)
            self.assertTrue(
                (d.array == np.full(shape, fill_value, dtype=dtype_in)).all()
            )

    def test_Data_ones(self):
        for shape, dtype_in, dtype_out in zip(
            [(), (3,), (4, 5)], [None, int, bool], [float, int, bool]
        ):
            d = cf.Data.ones(shape, dtype=dtype_in, chunks=-1)
            self.assertEqual(d.shape, shape)
            self.assertEqual(d.dtype, dtype_out)
            self.assertTrue((d.array == np.ones(shape, dtype=dtype_in)).all())

    def test_Data_zeros(self):
        for shape, dtype_in, dtype_out in zip(
            [(), (3,), (4, 5)], [None, int, bool], [float, int, bool]
        ):
            d = cf.Data.zeros(shape, dtype=dtype_in, chunks=-1)
            self.assertEqual(d.shape, shape)
            self.assertEqual(d.dtype, dtype_out)
            self.assertTrue((d.array == np.zeros(shape, dtype=dtype_in)).all())

    def test_Data__iter__(self):
        for d in (
            cf.Data([1, 2, 3], "metres"),
            cf.Data([[1, 2], [3, 4]], "metres"),
        ):
            for i, e in enumerate(d):
                self.assertTrue(e.equals(d[i]))

        # iteration over a 0-d Data
        with self.assertRaises(TypeError):
            list(cf.Data(99, "metres"))

    def test_Data__bool__(self):
        for x in (1, 1.5, True, "x"):
            self.assertTrue(bool(cf.Data(x)))
            self.assertTrue(bool(cf.Data([[x]])))

        for x in (0, 0.0, False, ""):
            self.assertFalse(bool(cf.Data(x)))
            self.assertFalse(bool(cf.Data([[x]])))

        with self.assertRaises(ValueError):
            bool(cf.Data([]))

        with self.assertRaises(ValueError):
            bool(cf.Data([1, 2]))

    def test_Data_compute(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Scalar numeric array
        d = cf.Data(9, "km")
        a = d.compute()
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.shape, ())
        self.assertEqual(a, np.array(9))
        d[...] = cf.masked
        a = d.compute()
        self.assertEqual(a.shape, ())
        self.assertIs(a[()], np.ma.masked)

        # Non-scalar numeric array
        b = np.arange(24).reshape(2, 1, 3, 4)
        d = cf.Data(b, "km", fill_value=-123)
        a = d.compute()
        self.assertTrue((a == b).all())

        # Fill value
        d[0, 0, 0, 0] = cf.masked
        self.assertEqual(d.compute().fill_value, d.fill_value)

        # Date-time array
        d = cf.Data([["2000-12-3 12:00"]], "days since 2000-12-01", dt=True)
        self.assertEqual(d.compute(), 2.5)

    def test_Data_persist(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(9, "km")
        self.assertIsNone(d.persist(inplace=True))

        # Scalar numeric array
        d = cf.Data([1, 2, 3.0, 4], "km", mask=[0, 1, 0, 0], chunks=2)
        e = d.persist()
        self.assertIsInstance(e, cf.Data)
        self.assertTrue(e.equals(d))

    def test_Data_cyclic(self):
        d = cf.Data(np.arange(12).reshape(3, 4))
        self.assertEqual(d.cyclic(), set())
        self.assertEqual(d.cyclic(0), set())
        self.assertEqual(d.cyclic(), {0})
        self.assertEqual(d.cyclic(1), {0})
        self.assertEqual(d.cyclic(), {0, 1})
        self.assertEqual(d.cyclic(0, iscyclic=False), {0, 1})
        self.assertEqual(d.cyclic(), {1})
        self.assertEqual(d.cyclic(1, iscyclic=False), {1})
        self.assertEqual(d.cyclic(), set())
        self.assertEqual(d.cyclic([0, 1]), set())
        self.assertEqual(d.cyclic(), {0, 1})
        self.assertEqual(d.cyclic([0, 1], iscyclic=False), {0, 1})
        self.assertEqual(d.cyclic(), set())

        # Invalid axis
        with self.assertRaises(ValueError):
            d.cyclic(2)

        # Scalar data
        d = cf.Data(9)
        self.assertEqual(d.cyclic(), set())

        # Scalar data invalid axis
        with self.assertRaises(ValueError):
            d.cyclic(0)

    def test_Data_change_calendar(self):
        d = cf.Data(
            [0, 1, 2, 3, 4], "days since 2004-02-27", calendar="standard"
        )
        e = d.change_calendar("360_day")
        self.assertTrue(np.allclose(e.array, [0, 1, 2, 4, 5]))
        self.assertEqual(e.Units, cf.Units("days since 2004-02-27", "360_day"))

        # An Exception should be raised when a date is stored that is
        # invalid to the calendar (e.g. 29th of February in the noleap
        # calendar).
        with self.assertRaises(ValueError):
            e = d.change_calendar("noleap").array

    def test_Data_chunks(self):
        dx = da.ones((4, 5), chunks=(2, 4))
        d = cf.Data.ones((4, 5), chunks=(2, 4))
        self.assertEqual(d.chunks, dx.chunks)

    def test_Data_rechunk(self):
        dx = da.ones((4, 5), chunks=(2, 4)).rechunk(-1)
        d = cf.Data.ones((4, 5), chunks=(2, 4)).rechunk(-1)
        self.assertEqual(d.chunks, dx.chunks)

        d = cf.Data.ones((4, 5), chunks=(2, 4))
        e = d.copy()
        self.assertIsNone(e.rechunk(-1, inplace=True))
        self.assertEqual(e.chunks, ((4,), (5,)))
        self.assertTrue(e.equals(d))

    def test_Data_get_units(self):
        for units in ("", "m", "days since 2000-01-01"):
            d = cf.Data(1, units)
            self.assertEqual(d.get_units(), units)

        d = cf.Data(1)
        with self.assertRaises(ValueError):
            d.get_units()

    def test_Data_set_calendar(self):
        d = cf.Data(1, "days since 2000-01-01")
        d.set_calendar("standard")

        with self.assertRaises(ValueError):
            d.set_calendar("noleap")

        d = cf.Data(1, "m")
        d.set_calendar("noleap")
        self.assertEqual(d.Units, cf.Units("m"))

    def test_Data_set_units(self):
        for units in (None, "", "m", "days since 2000-01-01"):
            d = cf.Data(1, units)
            self.assertEqual(d.Units, cf.Units(units))

        d = cf.Data(1, "m")
        d.set_units("km")
        self.assertEqual(d.array, 0.001)

        d = cf.Data(1, "days since 2000-01-01", calendar="noleap")
        d.set_units("days since 1999-12-31")
        self.assertEqual(d.array, 2)

        # Can't set to Units that are not equivalent
        with self.assertRaises(ValueError):
            d.set_units("km")

    def test_Data_to_dask_array(self):
        d = cf.Data([1, 2, 3, 4], "m")
        d.Units = cf.Units("km")
        dx = d.to_dask_array()
        self.assertIsInstance(dx, da.Array)
        self.assertTrue((d.array == dx.compute()).all())
        self.assertIs(da.asanyarray(d), dx)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
