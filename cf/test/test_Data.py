import datetime
import faulthandler
import inspect
import itertools
from operator import mul
import os
import unittest

from functools import reduce

import numpy

SCIPY_AVAILABLE = False
try:
    from scipy.ndimage import convolve1d

    SCIPY_AVAILABLE = True
# not 'except ImportError' as that can hide nested errors, catch anything:
except Exception:
    pass  # test with this dependency will then be skipped by unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


def reshape_array(a, axes):
    new_order = [i for i in range(a.ndim) if i not in axes]
    new_order.extend(axes)
    b = numpy.transpose(a, new_order)
    new_shape = b.shape[: b.ndim - len(axes)]
    new_shape += (reduce(mul, b.shape[b.ndim - len(axes) :]),)
    b = b.reshape(new_shape)
    return b


# Variables for _collapse
a = numpy.arange(-100, 200.0, dtype=float).reshape(3, 4, 5, 5)

w = numpy.arange(1, 301.0, dtype=float).reshape(a.shape)

w[-1, -1, ...] = w[-1, -1, ...] * 2
w /= w.min()

ones = numpy.ones(a.shape, dtype=float)

ma = numpy.ma.arange(-100, 200.0, dtype=float).reshape(3, 4, 5, 5)
ma[:, 1, 4, 4] = numpy.ma.masked
ma[0, :, 2, 3] = numpy.ma.masked
ma[0, 3, :, 3] = numpy.ma.masked
ma[1, 2, 3, :] = numpy.ma.masked


mw = numpy.ma.array(w, mask=ma.mask)

mones = numpy.ma.array(ones, mask=ma.mask)


class DataTest(unittest.TestCase):

    chunk_sizes = (100000, 300, 34)  # 17
    original_chunksize = cf.chunksize()

    axes_permutations = [
        axes
        for n in range(1, a.ndim + 1)
        for axes in itertools.permutations(range(a.ndim), n)
    ]

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
    mw = mw
    ones = ones
    mones = mones

    test_only = []
    #    test_only = ["NOTHING!!!!!"]
    #    test_only = [
    #        'test_Data_percentile',
    #        'test_Data_trigonometric_hyperbolic'
    #        'test_Data_AUXILIARY_MASK',
    #        'test_Data_datum',
    #        'test_Data_ERROR',
    #        'test_Data_array',
    #        'test_Data_varray',
    #        'test_Data_stats',
    #        'test_Data_datetime_array',
    #        'test_Data_cumsum',
    #        'test_Data_dumpd_loadd_dumps',
    #        'test_Data_root_mean_square',
    #        'test_Data_mean_mean_absolute_value',
    #        'test_Data_squeeze_insert_dimension',
    #        'test_Data_months_years',
    #        'test_Data_binary_mask',
    #        'test_Data_CachedArray',
    #        'test_Data_digitize',
    #        'test_Data_outerproduct',
    #        'test_Data_flatten',
    #        'test_Data_transpose',
    #        'test_Data__collapse_SHAPE',
    #        'test_Data_range_mid_range',
    #        'test_Data_median',
    #        'test_Data_mean_of_upper_decile',
    #        'test_Data__init__dtype_mask',
    #    ]

    #    test_only = ['test_Data_mean_mean_absolute_value']
    #    test_only = ['test_Data_AUXILIARY_MASK']
    #    test_only = ['test_Data_mean_of_upper_decile']
    #    test_only = ['test_Data__collapse_SHAPE']
    #    test_only = ['test_Data__collapse_UNWEIGHTED_MASKED']
    #    test_only = ['test_Data__collapse_UNWEIGHTED_UNMASKED']
    #    test_only = ['test_Data__collapse_WEIGHTED_UNMASKED']
    #    test_only = ['test_Data__collapse_WEIGHTED_MASKED']
    #    test_only = ['test_Data_ERROR']
    #    test_only = ['test_Data_diff', 'test_Data_compressed']
    #    test_only = ['test_Data__init__dtype_mask']
    #    test_only = ['test_Data_section']
    #    test_only = ['test_Data_sum_of_weights_sum_of_weights2']
    #    test_only = ['test_Data_max_min_sum_sum_of_squares']
    #    test_only = ['test_Data___setitem__']
    #    test_only = ['test_Data_year_month_day_hour_minute_second']
    #    test_only = ['test_Data_BINARY_AND_UNARY_OPERATORS']
    #    test_only = ['test_Data_clip']
    #    test_only = ['test_Data__init__dtype_mask']

    def test_Data_halo(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(numpy.arange(12).reshape(3, 4), "m")
        d[-1, -1] = cf.masked
        d[1, 1] = cf.masked

        e = d.copy()
        self.assertIsNone(e.halo(1, inplace=True))

        e = d.halo(0)
        self.assertTrue(d.equals(e, verbose=2))

        for i in (1, 2):
            e = d.halo(i)

            self.assertEqual(e.shape, (d.shape[0] + i * 2, d.shape[1] + i * 2))

            # Body
            self.assertTrue(d.equals(e[i:-i, i:-i], verbose=2))

            # Corners
            self.assertTrue(e[:i, :i].equals(d[:i, :i], verbose=2))
            self.assertTrue(e[:i, -i:].equals(d[:i, -i:], verbose=2))
            self.assertTrue(e[-i:, :i].equals(d[-i:, :i], verbose=2))
            self.assertTrue(e[-i:, -i:].equals(d[-i:, -i:], verbose=2))

        for i in (1, 2):
            e = d.halo(i, axes=0)

            self.assertEqual(e.shape, (d.shape[0] + i * 2, d.shape[1]))

            self.assertTrue(d.equals(e[i:-i, :], verbose=2))

        for j, i in zip([1, 1, 2, 2], [1, 2, 1, 2]):
            e = d.halo({0: j, 1: i})

            self.assertEqual(e.shape, (d.shape[0] + j * 2, d.shape[1] + i * 2))

            # Body
            self.assertTrue(d.equals(e[j:-j, i:-i], verbose=2))

            # Corners
            self.assertTrue(e[:j, :i].equals(d[:j, :i], verbose=2))
            self.assertTrue(e[:j, -i:].equals(d[:j, -i:], verbose=2))
            self.assertTrue(e[-j:, :i].equals(d[-j:, :i], verbose=2))
            self.assertTrue(e[-j:, -i:].equals(d[-j:, -i:], verbose=2))

        with self.assertRaises(Exception):
            _ = d.halo(4)

    #     e = d.halo(1, axes=0)
    #
    #    >>> print(e.array)
    #    [[ 0  1  2  3]
    #     [ 0  1  2  3]
    #     [ 4 --  6  7]
    #     [ 8  9 10 --]
    #     [ 8  9 10 --]]
    #    >>> d.equals(e[1:-1, :])
    #    True
    #    >>> f = d.halo({0: 1})
    #    >>> f.equals(e)
    #    True
    #
    #    >>> e = d.halo(1, tripolar={'X': 1, 'Y': 0})
    #    >>> print(e.array)
    #    [[ 0  0  1  2  3  3]
    #     [ 0  0  1  2  3  3]
    #     [ 4  4 --  6  7  7]
    #     [ 8  8  9 10 -- --]
    #     [-- -- 10  9  8  8]]
    #
    #    >>> e = d.halo(1, tripolar={'X': 1, 'Y': 0}, fold_index=0)
    #    >>> print(e.array)
    #    [[ 3  3  2  1  0  0]
    #     [ 0  0  1  2  3  3]
    #     [ 4  4 --  6  7  7]
    #     [ 8  8  9 10 -- --]
    #     [ 8  8  9 10 -- --]]

    def test_Data_apply_masking(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = self.ma

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(a, units="m")

                self.assertTrue((a == d.array).all())
                self.assertTrue((a.mask == d.mask.array).all())

                b = a.copy()
                e = d.apply_masking()
                self.assertTrue((b == e.array).all())
                self.assertTrue((b.mask == e.mask.array).all())

                b = numpy.ma.where(a == 0, numpy.ma.masked, a)
                e = d.apply_masking(fill_values=[0])
                self.assertTrue((b == e.array).all())
                self.assertTrue((b.mask == e.mask.array).all())

                b = numpy.ma.where((a == 0) | (a == 11), numpy.ma.masked, a)
                e = d.apply_masking(fill_values=[0, 11])
                self.assertTrue((b == e.array).all())
                self.assertTrue((b.mask == e.mask.array).all())

                b = numpy.ma.where(a < 30, numpy.ma.masked, a)
                e = d.apply_masking(valid_min=30)
                self.assertTrue((b == e.array).all())
                self.assertTrue((b.mask == e.mask.array).all())

                b = numpy.ma.where(a > -60, numpy.ma.masked, a)
                e = d.apply_masking(valid_max=-60)
                self.assertTrue((b == e.array).all())
                self.assertTrue((b.mask == e.mask.array).all())

                b = numpy.ma.where((a < -20) | (a > 80), numpy.ma.masked, a)
                e = d.apply_masking(valid_range=[-20, 80])
                self.assertTrue((b == e.array).all())
                self.assertTrue((b.mask == e.mask.array).all())

                d.set_fill_value(70)

                b = numpy.ma.where(a == 70, numpy.ma.masked, a)
                e = d.apply_masking(fill_values=True)
                self.assertTrue((b == e.array).all())
                self.assertTrue((b.mask == e.mask.array).all())

                b = numpy.ma.where(
                    (a == 70) | (a < 20) | (a > 80), numpy.ma.masked, a
                )
                e = d.apply_masking(fill_values=True, valid_range=[20, 80])
                self.assertTrue((b == e.array).all())
                self.assertTrue((b.mask == e.mask.array).all())

    def test_Data_convolution_filter(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        #        raise unittest.SkipTest("GSASL has no PLAIN support")
        if not SCIPY_AVAILABLE:
            raise unittest.SkipTest("SciPy must be installed for this test.")

        d = cf.Data(self.ma, units="m")

        window = [0.1, 0.15, 0.5, 0.15, 0.1]

        e = d.convolution_filter(window=window, axis=-1, inplace=True)
        self.assertIsNone(e)

        d = cf.Data(self.ma, units="m")

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                # Test user weights in different modes
                for mode in (
                    "reflect",
                    "constant",
                    "nearest",
                    "mirror",
                    "wrap",
                ):
                    b = convolve1d(d.array, window, axis=-1, mode=mode)
                    e = d.convolution_filter(
                        window=window, axis=-1, mode=mode, cval=0.0
                    )
                    self.assertTrue((e.array == b).all())
        # --- End: for

    def test_Data_diff(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = numpy.ma.arange(12.0).reshape(3, 4)
        a[1, 1] = 4.5
        a[2, 2] = 10.5
        a[1, 2] = numpy.ma.masked

        d = cf.Data(a)

        self.assertTrue((d.array == a).all())

        e = d.copy()
        x = e.diff(inplace=True)
        self.assertIsNone(x)
        self.assertTrue(e.equals(d.diff()))

        for n in (0, 1, 2):
            for axis in (0, 1, -1, -2):
                a_diff = numpy.diff(a, n=n, axis=axis)
                d_diff = d.diff(n=n, axis=axis)

                self.assertTrue((a_diff == d_diff).all())
                self.assertTrue((a_diff.mask == d_diff.mask).all())

                e = d.copy()
                x = e.diff(n=n, axis=axis, inplace=True)
                self.assertIsNone(x)
                self.assertTrue(e.equals(d_diff))
        # --- End: for

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(self.ma, "km")
                for n in (0, 1, 2):
                    for axis in (0, 1, 2, 3):
                        a_diff = numpy.diff(self.ma, n=n, axis=axis)
                        d_diff = d.diff(n=n, axis=axis)
                        self.assertTrue((a_diff == d_diff).all())
                        self.assertTrue((a_diff.mask == d_diff.mask).all())
        # --- End: for

    def test_Data_compressed(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = numpy.ma.arange(12).reshape(3, 4)

        d = cf.Data(a)
        self.assertTrue((d.array == a).all())
        self.assertTrue((a.compressed() == d.compressed()).all())

        e = d.copy()
        x = e.compressed(inplace=True)
        self.assertIsNone(x)
        self.assertTrue(e.equals(d.compressed()))

        a[1, 1] = numpy.ma.masked
        a[2, 3] = numpy.ma.masked

        d = cf.Data(a)
        self.assertTrue((d.array == a).all())
        self.assertTrue((d.mask.array == a.mask).all())
        self.assertTrue((a.compressed() == d.compressed()).all())

        e = d.copy()
        x = e.compressed(inplace=True)
        self.assertIsNone(x)
        self.assertTrue(e.equals(d.compressed()))

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(self.a, "km")
                self.assertTrue((self.a.flatten() == d.compressed()).all())

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(self.ma, "km")
                self.assertTrue((self.ma.compressed() == d.compressed()).all())

    def test_Data_stats(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data([[0, 1, 2], [3, -99, 5]], mask=[[0, 0, 0], [0, 1, 0]])

        self.assertIsInstance(d.stats(), dict)
        _ = d.stats(all=True)
        _ = d.stats(mean_of_upper_decile=True, range=False)

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

        a = numpy.ma.array(
            [[280.0, -99, -99, -99], [281.0, 279.0, 278.0, 279.0]],
            dtype=float,
            mask=[[0, 1, 1, 1], [0, 0, 0, 0]],
        )

        d = cf.Data([[280, -99, -99, -99], [281, 279, 278, 279]])
        self.assertEqual(d.dtype, numpy.dtype(int))

        d = cf.Data(
            [[280, -99, -99, -99], [281, 279, 278, 279]],
            dtype=float,
            mask=[[0, 1, 1, 1], [0, 0, 0, 0]],
        )

        self.assertEqual(d.dtype, a.dtype)
        self.assertEqual(d.mask.shape, a.mask.shape)
        self.assertTrue((d.array == a).all())
        self.assertTrue((d.mask.array == numpy.ma.getmaskarray(a)).all())

        a = numpy.array(
            [[280.0, -99, -99, -99], [281.0, 279.0, 278.0, 279.0]], dtype=float
        )
        mask = numpy.ma.masked_all(a.shape).mask

        d = cf.Data([[280, -99, -99, -99], [281, 279, 278, 279]], dtype=float)

        self.assertEqual(d.dtype, a.dtype)
        self.assertEqual(d.mask.shape, mask.shape)
        self.assertTrue((d.array == a).all())
        self.assertTrue((d.mask.array == numpy.ma.getmaskarray(a)).all())

        # Mask broadcasting
        a = numpy.ma.array(
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
        self.assertTrue((d.mask.array == numpy.ma.getmaskarray(a)).all())

    def test_Data_digitize(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for a in [
            numpy.arange(120).reshape(3, 2, 20),
            numpy.ma.arange(120).reshape(3, 2, 20),
        ]:

            if numpy.ma.isMA(a):
                a[0, 1, [2, 5, 6, 7, 8]] = numpy.ma.masked
                a[2, 0, [12, 14, 17]] = numpy.ma.masked

            for chunksize in self.chunk_sizes:
                with cf.chunksize(chunksize):
                    d = cf.Data(a, "km")

                    for upper in (False, True):
                        for bins in (
                            [2, 6, 10, 50, 100],
                            [[2, 6], [6, 10], [10, 50], [50, 100]],
                        ):
                            e = d.digitize(bins, upper=upper, open_ends=True)
                            b = numpy.digitize(
                                a, [2, 6, 10, 50, 100], right=upper
                            )

                            self.assertTrue((e.array == b).all())

                            e.where(
                                cf.set([e.minimum(), e.maximum()]),
                                cf.masked,
                                e - 1,
                                inplace=True,
                            )
                            f = d.digitize(bins, upper=upper)
                            self.assertTrue(e.equals(f, verbose=2))

    def test_Data_cumsum(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(self.a)
        e = d.copy()
        f = d.cumsum(axis=0)
        self.assertIsNone(e.cumsum(axis=0, inplace=True))
        self.assertTrue(e.equals(f, verbose=2))

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(self.a)

                for i in range(d.ndim):
                    b = numpy.cumsum(self.a, axis=i)
                    e = d.cumsum(axis=i)
                    self.assertTrue((e.array == b).all())

                d = cf.Data(self.ma)

                for i in range(d.ndim):
                    b = numpy.cumsum(self.ma, axis=i)
                    e = d.cumsum(axis=i, masked_as_zero=False)
                    self.assertTrue(cf.functions._numpy_allclose(e.array, b))

    def test_Data_flatten(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data(self.ma.copy())
        self.assertTrue(d.equals(d.flatten([]), verbose=2))
        self.assertIsNone(d.flatten(inplace=True))

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
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
                        shape = [
                            n for i, n in enumerate(d.shape) if i not in axes
                        ]
                        shape.insert(
                            sorted(axes)[0],
                            numpy.prod(
                                [n for i, n in enumerate(d.shape) if i in axes]
                            ),
                        )

                    self.assertEqual(e.shape, tuple(shape))
                    self.assertEqual(e.ndim, d.ndim - len(axes) + 1)
                    self.assertEqual(e.size, d.size)

    def test_Data_CachedArray(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        factor = 0.99999999999999

        cf.tempdir(self.tempdir)

        original_FMF = cf.free_memory_factor(1 - factor)
        d = cf.Data(numpy.arange(100))
        cf.free_memory_factor(factor)
        _ = d.array

        for partition in d.partitions.flat:
            self.assertTrue(partition.in_cached_file)

        _ = numpy.arange(1000000).reshape(100, 10000)

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                cf.free_memory_factor(1 - factor)
                d = cf.Data(numpy.arange(10000).reshape(100, 100))
                cf.free_memory_factor(factor)

                _ = d.array

                for partition in d.partitions.flat:
                    self.assertTrue(partition.in_cached_file)
        # --- End: for

        cf.free_memory_factor(original_FMF)

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

    def test_Data_AUXILIARY_MASK(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data()
        self.assertIsNone(d._auxiliary_mask)
        self.assertIsNone(d._auxiliary_mask_return())

        d = cf.Data.empty((90, 60))
        m = numpy.full(d.shape, fill_value=False, dtype=bool)

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
        m = numpy.full(d.shape, fill_value=False, dtype=bool)

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

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                # --------------------------------------------------------
                d = cf.Data(numpy.arange(120).reshape(30, 4))
                e = cf.Data(numpy.arange(120, 280).reshape(40, 4))

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
                d = cf.Data(numpy.arange(120).reshape(30, 4))
                e = cf.Data(numpy.arange(120, 280).reshape(40, 4))

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
                d = cf.Data(numpy.arange(120).reshape(30, 4))
                e = cf.Data(numpy.arange(120, 280).reshape(40, 4))

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

    def test_Data___contains__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data([[0.0, 1, 2], [3, 4, 5]], units="m")
                self.assertIn(4, d)
                self.assertNotIn(40, d)
                self.assertIn(cf.Data(3), d)
                self.assertIn(cf.Data([[[[3]]]]), d)
                value = d[1, 2]
                value.Units *= 2
                value.squeeze(0)
                self.assertIn(value, d)
                self.assertIn(numpy.array([[[2]]]), d)

    def test_Data_asdata(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(self.ma)

                self.assertIs(d.asdata(d), d)
                self.assertIs(cf.Data.asdata(d), d)
                self.assertIs(d.asdata(d, dtype=d.dtype), d)
                self.assertIs(cf.Data.asdata(d, dtype=d.dtype), d)

                self.assertIsNot(d.asdata(d, dtype="float32"), d)
                self.assertIsNot(cf.Data.asdata(d, dtype="float32"), d)
                self.assertIsNot(d.asdata(d, dtype=d.dtype, copy=True), d)
                self.assertIsNot(
                    cf.Data.asdata(d, dtype=d.dtype, copy=True), d
                )

                self.assertTrue(
                    cf.Data.asdata(
                        cf.Data([1, 2, 3]), dtype=float, copy=True
                    ).equals(cf.Data([1.0, 2, 3]), verbose=2)
                )

                self.assertTrue(
                    cf.Data.asdata([1, 2, 3]).equals(
                        cf.Data([1, 2, 3]), verbose=2
                    )
                )
                self.assertTrue(
                    cf.Data.asdata([1, 2, 3], dtype=float).equals(
                        cf.Data([1.0, 2, 3]), verbose=2
                    )
                )

    def test_Data_squeeze_insert_dimension(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
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
                self.assertEqual(
                    e.shape,
                    (
                        1,
                        1000,
                    ),
                )
                self.assertIsNone(e.squeeze(0, inplace=True))
                self.assertEqual(e.shape, (1000,))

                d = e
                d.insert_dimension(0, inplace=True)
                d.insert_dimension(-1, inplace=True)
                d.insert_dimension(-1, inplace=True)
                self.assertEqual(d.shape, (1, 1000, 1, 1))
                e = d.squeeze([0, 2])
                self.assertEqual(e.shape, (1000, 1))

                array = numpy.arange(1000).reshape(1, 100, 10)
                d = cf.Data(array)
                e = d.squeeze()
                f = e.insert_dimension(0)
                a = f.array
                self.assertTrue(numpy.allclose(a, array))

    def test_Data___getitem__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

    def test_Data___setitem__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                for hardmask in (False, True):
                    a = numpy.ma.arange(3000).reshape(50, 60)
                    if hardmask:
                        a.harden_mask()
                    else:
                        a.soften_mask()

                    d = cf.Data(a.filled(), "m")
                    d.hardmask = hardmask

                    for n, (j, i) in enumerate(
                        (
                            (34, 23),
                            (0, 0),
                            (-1, -1),
                            (slice(40, 50), slice(58, 60)),
                            (Ellipsis, slice(None)),
                            (slice(None), Ellipsis),
                        )
                    ):
                        n = -n - 1
                        for dvalue, avalue in (
                            (n, n),
                            (cf.masked, numpy.ma.masked),
                            (n, n),
                        ):
                            message = (
                                "hardmask={}, "
                                "cf.Data[{}, {}]]={}={} failed".format(
                                    hardmask, j, i, dvalue, avalue
                                )
                            )
                            d[j, i] = dvalue
                            a[j, i] = avalue

                            self.assertIn(
                                (d.array == a).all(),
                                (True, numpy.ma.masked),
                                message,
                            )
                            self.assertTrue(
                                (
                                    d.mask.array == numpy.ma.getmaskarray(a)
                                ).all(),
                                "d.mask.array={!r} \n"
                                "numpy.ma.getmaskarray(a)={!r}".format(
                                    d.mask.array, numpy.ma.getmaskarray(a)
                                ),
                            )
                    # --- End: for

                    a = numpy.ma.arange(3000).reshape(50, 60)
                    if hardmask:
                        a.harden_mask()
                    else:
                        a.soften_mask()

                    d = cf.Data(a.filled(), "m")
                    d.hardmask = hardmask

                    (j, i) = (slice(0, 2), slice(0, 3))
                    array = numpy.array([[1, 2, 6], [3, 4, 5]]) * -1
                    for dvalue in (
                        array,
                        numpy.ma.masked_where(array < -2, array),
                        array,
                    ):
                        message = "cf.Data[{}, {}]={} failed".format(
                            j, i, dvalue
                        )
                        d[j, i] = dvalue
                        a[j, i] = dvalue

                        self.assertIn(
                            (d.array == a).all(),
                            (True, numpy.ma.masked),
                            message,
                        )
                        self.assertTrue(
                            (d.mask.array == numpy.ma.getmaskarray(a)).all(),
                            message,
                        )

    def test_Data_outerproduct(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(numpy.arange(1200).reshape(40, 30))

                e = cf.Data(numpy.arange(5))
                f = d.outerproduct(e)
                self.assertEqual(f.shape, (40, 30, 5))

                e = cf.Data(numpy.arange(5).reshape(5, 1))
                f = d.outerproduct(e)
                self.assertEqual(f.shape, (40, 30, 5, 1))

                e = cf.Data(numpy.arange(30).reshape(6, 5))
                f = d.outerproduct(e)
                self.assertEqual(f.shape, (40, 30, 6, 5))

                e = cf.Data(7)
                f = d.outerproduct(e)
                self.assertEqual(f.shape, (40, 30), f.shape)

                e = cf.Data(numpy.arange(5))
                self.assertIsNone(d.outerproduct(e, inplace=True))
                self.assertEqual(d.shape, (40, 30, 5), d.shape)

    def test_Data_all(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(numpy.array([[0] * 1000]))
                self.assertTrue(not d.all())
                d[-1, -1] = 1
                self.assertFalse(d.all())
                d[...] = 1
                self.assertTrue(d.all())
                d[...] = cf.masked
                self.assertTrue(d.all())

    def test_Data_any(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(numpy.array([[0] * 1000]))
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
        self.assertEqual(a, numpy.array(9))
        d[...] = cf.masked
        a = d.array
        self.assertEqual(a.shape, ())
        self.assertIs(a[()], numpy.ma.masked)

        # Non-scalar numeric array
        b = numpy.arange(10 * 15 * 19).reshape(10, 1, 15, 19)
        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(b, "km")
                a = d.array
                a[0, 0, 0, 0] = -999
                a2 = d.array
                self.assertEqual(a2[0, 0, 0, 0], 0)
                self.assertEqual(a2.shape, b.shape)
                self.assertTrue((a2 == b).all())
                self.assertFalse((a2 == a).all())

                d = cf.Data(
                    [["2000-12-3 12:00"]], "days since 2000-12-01", dt=True
                )
                a = d.array
                self.assertTrue((a == numpy.array([[2.5]])).all())

    def test_Data_binary_mask(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = numpy.ma.ones((1000,), dtype="int32")
        a[[1, 900]] = numpy.ma.masked
        a[[0, 10, 910]] = 0

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(numpy.arange(1000.0), "radians")
                d[[1, 900]] = cf.masked
                d[[10, 910]] = 0

                b = d.binary_mask

                self.assertEqual(b.Units, cf.Units("1"))
                self.assertEqual(b.dtype, numpy.dtype("int32"))
                self.assertTrue((b.array == a).all())

    def test_Data_clip(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        c0 = -53.234
        c1 = 34.345456567

        a = self.a + 0.34567
        ac = numpy.clip(a, c0, c1)

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
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

    def test_Data_months_years(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        calendar = "360_day"
        d = cf.Data(
            [1.0, 2],
            units=cf.Units("months since 2000-1-1", calendar=calendar),
        )
        self.assertTrue((d.array == numpy.array([1.0, 2])).all())
        a = numpy.array(
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
        self.assertTrue((d.array == numpy.array([1.0, 2])).all())
        a = numpy.array(
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
        self.assertTrue((d.array == numpy.array([1.0, 2])).all())
        a = numpy.array(
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
        self.assertTrue((d.array == numpy.array([1.0, 2])).all())
        a = numpy.array(
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
                a, numpy.array(cf.dt("2000-12-1 12:00", calendar="standard"))
            )

            a = d.array
            self.assertEqual(a.shape, ())
            self.assertEqual(a, x)

            a = d.datetime_array
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
            a = d.array
            self.assertTrue((a == x).all())
            a = d.datetime_array
            a = d.array
            self.assertTrue((a == x).all())
            a = d.datetime_array
            self.assertTrue(
                (
                    a
                    == numpy.array(
                        [
                            [
                                cf.dt("2000-12-1 12:00", calendar="standard"),
                                cf.dt("2000-12-2 12:00", calendar="standard"),
                            ]
                        ]
                    )
                ).all()
            )

    def test_Data__asdatetime__asreftime__isdatetime(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data([[1.93, 5.17]], "days since 2000-12-29")
                self.assertEqual(d.dtype, numpy.dtype(float))
                self.assertFalse(d._isdatetime())

                self.assertIsNone(d._asreftime(inplace=True))
                self.assertEqual(d.dtype, numpy.dtype(float))
                self.assertFalse(d._isdatetime())

                self.assertIsNone(d._asdatetime(inplace=True))
                self.assertEqual(d.dtype, numpy.dtype(object))
                self.assertTrue(d._isdatetime())

                self.assertIsNone(d._asdatetime(inplace=True))
                self.assertEqual(d.dtype, numpy.dtype(object))
                self.assertTrue(d._isdatetime())

                self.assertIsNone(d._asreftime(inplace=True))
                self.assertEqual(d.dtype, numpy.dtype(float))
                self.assertFalse(d._isdatetime())

    def test_Data_ceil(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for x in (1, -1):
            a = 0.9 * x * self.a
            c = numpy.ceil(a)

            for chunksize in self.chunk_sizes:
                with cf.chunksize(chunksize):
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
            c = numpy.floor(a)

            for chunksize in self.chunk_sizes:
                with cf.chunksize(chunksize):
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
            c = numpy.trunc(a)

            for chunksize in self.chunk_sizes:
                with cf.chunksize(chunksize):
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
            c = numpy.rint(a)

            for chunksize in self.chunk_sizes:
                with cf.chunksize(chunksize):
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
            c = numpy.round(a, decimals=decimals)

            for chunksize in self.chunk_sizes:
                with cf.chunksize(chunksize):
                    d = cf.Data(a)
                    e = d.round(decimals=decimals)

                    self.assertIsNone(d.round(decimals=decimals, inplace=True))

                    self.assertTrue(d.equals(e, verbose=2))
                    self.assertEqual(d.shape, c.shape)
                    self.assertTrue((d.array == c).all())

    def test_Data_datum(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
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
                        self.assertEqual(
                            d.datum(*index),
                            d.array[index].item(),
                            "{}, {}".format(
                                d.datum(*index), d.array[index].item()
                            ),
                        )
                # --- End: for

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

    def test_Data_flip(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                array = numpy.arange(24000).reshape(120, 200)
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
        # --- End: for

        array = numpy.arange(3 * 4 * 5).reshape(3, 4, 5) + 1
        d = cf.Data(array.copy(), "metre", chunk=False)
        d.chunk(total=[0], omit_axes=[1, 2])

        self.assertEqual(d._pmshape, (3,))
        self.assertEqual(d[0].shape, (1, 4, 5))
        self.assertEqual(d[-1].shape, (1, 4, 5))
        self.assertEqual(d[0].maximum(), 4 * 5)
        self.assertEqual(d[-1].maximum(), 3 * 4 * 5)

        for i in (2, 1):
            e = d.flip(i)
            self.assertEqual(e._pmshape, (3,))
            self.assertEqual(e[0].shape, (1, 4, 5))
            self.assertEqual(e[-1].shape, (1, 4, 5))
            self.assertEqual(e[0].maximum(), 4 * 5)
            self.assertEqual(e[-1].maximum(), 3 * 4 * 5)

        i = 0
        e = d.flip(i)
        self.assertEqual(e._pmshape, (3,))
        self.assertEqual(e[0].shape, (1, 4, 5))
        self.assertEqual(e[-1].shape, (1, 4, 5))
        self.assertEqual(e[0].maximum(), 3 * 4 * 5)
        self.assertEqual(e[-1].maximum(), 4 * 5)

    def test_Data_max(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            for pp in (False, True):
                with cf.chunksize(chunksize):
                    d = cf.Data([[4, 5, 6], [1, 2, 3]], "metre")
                    self.assertEqual(
                        d.maximum(_preserve_partitions=pp), cf.Data(6, "metre")
                    )
                    self.assertEqual(
                        d.maximum(_preserve_partitions=pp).datum(), 6
                    )
                    d[0, 2] = cf.masked
                    self.assertEqual(d.maximum(_preserve_partitions=pp), 5)
                    self.assertEqual(
                        d.maximum(_preserve_partitions=pp).datum(), 5
                    )
                    self.assertEqual(
                        d.maximum(_preserve_partitions=pp),
                        cf.Data(0.005, "km"),
                    )

    def test_Data_min(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            for pp in (False, True):
                with cf.chunksize(chunksize):
                    d = cf.Data([[4, 5, 6], [1, 2, 3]], "metre")
                    self.assertEqual(
                        d.minimum(_preserve_partitions=pp), cf.Data(1, "metre")
                    )
                    self.assertEqual(
                        d.minimum(_preserve_partitions=pp).datum(), 1
                    )
                    d[1, 0] = cf.masked
                    self.assertEqual(d.minimum(_preserve_partitions=pp), 2)
                    self.assertEqual(
                        d.minimum(_preserve_partitions=pp).datum(), 2
                    )
                    self.assertEqual(
                        d.minimum(_preserve_partitions=pp),
                        cf.Data(0.002, "km"),
                    )

    def test_Data_ndindex(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            cf.chunksize(chunksize)
            for d in (
                cf.Data(5, "metre"),
                cf.Data([4, 5, 6, 1, 2, 3], "metre"),
                cf.Data([[4, 5, 6], [1, 2, 3]], "metre"),
            ):
                for i, j in zip(d.ndindex(), numpy.ndindex(d.shape)):
                    self.assertEqual(i, j)
        # --- End: for

        cf.chunksize(self.original_chunksize)

    def test_Data_roll(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = numpy.arange(10 * 15 * 19).reshape(10, 1, 15, 19)

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(a.copy())

                _ = d._pmshape

                e = d.roll(0, 4)
                e.roll(2, 120, inplace=True)
                e.roll(3, -77, inplace=True)

                a = numpy.roll(a, 4, 0)
                a = numpy.roll(a, 120, 2)
                a = numpy.roll(a, -77, 3)

                self.assertEqual(e.shape, a.shape)
                self.assertTrue((a == e.array).all())

                f = e.roll(3, 77)
                f.roll(2, -120, inplace=True)
                f.roll(0, -4, inplace=True)

                self.assertEqual(f.shape, d.shape)
                self.assertTrue(f.equals(d, verbose=2))

    def test_Data_swapaxes(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = numpy.arange(10 * 15 * 19).reshape(10, 1, 15, 19)

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(a.copy())

                for i in range(-a.ndim, a.ndim):
                    for j in range(-a.ndim, a.ndim):
                        b = numpy.swapaxes(a.copy(), i, j)
                        e = d.swapaxes(i, j)
                        message = "cf.Data.swapaxes({}, {}) failed".format(
                            i, j
                        )
                        self.assertEqual(b.shape, e.shape, message)
                        self.assertTrue((b == e.array).all(), message)

    def test_Data_transpose(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = numpy.arange(10 * 15 * 19).reshape(10, 1, 15, 19)

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(a.copy())

                for indices in (range(a.ndim), range(-a.ndim, 0)):
                    for axes in itertools.permutations(indices):
                        a = numpy.transpose(a, axes)
                        d.transpose(axes, inplace=True)
                        message = (
                            "cf.Data.transpose({}) failed: "
                            "d.shape={}, a.shape={}".format(
                                axes, d.shape, a.shape
                            )
                        )
                        self.assertEqual(d.shape, a.shape, message)
                        self.assertTrue((d.array == a).all(), message)

    def test_Data_unique(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data([[4, 2, 1], [1, 2, 3]], "metre")
                self.assertTrue(
                    (d.unique() == cf.Data([1, 2, 3, 4], "metre")).all()
                )
                d[1, -1] = cf.masked
                self.assertTrue(
                    (d.unique() == cf.Data([1, 2, 4], "metre")).all()
                )

    def test_Data_varray(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Scalar array
        d = cf.Data(9, "km")
        d.hardmask = False
        a = d.varray
        self.assertEqual(a.shape, ())
        self.assertEqual(a, numpy.array(9))
        d[...] = cf.masked
        a = d.varray
        self.assertEqual(a.shape, ())
        self.assertIs(a[()], numpy.ma.masked)
        a[()] = 18
        self.assertEqual(a, numpy.array(18))

        b = numpy.arange(10 * 15 * 19).reshape(10, 1, 15, 19)
        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(b, "km")
                e = d.copy()
                v = e.varray
                v[0, 0, 0, 0] = -999
                v = e.varray
                self.assertEqual(v[0, 0, 0, 0], -999)
                self.assertEqual(v.shape, b.shape)
                self.assertFalse((v == b).all())
                v[0, 0, 0, 0] = 0
                self.assertTrue((v == b).all())

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

        cf.chunksize(self.original_chunksize)

    def test_Data_BINARY_AND_UNARY_OPERATORS(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            cf.chunksize(chunksize)
            array = numpy.arange(3 * 4 * 5).reshape(3, 4, 5) + 1

            arrays = (
                numpy.arange(3 * 4 * 5).reshape(3, 4, 5) + 1.0,
                numpy.arange(3 * 4 * 5).reshape(3, 4, 5) + 1,
            )

            for a0 in arrays:
                for a1 in arrays[::-1]:
                    d = cf.Data(
                        a0[(slice(None, None, -1),) * a0.ndim], "metre"
                    )
                    d.flip(inplace=True)
                    x = cf.Data(a1, "metre")

                    message = "Failed in {!r}+{!r}".format(d, x)
                    self.assertTrue(
                        (d + x).equals(cf.Data(a0 + a1, "m"), verbose=1),
                        message,
                    )
                    message = "Failed in {!r}*{!r}".format(d, x)
                    self.assertTrue(
                        (d * x).equals(cf.Data(a0 * a1, "m2"), verbose=1),
                        message,
                    )
                    message = "Failed in {!r}/{!r}".format(d, x)
                    self.assertTrue(
                        (d / x).equals(cf.Data(a0 / a1, "1"), verbose=1),
                        message,
                    )
                    message = "Failed in {!r}-{!r}".format(d, x)
                    self.assertTrue(
                        (d - x).equals(cf.Data(a0 - a1, "m"), verbose=1),
                        message,
                    )
                    message = "Failed in {!r}//{!r}".format(d, x)
                    self.assertTrue(
                        (d // x).equals(cf.Data(a0 // a1, "1"), verbose=1),
                        message,
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
                for x in (
                    2,
                    2.0,
                ):
                    message = "Failed in {!r}+{}".format(d, x)
                    self.assertTrue(
                        (d + x).equals(cf.Data(a0 + x, "m"), verbose=1),
                        message,
                    )
                    message = "Failed in {!r}*{}".format(d, x)
                    self.assertTrue(
                        (d * x).equals(cf.Data(a0 * x, "m"), verbose=1),
                        message,
                    )
                    message = "Failed in {!r}/{}".format(d, x)
                    self.assertTrue(
                        (d / x).equals(cf.Data(a0 / x, "m"), verbose=1),
                        message,
                    )
                    message = "Failed in {!r}-{}".format(d, x)
                    self.assertTrue(
                        (d - x).equals(cf.Data(a0 - x, "m"), verbose=1),
                        message,
                    )
                    message = "Failed in {!r}//{}".format(d, x)
                    self.assertTrue(
                        (d // x).equals(cf.Data(a0 // x, "m"), verbose=1),
                        message,
                    )
                    message = "Failed in {!r}**{}".format(d, x)
                    self.assertTrue(
                        (d ** x).equals(cf.Data(a0 ** x, "m2"), verbose=1),
                        message,
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
                        (x + d).equals(cf.Data(x + a0, "m"), verbose=1),
                        message,
                    )
                    message = "Failed in {}*{!r}".format(x, d)
                    self.assertTrue(
                        (x * d).equals(cf.Data(x * a0, "m"), verbose=1),
                        message,
                    )
                    message = "Failed in {}/{!r}".format(x, d)
                    self.assertTrue(
                        (x / d).equals(cf.Data(x / a0, "m-1"), verbose=1),
                        message,
                    )
                    message = "Failed in {}-{!r}".format(x, d)
                    self.assertTrue(
                        (x - d).equals(cf.Data(x - a0, "m"), verbose=1),
                        message,
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
                        message = "Failed in {!r}.__itruediv__({})".format(
                            d, x
                        )
                        self.assertTrue(
                            e.equals(cf.Data(a, "m"), verbose=1), message
                        )
                # --- End: for

                for x in (cf.Data(2, "metre"), cf.Data(2.0, "metre")):
                    self.assertTrue(
                        (d + x).equals(cf.Data(a0 + x.datum(), "m"), verbose=1)
                    )
                    self.assertTrue(
                        (d * x).equals(
                            cf.Data(a0 * x.datum(), "m2"), verbose=1
                        )
                    )
                    self.assertTrue(
                        (d / x).equals(cf.Data(a0 / x.datum(), "1"), verbose=1)
                    )
                    self.assertTrue(
                        (d - x).equals(cf.Data(a0 - x.datum(), "m"), verbose=1)
                    )
                    self.assertTrue(
                        (d // x).equals(
                            cf.Data(a0 // x.datum(), "1"), verbose=1
                        )
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
        # --- End: for

        cf.chunksize(self.original_chunksize)

    def test_Data_BROADCASTING(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        A = [
            numpy.array(3),
            numpy.array([3]),
            numpy.array([3]).reshape(1, 1),
            numpy.array([3]).reshape(1, 1, 1),
            numpy.arange(5).reshape(5, 1),
            numpy.arange(5).reshape(1, 5),
            numpy.arange(5).reshape(1, 5, 1),
            numpy.arange(5).reshape(5, 1, 1),
            numpy.arange(5).reshape(1, 1, 5),
            numpy.arange(25).reshape(1, 5, 5),
            numpy.arange(25).reshape(5, 1, 5),
            numpy.arange(25).reshape(5, 5, 1),
            numpy.arange(125).reshape(5, 5, 5),
        ]

        for chunksize in self.chunk_sizes:
            cf.chunksize(chunksize)
            for a in A:
                for b in A:
                    d = cf.Data(a)
                    e = cf.Data(b)
                    ab = a * b
                    de = d * e
                    self.assertEqual(de.shape, ab.shape)
                    self.assertTrue((de.array == ab).all())
        # --- End: for

        cf.chunksize(self.original_chunksize)

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

        self.assertEqual(len(cf.Data([1, 2, 3])), 3)
        self.assertEqual(len(cf.Data([[1, 2, 3]])), 1)
        self.assertEqual(len(cf.Data([[1, 2, 3], [4, 5, 6]])), 2)

        with self.assertRaises(Exception):
            _ = len(cf.Data(1))

    def test_Data__float__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for x in (-1.9, -1.5, -1.4, -1, 0, 1, 1.0, 1.4, 1.9):
            self.assertEqual(float(cf.Data(x)), float(x))
            self.assertEqual(float(cf.Data(x)), float(x))

        with self.assertRaises(Exception):
            _ = float(cf.Data([1, 2]))

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

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):

                d = cf.Data(numpy.arange(1200).reshape(40, 5, 6))

                self.assertEqual(d.argmax(), 1199)
                self.assertEqual(d.argmax(unravel=True), (39, 4, 5))

                e = d.argmax(axis=1)
                self.assertEqual(e.shape, (40, 6))
                self.assertTrue(
                    e.equals(
                        cf.Data.full(shape=(40, 6), fill_value=4, dtype=int)
                    )
                )

    def test_Data__collapse_SHAPE(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        a = numpy.arange(-100, 200.0, dtype=float).reshape(3, 4, 5, 5)
        _ = numpy.ones(a.shape, dtype=float)

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

    def test_Data_max_min_sum_sum_of_squares(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            for pp in (True, False):
                cf.chunksize(chunksize)

                # unweighted, unmasked
                d = cf.Data(self.a)
                for np, h in zip(
                    (numpy.sum, numpy.amin, numpy.amax, numpy.sum),
                    ("sum", "min", "max", "sum_of_squares"),
                ):
                    for axes in self.axes_combinations:
                        b = reshape_array(self.a, axes)
                        if h == "sum_of_squares":
                            b = b ** 2

                        b = np(b, axis=-1)
                        e = getattr(d, h)(
                            axes=axes, squeeze=True, _preserve_partitions=pp
                        )
                        self.assertTrue(
                            e.allclose(b, rtol=1e-05, atol=1e-08),
                            "{}, axis={}, unweighted, unmasked "
                            "\ne={}, \nb={}".format(h, axes, e.array, b),
                        )
                # --- End: for

                # unweighted, masked
                d = cf.Data(self.ma)
                for np, h in zip(
                    (numpy.ma.sum, numpy.ma.amin, numpy.ma.amax, numpy.ma.sum),
                    ("sum", "min", "max", "sum_of_squares"),
                ):
                    for axes in self.axes_combinations:
                        b = reshape_array(self.ma, axes)
                        if h == "sum_of_squares":
                            b = b ** 2

                        b = np(b, axis=-1)
                        b = numpy.ma.asanyarray(b)
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
        # --- End: for

        cf.chunksize(self.original_chunksize)

    def test_Data_median(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                for pp in (True, False):
                    # unweighted, unmasked
                    d = cf.Data(self.a)
                    for axes in self.axes_combinations:
                        b = reshape_array(self.a, axes)
                        b = numpy.median(b, axis=-1)

                        e = d.median(
                            axes=axes, squeeze=True, _preserve_partitions=pp
                        )
                        self.assertTrue(
                            e.allclose(b, rtol=1e-05, atol=1e-08),
                            "median, axis={}, unweighted, unmasked "
                            "\ne={}, \nb={}".format(axes, e.array, b),
                        )

                    # unweighted, masked
                    d = cf.Data(self.ma)
                    for axes in self.axes_combinations:
                        b = reshape_array(self.ma, axes)
                        b = numpy.ma.filled(b, numpy.nan)
                        with numpy.testing.suppress_warnings() as sup:
                            sup.filter(
                                RuntimeWarning,
                                message=".*All-NaN slice encountered",
                            )
                            b = numpy.nanpercentile(b, 50, axis=-1)

                        b = numpy.ma.masked_where(
                            numpy.isnan(b), b, copy=False
                        )
                        b = numpy.ma.asanyarray(b)

                        e = d.median(
                            axes=axes, squeeze=True, _preserve_partitions=pp
                        )

                        self.assertTrue(
                            (e.mask.array == b.mask).all(),
                            "median, axis={}, \ne.mask={}, "
                            "\nb.mask={}".format(axes, e.mask.array, b.mask),
                        )

                        self.assertTrue(
                            e.allclose(b, rtol=1e-05, atol=1e-08),
                            "median, axis={}, unweighted, masked "
                            "\ne={}, \nb={}".format(axes, e.array, b),
                        )

    def test_Data_percentile(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.Data(self.a)

                # Percentiles taken across *all axes*
                ranks = [[30, 60, 90], [20], 80]  # include valid singular form

                for rank in ranks:
                    # Note: in cf the default is squeeze=False, but
                    # numpy has an inverse parameter called keepdims
                    # which is by default False also, one must be set
                    # to the non-default for equivalents.  So first
                    # cases (n1, n1) are both squeezed, (n2, n2) are
                    # not:
                    a1 = numpy.percentile(d, rank)  # keepdims=False default
                    b1 = d.percentile(rank, squeeze=True)
                    self.assertTrue(b1.allclose(a1, rtol=1e-05, atol=1e-08))
                    a2 = numpy.percentile(d, rank, keepdims=True)
                    b2 = d.percentile(rank)  # squeeze=False default
                    self.assertTrue(b2.shape, a2.shape)
                    self.assertTrue(b2.allclose(a2, rtol=1e-05, atol=1e-08))

        # TODO: add loop to check get same shape and close enough data
        # for every possible axes combo (as with test_Data_median above).

    def test_Data_mean_of_upper_decile(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                for pp in (True, False):
                    # unweighted, unmasked
                    d = cf.Data(self.a)
                    for axes in self.axes_combinations:
                        b = reshape_array(self.a, axes)
                        p = numpy.percentile(b, 90, axis=-1, keepdims=True)
                        b = numpy.ma.where(b < p, numpy.ma.masked, b)
                        b = numpy.average(b, axis=-1)

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
                        b = numpy.ma.filled(b, numpy.nan)
                        with numpy.testing.suppress_warnings() as sup:
                            sup.filter(
                                RuntimeWarning,
                                message=".*All-NaN slice encountered",
                            )
                            p = numpy.nanpercentile(
                                b, 90, axis=-1, keepdims=True
                            )

                        b = numpy.ma.masked_where(
                            numpy.isnan(b), b, copy=False
                        )

                        p = numpy.where(numpy.isnan(p), b.max() + 1, p)

                        with numpy.testing.suppress_warnings() as sup:
                            sup.filter(
                                RuntimeWarning,
                                message=".*invalid value encountered in less",
                            )
                            b = numpy.ma.where(b < p, numpy.ma.masked, b)

                        b = numpy.ma.average(b, axis=-1)
                        b = numpy.ma.asanyarray(b)

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

    def test_Data_range_mid_range(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                for pp in (True, False):
                    # unweighted, unmasked
                    d = cf.Data(self.a)
                    for h in ("range", "mid_range"):
                        for axes in self.axes_combinations:
                            b = reshape_array(self.a, axes)
                            mn = numpy.amin(b, axis=-1)
                            mx = numpy.amax(b, axis=-1)
                            if h == "range":
                                b = mx - mn
                            elif h == "mid_range":
                                b = (mx + mn) * 0.5

                            e = getattr(d, h)(
                                axes=axes,
                                squeeze=True,
                                _preserve_partitions=pp,
                            )
                            self.assertTrue(
                                e.allclose(b, rtol=1e-05, atol=1e-08),
                                "{}, axis={}, unweighted, unmasked "
                                "\ne={}, \nb={}".format(h, axes, e.array, b),
                            )
                    # --- End: for

                    # unweighted, masked
                    d = cf.Data(self.ma)
                    for h in ("range", "mid_range"):
                        for axes in self.axes_combinations:
                            b = reshape_array(self.ma, axes)
                            mn = numpy.amin(b, axis=-1)
                            mx = numpy.amax(b, axis=-1)
                            if h == "range":
                                b = mx - mn
                            elif h == "mid_range":
                                b = (mx + mn) * 0.5

                            b = numpy.ma.asanyarray(b)

                            e = getattr(d, h)(
                                axes=axes,
                                squeeze=True,
                                _preserve_partitions=pp,
                            )

                            self.assertTrue(
                                (e.mask.array == b.mask).all(),
                                "{}, axis={}, \ne.mask={}, "
                                "\nb.mask={}".format(
                                    h, axes, e.mask.array, b.mask
                                ),
                            )

                            self.assertTrue(
                                e.allclose(b, rtol=1e-05, atol=1e-08),
                                "{}, axis={}, unweighted, masked "
                                "\ne={}, \nb={}".format(h, axes, e.array, b),
                            )

    def test_Data_integral(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                for pp in (True, False):
                    # unmasked
                    d = cf.Data(self.a)
                    x = cf.Data(self.w)
                    for axes in self.axes_combinations:
                        b = reshape_array(self.a, axes)
                        v = reshape_array(self.w, axes)
                        b = numpy.sum(b * v, axis=-1)

                        e = d.integral(
                            axes=axes,
                            squeeze=True,
                            weights=x,
                            _preserve_partitions=pp,
                        )

                        self.assertTrue(
                            e.allclose(b, rtol=1e-05, atol=1e-08),
                            "axis={}, unmasked \ne={}, \nb={}".format(
                                axes, e.array, b
                            ),
                        )
                    # --- End: for

                    # masked
                    d = cf.Data(self.ma)
                    for axes in self.axes_combinations:
                        b = reshape_array(self.ma, axes)
                        v = reshape_array(self.w, axes)
                        b = numpy.sum(b * v, axis=-1)
                        b = numpy.ma.asanyarray(b)

                        e = d.integral(
                            axes=axes,
                            squeeze=True,
                            weights=x,
                            _preserve_partitions=pp,
                        )

                        self.assertTrue(
                            (e.mask.array == b.mask).all(),
                            "axis={} masked, \ne.mask={}, "
                            "\nb.mask={}".format(axes, e.mask.array, b.mask),
                        )

                        self.assertTrue(
                            e.allclose(b, rtol=1e-05, atol=1e-08),
                            "axis={}, masked \ne={}, \nb={}".format(
                                axes, e.array, b
                            ),
                        )

    def test_Data_sum_of_weights_sum_of_weights2(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                for pp in (True, False):
                    # unweighted, unmasked
                    d = cf.Data(self.a)
                    for h in ("sum_of_weights", "sum_of_weights2"):
                        for axes in self.axes_combinations:
                            b = reshape_array(self.ones, axes)
                            b = b.sum(axis=-1)
                            e = getattr(d, h)(
                                axes=axes,
                                squeeze=True,
                                _preserve_partitions=pp,
                            )

                            self.assertTrue(
                                e.allclose(b, rtol=1e-05, atol=1e-08),
                                "{}, axis={}, unweighted, unmasked, pp={}, "
                                "\ne={}, \nb={}".format(
                                    h, axes, pp, e.array, b
                                ),
                            )
                    # --- End: for

                    # unweighted, masked
                    d = cf.Data(self.ma)
                    for a, h in zip(
                        (self.mones, self.mones),
                        ("sum_of_weights", "sum_of_weights2"),
                    ):
                        for axes in self.axes_combinations:
                            b = reshape_array(a, axes)
                            b = numpy.ma.asanyarray(b.sum(axis=-1))
                            e = getattr(d, h)(
                                axes=axes,
                                squeeze=True,
                                _preserve_partitions=pp,
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
                                "\ne={}, \nb={}".format(
                                    h, axes, pp, e.array, b
                                ),
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
                            b = numpy.ma.asanyarray(b.sum(axis=-1))
                            e = getattr(d, h)(
                                axes=axes,
                                weights=x,
                                squeeze=True,
                                _preserve_partitions=pp,
                            )
                            self.assertTrue(
                                (e.mask.array == b.mask).all(),
                                "{}, axis={}, \ne.mask={}, "
                                "\nb.mask={}".format(
                                    h,
                                    axes,
                                    e.mask.array,
                                    b.mask,
                                ),
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

    def test_Data_mean_mean_absolute_value(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for absolute in (False, True):
            a = self.a
            ma = self.ma
            method = "mean"
            if absolute:
                a = numpy.absolute(a)
                ma = numpy.absolute(ma)
                method = "mean_absolute_value"

            for chunksize in self.chunk_sizes:
                cf.chunksize(chunksize)

                # unweighted, unmasked
                d = cf.Data(self.a)
                for axes in self.axes_combinations:
                    b = reshape_array(a, axes)
                    b = numpy.mean(b, axis=-1)
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
                    b = numpy.average(b, axis=-1, weights=v)

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
                    b = numpy.ma.average(b, axis=-1)
                    b = numpy.ma.asanyarray(b)

                    e = getattr(d, method)(axes=axes, squeeze=True)

                    self.assertTrue(
                        (e.mask.array == b.mask).all(),
                        "{} unweighted, masked axis={}, \ne.mask={}, "
                        "\nb.mask={}".format(
                            method, axes, e.mask.array, b.mask
                        ),
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
                    b = numpy.ma.average(b, axis=-1, weights=v)
                    b = numpy.ma.asanyarray(b)

                    e = getattr(d, method)(axes=axes, weights=x, squeeze=True)

                    self.assertTrue(
                        (e.mask.array == b.mask).all(),
                        "{} weighted, masked axis={}, \ne.mask={}, "
                        "\nb.mask={}".format(
                            method, axes, e.mask.array, b.mask
                        ),
                    )

                    self.assertTrue(
                        e.allclose(b, rtol=1e-05, atol=1e-08),
                        "{} weighted, masked axis={}, \ne={}, "
                        "\nb={}, ".format(method, axes, e.array, b),
                    )
        # --- End: for

        cf.chunksize(self.original_chunksize)

    def test_Data_root_mean_square(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                # unweighted, unmasked
                d = cf.Data(self.a)
                for axes in self.axes_combinations:
                    b = reshape_array(self.a, axes) ** 2
                    b = numpy.mean(b, axis=-1) ** 0.5
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
                    b = numpy.average(b, axis=-1, weights=v) ** 0.5

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
                    b = numpy.ma.average(b, axis=-1)
                    b = numpy.ma.asanyarray(b) ** 0.5

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
                    b = numpy.ma.average(b, axis=-1, weights=v)
                    b = numpy.ma.asanyarray(b) ** 0.5

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

    def test_Data_sample_size(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
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

    def test_Data_sd_var(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        ddofs = (0, 1)

        for chunksize in self.chunk_sizes:
            cf.chunksize(chunksize)
            for pp in (False, True):
                # unweighted, unmasked
                d = cf.Data(self.a, units="K")
                for np, h in zip((numpy.var, numpy.std), ("var", "sd")):
                    for ddof in ddofs:
                        for axes in self.axes_combinations:
                            b = reshape_array(self.a, axes)
                            b = np(b, axis=-1, ddof=ddof)
                            e = getattr(d, h)(
                                axes=axes,
                                squeeze=True,
                                ddof=ddof,
                                _preserve_partitions=pp,
                            )
                            self.assertTrue(
                                e.allclose(b, rtol=1e-05, atol=1e-08),
                                "{}, axis={}, unweighted, unmasked pp={}, "
                                "\ne={}, \nb={}".format(
                                    h, axes, pp, e.array, b
                                ),
                            )
                # --- End: for

                # unweighted, masked
                d = cf.Data(self.ma, units="K")
                for np, h in zip((numpy.ma.var, numpy.ma.std), ("var", "sd")):
                    for ddof in ddofs:
                        for axes in self.axes_combinations:
                            b = reshape_array(self.ma, axes)
                            b = np(b, axis=-1, ddof=ddof)
                            e = getattr(d, h)(
                                axes=axes,
                                squeeze=True,
                                ddof=ddof,
                                _preserve_partitions=pp,
                            )
                            self.assertTrue(
                                e.allclose(b, rtol=1e-05, atol=1e-08),
                                "{}, axis={}, unweighted, masked, pp={}, "
                                "\ne={}, \nb={}".format(
                                    h, axes, pp, e.array, b
                                ),
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

                            avg = numpy.average(b, axis=-1, weights=v)
                            if numpy.ndim(avg) < b.ndim:
                                avg = numpy.expand_dims(avg, -1)

                            b, V1 = numpy.average(
                                (b - avg) ** 2,
                                axis=-1,
                                weights=v,
                                returned=True,
                            )

                            if ddof == 1:
                                # Calculate the weighted unbiased
                                # variance. The unbiased variance
                                # weighted with _reliability_ weights
                                # is [V1**2/(V1**2-V2)]*var.
                                V2 = numpy.asanyarray((v * v).sum(axis=-1))
                                b *= V1 * V1 / (V1 * V1 - V2)
                            elif ddof == 0:
                                pass

                            if h == "sd":
                                b **= 0.5

                            b = numpy.ma.asanyarray(b)

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

                            not_enough_data = (
                                numpy.ma.count(b, axis=-1) <= ddof
                            )

                            avg = numpy.ma.average(b, axis=-1, weights=v)
                            if numpy.ndim(avg) < b.ndim:
                                avg = numpy.expand_dims(avg, -1)

                            b, V1 = numpy.ma.average(
                                (b - avg) ** 2,
                                axis=-1,
                                weights=v,
                                returned=True,
                            )

                            b = numpy.ma.where(
                                not_enough_data, numpy.ma.masked, b
                            )

                            if ddof == 1:
                                # Calculate the weighted unbiased
                                # variance. The unbiased variance
                                # weighted with _reliability_ weights
                                # is [V1**2/(V1**2-V2)]*var.
                                V2 = numpy.asanyarray((v * v).sum(axis=-1))
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

        cf.chunksize(self.original_chunksize)

    def test_Data_dumpd_loadd_dumps(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                d = cf.read(self.filename)[0].data

                dumpd = d.dumpd()
                self.assertTrue(d.equals(cf.Data(loadd=dumpd), verbose=2))
                self.assertTrue(d.equals(cf.Data(loadd=dumpd), verbose=2))

                d.to_disk()
                self.assertTrue(d.equals(cf.Data(loadd=dumpd), verbose=2))

    def test_Data_section(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in (300, 10000, 100000)[::-1]:
            with cf.chunksize(chunksize):
                f = cf.read(self.filename6)[0]
                self.assertEqual(
                    list(sorted(f.data.section((1, 2)).keys())),
                    [(x, None, None) for x in range(1800)],
                )
                d = cf.Data(numpy.arange(120).reshape(2, 3, 4, 5))
                x = d.section([1, 3])
                self.assertEqual(len(x), 8)
                e = cf.Data.reconstruct_sectioned_data(x)
                self.assertTrue(e.equals(d))

    def test_Data_count(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in (100000, 10000, 300):
            with cf.chunksize(chunksize):
                d = cf.Data(ma)
                self.assertEqual(d.count(), 284, d.count())
                self.assertEqual(
                    d.count_masked(), d.size - 284, d.count_masked()
                )

                d = cf.Data(a)
                self.assertEqual(d.count(), d.size)
                self.assertEqual(d.count_masked(), 0)

    def test_Data_exp(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for x in (1, -1):
            a = 0.9 * x * self.ma
            c = numpy.ma.exp(a)

            for chunksize in self.chunk_sizes:
                with cf.chunksize(chunksize):
                    d = cf.Data(a)
                    e = d.exp()
                    self.assertIsNone(d.exp(inplace=True))
                    self.assertTrue(d.equals(e, verbose=2))
                    self.assertEqual(d.shape, c.shape)
                    # The CI at one point gave a failure due to
                    # precision with:
                    # self.assertTrue((d.array==c).all()) so need a
                    # check which accounts for floating point calcs:
                    numpy.testing.assert_allclose(d.array, c)
        # --- End: for

        d = cf.Data(a, "m")
        with self.assertRaises(Exception):
            _ = d.exp()

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
                        a = numpy.cosh(a.data)  # convert non-masked x to >= 1
                    else:  # convert non-masked values x to range |x| < 1
                        a = numpy.sin(a.data)

                c = getattr(numpy.ma, method)(a)
                for chunksize in self.chunk_sizes:
                    with cf.chunksize(chunksize):
                        for units in (None, "", "1", "radians", "K"):
                            d = cf.Data(a, units=units)
                            # Suppress warnings that some values are
                            # invalid (NaN, +/- inf) or there is
                            # attempted division by zero, as this is
                            # expected with inverse trig:
                            with numpy.errstate(
                                invalid="ignore", divide="ignore"
                            ):
                                e = getattr(d, method)()
                                self.assertIsNone(
                                    getattr(d, method)(inplace=True)
                                )

                            self.assertTrue(
                                d.equals(e, verbose=2), "{}".format(method)
                            )
                            self.assertEqual(d.shape, c.shape)
                            self.assertTrue(
                                (d.array == c).all(),
                                "{}, {}, {}, {}".format(
                                    method, units, d.array, c
                                ),
                            )
                            self.assertTrue(
                                (d.mask.array == c.mask).all(),
                                "{}, {}, {}, {}".format(
                                    method, units, d.array, c
                                ),
                            )
        # --- End: for

        # Also test masking behaviour: masking of invalid data occurs for
        # numpy.ma module by default but we don't want that so there is logic
        # to workaround it. So check that invalid values do emerge.
        inverse_methods = [
            method
            for method in trig_and_hyperbolic_methods
            if method.startswith("arc")
        ]

        d = cf.Data([2, 1.5, 1, 0.5, 0], mask=[1, 0, 0, 0, 1])
        for method in inverse_methods:
            with numpy.errstate(invalid="ignore", divide="ignore"):
                e = getattr(d, method)()
            self.assertTrue(
                (e.mask.array == d.mask.array).all(),
                "{}, {}, {}".format(method, e.array, d),
            )

        # In addition, test that 'nan', inf' and '-inf' emerge distinctly
        f = cf.Data([-2, -1, 1, 2], mask=[0, 0, 0, 1])
        with numpy.errstate(invalid="ignore", divide="ignore"):
            g = f.arctanh().array  # expect [ nan, -inf,  inf,  --]

        self.assertTrue(numpy.isnan(g[0]))
        self.assertTrue(numpy.isneginf(g[1]))
        self.assertTrue(numpy.isposinf(g[2]))
        self.assertIs(g[3], cf.masked)

        # AT2
        #
        # # Treat arctan2 separately (as is a class method & takes two inputs)
        # for x in (1, -1):
        #     a1 = 0.9 * x * self.ma
        #     a2 = 0.5 * x * self.a
        #     # Transform data for 'a' into range more appropriate for inverse:
        #     a1 = numpy.sin(a1.data)
        #     a2 = numpy.cos(a2.data)

        #     c = numpy.ma.arctan2(a1, a2)
        #     for chunksize in self.chunk_sizes:
        #         cf.chunksize(chunksize)
        #         for units in (None, '', '1', 'radians', 'K'):
        #             d1 = cf.Data(a1, units=units)
        #             d2 = cf.Data(a2, units=units)
        #             e = cf.Data.arctan2(d1, d2)
        #             # Note: no inplace arg for arctan2 (operates on 2 arrays)
        #             self.assertEqual(d1.shape, c.shape)
        #             self.assertTrue((e.array == c).all())
        #             self.assertTrue((d1.mask.array == c.mask).all())

    def test_Data_filled(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = cf.Data([[1, 2, 3]])
        self.assertTrue((d.filled().array == [[1, 2, 3]]).all())

        d[0, 0] = cf.masked
        self.assertTrue(
            (
                d.filled().array
                == [
                    [
                        -9223372036854775806,
                        2,
                        3,
                    ]
                ]
            ).all()
        )

        d.set_fill_value(-99)
        self.assertTrue(
            (
                d.filled().array
                == [
                    [
                        -99,
                        2,
                        3,
                    ]
                ]
            ).all()
        )

        self.assertTrue(
            (
                d.filled(1e10).array
                == [
                    [
                        1e10,
                        2,
                        3,
                    ]
                ]
            ).all()
        )

        d = cf.Data(["a", "b", "c"], mask=[1, 0, 0])
        self.assertTrue((d.filled().array == ["", "b", "c"]).all())

    def test_Data_del_units(self):
        d = cf.Data(1)
        with self.assertRaises(ValueError):
            d.del_units()

        d = cf.Data(1, "")
        self.assertEqual(d.del_units(), "")
        d = cf.Data(1, "m")
        self.assertEqual(d.del_units(), "m")

        d = cf.Data(1, "days since 2000-1-1")
        self.assertTrue(d.del_units(), "days since 2000-1-1")

        d = cf.Data(1, "days since 2000-1-1", calendar="noleap")
        with self.assertRaises(ValueError):
            d.del_units()

    def test_Data_del_calendar(self):
        d = cf.Data(1)
        with self.assertRaises(ValueError):
            d.del_calendar()

        d = cf.Data(1, "")
        with self.assertRaises(ValueError):
            d.del_calendar()

        d = cf.Data(1, "m")
        with self.assertRaises(ValueError):
            d.del_calendar()

        d = cf.Data(1, "days since 2000-1-1")
        with self.assertRaises(ValueError):
            d.del_calendar()

        d = cf.Data(1, "days since 2000-1-1", calendar="noleap")
        self.assertTrue(d.del_calendar(), "noleap")

    def test_Data_has_units(self):
        d = cf.Data(1)
        self.assertFalse(d.has_units())
        d = cf.Data(1, "")
        self.assertTrue(d.has_units())
        d = cf.Data(1, "m")
        self.assertTrue(d.has_units())

    def test_Data_has_calendar(self):
        d = cf.Data(1)
        self.assertFalse(d.has_calendar())
        d = cf.Data(1, "")
        self.assertFalse(d.has_calendar())
        d = cf.Data(1, "m")
        self.assertFalse(d.has_calendar())

        d = cf.Data(1, "days since 2000-1-1")
        self.assertFalse(d.has_calendar())
        d = cf.Data(1, "days since 2000-1-1", calendar="noleap")
        self.assertTrue(d.has_calendar())


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
