import atexit
import datetime
import faulthandler
import itertools
import os
import re
import tempfile
import unittest

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

n_tmpfiles = 1
tmpfiles = [
    tempfile.mkstemp("_test_Field.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
[tmpfile] = tmpfiles


def _remove_tmpfiles():
    """TODO."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


def axes_combinations(f):
    return [
        axes
        for n in range(1, f.ndim + 1)
        for axes in itertools.permutations(range(f.ndim), n)
    ]


class FieldTest(unittest.TestCase):
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file.nc"
    )
    filename1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_file1.nc"
    )
    filename2 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file2.nc"
    )
    contiguous = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "DSG_timeSeries_contiguous.nc",
    )
    indexed = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "DSG_timeSeries_indexed.nc"
    )
    indexed_contiguous = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "DSG_timeSeriesProfile_indexed_contiguous.nc",
    )

    chunk_sizes = (100000, 300, 34, 17)
    original_chunksize = cf.chunksize()
    atol = cf.atol()
    rtol = cf.rtol()

    f = cf.read(filename)[0]

    f0 = cf.example_field(0)
    f1 = cf.example_field(1)

    def test_Field_creation_commands(self):
        for i in range(7):
            f = cf.example_field(i)
            f.creation_commands()

        f = self.f1

        for rd in (False, True):
            f.creation_commands(representative_data=rd)

        for indent in (0, 4):
            f.creation_commands(indent=indent)

        for s in (False, True):
            f.creation_commands(string=s)

        for ns in ("cf", ""):
            f.creation_commands(namespace=ns)

    def test_Field_get_filenames(self):
        f = self.f0

        cf.write(f, tmpfile)
        g = cf.read(tmpfile)[0]

        abspath_tmpfile = os.path.abspath(tmpfile)
        self.assertEqual(
            g.get_filenames(), set([abspath_tmpfile]), g.get_filenames()
        )

        g.data[...] = -99
        self.assertEqual(
            g.get_filenames(), set([abspath_tmpfile]), g.get_filenames()
        )

        for c in g.constructs.filter_by_data().values():
            c.data[...] = -99

        self.assertEqual(
            g.get_filenames(), set([abspath_tmpfile]), g.get_filenames()
        )

        for c in g.constructs.filter_by_data().values():
            if c.has_bounds():
                c.bounds.data[...] = -99

        self.assertEqual(g.get_filenames(), set(), g.get_filenames())

    def test_Field_halo(self):
        f = cf.example_field(7)

        g = f.copy()
        self.assertIsNone(g.halo(1, inplace=True))

        i = 1
        g = f.halo(i)
        self.assertTrue(
            (numpy.array(g.shape) == numpy.array(f.shape) + i * 2).all()
        )

        for key, c in g.constructs.filter_by_data().items():
            d = f.construct(key)
            self.assertTrue(
                (numpy.array(c.shape) == numpy.array(d.shape) + i * 2).all()
            )

    def test_Field_has_construct(self):
        f = self.f1

        self.assertTrue(f.has_construct("T"))
        self.assertTrue(f.has_construct("long_name=Grid latitude name"))
        self.assertTrue(f.has_construct("ncvar%a"))
        self.assertTrue(f.has_construct("measure:area"))
        self.assertTrue(f.has_construct("domainaxis0"))

        self.assertFalse(f.has_construct("height"))

    def test_Field_compress_uncompress(self):
        methods = ("contiguous", "indexed", "indexed_contiguous")

        for method in methods:
            message = "method=" + method
            for f in cf.read(getattr(self, method)):

                self.assertTrue(bool(f.data.get_compression_type()), message)

                u = f.uncompress()
                self.assertFalse(bool(u.data.get_compression_type()), message)
                self.assertTrue(f.equals(u, verbose=2), message)

                for method1 in methods:
                    message += ", method1=" + method1
                    if method1 == "indexed_contiguous":
                        if f.ndim != 3:
                            continue
                    elif f.ndim != 2:
                        continue

                    c = u.compress(method1)
                    self.assertTrue(
                        bool(c.data.get_compression_type()), message
                    )

                    self.assertTrue(u.equals(c, verbose=2), message)
                    self.assertTrue(f.equals(c, verbose=2), message)

                    c = f.compress(method1)
                    self.assertTrue(
                        bool(c.data.get_compression_type()), message
                    )

                    self.assertTrue(u.equals(c, verbose=2), message)
                    self.assertTrue(f.equals(c, verbose=2), message)

                    cf.write(c, tmpfile)
                    c = cf.read(tmpfile)[0]

                    self.assertTrue(
                        bool(c.data.get_compression_type()), message
                    )
                    self.assertTrue(f.equals(c, verbose=2), message)

    def test_Field_apply_masking(self):
        f = self.f0.copy()

        for prop in (
            "missing_value",
            "_FillValue",
            "valid_min",
            "valid_max",
            "valid_range",
        ):
            f.del_property(prop, None)

        d = f.data.copy()
        g = f.copy()
        self.assertIsNone(f.apply_masking(inplace=True))
        self.assertTrue(f.equals(g, verbose=1))

        x = 0.11
        y = 0.1
        z = 0.2

        f.set_property("_FillValue", x)
        d = f.data.copy()

        g = f.apply_masking()
        e = d.apply_masking(fill_values=[x])
        self.assertTrue(e.equals(g.data, verbose=1))
        self.assertEqual(g.data.array.count(), g.data.size - 1)

        f.set_property("valid_range", [y, z])
        d = f.data.copy()
        g = f.apply_masking()
        e = d.apply_masking(fill_values=[x], valid_range=[y, z])
        self.assertTrue(e.equals(g.data, verbose=1))

        f.del_property("valid_range")
        f.set_property("valid_min", y)
        g = f.apply_masking()
        e = d.apply_masking(fill_values=[x], valid_min=y)
        self.assertTrue(e.equals(g.data, verbose=1))

        f.del_property("valid_min")
        f.set_property("valid_max", z)
        g = f.apply_masking()
        e = d.apply_masking(fill_values=[x], valid_max=z)
        self.assertTrue(e.equals(g.data, verbose=1))

        f.set_property("valid_min", y)
        g = f.apply_masking()
        e = d.apply_masking(fill_values=[x], valid_min=y, valid_max=z)
        self.assertTrue(e.equals(g.data, verbose=1))

    def test_Field_flatten(self):
        f = self.f.copy()

        axis = f.set_construct(cf.DomainAxis(1))
        d = cf.DimensionCoordinate()
        d.standard_name = "time"
        d.set_data(cf.Data([123.0], "days since 2000-01-02"))
        f.set_construct(d, axes=axis)

        g = f.flatten()
        h = f.flatten(list(range(f.ndim)))
        self.assertTrue(h.equals(g, verbose=2))

        g = f.flatten("time")
        self.assertTrue(g.equals(f, verbose=2))

        for i in (0, 1, 2):
            g = f.flatten(i)
            self.assertTrue(g.equals(f, verbose=2))
            g = f.flatten([i, "time"])
            self.assertTrue(g.equals(f, verbose=2))

        for axes in axes_combinations(f):
            g = f.flatten(axes)

            if len(axes) <= 1:
                shape = f.shape
            else:
                shape = [n for i, n in enumerate(f.shape) if i not in axes]
                shape.insert(
                    sorted(axes)[0],
                    numpy.prod(
                        [n for i, n in enumerate(f.shape) if i in axes]
                    ),
                )

            self.assertEqual(g.shape, tuple(shape))
            self.assertEqual(g.ndim, f.ndim - len(axes) + 1)
            self.assertEqual(g.size, f.size)

        self.assertTrue(f.equals(f.flatten([]), verbose=2))
        self.assertIsNone(f.flatten(inplace=True))

    def test_Field_bin(self):
        f = self.f

        d = f.digitize(10)
        b = f.bin("sample_size", digitized=d)

        a = numpy.ma.masked_all((10,), dtype=int)
        a[...] = 9
        self.assertTrue((a == b.array).all())

        b = f.bin("sample_size", digitized=[d])

        a = numpy.ma.masked_all((10,), dtype=int)
        a[...] = 9
        self.assertTrue((a == b.array).all())

        b = f.bin("sample_size", digitized=[d, d])

        a = numpy.ma.masked_all((10, 10), dtype=int)
        for i in range(9):
            a[i, i] = 9

        self.assertTrue((a == b.array).all())

        b = f.bin("sample_size", digitized=[d, d, d])

        a = numpy.ma.masked_all((10, 10, 10), dtype=int)
        for i in range(9):
            a[i, i, i] = 9

        self.assertTrue((a == b.array).all())

    def test_Field_direction(self):
        f = self.f.copy()
        yaxis = f.domain_axis("Y", key=True)
        ydim = f.dimension_coordinate("Y", key=True)
        f.direction("X")
        f.del_construct(ydim)
        f.direction(yaxis)
        self.assertTrue(f.direction("qwerty"))

        f = self.f.copy()
        self.assertIsInstance(f.directions(), dict)
        f.directions()

    def test_Field_domain_axis_position(self):
        f = self.f

        for i in range(f.ndim):
            self.assertEqual(f.domain_axis_position(i), i)

        for i in range(1, f.ndim + 1):
            self.assertEqual(f.domain_axis_position(-i), -i + 3)

        data_axes = f.get_data_axes()
        for key in data_axes:
            self.assertEqual(f.domain_axis_position(key), data_axes.index(key))

        self.assertEqual(f.domain_axis_position("Z"), 0)
        self.assertEqual(f.domain_axis_position("grid_latitude"), 1)

    def test_Field_weights(self):
        f = self.f.copy()
        f += 1

        w = f.weights()
        self.assertIsInstance(w, cf.Field)

        w = f.weights(None)
        self.assertIsInstance(w, cf.Field)
        self.assertTrue(w.data.equals(cf.Data(1.0, "1"), verbose=2))

        w = f.weights(data=True)
        self.assertIsInstance(w, cf.Data)

        w = f.weights(None, data=True)
        self.assertIsInstance(w, cf.Data)
        self.assertTrue(w.equals(cf.Data(1.0, "1"), verbose=2))

        w = f.weights(components=True)
        self.assertIsInstance(w, dict)

        w = f.weights(None, components=True)
        self.assertIsInstance(w, dict)
        self.assertEqual(w, {})

        w = f.weights(methods=True)
        self.assertIsInstance(w, dict)

        w = f.weights(None, methods=True)
        self.assertIsInstance(w, dict)
        self.assertEqual(w, {})

        w = f.weights()
        x = f.weights(w)
        self.assertTrue(x.equals(w, verbose=2))

        for components in (False, True):
            for m in (False, True):
                for d in (False, True):
                    if components:
                        d = False

                    f.weights(w, components=components, measure=m, data=d)
                    f.weights(
                        w.transpose(), components=components, measure=m, data=d
                    )
                    f.weights(w.data, components=components, measure=m, data=d)
                    f.weights(
                        f.data.squeeze(),
                        components=components,
                        measure=m,
                        data=d,
                    )
                    f.weights(components=components, measure=m, data=d)
                    f.weights(
                        "grid_longitude",
                        components=components,
                        measure=m,
                        data=d,
                    )
                    f.weights(
                        ["grid_longitude"],
                        components=components,
                        measure=m,
                        data=d,
                    )

        with self.assertRaises(Exception):
            f.weights(components=True, data=True)

    def test_Field_replace_construct(self):
        f = self.f.copy()

        for x in (
            "grid_longitude",
            "latitude",
            "grid_mapping_name:rotated_latitude_longitude",
            "ncvar%a",
        ):
            for copy in (True, False):
                f.replace_construct(x, new=f.construct(x), copy=copy)

        with self.assertRaises(Exception):
            f.replace_construct("grid_longitude", new=f.construct("latitude"))

        with self.assertRaises(Exception):
            f.replace_construct(
                "grid_longitude", new=f.construct("grid_latitude")
            )

    def test_Field_allclose(self):
        f = self.f
        g = f.copy()

        self.assertTrue(f.allclose(f))
        self.assertTrue(f.allclose(g))
        self.assertTrue(f.allclose(g.data))
        self.assertTrue(f.allclose(g.array))

        g[-1, -1, -1] = 1
        self.assertFalse(f.allclose(g))
        self.assertFalse(f.allclose(g.data))
        self.assertFalse(f.allclose(g.array))

    def test_Field_collapse(self):
        f = self.f.copy()
        f[0, 3] *= -1
        f[0, 5, ::2] = cf.masked

        for axes in axes_combinations(f):
            for method in (
                "sum",
                "min",
                "max",
                "minimum_absolute_value",
                "maximum_absolute_value",
                "mid_range",
                "range",
                "sample_size",
                "sum_of_squares",
                "median",
                "sum_of_weights",
                "sum_of_weights2",
            ):
                for weights in (None, "area"):
                    a = f.collapse(method, axes=axes, weights=weights).data
                    b = getattr(f.data, method)(axes=axes)
                    self.assertTrue(
                        a.equals(b, rtol=1e-05, atol=1e-08, verbose=2),
                        "{} weights={}, axes={}, {!r}, {!r}".format(
                            method, weights, axes, a, b
                        ),
                    )

            for method in (
                "mean",
                "mean_absolute_value",
                # 'mean_of_upper_decile',
                "root_mean_square",
            ):
                for weights in (None, "area"):
                    if weights is not None:
                        d_weights = f.weights(weights, components=True)
                    else:
                        d_weights = weights

                    a = f.collapse(method, axes=axes, weights=weights).data
                    b = getattr(f.data, method)(axes=axes, weights=d_weights)
                    self.assertTrue(
                        a.equals(b, rtol=1e-05, atol=1e-08, verbose=2),
                        "{} weights={}, axes={}, {!r}, {!r}".format(
                            method, weights, axes, a, b
                        ),
                    )

            for method in ("integral",):
                weights = "area"
                d_weights = f.weights(weights, components=True, measure=True)
                a = f.collapse(
                    method, axes=axes, weights=weights, measure=True
                ).data
                b = getattr(f.data, method)(axes=axes, weights=d_weights)
                self.assertTrue(
                    a.equals(b, rtol=1e-05, atol=1e-08, verbose=2),
                    "{} weighted axes={}, {!r}, {!r}".format(
                        method, axes, a, b
                    ),
                )

        for axes in axes_combinations(f):
            if axes == (0,):
                continue

            for method in ("var", "sd"):
                for weights in (None, "area"):
                    if weights is not None:
                        d_weights = f.weights(weights, components=True)
                    else:
                        d_weights = None

                    a = f.collapse(method, axes=axes, weights=weights).data
                    b = getattr(f.data, method)(
                        axes=axes, ddof=1, weights=d_weights
                    )
                    self.assertTrue(
                        a.equals(b, rtol=1e-05, atol=1e-08, verbose=2),
                        "{} weights={}, axes={}, {!r}, {!r}".format(
                            method, weights, axes, a, b
                        ),
                    )

            for method in ("mean_of_upper_decile",):
                for weights in (None, "area"):
                    if weights is not None:
                        d_weights = f.weights(weights, components=True)
                    else:
                        d_weights = None

                    a = f.collapse(method, axes=axes, weights=weights).data
                    b = getattr(f.data, method)(axes=axes, weights=d_weights)
                    self.assertTrue(
                        a.equals(b, rtol=1e-05, atol=1e-08, verbose=2),
                        "{} weights={}, axes={}, {!r}, {!r}".format(
                            method, weights, axes, a, b
                        ),
                    )

    def test_Field_all(self):
        f = self.f.copy()

        self.assertFalse(f.all())

        f[0, 0, 0] = 99
        self.assertTrue(f.all())

        f.del_data()
        self.assertFalse(f.all())

    def test_Field_any(self):
        f = self.f.copy()

        self.assertTrue(f.any())

        f.del_data()
        self.assertFalse(f.any())

    def test_Field_atol_rtol(self):
        f = self.f

        g = f.copy()
        self.assertTrue(f.equals(g, verbose=2))
        g[0, 0, 0] += 0.001

        self.assertFalse(f.equals(g))
        self.assertTrue(f.equals(g, atol=0.1, verbose=2))
        self.assertFalse(f.equals(g))
        self.assertEqual(cf.atol(), cf.ATOL())
        atol = cf.atol(0.1)
        self.assertTrue(f.equals(g, verbose=2))
        cf.atol(atol)
        self.assertFalse(f.equals(g))

        self.assertTrue(f.equals(g, rtol=10, verbose=2))
        self.assertFalse(f.equals(g))
        self.assertEqual(cf.rtol(), cf.RTOL())
        rtol = cf.rtol(10)
        self.assertTrue(f.equals(g, verbose=2))
        cf.rtol(rtol)
        self.assertFalse(f.equals(g))

        cf.atol(self.atol)
        cf.rtol(self.rtol)

    def test_Field_concatenate(self):
        f = self.f.copy()

        g = cf.Field.concatenate([f.copy()], axis=0)
        self.assertEqual(g.shape, (1, 10, 9))

        x = [f.copy() for i in range(8)]

        g = cf.Field.concatenate(x, axis=0)
        self.assertEqual(g.shape, (8, 10, 9))

        key = x[3].construct_key("latitude")
        x[3].del_construct(key)
        g = cf.Field.concatenate(x, axis=0)
        self.assertEqual(g.shape, (8, 10, 9))

        with self.assertRaises(Exception):
            g = cf.Field.concatenate([], axis=0)

    def test_Field_AUXILIARY_MASK(self):
        ac = numpy.ma.masked_all((3, 7))
        ac[0, 0:5] = [1.0, 2.0, 3.0, -99, 5.0]
        ac[0, 3] = numpy.ma.masked
        ac[1, 1:5] = [1.5, 2.5, 3.5, 4.5]
        ac[2, 3:7] = [1.0, 2.0, 3.0, 5.0]

        ae = numpy.ma.masked_all((3, 8))
        ae[0, 0:5] = [1.0, 2.0, 3.0, -99, 5.0]
        ae[0, 3] = numpy.ma.masked
        ae[1, 1:5] = [1.5, 2.5, 3.5, 4.5]
        ae[2, 3:8] = [1.0, 2.0, 3.0, -99, 5.0]
        ae[2, 6] = numpy.ma.masked

        af = numpy.ma.masked_all((4, 9))
        af[1, 0:5] = [1.0, 2.0, 3.0, -99, 5.0]
        af[1, 3] = numpy.ma.masked
        af[2, 1:5] = [1.5, 2.5, 3.5, 4.5]
        af[3, 3:8] = [1.0, 2.0, 3.0, -99, 5.0]
        af[3, 6] = numpy.ma.masked

        query1 = cf.wi(1, 5) & cf.ne(4)

        for chunksize in self.chunk_sizes[0:2]:
            cf.chunksize(chunksize)

            f = cf.read(self.contiguous)[0]

            for (method, shape, a) in zip(
                ["compress", "envelope", "full"],
                [ac.shape, ae.shape, af.shape],
                [ac, ae, af],
            ):
                message = "method={!r}".format(method)

                f.indices(method, time=query1)

                g = f.subspace(method, time=query1)
                t = g.coordinate("time")

                self.assertEqual(g.shape, shape, message)
                self.assertEqual(t.shape, shape, message)

                self.assertTrue(
                    (t.data._auxiliary_mask_return().array == a.mask).all(),
                    message,
                )
                self.assertTrue(
                    (g.data._auxiliary_mask_return().array == a.mask).all(),
                    message,
                )

                self.assertTrue(
                    cf.functions._numpy_allclose(t.array, a), message
                )

        cf.chunksize(self.original_chunksize)

        query2 = cf.set([1, 3, 5])

        ac2 = numpy.ma.masked_all((2, 6))
        ac2[0, 0] = 1
        ac2[0, 1] = 3
        ac2[0, 3] = 5
        ac2[1, 2] = 1
        ac2[1, 4] = 3
        ac2[1, 5] = 5

        ae2 = numpy.ma.where(
            (ae == 1) | (ae == 3) | (ae == 5), ae, numpy.ma.masked
        )
        af2 = numpy.ma.where(
            (af == 1) | (af == 3) | (af == 5), af, numpy.ma.masked
        )

        for chunksize in self.chunk_sizes[0:2]:
            cf.chunksize(chunksize)
            f = cf.read(self.contiguous)[0]

            for (method, shape, a) in zip(
                ["compress", "envelope", "full"],
                [ac2.shape, ae2.shape, af2.shape],
                [ac2, ae2, af2],
            ):

                message = "method={!r}".format(method)

                h = f.subspace("full", time=query1)
                g = h.subspace(method, time=query2)
                t = g.coordinate("time")

                self.assertTrue(g.shape == shape, message)
                self.assertTrue(t.shape == shape, message)

                self.assertTrue(
                    (t.data._auxiliary_mask_return().array == a.mask).all(),
                    message,
                )
                self.assertTrue(
                    (g.data._auxiliary_mask_return().array == a.mask).all(),
                    message,
                )

                self.assertTrue(
                    cf.functions._numpy_allclose(t.array, a), message
                )

        cf.chunksize(self.original_chunksize)

        ac3 = numpy.ma.masked_all((2, 3))
        ac3[0, 0] = -2
        ac3[1, 1] = 3
        ac3[1, 2] = 4

        ae3 = numpy.ma.masked_all((3, 6))
        ae3[0, 0] = -2
        ae3[2, 4] = 3
        ae3[2, 5] = 4

        af3 = numpy.ma.masked_all((3, 8))
        af3[0, 0] = -2
        af3[2, 4] = 3
        af3[2, 5] = 4

        query3 = cf.set([-2, 3, 4])

        for chunksize in self.chunk_sizes[0:2]:
            cf.chunksize(chunksize)
            f = cf.read(self.contiguous)[0].subspace[[0, 2, 3], 1:]

            for (method, shape, a) in zip(
                ["compress", "envelope", "full"],
                [ac3.shape, ae3.shape, af3.shape],
                [ac3, ae3, af3],
            ):

                message = "method={!r}".format(method)

                g = f.subspace(method, time=query3)
                t = g.coordinate("time")

                self.assertEqual(g.shape, shape, message)
                self.assertEqual(t.shape, shape, message)

                self.assertTrue(
                    (t.data._auxiliary_mask_return().array == a.mask).all(),
                    message,
                )
                self.assertTrue(
                    (g.data._auxiliary_mask_return().array == a.mask).all(),
                    message,
                )

                self.assertTrue(
                    cf.functions._numpy_allclose(t.array, a), message
                )

        cf.chunksize(self.original_chunksize)

    def test_Field__getitem__(self):
        f = self.f.copy().squeeze()
        d = f.data
        f = self.f.copy().squeeze()

        g = f[...]
        self.assertTrue((g.data == d).all())

        g = f[:, :]
        self.assertTrue((g.data == d).all())

        g = f[slice(None), :]
        self.assertTrue((g.data == d).all())

        g = f[:, slice(0, f.shape[1], 1)]
        self.assertTrue((g.data == d).all())

        g = f[slice(0, None, 1), slice(0, None)]
        self.assertTrue((g.data == d).all())

        g = f[3:7, 2:5]
        self.assertTrue((g.data == d[3:7, 2:5]).all())

        g = f[6:2:-1, 4:1:-1]
        self.assertTrue((g.data == d[6:2:-1, 4:1:-1]).all())

        g = f[[0, 3, 8], [1, 7, 8]]

        g = f[[8, 3, 0], [8, 7, 1]]

        g = f[[7, 4, 1], slice(6, 8)]

        g = f.squeeze()
        g[0:3, 5]

        g = f[0].squeeze()
        g[5]

    def test_Field__setitem__(self):
        f = self.f.copy().squeeze()

        f[...] = 0
        self.assertTrue((f == 0).all())
        f[3:7, 2:5] = -1
        self.assertTrue((f.array[3:7, 2:5] == -1).all())
        f[6:2:-1, 4:1:-1] = numpy.array(-1)
        self.assertTrue((f.array[6:2:-1, 4:1:-1] == -1).all())
        f[[0, 3, 8], [1, 7, 8]] = numpy.array([[[[-2]]]])
        self.assertTrue((f[[0, 3, 8], [1, 7, 8]].array == -2).all())
        f[[8, 3, 0], [8, 7, 1]] = cf.Data(-3, None)
        self.assertTrue((f[[8, 3, 0], [8, 7, 1]].array == -3).all())
        f[[7, 4, 1], slice(6, 8)] = [-4]
        self.assertTrue((f[[7, 4, 1], slice(6, 8)].array == -4).all())

        f = self.f.copy().squeeze()
        g = f.copy()
        f[...] = g
        self.assertTrue(f.data.allclose(g.data))
        g.del_data()
        with self.assertRaises(Exception):
            f[...] = g

        f[..., 0:2] = [99, 999]

        g = cf.FieldAncillary()
        g.set_data(f.data[0, 0])
        f[...] = g
        g.del_data()
        with self.assertRaises(Exception):
            f[...] = g

        g = cf.FieldAncillary()
        g.set_data(f.data[0, 0:2])

        f[..., 0:2] = g
        g.del_data()
        with self.assertRaises(Exception):
            f[..., 0:2] = g

    def test_Field__add__(self):
        f = self.f.copy()

        g = f * 0
        self.assertTrue((f + g).equals(f, verbose=2))
        self.assertTrue((g + f).equals(f, verbose=2))

        g.transpose(inplace=True)
        self.assertTrue((f + g).equals(f, verbose=2))

        for g in (f, f.copy(), f * 0):
            self.assertTrue((f + g).equals(g + f, verbose=2))
            self.assertTrue((g + f).equals(f + g, verbose=2))

        g = f.subspace(grid_longitude=[0]) * 0

        a = f + g
        b = g + f

        axis = a.domain_axis("grid_longitude", key=1)
        for key in a.field_ancillaries(filter_by_axis=(axis,), axis_mode="or"):
            a.del_construct(key)

        for key in a.cell_measures(filter_by_axis=(axis,), axis_mode="or"):
            a.del_construct(key)

        self.assertTrue(a.equals(b, verbose=2))
        self.assertTrue(b.equals(a, verbose=2))

        with self.assertRaises(Exception):
            f + ("a string",)

    def test_Field__mul__(self):
        f = self.f.copy().squeeze()

        f.standard_name = "qwerty"
        g = f * f

        self.assertIsNone(g.get_property("standard_name", None))

    def test_Field__gt__(self):
        f = self.f.copy().squeeze()

        f.standard_name = "qwerty"
        g = f > f.mean()

        self.assertTrue(g.Units.equals(cf.Units()))
        self.assertIsNone(g.get_property("standard_name", None))

    def test_Field_domain_mask(self):
        f = self.f.copy()

        f.domain_mask()
        f.domain_mask(grid_longitude=cf.wi(25, 31))

    def test_Field_cumsum(self):
        f = self.f.copy()

        g = f.copy()
        h = g.cumsum(2)
        self.assertIsNone(g.cumsum(2, inplace=True))
        self.assertTrue(g.equals(h, verbose=2))

        for axis in range(f.ndim):
            a = numpy.cumsum(f.array, axis=axis)
            self.assertTrue((f.cumsum(axis=axis).array == a).all())

        f[0, 0, 3] = cf.masked
        f[0, 2, 7] = cf.masked

        for axis in range(f.ndim):
            a = f.array
            a = numpy.cumsum(a, axis=axis)
            g = f.cumsum(axis=axis)
            self.assertTrue(cf.functions._numpy_allclose(g.array, a))

        for axis in range(f.ndim):
            g = f.cumsum(axis=axis, masked_as_zero=True)

            a = f.array
            mask = a.mask
            a = a.filled(0)
            a = numpy.cumsum(a, axis=axis)
            size = a.shape[axis]
            shape = [1] * a.ndim
            shape[axis] = size
            new_mask = numpy.cumsum(mask, axis=axis) == numpy.arange(
                1, size + 1
            ).reshape(shape)
            a = numpy.ma.array(a, mask=new_mask, copy=False)
            self.assertTrue(
                cf.functions._numpy_allclose(g.array, a, verbose=2)
            )

    def test_Field_flip(self):
        f = self.f.copy()

        g = f[(slice(None, None, -1),) * f.ndim]

        h = f.flip()
        self.assertTrue(h.equals(g, verbose=1))

        h = f.flip(f.get_data_axes())
        self.assertTrue(h.equals(g, verbose=1))

        h = f.flip(list(range(f.ndim)))
        self.assertTrue(h.equals(g, verbose=1))

        h = f.flip(["X", "Z", "Y"])
        self.assertTrue(h.equals(g, verbose=1))

        h = f.flip((re.compile("^atmos"), "grid_latitude", "grid_longitude"))
        self.assertTrue(h.equals(g, verbose=1))

        g = f.subspace(grid_longitude=slice(None, None, -1))
        self.assertIsNone(f.flip("X", inplace=True))
        self.assertTrue(f.equals(g, verbose=1))

    def test_Field_anchor(self):
        dimarray = self.f.dimension_coordinate("grid_longitude").array

        f = self.f.copy()
        f.cyclic("grid_longitude", period=45)
        self.assertIsNone(f.anchor("grid_longitude", 32, inplace=True))
        self.assertIsInstance(
            f.anchor("grid_longitude", 32, dry_run=True), dict
        )

        g = f.subspace(grid_longitude=[0])
        g.anchor("grid_longitude", 32)
        g.anchor("grid_longitude", 32, inplace=True)
        g.anchor("grid_longitude", 32, dry_run=True)

        f = self.f.copy()

        for period in (dimarray.min() - 5, dimarray.min()):
            anchors = numpy.arange(
                dimarray.min() - 3 * period, dimarray.max() + 3 * period, 6.5
            )

            f.cyclic("grid_longitude", period=period)

            # Increasing dimension coordinate
            for anchor in anchors:
                g = f.anchor("grid_longitude", anchor)
                x0 = g.coordinate("grid_longitude").datum(-1) - period
                x1 = g.coordinate("grid_longitude").datum(0)
                self.assertTrue(
                    x0 < anchor <= x1,
                    "INCREASING period=%s, x0=%s, anchor=%s, x1=%s"
                    % (period, x0, anchor, x1),
                )

            # Decreasing dimension coordinate
            flipped_f = f.flip("grid_longitude")
            for anchor in anchors:
                g = flipped_f.anchor("grid_longitude", anchor)
                x1 = g.coordinate("grid_longitude").datum(-1) + period
                x0 = g.coordinate("grid_longitude").datum(0)
                self.assertTrue(
                    x1 > anchor >= x0,
                    "DECREASING period={}, x0={}, anchor={}, x1={}".format(
                        period, x1, anchor, x0
                    ),
                )

    def test_Field_cell_area(self):
        f = self.f.copy()

        ca = f.cell_area()

        self.assertEqual(ca.ndim, 2)
        self.assertEqual(len(ca.dimension_coordinates()), 2)
        self.assertEqual(len(ca.domain_ancillaries()), 0)
        self.assertEqual(len(ca.coordinate_references()), 1)

        f.del_construct("cellmeasure0")
        y = f.dimension_coordinate("Y")
        y.set_bounds(y.create_bounds())
        self.assertEqual(len(f.cell_measures()), 0)

        ca = f.cell_area()

        self.assertEqual(ca.ndim, 2)
        self.assertEqual(len(ca.dimension_coordinates()), 2)
        self.assertEqual(len(ca.domain_ancillaries()), 0)
        self.assertEqual(len(ca.coordinate_references()), 1)
        self.assertTrue(ca.Units.equivalent(cf.Units("m2")), ca.Units)

        y = f.dimension_coordinate("Y")
        self.assertTrue(y.has_bounds())

    def test_Field_radius(self):
        f = self.f.copy()

        with self.assertRaises(Exception):
            f.radius()

        for default in ("earth", cf.field._earth_radius):
            r = f.radius(default=default)
            self.assertEqual(r.Units, cf.Units("m"))
            self.assertEqual(r, cf.field._earth_radius)

        a = cf.Data(1234, "m")
        for default in (
            1234,
            cf.Data(1234, "m"),
            cf.Data([1234], "m"),
            cf.Data([[1234]], "m"),
            cf.Data(1234, "m"),
            cf.Data(1.234, "km"),
        ):
            r = f.radius(default=default)
            self.assertEqual(r.Units, cf.Units("m"))
            self.assertEqual(r, a)

        with self.assertRaises(ValueError):
            f.radius()

        with self.assertRaises(ValueError):
            f.radius(default=[12, 34])

        with self.assertRaises(ValueError):
            f.radius(default=[[12, 34]])

        with self.assertRaises(ValueError):
            f.radius(default="qwerty")

        cr = f.coordinate_reference(
            "grid_mapping_name:rotated_latitude_longitude"
        )
        cr.datum.set_parameter("earth_radius", a.copy())

        r = f.radius(default=None)
        self.assertEqual(r.Units, cf.Units("m"))
        self.assertEqual(r, a)

        cr = f.coordinate_reference(
            "standard_name:atmosphere_hybrid_height_coordinate"
        )
        cr.datum.set_parameter("earth_radius", a.copy())

        r = f.radius(default=None)
        self.assertEqual(r.Units, cf.Units("m"))
        self.assertEqual(r, a)

        cr = f.coordinate_reference(
            "standard_name:atmosphere_hybrid_height_coordinate"
        )
        cr.datum.set_parameter("earth_radius", cf.Data(5678, "km"))

        with self.assertRaises(ValueError):
            f.radius(default=None)

        cr = f.coordinate_reference(
            "standard_name:atmosphere_hybrid_height_coordinate"
        )
        cr.datum.del_parameter("earth_radius")

        cr = f.coordinate_reference(
            "grid_mapping_name:rotated_latitude_longitude"
        )
        cr.datum.set_parameter("earth_radius", cf.Data([123, 456], "m"))

        with self.assertRaises(ValueError):
            f.radius(default=None)

    def test_Field_set_get_del_has_data(self):
        f = self.f.copy()

        f.rank
        f.data
        del f.data

        f = self.f.copy()

        self.assertTrue(f.has_data())
        data = f.get_data()
        f.del_data()
        f.get_data(default=None)
        f.del_data(default=None)
        self.assertFalse(f.has_data())
        f.set_data(data, axes=None)
        f.set_data(data, axes=None, copy=False)
        self.assertTrue(f.has_data())

        f = self.f.copy()
        f.del_data_axes()
        self.assertFalse(f.has_data_axes())
        self.assertIsNone(f.del_data_axes(default=None))

        f = self.f.copy()
        for key in f.constructs.filter_by_data():
            self.assertTrue(f.has_data_axes(key))
            f.get_data_axes(key)
            f.del_data_axes(key)
            self.assertIsNone(f.del_data_axes(key, default=None))
            self.assertIsNone(f.get_data_axes(key, default=None))
            self.assertFalse(f.has_data_axes(key))

        g = cf.Field()
        g.set_data(cf.Data(9))
        with self.assertRaises(Exception):
            g.set_data(cf.Data(9), axes="X")

        g = self.f.copy()
        with self.assertRaises(Exception):
            g.set_data(cf.Data([9], axes="qwerty"))

        with self.assertRaises(Exception):
            g.set_data(cf.Data([9], axes=["X", "Y"]))

        g = cf.Field()
        g.set_data(cf.Data(9))
        with self.assertRaises(Exception):
            g.set_data(cf.Data(9), axes="X")

        g = cf.Field()
        a = g.set_construct(cf.DomainAxis(9))
        b = g.set_construct(cf.DomainAxis(10))
        g.set_data(cf.Data(list(range(9))), axes=a)
        with self.assertRaises(Exception):
            g.set_data(cf.Data(list(range(9))), axes=b)
        with self.assertRaises(Exception):
            g.set_data(cf.Data(list(range(9))), axes=[b, a])

        # Test inplace
        f = self.f.copy()
        d = f.del_data()
        g = f.set_data(d, inplace=False)
        self.assertIsInstance(g, cf.Field)
        self.assertFalse(f.has_data())
        self.assertTrue(g.has_data())
        self.assertTrue(g.data.equals(d))

        g = cf.Field()
        #        with self.assertRaises(Exception):
        #            g.set_data(cf.Data(list(range(9))))
        g.set_construct(cf.DomainAxis(9))
        g.set_construct(cf.DomainAxis(9))
        g.set_construct(cf.DomainAxis(10))
        g.set_construct(cf.DomainAxis(8))
        with self.assertRaises(Exception):
            g.set_data(cf.Data(numpy.arange(81).reshape(9, 9)))
        with self.assertRaises(Exception):
            g.set_data(cf.Data(numpy.arange(90).reshape(9, 10)))
        g.set_data(cf.Data(numpy.arange(80).reshape(10, 8)))
        with self.assertRaises(Exception):
            g.set_data(cf.Data(numpy.arange(8)))
        with self.assertRaises(Exception):
            g.set_data(cf.Data(numpy.arange(90).reshape(10, 9)))

    def test_Field_get_data_axes(self):
        f = self.f
        self.assertEqual(
            f.get_data_axes(),
            ("domainaxis0", "domainaxis1", "domainaxis2"),
            str(f.get_data_axes()),
        )

        f = cf.Field()
        f.set_data(cf.Data(9), axes=())
        self.assertEqual(f.get_data_axes(), ())

        f.del_data()
        self.assertEqual(f.get_data_axes(), ())

        f.del_data_axes()
        self.assertIsNone(f.get_data_axes(default=None))

    def test_Field_equals(self):
        f = self.f.copy()
        g = f.copy()
        self.assertTrue(f.equals(f, verbose=2))
        self.assertTrue(f.equals(g, verbose=2))
        g.set_property("foo", "bar")
        self.assertFalse(f.equals(g))
        g = f.copy()
        self.assertFalse(f.equals(g + 1))

        # Symmetry
        f = cf.example_field(2)
        g = f.copy()
        self.assertTrue(f.equals(g, verbose=2))
        self.assertTrue(g.equals(f, verbose=2))

        g.del_construct("dimensioncoordinate0")
        self.assertFalse(f.equals(g, verbose=2))
        self.assertFalse(g.equals(f, verbose=2))

    def test_Field_insert_dimension(self):
        f = self.f.copy()
        f.squeeze("Z", inplace=True)
        self.assertEqual(f.ndim, 2)
        g = f.copy()

        self.assertIsNone(g.insert_dimension("Z", inplace=True))

        self.assertEqual(g.ndim, f.ndim + 1)
        self.assertEqual(g.get_data_axes()[1:], f.get_data_axes())

        with self.assertRaises(ValueError):
            f.insert_dimension(1, "qwerty")

    # i#    def test_Field_indices(self):
    # i#        f = self.f.copy()
    # i#
    # i#        array = numpy.ma.array(f.array)
    # i#
    # i#        x = f.dimension_coordinate("X")
    # i#        a = x.varray
    # i#        a[...] = numpy.arange(0, 360, 40)
    # i#        x.set_bounds(x.create_bounds())
    # i#        f.cyclic("X", iscyclic=True, period=360)
    # i#
    # i#        f0 = f.copy()
    # i#
    # i#        # wi (increasing)
    # i#        indices = f.indices(grid_longitude=cf.wi(50, 130))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 2), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [80, 120]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.wi(-90, 50))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 4), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [-80, -40, 0, 40]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.wi(310, 450))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 4), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [-40, 0, 40, 80]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.wi(310 - 1080, 450 - 1080))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 4), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [-40, 0, 40, 80]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.wi(310 + 720, 450 + 720))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 4), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [-40, 0, 40, 80]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.wi(-90, 370))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 9), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue(
    # i#            (x == [-80, -40, 0, 40, 80, 120, 160, 200, 240.0]).all()
    # i#        )
    # i#
    # i#        with self.assertRaises(IndexError):
    # i#            f.indices(grid_longitude=cf.wi(90, 100))
    # i#
    # i#        indices = f.indices("full", grid_longitude=cf.wi(310, 450))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 9), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertEqual(x.shape, (9,), x.shape)
    # i#        self.assertTrue(
    # i#            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320]).all(), x
    # i#        )
    # i#        a = array.copy()
    # i#        a[..., [3, 4, 5, 6, 7]] = numpy.ma.masked
    # i#        self.assertTrue(cf.functions._numpy_allclose(g.array, a), g.array)
    # i#
    # i#        indices = f.indices("full", grid_longitude=cf.wi(70, 200))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 9), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertEqual(x.shape, (9,), x.shape)
    # i#        self.assertTrue(
    # i#            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320]).all(), x
    # i#        )
    # i#        a = array.copy()
    # i#        a[..., [0, 1, 6, 7, 8]] = numpy.ma.masked
    # i#        self.assertTrue(cf.functions._numpy_allclose(g.array, a), g.array)
    # i#
    # i#        # wi (decreasing)
    # i#        f.flip("X", inplace=True)
    # i#
    # i#        indices = f.indices(grid_longitude=cf.wi(50, 130))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 2), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [80, 120][::-1]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.wi(-90, 50))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 4), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [-80, -40, 0, 40][::-1]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.wi(310, 450))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 4), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.wi(310 - 1080, 450 - 1080))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 4), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.wi(310 + 720, 450 + 720))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 4), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())
    # i#
    # i#        with self.assertRaises(IndexError):
    # i#            f.indices(grid_longitude=cf.wi(90, 100))
    # i#
    # i#        indices = f.indices("full", grid_longitude=cf.wi(310, 450))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 9), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertEqual(x.shape, (9,), x.shape)
    # i#        self.assertTrue(
    # i#            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320][::-1]).all(), x
    # i#        )
    # i#
    # i#        indices = f.indices("full", grid_longitude=cf.wi(70, 200))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 9), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertEqual(x.shape, (9,), x.shape)
    # i#        self.assertTrue(
    # i#            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320][::-1]).all(), x
    # i#        )
    # i#
    # i#        # wo
    # i#        f = f0.copy()
    # i#
    # i#        indices = f.indices(grid_longitude=cf.wo(50, 130))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 7), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [-200, -160, -120, -80, -40, 0, 40]).all())
    # i#
    # i#        with self.assertRaises(IndexError):
    # i#            f.indices(grid_longitude=cf.wo(-90, 370))
    # i#
    # i#        # set
    # i#        indices = f.indices(grid_longitude=cf.set([320, 40, 80, 99999]))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 3), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [40, 80, 320]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.lt(90))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 3), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [0, 40, 80]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.gt(90))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 6), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [120, 160, 200, 240, 280, 320]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.le(80))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 3), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [0, 40, 80]).all())
    # i#
    # i#        indices = f.indices(grid_longitude=cf.ge(80))
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 7), g.shape)
    # i#        x = g.dimension_coordinate("X").array
    # i#        self.assertTrue((x == [80, 120, 160, 200, 240, 280, 320]).all())
    # i#
    # i#        # 2-d
    # i#        lon = f.construct("longitude").array
    # i#        lon = numpy.transpose(lon)
    # i#        lon = numpy.expand_dims(lon, 0)
    # i#
    # i#        lat = f.construct("latitude").array
    # i#        lat = numpy.expand_dims(lat, 0)
    # i#
    # i#        array = numpy.ma.where(
    # i#            (lon >= 92) & (lon <= 134), f.array, numpy.ma.masked
    # i#        )
    # i#
    # i#        for mode in ("", "compress", "full", "envelope"):
    # i#            indices = f.indices(mode, longitude=cf.wi(92, 134))
    # i#            g = f[indices]
    # i#            if mode == "full":
    # i#                shape = (1, 10, 9)
    # i#                array2 = array
    # i#            elif mode == "envelope":
    # i#                shape = (1, 10, 5)
    # i#                array2 = array[..., 3:8]
    # i#            else:
    # i#                shape = (1, 10, 5)
    # i#                array2 = array[..., 3:8]
    # i#
    # i#            self.assertEqual(g.shape, shape, str(g.shape) + "!=" + str(shape))
    # i#            self.assertTrue(
    # i#                cf.functions._numpy_allclose(array2, g.array), g.array
    # i#            )
    # i#
    # i#        array = numpy.ma.where(
    # i#            ((lon >= 72) & (lon <= 83)) | (lon >= 118),
    # i#            f.array,
    # i#            numpy.ma.masked,
    # i#        )
    # i#
    # i#        for mode in ("", "compress", "full", "envelope"):
    # i#            indices = f.indices(mode, longitude=cf.wi(72, 83) | cf.gt(118))
    # i#            g = f[indices]
    # i#            if mode == "full":
    # i#                shape = (1, 10, 9)
    # i#            elif mode == "envelope":
    # i#                shape = (1, 10, 8)
    # i#            else:
    # i#                shape = (1, 10, 6)
    # i#
    # i#            self.assertEqual(g.shape, shape, str(g.shape) + "!=" + str(shape))
    # i#
    # i#        indices = f.indices(
    # i#            "full",
    # i#            longitude=cf.wi(92, 134),
    # i#            latitude=cf.wi(-26, -20) | cf.ge(30),
    # i#        )
    # i#        g = f[indices]
    # i#        self.assertEqual(g.shape, (1, 10, 9), g.shape)
    # i#        array = numpy.ma.where(
    # i#            (
    # i#                ((lon >= 92) & (lon <= 134))
    # i#                & (((lat >= -26) & (lat <= -20)) | (lat >= 30))
    # i#            ),
    # i#            f.array,
    # i#            numpy.ma.masked,
    # i#        )
    # i#        self.assertTrue(cf.functions._numpy_allclose(array, g.array), g.array)
    # i#
    # i#        for mode in ("", "compress", "full", "envelope"):
    # i#            indices = f.indices(mode, grid_longitude=cf.contains(23.2))
    # i#            g = f[indices]
    # i#            if mode == "full":
    # i#                shape = f.shape
    # i#            else:
    # i#                shape = (1, 10, 1)
    # i#
    # i#            self.assertEqual(g.shape, shape, g.shape)
    # i#
    # i#            if mode != "full":
    # i#                self.assertEqual(
    # i#                    g.construct("grid_longitude").array, 40
    # i#                )  # TODO
    # i#
    # i#        for mode in ("", "compress", "full", "envelope"):
    # i#            indices = f.indices(mode, grid_latitude=cf.contains(3))
    # i#            g = f[indices]
    # i#            if mode == "full":
    # i#                shape = f.shape
    # i#            else:
    # i#                shape = (1, 1, 9)
    # i#
    # i#            self.assertEqual(g.shape, shape, g.shape)
    # i#
    # i#            if mode != "full":
    # i#                self.assertEqual(g.construct("grid_latitude").array, 3)
    # i#
    # i#        for mode in ("", "compress", "full", "envelope"):
    # i#            indices = f.indices(mode, longitude=cf.contains(83))
    # i#            g = f[indices]
    # i#            if mode == "full":
    # i#                shape = f.shape
    # i#            else:
    # i#                shape = (1, 1, 1)
    # i#
    # i#            self.assertEqual(g.shape, shape, g.shape)
    # i#
    # i#            if mode != "full":
    # i#                self.assertEqual(g.construct("longitude").array, 83)
    # i#
    # i#        # Calls that should fail
    # i#        with self.assertRaises(Exception):
    # i#            f.indices(longitude=cf.gt(23), grid_longitude=cf.wi(92, 134))
    # i#        with self.assertRaises(Exception):
    # i#            f.indices(grid_longitude=cf.gt(23), longitude=cf.wi(92, 134))
    # i#        with self.assertRaises(Exception):
    # i#            f.indices(grid_latitude=cf.contains(-23.2))

    def test_Field_indices(self):
        filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_file.nc"
        )
        f = cf.read(filename)[0]

        array = numpy.ma.array(f.array)

        x = f.dimension_coordinate("X")
        a = x.varray
        a[...] = numpy.arange(0, 360, 40)
        x.set_bounds(x.create_bounds())
        f.cyclic("X", iscyclic=True, period=360)

        f0 = f.copy()

        # wi (increasing)
        indices = f.indices(grid_longitude=cf.wi(50, 130))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 2), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [80, 120]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 50))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-80, -40, 0, 40]).all())

        indices = f.indices(grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(310 - 1080, 450 - 1080))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(310 + 720, 450 + 720))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 370))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue(
            (x == [-80, -40, 0, 40, 80, 120, 160, 200, 240.0]).all()
        )

        with self.assertRaises(IndexError):
            f.indices(grid_longitude=cf.wi(90, 100))

        indices = f.indices("full", grid_longitude=cf.wi(310, 450))
        self.assertTrue(indices[0], "mask")
        self.assertTrue(
            (
                indices[1][0].array
                == [
                    [
                        [
                            False,
                            False,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                        ]
                    ]
                ]
            ).all()
        )
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)

        x = g.dimension_coordinate("X").array
        self.assertEqual(x.shape, (9,), x.shape)

        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320]).all(), x
        )

        a = array.copy()
        a[..., 3:8] = numpy.ma.masked

        self.assertTrue(cf.functions._numpy_allclose(g.array, a), g.array)

        indices = f.indices("full", grid_longitude=cf.wi(70, 200))
        self.assertTrue(indices[0], "mask")
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertEqual(x.shape, (9,), x.shape)
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320]).all(), x
        )
        a = array.copy()
        a[..., [0, 1, 6, 7, 8]] = numpy.ma.masked
        self.assertTrue(cf.functions._numpy_allclose(g.array, a), g.array)

        # wi (decreasing)
        f.flip("X", inplace=True)

        indices = f.indices(grid_longitude=cf.wi(50, 130))
        self.assertTrue(indices[0], "mask")
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 2), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [80, 120][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 50))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-80, -40, 0, 40][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310 - 1080, 450 - 1080))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310 + 720, 450 + 720))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        with self.assertRaises(IndexError):
            f.indices(grid_longitude=cf.wi(90, 100))

        indices = f.indices("full", grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertEqual(x.shape, (9,), x.shape)
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320][::-1]).all(), x
        )

        indices = f.indices("full", grid_longitude=cf.wi(70, 200))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertEqual(x.shape, (9,), x.shape)
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320][::-1]).all(), x
        )

        # wo
        f = f0.copy()

        indices = f.indices(grid_longitude=cf.wo(50, 130))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 7), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-200, -160, -120, -80, -40, 0, 40]).all())

        with self.assertRaises(IndexError):
            f.indices(grid_longitude=cf.wo(-90, 370))

        # set
        indices = f.indices(grid_longitude=cf.set([320, 40, 80, 99999]))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 3), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [40, 80, 320]).all())

        indices = f.indices(grid_longitude=cf.lt(90))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 3), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.gt(90))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 6), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [120, 160, 200, 240, 280, 320]).all())

        indices = f.indices(grid_longitude=cf.le(80))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 3), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.ge(80))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 7), g.shape)
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [80, 120, 160, 200, 240, 280, 320]).all())

        # 2-d
        lon = f.construct("longitude").array
        lon = numpy.transpose(lon)
        lon = numpy.expand_dims(lon, 0)

        lat = f.construct("latitude").array
        lat = numpy.expand_dims(lat, 0)

        array = numpy.ma.where(
            (lon >= 92) & (lon <= 134), f.array, numpy.ma.masked
        )

        for mode in ("compress", "full", "envelope"):
            indices = f.indices(mode, longitude=cf.wi(92, 134))
            g = f[indices]
            if mode == "full":
                shape = (1, 10, 9)
                array2 = array
            elif mode == "envelope":
                shape = (1, 10, 5)
                array2 = array[..., 3:8]
            else:
                shape = (1, 10, 5)
                array2 = array[..., 3:8]

            self.assertEqual(g.shape, shape, str(g.shape) + "!=" + str(shape))
            self.assertTrue(
                cf.functions._numpy_allclose(array2, g.array), g.array
            )

        array = numpy.ma.where(
            ((lon >= 72) & (lon <= 83)) | (lon >= 118),
            f.array,
            numpy.ma.masked,
        )

        for mode in ((), ("compress",), ("full",), ("envelope",)):
            indices = f.indices(*mode, longitude=cf.wi(72, 83) | cf.gt(118))
            g = f[indices]
            if mode == ("full",):
                shape = (1, 10, 9)
            elif mode == ("envelope",):
                shape = (1, 10, 8)
            else:
                shape = (1, 10, 6)

            self.assertEqual(g.shape, shape, str(g.shape) + "!=" + str(shape))

        indices = f.indices(
            "full",
            longitude=cf.wi(92, 134),
            latitude=cf.wi(-26, -20) | cf.ge(30),
        )
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)
        array = numpy.ma.where(
            (
                ((lon >= 92) & (lon <= 134))
                & (((lat >= -26) & (lat <= -20)) | (lat >= 30))
            ),
            f.array,
            numpy.ma.masked,
        )
        self.assertTrue(cf.functions._numpy_allclose(array, g.array), g.array)

        for mode in ("compress", "full", "envelope"):
            indices = f.indices(mode, grid_longitude=cf.contains(23.2))
            g = f[indices]
            if mode == "full":
                shape = f.shape
            else:
                shape = (1, 10, 1)

            self.assertEqual(g.shape, shape, g.shape)

            if mode != "full":
                self.assertEqual(
                    g.construct("grid_longitude").array, 40
                )  # TODO

        for mode in ("compress", "full", "envelope"):
            indices = f.indices(mode, grid_latitude=cf.contains(3))
            g = f[indices]
            if mode == "full":
                shape = f.shape
            else:
                shape = (1, 1, 9)

            self.assertEqual(g.shape, shape, g.shape)

            if mode != "full":
                self.assertEqual(g.construct("grid_latitude").array, 3)

        for mode in ("compress", "full", "envelope"):
            indices = f.indices(mode, longitude=cf.contains(83))
            g = f[indices]
            if mode == "full":
                shape = f.shape
            else:
                shape = (1, 1, 1)

            self.assertEqual(g.shape, shape, g.shape)

            if mode != "full":
                self.assertEqual(g.construct("longitude").array, 83)

        # Calls that should fail
        with self.assertRaises(Exception):
            f.indices(longitude=cf.gt(23), grid_longitude=cf.wi(92, 134))
        with self.assertRaises(Exception):
            f.indices(grid_longitude=cf.gt(23), longitude=cf.wi(92, 134))
        with self.assertRaises(Exception):
            f.indices(grid_latitude=cf.contains(-23.2))

    def test_Field_match(self):
        f = self.f.copy()
        f.long_name = "qwerty"
        f.nc_set_variable("tas")

        # match, match_by_identity
        for identities in (
            [],
            ["eastward_wind"],
            ["standard_name=eastward_wind"],
            ["long_name=qwerty"],
            ["ncvar%tas"],
            [re.compile("^eastw")],
            ["eastward_wind", "long_name=qwerty"],
            ["None", "eastward_wind"],
        ):
            self.assertTrue(
                f.match(*identities), f"Failed with {identities!r}"
            )
            self.assertTrue(
                f.match_by_identity(*identities), f"Failed with {identities!r}"
            )

        # match_by_property
        for mode in ([], ["and"]):
            for properties in (
                {},
                {"standard_name": "eastward_wind"},
                {"long_name": "qwerty"},
                {"standard_name": re.compile("^eastw")},
                {"standard_name": "eastward_wind", "long_name": "qwerty"},
            ):
                self.assertTrue(
                    f.match_by_property(*mode, **properties),
                    f"Failed with {mode} {properties}",
                )

        for mode in (["or"],):
            for properties in (
                {},
                {"standard_name": "eastward_wind"},
                {"long_name": "qwerty"},
                {"standard_name": re.compile("^eastw")},
                {"standard_name": "eastward_wind", "long_name": "qwerty"},
                {"standard_name": "None", "long_name": "qwerty"},
            ):
                self.assertTrue(
                    f.match_by_property(*mode, **properties),
                    f"Failed with {mode} {properties}",
                )
        # match_by_units
        self.assertTrue(f.match_by_units("m s-1"))
        self.assertTrue(f.match_by_units("km h-1", exact=False))
        self.assertFalse(f.match_by_units("km h-1"))
        self.assertFalse(f.match_by_units("K s"))

        self.assertTrue(f.match_by_units(cf.Units("m s-1")))
        self.assertTrue(f.match_by_units(cf.Units("km h-1"), exact=False))
        self.assertFalse(f.match_by_units(cf.Units("km h-1")))
        self.assertFalse(f.match_by_units(cf.Units("K s")))

        # match_by_rank
        self.assertTrue(f.match_by_rank())
        self.assertTrue(f.match_by_rank(3))
        self.assertTrue(f.match_by_rank(99, 3))
        self.assertFalse(f.match_by_rank(99))
        self.assertFalse(f.match_by_rank(99, 88))

        # match_by_naxes
        self.assertTrue(f.match_by_naxes())
        self.assertTrue(f.match_by_naxes(3))
        self.assertTrue(f.match_by_naxes(99, 3))
        self.assertFalse(f.match_by_naxes(99))
        self.assertFalse(f.match_by_naxes(99, 88))
        g = f.copy()
        g.del_data()
        self.assertTrue(g.match_by_naxes())
        self.assertFalse(g.match_by_naxes(3))
        self.assertFalse(g.match_by_naxes(99, 88))

        # Match by construct
        for OR in (True, False):
            self.assertTrue(f.match_by_construct(OR=OR))
            self.assertTrue(f.match_by_construct("X", OR=OR))
            self.assertTrue(f.match_by_construct("latitude", OR=OR))
            self.assertTrue(f.match_by_construct("X", "latitude", OR=OR))
            self.assertTrue(f.match_by_construct("X", "Y", OR=OR))
            self.assertTrue(f.match_by_construct("X", "Y", "latitude", OR=OR))
            self.assertTrue(f.match_by_construct("grid_latitude: max", OR=OR))
            self.assertTrue(
                f.match_by_construct(
                    "grid_longitude: mean grid_latitude: max", OR=OR
                )
            )
            self.assertTrue(f.match_by_construct("X", "method:max", OR=OR))
            self.assertTrue(
                f.match_by_construct("X", "grid_latitude: max", OR=OR)
            )

        self.assertFalse(f.match_by_construct("qwerty"))
        self.assertFalse(f.match_by_construct("qwerty", OR=True))
        self.assertFalse(f.match_by_construct("X", "qwerty"))
        self.assertFalse(f.match_by_construct("time: mean"))

        self.assertTrue(f.match_by_construct("X", "qwerty", OR=True))
        self.assertTrue(
            f.match_by_construct(
                "X", "qwerty", "method:max", "over:years", OR=True
            )
        )
        self.assertTrue(
            f.match_by_construct(
                "X", "qwerty", "grid_latitude: max", "over:years", OR=True
            )
        )

    def test_Field_autocyclic(self):
        f = self.f.copy()

        self.assertFalse(f.autocyclic())
        f.dimension_coordinate("X").del_bounds()
        f.autocyclic()

    def test_Field_construct_key(self):
        self.f.construct_key("grid_longitude")

    def test_Field_convolution_filter(self):
        if not SCIPY_AVAILABLE:  # needed for 'convolution_filter' method
            raise unittest.SkipTest("SciPy must be installed for this test.")

        window = [0.1, 0.15, 0.5, 0.15, 0.1]

        f = cf.read(self.filename1)[0]

        # Test user weights in different modes
        for mode in ("reflect", "constant", "nearest", "mirror", "wrap"):
            g = f.convolution_filter(window, axis=-1, mode=mode, cval=0.0)
            self.assertTrue(
                (
                    g.array == convolve1d(f.array, window, axis=-1, mode=mode)
                ).all()
            )

    def test_Field_moving_window(self):
        if not SCIPY_AVAILABLE:  # needed for 'moving_window' method
            raise unittest.SkipTest("SciPy must be installed for this test.")

        weights = cf.Data([1, 2, 3, 10, 5, 6, 7, 8]) / 2

        f = cf.example_field(0)

        g = f.moving_window("mean", window_size=3, axis="X", inplace=True)
        self.assertIsNone(g)

        with self.assertRaises(ValueError):
            f.moving_window("mean", window_size=3, axis="X", cval=39)

        f = cf.example_field(0)
        a = f.array

        # ------------------------------------------------------------
        # Origin = 0
        # ------------------------------------------------------------
        for method in ("mean", "sum", "integral"):
            for mode in ("constant", "wrap", "reflect", "nearest", "mirror"):
                g = f.moving_window(
                    method, window_size=3, axis="X", weights=weights, mode=mode
                )

                for i in range(1, 7):
                    if method in ("mean", "integral"):
                        x = (a[:, i - 1 : i + 2] * weights[i - 1 : i + 2]).sum(
                            axis=1
                        )

                    if method == "sum":
                        x = a[:, i - 1 : i + 2].sum(axis=1)

                    if method == "mean":
                        x /= weights[i - 1 : i + 2].sum()

                    numpy.testing.assert_allclose(x, g.array[:, i])

            # Test 'wrap'
            for mode in (None, "wrap"):
                g = f.moving_window(
                    method, window_size=3, axis="X", weights=weights, mode=mode
                )

                for i, ii in zip([0, -1], ([0, 1, -1], [0, -2, -1])):
                    if method in ("mean", "integral"):
                        x = (a[:, ii] * weights[ii]).sum(axis=1)

                    if method == "sum":
                        x = a[:, ii].sum(axis=1)

                    if method == "mean":
                        x /= weights[ii].sum()

                    numpy.testing.assert_allclose(x, g.array[:, i])

            # ------------------------------------------------------------
            # Origin = 1
            # ------------------------------------------------------------
            for mode in ("constant", "wrap", "reflect", "nearest", "mirror"):
                g = f.moving_window(
                    method,
                    window_size=3,
                    axis="X",
                    weights=weights,
                    mode=mode,
                    origin=1,
                )

                for i in range(0, 6):
                    ii = slice(i, i + 3)

                    if method in ("mean", "integral"):
                        x = (a[:, ii] * weights[ii]).sum(axis=1)

                    if method == "sum":
                        x = a[:, ii].sum(axis=1)

                    if method == "mean":
                        x /= weights[ii].sum()

                    numpy.testing.assert_allclose(x, g.array[:, i])

            # Test 'wrap'
            for mode in (None, "wrap"):
                g = f.moving_window(
                    method,
                    window_size=3,
                    axis="X",
                    weights=weights,
                    mode=mode,
                    origin=1,
                )

                for i, ii in zip([-2, -1], ([0, -2, -1], [0, 1, -1])):
                    if method in ("mean", "integral"):
                        x = (a[:, ii] * weights[ii]).sum(axis=1)

                    if method == "sum":
                        x = a[:, ii].sum(axis=1)

                    if method == "mean":
                        x /= weights[ii].sum()

                    numpy.testing.assert_allclose(x, g.array[:, i])

            # ------------------------------------------------------------
            # Constant
            # ------------------------------------------------------------
            for constant in (None, 0):
                g = f.moving_window(
                    method,
                    window_size=3,
                    axis="X",
                    weights=weights,
                    mode="constant",
                    cval=constant,
                )
                for i, ii in zip([0, -1], ([0, 1], [-2, -1])):
                    if method in ("mean", "integral"):
                        x = (a[:, ii] * weights[ii]).sum(axis=1)

                    if method == "sum":
                        x = a[:, ii].sum(axis=1)

                    if method == "mean":
                        x /= weights[ii].sum()

                    numpy.testing.assert_allclose(x, g.array[:, i])

        # ------------------------------------------------------------
        # Weights broadcasting
        # ------------------------------------------------------------
        weights = cf.Data(numpy.arange(1, 6.0)) / 2
        g = f.moving_window("mean", window_size=3, axis="Y", weights=weights)

        with self.assertRaises(ValueError):
            f.moving_window("mean", window_size=3, axis="X", weights=weights)

        self.assertEqual(len(g.cell_methods()), len(f.cell_methods()) + 1)

    def test_Field_derivative(self):
        if not SCIPY_AVAILABLE:  # needed for 'derivative' method
            raise unittest.SkipTest("SciPy must be installed for this test.")

        x_min = 0.0
        x_max = 359.0
        dx = 1.0

        x_1d = numpy.arange(x_min, x_max, dx)

        data_1d = x_1d * 2.0 + 1.0

        dim_x = cf.DimensionCoordinate(
            data=cf.Data(x_1d, "s"), properties={"axis": "X"}
        )

        f = cf.Field()
        f.set_construct(cf.DomainAxis(size=x_1d.size))
        f.set_construct(dim_x)
        f.set_data(cf.Data(data_1d, "m"), axes="X")
        f.cyclic("X", period=360.0)

        g = f.derivative("X")
        self.assertTrue((g.array == 2.0).all())

        g = f.derivative("X", one_sided_at_boundary=True)
        self.assertTrue((g.array == 2.0).all())

        g = f.derivative("X", wrap=True)
        self.assertTrue((g.array == 2.0).all())

    def test_Field_convert(self):
        f = self.f.copy()

        c = f.convert("grid_latitude")
        self.assertTrue(c.ndim == 1)
        self.assertTrue(c.standard_name == "grid_latitude")
        self.assertTrue(len(c.dimension_coordinates()) == 1)
        self.assertTrue(len(c.auxiliary_coordinates()) == 1)
        self.assertTrue(len(c.cell_measures()) == 0)
        self.assertTrue(len(c.coordinate_references()) == 1)
        self.assertTrue(len(c.domain_ancillaries()) == 0)
        self.assertTrue(len(c.field_ancillaries()) == 0)
        self.assertTrue(len(c.cell_methods()) == 0)

        c = f.convert("latitude")
        self.assertTrue(c.ndim == 2)
        self.assertTrue(c.standard_name == "latitude")
        self.assertTrue(len(c.dimension_coordinates()) == 2)
        self.assertTrue(len(c.auxiliary_coordinates()) == 3)
        self.assertTrue(len(c.cell_measures()) == 1)
        self.assertTrue(len(c.coordinate_references()) == 1)
        self.assertTrue(len(c.domain_ancillaries()) == 0)
        self.assertTrue(len(c.field_ancillaries()) == 0)
        self.assertTrue(len(c.cell_methods()) == 0)

        # Cellsize
        c = f.convert("grid_longitude", cellsize=True)
        self.assertTrue(
            (c.data == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.5, 6.0]).all()
        )

        with self.assertRaises(ValueError):
            f.convert("qwerty")

    def test_Field_section(self):
        f = cf.read(self.filename2)[0][0:10]
        g = f.section(("X", "Y"))
        self.assertEqual(len(g), 10, "len(g)={}".format(len(g)))

    def test_Field_squeeze(self):
        f = self.f.copy()

        self.assertIsNone(f.squeeze(inplace=True))

        h = f.copy()
        h.squeeze(inplace=True)
        self.assertTrue(f.equals(h))

        f = self.f.copy()
        self.assertIsNone(f.squeeze(0, inplace=True))

    def test_Field_unsqueeze(self):
        f = self.f.copy()
        self.assertEqual(f.ndim, 3)

        f.squeeze(inplace=True)
        self.assertEqual(f.ndim, 2)

        g = f.copy()
        self.assertIsNone(g.unsqueeze(inplace=True))
        self.assertEqual(g.ndim, 3)

        g = f.unsqueeze()
        self.assertEqual(g.ndim, 3)
        self.assertEqual(f.ndim, 2)

    def test_Field_auxiliary_coordinate(self):
        f = self.f

        for identity in ("auxiliarycoordinate1", "latitude"):
            key, c = f.construct_item(identity)
            self.assertTrue(
                f.auxiliary_coordinate(identity).equals(c, verbose=2)
            )
            self.assertEqual(f.auxiliary_coordinate(identity, key=True), key)

        with self.assertRaises(ValueError):
            f.aux("long_name:qwerty")

    def test_Field_coordinate(self):
        f = self.f

        for identity in (
            "latitude",
            "grid_longitude",
            "auxiliarycoordinate1",
            "dimensioncoordinate1",
        ):
            key, c = f.construct(identity, item=True)

        with self.assertRaises(ValueError):
            f.coord("long_name:qweRty")

    def test_Field_coordinate_reference(self):
        f = self.f.copy()

        for identity in (
            "coordinatereference1",
            "key%coordinatereference0",
            "standard_name:atmosphere_hybrid_height_coordinate",
            "grid_mapping_name:rotated_latitude_longitude",
        ):
            key = f.construct_key(identity)
            c = f.construct(identity)

            self.assertTrue(
                f.coordinate_reference(identity).equals(c, verbose=2)
            )
            self.assertEqual(f.coordinate_reference(identity, key=True), key)

        key = f.construct_key(
            "standard_name:atmosphere_hybrid_height_coordinate"
        )
        self.assertEqual(
            f.coordinate_reference(
                "standard_name:atmosphere_hybrid_height_coordinate", key=True
            ),
            key,
        )

        key = f.construct_key("grid_mapping_name:rotated_latitude_longitude")
        self.assertEqual(
            f.coordinate_reference(
                "grid_mapping_name:rotated_latitude_longitude", key=True
            ),
            key,
        )

        # Delete
        self.assertIsNone(f.del_coordinate_reference("qwerty", default=None))

        self.assertEqual(len(f.coordinate_references()), 2)
        self.assertEqual(len(f.domain_ancillaries()), 3)
        c = f.coordinate_reference(
            "standard_name:atmosphere_hybrid_height_coordinate"
        )
        cr = f.del_coordinate_reference(
            "standard_name:atmosphere_hybrid_height_coordinate"
        )
        self.assertTrue(cr.equals(c, verbose=2))
        self.assertEqual(len(f.coordinate_references()), 1)
        self.assertEqual(len(f.domain_ancillaries()), 0)

        f.del_coordinate_reference(
            "grid_mapping_name:rotated_latitude_longitude"
        )
        self.assertEqual(len(f.coordinate_references()), 0)

        # Set
        f = self.f.copy()
        g = self.f.copy()

        f.del_construct("coordinatereference0")
        f.del_construct("coordinatereference1")

        cr = g.coordinate_reference(
            "grid_mapping_name:rotated_latitude_longitude"
        )
        f.set_coordinate_reference(cr, parent=g)
        self.assertEqual(len(f.coordinate_references()), 1)

        cr = g.coordinate_reference(
            "standard_name:atmosphere_hybrid_height_coordinate"
        )
        cr = cr.copy()
        cr.coordinate_conversion.set_domain_ancillary(
            "foo", "domainancillary99"
        )
        f.set_coordinate_reference(cr, parent=g)
        self.assertEqual(len(f.coordinate_references()), 2)
        self.assertEqual(len(f.domain_ancillaries()), 3)

        f.del_construct("coordinatereference0")
        f.del_construct("coordinatereference1")

        cr = g.coordinate_reference(
            "grid_mapping_name:rotated_latitude_longitude"
        )
        f.set_coordinate_reference(cr)
        self.assertEqual(len(f.coordinate_references()), 1)

        with self.assertRaises(ValueError):
            f.ref("long_name:qweRty")

    def test_Field_dimension_coordinate(self):
        f = self.f

        for identity in (
            "grid_latitude",
            "X",
            "dimensioncoordinate1",
        ):
            if identity == "X":
                key, c = f.construct("grid_longitude", item=True)
            else:
                key, c = f.construct(identity, item=True)

            self.assertTrue(
                f.dimension_coordinate(identity).equals(c, verbose=2)
            )
            self.assertEqual(f.dimension_coordinate(identity, key=True), key)

            k, v = f.dimension_coordinate(identity, item=True)
            self.assertEqual(k, key)
            self.assertTrue(v.equals(c))

        self.assertIsNone(
            f.dimension_coordinate("long_name=qwerty:asd", default=None)
        )
        self.assertEqual(
            len(f.dimension_coordinates("long_name=qwerty:asd")), 0
        )

        with self.assertRaises(ValueError):
            f.dim("long_name:qwerty")

    def test_Field_cell_measure(self):
        f = self.f

        for identity in ("measure:area", "cellmeasure0"):
            key, c = f.construct_item(identity)

            self.assertTrue(f.cell_measure(identity).equals(c, verbose=2))
            self.assertEqual(f.cell_measure(identity, key=True), key)

            self.assertTrue(f.cell_measure(identity).equals(c, verbose=2))
            self.assertEqual(f.cell_measure(identity, key=True), key)

        self.assertEqual(len(f.cell_measures()), 1)
        self.assertEqual(len(f.cell_measures("measure:area")), 1)
        self.assertEqual(len(f.cell_measures(*["measure:area"])), 1)

        self.assertIsNone(f.cell_measure("long_name=qwerty:asd", default=None))
        self.assertEqual(len(f.cell_measures("long_name=qwerty:asd")), 0)

        with self.assertRaises(ValueError):
            f.measure("long_name:qwerty")

    def test_Field_cell_method(self):
        f = self.f

        for identity in ("method:mean", "cellmethod0"):
            key, c = f.construct_item(identity)
            self.assertTrue(f.cell_method(identity).equals(c, verbose=2))
            self.assertEqual(f.cell_method(identity, key=True), key)

    def test_Field_domain_ancillary(self):
        f = self.f

        for identity in ("surface_altitude", "domainancillary0"):
            key, c = f.construct_item(identity)
            self.assertTrue(f.domain_ancillary(identity).equals(c, verbose=2))
            self.assertEqual(f.domain_ancillary(identity, key=True), key)

        with self.assertRaises(ValueError):
            f.domain_anc("long_name:qweRty")

    def test_Field_field_ancillary(self):
        f = self.f

        for identity in ("ancillary0", "fieldancillary0"):
            key, c = f.construct_item(identity)
            self.assertTrue(f.field_ancillary(identity).equals(c, verbose=2))
            self.assertEqual(f.field_ancillary(identity, key=True), key)

        with self.assertRaises(ValueError):
            f.field_anc("long_name:qweRty")

    def test_Field_transpose(self):
        f = self.f.copy()
        f0 = f.copy()

        # Null transpose
        g = f.transpose([0, 1, 2])

        self.assertTrue(f0.equals(g, verbose=2))
        self.assertIsNone(f.transpose([0, 1, 2], inplace=True))
        self.assertTrue(f0.equals(f))

        f = self.f.copy()
        h = f.transpose((1, 2, 0))
        h0 = h.transpose((re.compile("^atmos"), "grid_latitude", "X"))
        h.transpose((2, 0, 1), inplace=True)

        h.transpose(
            ("grid_longitude", re.compile("^atmos"), "grid_latitude"),
            inplace=True,
        )
        h.varray
        h.transpose(
            (re.compile("^atmos"), "grid_latitude", "grid_longitude"),
            inplace=True,
        )

        self.assertTrue(h.equals(h0, verbose=2))
        self.assertTrue((h.array == f.array).all())

        with self.assertRaises(Exception):
            f.transpose("qwerty")

        with self.assertRaises(Exception):
            f.transpose([2, 1])

    def test_Field_domain_axis(self):
        self.f.domain_axis(1)
        self.f.domain_axis("domainaxis2")

        with self.assertRaises(ValueError):
            self.f.domain_axis(99)

        with self.assertRaises(ValueError):
            self.f.axis("qwerty")

    def test_Field_where(self):
        f = self.f.copy()
        f0 = f.copy()

        landfrac = f.squeeze()
        landfrac[0:2] = cf.masked
        g = f.where(landfrac >= 54, cf.masked)
        self.assertTrue(g.data.count() == 9 * 6, g.data.count())

        self.assertTrue(f.equals(f.where(None), verbose=2))
        self.assertIsNone(f.where(None, inplace=True))
        self.assertTrue(f.equals(f0, verbose=2))

        g = f.where(cf.wi(25, 31), -99, 11, construct="grid_longitude")
        g = f.where(cf.wi(25, 31), f * 9, f * -7, construct="grid_longitude")
        g = f.where(
            cf.wi(25, 31), f.copy(), f.squeeze(), construct="grid_longitude"
        )

        g = f.where(cf.wi(-25, 31), -99, 11, construct="latitude")
        g = f.where(cf.wi(-25, 31), f * 9, f * -7, construct="latitude")
        g = f.where(
            cf.wi(-25, 31), f.squeeze(), f.copy(), construct="latitude"
        )

        for condition in (True, 1, [[[True]]], [[[[[456]]]]]):
            g = f.where(condition, -9)
            self.assertTrue(g[0].minimum() == -9, str(condition))
            self.assertTrue(g[0].maximum() == -9, str(condition))

        g = f.where(cf.le(34), 34)
        self.assertTrue(g[0].minimum() == 34)
        self.assertTrue(g[0].maximum() == 89)

        g = f.where(cf.le(34), cf.masked)
        self.assertTrue(g[0].minimum() == 35)
        self.assertTrue(g[0].maximum() == 89)

        self.assertIsNone(f.where(cf.le(34), cf.masked, 45, inplace=True))
        self.assertTrue(f[0].minimum() == 45)
        self.assertTrue(f[0].maximum() == 45)

    def test_Field_mask_invalid(self):
        f = self.f.copy()
        self.assertIsNone(f.mask_invalid(inplace=True))

    def test_Field_del_domain_axis(self):
        f = cf.example_field(0)

        g = f[0]
        self.assertIsInstance(
            g.del_domain_axis("Y", squeeze=True), cf.DomainAxis
        )
        self.assertIsInstance(
            g.del_domain_axis("T", squeeze=True), cf.DomainAxis
        )

        g = f.copy()
        self.assertIsInstance(
            g.del_domain_axis("T", squeeze=True), cf.DomainAxis
        )

        with self.assertRaises(Exception):
            f.del_domain_axis("T")

        with self.assertRaises(Exception):
            f.del_domain_axis("X")

        g = f[0]
        with self.assertRaises(Exception):
            g.del_domain_axis("Y")

        g = f[0]
        with self.assertRaises(Exception):
            f.del_domain_axis("T")

    def test_Field_percentile(self):
        f = cf.example_field(1)
        for chunksize in self.chunk_sizes:
            cf.chunksize(chunksize)
            # Percentiles taken across *all axes*
            ranks = [[30, 60, 90], [20], 80]  # include valid singular form

            for rank in ranks:
                # Note: in cf the default is squeeze=False, but numpy has an
                # inverse parameter called keepdims which is by default False
                # also, one must be set to the non-default for equivalents.
                # So first cases (n1, n1) are both squeezed, (n2, n2) are not:
                a1 = numpy.percentile(f, rank)  # has keepdims=False default
                b1 = f.percentile(rank, squeeze=True)
                self.assertTrue(b1.allclose(a1, rtol=1e-05, atol=1e-08))
                a2 = numpy.percentile(f, rank, keepdims=True)
                b2 = f.percentile(rank)  # has squeeze=False default
                self.assertTrue(b2.shape, a2.shape)
                self.assertTrue(b2.allclose(a2, rtol=1e-05, atol=1e-08))

        # TODO: add loop to check get same shape and close enough data
        # for every possible axis combo (see also test_Data_percentile).


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
