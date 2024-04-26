import atexit
import datetime
import faulthandler
import itertools
import os
import re
import tempfile
import unittest

import numpy
import numpy as np
from scipy.ndimage import convolve1d

faulthandler.enable()  # to debug seg faults and timeouts

import cf

n_tmpfiles = 1
tmpfiles = [
    tempfile.mkstemp("_test_Field.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
[tmpfile] = tmpfiles


def _remove_tmpfiles():
    """Try to remove defined temporary files by deleting their paths."""
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
        os.path.dirname(os.path.abspath(__file__)), "regrid.nc"
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
    ugrid_global = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ugrid_global_1.nc",
    )

    chunk_sizes = (100000, 300, 34, 17)
    original_chunksize = cf.chunksize()
    atol = cf.atol()
    rtol = cf.rtol()

    f = cf.read(filename)[0]

    f0 = cf.example_field(0)
    f1 = cf.example_field(1)

    def test_Field_creation_commands(self):
        for f in cf.example_fields():
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
                "min",
                "max",
                "minimum_absolute_value",
                "maximum_absolute_value",
                "mid_range",
                "range",
                "sample_size",
                "sum_of_squares",
                "median",
            ):
                for weights in (None, "area"):
                    a = f.collapse(method, axes=axes, weights=weights).data
                    b = getattr(f.data, method)(axes=axes)
                    self.assertTrue(
                        a.equals(b, rtol=1e-05, atol=1e-08, verbose=2),
                    )

            for method in (
                "sum",
                "mean",
                "mean_absolute_value",
                "mean_of_upper_decile",
                "root_mean_square",
                "sum_of_weights",
                "sum_of_weights2",
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
                    )

        # Test the remove_vertical_crs keyword
        f = cf.example_field(1)
        self.assertTrue(
            f.has_construct(
                "standard_name:atmosphere_hybrid_height_coordinate"
            )
        )
        self.assertEqual(len(f.coordinate_references()), 2)
        self.assertEqual(len(f.domain_ancillaries()), 3)

        u = f.collapse("X: mean")
        self.assertFalse(
            u.has_construct(
                "standard_name:atmosphere_hybrid_height_coordinate"
            )
        )
        self.assertEqual(len(u.coordinate_references()), 1)
        self.assertEqual(len(u.domain_ancillaries()), 0)

        u = f.collapse("X: mean", remove_vertical_crs=False)
        self.assertTrue(
            u.has_construct(
                "standard_name:atmosphere_hybrid_height_coordinate"
            )
        )
        self.assertEqual(len(u.coordinate_references()), 2)
        self.assertEqual(len(u.domain_ancillaries()), 2)

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

        # Test list indices that have a `to_dask_array` method
        y = f.dimension_coordinate("Y")
        self.assertEqual(f[y > 3].shape, (6, 9))

        # Indices result in a subspaced shape that has a size 0 axis
        with self.assertRaises(IndexError):
            f[..., [False] * f.shape[-1]]

        # Test with cyclic subspace
        f.cyclic("grid_longitude")
        g = f[:, -3:-5:1]
        self.assertEqual(g.shape, (10, 7))
        self.assertTrue(np.allclose(f[:, -3:].array, g[:, :3].array))
        self.assertTrue(f[:, :4].equals(g[:, 3:]))

        # Test setting of axis cyclicity
        f.cyclic("grid_longitude", iscyclic=True)
        self.assertEqual(f.data.cyclic(), {1})
        g = f[0, :]
        self.assertEqual(g.data.cyclic(), {1})
        g = f[:, 0]
        self.assertEqual(g.data.cyclic(), set())

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

        # Test list indices that have a `to_dask_array` method
        y = f.dimension_coordinate("Y")
        f[y > 3] = -314
        self.assertEqual(f.where(cf.ne(-314), cf.masked).count(), 6 * 9)

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

        with self.assertRaises(TypeError):
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
        f = cf.example_field(0)

        g = f.copy()
        h = g.cumsum(1)
        self.assertIsNone(g.cumsum(1, inplace=True))
        self.assertTrue(g.equals(h, verbose=2))

        # Check that a new cell method that has been added
        cell_methods = h.cell_methods(todict=True)
        self.assertEqual(len(cell_methods), len(f.cell_methods()) + 1)
        _, cm = cell_methods.popitem()
        self.assertEqual(cm.method, "sum")
        self.assertEqual(cm.axes, (h.get_data_axes()[1],))

        # Check increasing dimension coordinate bounds
        fx = f.dimension_coordinate("X")
        hx = h.dimension_coordinate("X")
        self.assertTrue((hx.lower_bounds == fx.lower_bounds[0]).all())
        self.assertTrue((hx.upper_bounds == fx.upper_bounds).all())

        # Check decreasing dimension coordinate bounds
        g = f.flip("X")
        h = g.cumsum("X")
        gx = g.dimension_coordinate("X")
        hx = h.dimension_coordinate("X")
        self.assertTrue((hx.upper_bounds == gx.upper_bounds[0]).all())
        self.assertTrue((hx.lower_bounds == gx.lower_bounds).all())

        a = f.array
        for axis in range(f.ndim):
            b = np.cumsum(a, axis=axis)
            self.assertTrue((f.cumsum(axis=axis).array == b).all())

        f[0, 3] = cf.masked
        f[2, 7] = cf.masked

        a = f.array
        for axis in range(f.ndim):
            b = np.cumsum(a, axis=axis)
            g = f.cumsum(axis=axis)
            self.assertTrue(cf.functions._numpy_allclose(g.array, b))

    def test_Field_flip(self):
        f = self.f.copy()

        kwargs = {
            axis: slice(None, None, -1) for axis in f.domain_axes(todict=True)
        }
        g = f.subspace(**kwargs)

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
        f = self.f.copy()

        dimarray = f.dimension_coordinate("grid_longitude").array

        f.cyclic("grid_longitude", period=45)
        f.anchor("grid_longitude", 32, dry_run=True)

        g = f.subspace(grid_longitude=[0])

        self.assertIsInstance(g.anchor("grid_longitude", 32), cf.Field)
        self.assertIsNone(g.anchor("grid_longitude", 32, inplace=True))
        self.assertIsInstance(
            g.anchor("grid_longitude", 32, dry_run=True), dict
        )
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
                    f"DECREASING period={period}, x0={x0}, anchor={anchor}, "
                    f"x1={x1}",
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

        ca = f.cell_area(return_cell_measure=True)
        self.assertIsInstance(ca, cf.CellMeasure)
        self.assertEqual(ca.get_measure(), "area")

        m = f.cell_area(methods=True)
        self.assertIsInstance(m, dict)

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

        # Test 'axes' parameter
        g = cf.Field()
        a = g.set_construct(cf.DomainAxis(9))
        b = g.set_construct(cf.DomainAxis(10))
        with self.assertRaises(Exception):
            g.set_data(cf.Data(list(range(9))), axes=a)
        with self.assertRaises(Exception):
            g.set_data(cf.Data(list(range(9))), axes=b)
        with self.assertRaises(Exception):
            g.set_data(cf.Data(list(range(9))), axes=[b, a])

        f = cf.example_field(0)
        f.set_data(f.data, axes=["Y", "X"])

        with self.assertRaises(ValueError):
            f.set_data(f.data.transpose(), axes=["Y", "X"])

        with self.assertRaises(ValueError):
            f.set_data(f.data, axes=["Y"])

        with self.assertRaises(ValueError):
            f.set_data(f.data[0], axes=["Y", "X"])

        with self.assertRaises(ValueError):
            f.set_data(f.data, axes=["T", "X"])

        with self.assertRaises(ValueError):
            f.set_data(f.data[0], axes=["T", "X"])

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

        self.assertEqual(g.cell_measure().ndim, 2)
        h = g.insert_dimension(None, constructs=True)
        self.assertEqual(h.cell_measure().ndim, 3)

        with self.assertRaises(ValueError):
            f.insert_dimension(1, "qwerty")

    def test_Field_indices(self):
        f = cf.read(self.filename)[0]

        array = np.ma.array(f.array, copy=False)

        x = f.dimension_coordinate("X")
        x[...] = np.arange(0, 360, 40)
        x.set_bounds(x.create_bounds())
        f.cyclic("X", iscyclic=True, period=360)

        f0 = f.copy()

        # wi (increasing)
        indices = f.indices(grid_longitude=cf.wi(50, 130))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 2))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [80, 120]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 50))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-80, -40, 0, 40]).all())

        indices = f.indices(grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(310 - 1080, 450 - 1080))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(310 + 720, 450 + 720))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 370))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9))
        x = g.dimension_coordinate("X").array
        self.assertTrue(
            (x == [-80, -40, 0, 40, 80, 120, 160, 200, 240.0]).all()
        )

        with self.assertRaises(ValueError):
            # No X coordinate values lie inside the range [90, 100]
            f.indices(grid_longitude=cf.wi(90, 100))

        indices = f.indices("full", grid_longitude=cf.wi(310, 450))
        self.assertTrue(indices[0], "mask")

        mask = indices[1]
        self.assertEqual(len(mask), 1)

        mask = mask[0]
        self.assertEqual(mask.shape, (1, 1, 9))
        self.assertTrue(
            (
                np.array(mask)
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
        self.assertEqual(g.shape, (1, 10, 9))

        x = g.dimension_coordinate("X").array
        self.assertEqual(x.shape, (9,))

        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320]).all(), x
        )

        a = array.copy()
        a[..., 3:8] = np.ma.masked

        self.assertTrue(cf.functions._numpy_allclose(g.array, a), g.array)

        indices = f.indices("full", grid_longitude=cf.wi(70, 200))
        self.assertTrue(indices[0], "mask")
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9))
        x = g.dimension_coordinate("X").array
        self.assertEqual(x.shape, (9,))
        self.assertTrue((x == [0, 40, 80, 120, 160, 200, 240, 280, 320]).all())
        a = array.copy()
        a[..., [0, 1, 6, 7, 8]] = np.ma.masked
        self.assertTrue(cf.functions._numpy_allclose(g.array, a), g.array)

        # wi (decreasing)
        f.flip("X", inplace=True)

        indices = f.indices(grid_longitude=cf.wi(50, 130))
        self.assertTrue(indices[0], "mask")
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 2))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [80, 120][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 50))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-80, -40, 0, 40][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310 - 1080, 450 - 1080))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310 + 720, 450 + 720))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        with self.assertRaises(ValueError):
            # No X coordinate values lie inside the range [90, 100]
            f.indices(grid_longitude=cf.wi(90, 100))

        indices = f.indices("full", grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9))
        x = g.dimension_coordinate("X").array
        self.assertEqual(x.shape, (9,))
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320][::-1]).all()
        )

        indices = f.indices("full", grid_longitude=cf.wi(70, 200))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9))
        x = g.dimension_coordinate("X").array
        self.assertEqual(x.shape, (9,))
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320][::-1]).all()
        )

        # wo
        f = f0.copy()

        indices = f.indices(grid_longitude=cf.wo(50, 130))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 7))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-200, -160, -120, -80, -40, 0, 40]).all())

        with self.assertRaises(ValueError):
            # No X coordinate values lie outside the range [-90, 370]
            f.indices(grid_longitude=cf.wo(-90, 370))

        # set
        indices = f.indices(grid_longitude=cf.set([320, 40, 80, 99999]))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 3))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [40, 80, 320]).all())

        indices = f.indices(grid_longitude=cf.lt(90))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 3))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.gt(90))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 6))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [120, 160, 200, 240, 280, 320]).all())

        indices = f.indices(grid_longitude=cf.le(80))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 3))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.ge(80))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 7))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [80, 120, 160, 200, 240, 280, 320]).all())

        # 2-d
        lon = f.construct("longitude").array
        lon = np.transpose(lon)
        lon = np.expand_dims(lon, 0)

        lat = f.construct("latitude").array
        lat = np.expand_dims(lat, 0)

        array = np.ma.where((lon >= 92) & (lon <= 134), f.array, np.ma.masked)

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

            self.assertEqual(g.shape, shape)
            self.assertTrue(
                cf.functions._numpy_allclose(array2, g.array), g.array
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

            self.assertEqual(g.shape, shape)

        indices = f.indices(
            "full",
            longitude=cf.wi(92, 134),
            latitude=cf.wi(-26, -20) | cf.ge(30),
        )
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9))
        array = np.ma.where(
            (
                ((lon >= 92) & (lon <= 134))
                & (((lat >= -26) & (lat <= -20)) | (lat >= 30))
            ),
            f.array,
            np.ma.masked,
        )
        self.assertTrue(cf.functions._numpy_allclose(array, g.array), g.array)

        for mode in ("compress", "full", "envelope"):
            indices = f.indices(mode, grid_longitude=cf.contains(23.2))
            g = f[indices]
            if mode == "full":
                shape = f.shape
            else:
                shape = (1, 10, 1)

            self.assertEqual(g.shape, shape)

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

            self.assertEqual(g.shape, shape)

            if mode != "full":
                self.assertEqual(g.construct("grid_latitude").array, 3)

        for mode in ("compress", "full", "envelope"):
            indices = f.indices(mode, longitude=cf.contains(83))
            g = f[indices]
            if mode == "full":
                shape = f.shape
            else:
                shape = (1, 1, 1)

            self.assertEqual(g.shape, shape)
            self.assertEqual(g.array.compressed(), 29)
            if mode != "full":
                self.assertEqual(g.construct("longitude").array, 83)

        for mode in ("compress", "full", "envelope"):
            indices = f.indices(
                mode, longitude=cf.contains(83) | cf.contains(100)
            )

            g = f[indices]
            if mode == "full":
                shape = f.shape
            elif mode == "envelope":
                shape = (1, 4, 3)
            else:
                shape = (1, 2, 2)

            self.assertEqual(g.shape, shape)
            self.assertTrue((g.array.compressed() == [4, 29]).all())

        # Add 2-d auxiliary coordinates with bounds, so we can
        # properly test cf.contains values
        x = f.coord("grid_longitude")
        y = f.coord("grid_latitude")
        y.set_bounds(y.create_bounds())

        x_bounds = x.bounds.array
        y_bounds = y.bounds.array

        lat = np.empty((y.size, x.size))
        lat[...] = y.array.reshape(y.size, 1)
        lon = np.empty((y.size, x.size))
        lon[...] = x.array

        lon_bounds = np.empty(lon.shape + (4,))
        lon_bounds[..., [0, 3]] = x_bounds[:, 0].reshape(1, x.size, 1)
        lon_bounds[..., [1, 2]] = x_bounds[:, 1].reshape(1, x.size, 1)

        lat_bounds = np.empty(lat.shape + (4,))
        lat_bounds[..., [0, 1]] = y_bounds[:, 0].reshape(y.size, 1, 1)
        lat_bounds[..., [2, 3]] = y_bounds[:, 1].reshape(y.size, 1, 1)

        lon_2d_coord = cf.AuxiliaryCoordinate(
            data=cf.Data(lon, units=x.Units), bounds=cf.Bounds(data=lon_bounds)
        )
        lat_2d_coord = cf.AuxiliaryCoordinate(
            data=cf.Data(lat, units=y.Units), bounds=cf.Bounds(data=lat_bounds)
        )

        lon_2d_coord.standard_name = "aux_x"
        lat_2d_coord.standard_name = "aux_y"

        axes = (f.domain_axis("Y", key=True), f.domain_axis("X", key=True))

        f.set_construct(lon_2d_coord, axes=axes, copy=False)
        f.set_construct(lat_2d_coord, axes=axes, copy=False)

        for mode in ("compress", "full", "envelope"):
            indices = f.indices(mode, aux_x=cf.contains(160.1))
            g = f[indices]
            if mode == "full":
                shape = f.shape
            else:
                shape = (1, 10, 1)

            self.assertEqual(g.shape, shape)

            if mode != "full":
                self.assertTrue((g.construct("aux_x").array == 160).all())

        for mode in ("compress", "full", "envelope"):
            indices = f.indices(mode, aux_x=cf.contains(160.1), aux_y=3)
            g = f[indices]
            if mode == "full":
                shape = f.shape
            else:
                shape = (1, 1, 1)

            self.assertEqual(g.shape, shape)

            if mode != "full":
                self.assertEqual(g.construct("aux_x").array, 160)
                self.assertEqual(g.construct("aux_y").array, 3)

        for mode in ("compress", "full", "envelope"):
            indices = f.indices(
                mode, aux_x=cf.contains(160.1), aux_y=cf.contains(3.1)
            )
            g = f[indices]
            if mode == "full":
                shape = f.shape
            else:
                shape = (1, 1, 1)

            self.assertEqual(g.shape, shape)

            if mode != "full":
                self.assertEqual(g.construct("aux_x").array, 160)
                self.assertEqual(g.construct("aux_y").array, 3)

        # Halos: monotonic increasing sequence
        index = [2, 3, 4, 5]
        indices = f.indices(0, grid_longitude=index)
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [80, 120, 160, 200]).all())

        indices = f.indices(1, grid_longitude=index)
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 6))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [40, 80, 120, 160, 200, 240]).all())

        indices = f.indices(999, grid_longitude=index)
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == f.dimension_coordinate("X").array).all())

        # Halos: non-monotonic sequence
        index = [2, 3, 4, 1]
        indices = f.indices(0, grid_longitude=index)
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [80, 120, 160, 40]).all())

        indices = f.indices(1, grid_longitude=index)
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 6))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [40, 80, 120, 160, 40, 0]).all())

        for halo in (2, 999):
            indices = f.indices(halo, grid_longitude=index)
            g = f[indices]
            self.assertEqual(g.shape, (1, 10, 7))
            x = g.dimension_coordinate("X").array
            self.assertTrue((x == [0, 40, 80, 120, 160, 40, 0]).all())

        # Halos: cyclic slice increasing
        for index in (cf.wi(70, 200), slice(2, 6)):
            indices = f.indices(0, grid_longitude=index)
            g = f[indices]
            self.assertEqual(g.shape, (1, 10, 4))
            x = g.dimension_coordinate("X").array
            self.assertTrue((x == [80, 120, 160, 200]).all())

            indices = f.indices(1, grid_longitude=index)
            g = f[indices]
            self.assertEqual(g.shape, (1, 10, 6))
            x = g.dimension_coordinate("X").array
            self.assertTrue((x == [40, 80, 120, 160, 200, 240]).all())

            indices = f.indices(999, grid_longitude=index)
            g = f[indices]
            self.assertEqual(g.shape, (1, 10, 9))
            x = g.dimension_coordinate("X").array
            self.assertTrue(
                (x == [-120, -80, -40, 0, 40, 80, 120, 160, 200]).all()
            )

        # Halos: cyclic slice increasing
        index = cf.wi(-170, 40)
        indices = f.indices(0, grid_longitude=index)
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 6))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-160, -120, -80, -40, 0, 40]).all())

        indices = f.indices(1, grid_longitude=index)
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 8))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [-200, -160, -120, -80, -40, 0, 40, 80]).all())

        indices = f.indices(2, grid_longitude=index)
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9))
        x = g.dimension_coordinate("X").array
        self.assertTrue(
            (x == [-240, -200, -160, -120, -80, -40, 0, 40, 80]).all()
        )

        # Halos: cyclic slice decreasing
        index = slice(1, -5, -1)
        indices = f.indices(0, grid_longitude=index)
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 6))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [40, 0, -40, -80, -120, -160]).all())

        indices = f.indices(1, grid_longitude=index)
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 8))
        x = g.dimension_coordinate("X").array
        self.assertTrue((x == [80, 40, 0, -40, -80, -120, -160, -200]).all())

        indices = f.indices(2, grid_longitude=index)
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9))
        x = g.dimension_coordinate("X").array
        self.assertTrue(
            (x == [120, 80, 40, 0, -40, -80, -120, -160, -200]).all()
        )

        # Halos: ancillary masking
        index = cf.wi(90, 100)
        indices = f.indices(longitude=index)
        g = f[indices]
        self.assertTrue(np.ma.is_masked(g.array))

        for halo in (0, 1):
            indices = f.indices(halo, longitude=index)
            g = f[indices]
            self.assertFalse(np.ma.is_masked(g.array))

        # Test API with 0/1/2 arguments
        kwargs = {"grid_latitude": [1]}
        i = f.indices(**kwargs)
        j = f.indices(0, **kwargs)
        k = f.indices("compress", 0, **kwargs)
        self.assertEqual(i, j)
        self.assertEqual(i, k)

        # Subspace has size 0 axis resulting from dask array index
        indices = f.indices(grid_latitude=cf.contains(-23.2))
        with self.assertRaises(IndexError):
            f[indices]

        # 'contains' when coords have no bounds, and the 'contains'
        # values do not equal any coordinate values
        with self.assertRaises(ValueError):
            f.indices(latitude=cf.contains(-99999))

        # Multiple constructs with incompatible domain axes
        with self.assertRaises(ValueError):
            f.indices(longitude=cf.gt(23), grid_longitude=cf.wi(92, 134))

        # Multiple constructs with incompatible domain axes
        with self.assertRaises(ValueError):
            f.indices(grid_longitude=cf.gt(23), longitude=cf.wi(92, 134))

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

    def test_Field_match_by_construct(self):
        f = self.f.copy()

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

        # Check match for size 1 axes that are not spanned by the data
        f = cf.example_field(0)
        self.assertTrue(f.match_by_construct(T=cf.dt("2019-01-01")))
        self.assertFalse(f.match_by_construct(T=cf.dt("9876-12-31")))

    def test_Field_autocyclic(self):
        f = self.f.copy()

        self.assertFalse(f.autocyclic())
        f.dimension_coordinate("X").del_bounds()
        f.autocyclic()

    def test_Field_construct_key(self):
        self.f.construct_key("grid_longitude")

    def test_Field_convolution_filter(self):
        f = cf.read(self.filename1)[0]

        window = [0.1, 0.15, 0.5, 0.15, 0.1]

        # Test user weights in different modes
        for mode in ("reflect", "constant", "nearest", "wrap"):
            g = f.convolution_filter(window, axis=-1, mode=mode, cval=0.0)
            a = convolve1d(f.array, window, axis=-1, mode=mode)
            self.assertTrue(np.allclose(g.array, a, atol=1.6e-5, rtol=0))

        # Test coordinate bounds
        f = cf.example_field(0)
        window = [0.5, 1, 2, 1, 0.5]

        g = f.convolution_filter(window, mode="wrap", axis="X")
        gx = g.coord("X").bounds.array
        self.assertTrue((gx[:, 0] == np.linspace(-90, 225, 8)).all())
        self.assertTrue((gx[:, 1] == np.linspace(135, 450, 8)).all())

        g = f[:, ::-1].convolution_filter(window, mode="wrap", axis="X")
        gx = g.coord("X").bounds.array
        self.assertTrue((gx[:, 0] == np.linspace(450, 135, 8)).all())
        self.assertTrue((gx[:, 1] == np.linspace(225, -90, 8)).all())

        g = f.convolution_filter(window, mode="constant", axis="X")
        gx = g.coord("X").bounds.array
        self.assertTrue((gx[:, 0] == [0, 0, 0, 45, 90, 135, 180, 225]).all())
        self.assertTrue(
            (gx[:, 1] == [135, 180, 225, 270, 315, 360, 360, 360]).all()
        )

    def test_Field_moving_window(self):
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
            for mode in ("constant", "wrap", "reflect", "nearest"):
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
            for mode in ("constant", "wrap", "reflect", "nearest"):
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
        f = cf.example_field(0)
        f[...] = np.arange(9)[1:] * 45

        # Check a cyclic periodic axis
        d = f.derivative("X")
        self.assertTrue(np.allclose(d[:, 1:-1].array, 1))
        self.assertTrue(np.allclose(d[:, [0, -1]].array, -3))

        # The reversed field should contain the same gradients in this
        # case
        f1 = f[:, ::-1]
        d1 = f1.derivative("X")
        self.assertTrue(d1.data.equals(d.data))

        # Check non-cyclic
        d = f.derivative("X", wrap=False)
        self.assertTrue(np.allclose(d.array, 1))
        self.assertEqual(d.array.sum(), 30)

        d = f.derivative("X", wrap=False, one_sided_at_boundary=True)
        self.assertTrue(np.allclose(d.array, 1))
        self.assertEqual(d.array.sum(), 40)

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

        # Test some constructs which can never have data
        with self.assertRaises(ValueError):
            f.convert("cellmethod0")
        with self.assertRaises(ValueError):
            f.convert("domainaxis0")

    def test_Field_section(self):
        f = cf.read(self.filename2)[0][0:10]
        g = f.section(("X", "Y"))
        self.assertEqual(len(g), 10, f"len(g)={len(g)}")

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

        # Get
        for identity in (
            "coordinatereference1",
            "key%coordinatereference0",
            "standard_name:atmosphere_hybrid_height_coordinate",
            "grid_mapping_name:rotated_latitude_longitude",
        ):
            key = f.construct_key(identity)
            c = f.construct(identity)

            self.assertTrue(
                f.get_coordinate_reference(identity).equals(c, verbose=2)
            )
            self.assertEqual(
                f.get_coordinate_reference(identity, key=True), key
            )

        with self.assertRaises(ValueError):
            f.get_coordinate_reference()  # since has two CR constructs
        g = f.copy()
        g.del_coordinate_reference("coordinatereference1")
        g.get_coordinate_reference()  # should work here as has only one CR

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

        for identity in ("grid_latitude", "X", "dimensioncoordinate1"):
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
            self.assertTrue((g.array == -9).all())

        g = f.where(cf.le(34), 34)
        self.assertTrue(g.min() == 34)
        self.assertTrue(g.max() == 89)

        g = f.where(cf.le(34), cf.masked)
        self.assertTrue(g.min() == 35)
        self.assertTrue(g.max() == 89)

        self.assertIsNone(f.where(cf.le(34), cf.masked, 45, inplace=True))
        self.assertTrue(f.min() == 45)
        self.assertTrue(f.max() == 45)

    def test_Field_masked_invalid(self):
        f = self.f.copy()
        self.assertIsNone(f.masked_invalid(inplace=True))

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

        # Percentiles taken across *all axes*
        ranks = ([30, 60, 90], [20], 80)  # include valid singular form

        for rank in ranks:
            # Note: Currently in cf the default is squeeze=False, but
            #       numpy has an inverse parameter called keepdims
            #       which is by default False also, one must be set to
            #       the non-default for equivalents.  So first cases
            #       (n1, n1) are both squeezed, (n2, n2) are not:
            a1 = numpy.percentile(f, rank)  # has keepdims=False default
            b1 = f.percentile(rank, squeeze=True)
            self.assertTrue(b1.allclose(a1, rtol=1e-05, atol=1e-08))
            a2 = numpy.percentile(f, rank, keepdims=True)
            b2 = f.percentile(rank)  # has squeeze=False default
            self.assertTrue(b2.shape, a2.shape)
            self.assertTrue(b2.allclose(a2, rtol=1e-05, atol=1e-08))

        # TODO: add loop to check get same shape and close enough data
        # for every possible axis combo (see also test_Data_percentile).

    def test_Field_grad_xy(self):
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

                self.assertTrue(x.Units == y.Units == cf.Units("m-1 rad-1"))

                x0 = f.derivative(
                    "X", wrap=wrap, one_sided_at_boundary=one_sided
                ) / (sin_theta * r)
                y0 = f.derivative("Y", one_sided_at_boundary=one_sided) / r

                # Check the data
                with cf.rtol(1e-10):
                    self.assertTrue((x.data == x0.data).all())
                    self.assertTrue((y.data == y0.data).all())

                # Check that x and y have the same metadata as f
                # (except standard_name, long_name, and units).
                f0 = f.copy()
                del f0.standard_name

                f0.set_data(x.data)
                del x.long_name
                self.assertTrue(x.equals(f0))

                f0.set_data(y.data)
                del y.long_name
                self.assertTrue(y.equals(f0))

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

                self.assertTrue(x.Units == y.Units == cf.Units("m-1"))

                x0 = f.derivative(
                    "X", wrap=wrap, one_sided_at_boundary=one_sided
                )
                y0 = f.derivative("Y", one_sided_at_boundary=one_sided)

                del x.long_name
                del y.long_name
                del x0.long_name
                del y0.long_name
                self.assertTrue(x.equals(x0, rtol=1e-10))
                self.assertTrue(y.equals(y0, rtol=1e-10))

        # Test case when spherical dimension coordinates have units
        # but no standard names
        f = cf.example_field(0)
        del f.dimension_coordinate("X").standard_name
        del f.dimension_coordinate("Y").standard_name
        x, y = f.grad_xy(radius="earth")
        self.assertEqual(x.shape, f.shape)
        self.assertEqual(y.shape, f.shape)
        self.assertEqual(x.dimension_coordinate("Y").standard_name, "latitude")
        self.assertEqual(
            x.dimension_coordinate("X").standard_name, "longitude"
        )
        self.assertEqual(y.dimension_coordinate("Y").standard_name, "latitude")
        self.assertEqual(
            y.dimension_coordinate("X").standard_name, "longitude"
        )

    def test_Field_laplacian_xy(self):
        f = cf.example_field(0)

        # Laplacian(f) = div(grad(f))

        # Spherical polar coordinates
        radius = 2
        for wrap in (False, True, None):
            for one_sided in (True, False):
                lp = f.laplacian_xy(
                    radius=radius, x_wrap=wrap, one_sided_at_boundary=one_sided
                )

                self.assertTrue(lp.Units == cf.Units("m-2 rad-2"))

                lp0 = cf.div_xy(
                    *f.grad_xy(
                        radius=radius,
                        x_wrap=wrap,
                        one_sided_at_boundary=one_sided,
                    ),
                    radius=2,
                    x_wrap=wrap,
                    one_sided_at_boundary=one_sided,
                )

                del lp.long_name
                del lp0.long_name
                self.assertTrue(lp.equals(lp0, rtol=1e-10))

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
                lp = f.laplacian_xy(
                    x_wrap=wrap, one_sided_at_boundary=one_sided
                )

                self.assertTrue(lp.Units == cf.Units("m-2"))

                lp0 = cf.div_xy(
                    *f.grad_xy(x_wrap=wrap, one_sided_at_boundary=one_sided),
                    x_wrap=wrap,
                    one_sided_at_boundary=one_sided,
                )

                del lp.long_name
                del lp0.long_name
                self.assertTrue(lp.equals(lp0, rtol=1e-10))

        # Test case when spherical dimension coordinates have units
        # but no standard names
        f = cf.example_field(0)
        del f.dimension_coordinate("X").standard_name
        del f.dimension_coordinate("Y").standard_name
        g = f.laplacian_xy(radius="earth")
        self.assertEqual(g.shape, f.shape)
        self.assertEqual(g.dimension_coordinate("Y").standard_name, "latitude")
        self.assertEqual(
            g.dimension_coordinate("X").standard_name, "longitude"
        )

    def test_Field_to_dask_array(self):
        f = self.f0.copy()
        self.assertIs(f.to_dask_array(), f.data.to_dask_array())

        f.del_data()
        with self.assertRaises(ValueError):
            f.to_dask_array()

    def test_Field_combine_with_Query(self):
        f = self.f0
        q = cf.lt(0.1)
        (q == f).array
        (f == q).array

        f = cf.example_field(2)

        x = f.coordinate("X")
        q = cf.eq(337.5)
        (q == x).array
        (x == q).array

        t = f.coordinate("T")
        q = cf.dt(1962, 11, 16)
        (q == t).array
        (t == q).array

        q = cf.wi(cf.dt(1960, 9, 1), cf.dt(1962, 11, 16, 12))
        (q == t).array
        (t == q).array

        q = cf.eq(cf.dt(1962, 11, 16, 12))
        (q == t).array
        (t == q).array

    def test_Field_get_original_filenames(self):
        """Test Field.orignal_filenames."""
        f = cf.example_field(0)
        f._original_filenames(define=["file1.nc", "file2.nc"])
        x = f.coordinate("longitude")
        x._original_filenames(define=["file1.nc", "file3.nc"])
        b = x.bounds
        b._original_filenames(define=["file1.nc", "file4.nc"])

        self.assertEqual(
            f.get_original_filenames(),
            set(
                (
                    cf.abspath("file1.nc"),
                    cf.abspath("file2.nc"),
                    cf.abspath("file3.nc"),
                    cf.abspath("file4.nc"),
                )
            ),
        )

        self.assertEqual(
            f.get_original_filenames(), f.copy().get_original_filenames()
        )

    def test_Field_set_construct_conform(self):
        """Test the 'conform' parameter of Field.set_construct."""
        f = cf.example_field(0)
        cm = cf.CellMethod("T", "maximum")
        self.assertEqual(cm.get_axes(), ("T",))

        key = f.set_construct(cm)
        cm2 = f.cell_method("method:maximum")
        taxis = f.domain_axis("T", key=True)
        self.assertEqual(cm2.get_axes(), (taxis,))

        f.del_construct(key)
        f.set_construct(cm, conform=False)
        cm2 = f.cell_method("method:maximum")
        self.assertEqual(cm2.get_axes(), ("T",))

    def test_Field_del_construct(self):
        """Test the `del_construct` Field method."""
        # Test a field without cyclic axes. These are equivalent tests to those
        # in the cfdm test suite, to check the behaviour is the same in cf.
        f = self.f1.copy()

        self.assertIsInstance(
            f.del_construct("auxiliarycoordinate1"), cf.AuxiliaryCoordinate
        )

        with self.assertRaises(ValueError):
            f.del_construct("auxiliarycoordinate1")

        self.assertIsNone(
            f.del_construct("auxiliarycoordinate1", default=None)
        )

        self.assertIsInstance(f.del_construct("measure:area"), cf.CellMeasure)

        # Test a field with cyclic axes, to ensure the cyclic() set is
        # updated accordingly if a cyclic axes is the one removed.
        g = cf.example_field(2)  # this has a cyclic axes 'domainaxis2'
        # To delete a cyclic axes, must first delete this dimension coordinate
        # because 'domainaxis2' spans it.
        self.assertIsInstance(
            g.del_construct("dimensioncoordinate2"), cf.DimensionCoordinate
        )
        self.assertEqual(g.cyclic(), set(("domainaxis2",)))
        self.assertIsInstance(g.del_construct("domainaxis2"), cf.DomainAxis)
        self.assertEqual(g.cyclic(), set())

    def test_Field_persist(self):
        """Test the `persist` Field method."""
        f = cf.example_field(0)
        f *= 2

        self.assertGreater(len(f.to_dask_array().dask.layers), 1)

        g = f.persist()
        self.assertIsInstance(g, cf.Field)
        self.assertEqual(len(g.to_dask_array().dask.layers), 1)
        self.assertTrue(g.equals(f))

        self.assertIsNone(g.persist(inplace=True))

    def test_Field_argmax(self):
        """Test the `argmax` Field method."""
        f = cf.example_field(2)
        i = f.argmax("T")
        self.assertEqual(i.shape, f.shape[1:])

        i = f.argmax(unravel=True)
        self.assertIsInstance(i, tuple)
        g = f[i]
        self.assertEqual(g.shape, (1, 1, 1))

        # Bad axis
        with self.assertRaises(ValueError):
            f.argmax(axis="foo")

    def test_Field_argmin(self):
        """Test the `argmin` Field method."""
        f = cf.example_field(2)
        i = f.argmin("T")
        self.assertEqual(i.shape, f.shape[1:])

        i = f.argmin(unravel=True)
        self.assertIsInstance(i, tuple)
        g = f[i]
        self.assertEqual(g.shape, (1, 1, 1))

        # Bad axis
        with self.assertRaises(ValueError):
            f.argmin(axis="foo")

    def test_Field_subspace(self):
        f = self.f

        g = f.subspace(grid_longitude=20)
        h = f.subspace(grid_longitude=np.float64(20))
        self.assertTrue(g.equals(h))

        # Test API with 0/1/2 arguments
        kwargs = {"grid_latitude": [1]}
        i = f.subspace(**kwargs)
        j = f.subspace(0, **kwargs)
        k = f.subspace("compress", 0, **kwargs)
        self.assertEqual(i, j)
        self.assertEqual(i, k)

    def test_Field_auxiliary_to_dimension_to_auxiliary(self):
        f = cf.example_field(0)
        nd = len(f.dimension_coordinates())

        g = f.dimension_to_auxiliary("latitude")
        self.assertEqual(len(g.dimension_coordinates()), nd - 1)
        self.assertEqual(len(g.auxiliary_coordinates()), 1)

        h = g.auxiliary_to_dimension("latitude")
        self.assertEqual(len(h.dimension_coordinates()), nd)
        self.assertEqual(len(h.auxiliary_coordinates()), 0)
        self.assertTrue(h.equals(f))
        self.assertIsNone(f.dimension_to_auxiliary("Y", inplace=True))
        self.assertIsNone(g.auxiliary_to_dimension("Y", inplace=True))

        f = cf.read("geometry_1.nc")[0]

        with self.assertRaises(ValueError):
            f.auxiliary_to_dimension("latitude")

    def test_Field_subspace_ugrid(self):
        f = cf.read(self.ugrid_global)[0]

        with self.assertRaises(ValueError):
            # Can't specify 2 conditions for 1 axis
            g = f.subspace(X=cf.wi(40, 70), Y=cf.wi(-20, 30))

        g = f.subspace(X=cf.wi(40, 70))
        g = g.subspace(Y=cf.wi(-20, 30))
        self.assertTrue(g.aux("X").data.range() < 30)
        self.assertTrue(g.aux("Y").data.range() < 50)

    def test_Field_file_location(self):
        f = cf.example_field(0)

        self.assertEqual(f.add_file_location("/data/model/"), "/data/model")

        cf.write(f, tmpfile)
        f = cf.read(tmpfile)[0]
        g = f.copy()
        location = os.path.dirname(os.path.abspath(tmpfile))

        self.assertEqual(f.file_locations(), set((location,)))
        self.assertEqual(f.add_file_location("/data/model/"), "/data/model")
        self.assertEqual(f.file_locations(), set((location, "/data/model")))

        # Check that we haven't changed 'g'
        self.assertEqual(g.file_locations(), set((location,)))

        self.assertEqual(f.del_file_location("/data/model/"), "/data/model")
        self.assertEqual(f.file_locations(), set((location,)))
        f.del_file_location("/invalid")
        self.assertEqual(f.file_locations(), set((location,)))

    def test_Field_pad_missing(self):
        """Test Field.pad_missing."""
        f = cf.example_field(0)

        g = f.pad_missing("X", to_size=10)
        self.assertEqual(g.shape, (5, 10))
        self.assertTrue(g[:, 8:].mask.all())

        self.assertIsNone(f.pad_missing("X", pad_width=(1, 2), inplace=True))
        self.assertEqual(f.shape, (5, 11))
        self.assertTrue(f[:, 0].mask.all())
        self.assertTrue(f[:, 9:].mask.all())

        g = f.pad_missing("Y", pad_width=(0, 1))
        self.assertEqual(g.shape, (6, 11))
        self.assertTrue(g[5, :].mask.all())

    def test_Field_cyclic_iscyclic(self):
        """Test the `cyclic` and `iscyclic` Field methods."""
        f1 = cf.example_field(1)  # no cyclic axes
        f2 = cf.example_field(2)  # one cyclic axis, 'domainaxis2' ('X')

        # Getting
        self.assertEqual(f1.cyclic(), set())
        self.assertFalse(f1.iscyclic("X"))
        self.assertFalse(f1.iscyclic("Y"))
        self.assertFalse(f1.iscyclic("Z"))
        self.assertFalse(f1.iscyclic("T"))
        self.assertEqual(f2.cyclic(), set(("domainaxis2",)))
        self.assertTrue(f2.iscyclic("X"))
        self.assertFalse(f2.iscyclic("Y"))
        self.assertFalse(f2.iscyclic("Z"))
        self.assertFalse(f2.iscyclic("T"))

        # Setting
        self.assertEqual(f2.cyclic("X", iscyclic=False), set(("domainaxis2",)))
        self.assertEqual(f2.cyclic(), set())
        self.assertEqual(f2.cyclic("X", period=360), set())
        self.assertEqual(f2.cyclic(), set(("domainaxis2",)))
        self.assertTrue(f2.iscyclic("X"))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
