import atexit
import datetime
import inspect
import itertools
import os
import re
import tempfile
import unittest

import numpy

from scipy.ndimage import convolve1d

import cf

tmpfile = tempfile.mktemp('.cfdm_test')
tmpfiles = [tmpfile]


def _remove_tmpfiles():
    '''
    '''
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


def axes_combinations(f):
    return [axes
            for n in range(1, f.ndim+1)
            for axes in itertools.permutations(range(f.ndim), n)]


class FieldTest(unittest.TestCase):
    def setUp(self):
        self.filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'test_file.nc')
        self.filename2 = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'test_file2.nc')
        self.contiguous = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'DSG_timeSeries_contiguous.nc'
        )
        self.indexed = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'DSG_timeSeries_indexed.nc'
        )
        self.indexed_contiguous = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'DSG_timeSeriesProfile_indexed_contiguous.nc'
        )

        self.filename1 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'regrid_file1.nc')

        self.chunk_sizes = (100000, 300, 34, 17)
        self.original_chunksize = cf.CHUNKSIZE()
        self.atol = cf.ATOL()
        self.rtol = cf.RTOL()
        self.f = cf.read(self.filename, verbose=0)[0]

        self.test_only = []
#        self.test_only = ['NOTHING!!!!']
#        self.test_only = ['test_Field_get_filenames']
#        self.test_only = [
#            'test_Field_convolution_filter', 'test_Field_derivative',
#            'test_Field_moving_window'
#        ]
#        self.test_only = ['test_Field_weights']
#        self.test_only = ['test_Field_collapse']
#        self.test_only = ['test_Field_radius']
#        self.test_only = ['test_Field_field_ancillary']
#        self.test_only = ['test_Field_AUXILIARY_MASK']
#        self.test_only = ['test_Field__getitem__']
#        self.test_only = ['test_Field_dimension_coordinate']
#        self.test_only = ['test_Field_insert_dimension']
#        self.test_only = ['test_Field_match']
#        self.test_only = ['test_Field_where']
#        self.test_only = ['test_Field_autocyclic']
#        self.test_only = ['test_Field_anchor']
#        self.test_only = ['test_Field_mask_invalid']
#        self.test_only = ['test_Field_test_Field_creation_commands']
#        self.test_only = ['test_Field_section']
#        self.test_only = ['test_Field_flip']
#        self.test_only = ['test_Field_Field_domain_mask']
#        self.test_only = ['test_Field_compress_uncompress']
#        self.test_only = ['test_Field_coordinate_reference']

    def test_Field_creation_commands(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        for rd in (False, True):
            for indent in (0, 4):
                for s in (False, True):
                    for ns in ('cf', ''):
                        _ = f.creation_commands(representative_data=rd,
                                                indent=indent,
                                                namespace=ns,
                                                string=s)
        # --- End: for

        for i in range(7):
            f = cf.example_field(i)
            for rd in (False, True):
                for indent in (0, 4):
                    for s in (False, True):
                        for ns in ('cf', ''):
                            _ = f.creation_commands(representative_data=rd,
                                                    indent=indent,
                                                    namespace=ns,
                                                    string=s)
        # --- End: for

    def test_Field_get_filenames(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.example_field(0)

        tmpfile = 'cfdm_test_Field_get_filenames.nc'
        tmpfiles.append(tmpfile)

        cf.write(f, tmpfile)
        g = cf.read(tmpfile)[0]

        abspath_tmpfile = os.path.abspath(tmpfile)
        self.assertTrue(g.get_filenames() == set([abspath_tmpfile]),
                        g.get_filenames())

        g.data[...] = -99
        self.assertTrue(g.get_filenames() == set([abspath_tmpfile]),
                        g.get_filenames())

        for c in g.constructs.filter_by_data().values():
            c.data[...] = -99

        self.assertTrue(g.get_filenames() == set([abspath_tmpfile]),
                        g.get_filenames())

        for c in g.constructs.filter_by_data().values():
            if c.has_bounds():
                c.bounds.data[...] = -99
        # --- End: for

        self.assertTrue(g.get_filenames() == set(),
                        g.get_filenames())

    def test_Field_halo(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.example_field(7)

        g = f.copy()
        self.assertIsNone(g.halo(1, inplace=True))

        i = 1
        g = f.halo(i)
        self.assertTrue(
            (numpy.array(g.shape) == numpy.array(f.shape) + i*2).all())

        for key, c in g.constructs.filter_by_data().items():
            d = f.construct(key)
            self.assertTrue(
                (numpy.array(c.shape) == numpy.array(d.shape) + i*2).all())
        # --- End: for

    def test_Field_has_construct(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.example_field(1)

        self.assertTrue(f.has_construct('T'))
        self.assertTrue(f.has_construct('long_name=Grid latitude name'))
        self.assertTrue(f.has_construct('ncvar%a'))
        self.assertTrue(f.has_construct('measure:area'))
        self.assertTrue(f.has_construct('domainaxis0'))

        self.assertFalse(f.has_construct('height'))

    def test_Field_compress_uncompress(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        methods = ('contiguous', 'indexed', 'indexed_contiguous')

        for method in methods:
            message = 'method='+method
            for f in cf.read(getattr(self, method)):

                self.assertTrue(bool(f.data.get_compression_type()), message)

                u = f.uncompress()
                self.assertFalse(bool(u.data.get_compression_type()), message)
                self.assertTrue(f.equals(u, verbose=2), message)

                for method1 in methods:
                    message += ', method1='+method1
                    if method1 == 'indexed_contiguous':
                        if f.ndim != 3:
                            continue
                    elif f.ndim != 2:
                        continue

                    c = u.compress(method1)
                    self.assertTrue(bool(c.data.get_compression_type()),
                                    message)

                    self.assertTrue(u.equals(c, verbose=2), message)
                    self.assertTrue(f.equals(c, verbose=2), message)

                    c = f.compress(method1)
                    self.assertTrue(bool(c.data.get_compression_type()),
                                    message)

                    self.assertTrue(u.equals(c, verbose=2), message)
                    self.assertTrue(f.equals(c, verbose=2), message)

                    cf.write(c, 'delme.nc')
                    c = cf.read('delme.nc')[0]

                    self.assertTrue(
                        bool(c.data.get_compression_type()), message)
                    self.assertTrue(f.equals(c, verbose=2), message)
        # --- End: for

    def test_Field_apply_masking(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.example_field(0)

        for prop in ('missing_value', '_FillValue', 'valid_min',
                     'valid_max', 'valid_range'):
            f.del_property(prop, None)

        d = f.data.copy()
        g = f.copy()
        self.assertIsNone(f.apply_masking(inplace=True))
        self.assertTrue(f.equals(g, verbose=1))

        x = 0.11
        y = 0.1
        z = 0.2

        f.set_property('_FillValue', x)
        d = f.data.copy()

        g = f.apply_masking()
        e = d.apply_masking(fill_values=[x])
        self.assertTrue(e.equals(g.data, verbose=1))
        self.assertTrue(g.data.array.count() == g.data.size - 1)

        f.set_property('valid_range', [y, z])
        d = f.data.copy()
        g = f.apply_masking()
        e = d.apply_masking(fill_values=[x], valid_range=[y, z])
        self.assertTrue(e.equals(g.data, verbose=1))

        f.del_property('valid_range')
        f.set_property('valid_min', y)
        g = f.apply_masking()
        e = d.apply_masking(fill_values=[x], valid_min=y)
        self.assertTrue(e.equals(g.data, verbose=1))

        f.del_property('valid_min')
        f.set_property('valid_max', z)
        g = f.apply_masking()
        e = d.apply_masking(fill_values=[x], valid_max=z)
        self.assertTrue(e.equals(g.data, verbose=1))

        f.set_property('valid_min', y)
        g = f.apply_masking()
        e = d.apply_masking(fill_values=[x], valid_min=y, valid_max=z)
        self.assertTrue(e.equals(g.data, verbose=1))

    def test_Field_flatten(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        axis = f.set_construct(cf.DomainAxis(1))
        d = cf.DimensionCoordinate()
        d.standard_name = 'time'
        d.set_data(cf.Data(123., 'days since 2000-01-02'))
        f.set_construct(d, axes=axis)

        g = f.flatten()
        h = f.flatten(list(range(f.ndim)))
        self.assertTrue(h.equals(g, verbose=2))

        g = f.flatten('time')
        self.assertTrue(g.equals(f, verbose=2))

        for i in (0, 1, 2):
            g = f.flatten(i)
            self.assertTrue(g.equals(f, verbose=2))
            g = f.flatten([i, 'time'])
            self.assertTrue(g.equals(f, verbose=2))

        for axes in axes_combinations(f):
            g = f.flatten(axes)

            if len(axes) <= 1:
                shape = f.shape
            else:
                shape = [n for i, n in enumerate(f.shape) if i not in axes]
                shape.insert(sorted(axes)[0],
                             numpy.prod([n for i, n in enumerate(f.shape)
                                         if i in axes]))

            self.assertTrue(g.shape == tuple(shape))
            self.assertTrue(g.ndim == f.ndim-len(axes)+1)
            self.assertTrue(g.size == f.size)

        self.assertTrue(f.equals(f.flatten([]), verbose=2))
        self.assertIsNone(f.flatten(inplace=True))

    def test_Field_bin(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        d = f.digitize(10)
        b = f.bin('sample_size', digitized=d)

        a = numpy.ma.masked_all((10,), dtype=int)
        a[...] = 9
        self.assertTrue((a == b.array).all())

        b = f.bin('sample_size', digitized=[d])

        a = numpy.ma.masked_all((10,), dtype=int)
        a[...] = 9
        self.assertTrue((a == b.array).all())

        b = f.bin('sample_size', digitized=[d, d])

        a = numpy.ma.masked_all((10, 10), dtype=int)
        for i in range(9):
            a[i, i] = 9

        self.assertTrue((a == b.array).all())

        b = f.bin('sample_size', digitized=[d, d, d])

        a = numpy.ma.masked_all((10, 10, 10), dtype=int)
        for i in range(9):
            a[i, i, i] = 9

        self.assertTrue((a == b.array).all())

    def test_Field_direction(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()
        yaxis = f.domain_axis('Y', key=True)
        ydim = f.dimension_coordinate('Y', key=True)
        f.direction('X')
        f.del_construct(ydim)
        f.direction(yaxis)
        self.assertTrue(f.direction('qwerty'))

        f = self.f.copy()
        self.assertIsInstance(f.directions(), dict)
        f.directions()

    def test_Field_domain_axis_position(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f

        for i in range(f.ndim):
            self.assertTrue(f.domain_axis_position(i) == i)

        for i in range(1, f.ndim+1):
            self.assertTrue(f.domain_axis_position(-i) == -i + 3)

        data_axes = f.get_data_axes()
        for key in data_axes:
            self.assertTrue(
                f.domain_axis_position(key) == data_axes.index(key))

        self.assertTrue(f.domain_axis_position('Z') == 0)
        self.assertTrue(f.domain_axis_position('grid_latitude') == 1)

    def test_Field_weights(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()
        f += 1

        w = f.weights()
        self.assertIsInstance(w, cf.Field)

        w = f.weights(None)
        self.assertIsInstance(w, cf.Field)
        self.assertTrue(w.data.equals(cf.Data(1.0, '1'), verbose=2))

        w = f.weights(data=True)
        self.assertIsInstance(w, cf.Data)

        w = f.weights(None, data=True)
        self.assertIsInstance(w, cf.Data)
        self.assertTrue(w.equals(cf.Data(1.0, '1'), verbose=2))

        w = f.weights(components=True)
        self.assertIsInstance(w, dict)

        w = f.weights(None, components=True)
        self.assertIsInstance(w, dict)
        self.assertTrue(w == {})

        w = f.weights(methods=True)
        self.assertIsInstance(w, dict)

        w = f.weights(None, methods=True)
        self.assertIsInstance(w, dict)
        self.assertTrue(w == {})

        w = f.weights()
        x = f.weights(w)
        self.assertTrue(x.equals(w, verbose=2))

        for components in (False, True):
            for m in (False, True):
                for d in (False, True):
                    if components:
                        d = False

                    y = f.weights(w, components=components, measure=m, data=d)
                    y = f.weights(w.transpose(), components=components,
                                  measure=m, data=d)
                    y = f.weights(w.data, components=components, measure=m,
                                  data=d)
                    y = f.weights(f.data.squeeze(), components=components,
                                  measure=m, data=d)
                    y = f.weights(components=components, measure=m, data=d)
                    y = f.weights('grid_longitude', components=components,
                                  measure=m, data=d)
                    y = f.weights(['grid_longitude'], components=components,
                                  measure=m, data=d)
        # --- End: for

        with self.assertRaises(Exception):
            f.weights(components=True, data=True)

    def test_Field_replace_construct(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        for x in ('grid_longitude',
                  'latitude',
                  'grid_mapping_name:rotated_latitude_longitude',
                  'ncvar%a'):
            for copy in (True, False):
                f.replace_construct(x, f.construct(x), copy=copy)
        # --- End: for

        with self.assertRaises(Exception):
            f.replace_construct('grid_longitude', f.construct('latitude'))

        with self.assertRaises(Exception):
            f.replace_construct(
                'grid_longitude', f.construct('grid_latitude'))

    def test_Field_allclose(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()
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
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()
        f[0, 3] *= -1
        f[0, 5, ::2] = cf.masked

        for axes in axes_combinations(f):
            for method in (
                    'sum',
                    'min',
                    'max',
                    'minimum_absolute_value',
                    'maximum_absolute_value',
                    'mid_range',
                    'range',
                    'sample_size',
                    'sum_of_squares',
                    'median',
                    'sum_of_weights',
                    'sum_of_weights2',
            ):
                for weights in (None, 'area'):
                    a = f.collapse(method, axes=axes, weights=weights).data
                    b = getattr(f.data, method)(axes=axes)
                    self.assertTrue(
                        a.equals(b, rtol=1e-05, atol=1e-08, verbose=2),
                        '{} weights={}, axes={}, {!r}, {!r}'.format(
                            method, weights, axes, a, b))
            # --- End: for

            for method in ('mean',
                           'mean_absolute_value',
                           # 'mean_of_upper_decile',
                           'root_mean_square'):
                for weights in (None, 'area'):
                    if weights is not None:
                        d_weights = f.weights(weights, components=True)
                    else:
                        d_weights = weights

                    a = f.collapse(method, axes=axes, weights=weights).data
                    b = getattr(f.data, method)(axes=axes, weights=d_weights)
                    self.assertTrue(
                        a.equals(b, rtol=1e-05, atol=1e-08, verbose=2),
                        '{} weights={}, axes={}, {!r}, {!r}'.format(
                            method, weights, axes, a, b))
            # --- End: for

            for method in ('integral',):
                weights = 'area'
                d_weights = f.weights(weights, components=True, measure=True)
                a = f.collapse(method, axes=axes, weights=weights,
                               measure=True).data
                b = getattr(f.data, method)(axes=axes, weights=d_weights)
                self.assertTrue(
                    a.equals(b, rtol=1e-05, atol=1e-08, verbose=2),
                    '{} weighted axes={}, {!r}, {!r}'.format(
                        method, axes, a, b))
        # --- End: for

        for axes in axes_combinations(f):
            if axes == (0,):
                continue

            for method in ('var',
                           'sd'):
                for weights in (None, 'area'):
                    if weights is not None:
                        d_weights = f.weights(weights, components=True)
                    else:
                        d_weights = None

                    a = f.collapse(method, axes=axes, weights=weights).data
                    b = getattr(f.data, method)(
                        axes=axes, ddof=1, weights=d_weights)
                    self.assertTrue(
                        a.equals(b, rtol=1e-05, atol=1e-08, verbose=2),
                        '{} weights={}, axes={}, {!r}, {!r}'.format(
                            method, weights, axes, a, b))
            # --- End: for

            for method in ('mean_of_upper_decile',):
                for weights in (None, 'area'):
                    if weights is not None:
                        d_weights = f.weights(weights, components=True)
                    else:
                        d_weights = None

                    a = f.collapse(method, axes=axes, weights=weights).data
                    b = getattr(f.data, method)(axes=axes, weights=d_weights)
                    self.assertTrue(
                        a.equals(b, rtol=1e-05, atol=1e-08, verbose=2),
                        '{} weights={}, axes={}, {!r}, {!r}'.format(
                            method, weights, axes, a, b))
        # --- End: for

    def test_Field_all(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        self.assertFalse(f.all())

        f[0, 0, 0] = 99
        self.assertTrue(f.all())

        f.del_data()
        self.assertFalse(f.all())

    def test_Field_any(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        self.assertTrue(f.any())

        f.del_data()
        self.assertFalse(f.any())

    def test_Field_axis(self):
        # v2 compatibility
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f

        self.assertTrue(f.domain_axes.equals(f.axes(), verbose=2))
        self.assertTrue(
            f.domain_axes('domainaxis1').equals(
                f.axes('domainaxis1'), verbose=2))

        self.assertTrue(
            f.domain_axis('domainaxis1').equals(
                f.axis('domainaxis1'), verbose=2))

    def test_Field_ATOL_RTOL(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f

        g = f.copy()
        self.assertTrue(f.equals(g, verbose=2))
        g[0, 0, 0] += 0.001

        self.assertFalse(f.equals(g))
        self.assertTrue(f.equals(g, atol=0.1, verbose=2))
        self.assertFalse(f.equals(g))
        atol = cf.ATOL(0.1)
        self.assertTrue(f.equals(g, verbose=2))
        cf.ATOL(atol)
        self.assertFalse(f.equals(g))

        self.assertTrue(f.equals(g, rtol=10, verbose=2))
        self.assertFalse(f.equals(g))
        rtol = cf.RTOL(10)
        self.assertTrue(f.equals(g, verbose=2))
        cf.RTOL(rtol)
        self.assertFalse(f.equals(g))

        cf.ATOL(self.atol)
        cf.RTOL(self.rtol)

    def test_Field_concatenate(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        g = cf.Field.concatenate([f.copy()], axis=0)
        self.assertTrue(g.shape == (1, 10, 9))

        x = [f.copy() for i in range(8)]

        g = cf.Field.concatenate(x, axis=0)
        self.assertTrue(g.shape == (8, 10, 9))

        key = x[3].construct_key('latitude')
        x[3].del_construct(key)
        g = cf.Field.concatenate(x, axis=0)
        self.assertTrue(g.shape == (8, 10, 9))

        with self.assertRaises(Exception):
            g = cf.Field.concatenate([], axis=0)

    def test_Field_AUXILIARY_MASK(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

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
            cf.CHUNKSIZE(chunksize)

            f = cf.read(self.contiguous, verbose=0)[0]

            for (method, shape, a) in zip(['compress', 'envelope', 'full'],
                                          [ac.shape, ae.shape, af.shape],
                                          [ac, ae, af]):
                message = 'method={!r}'.format(method)

                i = f.indices(method, time=query1)

                g = f.subspace(method, time=query1)
                t = g.coordinate('time')

                self.assertTrue(g.shape == shape, message)
                self.assertTrue(t.shape == shape, message)

                self.assertTrue(
                    (t.data._auxiliary_mask_return().array == a.mask).all(),
                    message
                )
                self.assertTrue(
                    (g.data._auxiliary_mask_return().array == a.mask).all(),
                    message
                )

                self.assertTrue(
                    cf.functions._numpy_allclose(t.array, a), message)
        # --- End: for

        cf.CHUNKSIZE(self.original_chunksize)

        query2 = cf.set([1, 3, 5])

        ac2 = numpy.ma.masked_all((2, 6))
        ac2[0, 0] = 1
        ac2[0, 1] = 3
        ac2[0, 3] = 5
        ac2[1, 2] = 1
        ac2[1, 4] = 3
        ac2[1, 5] = 5

        ae2 = numpy.ma.where(
            (ae == 1) | (ae == 3) | (ae == 5), ae, numpy.ma.masked)
        af2 = numpy.ma.where(
            (af == 1) | (af == 3) | (af == 5), af, numpy.ma.masked)

        for chunksize in self.chunk_sizes[0:2]:
            cf.CHUNKSIZE(chunksize)
            f = cf.read(self.contiguous)[0]

            for (method, shape, a) in zip(['compress', 'envelope', 'full'],
                                          [ac2.shape, ae2.shape, af2.shape],
                                          [ac2, ae2, af2]):

                message = 'method={!r}'.format(method)

                h = f.subspace('full', time=query1)
                g = h.subspace(method, time=query2)
                t = g.coordinate('time')

                self.assertTrue(g.shape == shape, message)
                self.assertTrue(t.shape == shape, message)

                self.assertTrue(
                    (t.data._auxiliary_mask_return().array == a.mask).all(),
                    message
                )
                self.assertTrue(
                    (g.data._auxiliary_mask_return().array == a.mask).all(),
                    message
                )

                self.assertTrue(
                    cf.functions._numpy_allclose(t.array, a), message)
        # --- End: for

        cf.CHUNKSIZE(self.original_chunksize)

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
            cf.CHUNKSIZE(chunksize)
            f = cf.read(self.contiguous)[0].subspace[[0, 2, 3], 1:]

            for (method, shape, a) in zip(['compress', 'envelope', 'full'],
                                          [ac3.shape, ae3.shape, af3.shape],
                                          [ac3, ae3, af3]):

                message = 'method={!r}'.format(method)

                g = f.subspace(method, time=query3)
                t = g.coordinate('time')

                self.assertTrue(g.shape == shape, message)
                self.assertTrue(t.shape == shape, message)

                self.assertTrue(
                    (t.data._auxiliary_mask_return().array == a.mask).all(),
                    message
                )
                self.assertTrue(
                    (g.data._auxiliary_mask_return().array == a.mask).all(),
                    message)

                self.assertTrue(
                    cf.functions._numpy_allclose(t.array, a),
                    message)
        # --- End: for
        cf.CHUNKSIZE(self.original_chunksize)

    def test_Field__getitem__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)[0].squeeze()
        d = f.data
        f = cf.read(self.filename)[0].squeeze()

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
        _ = g[0:3, 5]

        g = f[0].squeeze()
        _ = g[5]

    def test_Field__setitem__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)[0].squeeze()

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

        f = cf.read(self.filename)[0].squeeze()
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
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)[0]  # .squeeze()

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

        axis = a.domain_axis('grid_longitude', key=1)
        for key in a.field_ancillaries.filter_by_axis('or', axis):
            a.del_construct(key)

        for key in a.cell_measures.filter_by_axis('or', axis):
            a.del_construct(key)

        self.assertTrue(a.equals(b, verbose=2))
        self.assertTrue(b.equals(a, verbose=2))

        with self.assertRaises(Exception):
            _ = f + ('a string',)

    def test_Field__mul__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)[0].squeeze()

        f.standard_name = 'qwerty'
        g = f * f

        self.assertIsNone(g.get_property('standard_name', None))

    def test_Field__gt__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)[0].squeeze()

        f.standard_name = 'qwerty'
        g = f > f.mean()

        self.assertTrue(g.Units.equals(cf.Units()))
        self.assertIsNone(g.get_property('standard_name', None))

    def test_Field_domain_mask(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        _ = f.domain_mask()
        _ = f.domain_mask(grid_longitude=cf.wi(25, 31))

    def test_Field_cumsum(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)[0]

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
            new_mask = (numpy.cumsum(mask, axis=axis) ==
                        numpy.arange(1, size+1).reshape(shape))
            a = numpy.ma.array(a, mask=new_mask, copy=False)
            self.assertTrue(cf.functions._numpy_allclose(g.array, a,
                                                         verbose=2))

    def test_Field_flip(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        f = self.f.copy()
        g = f[(slice(None, None, -1),) * f.ndim]

        h = f.flip()
        self.assertTrue(h.equals(g, verbose=1))

        h = f.flip(f.get_data_axes())
        self.assertTrue(h.equals(g, verbose=1))

        h = f.flip(list(range(f.ndim)))
        self.assertTrue(h.equals(g, verbose=1))

        h = f.flip(['X', 'Z', 'Y'])
        self.assertTrue(h.equals(g, verbose=1))

        h = f.flip((re.compile('^atmos'), 'grid_latitude', 'grid_longitude'))
        self.assertTrue(h.equals(g, verbose=1))

        g = f.subspace(grid_longitude=slice(None, None, -1))
        self.assertIsNone(f.flip('X', inplace=True))
        self.assertTrue(f.equals(g, verbose=1))

    def test_Field_close(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)[0]
        self.assertIsNone(f.close())

        _ = repr(f.data)
        for c in f.constructs.filter_by_data().values():
            _ = repr(c.data)

    def test_Field_anchor(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        dimarray = self.f.dimension_coordinate('grid_longitude').array

        f = self.f.copy()
        f.cyclic('grid_longitude', period=45)
        self.assertIsNone(f.anchor('grid_longitude', 32, inplace=True))
        self.assertIsInstance(
            f.anchor('grid_longitude', 32, dry_run=True), dict)

        g = f.subspace(grid_longitude=[0])
        g.anchor('grid_longitude', 32)
        g.anchor('grid_longitude', 32, inplace=True)
        g.anchor('grid_longitude', 32, dry_run=True)

        f = self.f.copy()

        for period in (dimarray.min()-5, dimarray.min()):
            anchors = numpy.arange(dimarray.min()-3*period,
                                   dimarray.max()+3*period, 0.5)

            f.cyclic('grid_longitude', period=period)

            # Increasing dimension coordinate
            for anchor in anchors:
                g = f.anchor('grid_longitude', anchor)
                x0 = g.coordinate('grid_longitude').datum(-1) - period
                x1 = g.coordinate('grid_longitude').datum(0)
                self.assertTrue(
                    x0 < anchor <= x1,
                    'INCREASING period=%s, x0=%s, anchor=%s, x1=%s' % (
                        period, x0, anchor, x1)
                )
            # --- End: for

            # Decreasing dimension coordinate
            flipped_f = f.flip('grid_longitude')
            for anchor in anchors:
                g = flipped_f.anchor('grid_longitude', anchor)
                x1 = g.coordinate('grid_longitude').datum(-1) + period
                x0 = g.coordinate('grid_longitude').datum(0)
                self.assertTrue(
                    x1 > anchor >= x0,
                    "DECREASING period={}, x0={}, anchor={}, x1={}".format(
                        period, x1, anchor, x0))
        # --- End: for

    def test_Field_cell_area(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        ca = f.cell_area()

        self.assertTrue(ca.ndim == 2)
        self.assertTrue(len(ca.dimension_coordinates) == 2)
        self.assertTrue(len(ca.domain_ancillaries) == 0)
        self.assertTrue(len(ca.coordinate_references) == 1)

        f.del_construct('cellmeasure0')
        y = f.dimension_coordinate('Y')
        y.set_bounds(y.create_bounds())
        self.assertTrue(len(f.cell_measures) == 0)

        ca = f.cell_area()

        self.assertTrue(ca.ndim == 2)
        self.assertTrue(len(ca.dimension_coordinates) == 2)
        self.assertTrue(len(ca.domain_ancillaries) == 0)
        self.assertTrue(len(ca.coordinate_references) == 1)
        self.assertTrue(ca.Units.equivalent(cf.Units('m2')), ca.Units)

        y = f.dimension_coordinate('Y')
        self.assertTrue(y.has_bounds())

    def test_Field_radius(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        with self.assertRaises(Exception):
            _ = f.radius()

        for default in ('earth', cf.field._earth_radius):
            r = f.radius(default=default)
            self.assertTrue(r.Units == cf.Units('m'))
            self.assertTrue(r == cf.field._earth_radius)

        a = cf.Data(1234, 'm')
        for default in (1234,
                        cf.Data(1234, 'm'),
                        cf.Data([1234], 'm'),
                        cf.Data([[1234]], 'm'),
                        cf.Data(1234, 'm'),
                        cf.Data(1.234, 'km')):
            r = f.radius(default=default)
            self.assertTrue(r.Units == cf.Units('m'))
            self.assertTrue(r == a)

        with self.assertRaises(Exception):
            _ = f.radius()

        with self.assertRaises(Exception):
            _ = f.radius(default=[12, 34])

        with self.assertRaises(Exception):
            _ = f.radius(default=[[12, 34]])

        with self.assertRaises(Exception):
            _ = f.radius(default='qwerty')

        cr = f.coordinate_reference('rotated_latitude_longitude')
        cr.datum.set_parameter('earth_radius', a.copy())

        r = f.radius(default=None)
        self.assertTrue(r.Units == cf.Units('m'))
        self.assertTrue(r == a)

        cr = f.coordinate_reference('atmosphere_hybrid_height_coordinate')
        cr.datum.set_parameter('earth_radius', a.copy())

        r = f.radius(default=None)
        self.assertTrue(r.Units == cf.Units('m'))
        self.assertTrue(r == a)

        cr = f.coordinate_reference('atmosphere_hybrid_height_coordinate')
        cr.datum.set_parameter('earth_radius', cf.Data(5678, 'km'))

        with self.assertRaises(Exception):
            _ = f.radius(default=None)

        cr = f.coordinate_reference('atmosphere_hybrid_height_coordinate')
        cr.datum.del_parameter('earth_radius')

        cr = f.coordinate_reference('rotated_latitude_longitude')
        cr.datum.set_parameter('earth_radius', cf.Data([123, 456], 'm'))

        with self.assertRaises(Exception):
            _ = f.radius(default=None)

    def test_Field_DATA(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        _ = f.rank
        _ = f.data
        del f.data

        f = self.f.copy()

        self.assertTrue(f.has_data())
        data = f.get_data()
        _ = f.del_data()
        _ = f.get_data(default=None)
        _ = f.del_data(default=None)
        self.assertFalse(f.has_data())
        _ = f.set_data(data, axes=None)
        _ = f.set_data(data, axes=None, copy=False)
        self.assertTrue(f.has_data())

        f = self.f.copy()
        _ = f.del_data_axes()
        self.assertFalse(f.has_data_axes())
        self.assertIsNone(f.del_data_axes(default=None))

        f = self.f.copy()
        for key in f.constructs.filter_by_data():
            self.assertTrue(f.has_data_axes(key))
            _ = f.get_data_axes(key)
            _ = f.del_data_axes(key)
            self.assertIsNone(f.del_data_axes(key, default=None))
            self.assertIsNone(f.get_data_axes(key, default=None))
            self.assertFalse(f.has_data_axes(key))

        g = cf.Field()
        g.set_data(cf.Data(9))
        with self.assertRaises(Exception):
            g.set_data(cf.Data(9), axes='X')

        g = self.f.copy()
        with self.assertRaises(Exception):
            g.set_data(cf.Data([9], axes='qwerty'))

        with self.assertRaises(Exception):
            g.set_data(cf.Data([9], axes=['X', 'Y']))

        g = cf.Field()
        g.set_data(cf.Data(9))
        with self.assertRaises(Exception):
            g.set_data(cf.Data(9), axes='X')

        g = cf.Field()
        a = g.set_construct(cf.DomainAxis(9))
        b = g.set_construct(cf.DomainAxis(10))
        g.set_data(cf.Data(list(range(9))), axes=a)
        with self.assertRaises(Exception):
            g.set_data(cf.Data(list(range(9))), axes=b)
        with self.assertRaises(Exception):
            g.set_data(cf.Data(list(range(9))), axes=[b, a])

        g = cf.Field()
#        with self.assertRaises(Exception):
#            g.set_data(cf.Data(list(range(9))))
        a = g.set_construct(cf.DomainAxis(9))
        b = g.set_construct(cf.DomainAxis(9))
        c = g.set_construct(cf.DomainAxis(10))
        d = g.set_construct(cf.DomainAxis(8))
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
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f
        self.assertTrue(
            f.get_data_axes() == (
                'domainaxis0', 'domainaxis1', 'domainaxis2'),
            str(f.get_data_axes())
        )

        f = cf.Field()
        f.set_data(cf.Data(9), axes=())
        self.assertTrue(f.get_data_axes() == ())

        f.del_data()
        self.assertTrue(f.get_data_axes() == ())

        f.del_data_axes()
        self.assertIsNone(f.get_data_axes(default=None))

    def test_Field_equals(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)[0]
        g = f.copy()
        self.assertTrue(f.equals(f, verbose=2))
        self.assertTrue(f.equals(g, verbose=2))
        g.set_property('foo', 'bar')
        self.assertFalse(f.equals(g))
        g = f.copy()
        self.assertFalse(f.equals(g+1))

    def test_Field_insert_dimension(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()
        f.squeeze('Z', inplace=True)
        self.assertTrue(f.ndim == 2)
        g = f.copy()

        self.assertIsNone(g.insert_dimension('Z', inplace=True))

        self.assertTrue(g.ndim == f.ndim + 1)
        self.assertTrue(g.get_data_axes()[1:] == f.get_data_axes())

        with self.assertRaises(ValueError):
            f.insert_dimension(1, 'qwerty')

    def test_Field_indices(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        array = numpy.ma.array(f.array)

        x = f.dimension_coordinate('X')
        a = x.varray
        a[...] = numpy.arange(0, 360, 40)
        x.set_bounds(x.create_bounds())
        f.cyclic('X', iscyclic=True, period=360)

        f0 = f.copy()

        # wi (increasing)
        indices = f.indices(grid_longitude=cf.wi(50, 130))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 2), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [80, 120]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 50))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-80, -40, 0, 40]).all())

        indices = f.indices(grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(310-1080, 450-1080))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(310+720, 450+720))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 370))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 9), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue(
            (x == [-80, -40, 0, 40,  80, 120, 160, 200, 240.]).all())

        with self.assertRaises(IndexError):
            f.indices(grid_longitude=cf.wi(90, 100))

        indices = f.indices('full', grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 9), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue(x.shape == (9,), x.shape)
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320]).all(), x)
        a = array.copy()
        a[..., [3, 4, 5, 6, 7]] = numpy.ma.masked
        self.assertTrue(cf.functions._numpy_allclose(g.array, a), g.array)

        indices = f.indices('full', grid_longitude=cf.wi(70, 200))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 9), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue(x.shape == (9,), x.shape)
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320]).all(), x)
        a = array.copy()
        a[..., [0, 1, 6, 7, 8]] = numpy.ma.masked
        self.assertTrue(cf.functions._numpy_allclose(g.array, a), g.array)

        # wi (decreasing)
        f.flip('X', inplace=True)

        indices = f.indices(grid_longitude=cf.wi(50, 130))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 2), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [80, 120][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 50))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-80, -40, 0, 40][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310-1080, 450-1080))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310+720, 450+720))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        with self.assertRaises(IndexError):
            f.indices(grid_longitude=cf.wi(90, 100))

        indices = f.indices('full', grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 9), g.shape)
        x = g.dimension_coordinate('X').array
#        print (g.array)
#        print (x)
        self.assertTrue(x.shape == (9,), x.shape)
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320][::-1]).all(), x)

        indices = f.indices('full', grid_longitude=cf.wi(70, 200))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 9), g.shape)
        x = g.dimension_coordinate('X').array
#        print (g.array)
#        print (x)
        self.assertTrue(x.shape == (9,), x.shape)
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320][::-1]).all(), x)

        # wo
        f = f0.copy()

        indices = f.indices(grid_longitude=cf.wo(50, 130))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 7), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-200, -160, -120, -80, -40, 0, 40]).all())

        with self.assertRaises(IndexError):
            f.indices(grid_longitude=cf.wo(-90, 370))

        # set
        indices = f.indices(grid_longitude=cf.set([320, 40, 80, 99999]))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 3), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [40, 80, 320]).all())

        indices = f.indices(grid_longitude=cf.lt(90))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 3), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.gt(90))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 6), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [120, 160, 200, 240, 280, 320]).all())

        indices = f.indices(grid_longitude=cf.le(80))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 3), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.ge(80))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 7), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [80, 120, 160, 200, 240, 280, 320]).all())

        # 2-d
        lon = f.construct('longitude').array
        lon = numpy.transpose(lon)
        lon = numpy.expand_dims(lon, 0)

        lat = f.construct('latitude').array
        lat = numpy.expand_dims(lat, 0)

        array = numpy.ma.where((lon >= 92) & (lon <= 134),
                               f.array,
                               numpy.ma.masked)

        for mode in ('', 'compress', 'full', 'envelope'):
            indices = f.indices(mode, longitude=cf.wi(92, 134))
            g = f[indices]
            if mode == 'full':
                shape = (1, 10, 9)
                array2 = array
            elif mode == 'envelope':
                shape = (1, 10, 5)
                array2 = array[..., 3:8]
            else:
                shape = (1, 10, 5)
                array2 = array[..., 3:8]

            self.assertTrue(g.shape == shape, str(g.shape)+'!='+str(shape))
            self.assertTrue(
                cf.functions._numpy_allclose(array2, g.array), g.array)

        array = numpy.ma.where(((lon >= 72) & (lon <= 83)) | (lon >= 118),
                               f.array,
                               numpy.ma.masked)

        for mode in ('', 'compress', 'full', 'envelope'):
            indices = f.indices(mode, longitude=cf.wi(72, 83) | cf.gt(118))
            g = f[indices]
            if mode == 'full':
                shape = (1, 10, 9)
            elif mode == 'envelope':
                shape = (1, 10, 8)
            else:
                shape = (1, 10, 6)

            self.assertTrue(g.shape == shape, str(g.shape)+'!='+str(shape))

        indices = f.indices('full',
                            longitude=cf.wi(92, 134),
                            latitude=cf.wi(-26, -20) | cf.ge(30))
        g = f[indices]
        self.assertTrue(g.shape == (1, 10, 9), g.shape)
        array = numpy.ma.where(
            (((lon >= 92) & (lon <= 134)) &
             (((lat >= -26) & (lat <= -20)) | (lat >= 30))),
            f.array,
            numpy.ma.masked)
        self.assertTrue(cf.functions._numpy_allclose(array, g.array), g.array)

        for mode in ('', 'compress', 'full', 'envelope'):
            indices = f.indices(mode, grid_longitude=cf.contains(23.2))
            g = f[indices]
            if mode == 'full':
                shape = f.shape
            else:
                shape = (1, 10, 1)

            self.assertTrue(g.shape == shape, g.shape)

            if mode != 'full':
                self.assertTrue(
                    g.construct('grid_longitude').array == 40)  # TODO
        # --- End: for

        for mode in ('', 'compress', 'full', 'envelope'):
            indices = f.indices(mode, grid_latitude=cf.contains(3))
            g = f[indices]
            if mode == 'full':
                shape = f.shape
            else:
                shape = (1, 1, 9)

            self.assertTrue(g.shape == shape, g.shape)

            if mode != 'full':
                self.assertTrue(g.construct('grid_latitude').array == 3)
        # --- End: for

        for mode in ('', 'compress', 'full', 'envelope'):
            indices = f.indices(mode, longitude=cf.contains(83))
            g = f[indices]
            if mode == 'full':
                shape = f.shape
            else:
                shape = (1, 1, 1)

            self.assertTrue(g.shape == shape, g.shape)

            if mode != 'full':
                self.assertTrue(g.construct('longitude').array == 83)
        # --- End: for

#        print (f)
#        lon2 = f.construct('longitude').transpose()
#        a = lon2.array
#        b = numpy.empty(a.shape + (4,), dtype=float)
#        b[..., 0] = a - 0.5
#        b[..., 1] = a - 0.5
#        b[..., 2] = a + 0.5
#        b[..., 3] = a + 0.5
#        lon2.set_bounds(cf.Bounds(data=cf.Data(b)))
#        lon2.transpose(inplace=True)
#        a = lon2.array
#        b = lon2.bounds.array
#        q= lon2.transpose()
#        print (a)
#        print ('lon2.bounds[4, 2]=',lon2.bounds[4, 2].array)
#        print ('   q.bounds[3, 2]=',q.bounds[3, 2].array)
#
#        print (a[2, 3], b[2, 3])
#
#        lat2 = f.construct('latitude')
#        a = lat2.array
#        b = numpy.empty(a.shape + (4,), dtype=float)
#        b[..., 0] = a - 0.5
#        b[..., 1] = a - 0.5
#        b[..., 2] = a + 0.5
#        b[..., 3] = a + 0.5
#        lat2.set_bounds(cf.Bounds(data=cf.Data(b)))
#        print (a[3, 2], b[3, 2])
#
#        print ('looooon', repr((cf.contains(91.2)==lon2).array))
#
#        for mode in ('', 'compress', 'full', 'envelope'):
#            print (mode)
#            indices = f.indices('_debug', mode,
#                                longitude=cf.contains(91.2),
#                                latitude=cf.contains(-16.1))
#            g = f[indices]
#            if mode == 'full':
#                shape = f.shape
#            else:
#                shape = (1, 1, 1)
#
#            self.assertTrue(g.shape == shape, g.shape)
#
#            if mode != 'full':
#                self.assertTrue(g.construct('longitude').array == 83)
#        # --- End: for

        # Calls that should fail
        with self.assertRaises(Exception):
            f.indices(grid_longitudecf.gt(23), grid_longitude=cf.wi(92, 134))
        with self.assertRaises(Exception):
            f.indices(grid_longitude=cf.gt(23), longitude=cf.wi(92, 134))
        with self.assertRaises(Exception):
            f.indices(grid_latitude=cf.contains(-23.2))

    def test_Field_match(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)[0]
        f.long_name = 'qwerty'
        f.nc_set_variable('tas')

        # match, match_by_identity
        for identities in (
            [],
            ['eastward_wind'],
            ['standard_name=eastward_wind'],
            ['long_name=qwerty'],
            ['ncvar%tas'],
            [re.compile('^eastw')],
            ['eastward_wind', 'long_name=qwerty'],
            ['None', 'eastward_wind'],
        ):
            self.assertTrue(f.match(*identities),
                            'Failed with {}'.format(identities))
            self.assertTrue(f.match_by_identity(*identities),
                            'Failed with {}'.format(identities))

        # match_by_property
        for mode in ([], ['and']):
            for properties in (
                    {},
                    {'standard_name': 'eastward_wind'},
                    {'long_name': 'qwerty'},
                    {'standard_name': re.compile('^eastw')},
                    {'standard_name': 'eastward_wind',
                     'long_name': 'qwerty'},
            ):
                self.assertTrue(f.match_by_property(*mode, **properties),
                                'Failed with {} {}'.format(mode, properties))

        for mode in (['or'],):
            for properties in (
                    {},
                    {'standard_name': 'eastward_wind'},
                    {'long_name': 'qwerty'},
                    {'standard_name': re.compile('^eastw')},
                    {'standard_name': 'eastward_wind',
                     'long_name': 'qwerty'},
                    {'standard_name': 'None',
                     'long_name': 'qwerty'},
            ):
                self.assertTrue(f.match_by_property(*mode, **properties),
                                'Failed with {} {}'.format(mode, properties))
        # match_by_units
        self.assertTrue(f.match_by_units('m s-1'))
        self.assertTrue(f.match_by_units('km h-1', exact=False))
        self.assertFalse(f.match_by_units('km h-1'))
        self.assertFalse(f.match_by_units('K s'))

        self.assertTrue(f.match_by_units(cf.Units('m s-1')))
        self.assertTrue(f.match_by_units(cf.Units('km h-1'), exact=False))
        self.assertFalse(f.match_by_units(cf.Units('km h-1')))
        self.assertFalse(f.match_by_units(cf.Units('K s')))

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
            self.assertTrue(f.match_by_construct('X', OR=OR))
            self.assertTrue(f.match_by_construct('latitude', OR=OR))
            self.assertTrue(f.match_by_construct('X', 'latitude', OR=OR))
            self.assertTrue(f.match_by_construct('X', 'Y', OR=OR))
            self.assertTrue(f.match_by_construct('X', 'Y', 'latitude', OR=OR))
            self.assertTrue(f.match_by_construct('grid_latitude: max', OR=OR))
            self.assertTrue(
                f.match_by_construct(
                    'grid_longitude: mean grid_latitude: max', OR=OR))
            self.assertTrue(
                f.match_by_construct('X', 'method:max', OR=OR))
            self.assertTrue(
                f.match_by_construct('X', 'grid_latitude: max', OR=OR))

        self.assertFalse(f.match_by_construct('qwerty'))
        self.assertFalse(f.match_by_construct('qwerty', OR=True))
        self.assertFalse(f.match_by_construct('X', 'qwerty'))
        self.assertFalse(f.match_by_construct('time: mean'))

        self.assertTrue(f.match_by_construct('X', 'qwerty', OR=True))
        self.assertTrue(
            f.match_by_construct('X', 'qwerty', 'method:max', 'over:years',
                                 OR=True))
        self.assertTrue(
            f.match_by_construct('X', 'qwerty', 'grid_latitude: max',
                                 'over:years',
                                 OR=True))

#   def test_Field_period(self):
#       if self.test_only and inspect.stack()[0][3] not in self.test_only:
#           return
#
#       f = self.f.copy()
#       f.dimension_coordinate('X').period(None)
#       f.cyclic('X', False)
#       print('ATTENTION 0')
#       self.assertIsNone(f.period('X'))
#       f.cyclic('X', period=360)
#       print('ATTENTION 1')
#       self.assertTrue(f.period('X') == cf.Data(360, 'degrees'))
#       f.cyclic('X', False)
#       print('ATTENTION 2')
#       self.assertTrue(f.period('X') == cf.Data(360, 'degrees'))
#       f.dimension_coordinate('X').period(None)
#       print('ATTENTION 3')
#       self.assertIsNone(f.period('X'))

    def test_Field_autocyclic(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        self.assertFalse(f.autocyclic())
        f.dimension_coordinate('X').del_bounds()
        f.autocyclic()

    def test_Field_construct_key(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()
        x = f.construct('grid_longitude')
        i = f.item('grid_longitude')
        self.assertTrue(x.equals(i, verbose=2))

        x = f.construct_key('grid_longitude')
        i = f.item('grid_longitude', key=True)
        self.assertTrue(x == i)

    def test_Field_item(self):
        # v2 compatibility
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()
        x = f.construct('grid_longitude')
        i = f.item('grid_longitude')
        self.assertTrue(x.equals(i, verbose=2))

        x = f.construct_key('grid_longitude')
        i = f.item('grid_longitude', key=True)
        self.assertTrue(x == i)

        x = f.construct('grid_longitude', key=True)
        i = f.item('grid_longitude', key=True)
        self.assertTrue(x == i)

        self.assertTrue(
            f.constructs.filter_by_data().equals(f.items(), verbose=2))
        self.assertTrue(
            f.constructs('X', 'Y').equals(f.items(*['X', 'Y']), verbose=2))

    def test_Field_convolution_filter(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        window = [0.1, 0.15, 0.5, 0.15, 0.1]

        f = cf.read(self.filename1)[0]

        # Test user weights in different modes
        for mode in ('reflect', 'constant', 'nearest', 'mirror', 'wrap'):
            g = f.convolution_filter(window, axis=-1, mode=mode, cval=0.0)
            self.assertTrue((g.array == convolve1d(f.array, window, axis=-1,
                                                   mode=mode)).all())

#    def test_Field_moving_mean(self):
#        if self.test_only and inspect.stack()[0][3] not in self.test_only:
#            return
#
#        weights = cf.Data(numpy.arange(1, 9.0)) / 2
#
#        f = cf.example_field(0)
#
#        g = f.moving_mean(window_size=3, axis='X', inplace=True)
#        self.assertIsNone(g)
#
#        f = cf.example_field(0)
#        a = f.array
#
#        # ------------------------------------------------------------
#        # Origin = 0
#        # ------------------------------------------------------------
#        for mode in ('constant', 'wrap', 'reflect', 'nearest',
#                     'mirror'):
#            g = f.moving_mean(window_size=3, axis='X',
#                              weights=weights, mode=mode)
#
#            with self.assertRaises(ValueError):
#                g = f.moving_mean(window_size=3, axis='X', mode=mode,
#                                  cval=39)
#
#            for i in range(1, 7):
#                x = ((a[:, i-1:i+2] * weights[i-1:i+2]).sum(axis=1) /
#                      weights[i-1:i+2].sum())
#                numpy.testing.assert_allclose(x, g.array[:, i])
#        # --- End: for
#
#        # Test 'wrap'
#        for mode in (None, 'wrap'):
#            g = f.moving_mean(window_size=3, axis='X',
#                              weights=weights, mode=mode)
#
#            for i, ii in zip([0, -1], ([0, 1, -1], [0, -2, -1])):
#                x = (a[:, ii] * weights[ii]).sum(axis=1) / weights[ii].sum()
#                numpy.testing.assert_allclose(x, g.array[:, i])
#        # --- End: for
#
#        # ------------------------------------------------------------
#        # Origin = 1
#        # ------------------------------------------------------------
#        for mode in ('constant', 'wrap', 'reflect', 'nearest',
#                     'mirror'):
#            g = f.moving_mean(window_size=3, axis='X',
#                              weights=weights, mode=mode, origin=1)
#
#            for i in range(0, 6):
#                ii = slice(i, i+3)
#                x = (a[:, ii] * weights[ii]).sum(axis=1) / weights[ii].sum()
#                numpy.testing.assert_allclose(x, g.array[:, i])
#        # --- End: for
#
#        # Test 'wrap'
#        for mode in (None, 'wrap'):
#            g = f.moving_mean(window_size=3, axis='X',
#                              weights=weights, mode=mode, origin=1)
#
#            for i, ii in zip([-2, -1], ([0, -2, -1], [0, 1, -1])):
#                x = (a[:, ii] * weights[ii]).sum(axis=1) / weights[ii].sum()
#                numpy.testing.assert_allclose(x, g.array[:, i])
#        # --- End: for
#
#        # ------------------------------------------------------------
#        # Constant
#        # ------------------------------------------------------------
#        for constant in (None, 0):
#            g = f.moving_mean(window_size=3, axis='X',
#                              weights=weights, mode='constant',
#                              cval=constant)
#            for i, ii in zip([0, -1], ([0, 1], [-2, -1])):
#                x = (a[:, ii] * weights[ii]).sum(axis=1) / weights[ii].sum()
#                numpy.testing.assert_allclose(x, g.array[:, i])
#        # --- End: for
#
#        # ------------------------------------------------------------
#        # Weights broadcasting
#        # ------------------------------------------------------------
#        weights = cf.Data(numpy.arange(1, 6.)) / 2
#        g = f.moving_mean(window_size=3, axis='Y', weights=weights)
#
#        with self.assertRaises(ValueError):
#            _ = f.moving_mean(window_size=3, axis='X', weights=weights)
#
#
#        self.assertTrue(len(g.cell_methods) == len(f.cell_methods) + 1)

    def test_Field_moving_window(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        weights = cf.Data([1, 2, 3, 10, 5, 6, 7, 8]) / 2

        f = cf.example_field(0)

        g = f.moving_window('mean', window_size=3, axis='X',
                            inplace=True)
        self.assertIsNone(g)

        with self.assertRaises(ValueError):
            _ = f.moving_window('mean', window_size=3, axis='X',
                                cval=39)

        f = cf.example_field(0)
        a = f.array

        # ------------------------------------------------------------
        # Origin = 0
        # ------------------------------------------------------------
        for method in ('mean', 'sum', 'integral'):
            for mode in ('constant', 'wrap', 'reflect', 'nearest',
                         'mirror'):
                g = f.moving_window(method, window_size=3, axis='X',
                                    weights=weights, mode=mode)

                for i in range(1, 7):
                    if method in ('mean', 'integral'):
                        x = (a[:, i-1:i+2] * weights[i-1:i+2]).sum(axis=1)

                    if method == 'sum':
                        x = a[:, i-1:i+2].sum(axis=1)

                    if method == 'mean':
                        x /= weights[i-1:i+2].sum()

                    numpy.testing.assert_allclose(x, g.array[:, i])
            # --- End: for

            # Test 'wrap'
            for mode in (None, 'wrap'):
                g = f.moving_window(method, window_size=3, axis='X',
                                    weights=weights, mode=mode)

                for i, ii in zip([0, -1], ([0, 1, -1], [0, -2, -1])):
                    if method in ('mean', 'integral'):
                        x = (a[:, ii] * weights[ii]).sum(axis=1)

                    if method == 'sum':
                        x = a[:, ii].sum(axis=1)

                    if method == 'mean':
                        x /= weights[ii].sum()

                    numpy.testing.assert_allclose(x, g.array[:, i])
            # --- End: for

            # ------------------------------------------------------------
            # Origin = 1
            # ------------------------------------------------------------
            for mode in ('constant', 'wrap', 'reflect', 'nearest',
                         'mirror'):
                g = f.moving_window(method, window_size=3, axis='X',
                                    weights=weights, mode=mode, origin=1)

                for i in range(0, 6):
                    ii = slice(i, i+3)

                    if method in ('mean', 'integral'):
                        x = (a[:, ii] * weights[ii]).sum(axis=1)

                    if method == 'sum':
                        x = a[:, ii].sum(axis=1)

                    if method == 'mean':
                        x /= weights[ii].sum()

                    numpy.testing.assert_allclose(x, g.array[:, i])
            # --- End: for

            # Test 'wrap'
            for mode in (None, 'wrap'):
                g = f.moving_window(method, window_size=3, axis='X',
                                    weights=weights, mode=mode,
                                    origin=1)

                for i, ii in zip([-2, -1], ([0, -2, -1], [0, 1, -1])):
                    if method in ('mean', 'integral'):
                        x = (a[:, ii] * weights[ii]).sum(axis=1)

                    if method == 'sum':
                        x = a[:, ii].sum(axis=1)

                    if method == 'mean':
                        x /= weights[ii].sum()

                    numpy.testing.assert_allclose(x, g.array[:, i])
            # --- End: for

            # ------------------------------------------------------------
            # Constant
            # ------------------------------------------------------------
            for constant in (None, 0):
                g = f.moving_window(method, window_size=3, axis='X',
                                    weights=weights, mode='constant',
                                    cval=constant)
                for i, ii in zip([0, -1], ([0, 1], [-2, -1])):
                    if method in ('mean', 'integral'):
                        x = (a[:, ii] * weights[ii]).sum(axis=1)

                    if method == 'sum':
                        x = a[:, ii].sum(axis=1)

                    if method == 'mean':
                        x /= weights[ii].sum()

                    numpy.testing.assert_allclose(x, g.array[:, i])
        # --- End: for

        # ------------------------------------------------------------
        # Weights broadcasting
        # ------------------------------------------------------------
        weights = cf.Data(numpy.arange(1, 6.)) / 2
        g = f.moving_window('mean', window_size=3, axis='Y',
                            weights=weights)

        with self.assertRaises(ValueError):
            _ = f.moving_window('mean', window_size=3, axis='X',
                                weights=weights)

        self.assertTrue(len(g.cell_methods) == len(f.cell_methods) + 1)

    def test_Field_derivative(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        x_min = 0.0
        x_max = 359.0
        dx = 1.0

        x_1d = numpy.arange(x_min, x_max, dx)

        data_1d = x_1d*2.0 + 1.0

        dim_x = cf.DimensionCoordinate(data=cf.Data(x_1d, 's'),
                                       properties={'axis': 'X'})

        f = cf.Field()
        f.set_construct(cf.DomainAxis(size=x_1d.size))
        f.set_construct(dim_x)
        f.set_data(cf.Data(data_1d, 'm'), axes='X')
        f.cyclic('X', period=360.0)

        g = f.derivative('X')
        self.assertTrue((g.array == 2.0).all())

        g = f.derivative('X', one_sided_at_boundary=True)
        self.assertTrue((g.array == 2.0).all())

        g = f.derivative('X', wrap=True)
        self.assertTrue((g.array == 2.0).all())

    def test_Field_convert(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        c = f.convert('grid_latitude')
        self.assertTrue(c.ndim == 1)
        self.assertTrue(c.standard_name == 'grid_latitude')
        self.assertTrue(len(c.dimension_coordinates) == 1)
        self.assertTrue(len(c.auxiliary_coordinates) == 1)
        self.assertTrue(len(c.cell_measures) == 0)
        self.assertTrue(len(c.coordinate_references) == 1)
        self.assertTrue(len(c.domain_ancillaries) == 0)
        self.assertTrue(len(c.field_ancillaries) == 0)
        self.assertTrue(len(c.cell_methods) == 0)

        c = f.convert('latitude')
        self.assertTrue(c.ndim == 2)
        self.assertTrue(c.standard_name == 'latitude')
        self.assertTrue(len(c.dimension_coordinates) == 2)
        self.assertTrue(len(c.auxiliary_coordinates) == 3)
        self.assertTrue(len(c.cell_measures) == 1)
        self.assertTrue(len(c.coordinate_references) == 1)
        self.assertTrue(len(c.domain_ancillaries) == 0)
        self.assertTrue(len(c.field_ancillaries) == 0)
        self.assertTrue(len(c.cell_methods) == 0)

        # Cellsize
        c = f.convert('grid_longitude', cellsize=True)
        self.assertTrue(
            (c.data == [1.,  1.,  1.,  1.,  1.,  1.,  1.,  3.5, 6.]).all())

        with self.assertRaises(ValueError):
            f.convert('qwerty')

    def test_Field_section(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:
            cf.CHUNKSIZE(chunksize)
            f = cf.read(self.filename2)[0][0:100]
            g = f.section(('X', 'Y'))
            self.assertTrue(len(g) == 100,
                            'CHUNKSIZE={}, len(g)={}'.format(chunksize,
                                                             len(g)))
        # --- End: for
        cf.CHUNKSIZE(self.original_chunksize)

    def test_Field_squeeze(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        self.assertIsNone(f.squeeze(inplace=True))
        g = f.copy()
        h = f.copy()
        h.squeeze(inplace=True)
        self.assertTrue(f.equals(h))

        f = self.f.copy()
        self.assertIsNone(f.squeeze(0, inplace=True))

    def test_Field_unsqueeze(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

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
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        for identity in ('auxiliarycoordinate1', 'latitude'):
            key = f.construct_key(identity)
            c = f.construct(identity)

            self.assertTrue(
                f.auxiliary_coordinate(identity).equals(c, verbose=2))
            self.assertEqual(f.auxiliary_coordinate(identity, key=True), key)

            self.assertTrue(f.aux(identity).equals(c, verbose=2))
            self.assertEqual(f.aux(identity, key=True), key)

        self.assertEqual(len(f.auxs()), 3)
        self.assertEqual(len(f.auxs('longitude')), 1)
        self.assertEqual(len(f.auxs('longitude', 'latitude')), 2)

        identities = ['latitude', 'longitude']
        c = f.auxiliary_coordinates(*identities)
        self.assertTrue(f.auxs(*identities).equals(c, verbose=2))
        c = f.auxiliary_coordinates()
        self.assertTrue(f.auxs().equals(c, verbose=2))
        c = f.auxiliary_coordinates(identities[0])
        self.assertTrue(f.auxs(identities[0]).equals(c, verbose=2))

        self.assertIsNone(f.aux('long_name=qwerty:asd', None))
        self.assertEqual(len(f.auxs('long_name=qwerty:asd')), 0)

        with self.assertRaises(Exception):
            f.aux('long_name:qwerty')

        with self.assertRaises(Exception):
            f.auxs('long_name:qwerty')

    def test_Field_coordinate(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        for identity in ('domainaxis2', 'latitude', 'grid_longitude',
                         'auxiliarycoordinate1', 'dimensioncoordinate1', ):
            if identity == 'domainaxis2':
                key = f.dimension_coordinates.filter_by_axis(
                    'and', identity).key()
                c = f.dimension_coordinates.filter_by_axis(
                    'and', identity).value()
            else:
                key = f.construct_key(identity)
                c = f.construct(identity)

            self.assertTrue(f.coordinate(identity).equals(c, verbose=2))
            self.assertTrue(f.coordinate(identity, key=True) == key)

            self.assertTrue(f.coord(identity).equals(c, verbose=2))
            self.assertTrue(f.coord(identity, key=True) == key)

            identities = ['auxiliarycoordinate1', 'dimensioncoordinate1']
            c = f.coordinates(*identities)
            self.assertTrue(f.coords(*identities).equals(c, verbose=2))
            c = f.coordinates()
            self.assertTrue(f.coords().equals(c, verbose=2))
            c = f.coordinates(identities[0])
            self.assertTrue(f.coords(identities[0]).equals(c, verbose=2))

        self.assertIsNone(f.coord('long_name=qwerty:asd', None))
        self.assertTrue(len(f.coords('long_name=qwerty:asd')) == 0)

        with self.assertRaises(Exception):
            f.coord('long_name:qwerty')

        with self.assertRaises(Exception):
            f.coords('long_name:qwerty')

    def test_Field_coordinate_reference(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        for identity in ('coordinatereference1',
                         'key%coordinatereference0',
                         'standard_name:atmosphere_hybrid_height_coordinate',
                         'grid_mapping_name:rotated_latitude_longitude'):
            # 'atmosphere_hybrid_height_coordinate',
            # 'rotated_latitude_longitude'):
            key = f.construct_key(identity)
            c = f.construct(identity)

            self.assertTrue(
                f.coordinate_reference(identity).equals(c, verbose=2))
            self.assertTrue(f.coordinate_reference(identity, key=True) == key)

            self.assertTrue(f.ref(identity).equals(c, verbose=2))
            self.assertTrue(f.ref(identity, key=True) == key)

        key = f.construct_key(
            'standard_name:atmosphere_hybrid_height_coordinate')
        self.assertTrue(f.coordinate_reference(
            'atmosphere_hybrid_height_coordinate', key=True) == key)

        key = f.construct_key('grid_mapping_name:rotated_latitude_longitude')
        self.assertTrue(
            f.coordinate_reference('rotated_latitude_longitude', key=True) ==
            key
        )

        # Delete
        self.assertIsNone(f.del_coordinate_reference('qwerty', default=None))

        self.assertTrue(len(f.coordinate_references) == 2)
        self.assertTrue(len(f.domain_ancillaries) == 3)
        c = f.coordinate_reference(
            'standard_name:atmosphere_hybrid_height_coordinate')
        cr = f.del_coordinate_reference(
            'standard_name:atmosphere_hybrid_height_coordinate')
        self.assertTrue(cr.equals(c, verbose=2))
        self.assertTrue(len(f.coordinate_references) == 1)
        self.assertTrue(len(f.domain_ancillaries) == 0)

        f.del_coordinate_reference(
            'grid_mapping_name:rotated_latitude_longitude')
        self.assertTrue(len(f.coordinate_references) == 0)

        # Set
        f = self.f.copy()
        g = self.f.copy()

        f.del_construct('coordinatereference0')
        f.del_construct('coordinatereference1')

        cr = g.coordinate_reference(
            'grid_mapping_name:rotated_latitude_longitude')
        f.set_coordinate_reference(cr, field=g)
        self.assertTrue(len(f.coordinate_references) == 1)

        cr = g.coordinate_reference(
            'standard_name:atmosphere_hybrid_height_coordinate')
        cr = cr.copy()
        cr.coordinate_conversion.set_domain_ancillary(
            'foo', 'domainancillary99')
        f.set_coordinate_reference(cr, field=g)
        self.assertTrue(len(f.coordinate_references) == 2)
        self.assertTrue(len(f.domain_ancillaries) == 3)

        f.del_construct('coordinatereference0')
        f.del_construct('coordinatereference1')

        cr = g.coordinate_reference(
            'grid_mapping_name:rotated_latitude_longitude')
        f.set_coordinate_reference(cr)
        self.assertTrue(len(f.coordinate_references) == 1)

    def test_Field_dimension_coordinate(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        for identity in ('domainaxis2',
                         'grid_latitude',
                         'X',
                         'dimensioncoordinate1'):
            if identity == 'domainaxis2':
                key = f.dimension_coordinates.filter_by_axis(
                    'and', identity).key()
                c = f.dimension_coordinates.filter_by_axis(
                    'and', identity).value()
            elif identity == 'X':
                key = f.construct_key('grid_longitude')
                c = f.construct('grid_longitude')
            else:
                key = f.construct_key(identity)
                c = f.construct(identity)

            self.assertTrue(
                f.dimension_coordinate(identity).equals(c, verbose=2))
            self.assertTrue(f.dimension_coordinate(identity, key=True) == key)

            self.assertTrue(f.dim(identity).equals(c, verbose=2))
            self.assertTrue(f.dim(identity, key=True) == key)

            identities = ['grid_latitude', 'X']
            c = f.dimension_coordinates(*identities)
            self.assertTrue(f.dims(*identities).equals(c, verbose=2))
            c = f.dimension_coordinates()
            self.assertTrue(f.dims().equals(c, verbose=2))
            c = f.dimension_coordinates(identities[0])
            self.assertTrue(f.dims(identities[0]).equals(c, verbose=2))

        self.assertIsNone(f.dim('long_name=qwerty:asd', None))
        self.assertTrue(len(f.dims('long_name=qwerty:asd')) == 0)

        with self.assertRaises(Exception):
            f.dim('long_name:qwerty')

        with self.assertRaises(Exception):
            f.dims('long_name:qwerty')

    def test_Field_cell_measure(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        for identity in ('measure:area', 'cellmeasure0'):
            key = f.construct_key(identity)
            c = f.construct(identity)

            self.assertTrue(f.cell_measure(identity).equals(c, verbose=2))
            self.assertTrue(f.cell_measure(identity, key=True) == key)

            self.assertTrue(f.measure(identity).equals(c, verbose=2))
            self.assertTrue(f.measure(identity, key=True) == key)

        self.assertTrue(len(f.measures()) == 1)
        self.assertTrue(len(f.measures('measure:area')) == 1)
        self.assertTrue(len(f.measures(*['measure:area'])) == 1)

        self.assertIsNone(f.measure('long_name=qwerty:asd', None))
        self.assertTrue(len(f.measures('long_name=qwerty:asd')) == 0)

        with self.assertRaises(Exception):
            f.measure('long_name:qwerty')

        with self.assertRaises(Exception):
            f.measures('long_name:qwerty')

    def test_Field_cell_method(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        for identity in ('method:mean', 'cellmethod0'):
            key = f.construct_key(identity)
            c = f.construct(identity)

            self.assertTrue(f.cell_method(identity).equals(c, verbose=2))
            self.assertTrue(f.cell_method(identity, key=True) == key)

    def test_Field_domain_ancillary(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        for identity in ('surface_altitude', 'domainancillary0'):
            key = f.construct_key(identity)
            c = f.construct(identity)

            self.assertTrue(
                f.domain_ancillary(identity).equals(c, verbose=2))
            self.assertTrue(f.domain_ancillary(identity, key=True) == key)

            self.assertTrue(f.domain_anc(identity).equals(c, verbose=2))
            self.assertTrue(f.domain_anc(identity, key=True) == key)

        identities = ['surface_altitude', 'key%domainancillary1']
        c = f.domain_ancillaries(*identities)
        self.assertTrue(f.domain_ancs(*identities).equals(c, verbose=2))
        c = f.domain_ancillaries()
        self.assertTrue(f.domain_ancs().equals(c, verbose=2))
        c = f.domain_ancillaries(identities[0])
        self.assertTrue(f.domain_ancs(identities[0]).equals(c, verbose=2))

        self.assertIsNone(f.domain_anc('long_name=qwerty:asd', None))
        self.assertTrue(len(f.domain_ancs('long_name=qwerty:asd')) == 0)

        with self.assertRaises(Exception):
            f.domain_anc('long_name:qwerty')

        with self.assertRaises(Exception):
            f.domain_ancs('long_name:qwerty')

    def test_Field_field_ancillary(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()

        for identity in ('ancillary0', 'fieldancillary0'):
            key = f.construct_key(identity)
            c = f.construct(identity)

            self.assertTrue(
                f.field_ancillary(identity).equals(c, verbose=2))
            self.assertTrue(f.field_ancillary(identity, key=True) == key)

            self.assertTrue(f.field_anc(identity).equals(c, verbose=2))
            self.assertTrue(f.field_anc(identity, key=True) == key)

        self.assertTrue(len(f.field_ancs()) == 4)
        self.assertTrue(len(f.field_ancs('ancillary0')) == 1)
        self.assertTrue(len(f.field_ancs(*['ancillary0', 'ancillary1'])) == 2)

        identities = ['ancillary1', 'ancillary3']
        c = f.field_ancillaries(*identities)
        self.assertTrue(f.field_ancs(*identities).equals(c, verbose=2))
        c = f.field_ancillaries()
        self.assertTrue(f.field_ancs().equals(c, verbose=2))
        c = f.field_ancillaries(identities[0])
        self.assertTrue(f.field_ancs(identities[0]).equals(c, verbose=2))

        self.assertIsNone(f.field_anc('long_name=qwerty:asd', None))
        self.assertTrue(len(f.field_ancs('long_name=qwerty:asd')) == 0)

        with self.assertRaises(Exception):
            f.field_anc('long_name:qwerty')

        with self.assertRaises(Exception):
            f.field_ancs('long_name:qwerty')

    def test_Field_transpose(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)[0]
        f0 = f.copy()

        # Null transpose
        g = f.transpose([0, 1, 2])

        self.assertTrue(f0.equals(g, verbose=2))
        self.assertIsNone(f.transpose([0, 1, 2], inplace=True))
        self.assertTrue(f0.equals(f))

        f = cf.read(self.filename)[0]
        h = f.transpose((1, 2, 0))
#        h0 = h.transpose(
#            (re.compile('^atmos'), 'grid_latitude', 'grid_longitude'))
        h0 = h.transpose((re.compile('^atmos'), 'grid_latitude', 'X'))
        h.transpose((2, 0, 1), inplace=True)

        h.transpose(('grid_longitude', re.compile('^atmos'), 'grid_latitude'),
                    inplace=True)
        h.varray
        h.transpose((re.compile('^atmos'), 'grid_latitude', 'grid_longitude'),
                    inplace=True)

        self.assertTrue(h.equals(h0, verbose=2))
        self.assertTrue((h.array == f.array).all())

        with self.assertRaises(Exception):
            f.transpose('qwerty')

        with self.assertRaises(Exception):
            f.transpose([2, 1])

    def test_Field_domain_axis(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        _ = self.f.domain_axis(1)
        _ = self.f.domain_axis('domainaxis2')

        with self.assertRaises(ValueError):
            self.f.domain_axis(99)

        with self.assertRaises(ValueError):
            self.f.domain_axis('qwerty')

    def test_Field_where(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()
        a = f.array

        f = cf.read(self.filename)[0]
        f0 = f.copy()

        landfrac = f.squeeze()
        landfrac[0:2] = cf.masked
        g = f.where(landfrac >= 54, cf.masked)
        self.assertTrue(g.data.count() == 9*6, g.data.count())

        self.assertTrue(f.equals(f.where(None), verbose=2))
        self.assertIsNone(f.where(None, inplace=True))
        self.assertTrue(f.equals(f0, verbose=2))

        g = f.where(cf.wi(25, 31), -99, 11,
                    construct='grid_longitude')
        g = f.where(cf.wi(25, 31), f*9, f*-7,
                    construct='grid_longitude')
        g = f.where(cf.wi(25, 31), f.copy(), f.squeeze(),
                    construct='grid_longitude')

        g = f.where(cf.wi(-25, 31), -99, 11,
                    construct='latitude')
        g = f.where(cf.wi(-25, 31), f*9, f*-7,
                    construct='latitude')
        g = f.where(cf.wi(-25, 31), f.squeeze(), f.copy(),
                    construct='latitude')

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
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.f.copy()
        g = f.mask_invalid()
        self.assertIsNone(f.mask_invalid(inplace=True))

# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print('')
    unittest.main(verbosity=2)
