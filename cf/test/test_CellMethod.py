import datetime
import inspect
import os
import sys
import unittest

import cf


class CellMethodTest(unittest.TestCase):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'test_file.nc')

    strings = (
        't: mean',
        'time: point',
        'time: maximum',
        'time: sum',
        'lon: maximum time: mean',
        'time: mean lon: maximum',
        'lat: lon: standard_deviation',
        'lon: standard_deviation lat: standard_deviation',
        'time: standard_deviation (interval: 1 day)',
        'area: mean',
        'lon: lat: mean',
        'time: variance (interval: 1 hr comment: sampled '
        'instantaneously)',
        'time: mean',
        'time: mean time: maximum',
        'time: mean within years time: maximum over years',
        'time: mean within days time: maximum within years time: '
        'variance over years',
        'time: standard_deviation (interval: 1 day)',
        'time: standard_deviation (interval: 1 year)',
        'time: standard_deviation (interval: 30 year)',
        'time: standard_deviation (interval: 1.0 year)',
        'time: standard_deviation (interval: 30.0 year)',
        'lat: lon: standard_deviation (interval: 10 km)',
        'lat: lon: standard_deviation '
        '(interval: 10 km interval: 10 km)',
        'lat: lon: standard_deviation '
        '(interval: 0.1 degree_N interval: 0.2 degree_E)',
        'lat: lon: standard_deviation '
        '(interval: 0.123 degree_N interval: 0.234 degree_E)',
        'time: variance (interval: 1 hr comment: '
        'sampled instantaneously)',
        'area: mean where land',
        'area: mean where land_sea',
        'area: mean where sea_ice over sea',
        'area: mean where sea_ice over sea',
        'time: minimum within years time: mean over years',
        'time: sum within years time: mean over years',
        'time: mean within days time: mean over days',
        'time: minimum within days time: sum over days',
        'time: minimum within days time: maximum over days',
        'time: mean within days',
        'time: sum within days time: maximum over days',
    )

    test_only = []
#    test_only = ['test_CellMethod___str__']
#    test_only = ['test_CellMethod_equals']
#    test_only = ['test_CellMethod_equivalent']
#    test_only = ['test_CellMethod_get_set_delete']

    def test_CellMethod__repr__str__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for s in self.strings:
            cms = cf.CellMethod.create(s)
            t = ' '.join(map(str, cms))
            self.assertTrue(t == s, '{!r} != {!r}'.format(t, s))
            for cm in cms:
                _ = repr(cm)

    def test_CellMethod_equals(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for s in self.strings:
            cms = cf.CellMethod.create(s)
            for cm in cms:
                self.assertTrue(cm.equals(cm.copy(), verbose=2))

    def test_CellMethod_equivalent(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for s in self.strings:
            cms = cf.CellMethod.create(s)
            for cm in cms:
                self.assertTrue(cm.equivalent(cm.copy(), verbose=2))
        # --- End: for

        # Intervals
        for s0, s1 in (
                ['lat: lon: mean (interval: 10 km)',
                 'lat: lon: mean (interval: 10 km)'],
                ['lat: lon: mean (interval: 10 km)',
                 'lat: lon: mean (interval: 10 km interval: 10 km)'],
                ['lat: lon: mean (interval: 10 km interval: 10 km)',
                 'lat: lon: mean (interval: 10 km interval: 10 km)'],
                ['lat: lon: mean (interval: 20 km interval: 10 km)',
                 'lat: lon: mean (interval: 20 km interval: 10 km)'],
                ['lat: lon: mean (interval: 20 km interval: 10 km)',
                 'lat: lon: mean (interval: 20000 m interval: 10000 m)'],

                ['lat: lon: mean (interval: 10 km)',
                 'lon: lat: mean (interval: 10 km)'],
                ['lat: lon: mean (interval: 10 km)',
                 'lon: lat: mean (interval: 10 km interval: 10 km)'],
                ['lat: lon: mean (interval: 10 km interval: 10 km)',
                 'lon: lat: mean (interval: 10 km interval: 10 km)'],
                ['lat: lon: mean (interval: 20 km interval: 10 km)',
                 'lon: lat: mean (interval: 10 km interval: 20 km)'],
                ['lat: lon: mean (interval: 20 km interval: 10 km)',
                 'lon: lat: mean (interval: 10000 m interval: 20000 m)'],
        ):
            cms0 = cf.CellMethod.create(s0)
            cms1 = cf.CellMethod.create(s1)

            for cm0, cm1 in zip(cms0, cms1):
                self.assertTrue(
                    cm0.equivalent(cm1, verbose=2),
                    '{0!r} not equivalent to {1!r}'.format(cm0, cm1)
                )
        # --- End: for

    def test_CellMethod_get_set_delete(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        cm0, cm1 = cf.CellMethod.create(
            'time: minimum within days time: sum over years')

        self.assertTrue(cm0.within == 'days')
        self.assertIsNone(cm1.get_qualifier('within', None))
        self.assertIsNone(cm0.get_qualifier('where', None))
        self.assertIsNone(cm0.get_qualifier('over', None))
        self.assertTrue(cm1.over == 'years')
        self.assertTrue(cm0.method == 'minimum')
        self.assertTrue(cm1.method == 'sum')
        self.assertTrue(cm0.axes == ('time',))
        self.assertTrue(cm1.axes == ('time',))

    def test_CellMethod_intervals(self):
        cm = cf.CellMethod.create('lat: mean (interval: 1 hour)')[0]

        self.assertEqual('1 hour', str(cm.intervals[0]))


# --- End: class


if __name__ == "__main__":
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
