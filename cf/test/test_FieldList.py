import datetime
import inspect
import os
import re
import unittest

import numpy

import cf


class FieldTest(unittest.TestCase):
    def setUp(self):
        self.filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'test_file.nc')
        self.filename2 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'test_file2.nc')
        self.f = cf.read(self.filename)

        self.test_only = []
#        self.test_only = ['test_FieldList_select_by_construct']

    def test_FieldList(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)[0]
        g = cf.FieldList(f)

    def test_FieldList__add__iadd__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)

        f = f + f.copy()
        self.assertTrue(len(f) == 2)
        self.assertIsInstance(f, cf.FieldList)

        f += (f[0].copy(),)
        self.assertTrue(len(f) == 3)

        f += [f[0].copy()]
        self.assertTrue(len(f) == 4)
        self.assertIsInstance(f, cf.FieldList)

        f += list(f[0].copy())
        self.assertTrue(len(f) == 5)

        f += f.copy()
        self.assertTrue(len(f) == 10)

        f = f + f.copy()
        self.assertTrue(len(f) == 20)

    def test_FieldList__contains__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)
        f.append(f[0].copy())
        f[1] *= 10
        g = cf.read(self.filename)[0] * 10
        self.assertIn(g, f)
        self.assertNotIn(34.6, f)

    def test_FieldList_close(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)
        self.assertIsNone(f.close())

        _ = repr(f[0])

    def test_FieldList__len__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)

        self.assertTrue(len(cf.FieldList()) == 0)
        self.assertTrue(len(f) == 1)
        f.append(f[0].copy())
        self.assertTrue(len(f) == 2)
        f.extend(f.copy())
        self.assertTrue(len(f) == 4)

    def test_FieldList__mul__imul__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.FieldList()
        f = f * 4
        self.assertTrue(len(f) == 0)
        self.assertIsInstance(f, cf.FieldList)

        f = cf.FieldList()
        f *= 4
        self.assertTrue(len(f) == 0)
        self.assertIsInstance(f, cf.FieldList)

        f = cf.read(self.filename)
        f = f * 4
        self.assertTrue(len(f) == 4)
        self.assertIsInstance(f, cf.FieldList)

        f = cf.read(self.filename)
        f *= 4
        self.assertTrue(len(f) == 4)
        self.assertIsInstance(f, cf.FieldList)

        f = f * 2
        self.assertTrue(len(f) == 8)
        self.assertIsInstance(f, cf.FieldList)

        f *= 3
        self.assertTrue(len(f) == 24)
        self.assertIsInstance(f, cf.FieldList)

    def test_FieldList__repr__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)
        f += f

        _ = repr(f)

    def test_FieldList_append_extend(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Append
        f = cf.FieldList()

        f.append(cf.read(self.filename)[0])
        self.assertTrue(len(f) == 1)
        self.assertIsInstance(f, cf.FieldList)

        f.append(f[0].copy())
        self.assertTrue(len(f) == 2)
        self.assertIsInstance(f, cf.FieldList)

        f.append(f[0].copy())
        self.assertTrue(len(f) == 3)

        # Extend
        f = cf.FieldList()

        f.extend(cf.read(self.filename))
        self.assertTrue(len(f) == 1)
        self.assertIsInstance(f, cf.FieldList)

        f.extend(f.copy())
        self.assertTrue(len(f) == 2)
        self.assertIsInstance(f, cf.FieldList)

        f.extend(f.copy())
        self.assertTrue(len(f) == 4)

    def test_FieldList_copy(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)
        f.append(f[0].copy())
        g = f.copy()
        self.assertTrue(f.equals(f, verbose=2))
        self.assertTrue(f.equals(g, verbose=2))

    def test_FieldList__getslice__(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)
        f.append(f[0].copy())

        _ = f[0:1]
        _ = f[1:2]
        _ = f[:1]
        _ = f[1:]

    def test_FieldList_count(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)

        self.assertTrue(f.count(f[0]) == 1)

        f *= 7
        self.assertTrue(f.count(f[0]) == 7)

        f[3] = f[0] * 99
        f[5] = f[0] * 99
        self.assertTrue(f.count(f[0]) == 5)
        self.assertTrue(f.count(f[3]) == 2)

    def test_FieldList_equals(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)
        g = f.copy()
        self.assertTrue(f.equals(f, verbose=2))
        self.assertTrue(f.equals(g, verbose=2))

        f += g.copy()
        self.assertTrue(f.equals(f, verbose=2))
        self.assertTrue(len(f) == 2)
        g = f.copy()
        self.assertTrue(f.equals(g, verbose=2))
        self.assertTrue(f.equals(g, unordered=True, verbose=2))

        h = cf.read(self.filename2)
        f.extend(h)
        self.assertTrue(f.equals(f, verbose=2))
        self.assertTrue(f.equals(f.copy(), verbose=2))

        g = f.copy()[::-1]
        self.assertFalse(f.equals(g))
        self.assertTrue(f.equals(g, unordered=True, verbose=2))

        g = g[:-1]
        self.assertFalse(f.equals(g))
        self.assertFalse(f.equals(g, unordered=True))

        g.append(g[0])
        self.assertFalse(f.equals(g))
        self.assertFalse(f.equals(g, unordered=True))

        h *= 3
        self.assertFalse(f.equals(h))
        self.assertFalse(f.equals(h, unordered=True))

    def test_FieldList_insert_pop_remove(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Insert
        f = cf.read(self.filename)
        g = f[0].copy()

        f.insert(0, g.copy())
        self.assertTrue(len(f) == 2)
        self.assertIsInstance(f, cf.FieldList)

        g = g + 10
        f.insert(-1, g)
        self.assertTrue(len(f) == 3)
        self.assertTrue(f[0].maximum() == (f[1].maximum() - 10))
        self.assertTrue(isinstance(f, cf.FieldList))

        # Pop
        f = cf.read(self.filename)
        g = f[0]
        h = f[0] + 10
        f.append(h)

        z = f.pop(0)
        self.assertIs(z, g)
        self.assertTrue(len(f) == 1)
        self.assertIsInstance(f, cf.FieldList)

        z = f.pop(-1)
        self.assertIs(z, h)
        self.assertTrue(len(f) == 0)
        self.assertIsInstance(f, cf.FieldList)

        # Remove
        f = cf.read(self.filename)
        g = f[0].copy()
        g = g + 10

        f.append(g)
        self.assertTrue(len(f) == 2)

        f.remove(g)
        self.assertTrue(len(f) == 1)
        self.assertIsInstance(f, cf.FieldList)

        with self.assertRaises(Exception):
            f.remove(f[0]*-99)

        f.remove(f[0].copy())
        self.assertTrue(len(f) == 0)
        self.assertIsInstance(f, cf.FieldList)

    def test_FieldList_reverse(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)
        g = f[0]
        h = f[0] + 10
        f.append(h)

        self.assertIs(g, f[0])
        self.assertIs(h, f[1])

        f.reverse()
        self.assertIsInstance(f, cf.FieldList)
        self.assertTrue(len(f) == 2)
        self.assertIs(g, f[1])
        self.assertIs(h, f[0])

    def test_FieldList_select(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)

        g = f('not this one')
        self.assertIsInstance(g, cf.FieldList)
        self.assertTrue(len(g) == 0)

        g = f('eastward_wind')
        self.assertIsInstance(g, cf.FieldList)
        self.assertTrue(len(g) == 1, len(g))

        g = f(re.compile('^eastw'))
        self.assertIsInstance(g, cf.FieldList)
        self.assertTrue(len(g) == 1, len(g))

        f *= 9
        f[4] = f[0].copy()
        f[4].standard_name = 'this one'
        f[6] = f[0].copy()
        f[6].standard_name = 'this one'

        g = f(re.compile('^eastw'))
        self.assertIsInstance(g, cf.FieldList)
        self.assertTrue(len(g) == 7, len(g))

        g = f('this one')
        self.assertIsInstance(g, cf.FieldList)
        self.assertTrue(len(g) == 2)

        # select_by_Units
        f[1] = f[1].override_units(cf.Units('watt'))
        f[3] = f[3].override_units(cf.Units('mile hour-1'))

        self.assertTrue(len(f.select_by_units()) == 9)
        self.assertTrue(len(f.select_by_units(cf.Units('m s-1'))) == 7)
        self.assertTrue(
            len(f.select_by_units(cf.Units('m s-1'), exact=False)) == 8)
        self.assertTrue(len(f.select_by_units('m s-1')) == 7)
        self.assertTrue(len(f.select_by_units('m s-1', exact=False)) == 8)
        self.assertTrue(len(f.select_by_units(re.compile('^mile|watt'))) == 2)

        # select_by_Units
        f[1] = f[1].override_units(cf.Units('watt'))
        f[3] = f[3].override_units(cf.Units('mile hour-1'))

        self.assertTrue(len(f.select_by_units()) == 9)
        self.assertTrue(len(f.select_by_units(cf.Units('m s-1'))) == 7)
        self.assertTrue(
            len(f.select_by_units(cf.Units('m s-1'), exact=False)) == 8)
        self.assertTrue(len(f.select_by_units('m s-1')) == 7)
        self.assertTrue(len(f.select_by_units('m s-1', exact=False)) == 8)
        self.assertTrue(len(f.select_by_units(re.compile('^mile|watt'))) == 2)

        self.assertTrue(len(f.select_by_units('long_name=qwery:asd')) == 0)

        # select_by_ncvar
        f[1].nc_set_variable('qwerty')
        f[4].nc_set_variable('eastward_wind2')

        self.assertTrue(len(f.select_by_ncvar()) == 9)
        self.assertTrue(len(f.select_by_ncvar('qwerty')) == 1)
        self.assertTrue(len(f.select_by_ncvar('eastward_wind')) == 7)
        self.assertTrue(len(f.select_by_ncvar('eastward_wind2')) == 1)
        self.assertTrue(len(f.select_by_ncvar(re.compile('^east'))) == 8)

    def test_FieldList_select_by_construct(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read([self.filename, self.filename2])

        g = f.select_by_construct()
        self.assertTrue(len(g) == 2)

        g = f.select_by_construct('latitude')
        self.assertTrue(len(g) == 2)

        g = f.select_by_construct('latitude', 'longitude')
        self.assertTrue(len(g) == 2)

        g = f.select_by_construct('latitude', 'time')
        self.assertTrue(len(g) == 1)

        g = f.select_by_construct('latitude', 'time', OR=False)
        self.assertTrue(len(g) == 1)

        g = f.select_by_construct('latitude', 'time', OR=True)
        self.assertTrue(len(g) == 2)

        g = f.select_by_construct(longitude=cf.gt(0))
        self.assertTrue(len(g) == 2)

        g = f.select_by_construct(
            longitude=cf.gt(0), time=cf.le(cf.dt('2008-12-01')))
        self.assertTrue(len(g) == 1)

        g = f.select_by_construct(
            longitude=cf.gt(0), time=cf.le(cf.dt('2008-12-01')), OR=True)
        self.assertTrue(len(g) == 2)

        g = f.select_by_construct(
            'longitude', longitude=cf.gt(0), time=cf.le(cf.dt('2008-12-01')),
            OR=True
        )
        self.assertTrue(len(g) == 2)

        g = f.select_by_construct(
            'latitude', longitude=cf.gt(0), time=cf.le(cf.dt('2008-12-01')),
            OR=True
        )
        self.assertTrue(len(g) == 2)

        g = f.select_by_construct(
            'time', longitude=cf.gt(0), time=cf.le(cf.dt('2008-12-01')),
            OR=True
        )
        self.assertTrue(len(g) == 2)

        g = f.select_by_construct(
            'time', longitude=cf.gt(0), time=cf.le(cf.dt('2008-12-01')),
            OR=False
        )
        self.assertTrue(len(g) == 1)

        g = f.select_by_construct(
            'time', longitude=cf.gt(0), time=cf.le(cf.dt('2008-12-01')),
            OR=False
        )
        self.assertTrue(len(g) == 1)

        g = f.select_by_construct('qwerty')
        self.assertTrue(len(g) == 0)

        g = f.select_by_construct('qwerty', 'latitude')
        self.assertTrue(len(g) == 0)

        g = f.select_by_construct('qwerty', 'latitude', OR=True)
        self.assertTrue(len(g) == 2)

        g = f.select_by_construct('qwerty', 'time', 'longitude')
        self.assertTrue(len(g) == 0)

        g = f.select_by_construct('qwerty', 'time', 'longitude', OR=True)
        self.assertTrue(len(g) == 2)

        g = f.select_by_construct(longitude=cf.gt(70))
        self.assertTrue(len(g) == 1)

        g = f.select_by_construct(
            longitude=cf.gt(0), time=cf.le(cf.dt('1999-12-01')))
        self.assertTrue(len(g) == 0)

    def test_FieldList_select_field(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename)

        with self.assertRaises(Exception):
            _ = f.select_field('not this one')

        self.assertIsNone(f.select_field('not this one', None))

        g = f.select_field('eastward_wind')
        self.assertIsInstance(g, cf.Field)

        g = f.select_field(re.compile('^eastw'))
        self.assertIsInstance(g, cf.Field)

        with self.assertRaises(Exception):
            g = f.select_field(re.compile('^QWERTY'))

    def test_FieldList_concatenate(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename2)[0]

        g = cf.FieldList([f[0], f[1:456], f[456:]])

        h = g.concatenate(axis=0)
        self.assertTrue(f.equals(h, verbose=2))

        h = g.concatenate(axis=0, _preserve=False)
        self.assertTrue(f.equals(h, verbose=2))

    def test_FieldList_index(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename2)[0]

        a, b, c = [f[0], f[1:456], f[456:]]
        g = cf.FieldList([a, b, c])

        self.assertTrue(g.index(a) == 0)
        self.assertTrue(g.index(a, start=0) == 0)
        self.assertTrue(g.index(a, stop=1) == 0)
        self.assertTrue(g.index(a, stop=-2) == 0)
        self.assertTrue(g.index(a, stop=2) == 0)
        self.assertTrue(g.index(b) == 1)
        self.assertTrue(g.index(b, start=0) == 1)
        self.assertTrue(g.index(b, start=1, stop=2) == 1)
        self.assertTrue(g.index(c) == 2)
        self.assertTrue(g.index(c, start=0) == 2)
        self.assertTrue(g.index(c, start=1) == 2)
        self.assertTrue(g.index(c, start=2) == 2)
        self.assertTrue(g.index(c, start=-1) == 2)

        with self.assertRaises(Exception):
            _ = g.index(f)

        with self.assertRaises(Exception):
            _ = g.index(a, start=1)


# --- End: class

if __name__ == '__main__':
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
