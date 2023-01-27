import datetime
import faulthandler
import os
import re
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class FieldTest(unittest.TestCase):
    filename2 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file2.nc"
    )

    f2 = cf.read(filename2)

    x = cf.example_field(1)

    test_only = []

    def test_FieldList(self):
        cf.FieldList(self.x)
        cf.FieldList([self.x])

    def test_FieldList__add__iadd__(self):
        f = cf.FieldList(self.x)

        f = f + f.copy()
        self.assertEqual(len(f), 2)
        self.assertIsInstance(f, cf.FieldList)

        f += (f[0].copy(),)
        self.assertEqual(len(f), 3)

        f += [f[0].copy()]
        self.assertEqual(len(f), 4)
        self.assertIsInstance(f, cf.FieldList)

        f += list(f[0].copy())
        self.assertEqual(len(f), 5)

        f += f.copy()
        self.assertEqual(len(f), 10)

        f = f + f.copy()
        self.assertEqual(len(f), 20)

    def test_FieldList__contains__(self):
        f = cf.FieldList(self.x)

        f.append(self.x.copy())
        f[1] *= 10
        g = self.x * 10
        self.assertIn(g, f)
        self.assertNotIn(34.6, f)

    def test_FieldList__len__(self):
        f = cf.FieldList(self.x)

        self.assertEqual(len(cf.FieldList()), 0)
        self.assertEqual(len(f), 1)
        f.append(f[0])
        self.assertEqual(len(f), 2)
        f.extend(f)
        self.assertEqual(len(f), 4)

    def test_FieldList__mul__imul__(self):
        f = cf.FieldList()
        f = f * 4
        self.assertEqual(len(f), 0)
        self.assertIsInstance(f, cf.FieldList)

        f = cf.FieldList()
        f *= 4
        self.assertEqual(len(f), 0)
        self.assertIsInstance(f, cf.FieldList)

        f = cf.FieldList(self.x) * 4
        self.assertEqual(len(f), 4)
        self.assertIsInstance(f, cf.FieldList)

        f = cf.FieldList(self.x)
        f *= 4
        self.assertEqual(len(f), 4)
        self.assertIsInstance(f, cf.FieldList)

        f = f * 2
        self.assertEqual(len(f), 8)
        self.assertIsInstance(f, cf.FieldList)

        f *= 3
        self.assertEqual(len(f), 24)
        self.assertIsInstance(f, cf.FieldList)

    def test_FieldList__repr__(self):
        f = cf.FieldList(self.x)
        f += f
        repr(f)

    def test_FieldList_append_extend(self):
        # Append
        f = cf.FieldList()

        f.append(self.x)
        self.assertEqual(len(f), 1)
        self.assertIsInstance(f, cf.FieldList)

        f.append(f[0])
        self.assertEqual(len(f), 2)
        self.assertIsInstance(f, cf.FieldList)

        f.append(f[0])
        self.assertEqual(len(f), 3)

        # Extend
        f = cf.FieldList()

        f.extend([self.x])
        self.assertEqual(len(f), 1)
        self.assertIsInstance(f, cf.FieldList)

        f.extend(f)
        self.assertEqual(len(f), 2)
        self.assertIsInstance(f, cf.FieldList)

        f.extend(f)
        self.assertEqual(len(f), 4)

    def test_FieldList_copy(self):
        f = cf.FieldList(self.x)

        f.append(f[0].copy())
        g = f.copy()
        self.assertTrue(f.equals(f, verbose=2))
        self.assertTrue(f.equals(g, verbose=2))

    def test_FieldList__getslice__(self):
        f = cf.FieldList(self.x)

        f.append(f[0])

        f[0:1]
        f[1:2]
        f[:1]
        f[1:]

    def test_FieldList_count(self):
        f = cf.FieldList(self.x)

        self.assertEqual(f.count(f[0]), 1)

        f = f * 7
        self.assertEqual(f.count(f[0]), 7)

        f[3] = f[0] * 99
        f[5] = f[0] * 99
        self.assertEqual(f.count(f[0]), 5)
        self.assertEqual(f.count(f[3]), 2)

    def test_FieldList_equals(self):
        f = cf.FieldList(self.x)

        g = f.copy()
        self.assertTrue(f.equals(f, verbose=2))
        self.assertTrue(f.equals(g, verbose=2))

        f += g.copy()
        self.assertTrue(f.equals(f, verbose=2))
        self.assertEqual(len(f), 2)
        g = f.copy()
        self.assertTrue(f.equals(g, verbose=2))
        self.assertTrue(f.equals(g, unordered=True))

        h = self.f2.copy()
        for x in h:
            x.standard_name = "eastward_wind"

        f.extend(h)
        self.assertTrue(f.equals(f, verbose=2))
        self.assertTrue(f.equals(f.copy()))

        g = f.copy()[::-1]
        self.assertFalse(f.equals(g))
        self.assertTrue(f.equals(g, unordered=True))

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
        # Insert
        f = cf.FieldList(self.x)
        g = f[0].copy()

        f.insert(0, g)
        self.assertEqual(len(f), 2)
        self.assertIsInstance(f, cf.FieldList)

        g = g + 10
        f.insert(-1, g)
        self.assertEqual(len(f), 3)
        self.assertEqual(f[0].maximum(), (f[1].maximum() - 10))
        self.assertIsInstance(f, cf.FieldList)

        # Pop
        f = cf.FieldList(self.x)
        g = f[0]
        h = f[0] + 10
        f.append(h)

        z = f.pop(0)
        self.assertIs(z, g)
        self.assertEqual(len(f), 1)
        self.assertIsInstance(f, cf.FieldList)

        z = f.pop(-1)
        self.assertIs(z, h)
        self.assertEqual(len(f), 0)
        self.assertIsInstance(f, cf.FieldList)

        # Remove
        f = cf.FieldList(self.x)
        g = f[0] + 10

        f.append(g)
        self.assertEqual(len(f), 2)

        f.remove(g)
        self.assertEqual(len(f), 1)
        self.assertIsInstance(f, cf.FieldList)

        with self.assertRaises(Exception):
            f.remove(f[0] * -99)

        f.remove(f[0].copy())
        self.assertEqual(len(f), 0)
        self.assertIsInstance(f, cf.FieldList)

    def test_FieldList_reverse(self):
        f = cf.FieldList(self.x)

        g = f[0]
        h = f[0] + 10
        f.append(h)

        self.assertIs(g, f[0])
        self.assertIs(h, f[1])

        f.reverse()
        self.assertIsInstance(f, cf.FieldList)
        self.assertEqual(len(f), 2)
        self.assertIs(g, f[1])
        self.assertIs(h, f[0])

    def test_FieldList_select(self):
        f = cf.FieldList(self.x)

        g = f("not this one")
        self.assertIsInstance(g, cf.FieldList)
        self.assertEqual(len(g), 0)

        g = f("air_temperature")
        self.assertIsInstance(g, cf.FieldList)
        self.assertEqual(len(g), 1, len(g))

        g = f(re.compile("^air"))
        self.assertIsInstance(g, cf.FieldList)
        self.assertEqual(len(g), 1, len(g))

        f *= 9
        f[4] = f[0].copy()
        f[4].standard_name = "this one"
        f[6] = f[0].copy()
        f[6].standard_name = "this one"

        g = f(re.compile("^air"))
        self.assertIsInstance(g, cf.FieldList)
        self.assertEqual(len(g), 7, len(g))

        g = f("this one")
        self.assertIsInstance(g, cf.FieldList)
        self.assertEqual(len(g), 2)

        # select_by_Units
        f[1] = f[1].override_units(cf.Units("watt"))
        f[3] = f[3].override_units(cf.Units("K @ 273.15"))

        self.assertEqual(len(f.select_by_units()), 9)
        self.assertEqual(len(f.select_by_units(cf.Units("K"))), 7)
        self.assertEqual(len(f.select_by_units(cf.Units("K"), exact=False)), 8)
        self.assertEqual(len(f.select_by_units("K")), 7)
        self.assertEqual(len(f.select_by_units("K", exact=False)), 8)
        self.assertEqual(len(f.select_by_units(re.compile("^K @|watt"))), 2)

        self.assertEqual(len(f.select_by_units("long_name=qwery:asd")), 0)

        # select_by_ncvar
        for a in f:
            a.nc_set_variable("ta")

        f[1].nc_set_variable("qwerty")
        f[4].nc_set_variable("ta2")

        self.assertEqual(len(f.select_by_ncvar()), 9)
        self.assertEqual(len(f.select_by_ncvar("qwerty")), 1)
        self.assertEqual(len(f.select_by_ncvar("ta")), 7)
        self.assertEqual(len(f.select_by_ncvar("ta2")), 1)
        self.assertEqual(len(f.select_by_ncvar(re.compile("^ta"))), 8)

    def test_FieldList_select_by_construct(self):
        x = self.x.copy()
        x.del_construct("time")

        f = cf.FieldList(x)
        f.extend(self.f2.copy())

        g = f.select_by_construct()
        self.assertEqual(len(g), 2)

        g = f.select_by_construct("latitude")
        self.assertEqual(len(g), 2)

        g = f.select_by_construct("latitude", "longitude")
        self.assertEqual(len(g), 2)

        g = f.select_by_construct("latitude", "time")
        self.assertEqual(len(g), 1)

        g = f.select_by_construct("latitude", "time", OR=False)
        self.assertEqual(len(g), 1)

        g = f.select_by_construct("latitude", "time", OR=True)
        self.assertEqual(len(g), 2)

        g = f.select_by_construct(longitude=cf.gt(0))
        self.assertEqual(len(g), 2)

        g = f.select_by_construct(
            longitude=cf.gt(0), time=cf.le(cf.dt("2008-12-01"))
        )
        self.assertEqual(len(g), 1)

        g = f.select_by_construct(
            longitude=cf.gt(0), time=cf.le(cf.dt("2008-12-01")), OR=True
        )
        self.assertEqual(len(g), 2)

        g = f.select_by_construct(
            "longitude",
            longitude=cf.gt(0),
            time=cf.le(cf.dt("2008-12-01")),
            OR=True,
        )
        self.assertEqual(len(g), 2)

        g = f.select_by_construct(
            "latitude",
            longitude=cf.gt(0),
            time=cf.le(cf.dt("2008-12-01")),
            OR=True,
        )
        self.assertEqual(len(g), 2)

        g = f.select_by_construct(
            "time",
            longitude=cf.gt(0),
            time=cf.le(cf.dt("2008-12-01")),
            OR=True,
        )
        self.assertEqual(len(g), 2)

        g = f.select_by_construct(
            "time",
            longitude=cf.gt(0),
            time=cf.le(cf.dt("2008-12-01")),
            OR=False,
        )
        self.assertEqual(len(g), 1)

        g = f.select_by_construct(
            "time",
            longitude=cf.gt(0),
            time=cf.le(cf.dt("2008-12-01")),
            OR=False,
        )
        self.assertEqual(len(g), 1)

        g = f.select_by_construct("qwerty")
        self.assertEqual(len(g), 0)

        g = f.select_by_construct("qwerty", "latitude")
        self.assertEqual(len(g), 0)

        g = f.select_by_construct("qwerty", "latitude", OR=True)
        self.assertEqual(len(g), 2)

        g = f.select_by_construct("qwerty", "time", "longitude")
        self.assertEqual(len(g), 0)

        g = f.select_by_construct("qwerty", "time", "longitude", OR=True)
        self.assertEqual(len(g), 2)

        g = f.select_by_construct(longitude=cf.gt(7.6))
        self.assertEqual(len(g), 1)

        g = f.select_by_construct(
            longitude=cf.gt(0), time=cf.le(cf.dt("1999-12-01"))
        )
        self.assertEqual(len(g), 0)

    def test_FieldList_select_field(self):
        f = cf.FieldList(self.x)

        with self.assertRaises(Exception):
            f.select_field("not this one")

        self.assertIsNone(f.select_field("not this one", default=None))

        g = f.select_field("air_temperature")
        self.assertIsInstance(g, cf.Field)

        g = f.select_field(re.compile("^air_temp"))
        self.assertIsInstance(g, cf.Field)

        with self.assertRaises(Exception):
            g = f.select_field(re.compile("^QWERTY"))

    def test_FieldList_concatenate(self):
        f = self.f2[0]

        g = cf.FieldList([f[0], f[1:456], f[456:]])

        h = g.concatenate(axis=0)
        self.assertTrue(f.equals(h, verbose=2))

        h = g.concatenate(axis=0, cull_graph=False)
        self.assertTrue(f.equals(h, verbose=2))

    def test_FieldList_index(self):
        f = self.f2[0]

        a, b, c = [f[0], f[1:456], f[456:]]
        g = cf.FieldList([a, b, c])

        self.assertEqual(g.index(a), 0)
        self.assertEqual(g.index(a, start=0), 0)
        self.assertEqual(g.index(a, stop=1), 0)
        self.assertEqual(g.index(a, stop=-2), 0)
        self.assertEqual(g.index(a, stop=2), 0)
        self.assertEqual(g.index(b), 1)
        self.assertEqual(g.index(b, start=0), 1)
        self.assertEqual(g.index(b, start=1, stop=2), 1)
        self.assertEqual(g.index(c), 2)
        self.assertEqual(g.index(c, start=0), 2)
        self.assertEqual(g.index(c, start=1), 2)
        self.assertEqual(g.index(c, start=2), 2)
        self.assertEqual(g.index(c, start=-1), 2)

        with self.assertRaises(Exception):
            g.index(f)

        with self.assertRaises(Exception):
            g.index(a, start=1)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
