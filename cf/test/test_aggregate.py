import datetime
import faulthandler
import os
import unittest
import warnings

faulthandler.enable()  # to debug seg faults and timeouts

import cf

# To facilitate the testing of logging outputs (see test_aggregate_verbosity)
log_name = __name__
logger = cf.logging.getLogger(log_name)


class aggregateTest(unittest.TestCase):
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file.nc"
    )
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file.nc")
    file2 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "file2.nc"
    )

    def test_basic_aggregate(self):
        f = cf.read(self.filename, squeeze=True)[0]

        g = cf.FieldList(f[0])
        g.append(f[1:3])
        g.append(f[3])
        g[-1].flip(0, inplace=True)
        g.append(f[4:7])
        g[-1].flip(0, inplace=True)
        g.extend([f[i] for i in range(7, f.shape[0])])

        g0 = g.copy()
        self.assertTrue(g.equals(g0, verbose=-1), "g != g0")

        with warnings.catch_warnings():
            # Suppress noise throughout the test fixture from:
            #
            #   ~/cf-python/cf/__init__.py:1459: FutureWarning: elementwise
            #   comparison failed; returning scalar instead, but in the
            #   future will perform elementwise comparison
            #
            # TODO: it is not clear where the above emerges from, e.g.
            # since __init__ file ref'd does not have that many lines.
            # It seems like this warning arises from NumPy comparisons
            # done at some point in (only) some aggregate calls (see e.g:
            # https://github.com/numpy/numpy/issues/6784).
            warnings.filterwarnings("ignore", category=FutureWarning)
            h = cf.aggregate(g, verbose=2)

        self.assertEqual(len(h), 1)

        self.assertEqual(
            h[0].shape,
            (10, 9),
            "h[0].shape = " + repr(h[0].shape) + " != (10, 9)",
        )

        self.assertTrue(
            g.equals(g0, verbose=2), "g != itself after aggregation"
        )

        self.assertTrue(h[0].equals(f, verbose=2), "h[0] != f")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            i = cf.aggregate(g, verbose=2)

        self.assertTrue(
            i.equals(h, verbose=2), "The second aggregation != the first"
        )

        self.assertTrue(
            g.equals(g0, verbose=2),
            "g != itself after the second aggregation",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            i = cf.aggregate(g, verbose=2, axes="grid_latitude")

        self.assertTrue(
            i.equals(h, verbose=2), "The third aggregation != the first"
        )

        self.assertTrue(
            g.equals(g0, verbose=2),
            "g !=itself after the third aggregation",
        )

        self.assertEqual(
            i[0].shape, (10, 9), "i[0].shape is " + repr(i[0].shape)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            i = cf.aggregate(
                g,
                verbose=2,
                axes="grid_latitude",
                donotchecknonaggregatingaxes=1,
            )

        self.assertTrue(
            i.equals(h, verbose=2), "The fourth aggregation != the first"
        )

        self.assertTrue(
            g.equals(g0, verbose=2),
            "g != itself after the fourth aggregation",
        )

        self.assertEqual(
            i[0].shape, (10, 9), "i[0].shape is " + repr(i[0].shape)
        )

        q, t = cf.read(self.file)
        c = cf.read(self.file2)[0]

        d = cf.aggregate([c, t], verbose=1, relaxed_identities=True)
        e = cf.aggregate([t, c], verbose=1, relaxed_identities=True)

        self.assertEqual(len(d), 1)
        self.assertEqual(len(e), 1)
        self.assertEqual(d[0].shape, (3,) + t.shape)
        self.assertTrue(d[0].equals(e[0], verbose=2))

        x = cf.read(["file.nc", "file2.nc"], aggregate=False)
        self.assertEqual(len(x), 3)

        x = cf.read(
            ["file.nc", "file2.nc"], aggregate={"relaxed_identities": True}
        )
        self.assertEqual(len(x), 2)

        del t.standard_name
        del c.standard_name
        x = cf.aggregate([c, t], verbose=1)
        self.assertEqual(len(x), 2)

        t.long_name = "qwerty"
        c.long_name = "qwerty"
        x = cf.aggregate([c, t], field_identity="long_name")
        self.assertEqual(len(x), 1)

    def test_aggregate_exist_equal_ignore_opts(self):
        # TODO: extend the option-checking coverage so all options and all
        # reasonable combinations of them are tested. For now, this is
        # testing options that previously errored due to a bug.
        f = cf.read(self.filename, squeeze=True)[0]

        # Use f as-is: simple test that aggregate works and does not
        # change anything with the given options:
        g = cf.aggregate(f, exist_all=True)[0]
        self.assertEqual(g, f)
        h = cf.aggregate(f, equal_all=True)[0]
        self.assertEqual(h, f)

        with self.assertRaises(ValueError):  # contradictory options
            cf.aggregate(f, exist_all=True, equal_all=True)

    def test_aggregate_verbosity(self):
        f0 = cf.example_field(0)
        f1 = cf.example_field(1)

        detail_header = "DETAIL:cf.aggregate:STRUCTURAL SIGNATURE:"
        debug_header = "DEBUG:cf.aggregate:COMPLETE AGGREGATION METADATA:"

        # 'DEBUG' (-1) verbosity should output both log message headers...
        with self.assertLogs(level="NOTSET") as catch:
            cf.aggregate([f0, f1], verbose=-1)
            for header in (detail_header, debug_header):
                self.assertTrue(
                    any(
                        log_item.startswith(header)
                        for log_item in catch.output
                    ),
                    f"No log entry begins with '{header}'",
                )

        # ...but with 'DETAIL' (3), should get only the detail-level one.
        with self.assertLogs(level="NOTSET") as catch:
            cf.aggregate([f0, f1], verbose=3)
            self.assertTrue(
                any(
                    log_item.startswith(detail_header)
                    for log_item in catch.output
                ),
                f"No log entry begins with '{detail_header}'",
            )
            self.assertFalse(
                any(
                    log_item.startswith(debug_header)
                    for log_item in catch.output
                ),
                f"A log entry begins with '{debug_header}' but should not",
            )

        # and neither should emerge at the 'WARNING' (1) level.
        with self.assertLogs(level="NOTSET") as catch:
            logger.warning(
                "Dummy message to log something at warning level so that "
                "'assertLog' does not error when no logs messages emerge."
            )
            # Note: can use assertNoLogs in Python 3.10 to avoid this, see:
            # https://bugs.python.org/issue39385

            cf.aggregate([f0, f1], verbose=1)
            for header in (detail_header, debug_header):
                self.assertFalse(
                    any(
                        log_item.startswith(header)
                        for log_item in catch.output
                    ),
                    f"A log entry begins with '{header}' but should not",
                )

    def test_aggregate_bad_units(self):
        f = cf.read(self.filename, squeeze=True)[0]

        g = cf.FieldList(f[0])
        g.append(f[1:])

        h = cf.aggregate(g)
        self.assertEqual(len(h), 1)

        g[0].override_units(cf.Units("apples!"), inplace=True)
        g[1].override_units(cf.Units("oranges!"), inplace=True)

        h = cf.aggregate(g)
        self.assertEqual(len(h), 2)

    def test_aggregate_domain(self):
        f = cf.example_field(0)
        g = f[0:3].domain
        h = f[3:].domain

        x = cf.aggregate([g, h])

        self.assertEqual(len(x), 1, x)

    def test_aggregate_dimension(self):
        """Test the promotion of property to axis."""
        f = cf.example_field(0)
        g = f.copy()

        f.set_property("sim", "r1i1p1f1")
        g.set_property("sim", "r2i1p1f1")

        self.assertFalse(len(f.auxiliary_coordinates()))

        a = cf.aggregate([f, g], dimension="sim")
        self.assertEqual(len(a), 1)

        a = a[0]
        self.assertEqual(len(a.auxiliary_coordinates()), 1)

    def test_aggregate_keyword_consistency(self):
        """Test acceptable keyword combinations."""
        f = cf.example_field(0)
        a = cf.aggregate(
            [f[:2], f[2:]], relaxed_identities=True, ncvar_identities=True
        )
        self.assertEqual(len(a), 1)

    def test_aggregate_equal_equal_all(self):
        f = cf.example_field(0)
        f.set_property("foo", "bar")
        a, b, c, d = f[0], f[1], f[2:4], f[4]

        c.set_property("foo", "baz")
        d.set_property("foo", "baz")

        g = cf.aggregate([a, b, c, d])
        self.assertEqual(len(g), 1)

        g = cf.aggregate([a, b, c, d], equal=["foo"])
        self.assertEqual(len(g), 2)

        g = cf.aggregate([a, b, c, d], equal_all=True)
        self.assertEqual(len(g), 2)

        d.del_property("foo")

        g = cf.aggregate([a, b, c, d])
        self.assertEqual(len(g), 1)

        g = cf.aggregate([a, b, c, d], equal=["foo"])
        self.assertEqual(len(g), 3)

        g = cf.aggregate([a, b, c, d], equal_all=True)
        self.assertEqual(len(g), 3)

    def test_aggregate_exist_exist_all(self):
        f = cf.example_field(0)
        f.set_property("foo", "bar")
        a, b, c, d = f[0], f[1], f[2:4], f[4]

        c.set_property("foo", "baz")
        d.set_property("foo", "baz")

        g = cf.aggregate([a, b, c, d])
        self.assertEqual(len(g), 1)

        g = cf.aggregate([a, b, c, d], exist=["foo"])
        self.assertEqual(len(g), 1)

        g = cf.aggregate([a, b, c, d], exist_all=True)
        self.assertEqual(len(g), 1)

        d.del_property("foo")

        g = cf.aggregate([a, b, c, d])
        self.assertEqual(len(g), 1)

        g = cf.aggregate([a, b, c, d], exist=["foo"])
        self.assertEqual(len(g), 2)

        g = cf.aggregate([a, b, c, d], exist_all=True)
        self.assertEqual(len(g), 2)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
