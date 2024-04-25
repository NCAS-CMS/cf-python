import datetime
import faulthandler
import os
import unittest
import warnings

import numpy as np

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
        self.assertTrue(g.equals(g0, verbose=2))

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
        self.assertEqual(h[0].shape, (10, 9))
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

        self.assertEqual(i[0].shape, (10, 9))

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

        self.assertEqual(i[0].shape, (10, 9))

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

    def test_aggregate_relaxed_units(self):
        f = cf.example_field(0)
        bad_units = cf.Units("bad-units")
        f.override_units(bad_units, inplace=True)
        g = f[:2]
        h = f[2:]
        i = cf.aggregate([g, h], relaxed_units=True)
        self.assertEqual(len(i), 1)
        i = i[0]
        self.assertEqual(i.Units.__dict__, bad_units.__dict__)
        self.assertTrue((i.array == f.array).all())

    def test_aggregate_field_ancillaries(self):
        f = cf.example_field(0)
        self.assertFalse(f.field_ancillaries())

        a = f[:2]
        b = f[2:]
        a.set_property("foo", "bar_a")
        b.set_property("foo", "bar_b")

        c = cf.aggregate([a, b], field_ancillaries="foo")
        self.assertEqual(len(c), 1)
        c = c[0]
        self.assertEqual(len(c.field_ancillaries()), 1)

        anc = c.field_ancillary()
        self.assertEqual(anc.shape, c.shape)
        self.assertTrue((anc[:2] == "bar_a").all())
        self.assertTrue((anc[2:] == "bar_b").all())

    def test_aggregate_cells(self):
        """Test the 'cells' keyword of cf.aggregate"""
        f = cf.example_field(0)
        fl = (f[:2], f[2], f[3:])

        # 1-d aggregation resulting in one field
        for cells in (
            None,
            {"Y": {"cellsize": cf.lt(100, "degrees_north")}},
            {"Y": {"cellsize": cf.gt(100, "degrees_north")}},
            {"Y": {"cellsize": cf.wi(30, 60, "degrees_north")}},
            {"Y": {"cellsize": cf.set([30, 60], "degrees_north")}},
            {"Y": {"spacing": cf.set([30, 45], "degrees_north")}},
            {
                "Y": {
                    "cellsize": cf.wi(30, 60, "degrees_north"),
                    "spacing": cf.set([30, 45], "degrees_north"),
                }
            },
        ):
            x = cf.aggregate(fl, cells=cells)
            self.assertEqual(len(x), 1)

        # Test storage of cell conditions
        x = x[0]
        lat = x.dimension_coordinate("latitude")
        chars = lat.get_cell_characteristics()
        self.assertTrue(chars["cellsize"].equals(cf.wi(30, 60, "degrees_N")))
        self.assertTrue(chars["spacing"].equals(cf.set([30, 45], "degrees_N")))
        for identity in ("longitude", "time"):
            self.assertIsNone(
                x.dimension_coordinate(identity).get_cell_characteristics(None)
            )

        for cells in (
            {"Y": {"cellsize": cf.wi(39, 60, "km")}},
            {"foo": {"cellsize": 34}},
            {"T": {"cellsize": cf.D(0)}},
            {"T": {"cellsize": cf.Data(0, "days")}},
            {"T": {"cellsize": cf.Data(99, "days")}},
            {"T": {"cellsize": cf.Data([99], "days")}},
        ):
            self.assertEqual(len(cf.aggregate(fl, cells=cells)), 1)

        # 1-d aggregation resulting in two fields
        for cells in (
            {"Y": {"cellsize": cf.eq(30, "degreeN")}},
            {"Y": {"cellsize": cf.isclose(60, "degrees_N")}},
        ):
            self.assertEqual(len(cf.aggregate(fl, cells=cells)), 2)

        # 1-d aggregation with size 1 and size N axes and muliple
        # spacing conditions
        conditions = [
            {"spacing": cf.eq(-1, "degrees_north")},
            {"spacing": cf.wi(29, 46, "degrees_north")},
        ]
        cells = {"Y": conditions}
        self.assertEqual(len(cf.aggregate(fl, cells=cells)), 2)
        cells = {"Y": conditions[::-1]}
        self.assertEqual(len(cf.aggregate(fl, cells=cells)), 1)

        # 2-d aggregation resulting in one field
        fl_2d = []
        for g in fl:
            fl_2d.extend((g[:, :3], g[:, 3:]))

        for cells in (
            None,
            {"Y": {"cellsize": cf.wi(30, 60, "degrees_north")}},
            {"X": {"cellsize": cf.isclose(45, "degrees_east")}},
            {
                "Y": {"cellsize": cf.wi(30, 60, "degrees_north")},
                "X": {"cellsize": cf.isclose(45, "degrees_east")},
            },
        ):
            self.assertEqual(len(cf.aggregate(fl_2d, cells=cells)), 1)

        #  1-d aggregation with no bounds resulting in one field
        g = f.copy()
        g.dimension_coordinate("Y").del_bounds()
        fl = (g[:2], g[2], g[3:])
        for cells in (
            None,
            {"Y": {"cellsize": cf.lt(100, "degrees_north")}},
            {"Y": {"cellsize": cf.lt(-1, "degrees_north")}},
        ):
            self.assertEqual(len(cf.aggregate(fl, cells=cells)), 1)

        # 1-d aggregation: 'diff' condition depends on 'contiguous'
        h = f.copy()
        fl = [h[..., :3], h[..., 4], h[..., 5:]]  # Miss out h[..., [3]]
        cells = {"X": {"spacing": cf.Data(45, "degreeE")}}
        # .. aggregates with contiguous=False
        self.assertEqual(
            len(cf.aggregate(fl, cells=cells, contiguous=False)), 1
        )
        # ... does not aggregate with contiguous=True
        self.assertEqual(
            len(cf.aggregate(fl, cells=cells, contiguous=True)), 3
        )

        # Climatology cells
        f = cf.example_field(2)
        g = f.copy()
        g.dimension_coordinate("T").override_units(
            "hours since 1960-01-01", inplace=1
        )
        self.assertEqual(len(cf.aggregate([f[:12], f[12:], g])), 3)
        self.assertEqual(
            len(
                cf.aggregate([f[:12], f[12:], g], cells=cf.climatology_cells())
            ),
            2,
        )

        # Bad cells
        with self.assertRaises(TypeError):
            cf.aggregate(fl, cells=9)

        with self.assertRaises(TypeError):
            cf.aggregate(fl, cells=[9])

        # Bad condition units
        for condition in (cf.M(2), cf.Y(2)):
            with self.assertRaises(ValueError):
                cf.aggregate(fl, cells={"T": {"cellsize": condition}})

        # Bad key
        with self.assertRaises(ValueError):
            cf.aggregate(fl, cells={"T": {"foo": 99}})

    def test_climatology_cells(self):
        """Test cf.climatology_cells"""
        self.assertIsInstance(cf.climatology_cells(), dict)

        self.assertEqual(
            cf.climatology_cells(
                years=False, months=False, days=(), hours=(), seconds=()
            ),
            {"T": []},
        )

        self.assertEqual(
            cf.climatology_cells(
                years=False, months=False, days=(), hours=(), seconds=(1,)
            ),
            {
                "T": [
                    {"cellsize": cf.Data(1, "second")},
                    {
                        "cellsize": cf.Data(0, "second"),
                        "spacing": cf.Data(1, "second"),
                    },
                ]
            },
        )

        self.assertEqual(
            cf.climatology_cells(
                years=False,
                months=False,
                days=(),
                hours=(),
                seconds=(1,),
                seconds_instantaneous=False,
            ),
            {
                "T": [
                    {"cellsize": cf.Data(1, "second")},
                ]
            },
        )

        self.assertEqual(
            cf.climatology_cells(
                years=False,
                months=False,
                days=(),
                hours=(),
                minutes=(1,),
                minutes_instantaneous=False,
            ),
            {
                "T": [
                    {"cellsize": cf.Data(1, "minute")},
                ]
            },
        )

        self.assertEqual(
            cf.climatology_cells(
                years=False, months=False, days=(), hours=(1,)
            ),
            {
                "T": [
                    {"cellsize": cf.Data(1, "hour")},
                    {
                        "cellsize": cf.Data(0, "hour"),
                        "spacing": cf.Data(1, "hour"),
                    },
                ]
            },
        )

        self.assertEqual(
            cf.climatology_cells(
                years=False,
                months=False,
                days=(),
                hours=(1,),
                hours_instantaneous=False,
            ),
            {
                "T": [
                    {"cellsize": cf.Data(1, "hour")},
                ]
            },
        )

        self.assertEqual(
            cf.climatology_cells(
                years=False, months=False, days=(1,), hours=()
            ),
            {
                "T": [
                    {"cellsize": cf.Data(1, "day")},
                ]
            },
        )

        self.assertEqual(
            cf.climatology_cells(
                years=False,
                months=False,
                days=(1,),
                hours=(),
                days_instantaneous=True,
            ),
            {
                "T": [
                    {"cellsize": cf.Data(1, "day")},
                    {
                        "cellsize": cf.Data(0, "day"),
                        "spacing": cf.Data(1, "day"),
                    },
                ]
            },
        )

        cells = cf.climatology_cells(
            years=False,
            days=(),
            hours=(),
        )
        condition = cells["T"][0]["cellsize"]
        self.assertTrue(condition.equals(cf.M()))

        cells = cf.climatology_cells(
            years=True,
            months=False,
            days=(),
            hours=(),
        )
        condition = cells["T"][0]["cellsize"]
        self.assertTrue(condition.equals(cf.Y()))

    def test_aggregate_ugrid(self):
        """Test ugrid aggregation"""
        f = cf.example_field(8)

        # Test that aggregation over a non-ugrid axis (time, in this
        # case) works.
        g = f.copy()
        t = g.dim("T")
        cf.bounds_combination_mode("OR")
        t += 72000
        a = cf.aggregate([f, g])
        self.assertEqual(len(a), 1)
        a = a[0]
        self.assertEqual(len(a.domain_topologies()), 1)
        self.assertEqual(len(a.cell_connectivities()), 1)

        # Test that aggregation over a non-ugrid axis doesn't work
        # when the domain topology constructs are different
        h = g.copy()
        d = h.domain_topology()
        d = d.data
        d += 1
        self.assertEqual(len(cf.aggregate([f, h])), 2)

        # Test that aggregation over a non-ugrid axis doesn't work
        # when the cell connnectivty constructs are different
        h = g.copy()
        c = h.cell_connectivity()
        d = c.data
        d += 1
        self.assertEqual(len(cf.aggregate([f, h])), 2)

        # Test that aggregation over a ugrid axis doesn't work
        g = f.copy()
        x = g.aux("X")
        d = x.data
        d += 0.1
        self.assertEqual(len(cf.aggregate([f, g])), 2)

    def test_aggregate_trajectory(self):
        """Test DSG trajectory aggregation"""
        # Test that aggregation occurs when the tractory_id axes have
        # identical 1-d auxiliary coordinates
        f = cf.example_field(11)
        g = cf.aggregate([f, f], relaxed_identities=True)
        self.assertEqual(len(g), 1)

        g = g[0]
        self.assertTrue(
            g.subspace(**{"cf_role=trajectory_id": [0]}).equals(
                g.subspace(**{"cf_role=trajectory_id": [1]})
            )
        )

    def test_aggregate_actual_range(self):
        """Test aggregation of actual_range"""
        f = cf.example_field(0)
        f.set_property("actual_range", (5, 10))
        f.set_property("valid_range", (0, 15))
        f0 = f[:, :2]
        f1 = f[:, 2:4]
        f2 = f[:, 4:]

        g = cf.aggregate([f0, f1, f2])
        self.assertEqual(len(g), 1)
        self.assertEqual(g[0].get_property("actual_range"), (5, 10))

        f1.set_property("actual_range", [2, 13])
        g = cf.aggregate([f0, f1, f2])
        self.assertEqual(len(g), 1)
        self.assertEqual(g[0].get_property("actual_range"), (2, 13))

        f1.set_property("actual_range", [-2, 17])
        g = cf.aggregate([f0, f1, f2])
        self.assertEqual(len(g), 1)
        self.assertEqual(g[0].get_property("actual_range"), (-2, 17))

        g = cf.aggregate([f0, f1, f2], respect_valid=True)
        self.assertEqual(len(g), 1)
        self.assertEqual(g[0].get_property("valid_range"), (0, 15))
        self.assertFalse(g[0].has_property("actual_range"))

        f1.set_property("actual_range", [0, 15])
        g = cf.aggregate([f0, f1, f2], respect_valid=True)
        self.assertEqual(len(g), 1)
        self.assertEqual(g[0].get_property("valid_range"), (0, 15))
        self.assertEqual(g[0].get_property("actual_range"), (0, 15))

    def test_aggregate_numpy_array_property(self):
        """Test aggregation of numpy array-valued properties"""
        a = np.array([5, 10])
        f = cf.example_field(0)
        f.set_property("array", a)
        f0 = f[:, :2]
        f1 = f[:, 2:4]
        f2 = f[:, 4:]

        g = cf.aggregate([f0, f1, f2])
        self.assertEqual(len(g), 1)
        self.assertTrue((g[0].get_property("array") == a).all())

        f1.set_property("array", np.array([-5, 20]))
        g = cf.aggregate([f0, f1, f2])
        self.assertEqual(len(g), 1)
        self.assertEqual(
            g[0].get_property("array"),
            "[ 5 10] :AGGREGATED: [-5 20] :AGGREGATED: [ 5 10]",
        )

        f2.set_property("array", np.array([-5, 20]))
        g = cf.aggregate([f0, f1, f2])
        self.assertEqual(len(g), 1)
        self.assertEqual(
            g[0].get_property("array"),
            "[ 5 10] :AGGREGATED: [-5 20] :AGGREGATED: [-5 20]",
        )

        f1.set_property("array", np.array([5, 10]))
        g = cf.aggregate([f0, f1, f2])
        self.assertEqual(len(g), 1)
        self.assertEqual(
            g[0].get_property("array"),
            "[ 5 10] :AGGREGATED: [-5 20]",
        )


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
