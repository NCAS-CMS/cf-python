import datetime
import faulthandler
import os
import unittest

import numpy

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class Field_collapseTest(unittest.TestCase):
    def setUp(self):
        self.filename2 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_file2.nc"
        )

    def test_Field_collapse_CLIMATOLOGICAL_TIME(self):
        verbose = False

        f = cf.example_field(2)

        g = f.collapse(
            "T: mean within years time: minimum over years",
            within_years=cf.seasons(),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 4

        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse(
            "T: max within years time: minimum over years",
            within_years=cf.seasons(),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 4

        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f.collapse(
            "T: mean within years time: minimum over years",
            within_years=cf.M(),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 12

        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f.collapse(
            "T: max within years time: minimum over years", within_years=cf.M()
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 12

        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f[:12].collapse(
            "T: mean within years time: minimum over years",
            within_years=cf.seasons(),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 4

        if verbose:
            print("\n", f[:12])
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f[:12].collapse(
            "T: max within years time: minimum over years",
            within_years=cf.seasons(),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 4

        if verbose:
            print("\n", f[:12])
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f[:12].collapse(
            "T: mean within years time: minimum over years",
            within_years=cf.M(),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 12

        if verbose:
            print("\n", f[:12])
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f[:12].collapse(
            "T: max within years time: minimum over years", within_years=cf.M()
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 12

        if verbose:
            print("\n", f[:12])
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        for key in f.cell_methods(todict=True):
            f.del_construct(key)

        g = f.collapse(
            "T: max within years time: minimum over years",
            within_years=cf.seasons(),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 4

        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f.collapse(
            "T: max within years time: min over years", within_years=cf.M()
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 12

        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f[:12].collapse(
            "T: max within years time: minimum over years",
            within_years=cf.seasons(),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 4

        if verbose:
            print("\n", f[:12])
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f[:12].collapse(
            "T: max within years time: minimum over years", within_years=cf.M()
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 12

        if verbose:
            print("\n", f[:12])
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f.collapse(
            "T: max within years time: minimum over years",
            within_years=cf.seasons(),
            over_years=cf.Y(1),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 12

        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f.collapse(
            "T: max within years time: minimum over years",
            within_years=cf.seasons(),
            over_years=cf.Y(2),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 8

        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f.collapse(
            "T: max within years time: minimum over years",
            within_years=cf.seasons(),
            over_years=cf.Y(3),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 4

        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f.collapse(
            "T: max within years time: minimum over years",
            within_years=cf.seasons(),
            over_years=None,
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 4

        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

        g = f[::-1, ...].collapse(
            "T: max within years time: minimum over years",
            within_years=cf.seasons(),
            over_years=cf.Y(2),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 8

        if verbose:
            print("\n", f[::-1, ...])
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape)

    def test_Field_collapse(self):
        verbose = False

        f = cf.read(self.filename2)[0]

        g = f.collapse("mean")
        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(g.shape, (1, 1, 1), g.shape)

        g = f.collapse("mean", axes=["T", "X"])
        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(g.shape, (1, 4, 1))

        g = f.collapse("mean", axes=[0, 2])

        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(g.shape, (1, 4, 1))

        g = f.collapse("mean", axes=[0, 1])
        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(g.shape, (1, 1, 5))

        g = f.collapse("mean", axes="domainaxis1")
        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(g.shape, (1800, 1, 5))

        g = f.collapse("mean", axes=["domainaxis1"])
        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(g.shape, (1800, 1, 5))

        g = f.collapse("mean", axes=[1])
        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(g.shape, (1800, 1, 5))

        g = f.collapse("mean", axes=1)
        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(g.shape, (1800, 1, 5))

        g = f.collapse("T: mean")
        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(g.shape, (1, 4, 5))

        g = f.collapse("T: mean X: maximum")
        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(g.shape, (1, 4, 1))

        g = f.collapse(
            "T: mean within years time: minimum over years",
            within_years=cf.M(),
        )
        if verbose:
            print("\n", f)
            print(g)
            print(g.constructs)
        self.assertEqual(g.shape, (12, 4, 5))

        for m in range(1, 13):
            a = numpy.empty((5, 4, 5))
            for i, year in enumerate(
                f.subspace(T=cf.month(m)).coord("T").year.unique()
            ):
                cf.month(m) & cf.year(year)
                x = f.subspace(T=cf.month(m) & cf.year(year))
                x.data.mean(axes=0, inplace=True)
                a[i] = x.array

            a = a.min(axis=0)
            self.assertTrue(numpy.allclose(a, g.array[m % 12]))

        g = f.collapse("T: mean", group=360)

        for group in (
            cf.M(12),
            cf.M(12, month=12),
            cf.M(12, day=16),
            cf.M(12, month=11, day=27),
        ):
            g = f.collapse("T: mean", group=group)
            bound = g.coord("T").bounds.datetime_array[0, 1]
            self.assertEqual(
                bound.month,
                group.offset.month,
                f"{bound.month}!={group.offset.month}, group={group}",
            )
            self.assertEqual(
                bound.day,
                group.offset.day,
                f"{bound.day}!={group.offset.day}, group={group}",
            )

    def test_Field_collapse_WEIGHTS(self):
        verbose = False

        f = cf.example_field(2)

        if verbose:
            print(f)

        g = f.collapse("area: mean")
        g = f.collapse("area: mean", weights="area")
        if verbose:
            print(g)

        # Check area/volume collapses on fields with a different setup:
        h = cf.example_field(3)
        h.collapse("volume: minimum")
        i = cf.example_field(4)
        i.collapse("area: maximum")

    def test_Field_collapse_GROUPS(self):
        verbose = False

        f = cf.example_field(2)

        g = f.collapse("T: mean", group=cf.M(12), group_span=cf.Y())
        expected_shape = list(f.shape)
        expected_shape[0] = 2

        if verbose:
            print(f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse("T: mean", group=cf.M(12, month=12), group_span=cf.Y())
        expected_shape = list(f.shape)
        expected_shape[0] = 3

        if verbose:
            print(f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse("T: mean", group=cf.M(12, day=16), group_span=cf.Y())
        expected_shape = list(f.shape)
        expected_shape[0] = 2

        if verbose:
            print(f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse(
            "T: mean", group=cf.M(12, month=11, day=27), group_span=cf.Y()
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 3

        if verbose:
            print(f)
            print(g)
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse(
            "T: mean", group=cf.M(12, month=6, day=27), group_span=cf.Y()
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 2

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse(
            "T: mean",
            group=cf.M(5, month=12),
            group_span=cf.M(5),
            group_contiguous=1,
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 7

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse(
            "T: mean",
            group=cf.M(5, month=12),
            group_span=cf.M(5),
            group_contiguous=1,
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 7

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse(
            "T: mean",
            group=cf.M(5, month=3),
            group_span=cf.M(5),
            group_contiguous=1,
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 7

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse(
            "T: mean",
            group=cf.M(5, month=2),
            group_span=cf.M(5),
            group_contiguous=1,
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 7

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse(
            "T: mean",
            group=cf.M(5, month=12),
            group_span=cf.M(5),
            group_contiguous=2,
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 7

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse("T: mean", group=cf.M(5, month=3))
        expected_shape = list(f.shape)
        expected_shape[0] = 7

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)
        # TODO - look into month offset when M< 12

        g = f.collapse(
            "T: mean",
            group=cf.M(5, month=3),
            group_span=cf.M(5),
            group_contiguous=2,
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 7

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse("T: mean", group=cf.M(5, month=12), group_contiguous=1)
        expected_shape = list(f.shape)
        expected_shape[0] = 7

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse("T: mean", group=cf.M(5, month=3), group_contiguous=1)
        expected_shape = list(f.shape)
        expected_shape[0] = 7

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse("T: mean", group=cf.M(5, month=12), group_contiguous=2)
        expected_shape = list(f.shape)
        expected_shape[0] = 7

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        # Test method=integral with groups
        g = f.collapse(
            "T: integral", group=cf.M(5, month=12), weights=True, measure=True
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 7
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse("T: mean", group=cf.M(5, month=3), group_contiguous=2)
        expected_shape = list(f.shape)
        expected_shape[0] = 7

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse(
            "T: mean within years time: minimum over years",
            within_years=cf.M(3),
            group_span=True,
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 4

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

        g = f.collapse(
            "T: mean within years time: minimum over years",
            within_years=cf.seasons(),
            group_span=cf.M(3),
        )
        expected_shape = list(f.shape)
        expected_shape[0] = 4

        if verbose:
            print(f)
            print(g)
            print(
                g.dimension_coordinates("T").value().bounds.data.datetime_array
            )
            print(g.constructs)
        self.assertEqual(list(g.shape), expected_shape, g.shape)

    def test_Field_collapse_sum(self):
        f = cf.example_field(0)
        w = f.weights("area", measure=True).persist()
        a = f.array
        wa = w.array
        ws = a * wa
        ws_sum = ws.sum()

        g = f.collapse("area: sum")
        self.assertTrue((g.array == a.sum()).all())

        g = f.collapse("area: sum", weights=w)
        self.assertTrue((g.array == ws_sum).all())
        self.assertEqual(g.Units, cf.Units("1"))

        g = f.collapse("area: sum", weights=w, scale=1)
        self.assertTrue((g.array == (ws / wa.max()).sum()).all())
        self.assertEqual(g.Units, cf.Units("1"))

        g = f.collapse("area: sum", weights=w)
        self.assertTrue((g.array == ws_sum).all())
        self.assertEqual(g.Units, cf.Units("1"))

        # Can't set measure=True for 'sum' collapses
        with self.assertRaises(ValueError):
            g = f.collapse("area: sum", weights=w, measure=True)

    def test_Field_collapse_integral(self):
        f = cf.example_field(0)
        w = f.weights("area", measure=True).persist()
        a = f.array
        wa = w.array

        g = f.collapse("area: integral", weights=w, measure=True)
        self.assertTrue((g.array == (a * wa).sum()).all())
        self.assertEqual(g.Units, cf.Units("m2"))

        # Must set the 'weights' parameter for 'integral' collapses
        with self.assertRaises(ValueError):
            g = f.collapse("area: integral")

        # Must set measure=True for 'integral' collapses
        with self.assertRaises(ValueError):
            g = f.collapse("area: integral", weights=w)

        # 'scale' must be None when 'measure' is True
        with self.assertRaises(ValueError):
            g = f.collapse("area: integral", weights=w, measure=True, scale=1)

    def test_Field_collapse_sum_weights(self):
        f = cf.example_field(0)
        w = f.weights("area", measure=True).persist()
        wa = w.array

        g = f.collapse("area: sum_of_weights")
        self.assertTrue((g.array == 40).all())
        self.assertEqual(g.Units, cf.Units())

        g = f.collapse("area: sum_of_weights", weights=w)
        self.assertTrue((g.array == wa.sum()).all())
        self.assertEqual(g.Units, cf.Units("1"))

        g = f.collapse("area: sum_of_weights", weights=w, measure=True)
        self.assertTrue((g.array == wa.sum()).all())
        self.assertEqual(g.Units, cf.Units("m2"))

        g = f.collapse("area: sum_of_weights", weights=w, scale=1)
        self.assertTrue((g.array == (wa / wa.max()).sum()).all())
        self.assertEqual(g.Units, cf.Units("1"))

    def test_Field_collapse_sum_weights2(self):
        f = cf.example_field(0)
        w = f.weights("area", measure=True).persist()
        wa = w.array**2
        wa_sum = wa.sum()

        g = f.collapse("area: sum_of_weights2")
        self.assertTrue((g.array == 40).all())
        self.assertEqual(g.Units, cf.Units())

        g = f.collapse("area: sum_of_weights2", weights=w)
        self.assertTrue((g.array == wa_sum).all())
        self.assertEqual(g.Units, cf.Units("1"))

        g = f.collapse("area: sum_of_weights2", weights=w, measure=True)
        self.assertTrue((g.array == wa_sum).all())
        self.assertEqual(g.Units, cf.Units("m4"))

        g = f.collapse("area: sum_of_weights2", weights=w, scale=1)
        self.assertTrue((g.array == (wa / wa.max()).sum()).all())
        self.assertEqual(g.Units, cf.Units("1"))

    def test_Field_collapse_non_positive_weights(self):
        f = cf.example_field(0)
        w = f.weights("area").persist()

        for method in (
            "mean",
            "sum",
            "root_mean_square",
            "variance",
            "sum_of_weights",
        ):
            for x in (0, -3.14):
                w[0, 0] = x
                g = f.collapse(axes="area", method=method, weights=w)
                with self.assertRaises(ValueError):
                    # The check for non-positive weights occurs at
                    # compute time
                    g.array


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
