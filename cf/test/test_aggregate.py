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

    chunk_sizes = (100000, 300, 34)
    original_chunksize = cf.chunksize()

    def test_basic_aggregate(self):
        for chunksize in self.chunk_sizes:
            cf.chunksize(chunksize)

            f = cf.read(self.filename, squeeze=True)[0]

            g = cf.FieldList(f[0])
            g.append(f[1:3])
            g.append(f[3])
            g[-1].flip(0, inplace=True)
            g.append(f[4:7])
            g[-1].flip(0, inplace=True)
            g.extend([f[i] for i in range(7, f.shape[0])])

            g0 = g.copy()
            self.assertTrue(g.equals(g0, verbose=2), "g != g0")

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
            x = cf.aggregate([c, t])
            self.assertEqual(len(x), 2)

            t.long_name = "qwerty"
            c.long_name = "qwerty"
            x = cf.aggregate([c, t], field_identity="long_name")
            self.assertEqual(len(x), 1)

        cf.chunksize(self.original_chunksize)

    def test_aggregate_exist_equal_ignore_opts(self):
        # TODO: extend the option-checking coverage so all options and all
        # reasonable combinations of them are tested. For now, this is
        # testing options that previously errored due to a bug.
        for chunksize in self.chunk_sizes:
            cf.chunksize(chunksize)

            f = cf.read(self.filename, squeeze=True)[0]

            # Use f as-is: simple test that aggregate works and does not
            # change anything with the given options:
            g = cf.aggregate(f, exist_all=True)[0]
            self.assertEqual(g, f)
            h = cf.aggregate(f, equal_all=True)[0]
            self.assertEqual(h, f)

            with self.assertRaises(ValueError):  # contradictory options
                cf.aggregate(f, exist_all=True, equal_all=True)

        cf.chunksize(self.original_chunksize)

    def test_aggregate_verbosity(self):
        for chunksize in self.chunk_sizes:
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
                        "No log entry begins with '{}'".format(header),
                    )

            # ...but with 'DETAIL' (3), should get only the detail-level one.
            with self.assertLogs(level="NOTSET") as catch:
                cf.aggregate([f0, f1], verbose=3)
                self.assertTrue(
                    any(
                        log_item.startswith(detail_header)
                        for log_item in catch.output
                    ),
                    "No log entry begins with '{}'".format(detail_header),
                )
                self.assertFalse(
                    any(
                        log_item.startswith(debug_header)
                        for log_item in catch.output
                    ),
                    "A log entry begins with '{}' but should not".format(
                        debug_header
                    ),
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
                        "A log entry begins with '{}' but should not".format(
                            header
                        ),
                    )


# --- End: class


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
