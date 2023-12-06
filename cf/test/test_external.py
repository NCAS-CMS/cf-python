import datetime
import faulthandler
import os
import tempfile
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class ExternalVariableTest(unittest.TestCase):
    def setUp(self):
        self.parent_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "parent.nc"
        )
        self.external_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "external.nc"
        )
        self.combined_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "combined.nc"
        )
        self.external_missing_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "external_missing.nc"
        )

        self.test_only = []

        (fd, self.tempfilename) = tempfile.mkstemp(
            suffix=".nc", prefix="cf_", dir="."
        )
        os.close(fd)
        (fd, self.tempfilename_parent) = tempfile.mkstemp(
            suffix=".nc", prefix="cf_parent_", dir="."
        )
        os.close(fd)
        (fd, self.tempfilename_external) = tempfile.mkstemp(
            suffix=".nc", prefix="cf_external_", dir="."
        )
        os.close(fd)

    def tearDown(self):
        os.remove(self.tempfilename)
        os.remove(self.tempfilename_parent)
        os.remove(self.tempfilename_external)

    def test_EXTERNAL_READ(self):
        # Read the parent file on its own, without the external file
        f = cf.read(self.parent_file, verbose=0)

        self.assertEqual(len(f), 1)

        for i in f:
            repr(i)
            str(i)
            i.dump(display=False)

        f = f[0]

        cell_measure = f.cell_measure("measure:area")

        self.assertTrue(cell_measure.nc_get_external())
        self.assertEqual(cell_measure.nc_get_variable(), "areacella")
        self.assertEqual(cell_measure.properties(), {})
        self.assertFalse(cell_measure.has_data())

        # External file contains only the cell measure variable
        f = cf.read(self.parent_file, external=[self.external_file], verbose=0)

        c = cf.read(self.combined_file, verbose=0)

        for i in c + f:
            repr(i)
            str(i)
            i.dump(display=False)

        cell_measure = f[0].cell_measure("measure:area")

        self.assertEqual(len(f), 1)
        self.assertEqual(len(c), 1)

        for i in range(len(f)):
            self.assertTrue(c[i].equals(f[i], verbose=2))

        # External file contains other variables
        f = cf.read(self.parent_file, external=self.combined_file, verbose=0)

        for i in f:
            repr(i)
            str(i)
            i.dump(display=False)

        self.assertEqual(len(f), 1)
        self.assertEqual(len(c), 1)

        for i in range(len(f)):
            self.assertTrue(c[i].equals(f[i], verbose=2))

        # Two external files
        f = cf.read(
            self.parent_file,
            external=[self.external_file, self.external_missing_file],
            verbose=0,
        )

        for i in f:
            repr(i)
            str(i)
            i.dump(display=False)

        self.assertEqual(len(f), 1)
        self.assertEqual(len(c), 1)

        for i in range(len(f)):
            self.assertTrue(c[i].equals(f[i], verbose=2))

    def test_EXTERNAL_WRITE(self):
        parent = cf.read(self.parent_file)
        combined = cf.read(self.combined_file)

        # External file contains only the cell measure variable
        f = cf.read(self.parent_file, external=self.external_file)

        cf.write(f, self.tempfilename)
        g = cf.read(self.tempfilename)

        self.assertEqual(len(g), len(combined))

        for i in range(len(g)):
            self.assertTrue(combined[i].equals(g[i], verbose=2))

        cell_measure = g[0].cell_measure("measure:area")

        self.assertFalse(cell_measure.nc_get_external())
        cell_measure.nc_set_external(True)
        self.assertTrue(cell_measure.nc_get_external())
        self.assertTrue(cell_measure.properties())
        self.assertTrue(cell_measure.has_data())

        self.assertTrue(g[0].cell_measure("measure:area").nc_get_external())

        cf.write(
            g,
            self.tempfilename_parent,
            external=self.tempfilename_external,
            verbose=0,
        )

        h = cf.read(self.tempfilename_parent, verbose=0)

        self.assertEqual(len(h), len(parent))

        for i in range(len(h)):
            self.assertTrue(parent[i].equals(h[i], verbose=2))

        h = cf.read(self.tempfilename_external)
        external = cf.read(self.external_file)

        self.assertEqual(len(h), len(external))

        for i in range(len(h)):
            self.assertTrue(external[i].equals(h[i], verbose=2))

    def test_EXTERNAL_AGGREGATE(self):
        # Read parent file without the external file, taking first
        # field to test
        f = cf.read(self.parent_file, verbose=0)[0]
        measure_name = "measure:area"

        # Split f into parts (take longitude with 3 x 3 = 9 points) so
        # can test the difference between the aggregated result and
        # original f.  Note all parts retain the external variable
        # cell measure.
        f_lon_thirds = [f[:, :3], f[:, 3:6], f[:, 6:]]

        g = cf.aggregate(f_lon_thirds, verbose=2)

        self.assertEqual(len(g), 1)

        # Check cell measure construct from external variable has been
        # retained
        self.assertEqual(len(g[0].cell_measures()), 1)

        # Check aggregated field is identical to original
        self.assertEqual(g[0], f)

        # Also check aggregation did not remove the measure from the
        # inputs
        for part in f_lon_thirds:
            cell_measure = part.cell_measure("measure:area")
            self.assertTrue(cell_measure.nc_get_external())

        # Now try aggregating when one part doesn't have the cell
        # measure
        f_lon_thirds[1].del_construct(measure_name)
        g = cf.aggregate(f_lon_thirds)
        self.assertEqual(len(g), 2)
        self.assertFalse(g[1].cell_measures())

        # Also check measure was not removed from, or added to, any
        # input
        for part in [f_lon_thirds[0], f_lon_thirds[2]]:
            cm = part.cell_measure("measure:area")
            self.assertTrue(cm.nc_get_external())

        self.assertFalse(f_lon_thirds[1].cell_measures())


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
