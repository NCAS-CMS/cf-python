import datetime
import os
import tempfile
import unittest

import numpy

import cf


class ExternalVariableTest(unittest.TestCase):
    def setUp(self):
        self.parent_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'parent.nc')
        self.external_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'external.nc')
        self.combined_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'combined.nc')
        self.external_missing_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'external_missing.nc')

        self.test_only = []

        (fd, self.tempfilename) = tempfile.mkstemp(
            suffix='.nc', prefix='cf_', dir='.')
        os.close(fd)
        (fd, self.tempfilename_parent) = tempfile.mkstemp(
            suffix='.nc', prefix='cf_parent_', dir='.')
        os.close(fd)
        (fd, self.tempfilename_external) = tempfile.mkstemp(
            suffix='.nc', prefix='cf_external_', dir='.')
        os.close(fd)

    def tearDown(self):
        os.remove(self.tempfilename)
        os.remove(self.tempfilename_parent)
        os.remove(self.tempfilename_external)

    def test_EXTERNAL_READ(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Read the parent file on its own, without the external file
        f = cf.read(self.parent_file, verbose=0)

        for i in f:
            _ = repr(i)
            _ = str(i)
            _ = i.dump(display=False)

            self.assertEqual(len(f), 1)
        f = f[0]

        cell_measure = f.constructs.filter_by_identity('measure:area').value()

        self.assertTrue(cell_measure.nc_get_external())
        self.assertEqual(cell_measure.nc_get_variable(), 'areacella')
        self.assertEqual(cell_measure.properties(), {})
        self.assertFalse(cell_measure.has_data())

        # External file contains only the cell measure variable
        f = cf.read(self.parent_file, external=[self.external_file],
                    verbose=0)

        c = cf.read(self.combined_file, verbose=0)

        for i in c + f:
            _ = repr(i)
            _ = str(i)
            _ = i.dump(display=False)

        cell_measure = f[0].constructs.filter_by_identity(
            'measure:area').value()

        self.assertEqual(len(f), 1)
        self.assertEqual(len(c), 1)

        for i in range(len(f)):
            self.assertTrue(c[i].equals(f[i], verbose=2))

        # External file contains other variables
        f = cf.read(self.parent_file, external=self.combined_file,
                    verbose=0)

        for i in f:
            _ = repr(i)
            _ = str(i)
            _ = i.dump(display=False)

        self.assertEqual(len(f), 1)
        self.assertEqual(len(c), 1)

        for i in range(len(f)):
            self.assertTrue(c[i].equals(f[i], verbose=2))

        # Two external files
        f = cf.read(
            self.parent_file,
            external=[self.external_file, self.external_missing_file],
            verbose=0
        )

        for i in f:
            _ = repr(i)
            _ = str(i)
            _ = i.dump(display=False)

        self.assertEqual(len(f), 1)
        self.assertEqual(len(c), 1)

        for i in range(len(f)):
            self.assertTrue(c[i].equals(f[i], verbose=2))

    def test_EXTERNAL_WRITE(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        parent = cf.read(self.parent_file)
        combined = cf.read(self.combined_file)

        # External file contains only the cell measure variable
        f = cf.read(self.parent_file, external=self.external_file)

        cf.write(f, self.tempfilename)
        g = cf.read(self.tempfilename)

        self.assertEqual(len(g), len(combined))

        for i in range(len(g)):
            self.assertTrue(combined[i].equals(g[i], verbose=2))

        cell_measure = g[0].constructs('measure:area').value()

        self.assertFalse(cell_measure.nc_get_external())
        cell_measure.nc_set_external(True)
        self.assertTrue(cell_measure.nc_get_external())
        self.assertTrue(cell_measure.properties())
        self.assertTrue(cell_measure.has_data())

        self.assertTrue(
            g[0].constructs.filter_by_identity(
                'measure:area').value().nc_get_external()
        )

        cf.write(g, self.tempfilename_parent,
                 external=self.tempfilename_external,
                 verbose=0)

        h = cf.read(self.tempfilename_parent, verbose=0)

        self.assertEqual(len(h), len(parent))

        for i in range(len(h)):
            self.assertTrue(parent[i].equals(h[i], verbose=2))

        h = cf.read(self.tempfilename_external)
        external = cf.read(self.external_file)

        self.assertEqual(len(h), len(external))

        for i in range(len(h)):
            self.assertTrue(external[i].equals(h[i], verbose=2))

# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.now())
    print(cf.environment())
    print()
    unittest.main(verbosity=2)
