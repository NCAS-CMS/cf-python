import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import numpy as np

import cf

try:
    from activestorage import Active
except ModuleNotFoundError:
    Active = None

n_tmpfiles = 2
tmpfiles = [
    tempfile.mkstemp("_test_active_storage.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
(tmpfile, tmpfile2) = tmpfiles


def _remove_tmpfiles():
    """Try to remove defined temporary files by deleting their paths."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class ActiveStorageTest(unittest.TestCase):
    @unittest.skipUnless(Active is not None, "Requires activestorage package.")
    def test_active_storage(self):
        # No masked values
        f = cf.example_field(0)
        cf.write(f, tmpfile)

        f = cf.read(tmpfile, chunks={"latitude": (4, 1), "longitude": (3, 5)})
        f = f[0]
        self.assertEqual(f.data.chunks, ((4, 1), (3, 5)))

        cf.active_storage(False)
        self.assertFalse(cf.active_storage())
        array = f.collapse("mean", weights=False).array

        with cf.active_storage(True):
            self.assertTrue(cf.active_storage())
            self.assertTrue(f.data.active_storage)
            active_array = f.collapse("mean").array

        self.assertEqual(array, active_array)

        # Masked values (not yet working)
        # self.assertFalse(cf.active_storage())
        # f[0] = cf.masked
        # cf.write(f, tmpfile2)
        # f = cf.read(tmpfile2, chunks={"latitude": (4, 1), "longitude": (3, 5)})
        # f = f[0]
        #
        # array = f.collapse("mean", weights=False).array
        # with cf.active_storage(True):
        #     self.assertTrue(cf.active_storage())
        #     self.assertTrue(f.data.active_storage)
        #     active_array = f.collapse("mean").array
        #
        # self.assertEqual(array, active_array)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
