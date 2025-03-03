import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

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
    @unittest.skipUnless(Active is not None, "Requires activestorage.Active")
    def test_active_storage(self):
        # No masked values
        f = cf.example_field(0)
        cf.write(f, tmpfile)

        f = cf.read(
            tmpfile, dask_chunks={"latitude": (4, 1), "longitude": (3, 5)}
        )
        f = f[0]
        self.assertEqual(f.data.chunks, ((4, 1), (3, 5)))

        cf.active_storage(False)
        self.assertFalse(cf.active_storage())
        f.collapse("mean", weights=False)

        local_array = f.collapse("mean", weights=False).array

        with cf.configuration(active_storage=True, active_storage_url="dummy"):
            self.assertTrue(cf.active_storage())
            self.assertEqual(cf.active_storage_url(), "dummy")
            active_array = f.collapse("mean", weights=False).array

        self.assertEqual(active_array, local_array)

        # TODOACTIVE: Test with masked values (not yet working in
        #             activestorage.Active)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
