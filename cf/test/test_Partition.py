import datetime
import faulthandler
import inspect
import os
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class PartitionTest(unittest.TestCase):
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file.nc"
    )
    chunk_sizes = (17, 34, 300, 100000)[::-1]
    original_chunksize = cf.chunksize()

    test_only = []

    def test_Partition(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return


# --- End: class


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
