import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class IndexTest(unittest.TestCase):
    def setUp(self):
        # Disable log messages to silence expected warnings
        cf.log_level("DISABLE")
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows:
        #
        # cf.LOG_LEVEL('DEBUG')
        # < ... test code ... >
        # cf.log_level('DISABLE')

        self.indexed = "DSG_timeSeries_indexed.nc"

    def test_Index__repr__str__dump(self):
        f = cf.read(self.indexed)[0]

        index = f.data.get_index()

        _ = repr(index)
        _ = str(index)
        self.assertIsInstance(index.dump(display=False), str)


# --- End: class


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
