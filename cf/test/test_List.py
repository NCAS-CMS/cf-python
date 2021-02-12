import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class ListTest(unittest.TestCase):
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

        self.gathered = "gathered.nc"

    def test_List__repr__str__dump(self):
        f = cf.read(self.gathered)[0]

        list_ = f.data.get_list()

        _ = repr(list_)
        _ = str(list_)
        self.assertIsInstance(list_.dump(display=False), str)


# --- End: class


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
