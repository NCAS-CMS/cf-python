import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class DomainAncillaryTest(unittest.TestCase):
    def test_DomainAncillary(self):
        f = cf.DomainAncillary()

        repr(f)
        str(f)
        f.dump(display=False)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(module=__file__.split(".")[0], verbosity=2)
