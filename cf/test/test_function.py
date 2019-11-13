import atexit
import datetime
import unittest

import cf


class functionTest(unittest.TestCase):
    def setUp(self):
        self.test_only = ()

    def test_example_field(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for n in range(5):
            f = cf.example_field(n)
            a = f.array
            d = f.dump(display=False)
            

#--- End: class

if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    cf.environment()
    print()
    unittest.main(verbosity=2)
