import atexit
import datetime
import unittest
import inspect

import cf


class functionTest(unittest.TestCase):
    def setUp(self):
        self.test_only = ()

    def test_example_field(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for n in range(8):
            f = cf.example_field(n)
            _ = f.array
            _ = f.dump(display=False)

        with self.assertRaises(Exception):
            _ = cf.example_field(-999)

    def test_keyword_deprecation(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Use as test case 'i' kwarg, the deprecated old name for
        # 'inplace':
        a = cf.Data([list(range(100))])
        a.squeeze(inplace=True)  # new way to specify operation tested below

        b = cf.Data([list(range(100))])
        with self.assertRaises(cf.functions.DeprecationError):
            b.squeeze(i=True)


# --- End: class

if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    cf.environment()
    print()
    unittest.main(verbosity=2)
