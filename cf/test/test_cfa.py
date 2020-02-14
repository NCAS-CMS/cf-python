import datetime
import unittest
import inspect
import subprocess

import cf


class cfaTest(unittest.TestCase):
    def setUp(self):
        self.test_only = ()

    def test_cfa(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        subprocess.run(' '.join(['.', './cfa_test.sh']),
                       shell=True, check=True)


# --- End: class

if __name__ == '__main__':
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
