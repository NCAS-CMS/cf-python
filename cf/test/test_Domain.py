import datetime
import inspect
import os
import unittest

import cf


class DomainTest(unittest.TestCase):
    d = cf.example_field(1)

    def setUp(self):
        # Disable log messages to silence expected warnings
        cf.LOG_LEVEL('DISABLE')
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows:
        #
        # cf.LOG_LEVEL('DEBUG')
        # < ... test code ... >
        # cf.log_level('DISABLE')
        self.test_only = []

    def test_Domain__repr__str__dump(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = self.d

        _ = repr(d)
        _ = str(d)
        self.assertIsInstance(d.dump(display=False), str)

    def test_Domain__init__(self):
        d = cf.Domain(source='qwerty')

    def test_Domain_equals(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = self.d
        e = d.copy()

        self.assertTrue(d.equals(d, verbose=3))
        self.assertTrue(d.equals(e, verbose=3))
        self.assertTrue(e.equals(d, verbose=3))

    def test_Domain_properties(self):
        d = cf.Domain()

        d.set_property('long_name', 'qwerty')

        self.assertEqual(d.properties(), {'long_name': 'qwerty'})
        self.assertEqual(d.get_property('long_name'), 'qwerty')
        self.assertEqual(d.del_property('long_name'), 'qwerty')
        self.assertIsNone(d.get_property('long_name', None))
        self.assertIsNonec(d.del_property('long_name', None))

        d.set_property('long_name', 'qwerty')
        self.assertEqual(d.clear_properties(), {'long_name': 'qwerty'})

        d.set_properties({'long_name': 'qwerty'})
        d.set_properties({'foo': 'bar'})
        self.assertEqual(d.properties(),
                         {'long_name': 'qwerty', 'foo': 'bar'})

# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print('')
    unittest.main(verbosity=2)
