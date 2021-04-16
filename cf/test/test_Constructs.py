import datetime
import unittest

import faulthandler

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class ConstructsTest(unittest.TestCase):
    """TODO DOCS."""

    f = cf.example_field(1)

    def setUp(self):
        """TODO DOCS."""
        # Disable log messages to silence expected warnings
        cf.LOG_LEVEL("DISABLE")
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows:
        #
        # cf.LOG_LEVEL('DEBUG')
        # < ... test code ... >
        # cf.log_level('DISABLE')

    def test_Constructs__repr__(self):
        """TODO DOCS."""
        f = self.f

        repr(f.constructs)

    def test_Constructs_filter_by_naxes(self):
        """TODO DOCS."""
        c = self.f.constructs

        self.assertEqual(len(c.filter_by_naxes()), 12)
        self.assertEqual(len(c.filter_by_naxes(1)), 7)
        self.assertEqual(len(c.filter_by_naxes(cf.ge(2))), 5)
        self.assertEqual(len(c.filter_by_naxes(1, cf.ge(2))), 12)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
