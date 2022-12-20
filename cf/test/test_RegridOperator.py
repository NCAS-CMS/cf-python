import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class RegridOperatorTest(unittest.TestCase):
    f = cf.example_field(0)
    r = f.regrids(f, "conservative", return_operator=True)

    def test_RegridOperator__repr__(self):
        repr(self.r)

    def test_RegridOperator_name(self):
        self.assertEqual(self.r.name, "regrids")

    def test_RegridOperator_method(self):
        self.assertEqual(self.r.method, "conservative_1st")

    def test_RegridOperator_parameters(self):
        self.assertIsInstance(self.r.parameters, dict)

    def test_RegridOperator_check_method(self):
        self.assertTrue(self.r.check_method("conservative"))
        self.assertTrue(self.r.check_method("conservative_1st"))
        self.assertFalse(self.r.check_method("conservative_2nd"))

    def test_RegridOperator_destroy(self):
        self.r.destroy()

    def test_RegridOperator_get_parameter(self):
        self.r.get_parameter("dst")

        with self.assertRaises(ValueError):
            self.r.get_parameter(None)

    def test_RegridOperator_copy(self):
        self.r.copy()

    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
    def test_RegridOperator_regrid(self):
        from ESMF import Regrid

        self.assertIsInstance(self.r.regrid, Regrid)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
