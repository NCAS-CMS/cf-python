import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class RegridOperatorTest(unittest.TestCase):
    f = cf.example_field(0)
    r = f.regrids(f, "conservative", return_operator=True)

    def test_RegridOperator_coord_sys(self):
        self.assertEqual(self.r.coord_sys, "spherical")

    def test_RegridOperator_method(self):
        self.assertEqual(self.r.method, "conservative_1st")

    def test_RegridOperator_src_shape(self):
        self.assertEqual(self.r.src_shape, f.cyclic("X"))

    def test_RegridOperator_dst_shape(self):
        self.assertEqual(self.r.dst_shape, f.cyclic("X"))

    def test_RegridOperator_src_cyclic(self):
        self.assertTrue(self.r.src_cyclic, True)

    def test_RegridOperator_dst_cyclic(self):
        self.assertTrue(self.r.dst_cyclic, True)

    def test_RegridOperator_src_mask(self):
        self.assertIsNone(self.r.src_mask, None)

    def test_RegridOperator_dst_mask(self):
        self.assertIsNone(self.r.dst_mask, None)

    def test_RegridOperator_parameters(self):
        self.assertIsInstance(self.r.parameters(), dict)

    def test_RegridOperator_get_parameter(self):
        self.r.get_parameter("dst")
        self.assertIsNone(self.r.get_parameter("bad_name", None))
        with self.assertRaises(ValueError):
            self.r.get_parameter(None)

    def test_RegridOperator_copy(self):
        self.assertIsInstance(self.r.copy(), self.r)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
