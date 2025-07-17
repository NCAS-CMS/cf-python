import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class RegridOperatorTest(unittest.TestCase):
    src = cf.example_field(0)
    dst = cf.example_field(1)
    r = src.regrids(dst, "linear", return_operator=True)

    def test_RegridOperator_attributes(self):
        self.assertEqual(self.r.coord_sys, "spherical")
        self.assertEqual(self.r.method, "linear")
        self.assertEqual(self.r.dimensionality, 2)
        self.assertEqual(self.r.start_index, 1)
        self.assertTrue(self.r.src_cyclic)
        self.assertFalse(self.r.dst_cyclic)
        self.assertEqual(len(self.r.src_shape), 2)
        self.assertEqual(len(self.r.dst_shape), 2)
        self.assertEqual(len(self.r.src_coords), 2)
        self.assertEqual(len(self.r.src_bounds), 0)
        self.assertIsNone(self.r.src_axes)
        self.assertIsNone(self.r.dst_axes)
        self.assertIsNone(self.r.src_mask)
        self.assertIsNone(self.r.dst_mask)
        self.assertEqual(self.r.weights.ndim, 2)
        self.assertIsNone(self.r.row)
        self.assertIsNone(self.r.col)
        self.assertIsNone(self.r.weights_file)
        self.assertFalse(self.r.src_mesh_location)
        self.assertFalse(self.r.dst_mesh_location)
        self.assertFalse(self.r.src_featureType)
        self.assertFalse(self.r.dst_featureType)
        self.assertIsNone(self.r.src_z)
        self.assertIsNone(self.r.dst_z)
        self.assertFalse(self.r.ln_z)

    def test_RegridOperator_copy(self):
        self.assertIsInstance(self.r.copy(), self.r.__class__)

    def test_RegridOperator_equal_weights(self):
        r0 = self.r
        r1 = r0.copy()
        self.assertTrue(r0.equal_weights(r1))
        r1.weights.data += 0.1
        self.assertFalse(r0.equal_weights(r1))

    def test_RegridOperator_equal_dst_mask(self):
        r0 = self.r.copy()
        r1 = r0.copy()
        self.assertTrue(r0.equal_dst_mask(r1))
        mask = [True, False]
        r0._set_component("dst_mask", mask)
        self.assertFalse(r0.equal_dst_mask(r1))
        r1._set_component("dst_mask", mask)
        self.assertTrue(r0.equal_dst_mask(r1))
        r1._set_component("dst_mask", mask[::-1])
        self.assertFalse(r0.equal_dst_mask(r1))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
