import datetime
import unittest

import healpix
import numpy as np

import cf

# Create matching lists of selected nested, ring, nuniq and zuniq
# indices for every refinement level.
indices = [
    (r, i, healpix.nest2ring(healpix.order2nside(r), i))
    for r in range(30)
    for i in (0, 7, (12 * 4**r) - 1)
]
refinement_levels, nested_indices, ring_indices = map(list, zip(*indices))

nuniq_indices = [
    i + 4 ** (1 + r) for r, i in zip(refinement_levels, nested_indices)
]

zuniq_indices = [
    (2 * i + 1) * 4 ** (29 - r)
    for r, i in zip(refinement_levels, nested_indices)
]


class DataTest(unittest.TestCase):
    """Unit tests for HEALPix utilities."""

    def test_HEALPix_uniq2zuniq(self):
        """Test _uniq2zuniq"""
        from cf.data.dask_utils_healpix import _uniq2zuniq

        self.assertTrue(
            np.array_equal(_uniq2zuniq(nuniq_indices), zuniq_indices)
        )

    def test_HEALPix_zuniq2uniq(self):
        """Test _zuniq2uniq"""
        from cf.data.dask_utils_healpix import _zuniq2uniq

        self.assertTrue(
            np.array_equal(_zuniq2uniq(zuniq_indices), nuniq_indices)
        )

    def test_HEALPix_zuniq2pix(self):
        """Test _zuniq2pix"""
        from cf.data.dask_utils_healpix import _zuniq2pix

        # nested
        order, i = _zuniq2pix(zuniq_indices, nest=True)

        self.assertTrue(np.array_equal(order, refinement_levels))
        self.assertTrue(np.array_equal(i, nested_indices))

        # ring
        with self.assertRaises(NotImplementedError):
            _zuniq2pix(zuniq_indices, nest=False)

    def test_HEALPix_pix2zuniq(self):
        """Test _pix2zuniq"""
        from cf.data.dask_utils_healpix import _pix2zuniq

        # nested
        z = [
            _pix2zuniq(r, i, nest=True)
            for r, i in zip(refinement_levels, nested_indices)
        ]

        self.assertTrue(np.array_equal(z, zuniq_indices))

        # ring
        z = [
            _pix2zuniq(r, i, nest=False)
            for r, i in zip(refinement_levels, ring_indices)
        ]

        self.assertTrue(np.array_equal(z, zuniq_indices))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
