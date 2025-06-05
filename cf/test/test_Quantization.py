import datetime
import unittest

import cf


class QuantizationTest(unittest.TestCase):
    """Unit test for the Quantization class."""

    q = cf.Quantization({"quantization_nsd": 4, "algorithm": "bitgroom"})

    def test_Quantization_algorithm_parameters(self):
        """Test Quantization.algorithm_parameters."""
        self.assertEqual(
            cf.Quantization().algorithm_parameters(),
            {
                "bitgroom": "quantization_nsd",
                "bitround": "quantization_nsb",
                "digitround": "quantization_nsd",
                "granular_bitround": "quantization_nsd",
            },
        )

    def test_Quantization__str__(self):
        """Test Quantization.__str__."""
        self.assertEqual(str(self.q), "algorithm=bitgroom, quantization_nsd=4")

    def test_Quantization_dump(self):
        """Test Quantization.dump."""
        self.assertEqual(
            self.q.dump(display=False),
            "Quantization: \n"
            "    algorithm = 'bitgroom'\n"
            "    quantization_nsd = 4",
        )


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
