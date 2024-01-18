import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf

# Create test domain topology object
c = cf.DomainTopology()
c.set_properties({"long_name": "Maps every face to its corner nodes"})
c.nc_set_variable("Mesh2_face_nodes")
data = cf.Data(
    [[2, 3, 1, 0], [6, 7, 3, 2], [1, 3, 8, -99]],
    dtype="i4",
)
data.masked_values(-99, inplace=True)
c.set_data(data)
c.set_cell("face")


class DomainTopologyTest(unittest.TestCase):
    """Unit test for the DomainTopology class."""

    d = c

    def setUp(self):
        """Preparations called immediately before each test method."""
        # Disable log messages to silence expected warnings
        cf.log_level("DISABLE")
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows:
        #
        # cf.LOG_LEVEL('DEBUG')
        # < ... test code ... >
        # cf.log_level('DISABLE')

    def test_DomainTopology__repr__str__dump(self):
        """Test all means of DomainTopology inspection."""
        d = self.d
        self.assertEqual(repr(d), "<CF DomainTopology: cell:face(3, 4) >")
        self.assertEqual(str(d), "cell:face(3, 4) ")
        self.assertEqual(
            d.dump(display=False),
            """Domain Topology: cell:face
    long_name = 'Maps every face to its corner nodes'
    Data(3, 4) = [[2, ..., --]]""",
        )

    def test_DomainTopology_copy(self):
        """Test the copy of DomainTopology."""
        d = self.d
        self.assertTrue(d.equals(d.copy()))

    def test_DomainTopology_data(self):
        """Test the data of DomainTopology."""
        d = self.d
        self.assertEqual(d.ndim, 1)

    def test_DomainTopology_cell(self):
        """Test the 'cell' methods of DomainTopology."""
        d = self.d.copy()
        self.assertTrue(d.has_cell())
        self.assertEqual(d.get_cell(), "face")
        self.assertEqual(d.del_cell(), "face")
        self.assertFalse(d.has_cell())
        self.assertIsNone(d.get_cell(None))
        self.assertIsNone(d.del_cell(None))

        with self.assertRaises(ValueError):
            d.get_cell()

        with self.assertRaises(ValueError):
            d.set_cell("bad value")

        self.assertIsNone(d.set_cell("face"))
        self.assertTrue(d.has_cell())
        self.assertEqual(d.get_cell(), "face")

    def test_DomainTopology_transpose(self):
        """Test the 'transpose' method of DomainTopology."""
        d = self.d.copy()
        e = d.transpose()
        self.assertTrue(d.equals(e))
        self.assertIsNone(d.transpose(inplace=True))

        for axes in ([1], [1, 0], [3]):
            with self.assertRaises(ValueError):
                d.transpose(axes)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
