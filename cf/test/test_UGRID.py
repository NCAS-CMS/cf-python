import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import cf

warnings = False

# Set up temporary files
n_tmpfiles = 1
tmpfiles = [
    tempfile.mkstemp("_test_read_write.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
[tmpfile1] = tmpfiles


def _remove_tmpfiles():
    """Remove temporary files created during tests."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class UGRIDTest(unittest.TestCase):
    """Test UGRID field constructs."""

    filename1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_1.nc"
    )

    filename2 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_2.nc"
    )

    def setUp(self):
        """Preparations called immediately before each test method."""
        # Disable log messages to silence expected warnings
        cf.LOG_LEVEL("DISABLE")
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows: cf.LOG_LEVEL('DEBUG')
        #
        # < ... test code ... >
        # cf.log_level('DISABLE')

    def test_UGRID_read(self):
        """Test reading of UGRID files."""
        f1 = cf.read(self.filename1)

        self.assertEqual(len(f1), 3)
        for g in f1:
            self.assertEqual(len(g.domain_topologies()), 1)
            self.assertEqual(len(g.auxiliary_coordinates()), 2)
            self.assertEqual(len(g.dimension_coordinates()), 1)

            for aux in g.auxiliary_coordinates().values():
                self.assertTrue(aux.has_data())

            if g.domain_topology().get_cell() == "face":
                self.assertEqual(len(g.cell_connectivities()), 1)
                self.assertEqual(
                    g.cell_connectivity().get_connectivity(), "edge"
                )

        # Check that all fields have the same mesh id
        mesh_ids1 = set(g.get_mesh_id() for g in f1)
        self.assertEqual(len(mesh_ids1), 1)

        f2 = cf.read(self.filename2)
        self.assertEqual(len(f2), 3)
        for g in f2:
            self.assertEqual(len(g.domain_topologies()), 1)
            self.assertEqual(len(g.auxiliary_coordinates()), 2)
            self.assertEqual(len(g.dimension_coordinates()), 1)

            cell = g.domain_topology().get_cell()
            if cell in ("edge", "face"):
                for aux in g.auxiliary_coordinates().values():
                    self.assertFalse(aux.has_data())

            if cell == "face":
                self.assertEqual(len(g.cell_connectivities()), 1)
                self.assertEqual(
                    g.cell_connectivity().get_connectivity(), "edge"
                )

        # Check that all fields have the same mesh id
        mesh_ids2 = set(g.get_mesh_id() for g in f2)
        self.assertEqual(len(mesh_ids2), 1)

        # Check that the different files have different mesh ids
        self.assertNotEqual(mesh_ids1, mesh_ids2)

    def test_UGRID_data(self):
        """Test reading of UGRID data."""
        node1, face1, edge1 = cf.read(self.filename1)
        node2, face2, edge2 = cf.read(self.filename2)

        # Domain topology arrays
        domain_topology1 = face1.domain_topology()
        self.assertTrue(
            (
                domain_topology1.array
                == np.array([[2, 3, 1, 0], [4, 5, 3, 2], [1, 3, 6, -99]])
            ).all()
        )
        self.assertTrue(domain_topology1.equals(face2.domain_topology()))

        domain_topology1 = edge1.domain_topology()
        self.assertTrue(
            (
                domain_topology1.array
                == np.array(
                    [
                        [1, 6],
                        [3, 6],
                        [3, 1],
                        [0, 1],
                        [2, 0],
                        [2, 3],
                        [2, 4],
                        [5, 4],
                        [3, 5],
                    ]
                )
            ).all()
        )
        self.assertTrue(domain_topology1.equals(edge2.domain_topology()))

        # Cell connectivity arrays
        cell_connectivity1 = face1.cell_connectivity()
        self.assertTrue(
            (
                cell_connectivity1.array
                == np.array(
                    [
                        [0, 1, 2, -99, -99],
                        [1, 0, -99, -99, -99],
                        [2, 0, -99, -99, -99],
                    ]
                )
            ).all()
        )
        self.assertTrue(cell_connectivity1.equals(face2.cell_connectivity()))

    def test_read_UGRID_domain(self):
        """Test reading of UGRID files into domains."""
        d1 = cf.read(self.filename1, domain=True)

        self.assertEqual(len(d1), 3)
        for g in d1:
            self.assertIsInstance(g, cf.Domain)
            self.assertEqual(len(g.domain_topologies()), 1)
            self.assertEqual(len(g.auxiliary_coordinates()), 2)
            self.assertEqual(len(g.dimension_coordinates()), 0)

            for aux in g.auxiliary_coordinates().values():
                self.assertTrue(aux.has_data())

            if g.domain_topology().get_cell() == "face":
                self.assertEqual(len(g.cell_connectivities()), 1)
                self.assertEqual(
                    g.cell_connectivity().get_connectivity(), "edge"
                )

        # Check that all domains have the same mesh id
        mesh_ids1 = set(g.get_mesh_id() for g in d1)
        self.assertEqual(len(mesh_ids1), 1)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
