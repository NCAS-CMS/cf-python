import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class NetCDFArrayTest(unittest.TestCase):
    def test_NetCDFArray_del_file_location(self):
        a = cf.NetCDFArray(("/data1/file1", "/data2/file2"), ("tas1", "tas2"))
        b = a.del_file_location("/data1")
        self.assertIsNot(b, a)
        self.assertEqual(b.get_filenames(), ("/data2/file2",))
        self.assertEqual(b.get_addresses(), ("tas2",))

        a = cf.NetCDFArray(
            ("/data1/file1", "/data2/file1", "/data2/file2"),
            ("tas1", "tas1", "tas2"),
        )
        b = a.del_file_location("/data2")
        self.assertEqual(b.get_filenames(), ("/data1/file1",))
        self.assertEqual(b.get_addresses(), ("tas1",))

        # Can't be left with no files
        self.assertEqual(b.file_locations(), ("/data1",))
        with self.assertRaises(ValueError):
            b.del_file_location("/data1/")

    def test_NetCDFArray_file_locations(self):
        a = cf.NetCDFArray("/data1/file1")
        self.assertEqual(a.file_locations(), ("/data1",))

        a = cf.NetCDFArray(("/data1/file1", "/data2/file2"))
        self.assertEqual(a.file_locations(), ("/data1", "/data2"))

        a = cf.NetCDFArray(("/data1/file1", "/data2/file2", "/data1/file2"))
        self.assertEqual(a.file_locations(), ("/data1", "/data2", "/data1"))

    def test_NetCDFArray_set_file_location(self):
        a = cf.NetCDFArray("/data1/file1", "tas")
        b = a.set_file_location("/home/user")
        self.assertIsNot(b, a)
        self.assertEqual(
            b.get_filenames(), ("/data1/file1", "/home/user/file1")
        )
        self.assertEqual(b.get_addresses(), ("tas", "tas"))

        a = cf.NetCDFArray(("/data1/file1", "/data2/file2"), ("tas1", "tas2"))
        b = a.set_file_location("/home/user")
        self.assertEqual(
            b.get_filenames(),
            (
                "/data1/file1",
                "/data2/file2",
                "/home/user/file1",
                "/home/user/file2",
            ),
        )
        self.assertEqual(b.get_addresses(), ("tas1", "tas2", "tas1", "tas2"))

        a = cf.NetCDFArray(("/data1/file1", "/data2/file1"), ("tas1", "tas2"))
        b = a.set_file_location("/home/user")
        self.assertEqual(
            b.get_filenames(),
            ("/data1/file1", "/data2/file1", "/home/user/file1"),
        )
        self.assertEqual(b.get_addresses(), ("tas1", "tas2", "tas1"))

        a = cf.NetCDFArray(("/data1/file1", "/data2/file1"), ("tas1", "tas2"))
        b = a.set_file_location("/data1")
        self.assertEqual(b.get_filenames(), a.get_filenames())
        self.assertEqual(b.get_addresses(), a.get_addresses())


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
