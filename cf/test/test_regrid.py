import datetime
import faulthandler
import os
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class RegridTest(unittest.TestCase):
    filename1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_file1.nc"
    )
    filename2 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_file2.nc"
    )
    filename3 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_file3.nc"
    )
    filename4 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_file4.nc"
    )
    filename5 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file3.nc"
    )
    filename6 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file2.nc"
    )
    filename7 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_file5.nc"
    )
    filename8 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_file6.nc"
    )
    filename9 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_file7.nc"
    )
    filename10 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_file8.nc"
    )

    chunk_sizes = (300, 10000, 100000)[::-1]

    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
    def test_Field_regrids(self):
        self.assertFalse(cf.regrid_logging())
        with cf.atol(1e-12):
            for chunksize in self.chunk_sizes:
                with cf.chunksize(chunksize):
                    f1 = cf.read(self.filename1)[0]
                    f2 = cf.read(self.filename2)[0]
                    f3 = cf.read(self.filename3)[0]
                    f4 = cf.read(self.filename4)[0]
                    f5 = cf.read(self.filename5)[0]

                    r = f1.regrids(f2, "conservative")
                    self.assertTrue(
                        f3.equals(r),
                        "destination=global Field, CHUNKSIZE={}".format(
                            chunksize
                        ),
                    )

                    dst = {"longitude": f2.dim("X"), "latitude": f2.dim("Y")}
                    r = f1.regrids(dst, "conservative", dst_cyclic=True)
                    self.assertTrue(
                        f3.equals(r),
                        "destination=global dict, CHUNKSIZE={}".format(
                            chunksize
                        ),
                    )

                    r = f1.regrids(dst, method="conservative", dst_cyclic=True)
                    self.assertTrue(
                        f3.equals(r),
                        "destination=global dict, CHUNKSIZE={}".format(
                            chunksize
                        ),
                    )

                    # Regrid global to regional roated pole
                    r = f1.regrids(f5, method="linear")
                    self.assertTrue(
                        f4.equals(r, verbose=3),
                        "destination=regional Field, CHUNKSIZE={}".format(
                            chunksize
                        ),
                    )

        f6 = cf.read(self.filename6)[0]
        with self.assertRaises(Exception):
            f1.regridc(f6, axes="T", method="linear")

    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
    def test_Field_regridc(self):
        self.assertFalse(cf.regrid_logging())
        with cf.atol(1e-11):
            for chunksize in self.chunk_sizes:
                self.assertFalse(cf.regrid_logging())
                with cf.chunksize(chunksize):
                    f1 = cf.read(self.filename7)[0]
                    f2 = cf.read(self.filename8)[0]
                    f3 = cf.read(self.filename9)[0]
                    self.assertTrue(
                        f3.equals(f1.regridc(f2, axes="T", method="linear")),
                        "destination=time series, CHUNKSIZE={}".format(
                            chunksize
                        ),
                    )
                    f4 = cf.read(self.filename1)[0]
                    f5 = cf.read(self.filename2)[0]
                    f6 = cf.read(self.filename10)[0]
                    self.assertTrue(
                        f6.equals(
                            f4.regridc(
                                f5, axes=("X", "Y"), method="conservative"
                            )
                        ),
                        "destination=global Field, CHUNKSIZE={}".format(
                            chunksize
                        ),
                    )
                    self.assertTrue(
                        f6.equals(
                            f4.regridc(
                                f5, axes=("X", "Y"), method="conservative"
                            )
                        ),
                        "destination=global Field, CHUNKSIZE={}".format(
                            chunksize
                        ),
                    )
                    dst = {"X": f5.dim("X"), "Y": f5.dim("Y")}
                    self.assertTrue(
                        f6.equals(
                            f4.regridc(
                                dst, axes=("X", "Y"), method="conservative"
                            )
                        ),
                        "destination=global dict, CHUNKSIZE={}".format(
                            chunksize
                        ),
                    )
                    self.assertTrue(
                        f6.equals(
                            f4.regridc(
                                dst, axes=("X", "Y"), method="conservative"
                            )
                        ),
                        "destination=global dict, CHUNKSIZE={}".format(
                            chunksize
                        ),
                    )

    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
    def test_Field_regrids_operator(self):
        self.assertFalse(cf.regrid_logging())

        with cf.atol(1e-12):
            f1 = cf.read(self.filename1)[0]
            f2 = cf.read(self.filename2)[0]
            f3 = cf.read(self.filename3)[0]
            f4 = cf.read(self.filename4)[0]
            f5 = cf.read(self.filename5)[0]

            op = f1.regrids(f2, "conservative", return_operator=True)
            r = f1.regrids(op)
            self.assertTrue(f3.equals(r))

            # Repeat
            r = f1.regrids(op)
            self.assertTrue(f3.equals(r))

            dst = {"longitude": f2.dim("X"), "latitude": f2.dim("Y")}
            op = f1.regrids(
                dst, "conservative", dst_cyclic=True, return_operator=True
            )
            r = f1.regrids(op)
            self.assertTrue(f3.equals(r))

            op = f1.regrids(
                dst,
                method="conservative",
                dst_cyclic=True,
                return_operator=True,
            )
            r = f1.regrids(op)
            self.assertTrue(f3.equals(r))

            # Regrid global to regional rotated pole
            op = f1.regrids(f5, method="linear", return_operator=True)
            r = f1.regrids(op)
            self.assertTrue(f4.equals(r))

        # Raise exception when the source grid does not match that of
        # the regrid operator
        op = f1.regrids(f2, "conservative", return_operator=True)
        with self.assertRaises(ValueError):
            f2.regrids(op)

    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
    def test_Field_regridc_operator(self):
        self.assertFalse(cf.regrid_logging())

        with cf.atol(1e-12):
            f1 = cf.read(self.filename7)[0]
            f2 = cf.read(self.filename8)[0]
            f3 = cf.read(self.filename9)[0]
            f4 = cf.read(self.filename1)[0]
            f5 = cf.read(self.filename2)[0]
            f6 = cf.read(self.filename10)[0]

            op = f1.regridc(
                f2, axes="T", method="linear", return_operator=True
            )
            self.assertTrue(f3.equals(f1.regridc(op)))

            op = f4.regridc(
                f5,
                axes=("X", "Y"),
                method="conservative",
                return_operator=True,
            )
            self.assertTrue(f6.equals(f4.regridc(op)))

            op = f4.regridc(
                f5,
                axes=("X", "Y"),
                method="conservative",
                return_operator=True,
            )
            self.assertTrue(f6.equals(f4.regridc(op)))

            dst = {
                "X": f5.dimension_coordinate("X"),
                "Y": f5.dimension_coordinate("Y"),
            }
            op = f4.regridc(
                dst,
                axes=("X", "Y"),
                method="conservative",
                return_operator=True,
            )

            self.assertTrue(f6.equals(f4.regridc(op)))
            self.assertTrue(f6.equals(f4.regridc(op)))

        # Raise exception when the source grid does not match that of
        # the regrid operator
        op = f1.regridc(f2, axes="T", method="linear", return_operator=True)
        with self.assertRaises(ValueError):
            f2.regrids(op)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
