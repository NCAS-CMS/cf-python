import datetime
import os
import unittest
import inspect

import cf


class RegridTest(unittest.TestCase):
    filename1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'regrid_file1.nc')
    filename2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'regrid_file2.nc')
    filename3 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'regrid_file3.nc')
    filename4 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'regrid_file4.nc')
    filename5 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_file3.nc')
    filename6 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_file2.nc')
    filename7 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'regrid_file5.nc')
    filename8 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'regrid_file6.nc')
    filename9 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'regrid_file7.nc')
    filename10 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'regrid_file8.nc')

    chunk_sizes = (300, 10000, 100000)[::-1]
    original_chunksize = cf.CHUNKSIZE()

    test_only = []
#    test_only = ('NOTHING!!!!!',)
#    test_only = ('test_Field_regrids',)
#    test_only = ('test_Field_regridc',)
#    test_only('test_Field_section',)
#    test_only('test_Data_section',)

    @unittest.skipUnless(cf._found_ESMF, "Requires esmf package.")
    def test_Field_regrids(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Set tolerance for array equality
        original_atol = cf.ATOL(1e-12)

        for chunksize in self.chunk_sizes:
            cf.CHUNKSIZE(chunksize)
            f1 = cf.read(self.filename1)[0]
            f2 = cf.read(self.filename2)[0]
            f3 = cf.read(self.filename3)[0]

            r = f1.regrids(f2, 'conservative')

            self.assertTrue(
                f3.equals(r, verbose=2),
                'destination = global Field, CHUNKSIZE = %s' % chunksize
            )
            r = f1.regrids(f2, method='conservative')

            self.assertTrue(
                f3.equals(r, verbose=2),
                'destination = global Field, CHUNKSIZE = %s' % chunksize
            )
            dst = {'longitude': f2.dim('X'), 'latitude': f2.dim('Y')}
            r = f1.regrids(dst, 'conservative', dst_cyclic=True)

            self.assertTrue(
                f3.equals(r, verbose=2),
                'destination = global dict, CHUNKSIZE = %s' % chunksize
            )
            r = f1.regrids(dst, method='conservative', dst_cyclic=True)
            self.assertTrue(
                f3.equals(r, verbose=2),
                'destination = global dict, CHUNKSIZE = %s' % chunksize
            )

            f4 = cf.read(self.filename4)[0]
            f5 = cf.read(self.filename5)[0]

            r = f1.regrids(f5, 'linear')
            self.assertTrue(
                f4.equals(r, verbose=2),
                'destination = regional Field, CHUNKSIZE = %s' % chunksize
            )

            r = f1.regrids(f5, method='linear')
            self.assertTrue(
                f4.equals(r, verbose=2),
                'destination = regional Field, CHUNKSIZE = %s' % chunksize
            )
        # --- End: for
        cf.CHUNKSIZE(self.original_chunksize)

        f6 = cf.read(self.filename6)[0]
        with self.assertRaises(Exception):
            f1.regridc(f6, axes='T', method='linear')

        cf.ATOL(original_atol)

    @unittest.skipUnless(cf._found_ESMF, "Requires esmf package.")
    def test_Field_regridc(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        original_atol = cf.ATOL(1e-12)

        for chunksize in self.chunk_sizes:
            cf.CHUNKSIZE(chunksize)
            f1 = cf.read(self.filename7)[0]
            f2 = cf.read(self.filename8)[0]
            f3 = cf.read(self.filename9)[0]
            self.assertTrue(
                f3.equals(f1.regridc(
                    f2, axes='T', method='linear'), verbose=2),
                'destination = time series, CHUNKSIZE = %s' % chunksize
            )
            f4 = cf.read(self.filename1)[0]
            f5 = cf.read(self.filename2)[0]
            f6 = cf.read(self.filename10)[0]
            self.assertTrue(
                f6.equals(f4.regridc(
                    f5, axes=('X', 'Y'), method='conservative'), verbose=2),
                'destination = global Field, CHUNKSIZE = %s' % chunksize
            )
            self.assertTrue(
                f6.equals(f4.regridc(
                    f5, axes=('X', 'Y'), method='conservative'), verbose=2),
                'destination = global Field, CHUNKSIZE = %s' % chunksize
            )
            dst = {'X': f5.dim('X'), 'Y': f5.dim('Y')}
            self.assertTrue(
                f6.equals(
                    f4.regridc(dst, axes=('X', 'Y'), method='conservative'),
                    verbose=2
                ),
                'destination = global dict, CHUNKSIZE = %s' % chunksize
            )
            self.assertTrue(
                f6.equals(
                    f4.regridc(dst, axes=('X', 'Y'), method='conservative'),
                    verbose=2
                ),
                'destination = global dict, CHUNKSIZE = %s' % chunksize
            )
        # --- End: for
        cf.CHUNKSIZE(self.original_chunksize)

        cf.ATOL(original_atol)


# --- End: class

if __name__ == "__main__":
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print('')
    unittest.main(verbosity=2)
