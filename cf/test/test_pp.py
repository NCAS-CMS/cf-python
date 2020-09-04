import atexit
import datetime
import os
import tempfile
import unittest

import numpy

import cf

n_tmpfiles = 1
tmpfiles = [tempfile.mkstemp('_test_pp.nc', dir=os.getcwd())[1]
            for i in range(n_tmpfiles)]
[tmpfile] = tmpfiles


def _remove_tmpfiles():
    '''
'''
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class ppTest(unittest.TestCase):
    def setUp(self):
        self.ppfilename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'wgdos_packed.pp')

        self.new_table = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'new_STASH_to_CF.txt')

        text_file = open(self.new_table, 'w')
        text_file.write(
            '1!24!SURFACE TEMPERATURE AFTER TIMESTEP  !Pa!!!NEW_NAME!!')
        text_file.close()

        self.chunk_sizes = (100000, 300, 34)
        self.original_chunksize = cf.chunksize()

    def test_PP_load_stash2standard_name(self):
        f = cf.read(self.ppfilename)[0]
        self.assertEqual(f.identity(), 'surface_temperature')
        self.assertEqual(f.Units, cf.Units('K'))

        for merge in (True, False):
            cf.load_stash2standard_name(self.new_table, merge=merge)
            f = cf.read(self.ppfilename)[0]
            self.assertEqual(f.identity(), 'NEW_NAME')
            self.assertEqual(f.Units, cf.Units('Pa'))
            cf.load_stash2standard_name()
            f = cf.read(self.ppfilename)[0]
            self.assertEqual(f.identity(), 'surface_temperature')
            self.assertEqual(f.Units, cf.Units('K'))

        cf.load_stash2standard_name()

    def test_PP_WGDOS_UNPACKING(self):
        f = cf.read(self.ppfilename)[0]

        self.assertTrue(f.minimum() > 221.71,
                        'Bad unpacking of WGDOS packed data')
        self.assertTrue(f.maximum() < 310.45,
                        'Bad unpacking of WGDOS packed data')

        array = f.array

        for chunksize in self.chunk_sizes:
            cf.chunksize(chunksize)

            f = cf.read(self.ppfilename)[0]

            for fmt in ('NETCDF4', 'CFA4'):
                # print (fmt)
                # f.dump()
                # print (repr(f.dtype))
                # print (f._FillValue)
                # print (type(f._FillValue))
                # f._FillValue = numpy.array(f._FillValue , dtype='float32')
                cf.write(f, tmpfile, fmt=fmt)
                g = cf.read(tmpfile)[0]

                self.assertTrue((f.array == array).all(),
                                'Bad unpacking of PP WGDOS packed data')

                self.assertTrue(f.equals(g, verbose=2),
                                'Bad writing/reading. fmt='+fmt)

        cf.chunksize(self.original_chunksize)

# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
