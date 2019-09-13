import datetime
import tempfile
import os
import unittest
import atexit
import inspect

import numpy

import cf

tmpfile  = tempfile.mktemp('.cf-python_test')
tmpfiles = [tmpfile]
def _remove_tmpfiles():
    '''
'''
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass
#--- End: def
atexit.register(_remove_tmpfiles)

class read_writeTest(unittest.TestCase):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'test_file.nc')
    chunk_sizes = (17, 34, 300, 100000)[::-1]
#    chunk_sizes = (100000,)
    original_chunksize = cf.CHUNKSIZE()

    test_only = []
#    test_only = ['NOTHING!!!!!']
#    test_only = ['test_write_reference_datetime']
#    test_only = ['test_read_write_unlimited']
    test_only = ['test_read_write_format']
#    test_only = ['test_write_datatype']
#    test_only = ['test_read_directory']
    
    def test_read_directory(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read('dir', aggregate=False)
        self.assertTrue(len(f) == 3, f)

        f = cf.read('dir', recursive=True, aggregate=False)
        self.assertTrue(len(f) == 5, f)

        f = cf.read(['dir', 'dir/subdir'], aggregate=False)
        self.assertTrue(len(f) == 5, f)

        f = cf.read(['dir/subdir', 'dir'], aggregate=False)
        self.assertTrue(len(f) == 5, f)

        f = cf.read(['dir', 'dir/subdir'], recursive=True, aggregate=False)
        self.assertTrue(len(f) == 7, f)

        f = cf.read('dir/subdir', aggregate=False)
        self.assertTrue(len(f) == 2, f)
        
        f = cf.read('dir/subdir', recursive=True, aggregate=False)
        self.assertTrue(len(f) == 2, f)
    #--- End: def

    def test_read_select(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # select on field list
        f = cf.read(self.filename, select='eastward_wind')
        g = cf.read(self.filename)
        self.assertTrue(f.equals(g, verbose=True), 'Bad read with select keyword')
    #--- End: def

    def test_read_squeeze(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # select on field list
        f = cf.read(self.filename, squeeze=True)
        f = cf.read(self.filename, unsqueeze=True)
        with self.assertRaises(Exception):
            f = cf.read(self.filename, unsqueeze=True, squeeze=True)
    #--- End: def

    def test_read_aggregate(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # select on field list
        f = cf.read(self.filename, aggregate=True)
        f = cf.read(self.filename, aggregate=False)
        f = cf.read(self.filename, aggregate={})
    #--- End: def

    def test_read_extra(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Test field keyword of cfdm.read
        filename = self.filename
        
        f = cf.read(filename)
        self.assertTrue(len(f) == 1, '\n'+str(f))

        f = cf.read(filename, extra=['auxiliary_coordinate'])
        self.assertTrue(len(f) == 4, '\n'+str(f))
        
        f = cf.read(filename, extra='cell_measure')
        self.assertTrue(len(f) == 2, '\n'+str(f))

        f = cf.read(filename, extra=['field_ancillary'])
        self.assertTrue(len(f) == 5, '\n'+str(f))
                
        f = cf.read(filename, extra='domain_ancillary', verbose=0)
        self.assertTrue(len(f) == 4, '\n'+str(f))

        f = cf.read(filename, extra=['field_ancillary', 'auxiliary_coordinate'])
        self.assertTrue(len(f) == 8, '\n'+str(f))
        
        self.assertTrue(len(cf.read(filename, extra=['domain_ancillary', 'auxiliary_coordinate'])) == 7)
        f = cf.read(filename, extra=['domain_ancillary', 'cell_measure', 'auxiliary_coordinate'])
        self.assertTrue(len(f) == 8, '\n'+str(f))
        
        f = cf.read(filename, extra=('field_ancillary', 'dimension_coordinate',
                                     'cell_measure', 'auxiliary_coordinate',
                                     'domain_ancillary'))
        self.assertTrue(len(f) == 15, '\n'+str(f))
    #--- End: def

    def test_read_write_format(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return
        
        for chunksize in self.chunk_sizes:   
            cf.CHUNKSIZE(chunksize) 
            for fmt in (
                        #'NETCDF3_CLASSIC',
                        #'NETCDF3_64BIT',
                        #'NETCDF4',
                        #'NETCDF4_CLASSIC',
                        #'CFA3', 
                        'CFA',):
                f = cf.read(self.filename)[0]
                f0 = f.copy()
#                print ('FORMAT=', fmt)
#                tmpfile = 'delme'+str(chunksize)+fmt+'.nc'
                cf.write(f, tmpfile, fmt=fmt)
                g = cf.read(tmpfile)
                self.assertTrue(len(g) == 1, g)
                g0 = g[0]
#                print (f0.dump())
#                print (g0.dump())
                self.assertTrue(f0.equals(g0, verbose=1),
                                'Bad read/write of format {!r}'.format(fmt))
        #--- End: for
    #--- End: def

    def test_read_write_netCDF4_compress_shuffle(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:   
            cf.CHUNKSIZE(chunksize) 
            f = cf.read(self.filename)[0]
            for fmt in ('NETCDF4',
                        'NETCDF4_CLASSIC',
                        'CFA4'):
                for no_shuffle in (True, False):
                    for compress in (4,): #range(10):
                        cf.write(f, tmpfile, fmt=fmt,
                                 compress=compress,
                                 no_shuffle=no_shuffle)
                        g = cf.read(tmpfile)[0]
                        self.assertTrue(
                            f.equals(g, verbose=True),
                            'Bad read/write with lossless compression: {0}, {1}, {2}'.format(
                                fmt, compress, no_shuffle))
        #--- End: for
        cf.CHUNKSIZE(self.original_chunksize) 
    #--- End: def

    def test_write_datatype(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for chunksize in self.chunk_sizes:   
            cf.CHUNKSIZE(chunksize) 
            f = cf.read(self.filename)[0] 
            self.assertTrue(f.dtype == numpy.dtype(float))
            cf.write(f, tmpfile, fmt='NETCDF4', 
                     datatype={numpy.dtype(float): numpy.dtype('float32')})
            g = cf.read(tmpfile)[0]
            self.assertTrue(g.dtype == numpy.dtype('float32'), 
                            'datatype read in is '+str(g.dtype))
        #--- End: for
        cf.CHUNKSIZE(self.original_chunksize) 
    #--- End: def

    def test_write_reference_datetime(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for reference_datetime in ('1751-2-3', '1492-12-30'):
            for chunksize in self.chunk_sizes:   
                cf.CHUNKSIZE(chunksize) 
                f = cf.read(self.filename)[0]
                t = cf.DimensionCoordinate(data=cf.Data(123, 'days since 1750-1-1'))

                t.standard_name = 'time'
                axisT = f.set_construct(cf.DomainAxis(1))
                f.set_construct(t, axes=[axisT])
                cf.write(f, tmpfile, fmt='NETCDF4', reference_datetime=reference_datetime)
                g = cf.read(tmpfile)[0]
                t = g.dimension_coordinate('T')
                self.assertTrue(t.Units == cf.Units('days since '+reference_datetime),
                                'Units written were '+repr(t.Units.reftime)+' not '+repr(reference_datetime))
        #--- End: for
        cf.CHUNKSIZE(self.original_chunksize) 
    #--- End: def

#    def test_write_HDF_chunks(self):
#        if self.test_only and inspect.stack()[0][3] not in self.test_only:
#            return
#            
#        for chunksize in self.chunk_sizes:   
#            for fmt in ('NETCDF3_CLASSIC', 'NETCDF4'):
#                cf.CHUNKSIZE(chunksize) 
#                f = cf.read(self.filename)[0]
#                f.HDF_chunks({'T': 10000, 1: 3, 'grid_lat': 222, 45:45})
#                cf.write(f, tmpfile, fmt=fmt, HDF_chunksizes={'X': 6})
#        #--- End: for
#        cf.CHUNKSIZE(self.original_chunksize) 
#    #--- End: def

    def test_read_write_unlimited(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for fmt in ('NETCDF4', 'NETCDF3_CLASSIC'):
            f = cf.read(self.filename)[0]
            
            f.domain_axes['domainaxis0'].nc_set_unlimited(True)
            cf.write(f, tmpfile, fmt=fmt)
            
            f = cf.read(tmpfile)[0]
            self.assertTrue(f.domain_axes['domainaxis0'].nc_is_unlimited())

        fmt = 'NETCDF4'
        f = cf.read(self.filename)[0]
        f.domain_axes['domainaxis0'].nc_set_unlimited(True)
        f.domain_axes['domainaxis2'].nc_set_unlimited(True)
        cf.write(f, tmpfile, fmt=fmt)
        
        f = cf.read(tmpfile)[0]
        self.assertTrue(f.domain_axes['domainaxis0'].nc_is_unlimited())
        self.assertTrue(f.domain_axes['domainaxis2'].nc_is_unlimited())
    #--- End: def

#--- End: class

if __name__ == "__main__":
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
