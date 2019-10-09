import datetime
import tempfile
import os
import unittest
import atexit
import inspect
import subprocess

import numpy

import cf

tmpfile   = tempfile.mktemp('_cf-python_test')
tmpfileh  = tempfile.mktemp('_cf-python_test')
tmpfilec  = tempfile.mktemp('_cf-python_test')
tmpfiles = [tmpfile, tmpfileh, tmpfilec]
def _remove_tmpfiles():
    '''TODO
    '''
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass

        
atexit.register(_remove_tmpfiles)

class read_writeTest(unittest.TestCase):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'test_file.nc')
    chunk_sizes = (17, 34, 300, 100000)[::-1]
    original_chunksize = cf.CHUNKSIZE()

    test_only = []
#    test_only = ['NOTHING!!!!!']
#    test_only = ['test_write_reference_datetime']
#    test_only = ['test_read_write_unlimited']
#    test_only = ['test_read_write_format']
#    test_only = ['test_read_CDL']
#    test_only = ['test_read_directory']
#    test_only = ['test_read_write_netCDF4_compress_shuffle']

    def test_read_directory(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        pwd = os.getcwd() + '/'
        
        try:
            os.mkdir('dir')
        except FileExistsError:
            pass
        except:
            raise ValueError("Can not make 'dir'")
        else:
            f = 'test_file2.nc' 
            os.symlink(pwd+f, pwd+'dir/'+f)
            
        try:
            os.mkdir('dir/subdir')
        except FileExistsError:
            pass
        except:
            raise ValueError("Can not make 'dir/subdir'")
        else:
            for f in ('test_file3.nc', 'test_file4.nc'):            
                os.symlink(pwd+f, pwd+'dir/subdir/'+f)
                           
        f = cf.read('dir', aggregate=False)
        self.assertTrue(len(f) == 1, f)

        f = cf.read('dir', recursive=True, aggregate=False)
        self.assertTrue(len(f) == 3, f)

        f = cf.read(['dir', 'dir/subdir'], aggregate=False)
        self.assertTrue(len(f) == 3, f)

        f = cf.read(['dir/subdir', 'dir'], aggregate=False)
        self.assertTrue(len(f) == 3, f)

        f = cf.read(['dir', 'dir/subdir'], recursive=True, aggregate=False)
        self.assertTrue(len(f) == 5, f)

        f = cf.read('dir/subdir', aggregate=False)
        self.assertTrue(len(f) == 2, f)
        
        f = cf.read('dir/subdir', recursive=True, aggregate=False)
        self.assertTrue(len(f) == 2, f)


    def test_read_select(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # select on field list
        f = cf.read(self.filename, select='eastward_wind')
        g = cf.read(self.filename)
        self.assertTrue(f.equals(g, verbose=True), 'Bad read with select keyword')


    def test_read_squeeze(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # select on field list
        f = cf.read(self.filename, squeeze=True)
        f = cf.read(self.filename, unsqueeze=True)
        with self.assertRaises(Exception):
            f = cf.read(self.filename, unsqueeze=True, squeeze=True)


    def test_read_aggregate(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.filename, aggregate=True)
        f = cf.read(self.filename, aggregate=False)
        f = cf.read(self.filename, aggregate={})


    def test_read_extra(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Test field keyword of cf.read
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
                cf.write(f, tmpfile, fmt=fmt)
                g = cf.read(tmpfile)
                self.assertTrue(len(g) == 1, g)
                g0 = g[0]
                self.assertTrue(f0.equals(g0, verbose=1),
                                'Bad read/write of format {!r}'.format(fmt))
        #--- End: for


    def test_read_write_netCDF4_compress_shuffle(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        tmpfile = tempfile.mktemp('.cf-python_test')
        
        for chunksize in self.chunk_sizes:   
            cf.CHUNKSIZE(chunksize) 
            f = cf.read(self.filename)[0]
            for fmt in ('NETCDF4',
                        'NETCDF4_CLASSIC',
                        'CFA4'):
                for shuffle in (True, False):
                    for compress in (4,): #range(10):
                        cf.write(f, tmpfile, fmt=fmt,
                                 compress=compress,
                                 shuffle=shuffle)
                        g = cf.read(tmpfile)[0]
                        self.assertTrue(
                            f.equals(g, verbose=True),
                            'Bad read/write with lossless compression: {0}, {1}, {2}'.format(
                                fmt, compress, shuffle))
        #--- End: for
        cf.CHUNKSIZE(self.original_chunksize) 


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


    def test_read_pp(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        p = cf.read('wgdos_packed.pp')[0]            
        p0 = cf.read('wgdos_packed.pp',
                     um={'fmt': 'PP',
                         'endian': 'big',
                         'word_size': 4,
                         'version': 4.5,
                         'height_at_top_of_model': 23423.65})[0]

        self.assertTrue(p.equals(p0, verbose=True))
        

    def test_read_CDL(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        subprocess.run(' '.join(['ncdump', self.filename, '>', tmpfile]),
                       shell=True, check=True)
        subprocess.run(' '.join(['ncdump', '-h', self.filename, '>', tmpfileh]),
                                shell=True, check=True)
        subprocess.run(' '.join(['ncdump', '-c', self.filename, '>', tmpfilec]),
                       shell=True, check=True)

        f0 = cf.read(self.filename)[0]
        f = cf.read(tmpfile)[0]
        h = cf.read(tmpfileh)[0]
        c = cf.read(tmpfilec)[0]

        self.assertTrue(f0.equals(f, verbose=True))

        self.assertTrue(f.construct('grid_latitude').equals(c.construct('grid_latitude'), verbose=True))
        self.assertTrue(f0.construct('grid_latitude').equals(c.construct('grid_latitude'), verbose=True))

        with self.assertRaises(Exception):
            x = cf.read('test_read_write.py')
            
        
#--- End: class

if __name__ == "__main__":
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
