import datetime
import tempfile
import os
import unittest
import atexit
import inspect
import shutil
import subprocess

import numpy

import cf

tmpfile = tempfile.mktemp('.cf_test')
tmpfileh = tempfile.mktemp('.cf_test')
tmpfilec = tempfile.mktemp('.cf_test')
tmpfile0 = tempfile.mktemp('.cf_test')
tmpfile1 = tempfile.mktemp('.cf_test')
tmpfiles = [tmpfile, tmpfileh, tmpfilec, tmpfile0, tmpfile1]


def _remove_tmpfiles():
    '''TODO
    '''
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass
    # --- End: for


atexit.register(_remove_tmpfiles)


class read_writeTest(unittest.TestCase):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'test_file.nc')

    string_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'string_char.nc')

    chunk_sizes = (17, 34, 300, 100000)[::-1]
    original_chunksize = cf.CHUNKSIZE()

    test_only = []
#    test_only = ['NOTHING!!!!!']
#    test_only = ['test_write_filename']
#    test_only = ['test_read_write_unlimited']
#    test_only = ['test_write_datatype']
#    test_only = ['test_read_directory']
#    test_only = ['test_read_string']
#    test_only = ['test_read_write_netCDF4_compress_shuffle']

    def test_write_filename(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        tmpfile = tempfile.mktemp('.cf_test')
        tmpfiles.append(tmpfile)

        f = cf.example_field(0)
        a = f.array

        cf.write(f, tmpfile)
        g = cf.read(tmpfile)

        with self.assertRaises(Exception):
            cf.write(g, tmpfile)

        self.assertTrue((a == g[0].array).all())

    def test_read_mask(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.example_field(0)

        N = f.size

        f.data[1, 1] = cf.masked
        f.data[2, 2] = cf.masked

        f.del_property('_FillValue', None)
        f.del_property('missing_value', None)

        cf.write(f, tmpfile)

        g = cf.read(tmpfile)[0]
        self.assertTrue(numpy.ma.count(g.data.array) == N - 2)

        g = cf.read(tmpfile, mask=False)[0]
        self.assertTrue(numpy.ma.count(g.data.array) == N)

        g.apply_masking(inplace=True)
        self.assertTrue(numpy.ma.count(g.data.array) == N - 2)

        f.set_property('_FillValue', 999)
        f.set_property('missing_value', -111)
        cf.write(f, tmpfile)

        g = cf.read(tmpfile)[0]
        self.assertTrue(numpy.ma.count(g.data.array) == N - 2)

        g = cf.read(tmpfile, mask=False)[0]
        self.assertTrue(numpy.ma.count(g.data.array) == N)

        g.apply_masking(inplace=True)
        self.assertTrue(numpy.ma.count(g.data.array) == N - 2)

    def test_read_directory(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        pwd = os.getcwd() + '/'

        dir = 'dir_'+inspect.stack()[0][3]

        try:
            os.mkdir(dir)
        except FileExistsError:
            pass
        except:
            raise ValueError("Can not mkdir {}{}".format(pwd, dir))

        f = 'test_file2.nc'
        try:
            os.symlink(pwd+f, pwd+dir+'/'+f)
        except FileExistsError:
            pass

        subdir = dir+'/subdir'
        try:
            os.mkdir(subdir)
        except FileExistsError:
            pass
        except:
            raise ValueError("Can not mkdir {}{}".format(pwd, subdir))

        for f in ('test_file3.nc', 'test_file.nc'):
            try:
                os.symlink(pwd+f, pwd+subdir+'/'+f)
            except FileExistsError:
                pass
        # --- End: for

        f = cf.read(dir, aggregate=False)
        self.assertTrue(len(f) == 1, f)

        f = cf.read(dir, recursive=True, aggregate=False)
        self.assertTrue(len(f) == 3, f)

        f = cf.read([dir, subdir], aggregate=False)
        self.assertTrue(len(f) == 3, f)

        f = cf.read([subdir, dir], aggregate=False)
        self.assertTrue(len(f) == 3, f)

        f = cf.read([dir, subdir], recursive=True, aggregate=False)
        self.assertTrue(len(f) == 5, f)

        f = cf.read(subdir, aggregate=False)
        self.assertTrue(len(f) == 2, f)

        f = cf.read(subdir, recursive=True, aggregate=False)
        self.assertTrue(len(f) == 2, f)

        shutil.rmtree(dir)

    def test_read_select(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # select on field list
        f = cf.read(self.filename, select='eastward_wind')
        g = cf.read(self.filename)
        self.assertTrue(f.equals(g, verbose=2),
                        'Bad read with select keyword')

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

        f = cf.read(filename, extra=['field_ancillary',
                                     'auxiliary_coordinate'])
        self.assertTrue(len(f) == 8, '\n'+str(f))

        self.assertTrue(len(cf.read(filename,
                                    extra=['domain_ancillary',
                                           'auxiliary_coordinate'])) == 7)
        f = cf.read(filename, extra=['domain_ancillary', 'cell_measure',
                                     'auxiliary_coordinate'])
        self.assertTrue(len(f) == 8, '\n'+str(f))

        f = cf.read(filename, extra=('field_ancillary', 'dimension_coordinate',
                                     'cell_measure', 'auxiliary_coordinate',
                                     'domain_ancillary'))
        self.assertTrue(len(f) == 15, '\n'+str(f))

    def test_read_write_format(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for string in (True, False):
            for chunksize in self.chunk_sizes:
                cf.CHUNKSIZE(chunksize)
                for fmt in ('NETCDF3_CLASSIC',
                            'NETCDF3_64BIT',
                            'NETCDF3_64BIT_OFFSET',
                            'NETCDF3_64BIT_DATA',
                            'NETCDF4',
                            'NETCDF4_CLASSIC',
                            'CFA',):
                    # print (fmt, string)
                    f = cf.read(self.filename)[0]
                    f0 = f.copy()
                    cf.write(f, tmpfile, fmt=fmt, verbose=0, string=string)
                    g = cf.read(tmpfile, verbose=0)
                    self.assertTrue(len(g) == 1, 'g = '+repr(g))
                    g0 = g[0]

                    self.assertTrue(
                        f0.equals(g0, verbose=1),
                        'Bad read/write of format {!r}'.format(fmt))

    def test_read_write_netCDF4_compress_shuffle(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        tmpfile = tempfile.mktemp('.cf_test')
        tmpfiles.append(tmpfile)

        for chunksize in self.chunk_sizes:
            cf.CHUNKSIZE(chunksize)
            f = cf.read(self.filename)[0]
            for fmt in ('NETCDF4',
                        'NETCDF4_CLASSIC',
                        'CFA4'):
                for shuffle in (True, False):
                    for compress in (4,):  # range(10):
                        cf.write(f, tmpfile, fmt=fmt,
                                 compress=compress,
                                 shuffle=shuffle)
                        g = cf.read(tmpfile)[0]
                        self.assertTrue(
                            f.equals(g, verbose=2),
                            "Bad read/write with lossless compression: "
                            "{0}, {1}, {2}".format(
                                fmt, compress, shuffle))
        # --- End: for
        cf.CHUNKSIZE(self.original_chunksize)

    def test_write_datatype(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        tmpfile = tempfile.mktemp('.cf_test')
        tmpfiles.append(tmpfile)

        for chunksize in self.chunk_sizes:
            cf.CHUNKSIZE(chunksize)
            f = cf.read(self.filename)[0]
            self.assertTrue(f.dtype == numpy.dtype(float))
            cf.write(f, tmpfile, fmt='NETCDF4',
                     datatype={numpy.dtype(float): numpy.dtype('float32')})
            g = cf.read(tmpfile)[0]
            self.assertTrue(g.dtype == numpy.dtype('float32'),
                            'datatype read in is '+str(g.dtype))

        cf.CHUNKSIZE(self.original_chunksize)

        # Keyword single
        f = cf.read(self.filename)[0]
        self.assertTrue(f.dtype == numpy.dtype(float))
        cf.write(f, tmpfile, fmt='NETCDF4', single=True)
        g = cf.read(tmpfile)[0]
        self.assertTrue(g.dtype == numpy.dtype('float32'),
                        'datatype read in is '+str(g.dtype))

        tmpfile2 = tempfile.mktemp('.cf_test')
        tmpfiles.append(tmpfile2)

        # Keyword double
        f = g
        self.assertTrue(f.dtype == numpy.dtype('float32'))
        cf.write(f, tmpfile2, fmt='NETCDF4', double=True)
        g = cf.read(tmpfile2)[0]
        self.assertTrue(g.dtype == numpy.dtype(float),
                        'datatype read in is '+str(g.dtype))

        for single in (True, False):
            for dousble in (True, False):
                with self.assertRaises(Exception):
                    _ = cf.write(g, double=double, single=single)
        # --- End: for

        datatype = {numpy.dtype(float): numpy.dtype('float32')}
        with self.assertRaises(Exception):
            _ = cf.write(g, datatype=datatype, single=True)

        with self.assertRaises(Exception):
            _ = cf.write(g, datatype=datatype, double=True)

    def test_write_reference_datetime(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for reference_datetime in ('1751-2-3', '1492-12-30'):
            for chunksize in self.chunk_sizes:
                cf.CHUNKSIZE(chunksize)
                f = cf.read(self.filename)[0]
                t = cf.DimensionCoordinate(data=cf.Data(
                    123, 'days since 1750-1-1'))

                t.standard_name = 'time'
                axisT = f.set_construct(cf.DomainAxis(1))
                f.set_construct(t, axes=[axisT])
                cf.write(f, tmpfile, fmt='NETCDF4',
                         reference_datetime=reference_datetime)
                g = cf.read(tmpfile)[0]
                t = g.dimension_coordinate('T')
                self.assertTrue(
                    t.Units == cf.Units('days since ' + reference_datetime),
                    ('Units written were ' + repr(t.Units.reftime)
                     + ' not ' + repr(reference_datetime)))
        # --- End: for
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
#        # --- End: for
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

        self.assertTrue(p.equals(p0, verbose=2))

    def test_read_CDL(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        subprocess.run(
            ' '.join(['ncdump', self.filename, '>', tmpfile]),
            shell=True, check=True)
        subprocess.run(
            ' '.join(['ncdump', '-h', self.filename, '>', tmpfileh]),
            shell=True, check=True)
        subprocess.run(
            ' '.join(['ncdump', '-c', self.filename, '>', tmpfilec]),
            shell=True, check=True)

        f0 = cf.read(self.filename)[0]
        f = cf.read(tmpfile)[0]
        h = cf.read(tmpfileh)[0]
        c = cf.read(tmpfilec)[0]

        self.assertTrue(f0.equals(f, verbose=2))

        self.assertTrue(f.construct('grid_latitude').equals(
            c.construct('grid_latitude'), verbose=2))
        self.assertTrue(f0.construct('grid_latitude').equals(
            c.construct('grid_latitude'), verbose=2))

        with self.assertRaises(Exception):
            _ = cf.read('test_read_write.py')

    def test_read_write_string(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.string_filename)

        n = int(len(f)/2)

        for i in range(0, n):

            j = i + n
            self.assertTrue(f[i].data.equals(f[j].data, verbose=1),
                            "{!r} {!r}".format(f[i], f[j]))
            self.assertTrue(f[j].data.equals(f[i].data, verbose=1),
                            "{!r} {!r}".format(f[j], f[i]))

        for string0 in (True, False):
            for fmt0 in ('NETCDF4',
                         'NETCDF3_CLASSIC',
                         'NETCDF4_CLASSIC',
                         'NETCDF3_64BIT',
                         'NETCDF3_64BIT_OFFSET',
                         'NETCDF3_64BIT_DATA'):
                # print ('fmt0=', fmt0)
                f0 = cf.read(self.string_filename)
                cf.write(f0, tmpfile0, fmt=fmt0, string=string0)

                for string1 in (True, False):
                    for fmt1 in ('NETCDF4',
                                 'NETCDF3_CLASSIC',
                                 'NETCDF4_CLASSIC',
                                 'NETCDF3_64BIT',
                                 'NETCDF3_64BIT_OFFSET',
                                 'NETCDF3_64BIT_DATA'):
                        # print ('fmt1=', fmt1)
                        f1 = cf.read(self.string_filename)
                        cf.write(f0, tmpfile1, fmt=fmt1, string=string1)

                        for i, j in zip(cf.read(tmpfile1), cf.read(tmpfile0)):
                            self.assertTrue(i.equals(j, verbose=1))
        # --- End: for

# --- End: class


if __name__ == "__main__":
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
