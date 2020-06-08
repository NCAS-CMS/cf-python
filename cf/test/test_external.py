from __future__ import print_function
import datetime
import os
import tempfile
import time
import unittest

import numpy

import cf

# def _make_files():
#     '''
#     '''
#     def _pp(filename, parent=False, external=False, combined=False,
#             external_missing=False):
#         '''
#         '''
#         nc = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
#
#         nc.createDimension('grid_latitude', 10)
#         nc.createDimension('grid_longitude', 9)
#
#         nc.Conventions = 'CF-1.7'
#         if parent:
#             nc.external_variables = 'areacella'
#
#         if parent or combined or external_missing:
#             grid_latitude = nc.createVariable(dimensions=('grid_latitude',),
#                                               datatype='f8',
#                                               varname='grid_latitude')
#             grid_latitude.setncatts(
#                 {'units': 'degrees', 'standard_name': 'grid_latitude'})
#             grid_latitude[...] = range(10)
#
#             grid_longitude = nc.createVariable(
#                 dimensions=('grid_longitude',),
#                 datatype='f8',
#                 varname='grid_longitude'
#             )
#             grid_longitude.setncatts(
#                 {'units': 'degrees', 'standard_name': 'grid_longitude'})
#             grid_longitude[...] = range(9)
#
#             latitude = nc.createVariable(
#                 dimensions=('grid_latitude', 'grid_longitude'),
#                 datatype='i4',
#                 varname='latitude'
#             )
#             latitude.setncatts(
#                 {'units': 'degree_N', 'standard_name': 'latitude'})
#
#             latitude[...] = numpy.arange(90).reshape(10, 9)
#
#             longitude = nc.createVariable(
#                 dimensions=('grid_longitude', 'grid_latitude'),
#                 datatype='i4',
#                 varname='longitude'
#             )
#             longitude.setncatts(
#                 {'units': 'degreeE', 'standard_name': 'longitude'})
#             longitude[...] = numpy.arange(90).reshape(9, 10)
#
#             eastward_wind = nc.createVariable(
#                 dimensions=('grid_latitude', 'grid_longitude'),
#                 datatype='f8',
#                 varname=u'eastward_wind'
#             )
#             eastward_wind.coordinates = u'latitude longitude'
#             eastward_wind.standard_name = 'eastward_wind'
#             eastward_wind.cell_methods = (
#                 'grid_longitude: mean (interval: 1 day comment: ok) '
#                 'grid_latitude: maximum where sea'
#             )
#             eastward_wind.cell_measures = 'area: areacella'
#             eastward_wind.units = 'm s-1'
#             eastward_wind[...] = numpy.arange(90).reshape(10, 9) - 45.5
#
#         if external or combined:
#             areacella = nc.createVariable(
#                 dimensions=('grid_longitude', 'grid_latitude'),
#                 datatype='f8',
#                 varname='areacella'
#             )
#             areacella.setncatts(
#                 {'units': 'm2', 'standard_name': 'cell_area'})
#             areacella[...] = numpy.arange(90).reshape(9, 10) + 100000.5
#
#         nc.close()
#     # --- End: def
#
#     parent_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                'parent.nc')
#     external_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                  'external.nc')
#     combined_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                  'combined.nc')
#     external_missing_file = os.path.join(
#         os.path.dirname(os.path.abspath(__file__)), 'external_missing.nc')
#
#     _pp(parent_file          , parent=True)
#     _pp(external_file        , external=True)
#     _pp(combined_file        , combined=True)
#     _pp(external_missing_file, external_missing=True)
#
#     return parent_file, external_file, combined_file, external_missing_file
#
#
# (parent_file,
#  external_file,
#  combined_file,
#  external_missing_file) = _make_files()


class ExternalVariableTest(unittest.TestCase):
    def setUp(self):
        self.parent_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'parent.nc')
        self.external_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'external.nc')
        self.combined_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'combined.nc')
        self.external_missing_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'external_missing.nc')

        self.test_only = []

        (fd, self.tempfilename) = tempfile.mkstemp(
            suffix='.nc', prefix='cf_', dir='.')
        os.close(fd)
        (fd, self.tempfilename_parent) = tempfile.mkstemp(
            suffix='.nc', prefix='cf_parent_', dir='.')
        os.close(fd)
        (fd, self.tempfilename_external) = tempfile.mkstemp(
            suffix='.nc', prefix='cf_external_', dir='.')
        os.close(fd)

    def tearDown(self):
        os.remove(self.tempfilename)
        os.remove(self.tempfilename_parent)
        os.remove(self.tempfilename_external)

    def test_EXTERNAL_READ(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Read the parent file on its own, without the external file
        f = cf.read(self.parent_file, verbose=0)

        for i in f:
            _ = repr(i)
            _ = str(i)
            _ = i.dump(display=False)

        self.assertTrue(len(f) == 1)
        f = f[0]

        cell_measure = f.constructs.filter_by_identity('measure:area').value()

        self.assertTrue(cell_measure.nc_get_external())
        self.assertTrue(cell_measure.nc_get_variable() == 'areacella')
        self.assertTrue(cell_measure.properties() == {})
        self.assertFalse(cell_measure.has_data())

        # External file contains only the cell measure variable
        f = cf.read(self.parent_file, external=[self.external_file],
                    verbose=0)

        c = cf.read(self.combined_file, verbose=0)

        for i in c + f:
            _ = repr(i)
            _ = str(i)
            _ = i.dump(display=False)

        cell_measure = f[0].constructs.filter_by_identity(
            'measure:area').value()

        self.assertTrue(len(f) == 1)
        self.assertTrue(len(c) == 1)

        for i in range(len(f)):
            self.assertTrue(c[i].equals(f[i], verbose=2))

        # External file contains other variables
        f = cf.read(self.parent_file, external=self.combined_file,
                    verbose=0)

        for i in f:
            _ = repr(i)
            _ = str(i)
            _ = i.dump(display=False)

        self.assertTrue(len(f) == 1)
        self.assertTrue(len(c) == 1)

        for i in range(len(f)):
            self.assertTrue(c[i].equals(f[i], verbose=2))

        # Two external files
        f = cf.read(
            self.parent_file,
            external=[self.external_file, self.external_missing_file],
            verbose=0
        )

        for i in f:
            _ = repr(i)
            _ = str(i)
            _ = i.dump(display=False)

        self.assertTrue(len(f) == 1)
        self.assertTrue(len(c) == 1)

        for i in range(len(f)):
            self.assertTrue(c[i].equals(f[i], verbose=2))

    def test_EXTERNAL_WRITE(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        parent = cf.read(self.parent_file)
        combined = cf.read(self.combined_file)

        # External file contains only the cell measure variable
        f = cf.read(self.parent_file, external=self.external_file)

        cf.write(f, self.tempfilename)
        g = cf.read(self.tempfilename)

        self.assertTrue(len(g) == len(combined))

        for i in range(len(g)):
            self.assertTrue(combined[i].equals(g[i], verbose=2))

        cell_measure = g[0].constructs('measure:area').value()

        self.assertFalse(cell_measure.nc_get_external())
        cell_measure.nc_set_external(True)
        self.assertTrue(cell_measure.nc_get_external())
        self.assertTrue(cell_measure.properties())
        self.assertTrue(cell_measure.has_data())

        self.assertTrue(
            g[0].constructs.filter_by_identity(
                'measure:area').value().nc_get_external()
        )

        cf.write(g, self.tempfilename_parent,
                 external=self.tempfilename_external,
                 verbose=0)

        h = cf.read(self.tempfilename_parent, verbose=0)

        self.assertTrue(len(h) == len(parent))

        for i in range(len(h)):
            self.assertTrue(parent[i].equals(h[i], verbose=2))

        h = cf.read(self.tempfilename_external)
        external = cf.read(self.external_file)

        self.assertTrue(len(h) == len(external))

        for i in range(len(h)):
            self.assertTrue(external[i].equals(h[i], verbose=2))


# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    print(cf.environment(display=False))
    print()
    unittest.main(verbosity=2)
