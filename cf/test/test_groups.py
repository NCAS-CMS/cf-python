import atexit
import datetime
import os
import tempfile
import unittest

import netCDF4
#import netcdf_flattener

import cf


n_tmpfiles = 2
tmpfiles = [tempfile.mktemp('_test_groups.nc', dir=os.getcwd())
            for i in range(n_tmpfiles)]
(ungrouped_file,
 grouped_file,
) = tmpfiles

def _remove_tmpfiles():
    '''Remove temporary files created during tests.

    '''
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass

atexit.register(_remove_tmpfiles)


class GroupsTest(unittest.TestCase):
    def setUp(self):
        # Disable log messages to silence expected warnings
        cf.LOG_LEVEL('DISABLE')
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows:
        #
        # cf.LOG_LEVEL('DEBUG')
        # < ... test code ... >
        # cf.LOG_LEVEL('DISABLE')

    def test_groups(self):
        f = cf.example_field(1)

#        # Add a second grid mapping    
#        datum = cf.Datum(parameters={'earth_radius': 7000000})
#        conversion = cf.CoordinateConversion(
#            parameters={'grid_mapping_name': 'latitude_longitude'})
#        
#        grid = cf.CoordinateReference(
#            coordinate_conversion=conversion,
#            datum=datum,
#            coordinates=['auxiliarycoordinate0', 'auxiliarycoordinate1']
#        )
#
#        f.set_construct(grid)
#        
#        grid0 = f.construct('grid_mapping_name:rotated_latitude_longitude')
#        grid0.del_coordinate('auxiliarycoordinate0')
#        grid0.del_coordinate('auxiliarycoordinate1')
#        
#        f.dump()
        
        ungrouped_file = 'ungrouped1.nc'
        cf.write(f, ungrouped_file)
        g = cf.read(ungrouped_file)[0]
        self.assertTrue(f.equals(g, verbose=2))

        grouped_file = 'delme1.nc'
        filename = grouped_file

        # ------------------------------------------------------------
        # Move the field construct to the /forecast/model group
        # ------------------------------------------------------------
        g.nc_set_variable_groups(['forecast', 'model'])
        cf.write(g, filename)
        
        nc = netCDF4.Dataset(filename, 'r')
        self.assertIn(
            f.nc_get_variable(),
            nc.groups['forecast'].groups['model'].variables
        )
        nc.close()
        
        h = cf.read(filename, verbose=1)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))
        
        # ------------------------------------------------------------
        # Move constructs one by one to the /forecast group
        # ------------------------------------------------------------
        for name in ( #'time',  # Dimension coordinate
                     'grid_latitude',  # Dimension coordinate
                     'longitude', # Auxiliary coordinate
                     'measure:area',  # Cell measure
                     'surface_altitude',  # Domain ancillary
                     'air_temperature standard_error',  # Field ancillary
                     'grid_mapping_name:rotated_latitude_longitude',
    ):
#        for name in ('grid_latitude',):  # Dimension coordinate
            print(9999999999, name)
            g.construct(name).nc_set_variable_groups(['forecast'])
            cf.write(g, filename, verbose=1)
            g.dump()
            # Check that the variable is in the right group
            nc = netCDF4.Dataset(filename, 'r')
            self.assertIn(
                f.construct(name).nc_get_variable(),
                nc.groups['forecast'].variables)
            nc.close()

            # Check that the field construct hasn't changed
            h = cf.read(filename, verbose=-1)
            self.assertEqual(len(h), 1, repr(h))
            self.assertTrue(f.equals(h[0], verbose=2), name)
        
        # ------------------------------------------------------------
        # Move bounds to the /forecast group
        # ------------------------------------------------------------
        name = 'grid_latitude'
        g.construct(name).bounds.nc_set_variable_groups(['forecast'])
        cf.write(g, filename)
        
        nc = netCDF4.Dataset(filename, 'r')
        self.assertIn(
            f.construct(name).bounds.nc_get_variable(),
            nc.groups['forecast'].variables)
        nc.close()

        h = cf.read(filename)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))
        
        f.dump()
        
    def test_groups_geometry(self):
        f = cf.example_field(6)

#        return True
#        pnc = cf.PartNodeCountProperties()
#        pnc.set_property('long_name', 'part node count')
#        pnc.nc_set_variable('part_node_count')
#        f.construct('longitude').set_part_node_count(pnc)
            
        ungrouped_file = 'ungrouped1.nc'
        cf.write(f, ungrouped_file)
        g = cf.read(ungrouped_file)[0]
        self.assertTrue(f.equals(g, verbose=2))

        grouped_file = 'delme2.nc'
        filename = grouped_file

        # ------------------------------------------------------------
        # Move the field construct to the /forecast/model group
        # ------------------------------------------------------------
        g.nc_set_variable_groups(['forecast', 'model'])
        cf.write(g, filename)

        nc = netCDF4.Dataset(filename, 'r')
        self.assertIn(
            f.nc_get_variable(),
            nc.groups['forecast'].groups['model'].variables
        )
        nc.close()
        
        h = cf.read(filename)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))
        
        # ------------------------------------------------------------
        # Move the geometry container to the /forecast group
        # ------------------------------------------------------------
        g.nc_set_geometry_variable_groups(['forecast'])
        cf.write(g, filename)

        # Check that the variable is in the right group
        nc = netCDF4.Dataset(filename, 'r')
        self.assertIn(            
            f.nc_get_geometry_variable(),
            nc.groups['forecast'].variables)
        nc.close()

        # Check that the field construct hasn't changed
        h = cf.read(filename)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))
        
        # ------------------------------------------------------------
        # Move a node coordinate variable to the /forecast group
        # ------------------------------------------------------------
        g.construct('longitude').bounds.nc_set_variable_groups(['forecast'])
        cf.write(g, filename)

        # Check that the variable is in the right group
        nc = netCDF4.Dataset(filename, 'r')
        self.assertIn(            
            f.construct('longitude').bounds.nc_get_variable(),
            nc.groups['forecast'].variables)
        nc.close()

        # Check that the field construct hasn't changed
        h = cf.read(filename)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))

        # ------------------------------------------------------------
        # Move a node count variable to the /forecast group
        # ------------------------------------------------------------
        ncvar = g.construct('longitude').get_node_count().nc_get_variable()
        g.nc_set_component_variable_groups('node_count', ['forecast'])

        cf.write(g, filename)

        # Check that the variable is in the right group
        nc = netCDF4.Dataset(filename, 'r')
        self.assertIn(            
            ncvar,
            nc.groups['forecast'].variables)
        nc.close()

        # Check that the field construct hasn't changed
        h = cf.read(filename, verbose=1)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))

        # ------------------------------------------------------------
        # Move a part node count variable to the /forecast group
        # ------------------------------------------------------------
        ncvar = (
            g.construct('longitude').get_part_node_count().nc_get_variable()
        )
        g.nc_set_component_variable_groups('part_node_count', ['forecast'])

        cf.write(g, filename)

        # Check that the variable is in the right group
        nc = netCDF4.Dataset(filename, 'r')
        self.assertIn(            
            ncvar,
            nc.groups['forecast'].variables)
        nc.close()

        # Check that the field construct hasn't changed
        h = cf.read(filename)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))

        # ------------------------------------------------------------
        # Move interior ring variable to the /forecast group
        # ------------------------------------------------------------
        g.nc_set_component_variable('interior_ring', 'interior_ring')
        g.nc_set_component_variable_groups('interior_ring', ['forecast'])

        cf.write(g, filename)

        # Check that the variable is in the right group
        nc = netCDF4.Dataset(filename, 'r')
        self.assertIn(            
            f.construct('longitude').get_interior_ring().nc_get_variable(),
            nc.groups['forecast'].variables)
        nc.close()

        # Check that the field construct hasn't changed
        h = cf.read(filename, verbose=1)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))
        
    def test_groups_compression(self):
        f = cf.example_field(4)

#        return True
        f.compress('indexed_contiguous', inplace=True)
        f.data.get_count().nc_set_variable('count')
        f.data.get_index().nc_set_variable('index')
        
        ungrouped_file = 'ungrouped1.nc'
        cf.write(f, ungrouped_file , verbose=1)
        g = cf.read(ungrouped_file)[0]
        self.assertTrue(f.equals(g, verbose=2))

        grouped_file = 'delme3.nc'
        filename = grouped_file

        # ------------------------------------------------------------
        # Move the field construct to the /forecast/model group
        # ------------------------------------------------------------
        g.nc_set_variable_groups(['forecast', 'model'])
        # ------------------------------------------------------------
        # Move the count variable to the /forecast group
        # ------------------------------------------------------------        
        g.data.get_count().nc_set_variable_groups(['forecast'])
        # ------------------------------------------------------------
        # Move the index variable to the /forecast group
        # ------------------------------------------------------------        
        g.data.get_index().nc_set_variable_groups(['forecast'])
        # ------------------------------------------------------------
        # Move the coordinates that span the element dimension to the
        # /forecast group
        # ------------------------------------------------------------
        name = 'altitude'
        g.construct(name).nc_set_variable_groups(['forecast'])
        # ------------------------------------------------------------
        # Move the sample dimension to the /forecast group
        # ------------------------------------------------------------        
        g.data.get_count().nc_set_sample_dimension_groups(['forecast'])
        
        cf.write(g, filename, verbose=1)

        nc = netCDF4.Dataset(filename, 'r')
        self.assertIn(
            f.nc_get_variable(),
            nc.groups['forecast'].groups['model'].variables
        )
        self.assertIn(
            f.data.get_count().nc_get_variable(),
            nc.groups['forecast'].variables
        )
        self.assertIn(
            f.data.get_index().nc_get_variable(),
            nc.groups['forecast'].variables
        )
        self.assertIn(
            f.construct('altitude').nc_get_variable(),
            nc.groups['forecast'].variables)
        nc.close()
        
        h = cf.read(filename, verbose=1)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))

#--- End: class

#        netcdf_flattener.flatten(i, o)
#
#        o.close()
#
#        h = cf.read('tmp.nc')[0]
#
#        h.del_property('flattener_name_mapping_attributes')
#        h.del_property('flattener_name_mapping_variables')
#        h.del_property('flattener_name_mapping_dimensions')


#        i = netCDF4.Dataset(ungrouped_file, 'r')
#        o = netCDF4.Dataset('tmp.nc', 'w')
#
#        netcdf_flattener.flatten(i, o)
#
#        o.close()
#
#        h = cf.read('tmp.nc')[0]
#
#        h.del_property('flattener_name_mapping_attributes')
#        h.del_property('flattener_name_mapping_variables')
#        h.del_property('flattener_name_mapping_dimensions')


if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    cf.environment()
    print()
    unittest.main(verbosity=2)

