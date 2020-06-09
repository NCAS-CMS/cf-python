import datetime
import os
import sys
import unittest

import numpy

import cf


class create_fieldTest(unittest.TestCase):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'test_file.nc')
    chunk_sizes = (17, 34, 300, 100000)[::-1]

    def test_create_field(self):
        # Dimension coordinates
        dim1 = cf.DimensionCoordinate(
            data=cf.Data(numpy.arange(10.), 'degrees'))
        dim1.standard_name = 'grid_latitude'

        dim0 = cf.DimensionCoordinate(
            data=cf.Data(numpy.arange(9.) + 20, 'degrees'))
        dim0.standard_name = 'grid_longitude'
        dim0.data[-1] += 5
        bounds = cf.Data(numpy.array(
            [dim0.data.array-0.5, dim0.data.array+0.5]).transpose((1, 0)))
        bounds[-2, 1] = 30
        bounds[-1, :] = [30, 36]
        dim0.set_bounds(cf.Bounds(data=bounds))

        dim2 = cf.DimensionCoordinate(
            data=cf.Data([1.5]),
            bounds=cf.Bounds(data=cf.Data([[1, 2.]]))
        )
        dim2.standard_name = 'atmosphere_hybrid_height_coordinate'

        # Auxiliary coordinates
        ak = cf.DomainAncillary(data=cf.Data([10.], 'm'))
        ak.id = 'atmosphere_hybrid_height_coordinate_ak'
        bounds = cf.Bounds(data=cf.Data([[5, 15.]], units=ak.Units))
        ak.set_bounds(bounds)

        bk = cf.DomainAncillary(data=cf.Data([20.]))
        bk.id = 'atmosphere_hybrid_height_coordinate_bk'
        bounds = cf.Bounds(data=cf.Data([[14, 26.]]))
        bk.set_bounds(bounds)

        aux2 = cf.AuxiliaryCoordinate(
            data=cf.Data(numpy.arange(-45, 45, dtype='int32').reshape(10, 9),
                         units='degree_N'))
        aux2.standard_name = 'latitude'

        aux3 = cf.AuxiliaryCoordinate(
            data=cf.Data(numpy.arange(60, 150, dtype='int32').reshape(9, 10),
                         units='degreesE'))
        aux3.standard_name = 'longitude'

        aux4 = cf.AuxiliaryCoordinate(
            data=cf.Data(numpy.array(
                ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
                 'zeta', 'eta', 'theta', 'iota', 'kappa'],
                dtype='S'
            ))
        )
        aux4.standard_name = 'greek_letters'
        aux4[0] = cf.masked

        # Cell measures
        msr0 = cf.CellMeasure(
            data=cf.Data(1+numpy.arange(90.).reshape(9, 10)*1234, 'km 2'))
        msr0.measure = 'area'

        # Data
        data = cf.Data(numpy.arange(90.).reshape(10, 9), 'm s-1')

        properties = {'standard_name': 'eastward_wind'}

        f = cf.Field(properties=properties)

        axisX = f.set_construct(cf.DomainAxis(9))
        axisY = f.set_construct(cf.DomainAxis(10))
        axisZ = f.set_construct(cf.DomainAxis(1))

        f.set_data(data)

        x = f.set_construct(dim0)
        y = f.set_construct(dim1, axes=[axisY])
        z = f.set_construct(dim2, axes=[axisZ])

        lat = f.set_construct(aux2)
        lon = f.set_construct(aux3, axes=['X', axisY])
        f.set_construct(aux4, axes=['Y'])

        ak = f.set_construct(ak, axes=['Z'])
        bk = f.set_construct(bk, axes=[axisZ])

        # Coordinate references
        coordinate_conversion = cf.CoordinateConversion(
            parameters={'grid_mapping_name': 'rotated_latitude_longitude',
                        'grid_north_pole_latitude': 38.0,
                        'grid_north_pole_longitude': 190.0})
        ref0 = cf.CoordinateReference(
            coordinate_conversion=coordinate_conversion,
            coordinates=[x, y, lat, lon]
        )

        f.set_construct(msr0, axes=[axisX, 'Y'])

        f.set_construct(ref0)

        orog = cf.DomainAncillary()
        orog.standard_name = 'surface_altitude'
        orog.set_data(cf.Data(f.array*2, 'm'))
        orog.transpose([1, 0], inplace=True)

        orog_key = f.set_construct(orog, axes=['X', axisY])

        coordinate_conversion = cf.CoordinateConversion(
            parameters={
                'standard_name': 'atmosphere_hybrid_height_coordinate'
            },
            domain_ancillaries={
                'orog': orog_key,
                'a': ak,
                'b': bk
            }
        )
        ref1 = cf.CoordinateReference(
            coordinate_conversion=coordinate_conversion, coordinates=[z])

        f.set_construct(ref1)

        # Field ancillary variables
        g = cf.FieldAncillary()
        g.set_data(f.data)
        g.transpose([1, 0], inplace=True)
        g.standard_name = 'ancillary0'
        g *= 0.01
        f.set_construct(g)

        g = cf.FieldAncillary()
        g.set_data(f.data)
        g.standard_name = 'ancillary1'
        g *= 0.01
        f.set_construct(g)

        g = cf.FieldAncillary()
        g.set_data(f[0, :].data)
        g.squeeze(inplace=True)
        g.standard_name = 'ancillary2'
        g *= 0.001
        f.set_construct(g)

        g = cf.FieldAncillary()
        g.set_data(f[:, 0].data)
        g.squeeze(inplace=True)
        g.standard_name = 'ancillary3'
        g *= 0.001
        f.set_construct(g)

        f.flag_values = [1, 2, 4]
        f.flag_meanings = ['a', 'bb', 'ccc']

        for cm in cf.CellMethod.create(
                'grid_longitude: mean grid_latitude: max'):
            f.set_construct(cm)


#        print(f.constructs.data_axes())
#        print(repr(f.constructs))
#        print(f.cell_measures)
#        print(f.constructs)
#
#        f.dump()

        # Write the file, and read it in
        cf.write(f, self.filename, verbose=0, string=True)

        g = cf.read(self.filename, squeeze=True, verbose=0)[0]

        self.assertTrue(g.equals(f, verbose=0),
                        "Field not equal to itself read back in")

        x = g.dump(display=False)
        x = f.dump(display=False)


# --- End: class

if __name__ == "__main__":
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
