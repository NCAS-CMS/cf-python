import datetime
import os
import tempfile
import time
import unittest

import numpy

import cf

# def _make_gathered_file(filename):
#     '''
#     '''
#     def _jj(shape, list_values):
#         array = numpy.ma.masked_all(shape)
#         for i, (index, x) in enumerate(numpy.ndenumerate(array)):
#             if i in list_values:
#                 array[index] = i
#         return array
#     # --- End: def
#
#     n = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
#
#     n.Conventions = 'CF-1.6'
#
#     time    = n.createDimension('time'   ,  2)
#     height  = n.createDimension('height' ,  3)
#     lat     = n.createDimension('lat'    ,  4)
#     lon     = n.createDimension('lon'    ,  5)
#     p       = n.createDimension('p'      ,  6)
#
#     list1  = n.createDimension('list1',  4)
#     list2  = n.createDimension('list2',  9)
#     list3  = n.createDimension('list3', 14)
#
#     # Dimension coordinate variables
#     time = n.createVariable('time', 'f8', ('time',))
#     time.standard_name = "time"
#     time.units = "days since 2000-1-1"
#     time[...] = [31, 60]
#
#     height = n.createVariable('height', 'f8', ('height',))
#     height.standard_name = "height"
#     height.units = "metres"
#     height.positive = "up"
#     height[...] = [0.5, 1.5, 2.5]
#
#     lat = n.createVariable('lat', 'f8', ('lat',))
#     lat.standard_name = "latitude"
#     lat.units = "degrees_north"
#     lat[...] = [-90, -85, -80, -75]
#
#     p = n.createVariable('p', 'i4', ('p',))
#     p.long_name = "pseudolevel"
#     p[...] = [1, 2, 3, 4, 5, 6]
#
#     # Auxiliary coordinate variables
#
#     aux0 = n.createVariable('aux0', 'f8', ('list1',))
#     aux0.standard_name = "longitude"
#     aux0.units = "degrees_east"
#     aux0[...] = numpy.arange(list1.size)
#
#     aux1 = n.createVariable('aux1', 'f8', ('list3',))
#     aux1[...] = numpy.arange(list3.size)
#
#     aux2 = n.createVariable('aux2', 'f8', ('time', 'list3', 'p'))
#     aux2[...] = numpy.arange(time.size * list3.size * p.size).reshape(
#         time.size, list3.size, p.size)
#
#     aux3 = n.createVariable('aux3', 'f8', ('p', 'list3', 'time'))
#     aux3[...] = numpy.arange(p.size * list3.size * time.size).reshape(
#         p.size, list3.size, time.size)
#
#     aux4 = n.createVariable('aux4', 'f8', ('p', 'time', 'list3'))
#     aux4[...] = numpy.arange(p.size * time.size * list3.size).reshape(
#         p.size, time.size, list3.size)
#
#     aux5 = n.createVariable('aux5', 'f8', ('list3', 'p', 'time'))
#     aux5[...] = numpy.arange(list3.size * p.size * time.size).reshape(
#         list3.size, p.size, time.size)
#
#     aux6 = n.createVariable('aux6', 'f8', ('list3', 'time'))
#     aux6[...] = numpy.arange(list3.size * time.size).reshape(
#         list3.size, time.size)
#
#     aux7 = n.createVariable('aux7', 'f8', ('lat',))
#     aux7[...] = numpy.arange(lat.size)
#
#     aux8 = n.createVariable('aux8', 'f8', ('lon', 'lat',))
#     aux8[...] = numpy.arange(lon.size * lat.size).reshape(lon.size, lat.size)
#
#     aux9 = n.createVariable('aux9', 'f8', ('time', 'height'))
#     aux9[...] = numpy.arange(time.size * height.size).reshape(
#         time.size, height.size)
#
#     # List variables
#     list1 = n.createVariable('list1', 'i', ('list1',))
#     list1.compress = "lon"
#     list1[...] = [0, 1, 3, 4]
#
#     list2 = n.createVariable('list2', 'i', ('list2',))
#     list2.compress = "lat lon"
#     list2[...] = [0,  1,  5,  6, 13, 14, 17, 18, 19]
#
#     list3 = n.createVariable('list3', 'i', ('list3',))
#     list3.compress = "height lat lon"
#     array = _jj((3, 4, 5),
#                 [0, 1, 5, 6, 13, 14, 25, 26, 37, 38, 48, 49, 58, 59])
#     list3[...] = array.compressed()
#
#     # Data variables
#     temp1 = n.createVariable(
#         'temp1', 'f8', ('time', 'height', 'lat', 'list1', 'p'))
#     temp1.long_name = "temp1"
#     temp1.units = "K"
#     temp1.coordinates = "aux0 aux7 aux8 aux9"
#     temp1[...] = numpy.arange(2*3*4*4*6).reshape(2, 3, 4, 4, 6)
#
#     temp2 = n.createVariable('temp2', 'f8', ('time', 'height', 'list2', 'p'))
#     temp2.long_name = "temp2"
#     temp2.units = "K"
#     temp2.coordinates = "aux7 aux8 aux9"
#     temp2[...] = numpy.arange(2*3*9*6).reshape(2, 3, 9, 6)
#
#     temp3 = n.createVariable('temp3', 'f8', ('time', 'list3', 'p'))
#     temp3.long_name = "temp3"
#     temp3.units = "K"
#     temp3.coordinates = "aux0 aux1 aux2 aux3 aux4 aux5 aux6 aux7 aux8 aux9"
#     temp3[...] = numpy.arange(2*14*6).reshape(2, 14, 6)
#
#     n.close()
#
#     return filename
#
#
# gathered = _make_gathered_file('gathered.nc')


class DSGTest(unittest.TestCase):
    def setUp(self):
        self.gathered = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'gathered.nc')

        (fd, self.tempfilename) = tempfile.mkstemp(
            suffix='.nc', prefix='cf_', dir='.')
        os.close(fd)

        a = numpy.ma.masked_all((4, 9), dtype=float)
        a[0, 0:3] = [0.0, 1.0, 2.0]
        a[1, 0:7] = [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0]
        a[2, 0:5] = [2.0, 102.0, 202.0, 302.0, 402.0]
        a[3, 0:9] = [3.0, 1003.0, 2003.0, 3003.0, 4003.0, 5003.0, 6003.0,
                     7003.0, 8003.0]
        self.a = a

        b = numpy.ma.array(
            [[[207.12345561172262, -99, -99, -99],
              [100.65758285427566, 117.72137430364056, 182.1893456150461, -99],
              [109.93898265295516, 117.76872282697526, -99, -99],
              [163.020681064712, 200.09702526477145, -99, -99],
              [138.25879722836117, 182.59075988956565, -99, -99],
              [159.28122555425304, -99, -99, -99],
              [157.0114286059841, 212.14056704399377, -99, -99],
              [225.09002846189756, -99, -99, -99],
              [179.99301151546493, -99, -99, -99],
              [125.56310968736936, 216.60367471282225, -99, -99],
              [105.12035147782414, 129.460917520233, 210.13998569368403, -99],
              [159.75007622045126, 197.101264162631, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99]],
             [[52.1185292100177, 57.51542658633939, 108.49584371709457,
               137.7109686243953],
              [26.433960062549616, 91.57049700941819, -99, -99],
              [7.015322103368953, 39.551765142093345, -99, -99],
              [157.047493027102, -99, -99, -99],
              [25.18033994582771, 159.67348686580374, -99, -99],
              [45.84635421577662, 97.86781970832622, -99, -99],
              [5.61560792556281, 31.182013232254985, -99, -99],
              [37.78941964121314, -99, -99, -99],
              [57.2927165845568, 129.40831355790502, 181.2962705331917, -99],
              [38.714266913107686, 69.34591875157382, 169.26193063629765, -99],
              [72.52507309225012, 138.22169348672838, 159.82855521564647, -99],
              [45.23406469185547, 97.66633738254326, 112.64049631761776, -99],
              [14.920937817653984, -99, -99, -99],
              [9.071979535527532, 42.527916794472986, 61.8685137936187, -99],
              [17.175098751913993, 99.00403750149574, -99, -99],
              [92.95097491537247, -99, -99, -99],
              [7.11997786817564, -99, -99, -99],
              [156.81807261767003, -99, -99, -99],
              [6.832599021190903, 12.446963835216742, -99, -99],
              [45.19734905410353, 124.30321995608465, 130.4780046562618, -99],
              [35.18924597876244, 68.36858129904569, 78.88837365755683, -99],
              [81.15820119504805, 122.41242448019014, -99, -99],
              [58.95866448059819, -99, -99, -99],
              [10.465638726626635, 96.11859001483036, -99, -99],
              [55.64766876004607, 78.37174486781481, 91.09175506350066, -99],
              [71.46930436420837, 90.43816256387788, 103.76781788802138, -99]],
             [[351.97770529376936, -99, -99, -99],
              [347.0644742747811, 388.5698490238134, 481.0692542795372, -99],
              [352.42430719766776, 393.20047319955916, 395.71509960367075,
               -99],
              [402.8689447636048, 403.74922883226424, 479.8582815909853, -99],
              [300.0199333154121, 365.124061660449, -99, -99],
              [333.35006535728564, 433.143904011861, -99, -99],
              [376.9480484244583, -99, -99, -99],
              [334.99329771076077, -99, -99, -99],
              [319.36684737542186, 337.20913311790446, -99, -99],
              [340.66500823697623, 353.52589668400094, 410.44418671572373,
               -99],
              [301.9005914473572, 337.2055422899861, 386.9573429761627, -99],
              [324.3747437305056, 424.04244158178483, -99, -99],
              [331.52095586074626, 349.4826244342738, 396.81256849354895, -99],
              [331.99043697116906, -99, -99, -99],
              [384.76674803938937, -99, -99, -99],
              [373.0334288724908, 399.47980750739197, -99, -99],
              [300.0106221314076, 390.6371376624527, -99, -99],
              [364.25269358741537, 391.19723635099535, 456.466622863717, -99],
              [410.1246758522543, -99, -99, -99],
              [310.59214185542953, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99],
              [-99, -99, -99, -99]]]
        )

        b = numpy.ma.where(b == -99, numpy.ma.masked, b)
        self.b = b

        self.test_only = []

    def tearDown(self):
        os.remove(self.tempfilename)

    def test_GATHERING(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.gathered, verbose=0)

        self.assertTrue(len(f) == 3)

        g = f.select('long_name=temp3')[0]
#        for g in f:
#            if g.get_property('long_name') == 'temp3':
#                break

#        print (g)

#        print(g.data.array)
#        print(repr(g.data.get_list().data.array)
#        print(g.data.get_list().data.array)
#        print('compression_type=',g.data.get_compression_type())
#        print(g.data.get_compressed_axes())
#        print(g.data.shape)

        cf.write(f, self.tempfilename, verbose=0)
        g = cf.read(self.tempfilename, verbose=0)
#        print (repr(f))
#        print (repr(g))
        self.assertTrue(len(g) == len(f), str(len(g)) + ' ' + str(len(f)))

#        print ('\nf\n')
#        for x in f:
#            print(x)
#            a = x.data.array
#
#        print ('\ng\n')
#        for x in g:
#            print(x)
#            a = x.data.array
#
#        for x in g:
#            x.dump()

        for a, b in zip(f, g):
            self.assertTrue(b.equals(a, verbose=2))

    def test_GATHERING_create(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Define the gathered values
        gathered_array = numpy.array([[280, 282.5, 281], [279, 278, 277.5]],
                                     dtype='float32')
        # Define the list array values
        list_array = [1, 4, 5]

        # Initialise the list variable
        list_variable = cf.List(data=cf.Data(list_array))

        # Initialise the gathered array object
        array = cf.GatheredArray(
            compressed_array=cf.Data(gathered_array),
            compressed_dimension=1,
            shape=(2, 3, 2), size=12, ndim=3,
            list_variable=list_variable
        )

        # Create the field construct with the domain axes and the gathered
        # array
        tas = cf.Field(properties={'standard_name': 'air_temperature',
                                   'units': 'K'})

        # Create the domain axis constructs for the uncompressed array
        T = tas.set_construct(cf.DomainAxis(2))
        Y = tas.set_construct(cf.DomainAxis(3))
        X = tas.set_construct(cf.DomainAxis(2))

        uncompressed_array = numpy.ma.masked_array(
            data=[[[1, 280.0],
                   [1, 1],
                   [282.5, 281.0]],

                  [[1, 279.0],
                   [1, 1],
                   [278.0, 277.5]]],
            mask=[[[True, False],
                   [True, True],
                   [False, False]],

                  [[True, False],
                   [True, True],
                   [False, False]]],
            fill_value=1e+20,
            dtype='float32'
        )

        for chunksize in (1000000,):
            cf.CHUNKSIZE(chunksize)
            message = 'chunksize='+str(chunksize)

            # Set the data for the field
            tas.set_data(cf.Data(array), axes=[T, Y, X])
#            print (tas.data.dumpd())

            self.assertTrue(
                (tas.data.array == uncompressed_array).all(), message)

            self.assertTrue(
                tas.data.get_compression_type() == 'gathered', message)

            self.assertTrue((tas.data.compressed_array == numpy.array(
                [[280., 282.5, 281.],
                 [279., 278., 277.5]], dtype='float32')).all(), message)

            self.assertTrue((tas.data.get_list().data.array == numpy.array(
                [1, 4, 5])).all(), message)


# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    print(cf.environment(display=False))
    print()
    unittest.main(verbosity=2)
