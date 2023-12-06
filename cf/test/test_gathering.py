import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

import numpy

faulthandler.enable()  # to debug seg faults and timeouts

import cf

n_tmpfiles = 1
tmpfiles = [
    tempfile.mkstemp("_test_gathering.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
[tmpfile] = tmpfiles


def _remove_tmpfiles():
    """Try to remove defined temporary files by deleting their paths."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class GatheringTest(unittest.TestCase):
    gathered = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gathered.nc"
    )

    f = cf.read(gathered)

    a = numpy.ma.masked_all((4, 9), dtype=float)
    a[0, 0:3] = [0.0, 1.0, 2.0]
    a[1, 0:7] = [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0]
    a[2, 0:5] = [2.0, 102.0, 202.0, 302.0, 402.0]
    a[3, 0:9] = [
        3.0,
        1003.0,
        2003.0,
        3003.0,
        4003.0,
        5003.0,
        6003.0,
        7003.0,
        8003.0,
    ]

    b = numpy.ma.array(
        [
            [
                [207.12345561172262, -99, -99, -99],
                [
                    100.65758285427566,
                    117.72137430364056,
                    182.1893456150461,
                    -99,
                ],
                [109.93898265295516, 117.76872282697526, -99, -99],
                [163.020681064712, 200.09702526477145, -99, -99],
                [138.25879722836117, 182.59075988956565, -99, -99],
                [159.28122555425304, -99, -99, -99],
                [157.0114286059841, 212.14056704399377, -99, -99],
                [225.09002846189756, -99, -99, -99],
                [179.99301151546493, -99, -99, -99],
                [125.56310968736936, 216.60367471282225, -99, -99],
                [
                    105.12035147782414,
                    129.460917520233,
                    210.13998569368403,
                    -99,
                ],
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
                [-99, -99, -99, -99],
            ],
            [
                [
                    52.1185292100177,
                    57.51542658633939,
                    108.49584371709457,
                    137.7109686243953,
                ],
                [26.433960062549616, 91.57049700941819, -99, -99],
                [7.015322103368953, 39.551765142093345, -99, -99],
                [157.047493027102, -99, -99, -99],
                [25.18033994582771, 159.67348686580374, -99, -99],
                [45.84635421577662, 97.86781970832622, -99, -99],
                [5.61560792556281, 31.182013232254985, -99, -99],
                [37.78941964121314, -99, -99, -99],
                [57.2927165845568, 129.40831355790502, 181.2962705331917, -99],
                [
                    38.714266913107686,
                    69.34591875157382,
                    169.26193063629765,
                    -99,
                ],
                [
                    72.52507309225012,
                    138.22169348672838,
                    159.82855521564647,
                    -99,
                ],
                [
                    45.23406469185547,
                    97.66633738254326,
                    112.64049631761776,
                    -99,
                ],
                [14.920937817653984, -99, -99, -99],
                [9.071979535527532, 42.527916794472986, 61.8685137936187, -99],
                [17.175098751913993, 99.00403750149574, -99, -99],
                [92.95097491537247, -99, -99, -99],
                [7.11997786817564, -99, -99, -99],
                [156.81807261767003, -99, -99, -99],
                [6.832599021190903, 12.446963835216742, -99, -99],
                [
                    45.19734905410353,
                    124.30321995608465,
                    130.4780046562618,
                    -99,
                ],
                [35.18924597876244, 68.36858129904569, 78.88837365755683, -99],
                [81.15820119504805, 122.41242448019014, -99, -99],
                [58.95866448059819, -99, -99, -99],
                [10.465638726626635, 96.11859001483036, -99, -99],
                [55.64766876004607, 78.37174486781481, 91.09175506350066, -99],
                [
                    71.46930436420837,
                    90.43816256387788,
                    103.76781788802138,
                    -99,
                ],
            ],
            [
                [351.97770529376936, -99, -99, -99],
                [347.0644742747811, 388.5698490238134, 481.0692542795372, -99],
                [
                    352.42430719766776,
                    393.20047319955916,
                    395.71509960367075,
                    -99,
                ],
                [
                    402.8689447636048,
                    403.74922883226424,
                    479.8582815909853,
                    -99,
                ],
                [300.0199333154121, 365.124061660449, -99, -99],
                [333.35006535728564, 433.143904011861, -99, -99],
                [376.9480484244583, -99, -99, -99],
                [334.99329771076077, -99, -99, -99],
                [319.36684737542186, 337.20913311790446, -99, -99],
                [
                    340.66500823697623,
                    353.52589668400094,
                    410.44418671572373,
                    -99,
                ],
                [301.9005914473572, 337.2055422899861, 386.9573429761627, -99],
                [324.3747437305056, 424.04244158178483, -99, -99],
                [
                    331.52095586074626,
                    349.4826244342738,
                    396.81256849354895,
                    -99,
                ],
                [331.99043697116906, -99, -99, -99],
                [384.76674803938937, -99, -99, -99],
                [373.0334288724908, 399.47980750739197, -99, -99],
                [300.0106221314076, 390.6371376624527, -99, -99],
                [
                    364.25269358741537,
                    391.19723635099535,
                    456.466622863717,
                    -99,
                ],
                [410.1246758522543, -99, -99, -99],
                [310.59214185542953, -99, -99, -99],
                [-99, -99, -99, -99],
                [-99, -99, -99, -99],
                [-99, -99, -99, -99],
                [-99, -99, -99, -99],
                [-99, -99, -99, -99],
                [-99, -99, -99, -99],
            ],
        ]
    )

    b = numpy.ma.where(b == -99, numpy.ma.masked, b)

    def test_GATHERING(self):
        f = self.f.copy()

        self.assertEqual(len(f), 3)

        g = f.select("long_name=temp3")[0]

        cf.write(f, tmpfile, verbose=0)
        g = cf.read(tmpfile, verbose=0)

        self.assertEqual(len(g), len(f), str(len(g)) + " " + str(len(f)))

        for a, b in zip(f, g):
            self.assertTrue(b.equals(a, verbose=2))

    def test_GATHERING_create(self):
        # Define the gathered values
        gathered_array = numpy.array(
            [[280, 282.5, 281], [279, 278, 277.5]], dtype="float32"
        )
        # Define the list array values
        list_array = [1, 4, 5]

        # Initialise the list variable
        list_variable = cf.List(data=cf.Data(list_array))

        # Initialise the gathered array object
        array = cf.GatheredArray(
            compressed_array=cf.Data(gathered_array),
            compressed_dimensions={1: (1, 2)},
            shape=(2, 3, 2),
            list_variable=list_variable,
        )

        # Create the field construct with the domain axes and the
        # gathered array
        tas = cf.Field(
            properties={"standard_name": "air_temperature", "units": "K"}
        )

        # Create the domain axis constructs for the uncompressed array
        T = tas.set_construct(cf.DomainAxis(2))
        Y = tas.set_construct(cf.DomainAxis(3))
        X = tas.set_construct(cf.DomainAxis(2))

        # Set the data for the field
        tas.set_data(cf.Data(array), axes=[T, Y, X])

        self.assertTrue(
            (
                tas.data.array
                == numpy.ma.masked_array(
                    data=[
                        [[1, 280.0], [1, 1], [282.5, 281.0]],
                        [[1, 279.0], [1, 1], [278.0, 277.5]],
                    ],
                    mask=[
                        [[True, False], [True, True], [False, False]],
                        [[True, False], [True, True], [False, False]],
                    ],
                    fill_value=1e20,
                    dtype="float32",
                )
            ).all()
        )

        self.assertEqual(tas.data.get_compression_type(), "gathered")

        self.assertTrue(
            (
                tas.data.compressed_array
                == numpy.array(
                    [[280.0, 282.5, 281.0], [279.0, 278.0, 277.5]],
                    dtype="float32",
                )
            ).all()
        )

        self.assertTrue(
            (tas.data.get_list().data.array == numpy.array([1, 4, 5])).all(),
        )


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
