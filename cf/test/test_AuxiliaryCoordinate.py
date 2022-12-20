import datetime
import faulthandler
import unittest

import numpy

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class AuxiliaryCoordinateTest(unittest.TestCase):
    f = cf.example_field(1)

    aux1 = cf.AuxiliaryCoordinate()
    aux1.standard_name = "latitude"
    a = numpy.array(
        [
            -30,
            -23.5,
            -17.8123,
            -11.3345,
            -0.7,
            -0.2,
            0,
            0.2,
            0.7,
            11.30003,
            17.8678678,
            23.5,
            30,
        ]
    )
    aux1.set_data(cf.Data(a, "degrees_north"))
    bounds = cf.Bounds()
    b = numpy.empty(a.shape + (2,))
    b[:, 0] = a - 0.1
    b[:, 1] = a + 0.1
    bounds.set_data(cf.Data(b))
    aux1.set_bounds(bounds)

    def test_AuxiliaryCoordinate_masked_invalid(self):
        a = self.aux1.copy()

        a.masked_invalid()
        self.assertIsNone(a.masked_invalid(inplace=True))

        a.del_bounds()
        a.masked_invalid()
        self.assertIsNone(a.masked_invalid(inplace=True))

    def test_AuxiliaryCoordinate__repr__str__dump(self):
        x = self.f.auxiliary_coordinate("latitude")
        repr(x)
        str(x)
        x.dump(display=False)

    def test_AuxiliaryCoordinate_bounds(self):
        d = self.f.dimension_coordinate("X")
        x = cf.AuxiliaryCoordinate(source=d)

        x.upper_bounds
        x.lower_bounds

    def test_AuxiliaryCoordinate_properties(self):
        x = self.f.auxiliary_coordinate("latitude")

        x.positive = "up"
        self.assertEqual(x.positive, "up")
        del x.positive
        self.assertIsNone(getattr(x, "positive", None))

        x.axis = "Z"
        self.assertEqual(x.axis, "Z")
        del x.axis
        self.assertIsNone(getattr(x, "axis", None))

        d = self.f.dimension_coordinate("X")
        x = cf.AuxiliaryCoordinate(source=d)

    def test_AuxiliaryCoordinate_insert_dimension(self):
        d = self.f.dimension_coordinate("X")
        x = cf.AuxiliaryCoordinate(source=d)

        self.assertEqual(x.shape, (9,))
        self.assertEqual(x.bounds.shape, (9, 2))

        y = x.insert_dimension(0)
        self.assertEqual(y.shape, (1, 9))
        self.assertEqual(y.bounds.shape, (1, 9, 2), y.bounds.shape)

        x.insert_dimension(-1, inplace=True)
        self.assertEqual(x.shape, (9, 1))
        self.assertEqual(x.bounds.shape, (9, 1, 2), x.bounds.shape)

    def test_AuxiliaryCoordinate_transpose(self):
        x = self.f.auxiliary_coordinate("longitude").copy()

        bounds = cf.Bounds(
            data=cf.Data(numpy.arange(9 * 10 * 4).reshape(9, 10, 4))
        )
        x.set_bounds(bounds)

        self.assertEqual(x.shape, (9, 10))
        self.assertEqual(x.bounds.shape, (9, 10, 4))

        y = x.transpose()
        self.assertEqual(y.shape, (10, 9))
        self.assertEqual(y.bounds.shape, (10, 9, 4), y.bounds.shape)

        x.transpose([1, 0], inplace=True)
        self.assertEqual(x.shape, (10, 9))
        self.assertEqual(x.bounds.shape, (10, 9, 4), x.bounds.shape)

    def test_AuxiliaryCoordinate_squeeze(self):
        x = self.f.auxiliary_coordinate("longitude").copy()

        bounds = cf.Bounds(
            data=cf.Data(numpy.arange(9 * 10 * 4).reshape(9, 10, 4))
        )
        x.set_bounds(bounds)
        x.insert_dimension(1, inplace=True)
        x.insert_dimension(0, inplace=True)

        self.assertEqual(x.shape, (1, 9, 1, 10))
        self.assertEqual(x.bounds.shape, (1, 9, 1, 10, 4))

        y = x.squeeze()
        self.assertEqual(y.shape, (9, 10))
        self.assertEqual(y.bounds.shape, (9, 10, 4), y.bounds.shape)

        x.squeeze(2, inplace=True)
        self.assertEqual(x.shape, (1, 9, 10))
        self.assertEqual(x.bounds.shape, (1, 9, 10, 4), x.bounds.shape)

    def test_AuxiliaryCoordinate_floor(self):
        aux = self.aux1.copy()

        a = aux.array
        b = aux.bounds.array

        self.assertTrue((aux.floor().array == numpy.floor(a)).all())
        self.assertTrue((aux.floor().bounds.array == numpy.floor(b)).all())
        self.assertTrue(
            (aux.floor(bounds=False).array == numpy.floor(a)).all()
        )
        self.assertTrue((aux.floor(bounds=False).bounds.array == b).all())

        aux.del_bounds()
        self.assertTrue((aux.floor().array == numpy.floor(a)).all())
        self.assertTrue(
            (aux.floor(bounds=False).array == numpy.floor(a)).all()
        )

        self.assertIsNone(aux.floor(inplace=True))
        self.assertTrue((aux.array == numpy.floor(a)).all())

    def test_AuxiliaryCoordinate_ceil(self):
        aux = self.aux1.copy()

        a = aux.array
        b = aux.bounds.array

        self.assertTrue((aux.ceil().array == numpy.ceil(a)).all())
        self.assertTrue((aux.ceil().bounds.array == numpy.ceil(b)).all())
        self.assertTrue((aux.ceil(bounds=False).array == numpy.ceil(a)).all())
        self.assertTrue((aux.ceil(bounds=False).bounds.array == b).all())

        aux.del_bounds()
        self.assertTrue((aux.ceil().array == numpy.ceil(a)).all())
        self.assertTrue((aux.ceil(bounds=False).array == numpy.ceil(a)).all())

        self.assertIsNone(aux.ceil(inplace=True))
        self.assertTrue((aux.array == numpy.ceil(a)).all())

    def test_AuxiliaryCoordinate_trunc(self):
        aux = self.aux1.copy()

        a = aux.array
        b = aux.bounds.array

        self.assertTrue((aux.trunc().array == numpy.trunc(a)).all())
        self.assertTrue((aux.trunc().bounds.array == numpy.trunc(b)).all())
        self.assertTrue(
            (aux.trunc(bounds=False).array == numpy.trunc(a)).all()
        )
        self.assertTrue((aux.trunc(bounds=False).bounds.array == b).all())

        aux.del_bounds()
        self.assertTrue((aux.trunc().array == numpy.trunc(a)).all())
        self.assertTrue(
            (aux.trunc(bounds=False).array == numpy.trunc(a)).all()
        )

        self.assertIsNone(aux.trunc(inplace=True))
        self.assertTrue((aux.array == numpy.trunc(a)).all())

    def test_AuxiliaryCoordinate_rint(self):
        aux = self.aux1.copy()

        a = aux.array
        b = aux.bounds.array

        x0 = aux.rint()
        x = x0.array

        self.assertTrue((x == numpy.rint(a)).all(), x)
        self.assertTrue((aux.rint().bounds.array == numpy.rint(b)).all())
        self.assertTrue((aux.rint(bounds=False).array == numpy.rint(a)).all())
        self.assertTrue((aux.rint(bounds=False).bounds.array == b).all())

        aux.del_bounds()
        self.assertTrue((aux.rint().array == numpy.rint(a)).all())
        self.assertTrue((aux.rint(bounds=False).array == numpy.rint(a)).all())

        self.assertIsNone(aux.rint(inplace=True))
        self.assertTrue((aux.array == numpy.rint(a)).all())

    def test_AuxiliaryCoordinate_sin_cos_tan(self):
        aux = self.aux1.copy()

        aux.cos()
        self.assertIsNone(aux.cos(inplace=True))

        aux.sin()
        self.assertIsNone(aux.sin(inplace=True))

        aux.tan()
        self.assertIsNone(aux.tan(inplace=True))

    def test_AuxiliaryCoordinate_log_exp(self):
        aux = self.aux1.copy()

        aux.exp()
        self.assertIsNone(aux.exp(inplace=True))

        aux.log()
        self.assertIsNone(aux.log(inplace=True))

    def test_AuxiliaryCoordinate_count(self):
        aux = self.aux1.copy()

        aux.count()

        aux.del_data()
        with self.assertRaises(Exception):
            aux.count()

    def test_AuxiliaryCoordinate_cyclic(self):
        aux = self.aux1.copy()

        self.assertEqual(aux.cyclic(), set())
        self.assertEqual(aux.cyclic(0), set())
        self.assertEqual(aux.cyclic(), set([0]))

    def test_AuxiliaryCoordinate_roll(self):
        aux = self.aux1.copy()

        aux.roll(0, 3)
        self.assertIsNone(aux.roll(-1, 4, inplace=True))

    def test_AuxiliaryCoordinate_round(self):
        aux = self.aux1.copy()

        a = aux.array
        b = aux.bounds.array

        for decimals in (0, 1, 2, 3, 4, 5):
            aux = self.aux1.copy()

            self.assertTrue(
                (aux.round(decimals).array == numpy.round(a, decimals)).all()
            )
            self.assertTrue(
                (
                    aux.round(decimals).bounds.array
                    == numpy.round(b, decimals)
                ).all()
            )
            self.assertTrue(
                (
                    aux.round(decimals, bounds=False).array
                    == numpy.round(a, decimals)
                ).all()
            )
            self.assertTrue(
                (aux.round(decimals, bounds=False).bounds.array == b).all()
            )

            aux.del_bounds()
            self.assertTrue(
                (aux.round(decimals).array == numpy.round(a, decimals)).all()
            )
            self.assertTrue(
                (
                    aux.round(decimals, bounds=False).array
                    == numpy.round(a, decimals)
                ).all()
            )

            self.assertIsNone(aux.round(decimals, inplace=True))
            self.assertTrue((aux.array == numpy.round(a, decimals)).all())

    def test_AuxiliaryCoordinate_clip(self):
        aux = self.aux1.copy()

        a = aux.array
        b = aux.bounds.array

        self.assertTrue(
            (aux.clip(-15, 25).array == numpy.clip(a, -15, 25)).all()
        )
        self.assertTrue(
            (aux.clip(-15, 25).bounds.array == numpy.clip(b, -15, 25)).all()
        )
        self.assertTrue(
            (
                aux.clip(-15, 25, bounds=False).array == numpy.clip(a, -15, 25)
            ).all()
        )
        self.assertTrue(
            (aux.clip(-15, 25, bounds=False).bounds.array == b).all()
        )

        aux.del_bounds()
        self.assertTrue(
            (aux.clip(-15, 25).array == numpy.clip(a, -15, 25)).all()
        )
        self.assertTrue(
            (
                aux.clip(-15, 25, bounds=False).array == numpy.clip(a, -15, 25)
            ).all()
        )

        self.assertIsNone(aux.clip(-15, 25, inplace=True))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
