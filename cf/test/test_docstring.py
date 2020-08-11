import datetime
import unittest

import cf


class DocstringTest(unittest.TestCase):
    def setUp(self):
        self.package = 'cf'
        self.repr = 'CF '
        self.subclasses_of_Container = (
            cf.Field,
            cf.AuxiliaryCoordinate,
            cf.DimensionCoordinate,
            cf.DomainAncillary,
            cf.FieldAncillary,
            cf.CellMeasure,
            cf.DomainAxis,
            cf.CoordinateReference,
            cf.CellMethod,

            cf.NodeCountProperties,
            cf.PartNodeCountProperties,
            cf.Bounds,
            cf.InteriorRing,
            cf.List,
            cf.Index,
            cf.Count,

            cf.Data,
            cf.NetCDFArray,
            cf.GatheredArray,
            cf.RaggedContiguousArray,
            cf.RaggedIndexedArray,
            cf.RaggedIndexedContiguousArray,
        )
        self.subclasses_of_Properties = (
            cf.Field,
            cf.AuxiliaryCoordinate,
            cf.DimensionCoordinate,
            cf.DomainAncillary,
            cf.FieldAncillary,
            cf.CellMeasure,
            cf.NodeCountProperties,
            cf.PartNodeCountProperties,
            cf.Bounds,
            cf.InteriorRing,
            cf.List,
            cf.Index,
            cf.Count,
        )
        self.subclasses_of_PropertiesData = (
            cf.Field,
            cf.AuxiliaryCoordinate,
            cf.DimensionCoordinate,
            cf.DomainAncillary,
            cf.FieldAncillary,
            cf.CellMeasure,
            cf.Bounds,
            cf.InteriorRing,
            cf.List,
            cf.Index,
            cf.Count,
        )
        self.subclasses_of_PropertiesDataBounds = (
            cf.AuxiliaryCoordinate,
            cf.DimensionCoordinate,
            cf.DomainAncillary,
        )

    def test_docstring(self):
        # Test that all {{ occurences have been substituted
        for klass in self.subclasses_of_Container:
            for x in (klass, klass()):
                for name in dir(x):
                    if name.startswith('__'):
                        continue

                    f = getattr(klass, name, None)
                    if f is None or not hasattr(f, '__doc__'):
                        continue

                    self.assertIsNotNone(
                        f.__doc__,
                        '\nCLASS: {}\nMETHOD NAME: {}\nMETHOD: {}\n__doc__: {}'.format(
                            klass, name, f, f.__doc__))

                    self.assertNotIn(
                        '{{', f.__doc__,
                        '\nCLASS: {}\nMETHOD NAME: {}\nMETHOD: {}'.format(
                            klass, name, f))

    def test_docstring_package(self):
        string = '>>> f = {}.'.format(self.package)
        for klass in self.subclasses_of_Container:
            for x in (klass, klass()):
                f = x._has_component
                self.assertIn(
                    string, f.__doc__,
                    '\nCLASS: {}\nMETHOD NAME: {}'.format(
                        klass, '_has_component'))

        string = '>>> f = {}.'.format(self.package)
        for klass in self.subclasses_of_Properties:
            for x in (klass, klass()):
                self.assertIn(string, x.clear_properties.__doc__, klass)

    def test_docstring_class(self):
        for klass in self.subclasses_of_Properties:
            string = '>>> f = {}.{}'.format(self.package, klass.__name__)
            for x in (klass, klass()):
                self.assertIn(string, x.clear_properties.__doc__, klass)

        for klass in self.subclasses_of_Container:
            string = klass.__name__
            for x in (klass, klass()):
                self.assertIn(string, x.copy.__doc__, klass)

        for klass in self.subclasses_of_PropertiesDataBounds:
            string = '{}'.format(klass.__name__)
            for x in (klass, klass()):
                self.assertIn(
                    string, x.insert_dimension.__doc__,
                    '\n\nCLASS: {}\nMETHOD NAME: {}\nMETHOD: {}'.format(
                        klass, klass.__name__, 'insert_dimension'))

                self.assertIn(
                    string, x.swapaxes.__doc__,
                    '\n\nCLASS: {}\nMETHOD NAME: {}\nMETHOD: {}'.format(
                        klass, klass.__name__, 'swapaxes'))

    def test_docstring_plus_class(self):
        string = '>>> d = {}.{}'.format(self.package, 'Data')
        for klass in self.subclasses_of_PropertiesData:
            for x in (klass, klass()):
                self.assertIn(string, x.has_data.__doc__, klass)

    def test_docstring_repr(self):
        string = '<{}Data'.format(self.repr)
        for klass in self.subclasses_of_PropertiesData:
            for x in (klass, klass()):
                self.assertIn(
                    string, x.has_data.__doc__,
                    '\nCLASS: {}\nMETHOD NAME: {}\nMETHOD: {}'.format(
                        klass, 'has_data', x.has_data))

    def test_docstring_default(self):
        string = 'Return the value of the *default* parameter'
        for klass in self.subclasses_of_Properties:
            for x in (klass, klass()):
                self.assertIn(string, x.del_property.__doc__, klass)

    def test_docstring_staticmethod(self):
        string = 'Return the value of the *default* parameter'
        for klass in self.subclasses_of_PropertiesData:
            x = klass
            self.assertEqual(
                x._test_docstring_substitution_staticmethod(1, 2),
                (1, 2)
            )

    def test_docstring_classmethod(self):
        string = 'Return the value of the *default* parameter'
        for klass in self.subclasses_of_PropertiesData:
            for x in (klass, klass()):
                self.assertEqual(
                    x._test_docstring_substitution_classmethod(1, 2),
                    (1, 2)
                )

    def test_docstring_docstring_substitutions(self):
        for klass in self.subclasses_of_Container:
            for x in (klass, klass()):
                d = x._docstring_substitution()
                self.assertIsInstance(d, dict)
                self.assertIn(
                    '{{repr}}', d,
                    '\nCLASS: {}'.format(
                        klass))

# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print('')
    unittest.main(verbosity=2)
