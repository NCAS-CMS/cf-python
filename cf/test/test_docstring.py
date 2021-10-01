import datetime
import faulthandler
import inspect
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cfdm

import cf


def _recurse_on_subclasses(klass):
    """Return as a set all subclasses in a classes' subclass
    hierarchy."""
    return set(klass.__subclasses__()).union(
        [
            sub
            for cls in klass.__subclasses__()
            for sub in _recurse_on_subclasses(cls)
        ]
    )


def _get_all_abbrev_subclasses(klass):
    """Return set of all subclasses in class hierarchy, filtering some
    out.

    Filter out cf.mixin.properties*.Properties* (by means of there not
    being any abbreviated cf.Properties* classes) plus any cfdm classes,
    since this function needs to take cf subclasses from cfdm classes as
    well.

    """
    return tuple(
        [
            sub
            for sub in _recurse_on_subclasses(klass)
            if hasattr(cf, sub.__name__)
            and sub.__module__.split(".")[0] == "cf"  # i.e. not 'cfdm'
        ]
    )


class DocstringTest(unittest.TestCase):
    def setUp(self):
        self.package = "cf"
        self.repr = "CF "

        self.subclasses_of_Container = tuple(
            set(
                _get_all_abbrev_subclasses(cf.mixin_container.Container)
            ).union(
                set(
                    _get_all_abbrev_subclasses(cfdm.data.abstract.array.Array)
                ),
                [  # other key classes not in subclass hierarchy above
                    cf.coordinatereference.CoordinateReference,
                    cf.cellmethod.CellMethod,
                    cf.domainaxis.DomainAxis,
                ],
            )
        )

        self.subclasses_of_Properties = _get_all_abbrev_subclasses(
            cf.mixin.properties.Properties
        )

        self.subclasses_of_PropertiesData = _get_all_abbrev_subclasses(
            cf.mixin.propertiesdata.PropertiesData
        )

        self.subclasses_of_PropertiesDataBounds = _get_all_abbrev_subclasses(
            cf.mixin.propertiesdatabounds.PropertiesDataBounds
        )

    def test_docstring(self):
        # Test that all {{ occurrences have been substituted
        for klass in self.subclasses_of_Container:
            for x in (klass, klass()):
                for name in dir(x):
                    f = getattr(klass, name, None)

                    if f is None or not hasattr(f, "__doc__"):
                        continue

                    if name.startswith("__") and not inspect.isfunction(f):
                        continue

                    self.assertIsNotNone(
                        f.__doc__,
                        f"\nCLASS: {klass}"
                        f"\nMETHOD NAME: {name}"
                        f"\nMETHOD: {f}"
                        f"\n__doc__: {f.__doc__}",
                    )

                    self.assertNotIn(
                        "{{",
                        f.__doc__,
                        f"\nCLASS: {klass}"
                        f"\nMETHOD NAME: {name}"
                        f"\nMETHOD: {f}",
                    )

    def test_docstring_package(self):
        string = f">>> f = {self.package}."
        for klass in self.subclasses_of_Container:
            for x in (klass, klass()):
                f = x._has_component
                self.assertIn(
                    string,
                    f.__doc__,
                    "\nCLASS: {klass}\nMETHOD NAME: _has_component",
                )

        string = f">>> f = {self.package}."
        for klass in self.subclasses_of_Properties:
            for x in (klass, klass()):
                self.assertIn(string, x.clear_properties.__doc__, klass)

    def test_docstring_class(self):
        for klass in self.subclasses_of_Properties:
            string = f">>> f = {self.package}.{klass.__name__}"
            for x in (klass, klass()):
                self.assertIn(string, x.clear_properties.__doc__, klass)

        for klass in self.subclasses_of_Container:
            string = klass.__name__
            for x in (klass, klass()):
                self.assertIn(string, x.copy.__doc__, klass)

        for klass in self.subclasses_of_PropertiesDataBounds:
            string = f"{klass.__name__}"
            for x in (klass, klass()):
                self.assertIn(
                    string,
                    x.insert_dimension.__doc__,
                    f"\n\nCLASS: {klass}"
                    f"\nMETHOD NAME: {klass.__name__}"
                    f"\nMETHOD: insert_dimension",
                )

                self.assertIn(
                    string,
                    x.swapaxes.__doc__,
                    f"\n\nCLASS: {klass}"
                    f"\nMETHOD NAME: {klass.__name__}"
                    f"\nMETHOD: swapaxes",
                )

    def test_docstring_repr(self):
        string = f"<{self.repr}Data"
        for klass in self.subclasses_of_PropertiesData:
            for x in (klass, klass()):
                self.assertIn(
                    string,
                    x.has_data.__doc__,
                    f"\nCLASS: {klass}"
                    f"\nMETHOD NAME: has_data"
                    f"\nMETHOD: {x.has_data}",
                )

    def test_docstring_default(self):
        string = "Return the value of the *default* parameter"  # noqa: F841
        for klass in self.subclasses_of_Properties:
            for x in (klass, klass()):
                self.assertIn(string, x.del_property.__doc__, klass)

    def test_docstring_staticmethod(self):
        string = "Return the value of the *default* parameter"  # noqa: F841
        for klass in self.subclasses_of_PropertiesData:
            x = klass
            self.assertEqual(
                x._test_docstring_substitution_staticmethod(1, 2), (1, 2)
            )

    def test_docstring_classmethod(self):
        string = "Return the value of the *default* parameter"  # noqa: F841
        for klass in self.subclasses_of_PropertiesData:
            for x in (klass, klass()):
                self.assertEqual(
                    x._test_docstring_substitution_classmethod(1, 2), (1, 2)
                )

    def test_docstring_docstring_substitutions(self):
        for klass in self.subclasses_of_Container:
            for x in (klass,):
                d = x._docstring_substitutions(klass)
                self.assertIsInstance(d, dict)
                self.assertIn("{{repr}}", d, f"\nCLASS: {klass}")


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
