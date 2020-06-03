import cfdm

from .cfimplementation import implementation


_implementation = implementation()


def example_field(n, _implementation=_implementation):
    return cfdm.example_field(n, _implementation=_implementation)


example_field.__doc__ = cfdm.example_field.__doc__
