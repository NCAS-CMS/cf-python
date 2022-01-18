import cfdm

from .cfimplementation import implementation

_implementation = implementation()


def example_field(n, _implementation=_implementation):
    return cfdm.example_field(n, _implementation=_implementation)


example_field.__doc__ = cfdm.example_field.__doc__.replace("cfdm.", "cf.")


def example_fields(*n, _func=example_field):
    return cfdm.example_fields(*n, _func=_func)


example_fields.__doc__ = cfdm.example_fields.__doc__.replace("cfdm.", "cf.")
example_fields.__doc__ = example_fields.__doc__.replace(
    "<Field:", "<CF Field:"
)
example_fields.__doc__ = example_fields.__doc__.replace(
    "`list`", "`FieldList`"
)


def example_domain(n, _func=example_field):
    return cfdm.example_domain(n, _func=_func)


example_domain.__doc__ = cfdm.example_domain.__doc__.replace("cfdm.", "cf.")
example_domain.__doc__ = example_domain.__doc__.replace(
    "<Field:", "<CF Field:"
)
