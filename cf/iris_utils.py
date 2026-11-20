"""Utilities for conversion to and from iris."""

import cfdm


def from_iris(c):
    """Convert an `iris` dataset to CF field or domain constructs."""
    try:
        import ncdata  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install ncdata (https://pypi.org/project/ncdata) "
            "to allow conversions to and from iris cubes"
        )

    try:
        from ncdata.iris import from_iris
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install iris "
            "to allow conversions to and from iris cubes"
        )

    try:
        from ncdata.xarray import to_xarray
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install xarray "
            "to allow conversions to and from iris cubes"
        )

    from cf import from_xarray

    n = from_iris(c)
    ds = to_xarray(n)
    return from_xarray(ds)


# Reset the docstring from cfdm
from_iris.__doc__ = cfdm.from_iris.__doc__.replace("cfdm.", "cf.").replace(
    "<Field", "<CF Field"
)


def to_iris(f):
    """Convert CF field or domain constructs to an `iris` dataset."""
    return cfdm.to_iris(f)


# Reset the docstring from cfdm
to_iris.__doc__ = cfdm.to_iris.__doc__.replace("cfdm.", "cf.").replace(
    "<Field", "<CF Field"
)
