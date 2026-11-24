"""Utilities for conversion to and from xarray."""

import cfdm


def from_xarray(ds, domain=False):
    """Convert an `xarray` dataset to CF field or domain constructs."""
    try:
        import xarray  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install xarray "
            "to allow the conversion to and from an xarray dataset"
        )

    from . import read

    return read(ds, domain=domain, aggregate=False)


# Reset the docstring from cfdm
from_xarray.__doc__ = cfdm.from_xarray.__doc__.replace("cfdm.", "cf.").replace(
    "<Field", "<CF Field"
)


def to_xarray(f, group=True):
    """Convert CF field or domain constructs to an `xarray` dataset."""
    return cfdm.to_xarray(f, group=group)


# Reset the docstring from cfdm
to_xarray.__doc__ = cfdm.to_xarray.__doc__.replace("cfdm.", "cf.").replace(
    "<Field", "<CF Field"
)
