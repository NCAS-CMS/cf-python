from ctypes.util import find_library

from cfunits import Units as cfUnits

_libpath = find_library('udunits2')
if _libpath is None:
    raise FileNotFoundError(
        "cf requires UNIDATA UDUNITS-2. Can't find the 'udunits2' library.")


class Units:
    '''Store, combine and compare physical units and convert numeric
    values to different units.

    This is a convenience class that creates a `cfunits.Units`
    instance.

    The full documentation is available with ``help(cf.Units())``.

    '''
    def __new__(cls, *args, **kwargs):
        return cfUnits(*args, **kwargs)

    @staticmethod
    def conform(*args, **kwargs):
        return cfUnits.conform(*args, **kwargs)

# --- End: class
