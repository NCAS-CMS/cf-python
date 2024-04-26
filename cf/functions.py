import atexit
import csv
import ctypes.util
import importlib
import os
import platform
import re
import sys
import urllib.parse
import warnings
from collections.abc import Iterable
from itertools import product
from math import isnan
from numbers import Integral
from os import mkdir
from os.path import abspath as _os_path_abspath
from os.path import dirname as _os_path_dirname
from os.path import expanduser as _os_path_expanduser
from os.path import expandvars as _os_path_expandvars
from os.path import join as _os_path_join
from os.path import relpath as _os_path_relpath
from urllib.parse import urlparse

import cfdm
import netCDF4
import numpy as np
from dask import config as _config
from dask.base import is_dask_collection
from dask.utils import parse_bytes
from psutil import virtual_memory

from . import __cfa_version__, __file__, __version__
from .constants import (
    CONSTANTS,
    OperandBoundsCombination,
    _stash2standard_name,
)
from .docstring import _docstring_substitution_definitions


# Instruction to close /proc/mem at exit.
def _close_proc_meminfo():
    try:
        _meminfo_file.close()
    except Exception:
        pass


atexit.register(_close_proc_meminfo)


# --------------------------------------------------------------------
# Inherit classes from cfdm
# --------------------------------------------------------------------
class Constant(cfdm.Constant):
    def __docstring_substitutions__(self):
        return _docstring_substitution_definitions

    def __docstring_package_depth__(self):
        return 0

    def __repr__(self):
        """Called by the `repr` built-in function."""
        return super().__repr__().replace("<", "<CF ", 1)


class DeprecationError(Exception):
    pass


KWARGS_MESSAGE_MAP = {
    "relaxed_identity": "Use keywords 'strict' or 'relaxed' instead.",
    "axes": "Use keyword 'axis' instead.",
    "traceback": "Use keyword 'verbose' instead.",
    "exact": "Use 're.compile' objects instead.",
    "i": (
        "Use keyword 'inplace' instead. Note that when inplace=True, "
        "None is returned."
    ),
    "info": (
        "Use keyword 'verbose' instead. Note the informational levels "
        "have been remapped: V = I + 1 maps info=I to verbose=V inputs, "
        "excluding I >= 3 which maps to V = -1 (and V = 0 disables messages)"
    ),
}


if platform.system() == "Linux":
    # ----------------------------------------------------------------
    # GNU/LINUX
    # ----------------------------------------------------------------
    # Opening /proc/meminfo once per PE here rather than in
    # _free_memory each time it is called works with MPI on
    # Debian-based systems, which otherwise throw an error that there
    # is no such file or directory when run on multiple PEs.
    # ----------------------------------------------------------------
    _meminfo_fields = set(("SReclaimable:", "Cached:", "Buffers:", "MemFree:"))
    _meminfo_file = open("/proc/meminfo", "r", 1)

    def _free_memory():
        """The amount of available physical memory on GNU/Linux.

        This amount includes any memory which is still allocated but is no
        longer required.

        :Returns:

            `float`
                The amount of available physical memory in bytes.

        **Examples**

        >>> _free_memory()
        96496240.0

        """
        # https://github.com/giampaolo/psutil/blob/master/psutil/_pslinux.py

        # ----------------------------------------------------------------
        # The available physical memory is the sum of the values of
        # the 'SReclaimable', 'Cached', 'Buffers' and 'MemFree'
        # entries in the /proc/meminfo file
        # (http://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/tree/Documentation/filesystems/proc.txt).
        # ----------------------------------------------------------------
        free_KiB = 0.0
        n = 0

        # with open('/proc/meminfo', 'r', 1) as _meminfo_file:

        # Seeking the beginning of the file /proc/meminfo regenerates
        # the information contained in it.
        _meminfo_file.seek(0)
        for line in _meminfo_file:
            field_size = line.split()
            if field_size[0] in _meminfo_fields:
                free_KiB += float(field_size[1])
                n += 1
                if n > 3:
                    break

        free_bytes = free_KiB * 1024

        return free_bytes

else:
    # ----------------------------------------------------------------
    # NOT GNU/LINUX
    # ----------------------------------------------------------------
    def _free_memory():
        """The amount of available physical memory.

        :Returns:

            `float`
                The amount of available physical memory in bytes.

        **Examples**

        >>> _free_memory()
        96496240.0

        """
        return float(virtual_memory().available)


def configuration(
    atol=None,
    rtol=None,
    tempdir=None,
    chunksize=None,
    log_level=None,
    regrid_logging=None,
    relaxed_identities=None,
    bounds_combination_mode=None,
    of_fraction=None,
    collapse_parallel_mode=None,
    free_memory_factor=None,
):
    """View or set any number of constants in the project-wide
    configuration.

    The full list of global constants that can be set in any
    combination are:

    * `atol`
    * `rtol`
    * `tempdir`
    * `chunksize`
    * `log_level`
    * `regrid_logging`
    * `relaxed_identities`
    * `bounds_combination_mode`

    These are all constants that apply throughout cf, except for in
    specific functions only if overridden by the corresponding keyword
    argument to that function.

    The value of `None`, either taken by default or supplied as a
    value, will result in the constant in question not being changed
    from the current value. That is, it will have no effect.

    Note that setting a constant using this function is equivalent to
    setting it by means of a specific function of the same name,
    e.g. via `cf.atol`, but in this case multiple constants can be set
    at once.

    .. versionadded:: 3.6.0

    .. seealso:: `atol`, `rtol`, `tempdir`, `chunksize`,
                 `total_memory`, `log_level`, `regrid_logging`,
                 `relaxed_identities`, `bounds_combination_mode`

    :Parameters:

        atol: `float` or `Constant`, optional
            The new value of absolute tolerance. The default is to not
            change the current value.

        rtol: `float` or `Constant`, optional
            The new value of relative tolerance. The default is to not
            change the current value.

        tempdir: `str` or `Constant`, optional
            The new directory for temporary files. Tilde expansion (an
            initial component of ``~`` or ``~user`` is replaced by
            that *user*'s home directory) and environment variable
            expansion (substrings of the form ``$name`` or ``${name}``
            are replaced by the value of environment variable *name*)
            are applied to the new directory name.

            The default is to not change the directory.

        chunksize: `float` or `Constant`, optional
            The new chunksize in bytes. The default is to not change
            the current behaviour.

        bounds_combination_mode: `str` or `Constant`, optional
            Determine how to deal with cell bounds in binary
            operations. See `cf.bounds_combination_mode` for details.

        log_level: `str` or `int` or `Constant`, optional
            The new value of the minimal log severity level. This can
            be specified either as a string equal (ignoring case) to
            the named set of log levels or identifier ``'DISABLE'``,
            or an integer code corresponding to each of these, namely:

            * ``'DISABLE'`` (``0``);
            * ``'WARNING'`` (``1``);
            * ``'INFO'`` (``2``);
            * ``'DETAIL'`` (``3``);
            * ``'DEBUG'`` (``-1``).

        regrid_logging: `bool` or `Constant`, optional
            The new value (either True to enable logging or False to
            disable it). The default is to not change the current
            behaviour.

        relaxed_identities: `bool` or `Constant`, optional
            The new value; if True, use "relaxed" mode when getting a
            construct identity. The default is to not change the
            current value.

        of_fraction: `float` or `Constant`, optional
            Deprecated at version 3.14.0 and is no longer
            available.

        collapse_parallel_mode: `int` or `Constant`, optional
            Deprecated at version 3.14.0 and is no longer
            available.

        free_memory_factor: `float` or `Constant`, optional
            Deprecated at version 3.14.0 and is no longer
            available.

    :Returns:

        `Configuration`
            The dictionary-like object containing the names and values
            of the project-wide constants prior to the change, or the
            current names and values if no new values are specified.

    **Examples**

    >>> cf.configuration()  # view full global configuration of constants
    {'rtol': 2.220446049250313e-16,
     'atol': 2.220446049250313e-16,
     'tempdir': '/tmp',
     'regrid_logging': False,
     'relaxed_identities': False,
     'log_level': 'WARNING',
     'bounds_combination_mode': 'AND',
     'chunksize': 82873466.88000001}
    >>> cf.chunksize(7.5e7)  # any change to one constant...
    82873466.88000001
    >>> cf.configuration()['chunksize']  # ...is reflected in the configuration
    75000000.0

    >>> cf.configuration(tempdir='/usr/tmp', log_level='INFO')  # set items
    {'rtol': 2.220446049250313e-16,
     'atol': 2.220446049250313e-16,
     'tempdir': '/tmp',
     'regrid_logging': False,
     'relaxed_identities': False,
     'log_level': 'WARNING',
     'bounds_combination_mode': 'AND',
     'chunksize': 75000000.0}
    >>> cf.configuration()  # the items set have been updated accordingly
    {'rtol': 2.220446049250313e-16,
     'atol': 2.220446049250313e-16,
     'tempdir': '/usr/tmp',
     'regrid_logging': False,
     'relaxed_identities': False,
     'log_level': 'INFO',
     'bounds_combination_mode': 'AND',
     'chunksize': 75000000.0}

    Use as a context manager:

    >>> print(cf.configuration())
    {'rtol': 2.220446049250313e-16,
     'atol': 2.220446049250313e-16,
     'tempdir': '/usr/tmp',
     'regrid_logging': False,
     'relaxed_identities': False,
     'log_level': 'INFO',
     'bounds_combination_mode': 'AND',
     'chunksize': 75000000.0}
    >>> with cf.configuration(atol=9, rtol=10):
    ...     print(cf.configuration())
    ...
    {'rtol': 9.0,
     'atol': 10.0,
     'tempdir': '/usr/tmp',
     'regrid_logging': False,
     'relaxed_identities': False,
     'log_level': 'INFO',
     'bounds_combination_mode': 'AND',
     'chunksize': 75000000.0}
    >>> print(cf.configuration())
    {'rtol': 2.220446049250313e-16,
     'atol': 2.220446049250313e-16,
     'tempdir': '/usr/tmp',
     'regrid_logging': False,
     'relaxed_identities': False,
     'log_level': 'INFO',
     'bounds_combination_mode': 'AND',
     'chunksize': 75000000.0}

    """
    if of_fraction is not None:
        # TODODASKAPI
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "configuration",
            kwargs={"of_fraction": None},
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    if collapse_parallel_mode is not None:
        # TODODASKAPI
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "configuration",
            kwargs={"collapse_parallel_mode": None},
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    return _configuration(
        Configuration,
        new_atol=atol,
        new_rtol=rtol,
        new_tempdir=tempdir,
        new_chunksize=chunksize,
        new_log_level=log_level,
        new_regrid_logging=regrid_logging,
        new_relaxed_identities=relaxed_identities,
        bounds_combination_mode=bounds_combination_mode,
    )


def _configuration(_Configuration, **kwargs):
    """Internal helper function to provide the logic for
    `cf.configuration`.

    We delegate from the user-facing `cf.configuration` for two main reasons:

    1) to avoid a name clash there between the keyword arguments and the
    functions which they each call (e.g. `atol` and `cf.atol`) which
    would otherwise necessitate aliasing every such function name; and

    2) because the user-facing function must have the appropriate keywords
    explicitly listed, but the very similar logic applied for each keyword
    can be consolidated by iterating over the full dictionary of input kwargs.

    :Parameters:

        _Configuration: class
            The `Configuration` class to be returned.

    :Returns:

        `Configuration`
            The names and values of the project-wide constants prior
            to the change, or the current names and values if no new
            values are specified.

    """
    old = {name.lower(): val for name, val in CONSTANTS.items()}

    old.pop("total_memory", None)

    # Filter out 'None' kwargs from configuration() defaults. Note that this
    # does not filter out '0' or 'True' values, which is important as the user
    # might be trying to set those, as opposed to None emerging as default.
    kwargs = {name: val for name, val in kwargs.items() if val is not None}

    # Note values are the functions not the keyword arguments of same name:
    reset_mapping = {
        "new_atol": atol,
        "new_rtol": rtol,
        "new_tempdir": tempdir,
        "new_chunksize": chunksize,
        "new_log_level": log_level,
        "new_regrid_logging": regrid_logging,
        "new_relaxed_identities": relaxed_identities,
        "bounds_combination_mode": bounds_combination_mode,
    }

    old_values = {}

    try:
        # Run the corresponding func for all input kwargs
        for setting_alias, new_value in kwargs.items():
            reset_mapping[setting_alias](new_value)
            setting = setting_alias.replace("new_", "", 1)
            old_values[setting_alias] = old[setting]
    except ValueError:
        # Reset any constants that were changed prior to the exception
        for setting_alias, old_value in old_values.items():
            reset_mapping[setting_alias](old_value)

        # Raise the exception
        raise

    return _Configuration(**old)


# --------------------------------------------------------------------
# Inherit class from cfdm
# --------------------------------------------------------------------
class Configuration(cfdm.Configuration):
    def __new__(cls, *args, **kwargs):
        """Must override this method in subclasses."""
        instance = super().__new__(cls)
        instance._func = configuration
        return instance

    def __docstring_substitutions__(self):
        return _docstring_substitution_definitions

    def __docstring_package_depth__(self):
        return 0

    def __repr__(self):
        """Called by the `repr` built-in function."""
        return super().__repr__().replace("<", "<CF ", 1)


def free_memory():
    """The available physical memory.

    :Returns:

        `float`
            The amount of free memory in bytes.

    **Examples**

    >>> import numpy
    >>> print('Free memory =', cf.free_memory()/2**30, 'GiB')
    Free memory = 88.2728042603 GiB
    >>> a = numpy.arange(10**9)
    >>> print('Free memory =', cf.free_memory()/2**30, 'GiB')
    Free memory = 80.8082618713 GiB
    >>> del a
    >>> print('Free memory =', cf.free_memory()/2**30, 'GiB')
    Free memory = 88.2727928162 GiB

    """
    return _free_memory()


def FREE_MEMORY():
    """Alias for `cf.free_memory`."""
    return free_memory()


_disable_logging = cfdm._disable_logging
# We can inherit the generic logic for the cf-python log_level()
# function as contained in _log_level, but can't inherit the
# user-facing log_level() from cfdm as it operates on cfdm's CONSTANTS
# dict. Define cf-python's own.  This also means the log_level
# dostrings are independent which is important for providing
# module-specific documentation links and directives, etc.
_reset_log_emergence_level = cfdm._reset_log_emergence_level
_is_valid_log_level_int = cfdm._is_valid_log_level_int


# --------------------------------------------------------------------
# Functions inherited from cfdm
# --------------------------------------------------------------------
class ConstantAccess(cfdm.ConstantAccess):
    _CONSTANTS = CONSTANTS
    _Constant = Constant

    def __docstring_substitutions__(self):
        return _docstring_substitution_definitions

    def __docstring_package_depth__(self):
        return 0


class atol(ConstantAccess, cfdm.atol):
    pass


class rtol(ConstantAccess, cfdm.rtol):
    pass


class log_level(ConstantAccess, cfdm.log_level):
    _is_valid_log_level_int = _is_valid_log_level_int
    _reset_log_emergence_level = _reset_log_emergence_level


class regrid_logging(ConstantAccess):
    """Whether or not to enable `esmpy` regridding logging.

    If it is logging is performed after every call to `esmpy`.

    :Parameters:

        arg: `bool` or `Constant`, optional
            The new value (either `True` to enable logging or `False`
            to disable it). The default is to not change the current
            behaviour.

    :Returns:

        `Constant`
            The value prior to the change, or the current value if no
            new value was specified.

    **Examples**

    >>> cf.regrid_logging()
    False
    >>> cf.regrid_logging(True)
    False
    >>> cf.regrid_logging()
    True

    """

    _name = "REGRID_LOGGING"

    def _parse(cls, arg):
        """Parse a new constant value.

        .. versionaddedd:: 3.8.0

        :Parameters:

            cls:
                This class.

            arg:
                The given new constant value.

        :Returns:

                A version of the new constant value suitable for insertion
                into the `CONSTANTS` dictionary.

        """
        return bool(arg)


class collapse_parallel_mode(ConstantAccess):
    """Which mode to use when collapse is run in parallel. There are
    three possible modes:

    Deprecated at version 3.14.0 and is no longer available.

    0.  This attempts to maximise parallelism, possibly at the expense
        of extra communication. This is the default mode.

    1.  This minimises communication, possibly at the expense of the
        degree of parallelism. If collapse is running slower than you
        would expect, you can try changing to mode 1 to see if this
        improves performance. This is only likely to work if the
        output of collapse will be a sizeable array, not a single
        point.

    2.  This is here for debugging purposes, but we would expect this
        to maximise communication possibly at the expense of
        parallelism. The use of this mode is, therefore, not
        recommended.

    :Parameters:

        arg: `int` or `Constant`, optional
            The new value (0, 1 or 2).

    :Returns:

        `Constant`
            The value prior to the change, or the current value if no
            new value was specified.

    **Examples**

    >>> cf.collapse_parallel_mode()
    0
    >>> cf.collapse_parallel_mode(1)
    0
    >>> cf.collapse_parallel_mode()
    1

    """

    _name = "COLLAPSE_PARALLEL_MODE"

    def _parse(cls, arg):
        """Parse a new constant value.

        Deprecated at version 3.14.0 and is no longer available.

        .. versionaddedd:: 3.8.0

        :Parameters:

            cls:
                This class.

            arg:
                The given new constant value.

        :Returns:

                A version of the new constant value suitable for insertion
                into the `CONSTANTS` dictionary.

        """
        # TODODASKAPI
        _DEPRECATION_ERROR_FUNCTION(
            "collapse_parallel_mode", version="3.14.0", removed_at="5.0.0"
        )  # pragma: no cover


class relaxed_identities(ConstantAccess):
    """Use 'relaxed' mode when getting a construct identity.

    If set to True, sets ``relaxed=True`` as the default in calls to a
    construct's `identity` method (e.g. `cf.Field.identity`).

    This is used by construct arithmetic and field construct
    aggregation.

    :Parameters:

        arg: `bool` or `Constant`, optional

    :Returns:

        `Constant`
            The value prior to the change, or the current value if no
            new value was specified.

    **Examples**

    >>> org = cf.relaxed_identities()
    >>> org
    False
    >>> cf.relaxed_identities(True)
    False
    >>> cf.relaxed_identities()
    True
    >>> cf.relaxed_identities(org)
    True
    >>> cf.relaxed_identities()
    False

    """

    _name = "RELAXED_IDENTITIES"

    def _parse(cls, arg):
        """Parse a new constant value.

        .. versionaddedd:: 3.8.0

        :Parameters:

            cls:
                This class.

            arg:
                The given new constant value.

        :Returns:

                A version of the new constant value suitable for insertion
                into the `CONSTANTS` dictionary.

        """
        return bool(arg)


class chunksize(ConstantAccess):
    """Set the default chunksize used by `dask` arrays.

    If called without any arguments then the existing chunksize is
    returned.

    .. note:: Setting the chunk size will also change the `dask`
              global configuration value ``'array.chunk-size'``. If
              `chunksize` is used in a context manager then the `dask`
              configuration value is only altered within that context.
              Setting the chunk size directly from the `dask`
              configuration API will affect subsequent data creation,
              but will *not* change the value of `chunksize`.

    :Parameters:

        arg: number or `str` or `Constant`, optional
            The chunksize in bytes. Any size accepted by
            `dask.utils.parse_bytes` is accepted, for instance
            ``100``, ``'100'``, ``'1e6'``, ``'100 MB'``, ``'100M'``,
            ``'5kB'``, ``'5.4 kB'``, ``'1kiB'``, ``'1e6 kB'``, and
            ``'MB'`` are all valid sizes.

            Note that if *arg* is a `float`, or a string that implies
            a non-integral amount of bytes, then the integer part
            (rounded down) will be used.

            *Parameter example:*
               A chunksize of 2 MiB may be specified as ``'2097152'``
               or ``'2 MiB'``

            *Parameter example:*
               Chunksizes of ``'2678.9'`` and ``'2.6789 KB'`` are both
               equivalent to ``2678``.

    :Returns:

        `Constant`
            The value prior to the change, or the current value if no
            new value was specified.

    """

    _name = "CHUNKSIZE"

    def _parse(cls, arg):
        """Parse a new constant value.

        .. versionaddedd:: 3.8.0

        :Parameters:

            cls:
                This class.

            arg:
                The given new constant value.

        :Returns:

                A version of the new constant value suitable for insertion
                into the `CONSTANTS` dictionary.

        """
        _config.set({"array.chunk-size": arg})
        return parse_bytes(arg)


class tempdir(ConstantAccess):
    """The directory for internally generated temporary files.

    When setting the directory, it is created if the specified path
    does not exist.

    :Parameters:

        arg: `str`, optional
            The new directory for temporary files. Tilde expansion (an
            initial component of ``~`` or ``~user`` is replaced by
            that *user*'s home directory) and environment variable
            expansion (substrings of the form ``$name`` or ``${name}``
            are replaced by the value of environment variable *name*)
            are applied to the new directory name.

            The default is to not change the directory.

    :Returns:

        `str`
            The directory prior to the change, or the current
            directory if no new value was specified.

    **Examples**

    >>> cf.tempdir()
    '/tmp'
    >>> old = cf.tempdir('/home/me/tmp')
    >>> cf.tempdir(old)
    '/home/me/tmp'
    >>> cf.tempdir()
    '/tmp'

    """

    _name = "TEMPDIR"

    def _parse(cls, arg):
        """Parse a new constant value.

        .. versionaddedd:: 3.8.0

        :Parameters:

            cls:
                This class.

            arg:
                The given new constant value.

        :Returns:

                A version of the new constant value suitable for insertion
                into the `CONSTANTS` dictionary.

        """
        arg = _os_path_expanduser(_os_path_expandvars(arg))

        # Create the directory if it does not exist.
        try:
            mkdir(arg)
        except OSError:
            pass

        return arg


class of_fraction(ConstantAccess):
    """The amount of concurrently open files above which files
    containing data arrays may be automatically closed.

    Deprecated at version 3.14.0 and is no longer available.

    The amount is expressed as a fraction of the maximum possible
    number of concurrently open files.

    Note that closed files will be automatically reopened if
    subsequently needed by a variable to access its data array.

    .. seealso:: `cf.close_files`, `cf.close_one_file`,
                 `cf.open_files`, `cf.open_files_threshold_exceeded`

    :Parameters:

        arg: `float` or `Constant`, optional
            The new fraction (between 0.0 and 1.0). The default is to
            not change the current behaviour.

    :Returns:

        `Constant`
            The value prior to the change, or the current value if no
            new value was specified.

    **Examples**

    >>> cf.of_fraction()
    0.5
    >>> old = cf.of_fraction(0.33)
    >>> cf.of_fraction(old)
    0.33
    >>> cf.of_fraction()
    0.5

    The fraction may be translated to an actual number of files as
    follows:

    >>> old = cf.of_fraction(0.75)
    >>> import resource
    >>> max_open_files = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    >>> threshold = int(max_open_files * cf.of_fraction())
    >>> max_open_files, threshold
    (1024, 768)

    """

    _name = "OF_FRACTION"

    def _parse(cls, arg):
        """Parse a new constant value.

        Deprecated at version 3.14.0 and is no longer available.

        .. versionaddedd:: 3.8.0

        :Parameters:

            cls:
                This class.

            arg:
                The given new constant value.

        :Returns:

                A version of the new constant value suitable for insertion
                into the `CONSTANTS` dictionary.

        """
        # TODODASKAPI
        _DEPRECATION_ERROR_FUNCTION(
            "of_fraction", version="3.14.0", removed_at="5.0.0"
        )  # pragma: no cover


class free_memory_factor(ConstantAccess):
    """Set the fraction of memory kept free as a temporary workspace.

    Deprecated at version 3.14.0 and is no longer available.

    Users should set the free memory factor through cf.set_performance
    so that the upper limit to the chunksize is recalculated
    appropriately. The free memory factor must be a sensible value
    between zero and one. If no arguments are passed the existing free
    memory factor is returned.

    :Parameters:

        arg: `float` or `Constant`, optional
            The fraction of memory kept free as a temporary workspace.

    :Returns:

        `Constant`
            The value prior to the change, or the current value if no
            new value was specified.

    """

    _name = "FREE_MEMORY_FACTOR"

    def _parse(cls, arg):
        """Parse a new constant value.

        Deprecated at version 3.14.0 and is no longer available.

        .. versionaddedd:: 3.8.0

        :Parameters:

            cls:
                This class.

            arg:
                The given new constant value.

        :Returns:

                A version of the new constant value suitable for insertion
                into the `CONSTANTS` dictionary.

        """
        # TODODASKAPI
        _DEPRECATION_ERROR_FUNCTION(
            "free_memory_factor", version="3.14.0", removed_at="5.0.0"
        )  # pragma: no cover


class bounds_combination_mode(ConstantAccess):
    """Determine how to deal with cell bounds in binary operations.

    The flag returned by ``cf.bounds_combination_mode()`` is used to
    influence whether or not the result of a binary operation "op(x,
    y)", such as ``x + y``, ``x -= y``, ``x << y``, etc., will contain
    bounds, and if so how those bounds are calculated.

    The result of op(x, y) may only contain bounds if

    * ``x`` is a construct that may contain bounds, or

    * ``x`` does not support the operation and ``y`` is a construct
      that may contain bounds, e.g. ``2 + y``.

    and so the flag only has an effect in these specific cases. Only
    dimension coordinate, auxiliary coordinate and domain ancillary
    constructs support bounds.

    The behaviour for the different flag values is described by the
    following truth tables, for which it assumed that it is possible
    for the result of the operation to contain bounds:

    * If the flag is ``'AND'`` (the default) then

      ==========  ==========  ==========  ======================
      x           y           op(x, y)    Resulting bounds
      has bounds  has bounds  has bounds
      ==========  ==========  ==========  ======================
      Yes         Yes         **Yes**     op(x.bounds, y.bounds)
      Yes         No          *No*
      No          Yes         *No*
      No          No          *No*
      ==========  ==========  ==========  ======================

    * If the flag is ``'OR'`` then

      ==========  ==========  ==========  ======================
      x           y           op(x, y)    Resulting bounds
      has bounds  has bounds  has bounds
      ==========  ==========  ==========  ======================
      Yes         Yes         **Yes**     op(x.bounds, y.bounds)
      Yes         No          **Yes**     op(x.bounds, y)
      No          Yes         **Yes**     op(x, y.bounds)
      No          No          *No*
      ==========  ==========  ==========  ======================

    * If the flag is ``'XOR'`` then

      ==========  ==========  ==========  ======================
      x           y           op(x, y)    Resulting bounds
      has bounds  has bounds  has bounds
      ==========  ==========  ==========  ======================
      Yes         Yes         *No*
      Yes         No          **Yes**     op(x.bounds, y)
      No          Yes         **Yes**     op(x, y.bounds)
      No          No          *No*
      ==========  ==========  ==========  ======================

    * If the flag is ``'NONE'`` then

      ==========  ==========  ==========  ======================
      x           y           op(x, y)    Resulting bounds
      has bounds  has bounds  has bounds
      ==========  ==========  ==========  ======================
      Yes         Yes         *No*
      Yes         No          *No*
      No          Yes         *No*
      No          No          *No*
      ==========  ==========  ==========  ======================

    .. versionadded:: 3.8.0

    .. seealso:: `configuration`

    :Parameters:

        arg: `str` or `Constant`, optional
            Provide a new flag value that will apply to all subsequent
            binary operations.

    :Returns:

        `str`
            The value prior to the change, or the current value if no
            new value was specified.

    **Examples**

    >>> old = cf.bounds_combination_mode()
    >>> print(old)
    AND
    >>> print(cf.bounds_combination_mode('OR'))
    AND
    >>> print(cf.bounds_combination_mode())
    OR
    >>> print(cf.bounds_combination_mode(old))
    OR
    >>> print(cf.bounds_combination_mode())
    AND

    Use as a context manager:

    >>> print(cf.bounds_combination_mode())
    AND
    >>> with cf.bounds_combination_mode('XOR'):
    ...     print(cf.bounds_combination_mode())
    ...
    XOR
    >>> print(cf.bounds_combination_mode())
    AND

    """

    _name = "BOUNDS_COMBINATION_MODE"

    def _parse(cls, arg):
        """Parse a new constant value.

        .. versionaddedd:: 3.8.0

        :Parameters:

            cls:
                This class.

            arg:
                The given new constant value.

        :Returns:

                A version of the new constant value suitable for insertion
                into the `CONSTANTS` dictionary.

        """
        try:
            valid = hasattr(OperandBoundsCombination, arg)
        except (AttributeError, TypeError):
            valid = False

        if not valid:
            valid_vals = ", ".join(
                [repr(val.name) for val in OperandBoundsCombination]
            )
            raise ValueError(
                f"{arg!r} is not one of the valid values: {valid_vals}"
            )

        return arg


def CF():
    """The version of the CF conventions.

    This indicates which version of the CF conventions are represented
    by this release of the cf package, and therefore the version can not
    be changed.

    """
    return cfdm.CF()


CF.__doc__ = cfdm.CF.__doc__.replace("cfdm.", "cf.")


def CFA():
    """The version of the CFA conventions.

    This indicates which version of the CFA conventions are
    represented by this release of the cf package, and therefore the
    version can not be changed.

    .. versionadded:: 3.15.0

    .. seealso:: `cf.CF`

    :Returns:

        `str`
            The version of the CFA conventions represented by this
            release of the cf package.

    **Examples**

    >>> cf.CFA()
    '0.6.2'

    """
    return __cfa_version__


# Module-level alias to avoid name clashes with function keyword
# arguments (corresponding to 'import atol as cf_atol' etc. in other
# modules)
_cf_atol = atol
_cf_rtol = rtol


def _cf_chunksize(*new_chunksize):
    """Internal alias for `cf.chunksize`.

    Used in this module to prevent a name clash with a function keyword
    argument (corresponding to 'import X as cf_X' etc. in other
    modules). Note we don't use CHUNKSIZE() as it will likely be
    deprecated in future.

    """
    return chunksize(*new_chunksize)


def fm_threshold():
    """The amount of memory which is kept free as a temporary work
    space.

    Deprecated at version 3.14.0 and is no longer available.

    :Returns:

        `float`
            The amount of memory in bytes.

    **Examples**

    >>> cf.fm_threshold()
    10000000000.0

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "fm_threshold", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


def set_performance(chunksize=None, free_memory_factor=None):
    """Tune performance of parallelisation.

    Deprecated at version 3.14.0 and is no longer available.

    Sets the chunksize and free memory factor. By just providing the
    chunksize it can be changed to a smaller value than an upper
    limit, which is determined by the existing free memory factor. If
    just the free memory factor is provided then the chunksize is set
    to the corresponding upper limit.  Note that the free memory
    factor is the fraction of memory kept free as a temporary
    workspace and must be a sensible value between zero and one. If
    both arguments are provided then the free memory factor is changed
    first and then the chunksize is set provided it is consistent with
    the new free memory value. If any of the arguments is invalid then
    an error is raised and no parameters are changed.  When called
    with no arguments the existing values of the parameters are
    returned in a tuple.

    :Parameters:

        chunksize: `float`, optional
            The size in bytes of a chunk used by LAMA to partition the
            data array.

        free_memory_factor: `float`, optional
            The fraction of memory to keep free as a temporary
            workspace.

    :Returns:

        `tuple`
            A tuple of the previous chunksize and free_memory_factor.

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "set_performance", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


def min_total_memory():
    """The minimum total memory across nodes.

    Deprecated at version 3.14.0 and is no longer available.

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "min_total_memory", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


def total_memory():
    """The total amount of physical memory (in bytes)."""
    return CONSTANTS["TOTAL_MEMORY"]


# --------------------------------------------------------------------
# Aliases (for back-compatibility etc.):
# --------------------------------------------------------------------
def ATOL(*new_atol):
    """Alias for `cf.atol`."""
    return atol(*new_atol)


def RTOL(*new_rtol):
    """Alias for `cf.rtol`."""
    return rtol(*new_rtol)


def FREE_MEMORY_FACTOR(*new_free_memory_factor):
    """Alias for `cf.free_memory_factor`.

    Deprecated at version 3.14.0 and is no longer available.

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "FREE_MEMORY_FACTOR", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


def LOG_LEVEL(*new_log_level):
    """Alias for `cf.log_level`."""
    return log_level(*new_log_level)


def CHUNKSIZE(*new_chunksize):
    """Alias for `cf.chunksize`."""
    return chunksize(*new_chunksize)


def SET_PERFORMANCE(*new_set_performance):
    """Alias for `cf.set_performance`.

    Deprecated at version 3.14.0 and is no longer available.

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "SET_PERFORMANCE", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


def OF_FRACTION(*new_of_fraction):
    """Alias for `cf.of_fraction`.

    Deprecated at version 3.14.0 and is no longer available.

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "OF_FRACTION", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


def REGRID_LOGGING(*new_regrid_logging):
    """Alias for `cf.regrid_logging`."""
    return regrid_logging(*new_regrid_logging)


def COLLAPSE_PARALLEL_MODE(*new_collapse_parallel_mode):
    """Alias for `cf.collapse_parallel_mode`.

    Deprecated at version 3.14.0 and is no longer available.

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "COLLAPSE_PARALLEL_MODE", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


def RELAXED_IDENTITIES(*new_relaxed_identities):
    """Alias for `cf.relaxed_identities`."""
    return relaxed_identities(*new_relaxed_identities)


def MIN_TOTAL_MEMORY(*new_min_total_memory):
    """Alias for `cf.min_total_memory`.

    Deprecated at version 3.14.0 and is no longer available.

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "MIN_TOTAL_MEMORY", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


def TEMPDIR(*new_tempdir):
    """Alias for `cf.tempdir`."""
    return tempdir(*new_tempdir)


def TOTAL_MEMORY(*new_total_memory):
    """Alias for `cf.total_memory`."""
    return total_memory(*new_total_memory)


def FM_THRESHOLD(*new_fm_threshold):
    """Alias for `cf.fm_threshold`.

    Deprecated at version 3.14.0 and is no longer available.

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "FM_THRESHOLD", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


# def IGNORE_IDENTITIES(*arg):
#     '''TODO
#
#     :Parameters:
#
#         arg: `bool`, optional
#
#     :Returns:
#
#         `bool`
#             The value prior to the change, or the current value if no
#             new value was specified.
#
#     **Examples**
#
#     >>> org = cf.IGNORE_IDENTITIES()
#     >>> print(org)
#     False
#     >>> cf.IGNORE_IDENTITIES(True)
#     False
#     >>> cf.IGNORE_IDENTITIES()
#     True
#     >>> cf.IGNORE_IDENTITIES(org)
#     True
#     >>> cf.IGNORE_IDENTITIES()
#     False
#
#     '''
#     old = CONSTANTS['IGNORE_IDENTITIES']
#     if arg:
#         CONSTANTS['IGNORE_IDENTITIES'] = bool(arg[0])
#
#     return old


def dump(x, **kwargs):
    """Print a description of an object.

    If the object has a `!dump` method then this is used to create the
    output, so that ``cf.dump(f)`` is equivalent to ``print
    f.dump()``. Otherwise ``cf.dump(x)`` is equivalent to
    ``print(x)``.

    :Parameters:

        x:
            The object to print.

        kwargs : *optional*
            As for the input variable's `!dump` method, if it has one.

    :Returns:

        None

    **Examples**

    >>> x = 3.14159
    >>> cf.dump(x)
    3.14159

    >>> f
    <CF Field: rainfall_rate(latitude(10), longitude(20)) kg m2 s-1>
    >>> cf.dump(f)
    >>> cf.dump(f, complete=True)

    """
    if hasattr(x, "dump") and callable(x.dump):
        print(x.dump(**kwargs))
    else:
        print(x)


# _max_number_of_open_files = resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def open_files_threshold_exceeded():
    """Return True if the total number of open files is greater than the
    current threshold.

    Deprecated at version 3.14.0 and is no longer available.

    The threshold is defined as a fraction of the maximum possible number
    of concurrently open files (an operating system dependent amount). The
    fraction is retrieved and set with the `of_fraction` function.

    .. seealso:: `cf.close_files`, `cf.close_one_file`,
                 `cf.open_files`

    :Returns:

        `bool`
            Whether or not the number of open files exceeds the
            threshold.

    **Examples**

    In this example, the number of open files is 75% of the maximum
    possible number of concurrently open files:

    >>> cf.of_fraction()
    0.5
    >>> cf.open_files_threshold_exceeded()
    True
    >>> cf.of_fraction(0.9)
    >>> cf.open_files_threshold_exceeded()
    False

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "open_files_threshold_exceeded",
        version="3.14.0",
        removed_at="5.0.0",
    )  # pragma: no cover


def close_files(file_format=None):
    """Close open files containing sub-arrays of data arrays.

    Deprecated at version 3.14.0 and is no longer available.

    By default all such files are closed, but this may be restricted
    to files of a particular format.

    Note that closed files will be automatically reopened if
    subsequently needed by a variable to access the sub-array.

    If there are no appropriate open files then no action is taken.

    .. seealso:: `cf.close_one_file`, `cf.open_files`,
                 `cf.open_files_threshold_exceeded`

    :Parameters:

        file_format: `str`, optional
            Only close files of the given format. Recognised formats
            are ``'netCDF'`` and ``'PP'``. By default files of any
            format are closed.

    :Returns:

        None

    **Examples**

    >>> cf.close_files()
    >>> cf.close_files('netCDF')
    >>> cf.close_files('PP')

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "close_files", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


def close_one_file(file_format=None):
    """Close an arbitrary open file containing a sub-array of a data
    array.

    Deprecated at version 3.14.0 and is no longer available.

    By default a file of arbitrary format is closed, but the choice
    may be restricted to files of a particular format.

    Note that the closed file will be automatically reopened if
    subsequently needed by a variable to access the sub-array.

    If there are no appropriate open files then no action is taken.

    .. seealso:: `cf.close_files`, `cf.open_files`,
                 `cf.open_files_threshold_exceeded`

    :Parameters:

        file_format: `str`, optional
            Only close a file of the given format. Recognised formats
            are ``'netCDF'`` and ``'PP'``. By default a file of any
            format is closed.

    :Returns:

        `None`

    **Examples**

    >>> cf.close_one_file()
    >>> cf.close_one_file('netCDF')
    >>> cf.close_one_file('PP')

    >>> cf.open_files()
    {'netCDF': {'file1.nc': <netCDF4.Dataset at 0x181bcd0>,
                'file2.nc': <netCDF4.Dataset at 0x1e42350>,
                'file3.nc': <netCDF4.Dataset at 0x1d185e9>}}
    >>> cf.close_one_file()
    >>> cf.open_files()
    {'netCDF': {'file1.nc': <netCDF4.Dataset at 0x181bcd0>,
                'file3.nc': <netCDF4.Dataset at 0x1d185e9>}}

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "close_one_file", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


def open_files(file_format=None):
    """Return the open files containing sub-arrays of master data
    arrays.

    Deprecated at version 3.14.0 and is no longer available.

    By default all such files are returned, but the selection may be
    restricted to files of a particular format.

    .. seealso:: `cf.close_files`, `cf.close_one_file`,
                 `cf.open_files_threshold_exceeded`

    :Parameters:

        file_format: `str`, optional
            Only return files of the given format. Recognised formats
            are ``'netCDF'`` and ``'PP'``. By default all files are
            returned.

    :Returns:

        `dict`
            If *file_format* is set then return a dictionary of file
            names of the specified format and their open file
            objects. If *file_format* is not set then return a
            dictionary for which each key is a file format whose value
            is the dictionary that would have been returned if the
            *file_format* parameter was set.

    **Examples**

    >>> cf.open_files()
    {'netCDF': {'file1.nc': <netCDF4.Dataset at 0x187b6d0>}}
    >>> cf.open_files('netCDF')
    {'file1.nc': <netCDF4.Dataset at 0x187b6d0>}
    >>> cf.open_files('PP')
    {}

    """
    # TODODASKAPI
    _DEPRECATION_ERROR_FUNCTION(
        "open_files", version="3.14.0", removed_at="5.0.0"
    )  # pragma: no cover


def ufunc(name, x, *args, **kwargs):
    """The variable must have a `!copy` method and a method called.

    *name*. Any optional positional and keyword arguments are passed
    unchanged to the variable's *name* method.

    :Parameters:

        name: `str`

        x:
            The input variable.

        args, kwargs:

    :Returns:

            A new variable with size 1 axes inserted into the data
            array.

    """
    x = x.copy()
    getattr(x, name)(*args, **kwargs)
    return x


def _numpy_allclose(a, b, rtol=None, atol=None, verbose=None):
    """Returns True if two broadcastable arrays have equal values to
    within numerical tolerance, False otherwise.

    The tolerance values are positive, typically very small numbers. The
    relative difference (``rtol * abs(b)``) and the absolute difference
    ``atol`` are added together to compare against the absolute difference
    between ``a`` and ``b``.

    :Parameters:

        a, b : array_like
            Input arrays to compare.

        atol : float, optional
            The absolute tolerance for all numerical comparisons, By
            default the value returned by the `atol` function is used.

        rtol : float, optional
            The relative tolerance for all numerical comparisons, By
            default the value returned by the `rtol` function is used.

    :Returns:

        `bool`
            Returns True if the arrays are equal, otherwise False.

    **Examples**

    >>> cf._numpy_allclose([1, 2], [1, 2])
    True
    >>> cf._numpy_allclose(numpy.array([1, 2]), numpy.array([1, 2]))
    True
    >>> cf._numpy_allclose([1, 2], [1, 2, 3])
    False
    >>> cf._numpy_allclose([1, 2], [1, 4])
    False

    >>> a = numpy.ma.array([1])
    >>> b = numpy.ma.array([2])
    >>> a[0] = numpy.ma.masked
    >>> b[0] = numpy.ma.masked
    >>> cf._numpy_allclose(a, b)
    True

    """
    # TODO: we want to use @_manage_log_level_via_verbosity on this function
    # but we cannot, since importing it to this module would lead to a
    # circular import dependency with the decorators module. Tentative plan
    # is to move the function elsewhere. For now, it is not 'loggified'.

    # THIS IS WHERE SOME NUMPY FUTURE WARNINGS ARE COMING FROM

    a_is_masked = np.ma.isMA(a)
    b_is_masked = np.ma.isMA(b)

    if not (a_is_masked or b_is_masked):
        try:
            return np.allclose(a, b, rtol=rtol, atol=atol)
        except (IndexError, NotImplementedError, TypeError):
            return np.all(a == b)
    else:
        if a_is_masked and b_is_masked:
            if (a.mask != b.mask).any():
                if verbose:
                    print("Different masks (A)")

                return False
        else:
            if np.ma.is_masked(a) or np.ma.is_masked(b):
                if verbose:
                    print("Different masks (B)")

                return False

        try:
            return np.ma.allclose(a, b, rtol=rtol, atol=atol)
        except (IndexError, NotImplementedError, TypeError):
            # To prevent a bug causing some header/coord-only CDL reads or
            # aggregations to error. See also TODO comment below.
            if a.dtype == b.dtype:
                out = np.ma.all(a == b)
            else:
                # TODO: is this most sensible? Or should we attempt dtype
                # conversion and then compare? Probably we should avoid
                # altogether by catching the different dtypes upstream?
                out = False
            if out is np.ma.masked:
                return True
            else:
                return out


def indices_shape(indices, full_shape, keepdims=True):
    """Return the shape of the array subspace implied by indices.

    **Performance**

    Boolean `dask` arrays will be computed, and `dask` arrays with
    unknown size will have their chunk sizes computed.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.parse_indices`

    :Parameters:

        indices: `tuple`
            The indices to be applied to an array with shape
            *full_shape*.

        full_shape: sequence of `ints`
            The shape of the array to be subspaced.

        keepdims: `bool`, optional
            If True then an integral index is converted to a
            slice. For instance, ``3`` would become ``slice(3, 4)``.

    :Returns:

        `list`
            The shape of the subspace defined by the *indices*.

    **Examples**

    >>> import numpy as np
    >>> import dask.array as da

    >>> cf.indices_shape((slice(2, 5), 4), (10, 20))
    [3, 1]
    >>> cf.indices_shape(([2, 3, 4], np.arange(1, 6)), (10, 20))
    [3, 5]

    >>> index0 = [False] * 5
    >>> index0[2:5] = [True] * 3
    >>> cf.indices_shape((index0, da.arange(1, 6)), (10, 20))
    [3, 5]

    >>> index0 = da.full((5,), False, dtype=bool)
    >>> index0[2:5] = True
    >>> index1 = np.full((6,), False, dtype=bool)
    >>> index1[1:6] = True
    >>> cf.indices_shape((index0, index1), (10, 20))
    [3, 5]

    >>> index0 = da.arange(5)
    >>> index0 = index0[index0 < 3]
    >>> cf.indices_shape((index0, []), (10, 20))
    [3, 0]

    >>> cf.indices_shape((da.from_array(2), np.array(3)), (10, 20))
    [1, 1]
    >>> cf.indices_shape((da.from_array([]), np.array(())), (10, 20))
    [0, 0]
    >>> cf.indices_shape((slice(1, 5, 3), 3), (10, 20))
    [2, 1]
    >>> cf.indices_shape((slice(5, 1, -2), 3), (10, 20))
    [2, 1]
    >>> cf.indices_shape((slice(5, 1, 3), 3), (10, 20))
    [0, 1]
    >>> cf.indices_shape((slice(1, 5, -3), 3), (10, 20))
    [0, 1]

    >>> cf.indices_shape((slice(2, 5), 4), (10, 20), keepdims=False)
    [3]
    >>> cf.indices_shape((da.from_array(2), 3), (10, 20), keepdims=False)
    []
    >>> cf.indices_shape((2, np.array(3)), (10, 20), keepdims=False)
    []

    """
    shape = []
    for index, full_size in zip(indices, full_shape):
        if isinstance(index, slice):
            start, stop, step = index.indices(full_size)
            if (stop - start) * step < 0:
                # E.g. 5:1:3 or 1:5:-3
                size = 0
            else:
                size = abs((stop - start) / step)
                int_size = round(size)
                if size > int_size:
                    size = int_size + 1
                else:
                    size = int_size
        elif is_dask_collection(index) or isinstance(index, np.ndarray):
            if index.dtype == bool:
                # Size is the number of True values in the array
                size = int(index.sum())
            else:
                size = index.size
                if isnan(size):
                    index.compute_chunk_sizes()
                    size = index.size

            if not keepdims and not index.ndim:
                # Scalar array
                continue
        elif isinstance(index, list):
            size = len(index)
            if size:
                i = index[0]
                if isinstance(i, bool):
                    # Size is the number of True values in the list
                    size = sum(index)
        else:
            # Index is Integral
            if not keepdims:
                continue

            size = 1

        shape.append(size)

    return shape


def parse_indices(shape, indices, cyclic=False, keepdims=True):
    """Parse indices for array access and assignment.

    :Parameters:

        shape: sequence of `ints`
            The shape of the array.

        indices: `tuple`
            The indices to be applied.

        keepdims: `bool`, optional
            If True then an integral index is converted to a
            slice. For instance, ``3`` would become ``slice(3, 4)``.

    :Returns:

        `list` [, `dict`]
            The parsed indices. If *cyclic* is True then a dictionary
            is also returned that contains the parameters needed to
            interpret any cyclic slices.

    **Examples**

    >>> cf.parse_indices((5, 8), ([1, 2, 4, 6],))
    [array([1, 2, 4, 6]), slice(None, None, None)]
    >>> cf.parse_indices((5, 8), (Ellipsis, [2, 4, 6]))
    [slice(None, None, None), [2, 4, 6]]
    >>> cf.parse_indices((5, 8), (Ellipsis, 4))
    [slice(None, None, None), slice(4, 5, 1)]
    >>> cf.parse_indices((5, 8), (Ellipsis, 4), keepdims=False)
    [slice(None, None, None), 4]
    >>> cf.parse_indices((5, 8), (slice(-2, 2)), cyclic=False)
    [slice(-2, 2, None), slice(None, None, None)]
    >>> cf.parse_indices((5, 8), (slice(-2, 2)), cyclic=True)
    ([slice(0, 4, 1), slice(None, None, None)], {0: 2})
    >>> cf.parse_indices((5, 8), (cf.Data([1, 3]),))
    [dask.array<array, shape=(2,), dtype=int64, chunksize=(2,), chunktype=numpy.ndarray>, slice(None, None, None)]


    """
    parsed_indices = []
    roll = {}

    if not isinstance(indices, tuple):
        indices = (indices,)

    # Initialise the list of parsed indices as the input indices with any
    # Ellipsis objects expanded
    length = len(indices)
    n = len(shape)
    ndim = n
    for index in indices:
        if index is Ellipsis:
            m = n - length + 1
            parsed_indices.extend([slice(None)] * m)
            n -= m
        else:
            parsed_indices.append(index)
            n -= 1

        length -= 1

    len_parsed_indices = len(parsed_indices)

    if ndim and len_parsed_indices > ndim:
        raise IndexError(
            f"Invalid indices {parsed_indices} for array with shape {shape}"
        )

    if len_parsed_indices < ndim:
        parsed_indices.extend([slice(None)] * (ndim - len_parsed_indices))

    if not ndim and parsed_indices:
        raise IndexError(
            "Scalar array can only be indexed with () or Ellipsis"
        )

    for i, (index, size) in enumerate(zip(parsed_indices, shape)):
        if cyclic and isinstance(index, slice):
            # Check for a cyclic slice
            try:
                index = normalize_slice(index, size, cyclic=True)
            except IndexError:
                # Non-cyclic slice
                pass
            else:
                # Cyclic slice
                start = index.start
                stop = index.stop
                step = index.step
                if (
                    step > 0
                    and -size <= start < 0
                    and 0 <= stop <= size + start
                ):
                    # x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    # x[ -1:0:1] => [9]
                    # x[ -1:1:1] => [9, 0]
                    # x[ -1:3:1] => [9, 0, 1, 2]
                    # x[ -1:9:1] => [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
                    # x[ -4:0:1] => [6, 7, 8, 9]
                    # x[ -4:1:1] => [6, 7, 8, 9, 0]
                    # x[ -4:3:1] => [6, 7, 8, 9, 0, 1, 2]
                    # x[ -4:6:1] => [6, 7, 8, 9, 0, 1, 2, 3, 4, 5]
                    # x[ -9:0:1] => [1, 2, 3, 4, 5, 6, 7, 8, 9]
                    # x[ -9:1:1] => [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                    # x[-10:0:1] => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    index = slice(0, stop - start, step)
                    roll[i] = -start

                elif (
                    step < 0 and 0 <= start < size and start - size <= stop < 0
                ):
                    # x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    # x[0: -4:-1] => [0, 9, 8, 7]
                    # x[6: -1:-1] => [6, 5, 4, 3, 2, 1, 0]
                    # x[6: -2:-1] => [6, 5, 4, 3, 2, 1, 0, 9]
                    # x[6: -4:-1] => [6, 5, 4, 3, 2, 1, 0, 9, 8, 7]
                    # x[0: -2:-1] => [0, 9]
                    # x[0:-10:-1] => [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]
                    index = slice(start - stop - 1, None, step)
                    roll[i] = -1 - stop

        elif keepdims and isinstance(index, Integral):
            # Convert an integral index to a slice
            if index == -1:
                index = slice(-1, None, None)
            else:
                index = slice(index, index + 1, 1)

        elif hasattr(index, "to_dask_array"):
            to_dask_array = index.to_dask_array
            if callable(to_dask_array):
                # Replace index with its Dask array
                index = to_dask_array()

        parsed_indices[i] = index

    if not cyclic:
        return parsed_indices

    return parsed_indices, roll


def normalize_slice(index, size, cyclic=False):
    """Normalise a slice.

    If *index* is not a slice, or *cyclic* is True and *index* is not
    a cyclic slice, then an `IndexError` is raised.

    .. versionadded:: 3.16.2

    :Parameters:

        index:
            The index to be normalised.

        size: `int`
            The size of the axis to which the *index* applies.

        cyclic: `bool`
            If True then normalise a cyclic slice, and raise an
            exception if the *index* is a not a cyclic slice.

    :Returns:

        `slice`
            The normalised slice.

    **Examples**

    >>> cf.normalize_slice(slice(1, 4), 8)
    slice(1, 4, 1)
    >>> cf.normalize_slice(slice(None), 8)
    slice(0, 8, 1)
    >>> cf.normalize_slice(slice(6, None, -1), 8)
    slice(6, None, -1)
    >>> cf.normalize_slice(slice(-2, 4), 8)
    slice(6, 4, 1)

    >>> cf.normalize_slice([1, 2], 8)
    IndexError: [1, 2] is not a slice

    >>> cf.normalize_slice(slice(-2, 3), 8, cyclic=True)
    slice(-2, 3, 1)
    >>> cf.normalize_slice(slice(6, 3), 8, cyclic=True)
    slice(-2, 3, 1)

    >>> cf.normalize_slice(slice(2, -3, -1), 8, cyclic=True)
    slice(2, -3, -1)
    >>> cf.normalize_slice(slice(2, 5, -1), 8, cyclic=True)
    slice(2, -3, -1)

    >>> cf.normalize_slice(slice(1, 6), 8, cyclic=True)
    IndexError: slice(1, 6, None) is not a cyclic slice
    >>> cf.normalize_slice([1, 2, 3, 4], 8, cyclic=True)
    IndexError: [1, 2, 3, 4] is not a cyclic slice

    """
    if not isinstance(index, slice):
        step = 0
    elif cyclic:
        start = index.start
        stop = index.stop
        step = index.step
        if start is None or stop is None:
            step = 0
        elif step is None:
            step = 1

        if step > 0:
            if 0 < start < size and 0 <= stop <= start:
                # 6:0:1 => -4:0:1
                # 6:1:1 => -4:1:1
                # 6:3:1 => -4:3:1
                # 6:6:1 => -4:6:1
                start -= size
            elif -size <= start < 0 and -size <= stop <= start:
                # -4:-10:1  => -4:1:1
                # -4:-9:1   => -4:1:1
                # -4:-7:1   => -4:3:1
                # -4:-4:1   => -4:6:1
                # -10:-10:1 => -10:0:1
                stop += size
        elif step < 0:
            if -size <= start < 0 and start <= stop < 0:
                # -4:-1:-1   => 6:-1:-1
                # -4:-2:-1   => 6:-2:-1
                # -4:-4:-1   => 6:-4:-1
                # -10:-2:-1  => 0:-2:-1
                # -10:-10:-1 => 0:-10:-1
                start += size
            elif 0 <= start < size and start < stop < size:
                # 0:6:-1 => 0:-4:-1
                # 3:6:-1 => 3:-4:-1
                # 3:9:-1 => 3:-1:-1
                stop -= size
    else:
        start, stop, step = index.indices(size)
        if step < 0 and stop < 0:
            stop = None

        # Return the normalized non-cyclic slice
        return slice(start, stop, step)

    if not (
        (step > 0 and start < 0 and stop > 0)
        or (step < 0 and start > 0 and stop < 0)
    ):
        raise IndexError(
            f"{index!r} is not a {'cyclic ' if cyclic else ''}slice"
        )

    # Return the normalized cyclic slice
    return slice(start, stop, step)


def get_subspace(array, indices):
    """Return a subspace defined by the given indices of an array.

    Subset the input numpy array with the given indices. Indexing is
    similar to that of a numpy array. The differences to numpy array
    indexing are:

    1. An integer index i takes the i-th element but does not reduce
       the rank of the output array by one.

    2. When more than one dimension's slice is a 1-d boolean array or
       1-d sequence of integers then these indices work independently
       along each dimension (similar to the way vector subscripts work
       in Fortran).

    Indices must contain an index for each dimension of the input array.

    :Parameters:

        array: `numpy.ndarray`

        indices: `list`

    """
    gg = [i for i, x in enumerate(indices) if not isinstance(x, slice)]
    len_gg = len(gg)

    if len_gg < 2:
        # ------------------------------------------------------------
        # At most one axis has a list-of-integers index so we can do a
        # normal numpy subspace
        # ------------------------------------------------------------
        return array[tuple(indices)]

    else:
        # ------------------------------------------------------------
        # At least two axes have list-of-integers indices so we can't
        # do a normal numpy subspace
        # ------------------------------------------------------------
        if np.ma.isMA(array):
            take = np.ma.take
        else:
            take = np.take

        indices = indices[:]
        for axis in gg:
            array = take(array, indices[axis], axis=axis)
            indices[axis] = slice(None)

        if len_gg < len(indices):
            array = array[tuple(indices)]

        return array


_equals = cfdm.Data()._equals


def equals(x, y, rtol=None, atol=None, ignore_data_type=False, **kwargs):
    """True if two objects are equal within a given tolerance."""
    if rtol is None:
        rtol = _cf_rtol()

    if atol is None:
        atol = _cf_atol()

    return _equals(
        x, y, rtol=rtol, atol=atol, ignore_data_type=ignore_data_type, **kwargs
    )


def equivalent(x, y, rtol=None, atol=None, traceback=False):
    """True if and only if two objects are logically equivalent.

    If the first argument, *x*, has an `!equivalent` method then it is
    used, and in this case ``equivalent(x, y)`` is the same as
    ``x.equivalent(y)``.

    :Parameters:

        x, y :
            The objects to compare for equivalence.

        atol : float, optional
            The absolute tolerance for all numerical comparisons, By
            default the value returned by the `atol` function is used.

        rtol : float, optional
            The relative tolerance for all numerical comparisons, By
            default the value returned by the `rtol` function is used.

        traceback : bool, optional
            If True then print a traceback highlighting where the two
            objects differ.

    :Returns:

        `bool`
            Whether or not the two objects are equivalent.

    **Examples**

    >>> f
    <CF Field: rainfall_rate(latitude(10), longitude(20)) kg m2 s-1>
    >>> cf.equivalent(f, f)
    True

    >>> cf.equivalent(1.0, 1.0)
    True
    >>> cf.equivalent(1.0, 33)
    False

    >>> cf.equivalent('a', 'a')
    True
    >>> cf.equivalent('a', 'b')
    False

    >>> cf.equivalent(cf.Data(1000, units='m'), cf.Data(1, units='km'))
    True

    For a field, ``f``:

    >>> cf.equivalent(f, f.transpose())
    True

    """

    if rtol is None:
        rtol = _cf_rtol()

    if atol is None:
        atol = _cf_atol()

    atol = float(atol)
    rtol = float(rtol)

    eq = getattr(x, "equivalent", None)
    if callable(eq):
        # x has a callable equivalent method
        return eq(y, rtol=rtol, atol=atol, traceback=traceback)

    eq = getattr(y, "equivalent", None)
    if callable(eq):
        # y has a callable equivalent method
        return eq(x, rtol=rtol, atol=atol, traceback=traceback)

    return equals(
        x, y, rtol=rtol, atol=atol, ignore_fill_value=True, traceback=traceback
    )


def load_stash2standard_name(table=None, delimiter="!", merge=True):
    """Load a STASH to standard name conversion table from a file.

    This used when reading PP and UM fields files.

    Each mapping is defined by a separate line in a text file. Each
    line contains nine ``!``-delimited entries:

    1. ID: UM sub model identifier (1 = atmosphere, 2 = ocean, etc.)
    2. STASH: STASH code (e.g. 3236)
    3. STASHmaster description:STASH name as given in the STASHmaster
       files
    4. Units: Units of this STASH code (e.g. 'kg m-2')
    5. Valid from: This STASH valid from this UM version (e.g. 405)
    6. Valid to: This STASH valid to this UM version (e.g. 501)
    7. CF standard name: The CF standard name
    8. CF info: Anything useful (such as standard name modifiers)
    9. PP conditions: PP conditions which need to be satisfied for
       this translation

    Only entries "ID", "STASH", and "CF standard name" are mandatory,
    all other entries may be left blank. For example,
    ``1!999!!!!!ultraviolet_index!!`` is a valid mapping from
    atmosphere STASH code 999 to the standard name
    ultraviolet_index.

    If the "Valid from" and "Valid to" entries are omitted then the
    stash mapping is assumed to apply to all UM versions.

    .. seealso:: `stash2standard_name`

    :Parameters:

        table: `str`, optional
            Use the conversion table at this file location. By default
            the table will be looked for at
            ``os.path.join(os.path.dirname(cf.__file__),'etc/STASH_to_CF.txt')``

            Setting *table* to `None` will reset the table, removing
            any modifications that have previously been made.

        delimiter: `str`, optional
            The delimiter of the table columns. By default, ``!`` is
            taken as the delimiter.

        merge: `bool`, optional
            If False then the table is updated to only contain the
            mappings defined by the *table* parameter. By default the
            mappings defined by the *table* parameter are incorporated
            into the existing table, overwriting any entries which
            already exist.

            If *table* is `None` then *merge* is taken as False,
            regardless of its given value.

    :Returns:

        `dict`
            The new STASH to standard name conversion table.

    **Examples**

    >>> cf.load_stash2standard_name()
    >>> cf.load_stash2standard_name('my_table.txt')
    >>> cf.load_stash2standard_name('my_table2.txt', ',')
    >>> cf.load_stash2standard_name('my_table3.txt', merge=True)
    >>> cf.load_stash2standard_name('my_table4.txt', merge=False)

    """
    # 0  Model
    # 1  STASH code
    # 2  STASH name
    # 3  units
    # 4  valid from UM vn
    # 5  valid to   UM vn
    # 6  standard_name
    # 7  CF extra info
    # 8  PP extra info

    # Number matching regular expression
    number_regex = r"([-+]?\d*\.?\d+(e[-+]?\d+)?)"

    if table is None:
        # Use default conversion table
        merge = False
        package_path = os.path.dirname(__file__)
        table = os.path.join(package_path, "etc/STASH_to_CF.txt")
    else:
        # User supplied table
        table = abspath(os.path.expanduser(os.path.expandvars(table)))

    with open(table, "r") as open_table:
        lines = csv.reader(
            open_table, delimiter=delimiter, skipinitialspace=True
        )
        lines = list(lines)

    raw_list = []
    [raw_list.append(line) for line in lines]

    # Get rid of comments
    for line in raw_list[:]:
        if line[0].startswith("#"):
            raw_list.pop(0)
            continue

        break

    # Convert to a dictionary which is keyed by (submodel, STASHcode)
    # tuples
    (
        model,
        stash,
        name,
        units,
        valid_from,
        valid_to,
        standard_name,
        cf,
        pp,
    ) = list(range(9))

    stash2sn = {}
    for x in raw_list:
        key = (int(x[model]), int(x[stash]))

        if not x[units]:
            x[units] = None

        try:
            cf_info = {}
            if x[cf]:
                for d in x[7].split():
                    if d.startswith("height="):
                        cf_info["height"] = re.split(
                            number_regex, d, re.IGNORECASE
                        )[1:4:2]
                        if cf_info["height"] == "":
                            cf_info["height"][1] = "1"

                    if d.startswith("below_"):
                        cf_info["below"] = re.split(
                            number_regex, d, re.IGNORECASE
                        )[1:4:2]
                        if cf_info["below"] == "":
                            cf_info["below"][1] = "1"

                    if d.startswith("where_"):
                        cf_info["where"] = d.replace("where_", "where ", 1)
                    if d.startswith("over_"):
                        cf_info["over"] = d.replace("over_", "over ", 1)

            x[cf] = cf_info
        except IndexError:
            pass

        try:
            x[valid_from] = float(x[valid_from])
        except ValueError:
            x[valid_from] = None

        try:
            x[valid_to] = float(x[valid_to])
        except ValueError:
            x[valid_to] = None

        x[pp] = x[pp].rstrip()

        line = (x[name:],)

        if key in stash2sn:
            stash2sn[key] += line
        else:
            stash2sn[key] = line

    if not merge:
        _stash2standard_name.clear()

    _stash2standard_name.update(stash2sn)


def stash2standard_name():
    """Return a copy of the loaded STASH to standard name conversion
    table.

    .. versionadded:: 3.8.0

    .. seealso:: `load_stash2standard_name`

    """
    return _stash2standard_name.copy()


def flat(x):
    """Return an iterator over an arbitrarily nested sequence.

    :Parameters:

        x: scalar or arbitrarily nested sequence
            The arbitrarily nested sequence to be flattened. Note that
            a If *x* is a string or a scalar then this is equivalent
            to passing a single element sequence containing *x*.

    :Returns:

        generator
            An iterator over flattened sequence.

    **Examples**

    >>> cf.flat([1, [2, [3, 4]]])
    <generator object flat at 0x3649cd0>

    >>> list(cf.flat([1, (2, [3, 4])]))
    [1, 2, 3, 4]

    >>> import numpy
    >>> list(cf.flat((1, [2, numpy.array([[3, 4], [5, 6]])])))
    [1, 2, 3, 4, 5, 6]

    >>> for a in cf.flat([1, [2, [3, 4]]]):
    ...     print(a, end=' ')
    ...
    1 2 3 4

    >>> for a in cf.flat(['a', ['bc', ['def', 'ghij']]]):
    ...     print(a, end=' ')
    ...
    a bc def ghij

    >>> for a in cf.flat(2004):
    ...     print(a)
    ...
    2004

    >>> for a in cf.flat('abcdefghij'):
    ...     print(a, end=' ')
    ...
    abcdefghij

    >>> f
    <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>
    >>> for a in cf.flat(f):
    ...     print(repr(a))
    ...
    <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>

    >>> for a in cf.flat([f, [f, [f, f]]]):
    ...     print(repr(a))
    ...
    <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>
    <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>
    <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>
    <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>

    >>> fl = cf.FieldList(cf.flat([f, [f, [f, f]]]))
    >>> fl
    [<CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>,
     <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>,
     <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>,
     <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>]

    """
    if not isinstance(x, Iterable) or isinstance(x, str):
        x = (x,)

    for a in x:
        if not isinstance(a, str) and isinstance(a, Iterable):
            for sub in flat(a):
                yield sub
        else:
            yield a


def abspath(filename):
    """Return a normalized absolute version of a file name.

    If `None` or a string containing URL is provided then it is
    returned unchanged.

    .. seealso:: `cf.dirname`, `cf.pathjoin`, `cf.relpath`

    :Parameters:

        filename: `str` or `None`
            The name of the file, or `None`

    :Returns:

        `str`

            The normalized absolutised version of *filename*, or
            `None`.

    **Examples**

    >>> import os
    >>> os.getcwd()
    '/data/archive'
    >>> cf.abspath('file.nc')
    '/data/archive/file.nc'
    >>> cf.abspath('..//archive///file.nc')
    '/data/archive/file.nc'
    >>> cf.abspath('http://data/archive/file.nc')
    'http://data/archive/file.nc'

    """
    u = urlparse(filename)
    scheme = u.scheme
    if not scheme:
        return _os_path_abspath(filename)

    if scheme == "file":
        return u.path

    return filename


def relpath(filename, start=None):
    """Return a relative filepath to a file.

    The filepath is relative either from the current directory or from
    an optional start point.

    If a string containing URL is provided then it is returned unchanged.

    .. seealso:: `cf.abspath`, `cf.dirname`, `cf.pathjoin`

    :Parameters:

        filename: `str`
            The name of the file.

        start: `str`, optional
            The start point for the relative path. By default the
            current directory is used.

    :Returns:

        `str`
            The relative path.

    **Examples**

    >>> cf.relpath('/data/archive/file.nc')
    '../file.nc'
    >>> cf.relpath('/data/archive///file.nc', start='/data')
    'archive/file.nc'
    >>> cf.relpath('http://data/archive/file.nc')
    'http://data/archive/file.nc'

    """
    u = urllib.parse.urlparse(filename)
    if u.scheme != "":
        return filename

    if start is not None:
        return _os_path_relpath(filename, start)

    return _os_path_relpath(filename)


def dirname(filename):
    """Return the directory name of a file.

    If a string containing URL is provided then everything up to, but
    not including, the last slash (/) is returned.

    .. seealso:: `cf.abspath`, `cf.pathjoin`, `cf.relpath`

    :Parameters:

        filename: `str`
            The name of the file.

    :Returns:

        `str`
            The directory name.

    **Examples**

    >>> cf.dirname('/data/archive/file.nc')
    '/data/archive'
    >>> cf.dirname('..//file.nc')
    '..'
    >>> cf.dirname('http://data/archive/file.nc')
    'http://data/archive'

    """
    u = urllib.parse.urlparse(filename)
    if u.scheme != "":
        return filename.rpartition("/")[0]

    return _os_path_dirname(filename)


def pathjoin(path1, path2):
    """Join two file path components intelligently.

    If either of the paths is a URL then a URL will be returned

    .. seealso:: `cf.abspath`, `cf.dirname`, `cf.relpath`

    :Parameters:

        path1: `str`
            The first component of the path.

        path2: `str`
            The second component of the path.

    :Returns:

        `str`
            The joined paths.

    **Examples**

    >>> cf.pathjoin('/data/archive', '../archive/file.nc')
    '/data/archive/../archive/file.nc'
    >>> cf.pathjoin('/data/archive', '../archive/file.nc')
    '/data/archive/../archive/file.nc'
    >>> cf.abspath(cf.pathjoin('/data/', 'archive/'))
    '/data/archive'
    >>> cf.pathjoin('http://data', 'archive/file.nc')
    'http://data/archive/file.nc'

    """
    u = urllib.parse.urlparse(path1)
    if u.scheme != "":
        return urllib.parse.urljoin(path1, path2)

    return _os_path_join(path1, path2)


def hash_array(array, algorithm=None):
    """Return a hash value of a numpy array.

    Deprecated at version 3.15.0 and is no longer available. Use
    `dask.base.tokenize` instead.

    The hash value is dependent on the data type and the shape of the
    array. If the array is a masked array then the hash value is
    independent of the fill value and of data array values underlying
    any masked elements.

    :Parameters:

        array: `numpy.ndarray`
            The numpy array to be hashed. May be a masked array.

        algorithm: `hashlib` constructor function
            Constructor function for the desired hash algorithm,
            e.g. `hashlib.md5`, `hashlib.sha256`, etc.

            .. versionadded:: 3.14.0

    :Returns:

        `int`
            The hash value.

    **Examples**

    >>> a = np.array([[0, 1, 2, 3]])
    >>> cf.hash_array(a)
    -5620332080097671134

    >>> a = np.ma.array([[0, 1, 2, 3]], mask=[[0, 1, 0, 0]])
    >>> cf.hash_array(array)
    8372868545804866378

    >>> a[0, 1] = 999
    >>> a[0, 1] = np.ma.masked
    >>> print(a)
    [[0 -- 2 3]]
    >>> print(a.data)
    [[  0 999   2   3]]
    >>> cf.hash_array(a)
    8372868545804866378

    >>> a = a.astype(float)
    >>> cf.hash_array(a)
    5950106833921144220

    """
    _DEPRECATION_ERROR_FUNCTION(
        "hash_array",
        "Use 'dask.base.tokenize' instead.",
        version="3.15.0",
        removed_at="5.0.0",
    )  # pragma: no cover


def inspect(self):
    """Inspect the attributes of an object.

    :Returns:

        `None`

    """
    from pprint import pprint

    try:
        name = repr(self)
    except Exception:
        name = self.__class__.__name__

    print("\n".join([name, "".ljust(len(name), "-")]))

    if hasattr(self, "__dict__"):
        pprint(self.__dict__)


def broadcast_array(array, shape):
    """Broadcast an array to a given shape.

    It is assumed that ``numpy.ndim(array) <= len(shape)`` and that
    the array is broadcastable to the shape by the normal numpy
    broadcasting rules, but neither of these things is checked.

    For example, ``a[...] = broadcast_array(a, b.shape)`` is
    equivalent to ``a[...] = b``.

    :Parameters:

        a: numpy array-like

        shape: `tuple`

    :Returns:

        `numpy.ndarray`

    **Examples**


    >>> a = numpy.arange(8).reshape(2, 4)
    [[0 1 2 3]
     [4 5 6 7]]

    >>> print(cf.broadcast_array(a, (3, 2, 4)))
    [[[0 1 2 3]
      [4 5 6 0]]

     [[0 1 2 3]
      [4 5 6 0]]

     [[0 1 2 3]
      [4 5 6 0]]]

    >>> a = numpy.arange(8).reshape(2, 1, 4)
    [[[0 1 2 3]]

     [[4 5 6 7]]]

    >>> print(cf.broadcast_array(a, (2, 3, 4)))
    [[[0 1 2 3]
      [0 1 2 3]
      [0 1 2 3]]

     [[4 5 6 7]
      [4 5 6 7]
      [4 5 6 7]]]

    >>> a = numpy.ma.arange(8).reshape(2, 4)
    >>> a[1, 3] = numpy.ma.masked
    >>> print(a)
    [[0 1 2 3]
     [4 5 6 --]]

    >>> cf.broadcast_array(a, (3, 2, 4))
    [[[0 1 2 3]
      [4 5 6 --]]

     [[0 1 2 3]
      [4 5 6 --]]

     [[0 1 2 3]
      [4 5 6 --]]]

    """
    a_shape = np.shape(array)
    if a_shape == shape:
        return array

    tile = [(m if n == 1 else 1) for n, m in zip(a_shape[::-1], shape[::-1])]
    tile = shape[0 : len(shape) - len(a_shape)] + tuple(tile[::-1])

    return np.tile(array, tile)


def allclose(x, y, rtol=None, atol=None):
    """Returns True if two broadcastable arrays have equal values to
    within numerical tolerance, False otherwise.

    The tolerance values are positive, typically very small
    numbers. The relative difference (``rtol * abs(b)``) and the
    absolute difference ``atol`` are added together to compare against
    the absolute difference between ``a`` and ``b``.

    :Parameters:

        x, y: array_like
            Input arrays to compare.

        atol: `float`, optional
            The absolute tolerance for all numerical comparisons, By
            default the value returned by the `atol` function is used.

        rtol: `float`, optional
            The relative tolerance for all numerical comparisons, By
            default the value returned by the `rtol` function is used.

    :Returns:

        `bool`
            Returns True if the arrays are equal, otherwise False.

    **Examples**

    """
    if rtol is None:
        rtol = _cf_rtol()

    if atol is None:
        atol = _cf_atol()

    atol = float(atol)
    rtol = float(rtol)

    allclose = getattr(x, "allclose", None)
    if callable(allclose):
        # x has a callable allclose method
        return allclose(y, rtol=rtol, atol=atol)

    allclose = getattr(y, "allclose", None)
    if callable(allclose):
        # y has a callable allclose method
        return allclose(x, rtol=rtol, atol=atol)

    # x nor y has a callable allclose method
    return _numpy_allclose(x, y, rtol=rtol, atol=atol)


def _section(x, axes=None, stop=None, chunks=False, min_step=1):
    """Return a list of m dimensional sections of a Field of n
    dimensions or a dictionary of m dimensional sections of a Data
    object of n dimensions, where m <= n.

    In the case of a `Data` object, the keys of the dictionary are the
    indices of the sections in the original Data object. The m
    dimensions that are not sliced are marked with `None` as a
    placeholder making it possible to reconstruct the original data
    object. The corresponding values are the resulting sections of
    type `Data`.

    :Parameters:

        x: `Field` or `Data`
            The `Field` or `Data` object to be sectioned.

        axes: optional
            In the case of a Field this is a query for the m axes that
            define the sections of the Field as accepted by the Field
            object's axes method. The keyword arguments are also
            passed to this method. See `cf.Field.axes` for details. If
            an axis is returned that is not a data axis it is ignored,
            since it is assumed to be a dimension coordinate of size
            1. In the case of a Data object this should be a tuple or
            a list of the m indices of the m axes that define the
            sections of the Data object. If axes is None (the default)
            all axes are selected.

            Note: the axes specified by the *axes* parameter are the
            one which are to be kept whole. All other axes are
            sectioned

        data: `bool`, optional
            If True this indicates that a data object has been passed,
            if False it indicates that a field object has been
            passed. By default it is False.

        stop: `int`, optional
            Deprecated at version 3.14.0.

            Stop after taking this number of sections and return. If
            stop is None all sections are taken.

        chunks: `bool`, optional
            Deprecated at version 3.14.0. Consider using
            `cf.Data.rechunk` instead.

            If True return sections that are of the maximum possible
            size that will fit in one chunk of memory instead of
            sectioning into slices of size 1 along the dimensions that
            are being sectioned.

        min_step: `int`, optional
            The minimum step size when making chunks. By default this
            is 1. Can be set higher to avoid size 1 dimensions, which
            are problematic for linear regridding.

    :Returns:

        `list` or `dict`
            The list of m dimensional sections of the Field or the
            dictionary of m dimensional sections of the Data object.

    **Examples**

    >>> d = cf.Data(np.arange(120).reshape(2, 6, 10))
    >>> d
    <CF Data(2, 6, 10): [[[0, ..., 119]]]>
    >>> d.section([0, 1], min_step=2)
    {(None, None, 0): <CF Data(2, 6, 2): [[[0, ..., 111]]]>,
     (None, None, 2): <CF Data(2, 6, 2): [[[2, ..., 113]]]>,
     (None, None, 4): <CF Data(2, 6, 2): [[[4, ..., 115]]]>,
     (None, None, 6): <CF Data(2, 6, 2): [[[6, ..., 117]]]>,
     (None, None, 8): <CF Data(2, 6, 2): [[[8, ..., 119]]]>}

    """
    if stop is not None:
        raise DeprecationError(
            "The 'stop' keyword of cf._section() was deprecated at "
            "version 3.14.0 and is no longer available"
        )

    if chunks:
        raise DeprecationError(
            "The 'chunks' keyword of cf._section() was deprecated at "
            "version 3.14.0 and is no longer available. Consider using "
            "cf.Data.rechunk instead."
        )

    if axes is None:
        axes = list(range(x.ndim))

    axes = x.data._parse_axes(axes)

    ndim = x.ndim
    shape = x.shape

    # TODODASK: For v4.0.0, consider redefining the axes by removing
    #           the next line. I.e. the specified axes would be those
    #           that you want to be chopped, not those that you want
    #           to remain whole.
    axes = [i for i in range(ndim) if i not in axes]

    indices = [
        (
            (slice(j, j + min_step) for j in range(0, n, min_step))
            if i in axes
            else [slice(None)]
        )
        for i, n in enumerate(shape)
    ]

    keys = [
        range(0, n, min_step) if i in axes else [None]
        for i, n in enumerate(shape)
    ]

    out = {
        key: x[index] for key, index in zip(product(*keys), product(*indices))
    }
    return out


def _get_module_info(module, alternative_name=False, try_except=False):
    """Helper function for processing modules for cf.environment."""
    if try_except:
        module_name = None
        try:
            importlib.import_module(module)
            module_name = module
        except ImportError:
            if (
                alternative_name
            ):  # where a module has a different (e.g. old) name
                try:
                    importlib.import_module(alternative_name)
                    module_name = alternative_name
                except ImportError:
                    pass

        if not module_name:
            return ("not available", "")
    else:
        module_name = module

    return (
        importlib.import_module(module_name).__version__,
        importlib.util.find_spec(module_name).origin,
    )


def environment(display=True, paths=True):
    """Return the names and versions of the cf package and its
    dependencies.

    :Parameters:

        display: `bool`, optional
            If False then return the description of the environment as
            a string. By default the description is printed.

        paths: `bool`, optional
            If False then do not output the locations of each package.

            .. versionadded:: 3.0.6

    :Returns:

        `None` or `str`
            If *display* is True then the description of the
            environment is printed and `None` is returned. Otherwise
            the description is returned as a string.

    **Examples**

    >>> cf.environment()
    Platform: Linux-4.15.0-54-generic-x86_64-with-glibc2.10
    HDF5 library: 1.10.6
    netcdf library: 4.8.0
    udunits2 library: /home/username/anaconda3/envs/cf-env/lib/libudunits2.so.0
    esmpy/ESMF: 8.4.1 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/esmpy/__init__.py
    Python: 3.8.10 /home/username/anaconda3/envs/cf-env/bin/python
    dask: 2022.6.0 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/dask/__init__.py
    netCDF4: 1.5.6 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/netCDF4/__init__.py
    psutil: 5.9.0 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/psutil/__init__.py
    packaging: 21.3 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/packaging/__init__.py
    numpy: 1.22.2 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/numpy/__init__.py
    scipy: 1.10.0 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/scipy/__init__.py
    matplotlib: 3.4.3 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/matplotlib/__init__.py
    cftime: 1.6.0 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/cftime/__init__.py
    cfunits: 3.3.6 /home/username/cfunits/cfunits/__init__.py
    cfplot: 3.1.18 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/cfplot/__init__.py
    cfdm: 1.10.1.0 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/cfdm/__init__.py
    cf: 3.14.0 /home/username/anaconda3/envs/cf-env/lib/python3.8/site-packages/cf/__init__.py

    >>> cf.environment(paths=False)
    Platform: Linux-4.15.0-54-generic-x86_64-with-glibc2.10
    HDF5 library: 1.10.6
    netcdf library: 4.8.0
    udunits2 library: libudunits2.so.0
    esmpy/ESMF: 8.4.1
    Python: 3.8.10
    dask: 2022.6.0
    netCDF4: 1.5.6
    psutil: 5.9.0
    packaging: 21.3
    numpy: 1.22.2
    scipy: 1.10.0
    matplotlib: 3.4.3
    cftime: 1.6.0
    cfunits: 3.3.6
    cfplot: 3.1.18
    cfdm: 1.10.1.0
    cf: 3.14.0

    """
    dependency_version_paths_mapping = {
        # Platform first, then use an ordering to group libraries as follows...
        "Platform": (platform.platform(), ""),
        # Underlying C and Fortran based libraries first
        "HDF5 library": (netCDF4.__hdf5libversion__, ""),
        "netcdf library": (netCDF4.__netcdf4libversion__, ""),
        "udunits2 library": (ctypes.util.find_library("udunits2"), ""),
        "esmpy/ESMF": (
            _get_module_info("esmpy", alternative_name="ESMF", try_except=True)
        ),
        # Now Python itself
        "Python": (platform.python_version(), sys.executable),
        # Then Dask (cover first from below as it's important under-the-hood)
        "dask": _get_module_info("dask"),
        # Then Python libraries not related to CF
        "netCDF4": _get_module_info("netCDF4"),
        "psutil": _get_module_info("psutil"),
        "packaging": _get_module_info("packaging"),
        "numpy": _get_module_info("numpy"),
        "scipy": _get_module_info("scipy"),
        "matplotlib": _get_module_info("matplotlib", try_except=True),
        # Finally the CF related Python libraries, with the cf version last
        # as it is the most relevant (cfdm penultimate for similar reason)
        "cftime": _get_module_info("cftime"),
        "cfunits": _get_module_info("cfunits"),
        "cfplot": _get_module_info("cfplot", try_except=True),
        "cfdm": _get_module_info("cfdm"),
        "cf": (__version__, _os_path_abspath(__file__)),
    }
    string = "{0}: {1!s}"
    if paths:
        # Include path information, else exclude, when unpacking tuple
        string += " {2!s}"

    out = [
        string.format(dep, *info)
        for dep, info in dependency_version_paths_mapping.items()
    ]

    out = "\n".join(out)

    if display:
        print(out)  # pragma: no cover
    else:
        return out


def default_netCDF_fillvals():
    """Default data array fill values for each data type.

    :Returns:

        `dict`
            The fill values, keyed by `numpy` data type strings

    **Examples**

    >>> cf.default_netCDF_fillvals()
    {'S1': '\x00',
     'i1': -127,
     'u1': 255,
     'i2': -32767,
     'u2': 65535,
     'i4': -2147483647,
     'u4': 4294967295,
     'i8': -9223372036854775806,
     'u8': 18446744073709551614,
     'f4': 9.969209968386869e+36,
     'f8': 9.969209968386869e+36}

    """
    return netCDF4.default_fillvals


def size(a):
    """Return the number of elements.

    :Parameters:

        a: array_like
            Input data.

    :Returns:

        `int`
            The number of elements.

    **Examples**

    >>> cf.size(9)
    1
    >>> cf.size("foo")
    1
    >>> cf.size([9])
    1
    >>> cf.size((8, 9))
    2
    >>> import numpy as np
    >>> cf.size(np.arange(9))
    9
    >>> import dask.array as da
    >>> cf.size(da.arange(9))
    9

    """
    try:
        return a.size
    except AttributeError:
        return np.asanyarray(a).size


def unique_constructs(constructs, ignore_properties=None, copy=True):
    return cfdm.unique_constructs(
        constructs, ignore_properties=ignore_properties, copy=copy
    )


unique_constructs.__doc__ = cfdm.unique_constructs.__doc__.replace(
    "cfdm.", "cf."
)
unique_constructs.__doc__ = unique_constructs.__doc__.replace(
    "<Field:", "<CF Field:"
)
unique_constructs.__doc__ = unique_constructs.__doc__.replace(
    "<Domain:", "<CF Domain:"
)


def _DEPRECATION_ERROR(message="", version="3.0.0", removed_at="4.0.0"):
    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    raise DeprecationError(
        f"{message}. Deprecated at version {version}{removed_at}."
    )


def _DEPRECATION_ERROR_ARG(
    instance, method, arg, message="", version="3.0.0", removed_at="4.0.0"
):
    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    raise DeprecationError(
        f"Argument {arg!r} of method '{instance.__class__.__name__}.{method}' "
        f"has been deprecated at version {version} and is no longer available"
        f"{removed_at}. {message}"
    )


def _DEPRECATION_ERROR_FUNCTION_KWARGS(
    func,
    kwargs=None,
    message="",
    exact=False,
    traceback=False,
    info=False,
    version="3.0.0",
    removed_at="4.0.0",
):
    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    # Unsafe to set mutable '{}' as default in the func signature.
    if kwargs is None:  # distinguish from falsy '{}'
        kwargs = {}

    for kwarg, msg in KWARGS_MESSAGE_MAP.items():
        # This eval is safe as the kwarg is not a user input
        if kwarg in ("exact", "traceback") and eval(kwarg):
            kwargs = {kwarg: None}
            message = msg

    for key in kwargs.keys():
        raise DeprecationError(
            f"Keyword {key!r} of function '{func}' has been deprecated at "
            f"version {version} and is no longer available{removed_at}. "
            f"{message}"
        )


def _DEPRECATION_ERROR_FUNCTION_KWARG_VALUE(
    func,
    kwarg,
    value,
    message="",
    version=None,
    removed_at=None,
):
    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    raise DeprecationError(
        f"Value {value!r} of keyword {kwarg!r} of function {func!r} "
        f"has been deprecated at version {version} and is no longer "
        f"available{removed_at}. {message}"
    )


def _DEPRECATION_ERROR_FUNCTION_KWARG(
    func,
    kwarg=None,
    message="",
    version=None,
    removed_at=None,
):
    if version is None:
        raise ValueError("Must provide deprecation version, e.g. '3.14.0'")

    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    raise DeprecationError(
        f"Keyword {kwarg!r} of function {func} has been deprecated "
        f"at version {version} and is no longer available{removed_at}. "
        f"{message}"
    )


def _DEPRECATION_ERROR_KWARGS(
    instance,
    method,
    kwargs=None,
    message="",
    i=False,
    traceback=False,
    axes=False,
    exact=False,
    relaxed_identity=False,
    info=False,
    version=None,
    removed_at=None,
):
    if version is None:
        raise ValueError("Must provide deprecation version, e.g. '3.14.0'")

    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    # Unsafe to set mutable '{}' as default in the func signature.
    if kwargs is None:  # distinguish from falsy '{}'
        kwargs = {}

    for kwarg, msg in KWARGS_MESSAGE_MAP.items():
        if eval(kwarg):  # safe as this is not a kwarg input by the user
            kwargs = {kwarg: None}
            message = msg

    for key in kwargs.keys():
        raise DeprecationError(
            f"Keyword {key!r} of method "
            f"'{instance.__class__.__name__}.{method}' has been deprecated "
            f"at version {version} and is no longer available{removed_at}. "
            f"{message}"
        )


def _DEPRECATION_ERROR_KWARG_VALUE(
    instance,
    method,
    kwarg,
    value,
    message="",
    version="3.0.0",
    removed_at="4.0.0",
):
    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    raise DeprecationError(
        f"Value {value!r} of keyword {kwarg!r} of method "
        f"'{instance.__class__.__name__}.{method}' has been deprecated at "
        f"version {version} and is no longer available{removed_at}. {message}"
    )


def _DEPRECATION_ERROR_METHOD(
    instance, method, message="", version="3.0.0", removed_at=""
):
    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    raise DeprecationError(
        f"{instance.__class__.__name__} method {method!r} has been deprecated "
        f"at version {version} and is no longer available{removed_at}. "
        f"{message}"
    )


def _DEPRECATION_ERROR_ATTRIBUTE(
    instance, attribute, message="", version="3.0.0", removed_at=""
):
    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    raise DeprecationError(
        f"{instance.__class__.__name__} attribute {attribute!r} has been "
        f"deprecated at version {version}{removed_at}. {message}"
    )


def _DEPRECATION_ERROR_FUNCTION(
    func, message="", version="3.0.0", removed_at="4.0.0"
):
    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    raise DeprecationError(
        f"Function {func!r} has been deprecated at version {version} and is "
        f"no longer available{removed_at}. {message}"
    )


def _DEPRECATION_ERROR_CLASS(
    cls, message="", version="3.0.0", removed_at="4.0.0"
):
    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    raise DeprecationError(
        f"Class {cls!r} has been deprecated at version {version} and is no "
        f"longer available{removed_at}. {message}"
    )


def _DEPRECATION_WARNING_METHOD(
    instance, method, message="", new=None, version="3.0.0", removed_at="4.0.0"
):
    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    warnings.warn(
        f"{instance.__class__.__name__} method {method!r} has been deprecated "
        f"at version {version}{removed_at}. {message}",
        DeprecationWarning,
    )


def _DEPRECATION_ERROR_DICT(message="", version="3.0.0", removed_at="4.0.0"):
    if removed_at:
        removed_at = f"and will be removed at version {removed_at}"

    raise DeprecationError(
        "Use of a 'dict' to identify constructs has been deprecated at "
        f"version {version} and is no longer available and will be removed "
        f"at version {removed_at}. {message}"
    )


def _DEPRECATION_ERROR_SEQUENCE(instance, version="3.0.0", removed_at="4.0.0"):
    if removed_at:
        removed_at = f" and will be removed at version {removed_at}"

    raise DeprecationError(
        f"Use of a {instance.__class__.__name__!r} to identify constructs "
        f"has been deprecated at version {version} and is no longer available"
        f"{removed_at}. Use the * operator to unpack the arguments instead."
    )


# --------------------------------------------------------------------
# Deprecated functions
# --------------------------------------------------------------------
def default_fillvals():
    """Default data array fill values for each data type.

    Deprecated at version 3.0.2 and is no longer available. Use function
    `cf.default_netCDF_fillvals` instead.

    """
    _DEPRECATION_ERROR_FUNCTION(
        "default_fillvals",
        "Use function 'cf.default_netCDF_fillvals' instead.",
        version="3.0.2",
        removed_at="4.0.0",
    )  # pragma: no cover


def set_equals(
    x, y, rtol=None, atol=None, ignore_fill_value=False, traceback=False
):
    """Deprecated at version 3.0.0."""
    _DEPRECATION_ERROR_FUNCTION(
        "cf.set_equals", version="3.0.0", removed_at="4.0.0"
    )  # pragma: no cover
