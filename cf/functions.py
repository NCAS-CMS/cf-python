import csv
import logging
import os
import platform
import re
import resource
import ctypes.util
# import cPickle
import netCDF4
import warnings

import psutil

import cftime

from numpy import __file__          as _numpy__file__
from numpy import __version__       as _numpy__version__
from numpy import all               as _numpy_all
from numpy import allclose          as _x_numpy_allclose
from numpy import array             as _numpy_array
from numpy import ascontiguousarray as _numpy_ascontiguousarray
from numpy import integer           as _numpy_integer
from numpy import isclose           as _x_numpy_isclose
from numpy import ndim              as _numpy_ndim
from numpy import shape             as _numpy_shape
from numpy import sign              as _numpy_sign
from numpy import size              as _numpy_size
from numpy import take              as _numpy_take
from numpy import tile              as _numpy_tile
from numpy import where             as _numpy_where

from numpy.ma import all       as _numpy_ma_all
from numpy.ma import allclose  as _numpy_ma_allclose
from numpy.ma import is_masked as _numpy_ma_is_masked
from numpy.ma import isMA      as _numpy_ma_isMA
from numpy.ma import masked    as _numpy_ma_masked

from collections import Iterable
from hashlib     import md5 as hashlib_md5
from marshal     import dumps as marshal_dumps
from math        import ceil as math_ceil
from os          import getpid, listdir, mkdir
from os.path     import abspath      as _os_path_abspath
from os.path     import expanduser   as _os_path_expanduser
from os.path     import expandvars   as _os_path_expandvars
from os.path     import dirname      as _os_path_dirname
from os.path     import join         as _os_path_join
from os.path     import relpath      as _os_path_relpath
from psutil      import virtual_memory, Process
from sys         import executable as _sys_executable
import urllib.parse

import cfdm
import cfunits

from .          import __version__, __file__
from .constants import CONSTANTS, _file_to_fh, _stash2standard_name

from . import mpi_on
from . import mpi_size


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
    )
}

# Are we running on GNU/Linux?
_linux = (platform.system() == 'Linux')

if _linux:
    # ----------------------------------------------------------------
    # GNU/LINUX
    # ----------------------------------------------------------------
    # Opening /proc/meminfo once per PE here rather than in
    # _free_memory each time it is called works with MPI on
    # Debian-based systems, which otherwise throw an error that there
    # is no such file or directory when run on multiple PEs.
    # ----------------------------------------------------------------
    _meminfo_fields = set(('SReclaimable:', 'Cached:', 'Buffers:', 'MemFree:'))
    _meminfo_file = open('/proc/meminfo', 'r', 1)

    def _free_memory():
        '''The amount of available physical memory on GNU/Linux.

    This amount includes any memory which is still allocated but is no
    longer required.

    :Returns:

        `float`
            The amount of available physical memory in bytes.

    **Examples:**

    >>> _free_memory()
    96496240.0

        '''
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
        # --- End: for

        free_bytes = free_KiB * 1024

        return free_bytes


else:
    # ----------------------------------------------------------------
    # NOT GNU/LINUX
    # ----------------------------------------------------------------
    def _free_memory():
        '''The amount of available physical memory.

    :Returns:

        `float`
            The amount of available physical memory in bytes.

    **Examples:**

    >>> _free_memory()
    96496240.0

        '''
        return float(virtual_memory().available)


# --- End: if

def FREE_MEMORY():
    '''The available physical memory.

    :Returns:

        `float`
            The amount of free memory in bytes.

    **Examples:**

    >>> import numpy
    >>> print('Free memory =', cf.FREE_MEMORY()/2**30, 'GiB')
    Free memory = 88.2728042603 GiB
    >>> a = numpy.arange(10**9)
    >>> print('Free memory =', cf.FREE_MEMORY()/2**30, 'GiB')
    Free memory = 80.8082618713 GiB
    >>> del a
    >>> print('Free memory =', cf.FREE_MEMORY()/2**30, 'GiB')
    Free memory = 88.2727928162 GiB

    '''
    return _free_memory()


def _WORKSPACE_FACTOR_1():
    '''The value of workspace factor 1 used in calculating the upper limit
    to the chunksize given the free memory factor.

    :Returns:

        `float`
            workspace factor 1

    '''
    return CONSTANTS['WORKSPACE_FACTOR_1']


def _WORKSPACE_FACTOR_2():
    '''The value of workspace factor 2 used in calculating the upper limit
    to the chunksize given the free memory factor.

    :Returns:

        `float`
            workspace factor 2

    '''
    return CONSTANTS['WORKSPACE_FACTOR_2']


def FREE_MEMORY_FACTOR(*args):
    '''Set the fraction of memory kept free as a temporary
    workspace. Users should set the free memory factor through
    cf.SET_PERFORMANCE so that the upper limit to the chunksize is
    recalculated appropriately. The free memory factor must be a
    sensible value between zero and one. If no arguments are passed
    the existing free memory factor is returned.

    :Parameters:

        free_memory_factor: `float`
            The fraction of memory kept free as a temporary workspace.

    :Returns:

        `float`
             The previous value of the free memory factor.

    '''
    old = CONSTANTS['FREE_MEMORY_FACTOR']
    if args:
        free_memory_factor = float(args[0])
        if free_memory_factor <= 0.0 or free_memory_factor >= 1.0:
            raise ValueError(
                'Free memory factor must be between 0.0 and 1.0 not inclusive')

        CONSTANTS['FREE_MEMORY_FACTOR'] = free_memory_factor
        CONSTANTS['FM_THRESHOLD'] = free_memory_factor * TOTAL_MEMORY()

    return old


# --------------------------------------------------------------------
# Functions inherited from cfdm
# --------------------------------------------------------------------
ATOL = cfdm.ATOL
RTOL = cfdm.RTOL
CF = cfdm.CF

_disable_logging = cfdm._disable_logging
# We can inherit the generic logic for the cf-python LOG_LEVEL() function
# as contained in _log_level, but can't inherit the user-facing LOG_LEVEL()
# from cfdm as it operates on cfdm's CONSTANTS dict. Define cf-python's own.
# This also means the LOG_LEVEL dostrings are independent which is important
# for providing module-specific documentation links and directives, etc.
_log_level = cfdm._log_level
_reset_log_emergence_level = cfdm._reset_log_emergence_level


def LOG_LEVEL(*log_level):
    '''The minimal level of seriousness of log messages which are shown.

    This can be adjusted to filter out potentially-useful log messages
    generated by ``cf`` at runtime, such that any messages marked as
    having a severity below the level set will not be reported.

    For example, when set to ``'WARNING'`` (or equivalently ``1``),
    all messages categorised as ``'DEBUG'`` or ``'INFO'`` will be
    supressed, and only warnings will emerge.

    See https://ncas-cms.github.io/cf-python/tutorial.html#logging for
    a detailed breakdown on the levels and configuration possibilities.

    The default level is ``'WARNING'`` (``1``).

    .. versionadded:: 3.5.0

    :Parameters:

        log_level: `str` or `int`, optional
            The new value of the minimal log severity level. This can
            be specified either as a string equal (ignoring case) to
            the named set of log levels or identifier 'DISABLE', or an
            integer code corresponding to each of these, namely:

            * ``'DISABLE'`` (``0``);
            * ``'WARNING'`` (``1``);
            * ``'INFO'`` (``2``);
            * ``'DETAIL'`` (``3``);
            * ``'DEBUG'`` (``-1``).

    :Returns:

        `str`
            The value prior to the change, or the current value if no
            new value was specified (or if one was specified but was
            not valid). Note the string name, rather than the
            equivalent integer, will always be returned.

    **Examples:**

    >>> LOG_LEVEL()  # get the current value
    'WARNING'
    >>> LOG_LEVEL('INFO')  # change the value to 'INFO'
    'WARNING'
    >>> LOG_LEVEL()
    'INFO'
    >>> LOG_LEVEL(0)  # set to 'DISABLE' via corresponding integer
    'INFO'
    >>> LOG_LEVEL()
    'DISABLE'

    '''
    return _log_level(CONSTANTS, log_level)


def CHUNKSIZE(*args):
    '''Set the chunksize used by LAMA for partitioning the data
    array. This must be smaller than an upper limit determined by the
    free memory factor, which is the fraction of memory kept free as a
    temporary workspace, otherwise an error is raised. If called with
    None as the argument then the chunksize is set to its upper
    limit. If called without any arguments the existing chunksize is
    returned.

    The upper limit to the chunksize is given by:

    .. math:: upper\_chunksize = \dfrac{f \cdot total\_memory}{mpi\_size
                                 \cdot w_1 + w_2}

    where :math:`f` is the *free memory factor* and :math:`w_1` and
    :math:`w_2` the *workspace factors* *1* and *2* respectively.

    :Parameters:

        chunksize: `float`, optional
            The chunksize in bytes.

    :Returns:

        `float`
            The previous value of the chunksize in bytes.

    '''
    old = CONSTANTS['CHUNKSIZE']
    if args:
        upper_chunksize = ((FREE_MEMORY_FACTOR() * MIN_TOTAL_MEMORY())
                           / ((mpi_size * _WORKSPACE_FACTOR_1()) +
                              _WORKSPACE_FACTOR_2()))
        if args[0] is None:
            CONSTANTS['CHUNKSIZE'] = upper_chunksize
        else:
            chunksize = float(args[0])
            if chunksize > upper_chunksize and mpi_size > 1:
                raise ValueError(
                    'Specified chunk size is too large for given free memory '
                    'factor'
                )
            elif chunksize <= 0:
                raise ValueError('Chunk size must be positive')

            CONSTANTS['CHUNKSIZE'] = chunksize
    # --- End: if

    return old


def FM_THRESHOLD():
    '''The amount of memory which is kept free as a temporary work space.

    :Returns:

        `float`
            The amount of memory in bytes.

    **Examples:**

    >>> cf.FM_THRESHOLD()
    10000000000.0

    '''
    return CONSTANTS['FM_THRESHOLD']


def SET_PERFORMANCE(chunksize=None, free_memory_factor=None):
    '''Tune performance of parallelisation by setting chunksize and free
    memory factor. By just providing the chunksize it can be changed
    to a smaller value than an upper limit, which is determined by the
    existing free memory factor. If just the free memory factor is
    provided then the chunksize is set to the corresponding upper
    limit. Note that the free memory factor is the fraction of memory
    kept free as a temporary workspace and must be a sensible value
    between zero and one. If both arguments are provided then the free
    memory factor is changed first and then the chunksize is set
    provided it is consistent with the new free memory value. If any
    of the arguments is invalid then an error is raised and no
    parameters are changed. When called with no aruments the existing
    values of the parameters are returned in a tuple.

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

    '''
    old = CHUNKSIZE(), FREE_MEMORY_FACTOR()
    if free_memory_factor is None:
        if chunksize is not None:
            CHUNKSIZE(chunksize)
    else:
        FREE_MEMORY_FACTOR(free_memory_factor)
        try:
            CHUNKSIZE(chunksize)
        except ValueError:
            FREE_MEMORY_FACTOR(old[1])
            raise
    # --- End: if

    return old


def MIN_TOTAL_MEMORY():
    '''The minumum total memory across nodes.
    '''
    return CONSTANTS['MIN_TOTAL_MEMORY']
# --- End: def


def TOTAL_MEMORY():
    '''TODO
    '''
    return CONSTANTS['TOTAL_MEMORY']


def TEMPDIR(*arg):
    '''The directory for internally generated temporary files.

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

    **Examples:**

    >>> cf.TEMPDIR()
    '/tmp'
    >>> old = cf.TEMPDIR('/home/me/tmp')
    >>> cf.TEMPDIR(old)
    '/home/me/tmp'
    >>> cf.TEMPDIR()
    '/tmp'

    '''
    old = CONSTANTS['TEMPDIR']
    if arg:
        tempdir = _os_path_expanduser(_os_path_expandvars(arg[0]))

        # Create the directory if it does not exist.
        try:
            mkdir(tempdir)
        except OSError:
            pass

        CONSTANTS['TEMPDIR'] = tempdir

    return old


def OF_FRACTION(*arg):
    '''The amount of concurrently open files above which files containing
    data arrays may be automatically closed.

    The amount is expressed as a fraction of the maximum possible
    number of concurrently open files.

    Note that closed files will be automatically reopened if
    subsequently needed by a variable to access its data array.

    .. seealso:: `cf.close_files`, `cf.close_one_file`,
                 `cf.open_files`, `cf.open_files_threshold_exceeded`

    :Parameters:

        arg: `float`, optional
            The new fraction (between 0.0 and 1.0). The default is to
            not change the current behaviour.

    :Returns:

        `float`
            The value prior to the change, or the current value if no
            new value was specified.

    **Examples:**

    >>> cf.OF_FRACTION()
    0.5
    >>> old = cf.OF_FRACTION(0.33)
    >>> cf.OF_FRACTION(old)
    0.33
    >>> cf.OF_FRACTION()
    0.5

    The fraction may be translated to an actual number of files as
    follows:

    >>> old = cf.OF_FRACTION(0.75)
    >>> import resource
    >>> max_open_files = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    >>> threshold = int(max_open_files * cf.OF_FRACTION())
    >>> max_open_files, threshold
    (1024, 768)

    '''
    old = CONSTANTS['OF_FRACTION']
    if arg:
        CONSTANTS['OF_FRACTION'] = arg[0]

    return old


def REGRID_LOGGING(*arg):
    '''Whether or not to enable ESMPy logging.

    If it is logging is performed after every call to ESMPy.

    :Parameters:

        arg: `bool`, optional
            The new value (either True to enable logging or False to
            disable it).  The default is to not change the current
            behaviour.

    :Returns:

        `bool`
            The value prior to the change, or the current value if no
            new value was specified.

    **Examples:**

    >>> cf.REGRID_LOGGING()
    False
    >>> cf.REGRID_LOGGING(True)
    False
    >>> cf.REGRID_LOGGING()
    True

    '''
    old = CONSTANTS['REGRID_LOGGING']
    if arg:
        CONSTANTS['REGRID_LOGGING'] = bool(arg[0])

    return old


def COLLAPSE_PARALLEL_MODE(*arg):
    '''Which mode to use when collapse is run in parallel. There are three
    possible modes:

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

        arg: `int`, optional
            The new value (0, 1 or 2).

    :Returns:

        `int`
            The value prior to the change, or the current value if no
            new value was specified.

    **Examples:**

    >>> cf.COLLAPSE_PARALLEL_MODE()
    0
    >>> cf.COLLAPSE_PARALLEL_MODE(1)
    0
    >>> cf.COLLAPSE_PARALLEL_MODE()
    1

    '''
    old = CONSTANTS['COLLAPSE_PARALLEL_MODE']
    if arg:
        if arg[0] not in (0, 1, 2):
            raise ValueError('Invalid collapse parallel mode')

        CONSTANTS['COLLAPSE_PARALLEL_MODE'] = arg[0]

    return old


def RELAXED_IDENTITIES(*arg):
    '''Use 'relaxed' mode when getting a construct identity.

    If set to True, sets ``relaxed=True`` as the default in calls to a
    construct's `identity` method (e.g. `cf.Field.identity`).

    This is used by construct arithmetic and field construct
    aggregation.

    :Parameters:

        arg: `bool`, optional

    :Returns:

        `bool`
            The value prior to the change, or the current value if no
            new value was specified.

    **Examples:**

    >>> org = cf.RELAXED_IDENTITIES()
    >>> org
    False
    >>> cf.RELAXED_IDENTITIES(True)
    False
    >>> cf.RELAXED_IDENTITIES()
    True
    >>> cf.RELAXED_IDENTITIES(org)
    True
    >>> cf.RELAXED_IDENTITIES()
    False

    '''
    old = CONSTANTS['RELAXED_IDENTITIES']
    if arg:
        CONSTANTS['RELAXED_IDENTITIES'] = bool(arg[0])

    return old


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
#     **Examples:**
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
    '''Print a description of an object.

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

    **Examples:**

    >>> x = 3.14159
    >>> cf.dump(x)
    3.14159

    >>> f
    <CF Field: rainfall_rate(latitude(10), longitude(20)) kg m2 s-1>
    >>> cf.dump(f)
    >>> cf.dump(f, complete=True)

    '''
    if hasattr(x, 'dump') and callable(x.dump):
        print(x.dump(**kwargs))
    else:
        print(x)


_max_number_of_open_files = resource.getrlimit(resource.RLIMIT_NOFILE)[0]

if _linux:
    # ----------------------------------------------------------------
    # GNU/LINUX
    # ----------------------------------------------------------------

    # Directory containing a symbolic link for each file opened by the
    # current python session
    _fd_dir = '/proc/'+str(getpid())+'/fd'

    def open_files_threshold_exceeded():
        '''Return True if the total number of open files is greater than the
    current threshold. GNU/LINUX.

    The threshold is defined as a fraction of the maximum possible number
    of concurrently open files (an operating system dependent amount). The
    fraction is retrieved and set with the `OF_FRACTION` function.

    .. seealso:: `cf.close_files`, `cf.close_one_file`,
                 `cf.open_files`

    :Returns:

        `bool`
            Whether or not the number of open files exceeds the
            threshold.

    **Examples:**

    In this example, the number of open files is 75% of the maximum
    possible number of concurrently open files:

    >>> cf.OF_FRACTION()
    0.5
    >>> cf.open_files_threshold_exceeded()
    True
    >>> cf.OF_FRACTION(0.9)
    >>> cf.open_files_threshold_exceeded()
    False

        '''
        return (len(listdir(_fd_dir)) >
                _max_number_of_open_files * OF_FRACTION())


else:
    # ----------------------------------------------------------------
    # NOT GNU/LINUX
    # ----------------------------------------------------------------
    _process = Process(getpid())

    def open_files_threshold_exceeded():
        '''Return True if the total number of open files is greater than the
    current threshold.

    The threshold is defined as a fraction of the maximum possible number
    of concurrently open files (an operating system dependent amount). The
    fraction is retrieved and set with the `OF_FRACTION` function.

    .. seealso:: `cf.close_files`, `cf.close_one_file`,
                 `cf.open_files`

    :Returns:

        `bool`
            Whether or not the number of open files exceeds the
            threshold.

    **Examples:**

    In this example, the number of open files is 75% of the maximum
    possible number of concurrently open files:

    >>> cf.OF_FRACTION()
    0.5
    >>> cf.open_files_threshold_exceeded()
    True
    >>> cf.OF_FRACTION(0.9)
    >>> cf.open_files_threshold_exceeded()
    False

        '''
        return (len(_process.open_files()) >
                _max_number_of_open_files * OF_FRACTION())


# --- End: if

def close_files(file_format=None):
    '''Close open files containing sub-arrays of data arrays.

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

    **Examples:**

    >>> cf.close_files()
    >>> cf.close_files('netCDF')
    >>> cf.close_files('PP')

    '''
    if file_format is not None:
        if file_format in _file_to_fh:
            for fh in _file_to_fh[file_format].values():
                fh.close()

            _file_to_fh[file_format].clear()
    else:
        for file_format, value in _file_to_fh.items():
            for fh in value.values():
                fh.close()

            _file_to_fh[file_format].clear()
    # --- End: if


def close_one_file(file_format=None):
    '''Close an arbitrary open file containing a sub-array of a data
    array.

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

    **Examples:**

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

    '''
    if file_format is not None:
        if file_format in _file_to_fh and _file_to_fh[file_format]:
            filename, fh = next(iter(_file_to_fh[file_format].items()))
            fh.close()
            del _file_to_fh[file_format][filename]
    else:
        for values in _file_to_fh.values():
            if not values:
                continue

            filename, fh = next(iter(values.items()))
            fh.close()
            del values[filename]


def open_files(file_format=None):
    '''Return the open files containing sub-arrays of master data arrays.

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

    **Examples:**

    >>> cf.open_files()
    {'netCDF': {'file1.nc': <netCDF4.Dataset at 0x187b6d0>}}
    >>> cf.open_files('netCDF')
    {'file1.nc': <netCDF4.Dataset at 0x187b6d0>}
    >>> cf.open_files('PP')
    {}

    '''
    if file_format is not None:
        if file_format in _file_to_fh:
            return _file_to_fh[file_format].copy()
        else:
            return {}
    else:
        out = {}
        for file_format, values in _file_to_fh.items():
            out[file_format] = values.copy()

        return out


def ufunc(name, x, *args, **kwargs):
    '''The variable must have a `!copy` method and a method called
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

    '''
    x = x.copy()
    getattr(x, name)(*args, **kwargs)
    return x


def _numpy_allclose(a, b, rtol=None, atol=None, verbose=None):
    '''Returns True if two broadcastable arrays have equal values to
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
            default the value returned by the `ATOL` function is used.

        rtol : float, optional
            The relative tolerance for all numerical comparisons, By
            default the value returned by the `RTOL` function is used.

    :Returns:

        `bool`
            Returns True if the arrays are equal, otherwise False.

    **Examples:**

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

    '''
    # TODO: we want to use @_manage_log_level_via_verbosity on this function
    # but we cannot, since importing it to this module would lead to a
    # circular import dependency with the decorators module. Tentative plan
    # is to move the function elsewhere. For now, it is not 'loggified'.

    # THIS IS WHERE SOME NUMPY FUTURE WARNINGS ARE COMING FROM

    a_is_masked = _numpy_ma_isMA(a)
    b_is_masked = _numpy_ma_isMA(b)

    if not (a_is_masked or b_is_masked):
        try:
            return _x_numpy_allclose(a, b, rtol=rtol, atol=atol)
        except (IndexError, NotImplementedError, TypeError):
            return _numpy_all(a == b)
    else:
        if a_is_masked and b_is_masked:
            if (a.mask != b.mask).any():
                if verbose:
                    print('Different masks (A)')

                return False
        else:
            if _numpy_ma_is_masked(a) or _numpy_ma_is_masked(b):
                if verbose:
                    print('Different masks (B)')

                return False

#            if verbose:
#                print('Different masks 4')
#
#            return False

        try:
            return _numpy_ma_allclose(a, b, rtol=rtol, atol=atol)
        except (IndexError, NotImplementedError, TypeError):
            out = _numpy_ma_all(a == b)
            if out is _numpy_ma_masked:
                return True
            else:
                return out


def _numpy_isclose(a, b, rtol=None, atol=None):
    '''Returns a boolean array where two broadcastable arrays are
    element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers. The
    relative difference (``rtol * abs(b)``) and the absolute difference
    ``atol`` are added together to compare against the absolute difference
    between ``a`` and ``b``.

    :Parameters:

        a, b: array_like
            Input arrays to compare.

        atol: `float`, optional
            The absolute tolerance for all numerical comparisons, By
            default the value returned by the `ATOL` function is used.

        rtol: `float`, optional
            The relative tolerance for all numerical comparisons, By
            default the value returned by the `RTOL` function is used.

    :Returns:

        `numpy.ndarray`

    '''
    try:
        return _x_numpy_isclose(a, b, rtol=rtol, atol=atol)
    except (IndexError, NotImplementedError, TypeError):
        return a == b


def parse_indices(shape, indices, cyclic=False, reverse=False,
                  envelope=False, mask=False):
    '''TODO

    :Parameters:

        shape: sequence of `ints`

        indices: `tuple` (not a `list`!)

    :Returns:

        `list` [, `dict`]

    **Examples:**

    >>> cf.parse_indices((5, 8), ([1, 2, 4, 6],))
    [array([1, 2, 4, 6]), slice(0, 8, 1)]
    >>> cf.parse_indices((5, 8), ([2, 4, 6],))
    [slice(2, 7, 2), slice(0, 8, 1)]

    '''
    parsed_indices = []
    roll = {}
    flip = []
    compressed_indices = []
    mask_indices = []

    if not isinstance(indices, tuple):
        indices = (indices,)

    if mask and indices:
        arg0 = indices[0]
        if isinstance(arg0, str) and arg0 == 'mask':
            mask_indices = indices[1]
            indices = indices[2:]
    # --- End: if

    # Initialize the list of parsed indices as the input indices with any
    # Ellipsis objects expanded
    length = len(indices)
    n = len(shape)
    ndim = n
    for index in indices:
        if index is Ellipsis:
            m = n-length+1
            parsed_indices.extend([slice(None)] * m)
            n -= m
        else:
            parsed_indices.append(index)
            n -= 1

        length -= 1

    len_parsed_indices = len(parsed_indices)

    if ndim and len_parsed_indices > ndim:
        raise IndexError("Invalid indices {} for array with shape {}".format(
            parsed_indices, shape))

    if len_parsed_indices < ndim:
        parsed_indices.extend([slice(None)]*(ndim-len_parsed_indices))

    if not ndim and parsed_indices:
        # # If data is scalar then allow it to be indexed with an
        # # equivalent to [0]
        # if (len_parsed_indices == 1 and
        #     parsed_indices[0] in (0,
        #                           -1,
        #                           slice(0, 1),
        #                           slice(-1, None, -1),
        #                           slice(None, None, None))):
        #     parsed_indices = []
        # else:
        raise IndexError(
            "Scalar array can only be indexed with () or Ellipsis")

    # --- End: if

    for i, (index, size) in enumerate(zip(parsed_indices, shape)):
        is_slice = False
        if isinstance(index, slice):
            # --------------------------------------------------------
            # Index is a slice
            # --------------------------------------------------------
            is_slice = True
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
                    start = size-start
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
            # --- End: if

            if step > 0 and -size <= start < 0 and 0 <= stop <= size+start:
                # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                # -1:0:1  => [9]
                # -1:1:1  => [9, 0]
                # -1:3:1  => [9, 0, 1, 2]
                # -1:9:1  => [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
                # -4:0:1  => [6, 7, 8, 9]
                # -4:1:1  => [6, 7, 8, 9, 0]
                # -4:3:1  => [6, 7, 8, 9, 0, 1, 2]
                # -4:6:1  => [6, 7, 8, 9, 0, 1, 2, 3, 4, 5]
                # -9:0:1  => [1, 2, 3, 4, 5, 6, 7, 8, 9]
                # -9:1:1  => [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                # -10:0:1 => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                if cyclic:
                    index = slice(0, stop-start, step)
                    roll[i] = -start
                else:
                    index = slice(start, stop, step)

            elif step < 0 and 0 <= start < size and start-size <= stop < 0:
                # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                # 0:-4:-1  => [0, 9, 8, 7]
                # 6:-1:-1  => [6, 5, 4, 3, 2, 1, 0]
                # 6:-2:-1  => [6, 5, 4, 3, 2, 1, 0, 9]
                # 6:-4:-1  => [6, 5, 4, 3, 2, 1, 0, 9, 8, 7]
                # 0:-2:-1  => [0, 9]
                # 0:-10:-1 => [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]
                if cyclic:
                    index = slice(start-stop-1, None, step)
                    roll[i] = -1 - stop
                else:
                    index = slice(start, stop, step)

            else:
                start, stop, step = index.indices(size)
                if (start == stop or
                        (start < stop and step < 0) or
                        (start > stop and step > 0)):
                    raise IndexError(
                        "Invalid indices dimension with size {}: {}".format(
                            size, index))
                if step < 0 and stop < 0:
                    stop = None
                index = slice(start, stop, step)

        elif isinstance(index, (int, _numpy_integer)):
            # --------------------------------------------------------
            # Index is an integer
            # --------------------------------------------------------
            if index < 0:
                index += size

            index = slice(index, index+1, 1)
            is_slice = True
        else:
            convert2positve = True
            if (getattr(getattr(index, 'dtype', None), 'kind', None) == 'b' or
                    isinstance(index[0], bool)):
                # ----------------------------------------------------
                # Index is a sequence of booleans
                # ----------------------------------------------------
                # Convert booleans to non-negative integers. We're
                # assuming that anything with a dtype attribute also
                # has a size attribute.
                if _numpy_size(index) != size:
                    raise IndexError(
                        "Incorrect number ({}) of boolean indices for "
                        "dimension with size {}: {}".format(
                            _numpy_size(index), size, index)
                    )

                index = _numpy_where(index)[0]
                convert2positve = False

            if not _numpy_ndim(index):
                if index < 0:
                    index += size

                index = slice(index, index+1, 1)
                is_slice = True
            else:
                len_index = len(index)
                if len_index == 1:
                    index = index[0]
                    if index < 0:
                        index += size

                    index = slice(index, index+1, 1)
                    is_slice = True
                elif len_index:
                    if convert2positve:
                        # Convert to non-negative integer numpy array
                        index = _numpy_array(index)
                        index = _numpy_where(index < 0, index+size, index)

                    steps = index[1:] - index[:-1]
                    step = steps[0]
                    if step and not (steps - step).any():
                        # Replace the numpy array index with a slice
                        if step > 0:
                            start, stop = index[0], index[-1]+1
                        elif step < 0:
                            start, stop = index[0], index[-1]-1

                        if stop < 0:
                            stop = None

                        index = slice(start, stop, step)
                        is_slice = True
                    else:
                        if ((step > 0 and (steps <= 0).any()) or
                                (step < 0 and (steps >= 0).any()) or
                                not step):
                            raise ValueError(
                                "Bad index (not strictly monotonic): "
                                "{}".format(index)
                            )

                        if reverse and step < 0:
                            # The array is strictly monotoniticall
                            # decreasing, so reverse it so that it's
                            # strictly monotonically increasing.  Make
                            # a note that this dimension will need
                            # flipping later
                            index = index[::-1]
                            flip.append(i)
                            step = -step

                        if envelope:
                            # Create an envelope slice for a parsed
                            # index of a numpy array of integers
                            compressed_indices.append(index)

                            step = _numpy_sign(step)
                            if step > 0:
                                stop = index[-1] + 1
                            else:
                                stop = index[-1] - 1
                                if stop < 0:
                                    stop = None

                            index = slice(index[0], stop, step)
                            is_slice = True
                else:
                    raise IndexError(
                        "Invalid indices {} for array with shape {}".format(
                            parsed_indices, shape))
            # --- End: if
        # --- End: if

        if is_slice:
            if reverse and index.step < 0:
                # If the slice step is negative, then transform
                # the original slice to a new slice with a
                # positive step such that the result of the new
                # slice is the reverse of the result of the
                # original slice.
                #
                # For example, if the original slice is
                # slice(6,0,-2) then the new slice will be
                # slice(2,7,2):
                #
                # >>> a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                # >>> a[slice(6, 0, -2)]
                # [6, 4, 2]
                # >>> a[slice(2, 7, 2)]
                # [2, 4, 6]
                # a[slice(6, 0, -2)] == list(reversed(a[slice(2, 7, 2)]))
                # True
                start, stop, step = index.indices(size)
                step *= -1
                div, mod = divmod(start-stop-1, step)
                div_step = div*step
                start -= div_step
                stop = start + div_step + 1

                index = slice(start, stop, step)
                flip.append(i)
            # --- End: if

            # If step is greater than one then make sure that
            # index.stop isn't bigger than it needs to be
            if cyclic and index.step > 1:
                start, stop, step = index.indices(size)
                div, mod = divmod(stop-start-1, step)
                stop = start + div*step + 1
                index = slice(start, stop, step)
            # --- End: if

            #
            if envelope:
                # Create an envelope slice for a parsed
                # index of a numpy array of integers
                compressed_indices.append(index)
                index = slice(
                    start, stop, (1 if reverse else _numpy_sign(step)))
        # --- End: if

        parsed_indices[i] = index
    # --- End: for

    if not (cyclic or reverse or envelope or mask):
        return parsed_indices

    out = [parsed_indices]

    if cyclic:
        out.append(roll)

    if reverse:
        out.append(flip)

    if envelope:
        out.append(compressed_indices)

    if mask:
        out.append(mask_indices)

    return out


def get_subspace(array, indices):
    '''TODO

    Subset the input numpy array with the given indices. Indexing is
    similar to that of a numpy array. The differences to numpy array
    indexing are:

    1. An integer index i takes the i-th element but does not reduce
       the rank of the output array by one.

    2. When more than one dimension's slice is a 1-d boolean array or
       1-d sequence of integers then these indices work independently
       along each dimension (similar to the way vector subscripts work
       in Fortran).

    indices must contain an index for each dimension of the input array.

    :Parameters:

        array: `numpy.ndarray`

        indices: `list`

    '''
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
        if _numpy_ma_isMA(array):
            take = _numpy_ma_take
        else:
            take = _numpy_take

        indices = indices[:]
        for axis in gg:
            array = take(array, indices[axis], axis=axis)
            indices[axis] = slice(None)

        if len_gg < len(indices):
            array = array[tuple(indices)]

        return array


_equals = cfdm.Data()._equals


def equals(x, y, rtol=None, atol=None, ignore_data_type=False,
           **kwargs):
    '''
    '''
    if rtol is None:
        rtol = RTOL()
    if atol is None:
        atol = ATOL()

    return _equals(x, y, rtol=rtol, atol=atol,
                   ignore_data_type=ignore_data_type,
                   **kwargs)


def equivalent(x, y, rtol=None, atol=None, traceback=False):
    '''True if and only if two objects are logically equivalent.

    If the first argument, *x*, has an `!equivalent` method then it is
    used, and in this case ``equivalent(x, y)`` is the same as
    ``x.equivalent(y)``.

    :Parameters:

        x, y :
            The objects to compare for equivalence.

        atol : float, optional
            The absolute tolerance for all numerical comparisons, By
            default the value returned by the `ATOL` function is used.

        rtol : float, optional
            The relative tolerance for all numerical comparisons, By
            default the value returned by the `RTOL` function is used.

        traceback : bool, optional
            If True then print a traceback highlighting where the two
            objects differ.

    :Returns:

        `bool`
            Whether or not the two objects are equivalent.

    **Examples:**

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

    '''

    if rtol is None:
        rtol = RTOL()
    if atol is None:
        atol = ATOL()

    eq = getattr(x, 'equivalent', None)
    if callable(eq):
        # x has a callable equivalent method
        return eq(y, rtol=rtol, atol=atol, traceback=traceback)

    eq = getattr(y, 'equivalent', None)
    if callable(eq):
        # y has a callable equivalent method
        return eq(x, rtol=rtol, atol=atol, traceback=traceback)

    return equals(x, y, rtol=rtol, atol=atol, ignore_fill_value=True,
                  traceback=traceback)


def load_stash2standard_name(table=None, delimiter='!', merge=True):
    '''Load a STASH to standard name conversion table.

    This used when reading PP and UM fields files.

    :Parameters:

        table: `str`, optional
            Use the conversion table at this file location. By default
            the table will be looked for at
            ``os.path.join(os.path.dirname(cf.__file__),'etc/STASH_to_CF.txt')``

        delimiter: `str`, optional
            The delimiter of the table columns. By default, ``!`` is
            taken as the delimiter.

        merge: `bool`, optional
            If *table* is None then *merge* is taken as False,
            regardless of its given value.

    :Returns:

        `dict`
            The new STASH to standard name conversion table.

    **Examples:**

    >>> cf.load_stash2standard_name()
    >>> cf.load_stash2standard_name('my_table.txt')
    >>> cf.load_stash2standard_name('my_table2.txt', ',')
    >>> cf.load_stash2standard_name('my_table3.txt', merge=True)
    >>> cf.load_stash2standard_name('my_table4.txt', merge=False)

    '''
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
    number_regex = '([-+]?\d*\.?\d+(e[-+]?\d+)?)'

    if table is None:
        # Use default conversion table
        merge = False
        package_path = os.path.dirname(__file__)
        table = os.path.join(package_path, 'etc/STASH_to_CF.txt')
    # --- End: if

    with open(table, 'r') as open_table:
        lines = csv.reader(open_table, delimiter=delimiter,
                           skipinitialspace=True)
        lines = list(lines)

    raw_list = []
    [raw_list.append(line) for line in lines]

    # Get rid of comments
    for line in raw_list[:]:
        if line[0].startswith('#'):
            raw_list.pop(0)
            continue

        break

    # Convert to a dictionary which is keyed by (submodel, STASHcode)
    # tuples
    (model, stash, name,
     units,
     valid_from, valid_to,
     standard_name, cf, pp) = list(range(9))

    stash2sn = {}
    for x in raw_list:
        key = (int(x[model]), int(x[stash]))

        if not x[units]:
            x[units] = None

        try:
            cf_info = {}
            if x[cf]:
                for d in x[7].split():
                    if d.startswith('height='):
                        cf_info['height'] = re.split(number_regex, d,
                                                     re.IGNORECASE)[1:4:2]
                        if cf_info['height'] == '':
                            cf_info['height'][1] = '1'

                    if d.startswith('below_'):
                        cf_info['below'] = re.split(number_regex, d,
                                                    re.IGNORECASE)[1:4:2]
                        if cf_info['below'] == '':
                            cf_info['below'][1] = '1'

                    if d.startswith('where_'):
                        cf_info['where'] = d.replace('where_', 'where ', 1)
                    if d.startswith('over_'):
                        cf_info['over'] = d.replace('over_', 'over ', 1)

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
    # --- End: for

    if not merge:
        _stash2standard_name.clear()

    _stash2standard_name.update(stash2sn)

    return _stash2standard_name


def flat(x):
    '''Return an iterator over an arbitrarily nested sequence.

    :Parameters:

        x: scalar or arbitrarily nested sequence
            The arbitrarily nested sequence to be flattened. Note that
            a If *x* is a string or a scalar then this is equivalent
            to passing a single element sequence containing *x*.

    :Returns:

        generator
            An iterator over flattened sequence.

    **Examples:**

    >>> cf.flat([1, [2, [3, 4]]])
    <generator object flat at 0x3649cd0>

    >>> list(cf.flat([1, (2, [3, 4])]))
    [1, 2, 3, 4]

    >>> import numpy
    >>> list(cf.flat((1, [2, numpy.array([[3, 4], [5, 6]])]))
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

    >>> fl = cf.FieldList(cf.flat([f, [f, [f, f]]])
    >>> fl
    [<CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>,
     <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>,
     <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>,
     <CF Field: eastward_wind(air_pressure(5), latitude(110), longitude(106)) m s-1>]

    '''
    if not isinstance(x, Iterable) or isinstance(x, str):
        x = (x,)

    for a in x:
        if not isinstance(a, str) and isinstance(a, Iterable):
            for sub in flat(a):
                yield sub
        else:
            yield a
    # --- End: for


def abspath(filename):
    '''Return a normalized absolute version of a file name.

    If `None` or a string containing URL is provided then it is
    returned unchanged.

    .. seealso:: `cf.dirname`, `cf.pathjoin`, `cf.relpath`

    :Parameters:

        filename: `str` or `None`
            The name of the file, or `None`

    :Returns:

        `str`

            The normalized absolutized version of *filename*, or
            `None`.

    **Examples:**

    >>> import os
    >>> os.getcwd()
    '/data/archive'
    >>> cf.abspath('file.nc')
    '/data/archive/file.nc'
    >>> cf.abspath('..//archive///file.nc')
    '/data/archive/file.nc'
    >>> cf.abspath('http://data/archive/file.nc')
    'http://data/archive/file.nc'

    '''
    if filename is None:
        return

    u = urllib.parse.urlparse(filename)
    if u.scheme != '':
        return filename

    return _os_path_abspath(filename)


def relpath(filename, start=None):
    '''Return a relative filepath to a file.

    The filepath is relative either from the current directory or from
    an optional start point.

    If a string containing URL is provided then it is returned unchanged.

    .. seealso:: `cf.abspath`, `cf.dirname`, `cf.pathjoin`

    :Parameters:

        filename: `str`
            The name of the file.

        start: `str`, optional
            The start point for the relative path. By default the
            current directoty is used.

    :Returns:

        `str`
            The relative path.

    **Examples:**

    >>> cf.relpath('/data/archive/file.nc')
    '../file.nc'
    >>> cf.relpath('/data/archive///file.nc', start='/data')
    'archive/file.nc'
    >>> cf.relpath('http://data/archive/file.nc')
    'http://data/archive/file.nc'

    '''
    u = urllib.parse.urlparse(filename)
    if u.scheme != '':
        return filename

    if start is not None:
        return _os_path_relpath(filename, start)

    return _os_path_relpath(filename)


def dirname(filename):
    '''Return the directory name of a file.

    If a string containing URL is provided then everything up to, but
    not including, the last slash (/) is returned.

    .. seealso:: `cf.abspath`, `cf.pathjoin`, `cf.relpath`

    :Parameters:

        filename: `str`
            The name of the file.

    :Returns:

        `str`
            The directory name.

    **Examples:**

    >>> cf.dirname('/data/archive/file.nc')
    '/data/archive'
    >>> cf.dirname('..//file.nc')
    '..'
    >>> cf.dirname('http://data/archive/file.nc')
    'http://data/archive'

    '''
    u = urllib.parse.urlparse(filename)
    if u.scheme != '':
        return filename.rpartition('/')[0]

    return _os_path_dirname(filename)


def pathjoin(path1, path2):
    '''Join two file path components intelligently.

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

    **Examples:**

    >>> cf.pathjoin('/data/archive', '../archive/file.nc')
    '/data/archive/../archive/file.nc'
    >>> cf.pathjoin('/data/archive', '../archive/file.nc')
    '/data/archive/../archive/file.nc'
    >>> cf.abspath(cf.pathjoin('/data/', 'archive/')
    '/data/archive'
    >>> cf.pathjoin('http://data', 'archive/file.nc')
    'http://data/archive/file.nc'

    '''
    u = urllib.parse.urlparse(path1)
    if u.scheme != '':
        return urllib.parse.urljoin(path1, path2)

    return _os_path_join(path1, path2)


def hash_array(array):
    '''Return the hash value of a numpy array.

    The hash value is dependent on the data type, shape of the data
    array. If the array is a masked array then the hash value is
    independent of the fill value and of data array values underlying
    any masked elements.

    The hash value is not guaranteed to be portable across versions of
    Python, numpy and cf.

    :Parameters:

        array: `numpy.ndarray`
            The numpy array to be hashed. May be a masked array.

    :Returns:

        `int`
            The hash value.

    **Examples:**

    >>> print(array)
    [[0 1 2 3]]
    >>> cf.hash_array(array)
    -8125230271916303273
    >>> array[1, 0] = numpy.ma.masked
    >>> print(array)
    [[0 -- 2 3]]
    >>> cf.hash_array(array)
    791917586613573563
    >>> array.hardmask = False
    >>> array[0, 1] = 999
    >>> array[0, 1] = numpy.ma.masked
    >>> cf.hash_array(array)
    791917586613573563
    >>> array.squeeze()
    >>> print(array)
    [0 -- 2 3]
    >>> cf.hash_array(array)
    -7007538450787927902
    >>> array.dtype = float
    >>> print(array)
    [0.0 -- 2.0 3.0]
    >>> cf.hash_array(array)
    -4816859207969696442

    '''
    h = hashlib_md5()

    h_update = h.update

    h_update(marshal_dumps(array.dtype.name))
    h_update(marshal_dumps(array.shape))

    if _numpy_ma_isMA(array):
        if _numpy_ma_is_masked(array):
            mask = array.mask
            if not mask.flags.c_contiguous:
                mask = _numpy_ascontiguousarray(mask)

            h_update(mask)
            array = array.copy()
            array.set_fill_value()
            array = array.filled()
        else:
            array = array.data
    # --- End: if

    if not array.flags.c_contiguous:
        # array = array.copy()
        array = _numpy_ascontiguousarray(array)

    h_update(array)

    return hash(h.digest())


def inspect(self):
    '''Inspect the attributes of an object.

    :Returns:

        `None`

    '''
    name = repr(self)
    out = [name, ''.ljust(len(name), '-')]

    if hasattr(self, '__dict__'):
        for key, value in sorted(self.__dict__.items()):
            out.append('{}: {!r}'.format(key, value))

    print('\n'.join(out))


def broadcast_array(array, shape):
    '''Broadcast an array to a given shape.

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

    **Examples:**


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

    '''
    a_shape = _numpy_shape(array)
    if a_shape == shape:
        return array

    tile = [(m if n == 1 else 1)
            for n, m in zip(a_shape[::-1], shape[::-1])]
    tile = shape[0:len(shape)-len(a_shape)] + tuple(tile[::-1])

    return _numpy_tile(array, tile)


def allclose(x, y, rtol=None, atol=None):
    '''Returns True if two broadcastable arrays have equal values to
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
            default the value returned by the `ATOL` function is used.

        rtol: `float`, optional
            The relative tolerance for all numerical comparisons, By
            default the value returned by the `RTOL` function is used.

    :Returns:

        `bool`
            Returns True if the arrays are equal, otherwise False.

    **Examples:**

    '''
    if rtol is None:
        rtol = RTOL()
    if atol is None:
        atol = ATOL()

    allclose = getattr(x, 'allclose', None)
    if callable(allclose):
        # x has a callable allclose method
        return allclose(y, rtol=rtol, atol=atol)

    allclose = getattr(y, 'allclose', None)
    if callable(allclose):
        # y has a callable allclose method
        return allclose(x, rtol=rtol, atol=atol)

    # x nor y has a callable allclose method
    return _numpy_allclose(x, y, rtol=rtol, atol=atol)


def _section(x, axes=None, data=False, stop=None, chunks=False,
             min_step=1, **kwargs):
    '''Return a list of m dimensional sections of a Field of n dimensions
    or a dictionary of m dimensional sections of a Data object of n
    dimensions, where m <= n.

    In the case of a `Data` object, the keys of the dictionary are the
    indicies of the sections in the original Data object. The m
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
            Stop after taking this number of sections and return. If
            stop is None all sections are taken.

        chunks: `bool`, optional
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

    **Examples:**

    Section a field into 2D longitude/time slices, checking the units:

    >>> _section(f, {None: 'longitude', units: 'radians'},
    ...             {None: 'time',
    ...              'units': 'days since 2006-01-01 00:00:00'})

    Section a field into 2D longitude/latitude slices, requiring exact
    names:

    >>> _section(f, ['latitude', 'longitude'], exact=True)

    '''
    def loop_over_index(x, current_index, axis_indices, indices):
        '''Expects an index to loop over in the list indices. If this is less
        than 0 the horizontal slice defined by indices is appended to
        the FieldList fl, if it is the specified axis indices the
        value in indices is left as slice(None) and it calls itself
        recursively with the next index, otherwise each index is
        looped over. In this loop the routine is called recursively
        with the next index. If the count of the number of slices
        taken is greater than or equal to stop it returns before
        taking any more slices.

        '''
        if current_index < 0:
            if data:
                d[tuple([x.start for x in indices])] = x[tuple(indices)]
            else:
                fl.append(x[tuple(indices)])

            nl_vars['count'] += 1
            return

        if current_index in axis_indices:
            loop_over_index(x, current_index - 1, axis_indices, indices)
            return

        for i in range(0, sizes[current_index], steps[current_index]):
            if stop is not None and nl_vars['count'] >= stop:
                return

            indices[current_index] = slice(i, i + steps[current_index])
            loop_over_index(x, current_index - 1, axis_indices, indices)
    # --- End: def

    # Retrieve the index of each axis defining the sections
    if data:
        if isinstance(axes, int):
            axes = (axes,)

        if not axes:
            axis_indices = tuple(range(x.ndim))
        else:
            axis_indices = axes
    else:
        axis_keys = [x.domain_axis(axis, key=True) for axis in axes]
        axis_indices = list()
        for key in axis_keys:
            try:
                axis_indices.append(x.get_data_axes().index(key))
            except ValueError:
                pass
    # --- End: if

    # find the size of each dimension
    sizes = x.shape

    if chunks:
        steps = list(sizes)

        # Define the factor which, when multiplied by the size of the
        # data array, determines how many chunks are in the data
        # array.
        #
        # I.e. factor = 1/(the number of words per chunk)
        factor = (x.dtype.itemsize + 1.0)/CHUNKSIZE()

        # n_chunks = number of equal sized bits the partition needs to
        #            be split up into so that each bit's size is less
        #            than the chunk size.
        n_chunks = int(math_ceil(x.size * factor))

        for (index, axis_size) in enumerate(sizes):
            if index in axis_indices:
                # Do not attempt to "chunk" non-sectioned axes
                continue

            if int(math_ceil(float(axis_size)/min_step)) <= n_chunks:
                n_chunks = int(math_ceil(n_chunks/float(axis_size)*min_step))
                steps[index] = min_step

            else:
                steps[index] = int(axis_size/n_chunks)
                break
    else:
        steps = [size if i in axis_indices else 1 for i, size in
                 enumerate(sizes)]

    # Use recursion to slice out each section
    if data:
        d = dict()
    else:
        fl = []

    indices = [slice(None)] * len(sizes)

    nl_vars = {'count': 0}

    current_index = len(sizes) - 1
    loop_over_index(x, current_index, axis_indices, indices)

    if data:
        return d
    else:
        return fl


def environment(display=True, paths=True, string=True):
    '''Return the names and versions of the cf package and its
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

    **Examples:**

    >>> cf.environment()
    Platform: Linux-4.15.0-64-generic-x86_64-with-debian-stretch-sid
    HDF5 library: 1.10.2
    netcdf library: 4.6.1
    udunits2 library: libudunits2.so.0
    python: 3.7.3 /home/space/anaconda3/bin/python
    netCDF4: 1.4.2 /home/space/anaconda3/lib/python3.7/site-packages/netCDF4/__init__.py
    cftime: 1.0.3.4 /home/space/.local/lib/python3.7/site-packages/cftime-1.0.3.4-py3.7-linux-x86_64.egg/cftime/__init__.py
    numpy: 1.16.2 /home/space/anaconda3/lib/python3.7/site-packages/numpy/__init__.py
    psutil: 5.6.3 /home/space/anaconda3/lib/python3.7/site-packages/psutil/__init__.py
    scipy: 1.2.1 /home/space/anaconda3/lib/python3.7/site-packages/scipy/__init__.py
    matplotlib: 3.1.1 /home/space/anaconda3/lib/python3.7/site-packages/matplotlib/__init__.py
    ESMF: 7.1.0r /home/space/anaconda3/lib/python3.7/site-packages/ESMF/__init__.py
    cfdm: 1.7.8 /home/space/anaconda3/lib/python3.7/site-packages/cfdm/__init__.py
    cfunits: 3.2.2 /home/space/anaconda3/lib/python3.7/site-packages/cfunits/__init__.py
    cfplot: 3.0.0 /home/space/anaconda3/lib/python3.7/site-packages/cfplot/__init__.py
    cf: 3.0.1 /home/space/anaconda3/lib/python3.7/site-packages/cf/__init__.py

    >>> cf.environment(paths=False)
    Platform: Linux-4.15.0-64-generic-x86_64-with-debian-stretch-sid
    HDF5 library: 1.10.2
    netcdf library: 4.6.1
    udunits2 library: libudunits2.so.0
    python: 3.7.3
    netCDF4: 1.4.2
    cftime: 1.0.3.4
    numpy: 1.16.2
    psutil: 5.6.3
    scipy: 1.2.1
    matplotlib: 3.1.1
    ESMF: 7.1.0r
    cfdm: 1.7.8
    cfunits: 3.2.2
    cfplot: 3.0.0
    cf: 3.0.1

    '''
    out = []
    out.append('Platform: ' + str(platform.platform()))
    out.append('HDF5 library: ' + str(netCDF4. __hdf5libversion__))
    out.append('netcdf library: ' + str(netCDF4.__netcdf4libversion__))
    out.append(
        'udunits2 library: ' + str(ctypes.util.find_library('udunits2')))
    out.append('python: ' + str(platform.python_version()))
    if paths:
        out[-1] += ' ' + str(_sys_executable)

    out.append('netCDF4: ' + str(netCDF4.__version__))
    if paths:
        out[-1] += ' ' + str(_os_path_abspath(netCDF4.__file__))

    out.append('cftime: ' + str(cftime.__version__))
    if paths:
        out[-1] += ' ' + str(_os_path_abspath(cftime.__file__))

    out.append('numpy: ' + str(_numpy__version__))
    if paths:
        out[-1] += ' ' + str(_os_path_abspath(_numpy__file__))

    out.append('psutil: ' + str(psutil.__version__))
    if paths:
        out[-1] += ' ' + str(_os_path_abspath(psutil.__file__))

    try:
        import scipy
    except:
        out.append('scipy: not available')
    else:
        out.append('scipy: ' + str(scipy.__version__))
        if paths:
            out[-1] += ' ' + str(_os_path_abspath(scipy.__file__))
    # --- End: try

    try:
        import matplotlib
    except:
        out.append('matplotlib: not available')
    else:
        out.append('matplotlib: ' + str(matplotlib.__version__))
        if paths:
            out[-1] += ' ' + str(_os_path_abspath(matplotlib.__file__))
    # --- End: try

    try:
        import ESMF
    except:
        out.append('ESMF: not available')
    else:
        out.append('ESMF: ' + str(ESMF.__version__))
        if paths:
            out[-1] += ' ' + str(_os_path_abspath(ESMF.__file__))
    # --- End: try

    out.append('cfdm: ' + str(cfdm.__version__))
    if paths:
        out[-1] += ' ' + str(_os_path_abspath(cfdm.__file__))

    out.append('cfunits: ' + str(cfunits.__version__))
    if paths:
        out[-1] += ' ' + str(_os_path_abspath(cfunits.__file__))

    try:
        import cfplot
    except ImportError:
        out.append('cfplot: not available')
    else:
        out.append('cfplot: ' + str(cfplot.__version__))
        if paths:
            out[-1] += ' ' + str(_os_path_abspath(cfplot.__file__))
    # --- End: try

    out.append('cf: ' + str(__version__))
    if paths:
        out[-1] += ' ' + str(_os_path_abspath(__file__))

    out = '\n'.join(out)

    if display:
        print(out)
    else:
        return(out)


def default_netCDF_fillvals():
    '''Default data array fill values for each data type.

    :Returns:

        `dict`
            The fill values, keyed by `numpy` data type strings

    **Examples:**

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

    '''
    return netCDF4.default_fillvals


def _DEPRECATION_ERROR(message='', version='3.0.0'):
    raise DeprecationError("{}".format(message))


def _DEPRECATION_ERROR_ARG(instance, method, arg, message='', version='3.0.0'):
    raise DeprecationError(
        "Argument {2!r} of method '{0}.{1}' has been deprecated at version "
        "{4} and is no longer available. {3}".format(
            instance.__class__.__name__,
            method,
            arg,
            message,
            version
        )
    )


def _DEPRECATION_ERROR_FUNCTION_KWARGS(func, kwargs=None, message='',
                                       exact=False, traceback=False,
                                       info=False, version='3.0.0'):
    # Unsafe to set mutable '{}' as default in the func signature.
    if kwargs is None:  # distinguish from falsy '{}'
        kwargs = {}

    for kwarg, msg in KWARGS_MESSAGE_MAP.items():
        # This eval is safe as the kwarg is not a user input
        if kwarg in ('exact', 'traceback') and eval(kwarg):
            kwargs = {kwarg: None}
            message = msg

    for key in kwargs.keys():
        raise DeprecationError(
            "Keyword {1!r} of function '{0}' has been deprecated at version "
            "{3} and is no longer available. {2}".format(
                func,
                key,
                message,
                version
            )
        )


def _DEPRECATION_ERROR_KWARGS(instance, method, kwargs=None, message='',
                              i=False, traceback=False, axes=False,
                              exact=False, relaxed_identity=False,
                              info=False, version='3.0.0'):
    # Unsafe to set mutable '{}' as default in the func signature.
    if kwargs is None:  # distinguish from falsy '{}'
        kwargs = {}

    for kwarg, msg in KWARGS_MESSAGE_MAP.items():
        if eval(kwarg):  # safe as this is not a kwarg input by the user
            kwargs = {kwarg: None}
            message = msg

    for key in kwargs.keys():
        raise DeprecationError(
            "Keyword {2!r} of method '{0}.{1}' has been deprecated at "
            "version {4} and is no longer available. {3}".format(
                instance.__class__.__name__,
                method,
                key,
                message,
                version
            )
        )


def _DEPRECATION_ERROR_KWARG_VALUE(
        instance, method, kwarg, value, message='', version='3.0.0'):
    raise DeprecationError(
        "Value {!r} of keyword {!r} of method '{}.{}' has been deprecated at "
        "version {} and is no longer available. {}".format(
            value,
            kwarg,
            method,
            instance.__class__.__name__,
            version,
            message
        )
    )


def _DEPRECATION_ERROR_METHOD(instance, method, message='', version='3.0.0'):
    raise DeprecationError(
        "{} method {!r} has been deprecated at version {} and is no longer "
        "available. {}".format(
            instance.__class__.__name__,
            method,
            version,
            message
        )
    )


def _DEPRECATION_ERROR_ATTRIBUTE(
        instance, attribute, message='', version='3.0.0'):
    raise DeprecationError(
        "{} attribute {!r} has been deprecate at version {} and is no longer "
        "available. {}".format(
            instance.__class__.__name__,
            attribute,
            version,
            message
        )
    )


def _DEPRECATION_ERROR_FUNCTION(func, message='', version='3.0.0'):
    raise DeprecationError(
        "Function {!r} has been deprecated at version {} and is no longer "
        "available. {}".format(
            func,
            version,
            message
        )
    )


def _DEPRECATION_ERROR_CLASS(cls, message='', version='3.0.0'):
    raise DeprecationError(
        "Class {!r} has been deprecated at version {} and is no longer "
        "available. {}".format(
            cls,
            version,
            message
        )
    )


def _DEPRECATION_WARNING_METHOD(
        instance, method, message='', new=None, version='3.0.0'):
    warnings.warn(
        "{} method {!r} has been deprecated at version {} and will be "
        "removed in a future version. {}".format(
            instance.__class__.__name__,
            method,
            version,
            message),
        DeprecationWarning
    )


def _DEPRECATION_ERROR_DICT(message='', version='3.0.0'):
    raise DeprecationError(
        "Use of a 'dict' to identify constructs has been deprecated at "
        "version {} and is no longer available. {}".format(
            version,
            message
        )
    )


def _DEPRECATION_ERROR_SEQUENCE(instance, version='3.0.0'):
    raise DeprecationError(
        "Use of a {!r} to identify constructs has been deprecated at version "
        "{} and is no longer available. Use the * operator to unpack the "
        "arguments instead.".format(
            instance.__class__.__name__,
            version
        )
    )


# --------------------------------------------------------------------
# Deprecated functions
# --------------------------------------------------------------------
def default_fillvals():
    '''Default data array fill values for each data type.

    Deprecated at version 3.0.2 and is no longer available. Use
    function `cf.default_netCDF_fillvals` instead.

    '''
    _DEPRECATION_ERROR_FUNCTION(
        'default_fillvals',
        "Use function 'cf.default_netCDF_fillvals' instead.",
        version='3.0.2')  # pragma: no cover


def set_equals(x, y, rtol=None, atol=None, ignore_fill_value=False,
               traceback=False):
    '''Deprecated at version 3.0.0.

    '''
    _DEPRECATION_ERROR_FUNCTION('cf.set_equals')  # pragma: no cover
