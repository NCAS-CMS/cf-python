import datetime
from functools import partial

import numpy

import cftime

from numpy import around as numpy_around
from numpy import array as numpy_array
from numpy import asanyarray as numpy_asanyarray
from numpy import ndarray as numpy_ndarray
from numpy import ndim as numpy_ndim
from numpy import vectorize as numpy_vectorize

from numpy.ma import isMA as numpy_ma_isMA
from numpy.ma import is_masked as numpy_ma_is_masked
from numpy.ma import masked_all as numpy_ma_masked_all
from numpy.ma import masked_where as numpy_ma_masked_where
from numpy.ma import nomask as numpy_ma_nomask

from .functions import _DEPRECATION_ERROR_CLASS

# # Define some useful units
#  _calendar_years  = Units('calendar_years')
#  _calendar_months = Units('calendar_months')

_default_calendar = 'gregorian'
# _canonical_calendar = {'gregorian'          : 'gregorian'          ,
#                        'standard'           : 'gregorian'          ,
#                        'none'               : 'gregorian'          ,
#                        'proleptic_gregorian': 'proleptic_gregorian',
#                        '360_day'            : '360_day'            ,
#                        'noleap'             : '365_day'            ,
#                        '365_day'            : '365_day'            ,
#                        'all_leap'           : '366_day'            ,
#                        '366_day'            : '366_day'            ,
#                        'julian'             : 'julian'             ,
#                        }

# --------------------------------------------------------------------
# Mapping of CF calendars to date-time objects
# --------------------------------------------------------------------
_datetime_object = {
     ('',): partial(cftime.datetime, calendar=''),
     (None, 'gregorian', 'standard', 'none'): cftime.DatetimeGregorian,
     ('proleptic_gregorian',): cftime.DatetimeProlepticGregorian,
     ('360_day',): cftime.Datetime360Day,
     ('noleap', '365_day'): cftime.DatetimeNoLeap,
     ('all_leap', '366_day'): cftime.DatetimeAllLeap,
     ('julian',): cftime.DatetimeJulian,
}


class Datetime(cftime.datetime):
    '''A date-time object which supports CF calendars.

    Deprecated at version 3.0.0 and is no longer available. Use
    functions 'cf.dt' to create date-time objects instead.

    '''
    def __init__(self, year, month=1, day=1, hour=0, minute=0, second=0,
                 microsecond=0, dayofwk=-1, dayofyr=1, calendar=None):
        '''
        '''
        _DEPRECATION_ERROR_CLASS(
             'Datetime',
             "Use function 'cf.dt' to create date-time objects instead."
        )  # pragma: no cover


# --- End: class

def elements(x):
    return x.timetuple()[:6]  # + (getattr(x, 'microsecond', 0),)


def dt(arg, month=1, day=1, hour=0, minute=0, second=0,
       microsecond=0, calendar=''):  # None):
    '''Return a date-time object for a date and time according to a
    calendar.

    .. seealso:: `cf.dt_vector`

    :Parameters:

        arg:
            A multi-purpose argument that is one of:

            * An `int` specifying the calendar year, used in
              conjunction with the *month*, *day*, *hour*, *minute*,
              *second* and *microsecond* parameters.

            * A `str` specifying an ISO 8601-like date-time string (in
              which non-Gregorian calendar dates are allowed).

            * `datetime.datetime or sublcass `cftime.datetime` (such
              as `Datetime360Day`). A new date-time object is returned
              for the given date-time.


        calendar: `str`, optional
            The calendar for the date-time. By default the Gregorian
            calendar is used.

            *Parameter example:*
              ``calendar='360_day'``

    :Returns:

        (sublcass of) `cftime.datetime`
            The new date-time object.

    **Examples:**

    >>> d = cf.dt(2003)
    >>> d
    cftime.DatetimeGregorian(2003-01-01 00:00:00)
    >>> print(d)
    2003-01-01 00:00:00

    >>> d = cf.dt(2003, 2, 30, calendar='360_day')
    >>> d = cf.dt(2003, 2, 30, 0, 0, 0, calendar='360_day')
    >>> d = cf.dt('2003-2-30', calendar='360_day')
    >>> d = cf.dt('2003-2-30 0:0:0', calendar='360_day')
    >>> d
    cftime.Datetime360Day(2003:02:30 00:00:00)
    >>> print(d)
    2003-02-30 00:00:00

    >>> d = cf.dt(2003, 4, 5, 12, 30, 15)
    >>> d = cf.dt(year=2003, month=4, day=5, hour=12, minute=30, second=15)
    >>> d = cf.dt('2003-04-05 12:30:15')
    >>> d.year, d.month, d.day, d.hour, d.minute, d.second
    (2003, 4, 5, 12, 30, 15)

    '''
    if isinstance(arg, str):
        (year, month, day, hour, minute, second, microsecond) = st2elements(
            arg)

    elif isinstance(arg, cftime.datetime):
        (year, month, day, hour, minute, second, microsecond) = (
            arg.year, arg.month, arg.day, arg.hour, arg.minute, arg.second,
            arg.microsecond)
        if calendar == '':
            calendar = arg.calendar

    elif isinstance(arg, datetime.datetime):
        (year, month, day, hour, minute, second) = arg.timetuple()[:6]
        microsecond = arg.microsecond
        if calendar == '':
            calendar = _default_calendar

    else:
        year = arg

    for calendars, datetime_cls in _datetime_object.items():
        if calendar in calendars:
            return datetime_cls(year, month, day, hour, minute,
                                second, microsecond)
    # --- End: for

    raise ValueError(
        "Can't create date-time object with unknown calendar {!r}".format(
            calendar))


def dt_vector(arg, month=1, day=1, hour=0, minute=0, second=0,
              microsecond=0, calendar=None):
    '''Return a scalar or 1-d array of date-time objects.

    .. seealso:: `cf.dt`

    :Parameters:

        arg:
            A multi-purpose argument that is one of:

            * An `int`, or sequence of `int`, specifying the calendar
              years, used in conjunction with the *month*, *day*,
              *hour*, *minute*, *second* and *microsecond* parameters.

            * A `str`, or sequence of `str`, specifying ISO 8601-like
              date-time strings (in which non-Gregorian calendar dates
              are allowed).

            * A two dimensional array of `int`. There may be up to 7
              columns, each one specifying the years, months, days,
              hours minutes, seconds and microseconds respectively. If
              fewer than 7 trailing dimensions are provided then the
              default value for the missing components are used

        calendar: `str`, optional
            The calendar for the date-times. By default the Gregorian
            calendar is used.

            *Parameter example:*
              ``calendar='360_day'``

    :Returns:

        `numpy.ndarray`
            The scalar or 1-d array of date-time objects.

    **Examples:**

    TODO

    '''
    arg = numpy.array(arg)
    month = numpy.array(month)
    day = numpy.array(day)
    hour = numpy.array(hour)
    minute = numpy.array(minute)
    second = numpy.array(second)
    microsecond = numpy.array(microsecond)

    ndim = max(map(numpy.ndim, (month, day, hour, minute, second,
                                microsecond)))

    if ndim > 1:
        raise ValueError('TODO')

    if arg.ndim > 2:
        raise ValueError('TODO')

    sizes = set(map(numpy.size, (arg, month, day, hour, minute,
                                 second, microsecond)))

    if len(sizes) == 1 and 1 in sizes:
        # All arguments are scalars or size 1
        out = dt(arg.item(), month.item(), day.item(), hour.item(),
                 minute.item(), second.item(), microsecond.item(),
                 calendar=calendar)
        if ndim >= 1:
            out = [out]

        return numpy.array(out)

    if arg.ndim == 2 and arg.shape[1] > 7:
        raise ValueError('TODO (890)')

    if arg.ndim == 1:
        if arg.dtype.kind in 'UOS':
            out = [dt(a, calendar=calendar) for a in arg]

        else:
            if len(sizes) > 2:
                raise ValueError('TODO (891)')

            if len(sizes) == 2 and 1 not in sizes:
                raise ValueError('TODO (892)')

            _ = numpy.empty((max(sizes), 7), dtype=int)
            _[:, 0] = arg
            _[:, 1] = month
            _[:, 2] = day
            _[:, 3] = hour
            _[:, 4] = minute
            _[:, 5] = second
            _[:, 6] = microsecond
            arg = _

            out = [dt(*args, calendar=calendar) for args in arg]
    # --- End: if

    return numpy.array(out)


def st2dt(array, units_in=None, dummy0=None, dummy1=None):
    '''The returned array is always independent.

    :Parameters:

        array: numpy array-like

        units_in: `Units`, optional

        dummy0: optional
            Ignored.

        dummy1: optional
            Ignored.

    :Returns:

        `numpy.ndarray`
            An array of `cf.Datetime` or `datetime.datetime` objects
            with the same shape as *array*.

    **Examples:**

    '''
    func = partial(st2datetime, calendar=units_in._calendar)
    return numpy_vectorize(func, otypes=[object])(array)


# def st2datetime(date_string):
#     '''
#
# Parse an ISO 8601 date-time string into a datetime.datetime object.
#
# :Parameters:
#
#     date_string: `str`
#
# :Returns:
#
#     `datetime.datetime`
#
# '''
#     if date_string.count('-') != 2:
#         raise ValueError("A string must contain a year, a month and a day")
#
#     _ = cftime._parse_date(date_string)
#     if len(_) == 7:
#         year, month, day, hour, minute, second, utc_offset = _
#     else:
#         year, month, day, hour, minute, second, microsecond, utc_offset = _
#
#     if utc_offset:
#         raise ValueError("Can't specify a time offset from UTC")
#
#     return datetime(year, month, day, hour, minute, second)
#
#
# array_st2datetime = numpy_vectorize(st2datetime, otypes=[object])


def st2datetime(date_string, calendar=None):
    '''Parse an ISO 8601 date-time string into a `cftime` object.

    :Parameters:

        date_string: `str`

    :Returns:

        subclass of `cftime.datetime`

    '''
    if date_string.count('-') != 2:
        raise ValueError(
            "Input date-time string must contain at least a year, a month "
            "and a day"
        )

    _ = cftime._parse_date(date_string)
    if len(_) == 7:
        year, month, day, hour, minute, second, utc_offset = _
        microsecond = 0
    else:
        year, month, day, hour, minute, second, microsecond, utc_offset = _

    if utc_offset:
        raise ValueError("Can't specify a time offset from UTC")

    #    return Datetime(year, month, day, hour, minute, second)
    return dt(year, month, day, hour, minute, second, microsecond,
              calendar=calendar)


def st2elements(date_string):
    '''Parse an ISO 8601 date-time string into a `cftime` object.

    :Parameters:

        date_string: `str`

    :Returns:

        `tuple`

    '''
    if date_string.count('-') != 2:
        raise ValueError(
            "Input date-time string must contain at least a year, a month "
            "and a day"
        )

    _ = cftime._parse_date(date_string)
    if len(_) == 7:
        year, month, day, hour, minute, second, utc_offset = _
        microsecond = 0
    else:
        year, month, day, hour, minute, second, microsecond, utc_offset = _

    if utc_offset:
        raise ValueError("Can't specify a time offset from UTC")

    return (year, month, day, hour, minute,
            second, microsecond)  # round((second % 1 )* 1e6))


# array_st2Datetime = numpy_vectorize(st2Datetime, otypes=[object])

def rt2dt(array, units_in, units_out=None, dummy1=None):
    '''Convert reference times  to date-time objects

    The returned array is always independent.

    :Parameters:

        array: numpy array-like

        units_in: `Units`

        units_out: *optional*
            Ignored.

        dummy1:
            Ignored.

    :Returns:

        `numpy.ndarray`
            An array of `cf.Datetime` objects with the same shape as
            *array*.

    '''
    mask = None
    if numpy_ma_isMA(array):
        # num2date has issues if the mask is nomask
        mask = array.mask
        if mask is numpy_ma_nomask or not numpy_ma_is_masked(array):
            array = array.view(numpy_ndarray)

    units = units_in.units
    calendar = getattr(units_in, 'calendar', 'standard')

    array = cftime.num2date(array, units, calendar,
                            only_use_cftime_datetimes=True)
#    array = units_in._utime.num2date(array)

    if mask is not None:
        array = numpy_ma_masked_where(mask, array)

    ndim = numpy_ndim(array)

    if mask is None:
        # There is no missing data
        return numpy_array(array, dtype=object)
        # return numpy_vectorize(
        #     partial(dt2Dt, calendar=units_in._calendar),
        #     otypes=[object])(array)
    else:
        # There is missing data
        if not ndim:
            return numpy_ma_masked_all((), dtype=object)
        else:
            # array = numpy_array(array)
            # array = numpy_vectorize(
            #     partial(dt2Dt, calendar=units_in._calendar),
            #     otypes=[object])(array)
            return numpy_ma_masked_where(mask, array)


def dt2Dt(x, calendar=None):
    '''Convert a datetime.datetime object to a cf.Datetime object

    '''
    if not x:
        return False
#    return Datetime(*elements(x), calendar=calendar)
    return dt(x, calendar=calendar)


def dt2rt(array, units_in, units_out, dummy1=None):
    '''Round to the nearest millisecond. This is only necessary whilst
    netCDF4 time functions have an accuracy of no better than 1
    millisecond (which is still the case at version 1.2.2).

    The returned array is always independent.

    :Parameters:

        array: numpy array-like of date-time objects

        units_in:
            Ignored.

        units_out: `Units`

        dummy1:
            Ignored.

    :Returns:

        `numpy.ndarray`
            An array of numbers with the same shape as *array*.

    '''
    ndim = numpy_ndim(array)

    if not ndim and isinstance(array, numpy_ndarray):
        # This necessary because date2num gets upset if you pass
        # it a scalar numpy array
        array = array.item()

#   calendar = getattr(units_out, 'calendar', '')
#   if calendar

#   array = cftime.date2num(
#       array, units=units_out.units,
#       calendar=getattr(units_out, 'calendar', 'standard')
#   )

    array = units_out._utime.date2num(array)

    if not ndim:
        array = numpy_array(array)

    # Round to the nearest millisecond. This is only necessary whilst
    # netCDF4 time functions have an accuracy of no better than 1
    # millisecond (which is still the case at version 1.2.2).
    units = units_out._utime.units
    decimals = 3
    month = False
    year = False

    day = units in cftime.day_units
    if day:
        array *= 86400.0
    else:
        sec = units in cftime.sec_units
        if not sec:
            hr = units in cftime.hr_units
            if hr:
                array *= 3600.0
            else:
                m = units in cftime.min_units
                if m:
                    array *= 60.0
                else:
                    millisec = units in cftime.millisec_units
                    if millisec:
                        decimals = 0
                    else:
                        microsec = units in cftime.microsec_units
                        if microsec:
                            decimals = -3
                        else:
                            month = units in ('month', 'months')
                            if month:
                                array *= (365.242198781 / 12.0) * 86400.0
                            else:
                                year = units in ('year', 'years', 'yr')
                                if year:
                                    array *= 365.242198781 * 86400.0
    # --- End: if
    array = numpy_around(array, decimals, array)

    if day:
        array /= 86400.0
    elif sec:
        pass
    elif hr:
        array /= 3600.0
    elif m:
        array /= 60.0
    elif month:
        array /= (365.242198781 / 12.0) * 86400.0
    elif year:
        array /= 365.242198781 * 86400.0

    if not ndim:
        array = numpy_asanyarray(array)

    return array


def st2rt(array, units_in, units_out, dummy1=None):
    '''The returned array is always independent.

    :Parameters:

        array: numpy array-like of ISO 8601 date-time strings

        units_in: `Units` or `None`

        units_out: `Units`

        dummy1:
            Ignored.

    :Returns:

        `numpy.ndarray`
            An array of floats with the same shape as *array*.

    '''
    array = st2dt(array, units_in)

    ndim = numpy_ndim(array)

    if not ndim and isinstance(array, numpy_ndarray):
        # This necessary because date2num gets upset if you pass
        # it a scalar numpy array
        array = array.item()

    array = units_out._utime.date2num(array)

    if not ndim:
        array = numpy_array(array)

    return array
