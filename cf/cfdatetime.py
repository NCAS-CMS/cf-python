import datetime
from functools import partial

import cftime
import numpy as np

from .functions import _DEPRECATION_ERROR_CLASS
from .functions import size as cf_size

default_calendar = "gregorian"

# --------------------------------------------------------------------
# Mapping of CF calendars to cftime date-time objects
# --------------------------------------------------------------------
_datetime_object = {
    ("",): partial(cftime.datetime, calendar=""),
    (None, "gregorian", "standard", "none"): cftime.DatetimeGregorian,
    ("proleptic_gregorian",): cftime.DatetimeProlepticGregorian,
    ("360_day",): cftime.Datetime360Day,
    ("noleap", "365_day"): cftime.DatetimeNoLeap,
    ("all_leap", "366_day"): cftime.DatetimeAllLeap,
    ("julian",): cftime.DatetimeJulian,
}

canonical_calendar = {
    None: "standard",
    "gregorian": "standard",
    "standard": "standard",
    "proleptic_gregorian": "proleptic_gregorian",
    "julian": "julian",
    "noleap": "noleap",
    "365_day": "noleap",
    "all_366_day": "all_leap",
    "all_leap": "all_leap",
    "": "",
    "none": "",
}


_calendar_map = {None: "gregorian"}


class Datetime(cftime.datetime):
    """A date-time object which supports CF calendars.

    Deprecated at version 3.0.0. Use function 'cf.dt' to create date-
    time objects instead.

    """

    def __init__(
        self,
        year,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        dayofwk=-1,
        dayofyr=1,
        calendar=None,
    ):
        """**Initialisation**"""
        _DEPRECATION_ERROR_CLASS(
            "Datetime",
            "Use function 'cf.dt' to create date-time objects instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover


def elements(x):
    return x.timetuple()[:6]


def dt(
    arg, month=1, day=1, hour=0, minute=0, second=0, microsecond=0, calendar=""
):
    """Return a date-time object for a date and time according to a
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

            * `datetime.datetime` or `cftime.datetime`. A new date-time
              object is returned for the given date-time.


        calendar: `str`, optional
            The calendar for the date-time. By default the Gregorian
            calendar is used.

            *Parameter example:*
              ``calendar='360_day'``

    :Returns:

        `cftime.datetime`
            The new date-time object.

    **Examples**

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

    """
    if isinstance(arg, str):
        (year, month, day, hour, minute, second, microsecond) = st2elements(
            arg
        )

    elif isinstance(arg, cftime.datetime):
        (year, month, day, hour, minute, second, microsecond) = (
            arg.year,
            arg.month,
            arg.day,
            arg.hour,
            arg.minute,
            arg.second,
            arg.microsecond,
        )
        if calendar == "":
            calendar = arg.calendar

    elif isinstance(arg, datetime.datetime):
        (year, month, day, hour, minute, second) = arg.timetuple()[:6]
        microsecond = arg.microsecond
        if calendar == "":
            calendar = default_calendar

    else:
        year = arg

    #    calendar=_calendar_map.get(calendar, calendar)
    #
    #    return cftime.datetime(year, month, day, hour, minute, second,
    #                           microsecond, calendar=calendar)

    for calendars, datetime_cls in _datetime_object.items():
        if calendar in calendars:
            return datetime_cls(
                year, month, day, hour, minute, second, microsecond
            )

    raise ValueError(
        f"Can't create date-time object with unknown calendar {calendar!r}"
    )


def dt_vector(
    arg, month=1, day=1, hour=0, minute=0, second=0, microsecond=0, calendar=""
):
    """Return a 1-d array of date-time objects.

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
            1-d array of date-time objects.

    **Examples**

    TODO

    """
    arg = np.array(arg)
    month = np.array(month)
    day = np.array(day)
    hour = np.array(hour)
    minute = np.array(minute)
    second = np.array(second)
    microsecond = np.array(microsecond)

    ndim = max(map(np.ndim, (month, day, hour, minute, second, microsecond)))

    if ndim > 1:
        raise ValueError(
            "If set, the 'month', 'day', 'hour', 'minute', 'second', "
            "'microsecond' parameters must be scalar or 1-d"
        )

    if arg.ndim > 2:
        raise ValueError(
            "The 'arg' parameter must be scalar, 1-d or 2-d. " f"Got: {arg!r}"
        )

    sizes = set(
        map(cf_size, (arg, month, day, hour, minute, second, microsecond))
    )

    if len(sizes) == 1 and 1 in sizes:
        # All arguments are scalars or size 1
        out = dt(
            arg.item(),
            month.item(),
            day.item(),
            hour.item(),
            minute.item(),
            second.item(),
            microsecond.item(),
            calendar=calendar,
        )
        if ndim >= 1:
            out = [out]

        out = np.array(out)

        if not out.ndim:
            out = np.expand_dims(out, 0)

        return out

    # Still here?
    if arg.ndim == 2 and arg.shape[1] > 7:
        raise ValueError(
            "The size of the second dimension of 'arg' must be less than 8. "
            f"Got: {arg.shape[1]!r}"
        )

    if arg.ndim == 1:
        if arg.dtype.kind in "UOS":
            out = [dt(a, calendar=calendar) for a in arg]
        else:
            if len(sizes) > 2:
                raise ValueError(
                    "The 'arg', 'month', 'day', 'hour', 'minute', 'second', "
                    "'microsecond' parameters have incompatible sizes."
                    "At least two of them have different sizes greater than 1"
                )

            if len(sizes) == 2 and 1 not in sizes:
                raise ValueError(
                    "The 'arg', 'month', 'day', 'hour', 'minute', 'second', "
                    "'microsecond' parameters have incompatible sizes. "
                    "At least two of them have different sizes greater than 1"
                )

            x = np.empty((max(sizes), 7), dtype=int)
            x[:, 0] = arg
            x[:, 1] = month
            x[:, 2] = day
            x[:, 3] = hour
            x[:, 4] = minute
            x[:, 5] = second
            x[:, 6] = microsecond
            arg = x

            out = [dt(*args, calendar=calendar) for args in arg]
    else:
        out = [dt(*args, calendar=calendar) for args in arg]

    out = np.array(out)

    if not out.ndim:
        out = np.expand_dims(out, 0)

    return out


def st2dt(array, units_in=None, dummy0=None, dummy1=None):
    """The returned array is always independent.

    :Parameters:

        array: numpy array-like

        units_in: `Units`, optional

        dummy0: optional
            Ignored.

        dummy1: optional
            Ignored.

    :Returns:

        `numpy.ndarray`
            An array of `cftime.datetime` objects with the same shape
            as *array*.

    **Examples**

    """
    func = partial(st2datetime, calendar=units_in._calendar)
    return np.vectorize(func, otypes=[object])(array)


def st2datetime(date_string, calendar=None):
    """Parse an ISO 8601 date-time string into a `cftime` object.

    :Parameters:

        date_string: `str`

    :Returns:

        `cftime.datetime`

    """
    if date_string.count("-") != 2:
        raise ValueError(
            "Input date-time string must contain at least a year, a month "
            "and a day"
        )

    x = cftime._parse_date(date_string)
    if len(x) == 7:
        year, month, day, hour, minute, second, utc_offset = x
        microsecond = 0
    else:
        year, month, day, hour, minute, second, microsecond, utc_offset = x

    if utc_offset:
        raise ValueError("Can't specify a time offset from UTC")

    #    return Datetime(year, month, day, hour, minute, second)
    return dt(
        year, month, day, hour, minute, second, microsecond, calendar=calendar
    )


def st2elements(date_string):
    """Parse an ISO 8601 date-time string into a `cftime` object.

    :Parameters:

        date_string: `str`

    :Returns:

        `tuple`

    """
    if date_string.count("-") != 2:
        raise ValueError(
            "Input date-time string must contain at least a year, a month "
            "and a day"
        )

    x = cftime._parse_date(date_string)
    if len(x) == 7:
        year, month, day, hour, minute, second, utc_offset = x
        microsecond = 0
    else:
        year, month, day, hour, minute, second, microsecond, utc_offset = x

    if utc_offset:
        raise ValueError("Can't specify a time offset from UTC")

    return (year, month, day, hour, minute, second, microsecond)


def rt2dt(array, units_in, units_out=None, dummy1=None):
    """Convert reference times to date-time objects.

    The returned array is always independent.

    .. seealso:: `dt2rt`

    :Parameters:

        array: numpy array-like

        units_in: `Units`

        units_out: *optional*
            Ignored.

        dummy1:
            Ignored.

    :Returns:

        `numpy.ndarray`
            An array of `cftime.datetime` objects with the same shape
            as *array*.

    **Examples**

    >>> print(
    ...   cf.cfdatetime.rt2dt(
    ...     np.ma.array([0, 685.5], mask=[True, False]),
    ...     units_in=cf.Units('days since 2000-01-01')
    ...   )
    ... )
    [--
     cftime.DatetimeGregorian(2001, 11, 16, 12, 0, 0, 0, has_year_zero=False)]

    """
    ndim = np.ndim(array)
    if not ndim and np.ma.is_masked(array):
        # num2date has issues with scalar masked arrays with a True
        # mask
        return np.ma.masked_all((), dtype=object)

    units = units_in.units
    calendar = getattr(units_in, "calendar", "standard")

    array = cftime.num2date(
        array, units, calendar, only_use_cftime_datetimes=True
    )

    if not isinstance(array, np.ndarray):
        array = np.array(array, dtype=object)

    return array


def dt2Dt(x, calendar=None):
    """Convert a datetime.datetime object to a cf.Datetime object."""
    if not x:
        return False

    return dt(x, calendar=calendar)


def dt2rt(array, units_in, units_out, dummy1=None):
    """Return numeric time values from datetime objects.

    .. seealso:: `rt2dt`

    :Parameters:

        array: numpy array-like of date-time objects
            The datetime objects must be in UTC with no time-zone
            offset.

      units_in:
            Ignored.

        units_out: `Units`
            The units of the numeric time values. If there is a
            time-zone offset in *units_out*, it will be applied to the
            returned numeric values.

        dummy1:
            Ignored.

    :Returns:

        `numpy.ndarray`
            An array of numbers with the same shape as *array*.

    **Examples**

    >>> print(
    ...   cf.cfdatetime.dt2rt(
    ...     np.ma.array([0, cf.dt('2001-11-16 12:00')], mask=[True, False]),
    ...     None,
    ...     units_out=cf.Units('days since 2000-01-01')
    ...   )
    ... )
    [-- 685.5]

    """
    isscalar = not np.ndim(array)

    array = cftime.date2num(
        array, units=units_out.units, calendar=units_out._utime.calendar
    )

    if isscalar:
        if array is np.ma.masked:
            array = np.ma.masked_all(())
        else:
            array = np.asanyarray(array)

    return array


def st2rt(array, units_in, units_out, dummy1=None):
    """The returned array is always independent.

    :Parameters:

        array: numpy array-like of ISO 8601 date-time strings

        units_in: `Units` or `None`

        units_out: `Units`

        dummy1:
            Ignored.

    :Returns:

        `numpy.ndarray`
            An array of floats with the same shape as *array*.

    """
    array = st2dt(array, units_in)
    #    array = units_out._utime.date2num(array)
    array = cftime.date2num(
        array, units=units_out.units, calendar=units_out._utime.calendar
    )

    if not np.ndim(array):
        array = np.asanyarray(array)

    return array
