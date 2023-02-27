"""General functions useful for `Collapse` functionality."""


def double_precision_dtype(a, default=None, bool_type="i"):
    """Returns the corresponding double precision data type of an array.

    .. versionadded:: 3.14.0

    :Parameters:

        a: `dask.array.Array` or `None`
            The data. If `None` then the value of *default* is
            returned*.

        default: `str`, optional
            If *a* is `None`, then return this data type.

        bool_type: `str`, optional
            The corresponding double data type kind for Boolean
            data. Defaults to ``'i'``, meaning ``'i8'`` is
            returned. Set to ``'f'` to return ``'f8'`` instead.

    :Returns:

        `str`
            The double precision type.

    **Examples**

    >>> for d in (int, 'int32', float, 'float32', bool):
    ...     print(double_precision_dtype(np.array(1, dtype=d)))
    ...
    i8
    i8
    f8
    f8
    i8

    >>> double_precision_dtype(np.array(1, dtype=bool), bool_type='f')
    'f8'
    >>> double_precision_dtype(None, default="i8")
    'i8'

    """
    if a is None:
        return default

    kind = a.dtype.kind
    if kind == "b":
        return bool_type + "8"

    if kind in "fi":
        return kind + "8"

    raise TypeError(f"Can't collapse data with {a.dtype!r}")


def check_input_dtype(a, allowed="fib"):
    """Check that data has a data type allowed by a collapse method.

    The collapse method is assumed to be defined by the calling
    function.

    .. versionadded:: 3.14.0

    :Parameters:

        a: `dask.array.Array`
            The data.

        allowed: `str`, optional
            The data type kinds allowed by the collapse
            method. Defaults to ``'fib'``, meaning that only float,
            integer and Boolean data types are allowed.

    :Returns:

        `None`

    """
    if a.dtype.kind not in allowed:
        from inspect import currentframe

        method = currentframe().f_back.f_code.co_name
        raise TypeError(f"Can't calculate {method} of data with {a.dtype!r}")
