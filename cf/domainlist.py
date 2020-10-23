import logging

import cfdm

#from .mixin_container import Container
from . import abstract

from .functions import (_DEPRECATION_ERROR,
                        _DEPRECATION_ERROR_KWARGS,
                        _DEPRECATION_ERROR_METHOD,
                        _DEPRECATION_ERROR_DICT)

from .decorators import (_deprecated_kwarg_check,
                         _manage_log_level_via_verbosity)


logger = logging.getLogger(__name__)


class DomainList(abstract.ConstructList):
    '''An ordered sequence of TODO

    Each element of a field list is a field construct.

    A field list supports the python list-like operations (such as
    indexing and methods like `!append`).

    >>> fl = cf.FieldList()
    >>> len(fl)
    0
    >>> f
    <CF Field: air_temperaturetime(12), latitude(73), longitude(96) K>
    >>> fl = cf.FieldList(f)
    >>> len(fl)
    1
    >>> fl = cf.FieldList([f, f])
    >>> len(fl)
    2
    >>> fl = cf.FieldList(cf.FieldList([f] * 3))
    >>> len(fl)
    3
    >>> len(fl + fl)
    6

    These methods provide functionality similar to that of a
    :ref:`built-in list <python:tut-morelists>`. The main difference
    is that when a field element needs to be assesed for equality its
    `~cf.Field.equals` method is used, rather than the ``==``
    operator.

    '''
    def __init__(self, domains=None):
        '''**Initialization**

    :Parameters:

        domains: (sequence of) `Domain`, optional
             Create a new list with these domain constructs.

        '''
        super().__init__(constructs=domains)

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def select_by_construct(self, *identities, OR=False, **conditions):
        '''Select list elements by metadata constructs.

    To find the inverse of the selection, use a list comprehension
    with the `!match_by_construct` method of the constucts. For
    example, to select all field constructs that do *not* have a
    "latitude" metadata construct:

       >>> gl = cf.FieldList(
       ...     f for f in fl if not f.match_by_constructs('latitude')
       ... )

    .. note:: The API changed at version 3.1.0

    .. versionadded:: 3.0.0

    .. seealso: `select`, `__call__`, `select_by_rank`,
                `select_by_property`, `select_by_ncvar`

    :Parameters:

        identities: optional
            Identify the metadata constructs that have any of the
            given identities or construct keys.

            A construct identity is specified by a string
            (e.g. ``'latitude'``, ``'long_name=time'``,
            ``'ncvar%lat'``, etc.); or a compiled regular expression
            (e.g. ``re.compile('^atmosphere')``) that selects the
            relevant constructs whose identities match via
            `re.search`.

            Each construct has a number of identities, and is selected
            if any of them match any of those provided. A construct's
            identities are those returned by its `!identities`
            method. In the following example, the construct ``x`` has
            six identities:

               >>> x.identities()
               ['time',
                'long_name=Time',
                'foo=bar',
                'standard_name=time',
                'ncvar%t',
                'T']

            A construct key may optionally have the ``'key%'``
            prefix. For example ``'dimensioncoordinate2'`` and
            ``'key%dimensioncoordinate2'`` are both acceptable keys.

            Note that in the output of a `print` call or `!dump`
            method, a construct is always described by one of its
            identities, and so this description may always be used as
            an *identity* argument.

            *Parameter example:*
              ``'measure:area'``

            *Parameter example:*
              ``'latitude'``

            *Parameter example:*
              ``'long_name=Longitude'``

            *Parameter example:*
              ``'domainancillary2', 'ncvar%areacello'``

        conditions: optional
            Identify the metadata constructs that have any of the
            given identities or construct keys, and whose data satisfy
            conditions.

            A construct identity or construct key (as defined by the
            *identities* parameter) is given as a keyword name and a
            condition on its data is given as the keyword value.

            The condition is satisfied if any of its data values
            equals the value provided.

            *Parameter example:*
              ``longitude=180.0``

            *Parameter example:*
              ``time=cf.dt('1959-12-16')``

            *Parameter example:*
              ``latitude=cf.ge(0)``

            *Parameter example:*
              ``latitude=cf.ge(0), air_pressure=500``

            *Parameter example:*
              ``**{'latitude': cf.ge(0), 'long_name=soil_level': 4}``

        OR: `bool`, optional
            If True then return `True` if at least one metadata
            construct matches at least one of the criteria given by
            the *identities* or *conditions* arguments. By default
            `True` is only returned if the field constructs matches
            each of the given criteria.

        mode: deprecated at version 3.1.0
            Use the *OR* parameter instead.

        constructs: deprecated at version 3.1.0

    :Returns:

        `bool`
            The matching field constructs.

    **Examples:**

        TODO

        '''
        return type(self)(
            f for f in self
            if f.match_by_construct(*identities, OR=OR, **conditions)
        )

    select_by_property = abstract.ConstructList.select_by_property
    select_by_property.__doc__ += '''

    **Examples:**

    TODO

        '''
    
    def select_by_rank(self, *ranks):
        '''Select list elements by the number of domain axis constructs.

    .. versionadded:: 3.0.0

    .. seealso: `select`, `__call__`, `select_by_construct`,
                `select_by_property`, `select_by_ncvar`

    :Parameters:

        ranks: optional
            Define conditions on the number of domain axis constructs.

            A condition is one of:

              * `int`
              * a `Query` object

            The condition is satisfied if the number of domain axis
            constructs equals the condition value.

            *Parameter example:*
              To see if the field construct has 4 domain axis
              constructs: ``4``

            *Parameter example:*
              To see if the field construct has at least 3 domain axis
              constructs: ``cf.ge(3)``

    :Returns:

        `bool`
            The matching field constructs.

    **Examples:**

        TODO

        '''

        return type(self)(f for f in self if f.match_by_rank(*ranks))

# --- End: class
