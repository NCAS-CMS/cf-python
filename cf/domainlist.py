from . import mixin
from . import abstract


class DomainList(mixin.FieldDomainList,
                 abstract.ConstructList):
    '''An ordered sequence of TODO

    Each element of a domain list is a domain construct.

    A domain list supports the python list-like operations (such as
    indexing and methods like `!append`). These methods provide
    functionality similar to that of a :ref:`built-in list
    <python:tut-morelists>`. The main difference is that when a domain
    construct element needs to be assesed for equality its
    `~cf.Domain.equals` method is used, rather than the ``==``
    operator.

    .. versionadded:: 3.TODO.0

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
    with the `!match_by_construct` method of the constuct
    elements. For example, to select all constructs that do *not* have
    a "latitude" metadata construct:

       >>> gl = cf.FieldList(
       ...     f for f in fl if not f.match_by_constructs('latitude')
       ... )

    .. versionadded:: 3.TODO.0

    .. seealso: `select`, `__call__`, `select_by_rank`,
                `select_by_property`, `select_by_ncvar`

    :Parameters:

        identities: optional
            Identify the metadata constructs by one or more of

            * A metadata construct identity.

              {{construct selection identity}}

            * The key of a metadata construct (although beware that
              construct keys may differ arbitrarily between list
              elements).
        
            If no identities nor conditions (see the *conditions*
            parameter) are provided then all list elements are
            selected.

            *Parameter example:*
              ``identity='latitude'``

            *Parameter example:*
              ``'T'

            *Parameter example:*
              ``'latitude'``

            *Parameter example:*
              ``'long_name=Cell Area'``

            *Parameter example:*
              ``'cellmeasure1'``

            *Parameter example:*
              ``'measure:area'``

            *Parameter example:*
              ``cf.eq('time')'``

            *Parameter example:*
              ``re.compile('^lat')``

            *Parameter example:*
              ``'domainancillary2', 'longitude'``

        conditions: optional
            Identify the metadata constructs that have any of the
            given identities or construct keys, and whose data satisfy
            conditions.

            A construct identity or construct key (as defined by the
            *identities* parameter) is given as a keyword name and a
            condition on its data is given as the keyword value.

            The condition is satisfied if any of its data values
            equals the value provided.

            If no conditions nor identities (see the *identities*
            parameter) are provided then all list elements are
            selected.

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

    :Returns:

        `bool`
            The matching domain constructs.

    **Examples:**

        TODO

        '''
        return type(self)(
            f for f in self
            if f.match_by_construct(*identities, OR=OR, **conditions)
        )

    # ----------------------------------------------------------------
    # Methods that need extra docstrings
    # ----------------------------------------------------------------
    select_by_property = abstract.ConstructList.select_by_property
    select_by_property.__doc__ += '''

    **Examples:**

    TODO

        '''
    
# --- End: class
