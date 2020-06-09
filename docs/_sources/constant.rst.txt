.. currentmodule:: cf
.. default-role:: obj
		  
**cf constants**
================

----

Version |release| for version |version| of the CF conventions.

.. data:: cf.masked

    The :attr:`cf.masked` constant allows data array values to be
    masked by direct assignment. This is consistent with the
    :ref:`behaviour of numpy masked arrays
    <numpy:maskedarray.generic.constructing>`.

    .. seealso:: `cf.Field.hardmask`, `cf.Field.subspace`,
                 `cf.Field.where`

    **Examples**:

    Masking every element of a field construct's data could be done as
    follows:

    >>> f[...] = cf.masked

    To mask every element of a field construct's data whose value is
    less than zero:
    
    >>> g = f.where(cf.lt(0), cf.masked)


