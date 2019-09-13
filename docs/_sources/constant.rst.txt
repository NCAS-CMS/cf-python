.. currentmodule:: cf

Constants of the :mod:`cf` module
=================================

.. data:: cf.masked

    The :attr:`cf.masked` constant allows data array values to be
    masked by direct assignment. This is consistent with the
    :ref:`behaviour of numpy masked arrays
    <numpy:maskedarray.generic.constructing>`.

    For example, masking every element of a field's data array could
    be done as follows:

    >>> f[...] = cf.masked

    To mask every element of a field's data array whose value is less
    than zero:
    
    >>> f.where(cf.lt(0), cf.masked, i=True)

    .. seealso:: `cf.Field.hardmask`, `cf.Field.subspace`,
                 `cf.Field.where`, `cf.lt`

