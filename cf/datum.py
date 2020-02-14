import cfdm


class Datum(cfdm.Datum):
    '''A datum component of a coordinate reference construct of the CF
    data model.

    A datum is a complete or partial definition of the zeroes of the
    dimension and auxiliary coordinate constructs which define a
    coordinate system.

    The datum may contain the definition of a geophysical surface
    which corresponds to the zero of a vertical coordinate construct,
    and this may be required for both horizontal and vertical
    coordinate systems.

    Elements of the datum not specified may be implied by the
    properties of the dimension and auxiliary coordinate constructs
    referenced by the `cf.CoordinateReference` instance that contains
    the datum.

    .. versionadded:: 3.0.0

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)


# --- End: class
