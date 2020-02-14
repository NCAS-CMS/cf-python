import cfdm


class Domain(cfdm.Domain):
    '''A domain of the CF data model.

    The domain represents a set of discrete "locations" in what
    generally would be a multi-dimensional space, either in the real
    world or in a model's simulated world. These locations correspond
    to individual data array elements of a field construct

    The domain is defined collectively by the following constructs of
    the CF data model: domain axis, dimension coordinate, auxiliary
    coordinate, cell measure, coordinate reference and domain
    ancillary constructs.

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)


# --- End: class
