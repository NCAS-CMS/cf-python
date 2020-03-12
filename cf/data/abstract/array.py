import cfdm


class Array(cfdm.Array):
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

    .. versionadded:: 3.0.0

        '''
        return super().__repr__().replace('<', '<CF ', 1)


# --- End: class
