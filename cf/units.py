from  cfunits import Units as cfUnits


class Units:
    '''Store, combine and compare physical units and convert numeric
    values to different units.
    
    This is a convenience class that creates a `cfunits.Units`
    instance.

    The full documentation is available with ``help(cf.Units())``.

    '''
    def __new__(cls, *args, **kwargs):
        return cfUnits(*args, **kwargs)


    @staticmethod
    def conform(*args, **kwargs):
        return cfUnits.conform(*args, **kwargs)

    
#--- End: class
