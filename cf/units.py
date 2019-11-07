from  cfunits import Units as cfUnits


class Units:
    '''Store, combine and compare physical units and convert numeric
    values to different units.
    
    This is a convenience class that creates a `cfunits.Units`
    instance (https://ncas-cms.github.io/cfunits). For further
    details, type
    
       >>> help(cf.Units())

    '''
    def __new__(cls, *args, **kwargs):
        return cfUnits(*args, **kwargs)


    @staticmethod
    def conform(*args, **kwargs):
        return cfUnits.conform(*args, **kwargs)

    
#--- End: class
