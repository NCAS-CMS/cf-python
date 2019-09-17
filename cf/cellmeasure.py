import cfdm

from . import mixin

from .functions import _DEPRECATION_ERROR_KWARGS


class CellMeasure(mixin.PropertiesData,
                  cfdm.CellMeasure):
    '''A cell measure construct of the CF data model.

A cell measure construct provides information that is needed about the
size or shape of the cells and that depends on a subset of the domain
axis constructs. Cell measure constructs have to be used when the size
or shape of the cells cannot be deduced from the dimension or
auxiliary coordinate constructs without special knowledge that a
generic application cannot be expected to have.

The cell measure construct consists of a numeric array of the metric
data which spans a subset of the domain axis constructs, and
properties to describe the data. The cell measure construct specifies
a "measure" to indicate which metric of the space it supplies,
e.g. cell horizontal areas, and must have a units property consistent
with the measure, e.g. square metres. It is assumed that the metric
does not depend on axes of the domain which are not spanned by the
array, along which the values are implicitly propagated. CF-netCDF
cell measure variables correspond to cell measure constructs.

**NetCDF interface**

The netCDF variable name of the construct may be accessed with the
`nc_set_variable`, `nc_get_variable`, `nc_del_variable` and
`nc_has_variable` methods.

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)


    @property
    def ismeasure(self): 
        '''Always True.

    .. seealso:: `isauxiliary`, `isdimension`

    **Examples:**

    >>> c.ismeasure
    True
        
        '''
        return True


    @property
    def measure(self):
        '''TODO
        '''
        return self.get_measure(default=AttributeError())
    @measure.setter
    def measure(self, value): self.set_measure(value)
    @measure.deleter
    def measure(self):        self.del_measure(default=AttributeError())

    def identity(self, default='', strict=None, nc_only=False,
                 relaxed_identity=None): 
        '''TODO        
        ''' 
        if relaxed_identity:
            _DEPRECATION_ERROR_KWARGS(self, 'identity',
                                      {'relaxed_identity':  relaxed_identity},
                                      "Use the 'strict' keyword") # pragma: no cover
            
        if not nc_only:
            n = self.get_measure(default=None)
            if n is not None:
                return 'measure:'+n
        #--- End: if

        return super().identity(default=default, strict=strict,
                                nc_only=nc_only)


#--- End: class
