from numpy import empty   as numpy_empty
from numpy import full    as numpy_full
from numpy import load    as numpy_load
from numpy import ndarray as numpy_ndarray
from numpy import save    as numpy_save

from numpy.ma import array      as numpy_ma_array
from numpy.ma import is_masked  as numpy_ma_is_masked
from numpy.ma import masked_all as numpy_ma_masked_all

from ..functions import parse_indices, get_subspace #, abspath
from ..functions import inspect as cf_inspect
from ..constants import CONSTANTS

_debug = False

from . import abstract

class FilledArray(abstract.Array):
    '''**Initialization**

:Parameters:

    dtype : numpy.dtype
        The numpy data type of the data array.

    ndim : int
        Number of dimensions in the data array.

    shape : tuple
        The data array's dimension sizes.

    size : int
        Number of elements in the data array.

    fill_value : scalar, optional

    masked_all: `bool`

    '''
    def __init__(self, dtype=None, ndim=None, shape=None, size=None,
                 fill_value=None, masked_all=False):
        '''
        '''
        super().__init__(dtype=dtype, ndim=ndim, shape=shape,
                         size=size, fill_value=fill_value,
                         masked_all=masked_all)
    #--- End: def

    def __getitem__(self, indices):
        '''x.__getitem__(indices) <==> x[indices]

Returns a numpy array.

        '''
        if indices is Ellipsis:
            array_shape = self.shape
        else:
            array_shape = []
            for index in parse_indices(self.shape, indices):
                if isinstance(index, slice):                
                    step = index.step
                    if step == 1:
                        array_shape.append(index.stop - index.start)
                    elif step == -1:
                        stop = index.stop
                        if stop is None:
                            array_shape.append(index.start + 1)
                        else:
                            array_shape.append(index.start - index.stop)
                    else:                    
                        stop = index.stop
                        if stop is None:
                            stop = -1
                           
                        a, b = divmod(stop - index.start, step)
                        if b:
                            a += 1
                        array_shape.append(a)
                else:
                    array_shape.append(len(index))
            #-- End: for
        #-- End: if

        if self.masked_all():
            return numpy_ma_masked_all(array_shape, dtype=self.dtype)
        elif self.fill_value is not None:
            return numpy_full(array_shape, fill_value=self.fill_value(),
                              dtype=self.dtype)
        else:
            return numpy_empty(array_shape, dtype=self.dtype)
    #--- End: def

#    def __repr__(self):
#        '''
#
#x.__repr__() <==> repr(x)
#
#'''
#        return "<CF {0}: shape={1}, dtype={2}, fill_value={3}>".format(
#            self.__class__.__name__, self.shape, self.dtype, self.fill_value)
#    #--- End: def
#
#    def __str__(self):
#        '''
#
#x.__str__() <==> str(x)
#
#'''
#        return repr(self)
#    #--- End: def

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def dtype(self):
        '''Data-type of the data elements.
         
**Examples:**

>>> a.dtype
dtype('float64')
>>> print(type(a.dtype))
<type 'numpy.dtype'>

        '''
        return self._get_component('dtype')
    #--- End: def
    
    @property
    def ndim(self):
        '''Number of array dimensions
        
**Examples:**

>>> a.shape
(73, 96)
>>> a.ndim
2
>>> a.size
7008

>>> a.shape
(1, 1, 1)
>>> a.ndim
3
>>> a.size
1

>>> a.shape
()
>>> a.ndim
0
>>> a.size
1
        '''
        return self._get_component('ndim')
    #--- End: def
    
    @property
    def shape(self):
        '''Tuple of array dimension sizes.

**Examples:**

>>> a.shape
(73, 96)
>>> a.ndim
2
>>> a.size
7008

>>> a.shape
(1, 1, 1)
>>> a.ndim
3
>>> a.size
1

>>> a.shape
()
>>> a.ndim
0
>>> a.size
1
        '''
        return self._get_component('shape')
    #--- End: def
    
    @property
    def size(self):        
        '''Number of elements in the array.

**Examples:**

>>> a.shape
(73, 96)
>>> a.size
7008
>>> a.ndim
2

>>> a.shape
(1, 1, 1)
>>> a.ndim
3
>>> a.size
1

>>> a.shape
()
>>> a.ndim
0
>>> a.size
1

        '''
        return self._get_component('size')
    #--- End: def

    def fill_value(self):        
        '''TODO        '''
        return self._get_component('fill_value')
    #--- End: def

    def masked_all(self):        
        '''TODO        '''
        return self._get_component('masked_all')
    #--- End: def

    @property
    def array(self):
        '''TODO
'''
        return self[...]

    def reshape(self, newshape):
        '''
'''
        new = self.copy()        
        new.shape = newshape
        new.ndim  = len(newshape)
        return new
    #--- End: def

    def resize(self, newshape):
        '''
'''
        self.shape = newshape
        self.ndim  = len(newshape)
    #--- End: def

    def view(self):
        '''
'''
        return self[...]
    #--- End: def

#--- End: class
