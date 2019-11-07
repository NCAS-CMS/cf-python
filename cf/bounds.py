import cfdm

from . import mixin


class Bounds(mixin.PropertiesData,
             cfdm.Bounds):
    '''A cell bounds component of a coordinate or domain ancillary
    construct of the CF data model.
    
    An array of cell bounds spans the same domain axes as its
    coordinate array, with the addition of an extra dimension whose
    size is that of the number of vertices of each cell. This extra
    dimension does not correspond to a domain axis construct since it
    does not relate to an independent axis of the domain. Note that,
    for climatological time axes, the bounds are interpreted in a
    special way indicated by the cell method constructs.
    
    In the CF data model, a bounds component does not have its own
    properties because they can not logically be different to those of
    the coordinate construct itself. However, it is sometimes desired
    to store attributes on a CF-netCDF bounds variable, so it is also
    allowed to provide properties to a bounds component.
    
    **NetCDF interface**
    
    The netCDF variable name of the bounds may be accessed with the
    `nc_set_variable`, `nc_get_variable`, `nc_del_variable` and
    `nc_has_variable` methods.
    
    The name of the trailing netCDF dimension spanned by bounds (which
    does not correspond to a domain axis construct) may be accessed
    with the `nc_set_dimension`, `nc_get_dimension`,
    `nc_del_dimension` and `nc_has_dimension` methods.

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)

    
    def contiguous(self, overlap=True, direction=None):
        '''Return True if the bounds are contiguous.

    Bounds are contiguous if the cell boundaries match up, or overlap,
    with the boundaries of adjacent cells.
    
    In general, it is only possible for 1 or 0 variable dimensional
    variables with bounds to be contiguous, but size 1 variables with
    any number of dimensions are always contiguous.
    
    An exception is raised if the variable is multdimensional and has
    more than one element.
    
    .. versionadded:: 2.0

        '''
        data = self.get_data(None)
        if data is None:
            return False    

        nbounds = data.shape[-1]

        if data.size == nbounds:
            return True

        if nbounds == 4 and data.ndim == 3:
            if overlap:
                raise ValueError(
                    "overlap=True and can't tell if 2-d bounds are contiguous")
            
            bnd = data.array
            for j in range(data.shape[0] - 1):
                for i in range(data.shape[1] - 1):
                    # Check cells (j, i) and cells (j, i+1) are contiguous
                    if (bnd[j,i,1] != bnd[j,i+1,0] or 
                        bnd[j,i,2] != bnd[j,i+1,3]):
                        return False
                    
                    # Check cells (j, i) and (j+1, i) are contiguous
                    if (bnd[j,i,3] != bnd[j+1,i,0] or 
                        bnd[j,i,2] != bnd[j+1,i,1]):
                        return False
            #--- End: for
            
            return True

        if nbounds > 2 or data.ndim > 2:
            raise ValueError("Can't tell if multidimensional bounds are contiguous")
        
        if not overlap: 
            return data[1:, 0].equals(data[:-1, 1])
        else:
            if direction is None:
                b = data[(0,)*(data.ndim-1)].array
                direction =  b.item(0,) < b.item(1,)
        
            if direction:
                return (data[1:, 0] <= data[:-1, 1]).all()
            else:
                return (data[1:, 0] >= data[:-1, 1]).all()


#--- End: class
