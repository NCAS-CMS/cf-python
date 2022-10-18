from .netcdfarray import NetCDFArray

from ..units import Units

SHOULD THESE BE SUBARRAYS, rather than fragments??

class NetCDFFragmentArray(NetCDFArray):
    """An array stored in a netCDF file."""
    def __getitem__(self, indices):
        array = super().__getitem__(indices)
        
        if array.ndim < self.ndim and array.size == self.size:
            # Add missing size 1 dimensions
            array = array.reshape(self.shape)

        # Get the fragment's units. These are available after any
        # prior call to `super().__getitem__`.
        units = self._get_units()

        if units:
            # Convert array to have parent units 
            parent_units = self.get_parent_units()
            if parent_units and parent_units != units:
                array = Units.conform(array, units, parent_units, inplace=True)

        return array
    
    def _get_units(self):
        """TODO

        The units are only available if there has been a prior call to
        `super().__getitem__`

        :Returns:

            `Units`

        """
        return Units(self._get_component("units"),
                     self._get_component("calendar"))

    def get_parent_units(self):
        """TODODASKDOCS

        .. versionadded:: TODODASKVER

        .. seealso:: `set_parent_units`

        :Returns:
        
            `Units`
                TODODASKDOCS

        **Examples**

        >>> f.set_parent_units(cf.Units('K'))
        >>> f.get_parent_units()
        <Units: K>

        """
        parent_units = self._get_component("parent_units", None)
        if parent_units is None:
            raise ValueError(
                f"Must set parent units on {self.__class__.__name__} "
                "before getting them"
            )

        return 

    def set_parent_units(self, value):
        """TODODASKDOCS

        .. versionadded:: TODODASKVER

        .. seealso:: `get_parent_units`

        :Parameters:

            value: `Units`
                TODODASKDOCS

        :Returns:
        
            `None`

        **Examples**

        >>> f.set_parent_units(cf.Units('K'))
        >>> f.get_parent_units()
        <Units: K>

        """
        self._set_component("parent_units", value, copy=False)
