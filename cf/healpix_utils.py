"""General functions useful for HEALPix functionality."""

from functools import  partial
import numpy as np
import dask.array as da

def _healpix_info(f):
    """Get information about the HEALPix axis, if there is one.

    .. versionadded:: NEXTVERSION
    
    :Parameters:

        f: `Field` or `Domain`
            The field or domain.
    
    :Returns:

        `dict`
            The information about the HEALPix axis. The dictionary
            will be empty if there is no HEALPix axis.

    **Examples**

    >>> f = cf.example_field(12)
    >>> _healpix_info(f)
    {'coordinate_reference_key': 'coordinatereference0',
     'grid_mapping_name:healpix': <CF CoordinateReference: grid_mapping_name:healpix>,
     'indexing_scheme': 'nested',
     'refinement_level': 1,
     'axis_key': 'domainaxis1',
     'coordinate_key': 'auxiliarycoordinate0',
     'healpix_index': <CF AuxiliaryCoordinate: healpix_index(48) 1>}

    """
    info = {}
    
    cr_key, cr = f.coordinate_reference("grid_mapping_name:healpix",
                                        item=True, default=(None, None))    
    if cr is not None:
        info['coordinate_reference_key'] = cr_key
        info['grid_mapping_name:healpix'] = cr
        parameters = cr.coordinate_conversion.parameters() 
        for param in ('indexing_scheme','refinement_level'):
            value = parameters.get(param)
            if value is not None:
                info[param] = value
        
    hp_key, healpix_index = f.coordinate(
        "healpix_index",
        filter_by_naxes=(1,),
        item=True,
        default=(None, None),
    )
    if healpix_index is not None:
        info['axis_key'] = f.get_data_axes(hp_key)[0]        
        info['coordinate_key'] = hp_key
        info['healpix_index'] = healpix_index
    
    return info


def _healpix_contains_latlon(f, lat, lon):
    """TODOHEALPIX"""

    hp = _healpix_info(f)
    if not hp:
        raise ValueError("TODOHEALPIX")

    try:
        import healpix
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install healpix (https://pypi.org/project/healpix) "
            "to allow the calculation of which cells contain "
            "latitude/longitude locations for a HEALPix grid"
        )
    
    healpix_index = hp.get('healpix_index')
    if healpix_index is None:
        raise ValueError("TODOHEALPIX")

    indexing_scheme = hp.get("indexing_scheme")
    if indexing_scheme is None:
        raise ValueError("TODOHEALPIX")    
          
    if indexing_scheme == 'nested_unique':
        index = []
        healpix_index = healpix_index.array
        orders = healpix.uniq2pix(healpix_index, nest=True)[0]
        orders = np.unique(orders)
        for order in orders:
            nside = healpix.order2nside(order) 
            pix = healpix.ang2pix(nside, lon, lat, nest=True, lonlat=True)
            pix = np.unique(pix)
            pix = healpix._chp.nest2uniq(order, pix, pix)
            index.append(da.where(da.isin(healpix_index, pix))[0])

        index = da.unique(da.concatenate(index, axis=0))
    else:
        refinement_level = hp.get("refinement_level")
        if refinement_level is None:
            raise ValueError("TODOHEALPIX")
    
        nest = indexing_scheme == 'nested'
        nside = healpix.order2nside(refinement_level)
        pix = healpix.ang2pix(nside, lon, lat, nest=nest, lonlat=True)
        pix = np.unique(pix)
        index = da.where(da.isin(healpix_index, pix))[0]

    return index.compute()


    
def del_healpix_coordinate_reference(f):
    """Remove a healpix grid mapping coordinate reference construct.

    A new latitude_longitude grid mapping coordinate reference will be
    created in-place, if required, to store any generic coordinate
    conversion or datum parameters found in the healpix grid mapping
    coordinate reference.

    .. versionadded:: NEXTVERSION

    :Parameters:

        f: `Field` or `Domain`
            The Field or Domain from which to delete the healpix grid
            mapping coordinate reference.

    :Returns:

        `CoordinateReference` or `None`
            The removed healpix grid mapping coordinate reference
            construct, or `None` if there wasn't one.

    """
    cr_key, cr = f.coordinate_reference(
        "grid_mapping_name:healpix", item=True, default=(None, None)
    )
    latlon = f.coordinate_reference(
        "grid_mapping_name:latitude_longitude", default=None
    )

    if cr is not None:
        f.del_construct(cr_key)

        if latlon is None:
            latlon = cr.copy()
            cc = latlon.coordinate_conversion
            cc.del_parameter("grid_mapping_name", None)
            cc.del_parameter("indexing_scheme", None)
            cc.del_parameter("refinement_level", None)
            if cc.parameters() or latlon.datum.parameters():
                # The healpix coordinate reference contains generic
                # coordinate conversion or datum parameters
                latlon.coordinate_conversion.set_parameter(
                    "grid_mapping_name", "latitude_longitude"
                )

                # Remove healpix_index coordinates from the coordinate
                # reference
                for key in f.coordinates(
                    "healpix_index", filter_by_naxes=(1,), todict=True
                ):
                    latlon.del_coordinate(key, None)

                f.set_construct(latlon)

    return cr
