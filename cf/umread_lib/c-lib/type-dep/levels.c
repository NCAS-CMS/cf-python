#include "umfileint.h"


int lev_set(Level *lev, const Rec *rec) 
{  
  lev->type = level_type(rec);

  switch (lev->type) 
    {
    case hybrid_height_lev_type:
      lev->values.hybrid_height.a = RLOOKUP(rec, INDEX_BLEV);
      lev->values.hybrid_height.b = RLOOKUP(rec, INDEX_BHLEV);
#ifdef BDY_LEVS
      lev->values.hybrid_height.ubdy_a = RLOOKUP(rec, INDEX_BULEV);
      lev->values.hybrid_height.ubdy_b = RLOOKUP(rec, INDEX_BHULEV);
      lev->values.hybrid_height.lbdy_a = RLOOKUP(rec, INDEX_BRLEV);
      lev->values.hybrid_height.lbdy_b = RLOOKUP(rec, INDEX_BHRLEV);
#endif
      break;

    case hybrid_sigmap_lev_type:
      lev->values.hybrid_sigmap.a = RLOOKUP(rec, INDEX_BHLEV);
      lev->values.hybrid_sigmap.b = RLOOKUP(rec, INDEX_BLEV);
#ifdef BDY_LEVS
      lev->values.hybrid_sigmap.ubdy_a = RLOOKUP(rec, INDEX_BHULEV);
      lev->values.hybrid_sigmap.ubdy_b = RLOOKUP(rec, INDEX_BULEV);
      lev->values.hybrid_sigmap.lbdy_a = RLOOKUP(rec, INDEX_BHRLEV);
      lev->values.hybrid_sigmap.lbdy_b = RLOOKUP(rec, INDEX_BRLEV);
#endif
      break;

    case pseudo_lev_type:
      lev->values.pseudo.index = LOOKUP(rec, INDEX_LBUSER5);
      break;
      
    default:
      if (RLOOKUP(rec, INDEX_BLEV) == 0 
	  && LOOKUP(rec, INDEX_LBLEV) != 9999
	  && LOOKUP(rec, INDEX_LBLEV) != 8888) 
	lev->values.misc.level = LOOKUP(rec, INDEX_LBLEV);
      else
	lev->values.misc.level = RLOOKUP(rec, INDEX_BLEV);
#ifdef BDY_LEVS
      lev->values.misc.ubdy_level = RLOOKUP(rec, INDEX_BULEV);
      lev->values.misc.lbdy_level = RLOOKUP(rec, INDEX_BRLEV);
#endif
      break;
    }
  return 0;
}


Lev_type level_type(const Rec *rec) 
{
  if (LOOKUP(rec, INDEX_LBUSER5) != 0 
      && LOOKUP(rec, INDEX_LBUSER5) != INT_MISSING_DATA)
    return pseudo_lev_type;
  
  switch (LOOKUP(rec, INDEX_LBVC))
    {
      /*
       *         1  Height (m)              8  Pressure (mb)
       *         9  Hybrid co-ordinates    10  Sigma (=p/p*)
       *         128  Mean sea level      129  Surface
       *         130  Tropopause level    131  Maximum wind level
       *         132  Freezing level      142  Upper hybrid level
       *         143  Lower hybrid level  176  Latitude (deg)
       *         177  Longitude (deg)
       */
      /* also new dynamics:  65  hybrid height */
      
    case 1:
      return height_lev_type;
    case 2:
      return depth_lev_type;
    case 5:
      return boundary_layer_top_lev_type;
    case 6:
      return soil_lev_type;
    case 8:
      return pressure_lev_type;
    case 9:
      return hybrid_sigmap_lev_type;
    case 65:
      return hybrid_height_lev_type;
    case 128:
      return mean_sea_lev_type;
    case 129:
      return surface_lev_type;
    case 130:
      return tropopause_lev_type;
    case 133:
      return top_of_atmos_lev_type;
    default:
      return other_lev_type;
    }
}
