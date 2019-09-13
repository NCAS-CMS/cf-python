#include <string.h>

#include "umfileint.h"

/*
 * Functions relating to time and z axes.
 *
 * These functions have very similar content for Z and T axes, because in fact 
 * the Z_axis and T_axis struct each only contains a list of values.  In cdunifpp,
 * there are other elements (e.g. a time axis origin) that are used when comparing 
 * axes (maybe unnecessarily).  In umread, we do not even bother to compare axes 
 * because axes are not returned to the caller: they are just used to evaluate the 
 * shape of the variable (nz, nt) and ensure that it is regular.
 */

T_axis *new_t_axis(List *heaplist)
{
  T_axis *t_axis;
  CKP(  t_axis = malloc_(sizeof(T_axis), heaplist)  );
  t_axis->values = list_new(heaplist);
  return t_axis;
  ERRBLKP;
}

int free_t_axis(T_axis *t_axis, List *heaplist)
{
  CKI(  list_free(t_axis->values, 1, heaplist)  );
  CKI(  free_(t_axis, heaplist)   );
  return 0;
  ERRBLKI;
}


Z_axis *new_z_axis(List *heaplist)
{
  Z_axis *z_axis;
  CKP(  z_axis = malloc_(sizeof(Z_axis), heaplist)  );
  z_axis->values = list_new(heaplist);
  return z_axis;
  ERRBLKP;
}

int free_z_axis(Z_axis *z_axis, List *heaplist)
{
  CKI(  list_free(z_axis->values, 1, heaplist)  );
  CKI(  free_(z_axis, heaplist)  );
  return 0;
  ERRBLKI;
}


int t_axis_add(T_axis *t_axis, const Time *time, 
	       int *index_return, List *heaplist)
{
  Time *timecopy;

  CKP(   timecopy = dup_(time, sizeof(Time), heaplist)   );
  return list_add_or_find(t_axis->values, &timecopy, compare_times, 0, 
			  free_, index_return, heaplist);
  ERRBLKI;
}



int z_axis_add(Z_axis *z_axis, const Level *lev, 
	      int *index_return, List *heaplist)
{
  Level *levcopy;

  CKP(   levcopy = dup_(lev, sizeof(Level), heaplist)   );
  return list_add_or_find(z_axis->values, &levcopy, compare_levels, 0, 
			  free_, index_return, heaplist);
  ERRBLKI;
}

