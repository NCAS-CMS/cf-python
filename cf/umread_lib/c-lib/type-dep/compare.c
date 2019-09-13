/*
 * COMPARISON FUNCTIONS.
 *
 * NOTE: functions which take arguments of type void* (except for
 *   compare_ptrs) are designed to be used with generic routines:
 *   compare_records is envisaged for use with qsort; several other functions
 *   are envisaged for use with compare_lists (below).
 *
 *   In these cases if supplying pointers directly to the relevant structures,
 *   need to generate an extra level of pointer with "&" syntax.
 *
 * But not all functions below are like that.  Don't assume functions can be
 * used analogously without first examining the argument lists.
 */

/* The code profiler suggests that compare_ints and compare_reals are
 * candidates for inlining; however, unfortunately this sometimes gets
 * compiled with c89 which doesn't support inline functions.  Use a #define
 * for compare_ints.  compare_reals, which is more awkward to #define, is just
 * going to have to stay as it is for now (it's called less often).
 */

#include <math.h>

#include "umfileint.h"

#define compare_ints(a, b) ((a) < (b) ? (-1) : (a) > (b) ? 1 : 0)

/*
 * static int compare_ints(INTEGER a, INTEGER b)
 * {
 *   if (a<b) return -1;
 *   if (a>b) return 1;
 *   return 0;
 * }
 */

static int compare_reals(REAL a, REAL b)
{
  REAL delta;

  /* first test for special case (unnecessary, but code profiler shows 
   * slightly more efficient)
   */
  if (a == b)
    return 0;

  delta = fabs(b * REAL_TOLERANCE);
  if (a < b - delta) return -1;
  if (a > b + delta) return 1;

  return 0;
}

#define COMPARE_INTS(tag) {int cmp = compare_ints(LOOKUP(a, tag), LOOKUP(b, tag)); if (cmp != 0) return cmp;}
#define COMPARE_REALS(tag) {int cmp = compare_reals(RLOOKUP(a, tag), RLOOKUP(b, tag)); if (cmp != 0) return cmp;}


/* routine to compare two PP records, to see if they are in the same 
 * variable
 *
 * returns: 
 *
 *    -1 or 1  headers are from different variables;
 *               sign of return value gives consistent ordering
 *
 *       0     headers are from same variable
 */
int compare_records_between_vars(const Rec *a, const Rec *b) 
{
  int cmp;
  COMPARE_INTS(INDEX_LBUSER4);
  COMPARE_INTS(INDEX_LBUSER7);
  COMPARE_INTS(INDEX_LBCODE);
  COMPARE_INTS(INDEX_LBVC);
  COMPARE_INTS(INDEX_LBTIM);
  COMPARE_INTS(INDEX_LBPROC);
  COMPARE_REALS(INDEX_BPLAT);
  COMPARE_REALS(INDEX_BPLON);
  COMPARE_INTS(INDEX_LBHEM);
  COMPARE_INTS(INDEX_LBROW);
  COMPARE_INTS(INDEX_LBNPT);

  COMPARE_REALS(INDEX_BGOR);
  COMPARE_REALS(INDEX_BZY);
  COMPARE_REALS(INDEX_BDY);
  COMPARE_REALS(INDEX_BZX);
  COMPARE_REALS(INDEX_BDX);
    
  cmp = compare_mean_periods(a, b);
  if (cmp != 0) return cmp;

  /* Disambig index is used to force distinction between variables for records
   * whose headers are the same.  It is initialised to the same value for all
   * records (in fact -1), but may later be set to different values according
   * to some heuristic.
   */

  cmp = compare_ints(a->internp->disambig_index, b->internp->disambig_index);
  if (cmp != 0) return cmp;

  return 0;
}


/* helper routine for compare_mean_periods - test if both periods are in specified range
 * note - assumes that low, high are positive
 */
static int both_values_in_range(REAL low, REAL high, REAL a, REAL b)
{
  REAL low1 = low * (1. - REAL_TOLERANCE);
  REAL high1 = high * (1. + REAL_TOLERANCE);
  return (a >= low1) && (a <= high1) && (b >= low1) & (b <= high1);  
}


/* Routine to compare if two PP records have different meaning periods, such
 * that they should be considered to be part of different variables.  Normally
 * this will be true if the mean periods differ by more than "delta", but in
 * the case of Gregorian calendar, we allow some tolerance relating to
 * climatology data.
 *
 * This should only get called if both records have already been checked for
 * having the same LBTIM and LBPROC.
 */

int compare_mean_periods(const Rec *a, const Rec *b)
{    
  int cmp;

  cmp = compare_reals(a->internp->mean_period, b->internp->mean_period);
  if (cmp == 0) return 0;

  /* if we get here, times differ - but for gregorian cut some slack */
  if (calendar_type(LOOKUP(a, INDEX_LBTIM)) == gregorian)
    {
      if (both_values_in_range(28., 31., 
			       a->internp->mean_period, b->internp->mean_period)  /* monthly */
	  || both_values_in_range(90., 92.,
				  a->internp->mean_period, b->internp->mean_period)  /* seasonal */
	  || both_values_in_range(365., 366.,
				  a->internp->mean_period, b->internp->mean_period)) /* annual */
	return 0;
    }
  return cmp;
}


/* routine to compare two PP records that the calling routine 
 * has already established are in the same variable.
 * 
 * returns: 
 *
 *    -1 or 1  times or levels differ;
 *               sign of return value gives consistent ordering
 *               of times and levels
 *
 *       0     records do not differ within values tested
 */
int compare_records_within_var(const Rec *a, const Rec *b)
{  
  int a_surface, b_surface;

  COMPARE_INTS(INDEX_LBFT);

  COMPARE_INTS(INDEX_LBYR);
  COMPARE_INTS(INDEX_LBMON);
  COMPARE_INTS(INDEX_LBDAT);
  COMPARE_INTS(INDEX_LBDAY);
  COMPARE_INTS(INDEX_LBHR);
  COMPARE_INTS(INDEX_LBMIN);

  COMPARE_INTS(INDEX_LBYRD);
  COMPARE_INTS(INDEX_LBMOND);
  COMPARE_INTS(INDEX_LBDATD);
  COMPARE_INTS(INDEX_LBDAYD);
  COMPARE_INTS(INDEX_LBHRD);
  COMPARE_INTS(INDEX_LBMIND);

  /*
   *  Ordering of levels:
   * 
   *  Generally we want to sort on LBLEV before sorting on BLEV.
   *
   *  This is because in the case of hybrid levels, BLEV contains the B values
   *  (in p = A + B p_s), which won't do for sorting, and fortunately in this
   *  case LBLEV contains the model level index which is fine.
   *
   *  But there is a nasty special case: surface and boundary layer heat flux
   *  has LBLEV = 9999, 2, 3, 4, ... where 9999 is the surface layer.  In this
   *  case we *could* in fact sort on BLEV, but then we need to know when it's 
   *  okay to do this (STASH code?).  
   *
   *  Maybe safer, treat 9999 lower than any level if comparing it with
   *  another level.  (9999 should always be a special value and it is rare
   *  for it to be mixed with non-special values in the same variable.)
   */
  a_surface = (LOOKUP(a, INDEX_LBLEV) == 9999);
  b_surface = (LOOKUP(b, INDEX_LBLEV) == 9999);
  if (a_surface && !b_surface)
    return -1;
  else if (b_surface && !a_surface)
    return 1;

  COMPARE_INTS(INDEX_LBLEV);
  COMPARE_REALS(INDEX_BLEV);
  COMPARE_REALS(INDEX_BHLEV);

  return 0;
}


/* routine to compare two PP records.
 * returns: 
 *    -2 or 2  headers are from different variable
 *    -1 or 1  headers are from same variable
 *       0     difference not found in elements inspected
 *
 */
int compare_records(const void *p1, const void *p2)
{
  const Rec *a = * (Rec **) p1;
  const Rec *b = * (Rec **) p2;

  int cmp;

  cmp = compare_records_between_vars(a, b);
  if (cmp != 0) {
    //    debug("compare_records - variables differ %d %d", LOOKUP(a,INDEX_LBUSER4),
    //	  LOOKUP(b,INDEX_LBUSER4));

    return cmp * 2;
  }
  cmp = compare_records_within_var(a, b);
  if (cmp != 0){
    //    debug("compare_records - variables same %d %d", LOOKUP(a,INDEX_LBUSER4),
    //LOOKUP(b,INDEX_LBUSER4));

    return cmp;
  }
//debug("compare_records - records same");
  return 0;
}


int records_from_different_vars(const Rec *a, const Rec *b)
{
  return (compare_records_between_vars(a, b) != 0);
}


int compare_lists(const List *l1, const List *l2, int (*compfunc)(const void*, const void*))
{
  int i, n, cmp;
  const void *item1, *item2;
  List_handle handle1, handle2;

  /* differ if number of items differs */
  n = list_size(l1);
  if ((cmp = compare_ints(n, list_size(l2))) != 0) return cmp;
  
  /* differ if any individual item differs */
  list_startwalk(l1, &handle1);
  list_startwalk(l2, &handle2);
  for (i = 0; i < n; i++) {
    item1 = list_walk(&handle1, 0);
    item2 = list_walk(&handle2, 0);
    if ((cmp = compfunc(&item1, &item2)) != 0) return cmp;
  }
  return 0;
}

int compare_levels(const void *p1, const void *p2) 
{
  const Level *a = *(Level **)p1;
  const Level *b = *(Level **)p2;

  /* macros called LCOMPARE_INTS and LCOMPARE_REALS to emphasise difference from those in compare_records */

#define LCOMPARE_INTS(tag) {int cmp = compare_ints(a->tag, b->tag); if (cmp != 0) return cmp;}
#define LCOMPARE_REALS(tag) {int cmp = compare_reals(a->tag, b->tag); if (cmp != 0) return cmp;}

  LCOMPARE_INTS(type);

  switch (a->type) {
  case hybrid_height_lev_type:
    LCOMPARE_REALS(values.hybrid_height.a);
    LCOMPARE_REALS(values.hybrid_height.b);
#ifdef BDY_LEVS
    LCOMPARE_REALS(values.hybrid_height.ubdy_a);
    LCOMPARE_REALS(values.hybrid_height.ubdy_b);
    LCOMPARE_REALS(values.hybrid_height.lbdy_a);
    LCOMPARE_REALS(values.hybrid_height.lbdy_b);
#endif
    break;
  case hybrid_sigmap_lev_type:
    LCOMPARE_REALS(values.hybrid_sigmap.a);
    LCOMPARE_REALS(values.hybrid_sigmap.b);
#ifdef BDY_LEVS
    LCOMPARE_REALS(values.hybrid_sigmap.ubdy_a);
    LCOMPARE_REALS(values.hybrid_sigmap.ubdy_b);
    LCOMPARE_REALS(values.hybrid_sigmap.lbdy_a);
    LCOMPARE_REALS(values.hybrid_sigmap.lbdy_b);
#endif
    break;
  case pseudo_lev_type:
    LCOMPARE_INTS(values.pseudo.index);
    break;
  default:
    LCOMPARE_REALS(values.misc.level);
#ifdef BDY_LEVS
    LCOMPARE_REALS(values.misc.ubdy_level);
    LCOMPARE_REALS(values.misc.lbdy_level);
#endif
    break;
  }
  return 0;
}

int compare_times(const void *p1, const void *p2)
{
  const Time *a = * (Time **) p1;
  const Time *b = * (Time **) p2;
  int cmp;

  /* LBTYP: ignore 100s digit = sampling frequency, as we don't use it for anything */
  if ((cmp = compare_ints(a->type % 100, b->type % 100)) != 0) return cmp;

  if ((cmp = compare_dates(&a->time1, &b->time1)) != 0) return cmp;
  if ((cmp = compare_dates(&a->time2, &b->time2)) != 0) return cmp;
  return 0;
}

int compare_dates(const Date *a, const Date *b)
{
  int cmp;
  if ((cmp = compare_ints(a->year  ,b->year  )) != 0) return cmp;
  if ((cmp = compare_ints(a->month ,b->month )) != 0) return cmp;
  if ((cmp = compare_ints(a->day   ,b->day   )) != 0) return cmp;
  if ((cmp = compare_ints(a->hour  ,b->hour  )) != 0) return cmp;
  if ((cmp = compare_ints(a->minute,b->minute)) != 0) return cmp;
  if ((cmp = compare_ints(a->second,b->second)) != 0) return cmp;
  return 0;
}

