#include "umfileint.h"

/*
 * Aside from the time_set() function, most of the infrastructure in this file 
 * is for the purpose of calculating the length of the time mean period, just so
 * that e.g. monthly and daily means in the same file can be separated out into 
 * different variables.
 */

REAL mean_period(const Time *time) 
{
  /* returns the averaging period in days, or 0. if it is not a mean field */
  INTEGER lbtim = time->type;
  if (!is_time_mean(lbtim)) 
    return 0.;
  return time_diff(lbtim, &time->time2, &time->time1);
}

int is_time_mean(INTEGER LBTIM) 
{
  int ib;
  ib = (LBTIM / 10) % 10;
  return (ib == 2) || (ib == 3);
}

REAL time_diff(INTEGER lbtim, const Date *date, const Date *orig_date)
{
  int64_t secs;

  switch(calendar_type(lbtim))
    {
    case gregorian:
      return sec_to_day(gregorian_to_secs(date) - gregorian_to_secs(orig_date));
      break; /* notreached */
    case cal360day:
      secs =
	date->second - orig_date->second + 
	60 * (date->minute - orig_date->minute + 
	      60 * (date->hour - orig_date->hour + 
		    24 * (date->day - orig_date->day + 
			  30 * (date->month - orig_date->month +
				12 * (int64_t) (date->year - orig_date->year) ))));
      
      return sec_to_day(secs);
      break; /* notreached */
    case model:
      secs =
	date->second - orig_date->second + 
	60 * (date->minute - orig_date->minute + 
	      60 * (date->hour - orig_date->hour + 
		    24 * (int64_t) (date->day - orig_date->day)));
      
      return sec_to_day(secs);
      break; /* notreached */
    default:
      SWITCH_BUG;
    }  
  ERRBLKF;
}

REAL sec_to_day(int64_t seconds)
{
  /* convert seconds to days, avoiding rounding where possible 
   * by using integer arithmetic for the whole days
   */
  const int secs_per_day = 86400;
  
  int64_t days, remainder;
  days = seconds / secs_per_day;
  remainder = seconds % secs_per_day;

  return days + remainder / (REAL) secs_per_day;
}

Calendar_type calendar_type(INTEGER type)
{
  switch(type % 10)
    {
    case 0:
      /* fallthrough */
    case 3:
      return model;
      break; /* notreached */
    case 1:
      return gregorian;
      break; /* notreached */
    case 2:
      return cal360day;
      break; /* notreached */
    default:
      SWITCH_BUG;
  }

  /* on error return -1 (though only useful to calling routine if stored in an int
   * not a Calendar_type)
   */
  ERRBLKI;
}

int64_t gregorian_to_secs(const Date *date)
{
  /* Convert from Gregorian calendar to seconds since a fixed origin
   * 
   * Can be with respect to any arbitary origin, because return values from this are
   * differenced.
   *
   * Arbitrary origin is what would be 1st Jan in the year 0 if hypothetically the
   * system was completely consistent going back this far.  This simplifies the 
   * calculation.
   * 
   * Strictly, this is proleptic_gregorian rather than gregorian (see CF docs)
   * as this is more likely to match the code actually in the model.  The UM
   * docs call it "gregorian" but I'm speculating (without checking model code)
   * that the model really doesn't have all that jazz with Julian calendar
   * before fifteen-something.
   */
  const int sid = 86400; /* seconds in day */
  const int sih = 3600;
  const int sim = 60;

  int64_t nsec;
  int year,nleap,nday,isleap;

  /* offsets from 1st Jan to 1st of each month in non-leap year */
  int dayno[12] = { 0,  31,  59,  90, 120, 151, 181, 212, 243, 273, 304, 334 };

  year = date->year;

  /* is the year leap? */
  if (year % 400 == 0) isleap = 1;
  else if (year % 100 == 0) isleap = 0;
  else if (year % 4 == 0) isleap = 1;
  else isleap=0;

  /* nleap is number of 29th Febs passed between origin date and supplied date. */
  nleap = year / 4 - year / 100 + year / 400;
  if (isleap && date->month <= 2)
    nleap--;

  nday = (year * 365) + dayno[date->month - 1] + (date->day - 1) + nleap;

  nsec = (int64_t) nday * sid + date->hour * sih + date->minute * sim + date->second;

  return nsec;
}


int time_set(Time *time, const Rec *rec)
{
  time->type    = LOOKUP(rec, INDEX_LBTIM);
  
  time->time1.year    = LOOKUP(rec, INDEX_LBYR);
  time->time1.month   = LOOKUP(rec, INDEX_LBMON);
  time->time1.day     = LOOKUP(rec, INDEX_LBDAT);
  time->time1.hour    = LOOKUP(rec, INDEX_LBHR);
  time->time1.minute  = LOOKUP(rec, INDEX_LBMIN);
  time->time1.second  = 0;
  
  time->time2.year    = LOOKUP(rec, INDEX_LBYRD);
  time->time2.month   = LOOKUP(rec, INDEX_LBMOND);
  time->time2.day     = LOOKUP(rec, INDEX_LBDATD);
  time->time2.hour    = LOOKUP(rec, INDEX_LBHRD);
  time->time2.minute  = LOOKUP(rec, INDEX_LBMIND);
  time->time2.second  = 0;
  
  return 0;
}


