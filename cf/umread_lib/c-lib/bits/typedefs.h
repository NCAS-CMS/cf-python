#include <inttypes.h>

typedef float float32_t;
typedef double float64_t;

enum { single_precision, double_precision };

/*---------------------------*/
/* for linked list */

struct _list_element
{
  void *ptr;
  struct _list_element *prev;
  struct _list_element *next;
};
typedef struct _list_element List_element;

typedef struct 
{
  int n;
  List_element *first;
  List_element *last;
}
  List;

typedef struct 
{
  /* This is a little structure which stores the information needed for
   * pp_list_walk.  Its main purpose is to store the position outside the list
   * structure itself, so that for read-only scanning of the list, the PPlist*
   * can be declared as const.
   */
  List_element *current;
  const List *list;
}
  List_handle;

/*---------------------------*/

typedef enum
{
  pseudo_lev_type,
  height_lev_type,
  depth_lev_type,
  hybrid_sigmap_lev_type,
  hybrid_height_lev_type,
  pressure_lev_type,
  soil_lev_type,
  boundary_layer_top_lev_type,
  top_of_atmos_lev_type,
  mean_sea_lev_type,
  surface_lev_type,
  tropopause_lev_type,
  other_lev_type
}
  Lev_type;

typedef enum
{
  gregorian, 
  cal360day,
  model
}
  Calendar_type;

typedef enum
{
  lev_type,
  hybrid_sigmap_a_type,
  hybrid_sigmap_b_type,
  hybrid_height_a_type,
  hybrid_height_b_type
}
  Lev_val_type;


#if defined(INTEGER)

typedef struct
{
  INTEGER year;
  INTEGER month;
  INTEGER day;
  INTEGER hour;
  INTEGER minute;
  INTEGER second;
} 
  Date;

typedef struct
{
  /* this is a value on time axis */
  INTEGER type;
  Date time1;
  Date time2;
}
  Time;

typedef struct
{
  List *values;
}
  T_axis;

typedef struct
{
  Lev_type type;

  union 
  {
    struct
    { 
      REAL level;
#ifdef BDY_LEVS
      REAL ubdy_level;
      REAL lbdy_level;
#endif
    } 
      misc;

    struct
    {
      REAL a;
      REAL b;
#ifdef BDY_LEVS
      REAL ubdy_a;
      REAL ubdy_b;
      REAL lbdy_a;
      REAL lbdy_b;
#endif
    } 
      hybrid_sigmap;

    struct
    { 
      REAL a;
      REAL b;
#ifdef BDY_LEVS
      REAL ubdy_a;
      REAL ubdy_b;
      REAL lbdy_a;
      REAL lbdy_b;
#endif
    } 
      hybrid_height;
    
    struct
    { 
      INTEGER index;
    }
      pseudo;    
  } 
    values;
}
  Level;

typedef struct
{
  List *values;
}
  Z_axis;

#else

typedef void Z_axis;
typedef void T_axis;
typedef void Time;
typedef void Level;

#endif

/*---------------------------*/

struct _File
{
  List *heaplist;
  int nrec;
  Rec **recs;
};

struct _Var 
{
  Z_axis *z_axis;
  T_axis *t_axis;
  Rec *first_rec;
  int first_rec_no;
  int last_rec_no;
};

struct _Rec 
{
  Level *lev;
  Time *time;
  int zindex; /* index on z axis within a variable - used for detecting vars with irreg z,t */
  int tindex; /* index on t axis within a variable - used for detecting vars with irreg z,t */
  int disambig_index; /* index used for splitting variables with irreg z,t into 
                       * sets of variables with regular z,t */
  int supervar_index; /* when a variable is split, this is set to an index which is common
                       * across the set, but different from sets generated from other
                       * super-variables
                       */
  float64_t mean_period; /* period (in days) of time mean 
			    (store here so as to calculate once only) */  
};

/*---------------------------*/

