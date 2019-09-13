#include <stdlib.h>

#include "umfileint.h"


File *file_parse_core(int fd,
		      File_type file_type)
{
  File *file;
  List *heaplist;

  CKP(  file = new_file()  );
  file->fd = fd;
  file->file_type = file_type;

  heaplist = file->internp->heaplist;

  CKI(  read_all_headers(file, heaplist)  );
  CKI(  process_vars(file, heaplist)  );

  return file;

 err:
  if (file)
    free_file(file);
  return NULL;
}


int process_vars(File *file, List *heaplist)
{
  int nrec;
  Rec **recs;
  List *vars;

  nrec = file->internp->nrec;
  recs = file->internp->recs;
  
  /* initialise elements in the records before sorting */
  CKI(   initialise_records(recs, nrec, heaplist)   );

  /* sort the records */
  qsort(recs, nrec, sizeof(Rec*), compare_records);

  /* now sort out the list of variables and dimensions */
  CKP(   vars = list_new(heaplist)   );
  CKI(   get_vars(nrec, recs, vars, heaplist)   );
  /* move the variables from the linked list to the array */  
  CKI(   list_copy_to_ptr_array(vars, &file->nvars, &file->vars, heaplist)   );
  CKI(   list_free(vars, 0, heaplist)   );
  return 0;
  ERRBLKI;
}


/*
 * scan records for vars and add all to list
 */

int get_vars(int nrec, Rec **recs, 
	     List *vars, 
	     List *heaplist)
{
  int recno;
  int at_start_rec, at_end_rec = 0;
  Rec *rec, **vrecs;
  Z_axis *z_axis;
  T_axis *t_axis;
  Var *var;
  int zindex, tindex;
  int nvrec;  
  int svindex = 1;
  
  var = NULL;
  z_axis = NULL;
  t_axis = NULL;
  
  for (recno=0; recno < nrec ; recno++) 
    {
      rec = recs[recno];

      /* Some fieldsfiles have header fields with missing values */
      if (var_is_missing(rec->int_hdr))
	{
	  error_mesg("skipping variable stash code=%d, %d, %d "
		     "because of missing header data", 
		     get_var_stash_model(rec->int_hdr), 
		     get_var_stash_section(rec->int_hdr), 
		     get_var_stash_item(rec->int_hdr));
	  continue;
	}

      /* we are at start record of a variable at the very start, or if at we
       * were at the end record last time
       */
      at_start_rec = ( recno == 0 || at_end_rec );
      
      /* we are at end record of a variable at the very end, or if the header
       * shows a difference from the next record which constitutes a different
       * variable
       *
       * We also force end record of a variable if the grid type is not
       * supported; any such records are passed back as single-record
       * variables for the caller to deal with.
       */
      at_end_rec = ( recno == nrec - 1 ||
		     records_from_different_vars(recs[recno + 1], rec) ||
		     !grid_supported(rec->int_hdr));
      
      /* allow for variables which are unsupported for some reason */
      if (at_start_rec && test_skip_var(rec))
	continue;
      
      /* initialise new variable and axes if at first record of a variable */
      if (at_start_rec)
	{
	  CKP(  var = new_var(heaplist)   );
	  CKP(  z_axis = new_z_axis(heaplist)  );
	  CKP(  t_axis = new_t_axis(heaplist)  );

	  var->internp->first_rec_no = recno;
	  var->internp->last_rec_no = -1;
	  var->internp->first_rec = rec;
	}

    /* for every record, add the z, t values to the axes */

    CKI(  z_axis_add(z_axis, rec->internp->lev, &zindex, heaplist)  );
    rec->internp->zindex = zindex;

    CKI(  t_axis_add(t_axis, rec->internp->time, &tindex, heaplist)  );
    rec->internp->tindex = tindex;

    if (at_end_rec) 
      {
	var->internp->last_rec_no = recno;
	nvrec = var->internp->last_rec_no - var->internp->first_rec_no + 1;
	vrecs = recs + var->internp->first_rec_no;
	
      /* now if the axes are not regular, free the axes, split the variable
       * into a number of variables and try again...
       */
      if (set_disambig_index(z_axis, t_axis, vrecs, nvrec, svindex))
	{
	  /* increment the supervar index, used later to show the connection
	   *  between the separate variables into which this one will be split
	   */
	  svindex++;

	  /* now re-sort this part of the record list, 
	   * now that we have set the disambig index */
	  qsort(vrecs, nvrec, sizeof(Rec *), compare_records);  

	  /* now go back to the start record of the variable; set to one less
	   * because it will get incremented in the "for" loop reinitialisation
	   */
	  recno = var->internp->first_rec_no - 1;

	  /* and free the stuff associated with the var we won't be using */
	  CKI(  free_z_axis(z_axis, heaplist)  );
	  CKI(  free_t_axis(t_axis, heaplist)  );
	  CKI(  free_var(var, heaplist)  );

	continue;
      }

      /* add the metadata the caller needs */

      var->nz = list_size(z_axis->values);
      var->nt = list_size(t_axis->values);
      var->recs = &recs[var->internp->first_rec_no];
      svindex = var->internp->first_rec->internp->supervar_index;
      if (svindex >= 0)
	var->supervar_index = svindex;

      /* add the variable */
      CKI(   list_add(vars, var, heaplist)   );

      /* don't need the axes any more */
      CKI(  free_z_axis(z_axis, heaplist)  );
      CKI(  free_t_axis(t_axis, heaplist)  );
      }
  }
  return 0;
  ERRBLKI;
}


int test_skip_var(const Rec *rec)
{
  char *skip_reason;
  INTEGER *int_hdr;

  int_hdr = rec->int_hdr;      
  skip_reason = NULL;

  if (var_is_missing(int_hdr))
    skip_reason = "PP record has essential header data set to missing data value";

  /* Compressed field index */
  if (get_var_compression(int_hdr) == 1)
    skip_reason = "compressed field index not supported";

  /* remove grid_supported test - now used to split up variables, not to 
   * exclude them.
   * 
   * if (grid_supported(int_hdr) == 0)
   *  skip_reason = "grid code not supported";
   */

  /* ADD ANY MORE VARIABLE SKIPPING CASES HERE. */

  if (skip_reason != NULL)
    {
      error_mesg("skipping variable stash code=%d, %d, %d because: %s", 
		 get_var_stash_model(int_hdr), 
		 get_var_stash_section(int_hdr), 
		 get_var_stash_item(int_hdr), 
		 skip_reason);
      return 1;
    }
  return 0;
}


int initialise_records(Rec **recs, int nrec, List *heaplist) 
{
  int recno;
  Rec *rec;

  for (recno = 0; recno < nrec ; recno++)
    {
      rec = recs[recno];
      rec->internp = rec->internp;

      rec->internp->disambig_index = -1;
      rec->internp->supervar_index = -1;

      /* store level info */
      CKP(  rec->internp->lev = malloc_(sizeof(Level), heaplist)  );
      CKI(  lev_set(rec->internp->lev, rec)  );

      /* store time info */
      CKP(  rec->internp->time = malloc_(sizeof(Time), heaplist)  );
      CKI(  time_set(rec->internp->time, rec)  );
      rec->internp->mean_period = mean_period(rec->internp->time);
  }
  return 0;
  ERRBLKI;
}


/*
 * set the disambig index on all records within a super-variable
 */
int set_disambig_index(Z_axis *z_axis, T_axis *t_axis, 
		       Rec **recs, int nvrec, int svindex)
{
  int var_rec_no;
  Rec *vrec;
  int zindex, tindex, dindex;
  int prev_zindex, prev_tindex, prev_dindex;

  prev_zindex = prev_tindex = prev_dindex = 0;

  /* do nothing if axes are regular */
  if (var_has_regular_z_t(z_axis, t_axis, recs, nvrec))
    return 0;

  for (var_rec_no=0; var_rec_no < nvrec; var_rec_no++) 
    {
      vrec = recs[var_rec_no];
      
      zindex = vrec->internp->zindex;
      tindex = vrec->internp->tindex;

    /* check for dups coord pairs */
    /* the exact expressions for dindex are fairly arbitrary -- just need to
     * ensure that indices for dup coordinate pairs will be different from
     * indices for non-dups on other levels
     */
      if (var_rec_no > 0 
	  && zindex == prev_zindex
	  && tindex == prev_tindex)
	dindex = prev_dindex + 1;
      else
	dindex = zindex * nvrec;
    
      vrec->internp->disambig_index = dindex;
    
      if (vrec->internp->supervar_index < 0)
	vrec->internp->supervar_index = svindex;

      /* save vals for next iter */
      prev_zindex = zindex;
      prev_tindex = tindex;
      prev_dindex = dindex;
    }  
  return 1;
}


int grid_supported(INTEGER *int_hdr)
{
  int gridcode;
  gridcode = get_var_gridcode(int_hdr);

  switch(gridcode) {

  case 1:
  case 101:
  case 11110:
    return 1;

  default:
    return 0;
  }
}


/* routine to test t and z indices to check whether the variable is on regular
 * array of times and levels (NB "regular" here refers to the ordering, not to
 * whether the spacing is uniform)
 */

int var_has_regular_z_t(Z_axis *z_axis, T_axis *t_axis, Rec **recs, int nvrec)
{
  int var_rec_no, nz, nt; /* needed for check on variables */
  Rec *rec;

  nz = list_size(z_axis->values);
  nt = list_size(t_axis->values);

  /*------------------------------------------------------------*/

  /* first test the most obvious case of irregular (for possible speed) */
  if (nvrec != nz * nt)
    return 0;

  /* z indices (faster varying) should loop be vrec % nz */
  /* t indices (slower varying) should loop be vrec / nz */
  for (var_rec_no=0; var_rec_no < nvrec; var_rec_no++) 
    {
      rec = recs[var_rec_no];
      if (rec->internp->zindex != var_rec_no % nz
	  || rec->internp->tindex != var_rec_no / nz)
	return 0;
    }
  return 1;
}
