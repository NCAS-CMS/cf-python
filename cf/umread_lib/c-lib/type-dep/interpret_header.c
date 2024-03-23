#include "umfileint.h"

Data_type get_type(const INTEGER *int_hdr)
{
  switch (int_hdr[INDEX_LBUSER1])
    {
    case(2):
    case(-2):
    case(3):
    case(-3):
      return int_type;
      /* break; */
    case(1):
    case(-1):
      return real_type;
      /* break; */
    default:
      error_mesg("Warning: datatype %d not recognised, assuming real", 
		 int_hdr[INDEX_LBUSER1]);
      return real_type;
    }
}

/* Get number of data words. Does not include extra data. */
size_t get_num_data_words(const INTEGER *int_hdr) 
{
  if (int_hdr[INDEX_LBPACK] != 0 
      && int_hdr[INDEX_LBROW] > 0 
      && int_hdr[INDEX_LBNPT] > 0)
    /* if packed and horizontal grid sizes set, use them */
    return int_hdr[INDEX_LBROW] * int_hdr[INDEX_LBNPT];
  else
    /* otherwise use LBLREC */
    return int_hdr[INDEX_LBLREC] - get_extra_data_length(int_hdr) / WORD_SIZE;
}

/* get length of (any) extra data in bytes */
size_t get_extra_data_length(const INTEGER *int_hdr)
{
  if (int_hdr[INDEX_LBEXT] > 0)
    return int_hdr[INDEX_LBEXT] * WORD_SIZE;
  return 0;
}

size_t get_extra_data_offset_and_length_core(const INTEGER *int_hdr,
					     size_t data_offset,
					     size_t disk_length,
					     size_t *extra_data_offset_rtn,
					     size_t *extra_data_length_rtn)
{
  size_t extra_data_length;

  extra_data_length = get_extra_data_length(int_hdr);
  *extra_data_length_rtn = extra_data_length;
 
  /* If data is packed, the only way of telling where the extra data is 
   * is to assume that it is at the very end of data_length.
   *
   * If data is not packed, then can use data_length to work out where the 
   * extra data starts, allowing resilience against the possibility that 
   * disk_length might include some possible padding
   */
  if (int_hdr[INDEX_LBPACK] != 0)
      *extra_data_offset_rtn = data_offset + disk_length - extra_data_length;
  else
    *extra_data_offset_rtn = data_offset + get_num_data_words(int_hdr) * WORD_SIZE;

  return 0;
}

int get_type_and_num_words_core(const INTEGER *int_hdr,
				Data_type *type_rtn,
				size_t *num_words_rtn)
{
  *type_rtn = get_type(int_hdr);
  *num_words_rtn = get_num_data_words(int_hdr);
  return 0;
}


/* sometimes a variable is included but which has some
 * really essential header elements to missing data flag,
 * so the variable is essentially missing in that any
 * attempt to process the variable is only going to
 * lead to errors
 *
 * pp_var_missing() tests for this.
 *
 * FIXME: expand to test other header elements
 */
int var_is_missing(const INTEGER *int_hdr)
{
  if (int_hdr[INDEX_LBNPT] == INT_MISSING_DATA)
    return 1;

  if (int_hdr[INDEX_LBROW] == INT_MISSING_DATA)
    return 1;

  return 0;
}

int get_var_stash_model(const INTEGER *int_hdr)
{
  return int_hdr[INDEX_LBUSER7];
}

int get_var_stash_section(const INTEGER *int_hdr)
{
  return int_hdr[INDEX_LBUSER4] / 1000;
}

int get_var_stash_item(const INTEGER *int_hdr)
{
  return int_hdr[INDEX_LBUSER4] % 1000;
}

int get_var_compression(const INTEGER *int_hdr)
{
  return (int_hdr[INDEX_LBPACK] / 10) % 10;
}


int get_var_gridcode(const INTEGER *int_hdr)
{
  return int_hdr[INDEX_LBCODE];
}


int get_var_packing(const INTEGER *int_hdr)
{
   return int_hdr[INDEX_LBPACK] % 10;
}


/* get the fill value from the floating point header.
 * caller needs to check that it is actually floating point data!
 */
REAL get_var_real_fill_value(const REAL *real_hdr)
{
  return real_hdr[INDEX_BMDI];
}
