#include <unistd.h>
#include <stdlib.h>

#include "umfileint.h"


#define file_pos(f) (lseek(f, 0, SEEK_CUR))


#if defined(SINGLE)
#define swap_bytes swap_bytes_sgl
#elif defined(DOUBLE)
#define swap_bytes swap_bytes_dbl
#else
#error Need to compile this file with -DSINGLE or -DDOUBLE
#endif


/*
 * reads n words from file, storing them at ptr, with byte swapping as required
 * returns number of words read (i.e. n, unless there's a short read)
 */
size_t read_words(int fd, 
		  void *ptr,
		  size_t num_words,
		  Byte_ordering byte_ordering)
{
  size_t nread;

  CKP(ptr);
  nread = read(fd, ptr, num_words * WORD_SIZE) / WORD_SIZE;
  if (byte_ordering == REVERSE_ORDERING)
    swap_bytes(ptr, nread);
  return nread;
  ERRBLKI;
}


int read_extra_data_core(int fd,
			 size_t extra_data_offset,
			 size_t extra_data_length, 
			 Byte_ordering byte_ordering, 
			 void *extra_data_rtn)
{
  /* reads extra data into storage provided by the caller
   * The caller must provide the offset and length, obtained 
   * by a previous call to get_extra_data_offset_and_length()
   */
  size_t extra_data_words;

  extra_data_words = extra_data_length / WORD_SIZE;

  CKI(  lseek(fd, extra_data_offset, SEEK_SET)  );
  ERRIF(   extra_data_length % WORD_SIZE != 0   );
  ERRIF(   read_words(fd, extra_data_rtn, 
		      extra_data_words, byte_ordering)   != extra_data_words);
  return 0;
  ERRBLKI;    
}


int read_hdr_at_offset(int fd,
		       size_t header_offset,
		       Byte_ordering byte_ordering, 
		       INTEGER *int_hdr_rtn,
		       REAL *real_hdr_rtn)
{
  /* as read_hdr below, but also specifying the file offset in bytes */

  CKI(  lseek(fd, header_offset, SEEK_SET)  );
  return read_hdr(fd, byte_ordering, int_hdr_rtn, real_hdr_rtn);
  ERRBLKI;
}


int read_hdr(int fd,
	     Byte_ordering byte_ordering, 
	     INTEGER *int_hdr_rtn,
	     REAL *real_hdr_rtn)
{
  /* reads a PP header at specified word offset into storage 
     provided by the caller */

  ERRIF(   read_words(fd, int_hdr_rtn, 
		      N_INT_HDR, byte_ordering)   != N_INT_HDR);
  
  ERRIF(   read_words(fd, real_hdr_rtn, 
		      N_REAL_HDR, byte_ordering)   != N_REAL_HDR);

  return 0;
  ERRBLKI;    
}


Rec *get_record(File *file, List *heaplist)
{
  /* reads PP headers and returns a Rec structure -- 
   *
   * file must be positioned at start of header (after any fortran record length integer) on entry,
   * and will be positioned at end of header on return
   *
   * the Rec structure will contain the headers in elements int_hdr and real_hdr, 
   * but other elements will be left as initialised by new_rec()
   */

  Rec *rec;

  CKP(   rec = new_rec(WORD_SIZE, heaplist)   );

  CKI(
      read_hdr(file->fd,
	       file->file_type.byte_ordering,
	       rec->int_hdr,
	       rec->real_hdr)
      );
  
  return rec; /* success */
  ERRBLKP;    
}

int read_all_headers(File *file, List *heaplist)
{
  switch (file->file_type.format)
    {
    case plain_pp:
      return read_all_headers_pp(file, heaplist);
    case fields_file:
      return read_all_headers_ff(file, heaplist);
    default:
      switch_bug("read_all_headers");
      ERR;
    }
  return 0;  
  ERRBLKI;
}

/* skip_fortran_record: skips a fortran record, and returns how big it was (in bytes),
 *  or -1 for end of file, or -2 for any error which may imply corrupt file
 * (return value of 0 is a legitimate empty record).
 */
size_t skip_fortran_record(File *file)
{
  INTEGER rec_bytes, rec_bytes_2;

  if(   read_words(file->fd, &rec_bytes, 1, 
		   file->file_type.byte_ordering)   != 1) return -1;
  CKI(   lseek(file->fd, rec_bytes, SEEK_CUR)   );
  ERRIF(   read_words(file->fd, &rec_bytes_2, 1, 
		      file->file_type.byte_ordering)   != 1);
  ERRIF(rec_bytes != rec_bytes_2);
  return rec_bytes;
  ERRBLK(-2);
}

int skip_word(File *file)
{
  CKI(   lseek(file->fd, WORD_SIZE, SEEK_CUR)   );
  return 0;

  ERRBLKI;
}

int read_all_headers_pp(File *file, List *heaplist)
{
  int fd;
  size_t nrec, rec_bytes, recno, header_offset;
  Rec **recs, *rec;

  fd = file->fd;

  /* count the PP records in the file */
  lseek(fd, 0, SEEK_SET);
  for (nrec = 0; (rec_bytes = skip_fortran_record(file)) != -1; nrec++) 
    {
      ERRIF(rec_bytes == -2);
      if (rec_bytes != N_HDR * WORD_SIZE) {
	error_mesg("unsupported header length in PP file: %d words", 
		   rec_bytes / WORD_SIZE);
	ERR;
      }
      ERRIF(   skip_fortran_record(file)   < 0); /* skip the data record */
    }
  
  /* now rewind, and read in all the PP header data */
  CKP(   recs = malloc_(nrec * sizeof(Rec *), heaplist)   );
  file->internp->nrec = nrec;
  file->internp->recs = recs;

  lseek(fd, 0, SEEK_SET);
  for (recno = 0; recno < nrec; recno++)
    {
      CKI(   skip_word(file)   );
      header_offset = file_pos(fd);
      CKP(   rec = get_record(file, heaplist)   );
      CKI(   skip_word(file)   );
      recs[recno] = rec;

      /* skip data record but store length */
      rec->header_offset = header_offset;
      rec->data_offset = file_pos(fd) + WORD_SIZE;
      rec->disk_length = skip_fortran_record(file);
    }
  return 0;
  ERRBLKI;
}


#define READ_ITEM(x) \
  ERRIF(   read_words(fd, &x, 1, byte_ordering)   != 1);

int read_all_headers_ff(File *file, List *heaplist)
{
  int fd;
  size_t hdr_start, hdr_size, header_offset, data_offset_calculated, data_offset_specified;
  int *valid, n_valid_rec, n_raw_rec, i_valid_rec, i_raw_rec;
  Byte_ordering byte_ordering;
  Rec *rec, **recs;
  INTEGER start_lookup, nlookup1, nlookup2, dataset_type, start_data;

  fd = file->fd;
  byte_ordering = file->file_type.byte_ordering;
  
  /* pick out certain information from the fixed length header */    
  CKI(   lseek(fd, 4 * WORD_SIZE, SEEK_SET)  );
  READ_ITEM(dataset_type);

  ERRIF(   read_words(fd, &dataset_type, 1, byte_ordering)   != 1);
  
  CKI(   lseek(fd, 149 * WORD_SIZE, SEEK_SET)  );
  READ_ITEM(start_lookup);
  READ_ITEM(nlookup1);
  READ_ITEM(nlookup2);

  CKI(   lseek(fd, 159 * WORD_SIZE, SEEK_SET)  );
  READ_ITEM(start_data);

  /* (first dim of lookup documented as being 64 or 128, so 
   * allow header longer than n_hdr (64) -- discarding excess -- but not shorter)
   */

  if (nlookup1 < N_HDR)
    {
      error_mesg("unsupported header length: %d words", nlookup1);
      ERR;
    }

  CKP(  valid = malloc_(nlookup2 * sizeof(int), heaplist)   );

  hdr_start = (start_lookup - 1) * WORD_SIZE;
  hdr_size = nlookup1 * WORD_SIZE;
  n_raw_rec = nlookup2;
  CKI(  get_valid_records_ff(fd, byte_ordering, hdr_start, hdr_size, n_raw_rec,
				       valid, &n_valid_rec)  );  

  /* now read in all the PP header data */
  
  CKP(   recs = malloc_(n_valid_rec * sizeof(Rec *), heaplist)   );
  /* debug("n_raw_rec=%d n_valid_rec=%d", n_raw_rec, n_valid_rec); */
  file->internp->nrec = n_valid_rec;
  file->internp->recs = recs;
  
  i_valid_rec = 0;
  data_offset_calculated = (start_data - 1) * WORD_SIZE;
  for (i_raw_rec = 0; i_raw_rec < n_raw_rec; i_raw_rec++)
    {
      if (valid[i_raw_rec])
	{
	  header_offset = hdr_start + i_raw_rec * hdr_size;
	  CKI(   lseek(fd, header_offset, SEEK_SET)  );
	  CKP(   rec = get_record(file, heaplist)   );
	  recs[i_valid_rec] = rec;
	  
	  rec->header_offset = header_offset;
	  rec->disk_length = get_ff_disk_length(rec->int_hdr);
	  
	  data_offset_specified = (size_t) LOOKUP(rec, INDEX_LBBEGIN) * WORD_SIZE;
	  /* use LBBEGIN if available */
	  rec->data_offset =
	    (data_offset_specified != 0) ? data_offset_specified : data_offset_calculated;
	  
	  data_offset_calculated += rec->disk_length;

	  i_valid_rec++;
	}
    }
      
  CKI(  free_(valid, heaplist)  );
  return 0;
  ERRBLKI;
}


size_t get_ff_disk_length(INTEGER *ihdr)
{
  /* work out disk length in bytes */
  /* Input array size (packed field):
   *   First try LBNREC
   *   then if Cray 32-bit packing, know ratio of packed to unpacked lengths;
   *   else use LBLREC
   */
  if (ihdr[INDEX_LBPACK] != 0 && ihdr[INDEX_LBNREC] != 0) 
    return ihdr[INDEX_LBNREC] * WORD_SIZE;
  if (ihdr[INDEX_LBPACK] % 10 == 2)
      return get_num_data_words(ihdr) * 4;
  return ihdr[INDEX_LBLREC] * WORD_SIZE;
}


/* 
 *  check which PP records are valid; populate an array provided by the caller with 1s and 0s
 *  and also provide the total count
 */
int get_valid_records_ff(int fd,
			 Byte_ordering byte_ordering,
			 size_t hdr_start, size_t hdr_size, int n_raw_rec,
			 int valid[], int *n_valid_rec_return)
{
  int n_valid_rec, irec;
  INTEGER lbbegin;
  n_valid_rec = 0;
  
  for (irec = 0; irec < n_raw_rec; irec++)
    {
      valid[irec] = 0;
      CKI(   lseek(fd, hdr_start + irec * hdr_size + INDEX_LBBEGIN * WORD_SIZE, SEEK_SET)   );
      READ_ITEM(lbbegin);
      if (lbbegin != -99)
	{
	  /* valid record */
	  valid[irec] = 1;
	  n_valid_rec++;
	} 
      else
	valid[irec] = 0;
    }
  *n_valid_rec_return = n_valid_rec;
  return 0;
  ERRBLKI;
}


int read_record_data_core(int fd, 
			  size_t data_offset, 
			  size_t disk_length, 
			  Byte_ordering byte_ordering, 
			  const void *int_hdr,
			  const void *real_hdr,
			  size_t nwords,
			  void *data_return)
{
  int pack;
  size_t packed_bytes, ipt, packed_words;
  void *packed_data;
  REAL mdi;

  packed_data = NULL;

  CKI(   lseek(fd, data_offset, SEEK_SET)   );
  pack = get_var_packing(int_hdr);

  if (pack == 0)
    {
      /* unpacked data -- read, and byte swap if necessary */
      ERRIF(   read_words(fd, data_return, nwords, byte_ordering)  != nwords);
    }
  else
    {
      /* PACKING IN USE */

      /* Complain if not REAL data. In cdunifpp, this test was applied only to Cray 32-bit packing,
       * but in fact also unwgdos assumes real, so apply to both packing types.
       */
      if (get_type(int_hdr) != real_type)
	{
	  error_mesg("Unpacking supported only for REAL type data");
	  ERR;
	}
      
      /* first allocate array and read in packed data */

      /* disk_length includes extra data, so subtract off */
      packed_bytes = disk_length - get_extra_data_length(int_hdr);

      /* ----------------------------
       * Possible alternative envisaged, that reads in slightly more data to
       * give tolerance against any situation where LBNREC does not include
       * the extra data.  However, this is now of dubious gain because reading in 
       * the extra data requires LBNREC to be used consistently so that the extra 
       * data can be found at the end of the record where the packed data that precedes
       * it is of variable length.
       * 
       *       packed_bytes = disk_length;
       *-----------------------------
       */


      /* An exception to the usual strategy heap memory management: no heaplist available, so use plain malloc
       * and be careful to free even if an error arose.
       *
       * (This is fairly much unavoidable: heaplist is attached to a File struct, but these exist only while 
       * parsing the file metadata, not while the read callback is executed.  A Python File object will exist
       * in the calling code, but that's not the same thing: it may have been instantiated with parse=False;
       * see the Python code.)
       */
      CKP(   packed_data = malloc(packed_bytes)  );
      ERRIF(   read(fd, packed_data, packed_bytes)  != packed_bytes   );

      /* NOW UNPACK ACCORDING TO PACKING TYPE (including byte swapping where necessary). */
      
      switch(pack)
	{
	case 1:
	  /* WGDOS */
	  
	  /* unwgdos routine wants to know number of native integers in input.
	   * input type might not be native int, so calculate:
	   */
	  mdi = get_var_real_fill_value(real_hdr);
	  
	  /* Note - even though we read in raw (unswapped) data from the file, we do not 
	   * byte swap prior to calling unwgdos, as the packed data contains a mixture
	   * of types of different lengths, so leave it to unwgdos() that knows about
	   * this and has appropriate byte swapping code.
	   */
	  CKI(   unwgdos(packed_data, packed_bytes, data_return, nwords, mdi)   );
	  
	  break;
	  
	case 2:
	  if (byte_ordering == REVERSE_ORDERING)
	    swap_bytes_sgl(packed_data, packed_bytes / 4);
	  
	  for (ipt = 0; ipt < nwords ; ipt++)
	    ((REAL*) data_return)[ipt] = ((float32_t *) packed_data)[ipt];
	  
	  break;
	  
	case 3:
	  error_mesg("GRIB unpacking not supported");
	  ERR;
	  
	  /* break; */

	case 4:
	  packed_words = packed_bytes / WORD_SIZE;
	  if (byte_ordering == REVERSE_ORDERING)
	    swap_bytes(packed_data, packed_words);
	  mdi = get_var_real_fill_value(real_hdr);
	  CKI(   unpack_run_length_encoded(packed_data, packed_words, data_return, nwords, mdi)   );
	  break;

	default:
	  SWITCH_BUG;
	}
      free(packed_data);
    }
  return 0;
 err:
  GRIPE;
  if (packed_data != NULL)
    free(packed_data);
  return -1;
}


int unpack_run_length_encoded(REAL *datain, INTEGER nin, REAL *dataout, INTEGER nout, REAL mdi)
{
  REAL *src, *dest, *end_src, *end_dest, data;
  INTEGER repeat;

  /* some pointers:
   * src and dest are current positions;
   * end_src and end_dest are the first position off the end of each array
   */
  src = datain;
  dest = dataout;
  end_src = src + nin;
  end_dest = dest + nout;

  /* syntax reminder: *p++ means first dereference p and then increment p
   */
  while (src < end_src && dest < end_dest)
    {
      data = *src++;
      if (data != mdi)
	*dest++ = data;
      else
	{
	  /* check we didn't read the MDI as the last item in the input */
	  ERRIF(src == end_src);

	  /* read in next word, round to nearest integer, and output MDI that many times 
	   * while checking we don't go beyond end of output data array
	   */
	  for (repeat = (INTEGER)(0.5 + *src++); repeat > 0 && dest < end_dest; repeat--)
	    *dest++ = mdi;

	  /* check we didn't reach end of output data array with copies of the MDI still to write 
	   * (or read in a negative repeat count)
	   */
	  ERRIF(repeat != 0);
	}
    }
  /* check we reached end of output data,
   * (not necessarily end of input data - it could be padded)
   */
  ERRIF (dest != end_dest);

  return 0;
  ERRBLKI;
}
