/* datatype dependent parts of the test code */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "umfileint.h"

//Rec *rec_alloc()
//{
//  Rec *rec;
//  rec = xmalloc(sizeof(Rec));
//  rec->internp = xmalloc(sizeof(struct _Rec));
//  rec->int_hdr = xmalloc(45 * sizeof(INTEGER));
//  rec->real_hdr = xmalloc(19 * sizeof(REAL));
//  return rec;
//}
//
//void rec_free(Rec *rec)
//{
//  _rec_internals_free(rec);
//  {
//    INTEGER *ihdr = rec->int_hdr;
//    REAL *rhdr = rec->real_hdr;
//    printf("freeing rec with ihdr=%d %d... rhdr=%f %f...\n", 
//	   ihdr[0], ihdr[1], rhdr[0], rhdr[1]);
//  }
//  xfree(rec->int_hdr);
//  xfree(rec->real_hdr);
//  xfree(rec);
//}
//
//
//Rec *rec_create_dummy(int k)
//{
//  int i;
//  Rec *rec;
//  rec = rec_alloc();
//  for (i = 0; i < 45 ; i++)
//    ((INTEGER *)rec->int_hdr)[i] = k * 100 + i;
//  for (i = 0; i < 19 ; i++)
//    ((REAL *)rec->real_hdr)[i] = k + i / 100.;
//  rec->header_offset = k;
//  rec->data_offset = 500 + k;
//  rec->internp->blahblah = 200 + k;
//  return rec;
//}
//
//int get_type_and_length_dummy(const void *int_hdr, Data_type *type_rtn, size_t *num_words_rtn)
//{
//  const INTEGER *int_hdr_4 = int_hdr;
//  *num_words_rtn = int_hdr_4[0];
//  *type_rtn = real_type;
//  return 0;
//}
//
void read_record_data_dummy(size_t nwords, 
				       void *data_return)
{
  int i;
  REAL *data_return_4 = data_return;
  for (i = 0; i < nwords; i++)
    {
      data_return_4[i] = i / 100.;    
    }
} 


// int read_record_data_core(int fd, 
// 				    size_t data_offset, 
// 				    size_t disk_length,
// 				    Byte_ordering byte_ordering, 
// 				    int word_size, 
// 				    const void *int_hdr,
// 				    const void *real_hdr,
// 				    size_t nwords, 
// 				    void *data_return)
// {
//   int i;
//   assert(byte_ordering == little_endian);
//   assert(word_size == 4);
// 
//   printf("start of int header seen in read_record_data_dummy():");
//   for (i = 0; i < 5; i++)
//     printf("  %d", ((INTEGER *) int_hdr)[i]);
//   printf("\n");
//   printf("start of real header seen in read_record_data_dummy():");
//   for (i = 0; i < 5; i++)
//     printf("  %f", ((REAL *) real_hdr)[i]);
//   printf("\n");
// 
//   read_record_data_dummy(nwords, data_return);
//   return 0;
// }
// 

#ifdef MAIN
int main()
{
  int i, j, k, nrec;
  int fd;
  File *file;
  File_type file_type;
  Var *var;
  Rec *rec;
  REAL *data;
  int word_size;
  Data_type data_type;
  size_t nwords, nbytes;
 
  fd = 3;
  detect_file_type(fd, &file_type);
  printf("word size = %d\n", file_type.word_size);
  
  file = file_parse(fd, file_type);
  for (i = 0; i < file->nvars; i++)
    {
      printf("var %d\n", i);
      var = file->vars[i];
      printf("nz = %d, nt = %d\n", var->nz, var->nt);
      nrec = var->nz * var->nt;
      for (j = 0; j < nrec; j++)
	{
	  rec = var->recs[j];
	  printf("var %d rec %d\n", i, j);
	  printf("int header:\n");
	  for (k = 0; k < 45; k++)
	    printf(" ihdr[%d] = %d\n", k, ((INTEGER *)rec->int_hdr)[k]);
	  printf("real header:\n");
	  for (k = 0; k < 19; k++)
	    printf(" rhdr[%d] = %f\n", k, ((REAL *)rec->real_hdr)[k]);

	  word_size = file_type.word_size;
	  get_type_and_length(word_size, rec->int_hdr, &data_type, &nwords);
	  nbytes = word_size * nwords;;
	  printf("data (%ld items)\n", nwords);
	  data = xmalloc(nbytes * sizeof(float));
	  read_record_data(fd, 
			   rec->data_offset,
			   file_type.byte_ordering,
			   file_type.word_size,
			   rec->int_hdr,
			   rec->real_hdr,
			   nwords,
			   data);
	  for (k = 0; k < nwords; k++)
	    printf(" data[%d] = %f\n", k, data[k]);
	  xfree(data);	  
	}
    }
  return 0;
}
#endif
