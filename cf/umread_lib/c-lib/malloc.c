#include <stdlib.h>
#include <string.h>

#include "umfileint.h"

/* Malloc functions 
 *
 * These routines are closely integrated with the link list functions; they
 * are called with a linked list "heaplist"; the malloc_ function adds the
 * newly allocated pointer to this list, and the free_ function removes it
 * from the list.  (They can also be called with NULL in which case they
 * ignore the heaplist; this is necessary when allocating or freeing memory
 * for the heaplist itself.)
 *
 * The idea is that all the dynamically memory allocation associated with
 * a given file should be through these functions.  Then whenever the file
 * is closed properly or because an error condition gave an abort, the
 * memory can be freed without needing complicated tests to work out what
 * has been allocated: just go through the linked list freeing pointers.
 *
 * NOTE: this routine now allocates a little more memory than requested,
 * and saves the pointer to the list element on the heaplist at the start,
 * before returning to the calling routine the pointer to the actual block
 * of memory that the caller is interested in.  This ensures that when freeing
 * the memory, list_del_by_listel can be used instead of list_del, giving
 * efficiency gains.
 */

static const int extrasize = sizeof(List_element*);

void *malloc_(size_t size, List *heaplist){

  void *ptr;
  List_element* *elp;

  if (size == 0)
    return NULL;

  /* The only call to malloc in umfile c-lib (except in unwgdos.c and packed_data in read.c) */
  ptr = malloc(size + extrasize);

  if (ptr == NULL)
    {
      error_mesg("unable to allocate of %d bytes of memory",
		 size);
    }
  else
    {
      /* copy the pointer so we can use the start of the address to store
       * the List_element* 
       */
      elp = (List_element**) ptr;

      /* Now increment the pointer (to after our stored List_element*) to give
       * what the calling routine calling routine sees the start of memory
       * (cast to char* for ptr arithmetic.  Do this *before* storing it
       * on the heaplist, because pointers on will be freed with free
       */
      ptr = (void*) ((char*)ptr + extrasize);

      if (heaplist != NULL) 
	{
	  CKI(   list_add(heaplist, ptr, NULL)   );

	  /* we just added to the list, so that heaplist->last will
	   * contain pointer to the relevant List_element*
	   */
	  *elp = heaplist->last;
	}
      else
	*elp = NULL;
    }

  return ptr;
  ERRBLKP;
}


void *dup_(const void *inptr, size_t size, List *heaplist)
{  
  void *outptr;
  
  CKP(   outptr = malloc_(size, heaplist)   );
  memcpy(outptr, inptr, size);
  return outptr;
  ERRBLKP;
}


int free_(void *ptr, List *heaplist)
{
  List_element *el;

  CKP(ptr);
  /* first subtract off the extra size we added (see malloc_) */
  ptr = (void*) ((char*) ptr - extrasize);

  /* this is our list element */
  el = * (List_element**) ptr;
  
  /* The only call to free in umfile c-lib 
   *  (except in unwgdos.c and packed_data in read.c)
   */
  free(ptr);

  /*   printf ("free: %p\n",ptr);   */
  if (heaplist != NULL)
    CKI(  list_del_by_listel(heaplist, el, NULL)  );

  return 0;
  ERRBLKI;
}


int free_all(List *heaplist) 
{
  return list_free(heaplist, 1, NULL);
}
