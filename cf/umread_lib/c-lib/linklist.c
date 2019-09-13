#include "umfileint.h"

/* LINKED LIST FUNCTIONS */

void *list_new(List *heaplist)
{
  List *list;
  CKP(   list = malloc_(sizeof(List), heaplist)   );
  list->first = NULL;
  list->last = NULL;
  list->n = 0;
  return list;
  ERRBLKP;
}

/* This function frees a list; 
 * Set free_ptrs if the pointers which have been explicitly stored on the 
 * list (2nd argument to list_add) are to be freed, not just the pointers
 * which are implicit in the linked list structure.  NB there is no further
 * recursion, in the sense that if the stored pointers are to datatypes which
 * contain further pointers then these may have to be freed explicitly.
 */
int list_free(List *list, int free_ptrs, List *heaplist)
{
  List_element *p, *next;
  CKP(list);
  for (p = list->first ; p != NULL ; p = next) 
    {
      next = p->next;
      if (free_ptrs)
	CKI(  free_(p->ptr, heaplist)  );
      CKI(  free_(p,heaplist)  );
    }
  CKI(  free_(list, heaplist)  );
  return 0;
  ERRBLKI;
}


int list_size(const List *list)
{
  CKP(list);
  return list->n;
  ERRBLKI;
}


int list_add(List *list, void *ptr, List *heaplist)
{
  List_element *el;
  CKP(list);
  CKP(   el = malloc_(sizeof(List_element), heaplist)   );
  list->n ++;
  el->ptr = ptr;
  el->next = NULL;
  if (list->first == NULL) 
    {
      el->prev = NULL;
      list->first = list->last = el;
    }
  else 
    {
      list->last->next = el;
      el->prev = list->last;
      list->last = el;
    }
  return 0;
  ERRBLKI;
}

/* list_add_or_find takes a pointer to an item and tries to find it on the
 * list, using the comparision function as in list_find.
 *
 * If it already exists, it changes the item to point to the old value, and
 *    calls the supplied function (if non-null) to free the item.  If it does
 *    not exist, it adds the item to the list.
 *
 * Return values: 
 *   0  time already existed in axis
 *   1  time has been added to axis
 *  -1  an error occurred (probably in memory allocation)
 *
 * NOTE: the return value of this function may be tested with the CKI() macro.
 * Do not add non-error cases with negative return values.
 *
 * NOTE 2: The item is formally declared as a void* but the
 *         thing pointed to should itself be a pointer (to heap memory),
 *         so you should pass in a foo** of some sort.  The only reason for
 *         not declaring as void** is that void* has the special property of
 *         being treated by the compiler as a generic pointer hence no
 *         warnings about incompatible pointer type
 */
int list_add_or_find(List *list, 
		     void *item_in,
		     int (*compar)(const void *, const void *), 
		     int matchval, 
		     free_func free_function,
		     int *index_return,
		     List *heaplist)
{
  void *oldptr;
  void **item = (void**) item_in;

  if ((oldptr = list_find(list, *item, compar,
			  matchval, index_return)) != NULL) 
    {
      if (free_function != NULL)
	CKI(  free_function(*item, heaplist)  );
      *item = oldptr;
      return 0;      
    } 
  else
    {
      CKI(  list_add(list, *item, heaplist)  );
      if (index_return != NULL)
	*index_return = list_size(list) - 1;
      return 1;
    }
  ERRBLKI;
}


/* call list_del to find a pointer ("ptr" element contained within the
 * listel structure) on the list, and then delete that element from the list,
 * or call list_del_by_listel directly (more efficient) if you already
 * have the listel structure pointer for what you want to delete.
 */

int list_del(List *list, void *ptr, List *heaplist)
{
  List_element *p;
  CKP(list);
  for (p = list->first; p != NULL; p = p->next)
    if (p->ptr == ptr)
      return list_del_by_listel(list, p, heaplist);

  /* if what we're trying to remove is not found, fall through
   * to error exit
   */
  ERRBLKI;
}


int list_del_by_listel(List *list, List_element *p, List *heaplist)
{
  List_element *prev, *next;
  next = p->next;
  prev = p->prev;
  if (next != NULL) next->prev = prev;
  if (prev != NULL) prev->next = next;
  if (p == list->first) list->first = next;
  if (p==list->last) list->last = prev;
  CKI(  free_(p, heaplist)  );
  list->n --;
  return 0;
  ERRBLKI;
}

/* call list_startwalk before a sequence of calls to list_walk */
int list_startwalk(const List *list, List_handle *handle)
{
  CKP(list);
  CKP(handle);
  handle->current = list->first;
  handle->list = list;
  return 0;
  ERRBLKI;
}


/* list_walk:
 *   designed to be called repeatedly, and returns the next element of the
 * list each time (but must not call either add or del between calls)
 *
 * (Set return_listel to nonzero to return the list element structure rather
 *  than the pointer it contains.  This is just so that if you put null
 *  pointers on the list you can tell the difference from end of list.)
 */
void *list_walk(List_handle *handle, int return_listel)
{
  void *ptr;
  CKP(handle);
  if (handle->current == NULL)
    return NULL;
  else 
    {
      ptr = (return_listel) ? (void *) handle->current : handle->current->ptr;
      handle->current = handle->current->next;
      return ptr;
    }
  ERRBLKP;
}

/*------------------------------------------------------------------------------*/
/* list_find: find first item on the list matching specified item, where
 * "compar" is the matching function, and "matchval" is return value from
 * compar in the event of a match
 *
 * The pointer index_return, if non-NULL, is used to return the index number
 * on the list (set to -1 if not found).
 */
void *list_find(List *list,
		const void *item,
		int (*compar)(const void *, const void *), 
		int matchval,
		int *index_return) 
{
  int index;
  List_element *listel;
  List_handle handle;
  void *ptr;

  list_startwalk(list, &handle);
  index = 0;
  while ((listel = list_walk(&handle, 1)) != NULL)
    {
      ptr = listel->ptr;
      if (compar(&item, &ptr) == matchval) 
	{
	  if (index_return != NULL)
	    *index_return = index;
	  return ptr;
	}
      index++;
    }
  if (index_return != NULL)
    *index_return = -1;
  return NULL;
}


int list_copy_to_ptr_array(const List *list, 
			   int *n_return, 
			   void *ptr_array_return,
			   List *heaplist)
{
  int n;
  List_handle handle;
  List_element *listel;
  void **ptr_array, **p;

  n = list_size(list);
  if (n == 0)
    ptr_array = NULL;
  else
    {
      CKP(  ptr_array = malloc_(n * sizeof(void *), heaplist)  );
      p = ptr_array;
  
      list_startwalk(list, &handle);
      while ((listel = list_walk(&handle, 1)) != NULL)
	{
	  *p = listel->ptr;
	  p++;
	}  
    }
  *n_return = n;
  * (void ***) ptr_array_return = ptr_array;
  return 0;
  ERRBLKI;
}
