#include "umfileint.h"

#define DO_SWAP(x, y) {t = p[x]; p[x] = p[y]; p[y] = t;}

void swap_bytes_sgl(void *ptr, size_t num_words)
{
  int i;
  char *p;
  char t;

  p = (char*) ptr;
  for (i = 0; i < num_words; i++)
    {
      DO_SWAP(3, 0);
      DO_SWAP(2, 1);
      p += 4;  
    }
}

void swap_bytes_dbl(void *ptr, size_t num_words)
{
  int i;
  char *p;
  char t;

  p = (char*) ptr;
  for (i = 0; i < num_words; i++)
    {
      DO_SWAP(7, 0);
      DO_SWAP(6, 1);
      DO_SWAP(5, 2);
      DO_SWAP(4, 3);
      p += 8;      
    }
}


