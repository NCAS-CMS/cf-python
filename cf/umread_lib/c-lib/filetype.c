  /* Routines for auto-determining file type from the start of the file 
   * contents.
   * 
   * =================
   * Fields file tests
   * =================
   *
   * These are done first. We test the second word, which should be the
   * submodel ID - is this 1, 2 or 4?
   *
   * ==> Could the fields file test give a false +ve with PP file?
   *
   *     Test for fields file only true if the first 16 bytes, when viewed 
   *     as 4 32-bit integers, are one of the following:
   *
   *     bytes 1-4    bytes 5-8    bytes 9-12   bytes 13-16
   *     ---------------------------------------------------
   *         any         any          0         1/2/4(BE)   <-- 64-bit BE FF
   *         any         any      1/2/4(LE)        0        <-- 64-bit LE FF
   *         any      1/2/4(BE)      any          any       <-- 32-bit BE FF
   *         any      1/2/4(LE)      any          any       <-- 32-bit LE FF
   *   
   *     For PP files, we in fact have:
   *   
   *          0       512/1024(BE)    0          lbyr(BE)  <--- 64-bit BE PP
   *      512/1024(LE)      0      lbyr(LE)         0      <--- 64-bit LE PP
   *       256/512(BE)  lbyr(BE)   lbmon(BE)    lbdat(BE)  <--- 32-bit BE PP
   *       256/512(LE)  lbyr(LE)   lbmon(LE)    lbdat(LE)  <--- 32-bit LE PP
   *   
   *     Possible false positives:
   *
   *     - any PP with lbyr=1/2/4 looks like FF of same length and endianness
   *     - 32-bit BE PP with lbmon=0, lbdat=1/2/4 looks like 64-bit BE FF
   *     - 32-bit LE PP with lbmon=1/2/4, lbdat=0 looks like 64-bit LE FF
   *   
   *      Do we care about these cases?
   *          lbyr=1/2/4: probably NO
   *          lbmon=0, lbdat non-zero: probably NO
   *          lbmon=1/2/4, lbdat=0 - possible monthly climatology? <== YES
   *
   *      Always option for user to force file type, but:
   *        **FIXME**: additional test could help.
   *
   * ========
   * PP tests
   * ========
   * 
   * If the fields-file test is false, then test for types of plain PP file.
   * Here we test the first word, which should be record length (put there by 
   * fortran).
   *
   * Check first for a 64-bit PP file, but in addition to the first word being
   * a valid possibility, this must also pass the stringent test of every
   * other 32-bit value being zero throughout the first 14 64-bit words,
   * although because of endianness issues we accept the sequence of
   * alternating zeros to start either at the first or second 32-bit value.
   * The point is that for a true 64-bit file, these should all be small
   * integers, so the most significant bytes will be 0.  (The first possibly
   * large integer is the 15th: LBLREC.)  However, for a 32-bit file this test
   * will span the first 28 elements.  Even if the date elements (first 12
   * words) are all 0, LBROW (18) and LBNPT (19) should both be non-zero, so
   * both the set of even-positioned integers and the set of odd-positioned
   * integers will each contain at least one non-zero value, and the test will
   * fail. 
   * 
   * If the 64-bit tests fail, try 32-bit.
   */

#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "umfileint.h"

/* values passed to valid_um_word2 and valid_pp_word1 could be 32 or
 * 64-bit.  Declare as longer of these two (int64_t), and shorter will be
 * accommodated also.
 */

static int valid_um_word2(int64_t val)
{
  /* second word should be 1,2 or 4, reflecting model ID in fixed length
     header */
  return (val == 1 || val == 2 || val == 4);
}

static int valid_pp_word1(int64_t val, int wsize)
{
  /* first word should be integer from Fortan representing length of header
     record */
  return (val == 64 * wsize || val == 128 * wsize);
}

/* tests whether sequence of integers has every other value = 0, but 
 * only when starting at first value
 */
static int is_alternating_zeros_without_offset(int32_t *vals, int num_pairs)
{
  int i;
  int32_t *p;
  p = vals;
  for (i = 0; i < num_pairs; i++)
    {
      if (*p != 0) return 0;
      p += 2;
    }
  return 1;
}

/* tests whether sequence of integers has every other value = 0, but 
 * can either be when starting at first or second value
 */
static int is_alternating_zeros(int32_t *vals, int num_pairs)
{
  return (is_alternating_zeros_without_offset(vals, num_pairs) || 
	  is_alternating_zeros_without_offset(vals + 1, num_pairs));
}

#define N_PAIRS 14
int detect_file_type_(int fd, File_type *file_type)
{
  int32_t data4[2 * N_PAIRS], data4s[2];
  int64_t data8[2], data8s[2];

  /* read and store first 24 4-byte words
   * and store first two integers of this according to possible suppositions 
   * of 4- or 8- byte, and of native or swapped byte ordering
   */
  lseek(fd, 0, SEEK_SET);
  if(read(fd, data4, 8 * N_PAIRS) != 8 * N_PAIRS) return 1;

  memcpy(data8, data4, 16);

  memcpy(data4s, data4, 8);
  swap_bytes_sgl(data4s, 2);

  memcpy(data8s, data4, 16);
  swap_bytes_dbl(data8s, 2);


  /* --- Fields file cases -- */

  if (valid_um_word2(data4[1]))
    {
      file_type->format = fields_file;
      file_type->byte_ordering = NATIVE_ORDERING;
      file_type->word_size = 4;
    }
  else if (valid_um_word2(data8[1]))
    {
      file_type->format = fields_file;
      file_type->byte_ordering = NATIVE_ORDERING;
      file_type->word_size = 8;
    }
  else if (valid_um_word2(data4s[1]))
    {
      file_type->format = fields_file;
      file_type->byte_ordering = REVERSE_ORDERING;
      file_type->word_size = 4;
    }
  else if (valid_um_word2(data8s[1]))
    {
      file_type->format = fields_file;
      file_type->byte_ordering = REVERSE_ORDERING;
      file_type->word_size = 8;
    }

  /* --- Plain PP cases -- */

  else if (valid_pp_word1(data8[0], 8) && is_alternating_zeros(data4, N_PAIRS))
    {
      file_type->format = plain_pp;
      file_type->byte_ordering = NATIVE_ORDERING;
      file_type->word_size = 8;
    }
  else if (valid_pp_word1(data8s[0], 8) && is_alternating_zeros(data4, N_PAIRS))
    {
      file_type->format = plain_pp;
      file_type->byte_ordering = REVERSE_ORDERING;
      file_type->word_size = 8;
    }
  else if (valid_pp_word1(data4[0], 4))
    {
      file_type->format = plain_pp;
      file_type->byte_ordering = NATIVE_ORDERING;
      file_type->word_size = 4;
    }
  else if (valid_pp_word1(data4s[0], 4))
    {
      file_type->format = plain_pp;
      file_type->byte_ordering = REVERSE_ORDERING;
      file_type->word_size = 4;
    }
  else
    {
      /* type not identified */
      return 1;
    }
  return 0;
}
