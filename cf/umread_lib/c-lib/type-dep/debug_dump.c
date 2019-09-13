#include "umfileint.h"

void debug_dump_all_headers(File *file)
{
  int irec;
  Rec *rec;

  debug("fd = %d", file->fd);
  debug("format = %d", file->file_type.format);
  debug("byte_ordering = %d", file->file_type.byte_ordering);
  debug("word_size = %d", file->file_type.word_size);
  debug("nrec = %d", file->internp->nrec);
  debug("");

  for (irec = 0; irec < file->internp->nrec; irec++)
    {
      rec = file->internp->recs[irec];
      debug("rec %d", irec);
      debug("header_offset = %d", rec->header_offset);
      debug("data_offset = %d", rec->data_offset);
      debug("disk_length = %d", rec->disk_length);

      debug("LBYR = %d", LOOKUP(rec, INDEX_LBYR));
      debug("LBMON = %d", LOOKUP(rec, INDEX_LBMON));
      debug("LBDAT = %d", LOOKUP(rec, INDEX_LBDAT));
      debug("LBHR = %d", LOOKUP(rec, INDEX_LBHR));
      debug("LBMIN = %d", LOOKUP(rec, INDEX_LBMIN));
      debug("LBDAY = %d", LOOKUP(rec, INDEX_LBDAY));
      debug("LBYRD = %d", LOOKUP(rec, INDEX_LBYRD));
      debug("LBMOND = %d", LOOKUP(rec, INDEX_LBMOND));
      debug("LBDATD = %d", LOOKUP(rec, INDEX_LBDATD));
      debug("LBHRD = %d", LOOKUP(rec, INDEX_LBHRD));
      debug("LBMIND = %d", LOOKUP(rec, INDEX_LBMIND));
      debug("LBDAYD = %d", LOOKUP(rec, INDEX_LBDAYD));
      debug("LBTIM = %d", LOOKUP(rec, INDEX_LBTIM));
      debug("LBFT = %d", LOOKUP(rec, INDEX_LBFT));
      debug("LBLREC = %d", LOOKUP(rec, INDEX_LBLREC));
      debug("LBCODE = %d", LOOKUP(rec, INDEX_LBCODE));
      debug("LBHEM = %d", LOOKUP(rec, INDEX_LBHEM));
      debug("LBROW = %d", LOOKUP(rec, INDEX_LBROW));
      debug("LBNPT = %d", LOOKUP(rec, INDEX_LBNPT));
      debug("LBEXT = %d", LOOKUP(rec, INDEX_LBEXT));
      debug("LBPACK = %d", LOOKUP(rec, INDEX_LBPACK));
      debug("LBREL = %d", LOOKUP(rec, INDEX_LBREL));
      debug("LBFC = %d", LOOKUP(rec, INDEX_LBFC));
      debug("LBCFC = %d", LOOKUP(rec, INDEX_LBCFC));
      debug("LBPROC = %d", LOOKUP(rec, INDEX_LBPROC));
      debug("LBVC = %d", LOOKUP(rec, INDEX_LBVC));
      debug("LBRVC = %d", LOOKUP(rec, INDEX_LBRVC));
      debug("LBEXP = %d", LOOKUP(rec, INDEX_LBEXP));
      debug("LBBEGIN = %d", LOOKUP(rec, INDEX_LBBEGIN));
      debug("LBNREC = %d", LOOKUP(rec, INDEX_LBNREC));
      debug("LBPROJ = %d", LOOKUP(rec, INDEX_LBPROJ));
      debug("LBTYP = %d", LOOKUP(rec, INDEX_LBTYP));
      debug("LBLEV = %d", LOOKUP(rec, INDEX_LBLEV));
      debug("LBRSVD1 = %d", LOOKUP(rec, INDEX_LBRSVD1));
      debug("LBRSVD2 = %d", LOOKUP(rec, INDEX_LBRSVD2));
      debug("LBRSVD3 = %d", LOOKUP(rec, INDEX_LBRSVD3));
      debug("LBRSVD4 = %d", LOOKUP(rec, INDEX_LBRSVD4));
      debug("LBSRCE = %d", LOOKUP(rec, INDEX_LBSRCE));
      debug("LBUSER1 = %d", LOOKUP(rec, INDEX_LBUSER1));
      debug("LBUSER2 = %d", LOOKUP(rec, INDEX_LBUSER2));
      debug("LBUSER3 = %d", LOOKUP(rec, INDEX_LBUSER3));
      debug("LBUSER4 = %d", LOOKUP(rec, INDEX_LBUSER4));
      debug("LBUSER5 = %d", LOOKUP(rec, INDEX_LBUSER5));
      debug("LBUSER6 = %d", LOOKUP(rec, INDEX_LBUSER6));
      debug("LBUSER7 = %d", LOOKUP(rec, INDEX_LBUSER7));
      debug("BULEV = %f", RLOOKUP(rec, INDEX_BULEV));
      debug("BHULEV = %f", RLOOKUP(rec, INDEX_BHULEV));
      debug("BRSVD3 = %f", RLOOKUP(rec, INDEX_BRSVD3));
      debug("BRSVD4 = %f", RLOOKUP(rec, INDEX_BRSVD4));
      debug("BDATUM = %f", RLOOKUP(rec, INDEX_BDATUM));
      debug("BACC = %f", RLOOKUP(rec, INDEX_BACC));
      debug("BLEV = %f", RLOOKUP(rec, INDEX_BLEV));
      debug("BRLEV = %f", RLOOKUP(rec, INDEX_BRLEV));
      debug("BHLEV = %f", RLOOKUP(rec, INDEX_BHLEV));
      debug("BHRLEV = %f", RLOOKUP(rec, INDEX_BHRLEV));
      debug("BPLAT = %f", RLOOKUP(rec, INDEX_BPLAT));
      debug("BPLON = %f", RLOOKUP(rec, INDEX_BPLON));
      debug("BGOR = %f", RLOOKUP(rec, INDEX_BGOR));
      debug("BZY = %f", RLOOKUP(rec, INDEX_BZY));
      debug("BDY = %f", RLOOKUP(rec, INDEX_BDY));
      debug("BZX = %f", RLOOKUP(rec, INDEX_BZX));
      debug("BDX = %f", RLOOKUP(rec, INDEX_BDX));
      debug("BMDI = %f", RLOOKUP(rec, INDEX_BMDI));
      debug("BMKS = %f", RLOOKUP(rec, INDEX_BMKS));
    }
}
