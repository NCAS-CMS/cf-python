/* ----------------------------------------------------------- */
/* PP header interpretation */

#define N_INT_HDR 45
#define N_REAL_HDR 19
#define N_HDR (N_INT_HDR + N_REAL_HDR)

#define INDEX_LBYR    0
#define INDEX_LBMON   1
#define INDEX_LBDAT   2
#define INDEX_LBHR    3
#define INDEX_LBMIN   4
#define INDEX_LBDAY   5
#define INDEX_LBYRD   6
#define INDEX_LBMOND  7
#define INDEX_LBDATD  8
#define INDEX_LBHRD   9
#define INDEX_LBMIND  10
#define INDEX_LBDAYD  11
#define INDEX_LBTIM   12
#define INDEX_LBFT    13
#define INDEX_LBLREC  14
#define INDEX_LBCODE  15
#define INDEX_LBHEM   16
#define INDEX_LBROW   17
#define INDEX_LBNPT   18
#define INDEX_LBEXT   19
#define INDEX_LBPACK  20
#define INDEX_LBREL   21
#define INDEX_LBFC    22
#define INDEX_LBCFC   23
#define INDEX_LBPROC  24
#define INDEX_LBVC    25
#define INDEX_LBRVC   26
#define INDEX_LBEXP   27
#define INDEX_LBBEGIN 28
#define INDEX_LBNREC  29
#define INDEX_LBPROJ  30
#define INDEX_LBTYP   31
#define INDEX_LBLEV   32
#define INDEX_LBRSVD1 33
#define INDEX_LBRSVD2 34
#define INDEX_LBRSVD3 35
#define INDEX_LBRSVD4 36
#define INDEX_LBSRCE  37
#define INDEX_LBUSER1 38
#define INDEX_LBUSER2 39
#define INDEX_LBUSER3 40
#define INDEX_LBUSER4 41
#define INDEX_LBUSER5 42
#define INDEX_LBUSER6 43
#define INDEX_LBUSER7 44
#define INDEX_BULEV    0
#define INDEX_BHULEV   1
#define INDEX_BRSVD3   2
#define INDEX_BRSVD4   3
#define INDEX_BDATUM   4
#define INDEX_BACC     5
#define INDEX_BLEV     6
#define INDEX_BRLEV    7
#define INDEX_BHLEV    8
#define INDEX_BHRLEV   9
#define INDEX_BPLAT   10
#define INDEX_BPLON   11
#define INDEX_BGOR    12
#define INDEX_BZY     13
#define INDEX_BDY     14
#define INDEX_BZX     15
#define INDEX_BDX     16
#define INDEX_BMDI    17
#define INDEX_BMKS    18

#define LOOKUP(rec, index) (((INTEGER *)((rec)->int_hdr))[index])
#define RLOOKUP(rec, index) (((REAL *)((rec)->real_hdr))[index])

/* ----------------------------------------------------------- */
