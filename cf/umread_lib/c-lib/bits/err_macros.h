
/* error-checking macros */

/* these are to allow a compact way of incorporating error-checking of 
 * the return value of a function call, without obfuscating the basic purpose
 * of the line of code, which is executing the function call.
 *
 * CKI used for integer functions which return negative value on failure
 * CKP used for pointer functions which return NULL on failure
 * CKF for floats for good measure (probably not used)
 *
 * put the ERRBLK (or ERRBLKI or ERRBLKP) at the end of the subroutine
 */

#define FLT_ERR -1e38

#ifdef DEBUG
#define ERR abort();
#else
/* ERR: unconditional branch */
#define ERR goto err;
#endif

#define CKI(i) if ((i) < 0){ ERR }
#define CKP(p) if ((p) == NULL){ ERR }
#define CKF(f) if ((f) == FLT_ERR){ ERR }

/* ERRIF: conditional branch */
#define ERRIF(i) if (i){ ERR }

#define GRIPE gripe(__func__);
#define SWITCH_BUG switch_bug(__func__); ERR;
#define ERRBLK(rtn) err: GRIPE; return (rtn);
#define ERRBLKI ERRBLK(-1);
#define ERRBLKP ERRBLK(NULL);
#define ERRBLKF ERRBLK(FLT_ERR);
