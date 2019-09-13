/* int_missing_data is convention in input file */
#define INT_MISSING_DATA -32768

/* for float comparisons */
#if defined(SINGLE)
#define REAL_TOLERANCE 1e-5
#elif defined(DOUBLE)
#define REAL_TOLERANCE 1e-13
#endif
