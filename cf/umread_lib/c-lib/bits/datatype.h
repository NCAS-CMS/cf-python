#if defined(DOUBLE)

#define REAL float64_t
#define INTEGER int64_t
#define COMPILED_TYPE (double_precision)
#define WORD_SIZE 8

#elif defined(SINGLE)

#define REAL float32_t
#define INTEGER int32_t
#define COMPILED_TYPE  (single_precision)
#define WORD_SIZE 4

#endif
