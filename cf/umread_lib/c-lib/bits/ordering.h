#include <endian.h>
#if BYTE_ORDER == LITTLE_ENDIAN
#define NATIVE_ORDERING little_endian
#define REVERSE_ORDERING big_endian
#elif BYTE_ORDER == BIG_ENDIAN
#define NATIVE_ORDERING big_endian
#define REVERSE_ORDERING little_endian
#else
#error BYTE_ORDER not defined properly
#endif
