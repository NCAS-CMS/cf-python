#if defined(LINUX)
#include <endian.h>
#elif defined(OSX)
#include <machine/endian.h>
#else
#error build type not specified
#endif
#if BYTE_ORDER == LITTLE_ENDIAN
#define NATIVE_ORDERING little_endian
#define REVERSE_ORDERING big_endian
#elif BYTE_ORDER == BIG_ENDIAN
#define NATIVE_ORDERING big_endian
#define REVERSE_ORDERING little_endian
#else
#error BYTE_ORDER not defined properly
#endif
