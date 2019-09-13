#include "umfile.h"

/*---------------------------*/

#include "bits/constants.h"
#include "bits/datatype.h"
#include "bits/ordering.h"
#include "bits/typedefs.h"
#include "bits/type_indep_protos.h"
#include "bits/pp_header.h"
#include "bits/err_macros.h"

/* ----------------------------------------------------------- */

#if defined(SINGLE) || defined(DOUBLE)
#include "bits/type_dep_protos.h"
#else
#define WITH_LEN(x) x ## _sgl
#include "bits/type_dep_entry_protos.h"
#undef WITH_LEN
#define WITH_LEN(x) x ## _dbl
#include "bits/type_dep_entry_protos.h"
#undef WITH_LEN
#endif
