/* testmp.cl, test the ul and cofact on ocl */

#include "include/ocl/tests.hl"


#if TEST_SET & TEST_32
#include "include/ocl/ul32_test.hl"
#endif
#if TEST_SET & TEST_64
#include "include/ocl/ul64_test.hl"
#endif
#if TEST_SET & TEST_96
#include "include/ocl/ul96_test.hl"
#endif
#if TEST_SET & TEST_128
#include "include/ocl/ul128_test.hl"
#endif
#if TEST_SET & TEST_160
#include "include/ocl/ul160_test.hl"
#endif
#if TEST_SET & TEST_192
#include "include/ocl/ul192_test.hl"
#endif
#if TEST_SET & TEST_224
#include "include/ocl/ul224_test.hl"
#endif
#if TEST_SET & TEST_256
#include "include/ocl/ul256_test.hl"
#endif

