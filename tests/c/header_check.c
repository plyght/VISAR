/* Compile-only check that include/visar.h is valid C99 and C++17 (built by `zig build test`). */
#include "visar.h"

int visar_header_check(void) { return (int)sizeof(visar_moments_t); }
