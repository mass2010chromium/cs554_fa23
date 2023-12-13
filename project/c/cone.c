#include "cone.h"
#include "types.h"
#include "math/vectorops.h"

#include <string.h>

// Copy from SCS source code.
// but more readable and using vectorops.
// NOTE: "Z" coordinate is the first coordinate in the array.
static void project_soc(numeric* x, size_t n_dims) {
    if (n_dims == 0) {
        return;
    }
    if (n_dims == 1) {
        if (x[0] < 0) { x[0] = 0; }
        return;
    }
    numeric z = x[0];
    numeric x_len = __vo_norm(&x[1], n_dims-1);

    if (x_len <= z) {
        // We are in the cone. No change needed.
        return;
    }
    if (x_len <= -z) {
        // We are beyond the cone's "support". Return zero.
        memset(x, 0, n_dims*sizeof(numeric));
    }
    else {
        // middle case. Perform cone projection
        numeric alpha = (z + x_len) / 2.0;
        x[0] = alpha;
        __vo_mul(&x[1], &x[1], alpha / x_len, n_dims-1);
    }
}

void project_to_socs(numeric* x, size_t* cone_dims, size_t num_cones) {
    numeric* x_ptr = x;
    for (size_t i = 0; i < num_cones; ++i) {
        project_soc(x_ptr, cone_dims[i]);
        x_ptr += cone_dims[i];
    }
}
