#include <stdint.h>

extern "C" {

void checksum_zero(int32_t *out, int32_t worker_id) {
    out[0] = 0;          // sum
    out[1] = 0;          // count
    out[2] = worker_id;  // first value, overwritten on first chunk
    out[3] = worker_id;  // last value
}

void checksum_accum(int32_t *half, int32_t *out, int32_t chunk_idx) {
    for (int i = 0; i < 2048; i++) {
        int32_t value = half[i];
        out[0] += value;
        if (out[1] == 0 && i == 0) {
            out[2] = value;
        }
        out[3] = value;
        out[1] += 1;
    }
    out[0] += chunk_idx;
}

}
