#include <stdint.h>

extern "C" {

void checksum_chunk(int32_t *data, int32_t *out, int32_t chunk_idx, int32_t n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    out[chunk_idx] = sum;
}

}
