#include <stdint.h>

namespace {

constexpr int WORDS_PER_RECORD = 4;

} // namespace

extern "C" {

void record_descriptor_chunk(int32_t *chunk, int32_t *output, int32_t chunk_idx, int32_t chunk_len) {
    int32_t sum = 0;
    for (int idx = 0; idx < chunk_len; idx++) {
        sum += chunk[idx];
    }

    int32_t base = chunk_idx * WORDS_PER_RECORD;
    output[base + 0] = 0x55000000 | chunk_idx;
    output[base + 1] = chunk[0];
    output[base + 2] = chunk[chunk_len - 1];
    output[base + 3] = sum;
}

} // extern "C"
