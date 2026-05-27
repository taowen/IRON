#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int WORDS_PER_RECORD = 4;

} // namespace

extern "C" {

void record_patch_chunk(bfloat16 *chunk, int32_t *output, int32_t row, int32_t chunk_bf16) {
    uint16_t *raw = reinterpret_cast<uint16_t *>(chunk);
    int32_t sum = 0;
    for (int idx = 0; idx < chunk_bf16; idx++) {
        sum += static_cast<int32_t>(raw[idx]);
    }

    output[0] = row;
    output[1] = static_cast<int32_t>(raw[0]);
    output[2] = static_cast<int32_t>(raw[chunk_bf16 - 1]);
    output[3] = sum;
}

} // extern "C"
