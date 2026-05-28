#include <stdint.h>

extern "C" {

void bridge_init_summary(int32_t *summary, int32_t group, int32_t row) {
    summary[0] = group;
    summary[1] = row;
    summary[2] = 0;
    summary[3] = 0;
    summary[4] = 0;
    summary[5] = 0;
    summary[6] = 0;
    summary[7] = 0;
}

void bridge_accum_chunk(int32_t *chunk, int32_t *summary, int32_t dwords) {
    if (summary[2] == 0) {
        summary[3] = chunk[0];
    }
    summary[4] = chunk[dwords - 1];
    for (int idx = 0; idx < dwords; idx++) {
        summary[5] += chunk[idx];
        summary[6] ^= chunk[idx];
    }
    summary[2] += 1;
    summary[7] += dwords;
}

}
