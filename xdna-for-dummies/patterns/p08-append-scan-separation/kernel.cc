#include <stdint.h>

// Two separate roles demonstrate the pattern:
// Writer: produces new data to append (runs in Phase 1)
// Scanner: reads entire cache and computes sum (runs in Phase 2, after sync)

extern "C" {

void writer_produce(int32_t *output, int32_t position, int32_t len) {
    // Produce deterministic data based on position
    for (int32_t i = 0; i < len; i++) {
        output[i] = 9000 + position + i;
    }
}

void scanner_sum(int32_t *cache, int32_t *result, int32_t len) {
    int32_t sum = 0;
    for (int32_t i = 0; i < len; i++) {
        sum += cache[i];
    }
    result[0] = sum;
}

}
