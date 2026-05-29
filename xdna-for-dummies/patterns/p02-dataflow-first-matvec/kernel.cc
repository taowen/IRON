#include <stdint.h>

extern "C" {

// Tile-local accumulator: persists across weight chunks
static int32_t accum[4];

void matvec_clear(int32_t rows) {
    for (int32_t i = 0; i < rows; i++) {
        accum[i] = 0;
    }
}

// MAC one weight chunk against one activation chunk.
// Weight stream order aligns with compute loop:
// weights arrive chunk-by-chunk, each used exactly once then discarded.
void matvec_mac_chunk(int32_t *weights, int32_t *activation,
                      int32_t rows, int32_t cols) {
    for (int32_t row = 0; row < rows; row++) {
        int32_t sum = 0;
        for (int32_t col = 0; col < cols; col++) {
            sum += weights[row * cols + col] * activation[col];
        }
        accum[row] += sum;
    }
}

void matvec_emit_record(int32_t *record, int32_t tile_id, int32_t rows) {
    record[0] = (tile_id << 16) | 0x4D00 | rows;
    for (int32_t i = 0; i < rows; i++) {
        record[1 + i] = accum[i];
    }
}

}
