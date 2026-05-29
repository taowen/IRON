#include <stdint.h>

extern "C" {

// Each tile is responsible for ROWS_PER_TILE output rows.
// It receives activation chunks (CHUNK_COLS wide) and weight chunks
// (ROWS_PER_TILE * CHUNK_COLS), accumulates MAC results locally,
// then emits a fixed-format record (1 header + ROWS_PER_TILE payload).

static int32_t accum[4];

void tile_clear_accum(int32_t rows) {
    for (int32_t i = 0; i < rows; i++) {
        accum[i] = 0;
    }
}

void tile_mac_chunk(int32_t *weights, int32_t *activation,
                    int32_t rows, int32_t cols) {
    // weights layout: [row][col] within chunk
    for (int32_t row = 0; row < rows; row++) {
        int32_t sum = 0;
        for (int32_t col = 0; col < cols; col++) {
            sum += weights[row * cols + col] * activation[col];
        }
        accum[row] += sum;
    }
}

void tile_emit_record(int32_t *record, int32_t tile_id, int32_t rows) {
    // Record ABI: [header | payload...]
    // header encodes which tile produced this record
    record[0] = (tile_id << 16) | 0xAB00 | rows;
    for (int32_t i = 0; i < rows; i++) {
        record[1 + i] = accum[i];
    }
}

}
