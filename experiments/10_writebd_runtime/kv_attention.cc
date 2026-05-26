#include <stdint.h>

extern "C" {

void zero_output(int32_t *out_buf) {
    for (int i = 0; i < 128; i++) {
        out_buf[i] = 0;
    }
}

void kv_attention(
    int32_t *k_buf,
    int32_t *v_buf,
    int32_t *out_buf,
    int32_t tile_idx,
    int32_t num_tiles,
    int32_t last_valid,
    int32_t head_offset
) {
    const int NUM_HEADS = 4;
    const int HEAD_DIM = 32;
    const int STRIDE = NUM_HEADS * HEAD_DIM;

    int valid = (tile_idx < num_tiles - 1) ? 16 : last_valid;

    for (int t = 0; t < valid; t++) {
        for (int h = 0; h < NUM_HEADS; h++) {
            int32_t score = 0;
            int32_t q_val = head_offset + h + 1;
            for (int d = 0; d < HEAD_DIM; d++) {
                score += q_val * k_buf[t * STRIDE + h * HEAD_DIM + d];
            }
            for (int d = 0; d < HEAD_DIM; d++) {
                out_buf[h * HEAD_DIM + d] += score * v_buf[t * STRIDE + h * HEAD_DIM + d];
            }
        }
    }
}

}
