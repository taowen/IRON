#include <stdint.h>

extern "C" {

void shape_a_make_state(int32_t *current, int32_t *history, int32_t *state) {
    int64_t cur_sum = 0;
    int64_t hist_sum = 0;
    for (int i = 0; i < 512; i += 32) {
        cur_sum += current[i];
    }
    for (int i = 0; i < 2048; i += 128) {
        hist_sum += history[i];
    }

    state[0] = 0x41000000 | ((static_cast<int32_t>(cur_sum) & 0xff) << 8) |
               (static_cast<int32_t>(hist_sum) & 0xff);
    int32_t cur_bias = static_cast<int32_t>(cur_sum) & 0xffff;
    int32_t hist_bias = static_cast<int32_t>(hist_sum) & 0x7fff;
    for (int lane = 0; lane < 16; lane++) {
        state[1 + lane] = current[(lane * 31) & 511] +
                          3 * history[(lane * 97) & 2047] + lane * 17 +
                          cur_bias - hist_bias;
    }
}

void shape_b_consume_state(int32_t *state, int32_t *history, int32_t *output) {
    int32_t header_low = state[0] & 0xffff;
    for (int i = 0; i < 512; i++) {
        int lane = i & 15;
        int block = i >> 4;
        output[i] = state[1 + lane] +
                    history[(block * 37 + lane * 13) & 2047] + header_low +
                    block;
    }
}

}
