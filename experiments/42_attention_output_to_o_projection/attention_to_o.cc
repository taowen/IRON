#include <stdint.h>

extern "C" {

static inline int32_t read_attention(int32_t *attention_lo, int32_t *attention_hi, int index) {
    return index < 256 ? attention_lo[index] : attention_hi[index - 256];
}

void edge_make_attention(int32_t *current, int32_t *history, int32_t *attention_lo, int32_t *attention_hi) {
    for (int i = 0; i < 512; i++) {
        int32_t value = 3 * current[i] - 2 * current[(i + 13) & 511] +
                        5 * history[(i * 7) & 2047] +
                        history[(i * 11 + 3) & 2047] + (i % 17);
        if (i < 256) {
            attention_lo[i] = value;
        } else {
            attention_hi[i - 256] = value;
        }
    }
}

void o_project_attention(int32_t *attention_lo, int32_t *attention_hi, int32_t *weight, int32_t *output) {
    for (int i = 0; i < 512; i++) {
        int base = i * 4;
        output[i] = read_attention(attention_lo, attention_hi, i) * weight[base] +
                    read_attention(attention_lo, attention_hi, (i + 17) & 511) * weight[base + 1] -
                    read_attention(attention_lo, attention_hi, (i + 29) & 511) * weight[base + 2] +
                    read_attention(attention_lo, attention_hi, (i + 43) & 511) * weight[base + 3] +
                    (i & 31);
    }
}

}
