#include <stdint.h>

extern "C" {

static inline int32_t read_half(int32_t *lo, int32_t *hi, int index) {
    return index < 256 ? lo[index] : hi[index - 256];
}

static inline void write_half(int32_t *lo, int32_t *hi, int index, int32_t value) {
    if (index < 256) {
        lo[index] = value;
    } else {
        hi[index - 256] = value;
    }
}

void edge_make_attention(int32_t *current, int32_t *history, int32_t *attention_lo, int32_t *attention_hi) {
    for (int i = 0; i < 512; i++) {
        int32_t value = 3 * current[i] - 2 * current[(i + 13) & 511] +
                        5 * history[(i * 7) & 2047] +
                        history[(i * 11 + 3) & 2047] + (i % 17);
        write_half(attention_lo, attention_hi, i, value);
    }
}

void o_project_attention(int32_t *attention_lo, int32_t *attention_hi,
                         int32_t *weight, int32_t *o_lo, int32_t *o_hi) {
    for (int i = 0; i < 512; i++) {
        int base = i * 4;
        int32_t value =
            read_half(attention_lo, attention_hi, i) * weight[base] +
            read_half(attention_lo, attention_hi, (i + 17) & 511) * weight[base + 1] -
            read_half(attention_lo, attention_hi, (i + 29) & 511) * weight[base + 2] +
            read_half(attention_lo, attention_hi, (i + 43) & 511) * weight[base + 3] +
            (i & 31);
        write_half(o_lo, o_hi, i, value);
    }
}

void ffn_tail(int32_t *o_lo, int32_t *o_hi, int32_t *weight,
              int32_t *gate, int32_t *up, int32_t *swiglu, int32_t *output) {
    for (int i = 0; i < 512; i++) {
        int32_t o_value = read_half(o_lo, o_hi, i);
        int32_t norm = o_value - ((i % 17) - 8);
        gate[i] = norm * weight[i] + weight[512 + i] + (i & 7);
        up[i] = norm * weight[1024 + i] - weight[1536 + i] + (i & 15);
        swiglu[i] = (gate[i] * up[i]) / 64;
    }

    for (int i = 0; i < 512; i++) {
        int32_t o_value = read_half(o_lo, o_hi, i);
        output[i] =
            o_value + swiglu[i] * weight[2048 + i] +
            swiglu[(i + 19) & 511] * weight[2560 + i] -
            swiglu[(i + 37) & 511] * weight[3072 + i] +
            weight[3584 + i] + (i & 31);
    }
}

}
