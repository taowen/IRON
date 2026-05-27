#include <stdint.h>

namespace {

constexpr int CONTEXT_LEN = 31;
constexpr int HEAD_DIM = 128;
constexpr int HIDDEN_DWORDS = 512;
constexpr int QUERY_DWORDS = 512;
constexpr int OUTPUT_DWORDS = 512;

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

} // namespace

extern "C" {

void project_query_current(int32_t *hidden, int32_t *query_lo, int32_t *query_hi,
                           int32_t *current_k, int32_t *current_v) {
    for (int i = 0; i < QUERY_DWORDS; i++) {
        int head = i / HEAD_DIM;
        int dim = i & (HEAD_DIM - 1);
        int32_t value = 2 * hidden[(dim * 3 + head * 17) & (HIDDEN_DWORDS - 1)] -
                        hidden[(dim * 5 + head * 29 + 7) & (HIDDEN_DWORDS - 1)] +
                        ((dim + head) & 7);
        write_half(query_lo, query_hi, i, value);
    }

    for (int dim = 0; dim < HEAD_DIM; dim++) {
        current_k[dim] = hidden[(dim * 7 + 11) & (HIDDEN_DWORDS - 1)] -
                         2 * hidden[(dim * 13 + 3) & (HIDDEN_DWORDS - 1)] + (dim & 3);
        current_v[dim] = 3 * hidden[(dim * 5 + 19) & (HIDDEN_DWORDS - 1)] +
                         hidden[(dim * 9 + 23) & (HIDDEN_DWORDS - 1)] - (dim & 5);
    }
}

void attention_from_cache(int32_t *query_lo, int32_t *query_hi,
                          int32_t *k_history, int32_t *v_history,
                          int32_t *attention_lo, int32_t *attention_hi) {
    for (int i = 0; i < QUERY_DWORDS; i++) {
        int head = i / HEAD_DIM;
        int dim = i & (HEAD_DIM - 1);
        int32_t q = read_half(query_lo, query_hi, i);
        int32_t acc = 0;
        for (int token = 0; token < CONTEXT_LEN; token++) {
            int idx = token * HEAD_DIM + dim;
            acc += (q + head + 1) * k_history[idx];
            acc += (head + 2) * v_history[idx];
            acc += (token + dim) & 7;
        }
        write_half(attention_lo, attention_hi, i, acc / 32);
    }
}

void o_project_attention(int32_t *attention_lo, int32_t *attention_hi,
                         int32_t *weight, int32_t *o_lo, int32_t *o_hi) {
    for (int i = 0; i < OUTPUT_DWORDS; i++) {
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
    for (int i = 0; i < OUTPUT_DWORDS; i++) {
        int32_t o_value = read_half(o_lo, o_hi, i);
        int32_t norm = o_value - ((i % 17) - 8);
        gate[i] = norm * weight[i] + weight[512 + i] + (i & 7);
        up[i] = norm * weight[1024 + i] - weight[1536 + i] + (i & 15);
        swiglu[i] = (gate[i] * up[i]) / 64;
    }

    for (int i = 0; i < OUTPUT_DWORDS; i++) {
        int32_t o_value = read_half(o_lo, o_hi, i);
        output[i] =
            o_value + swiglu[i] * weight[2048 + i] +
            swiglu[(i + 19) & 511] * weight[2560 + i] -
            swiglu[(i + 37) & 511] * weight[3072 + i] +
            weight[3584 + i] + (i & 31);
    }
}

}
