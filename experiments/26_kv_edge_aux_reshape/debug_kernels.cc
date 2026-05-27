#include <stdint.h>

extern "C" {

void copy_current_512(int32_t *src, int32_t *dst) {
    for (int i = 0; i < 512; i++) {
        dst[i] = src[i];
    }
}

void copy_sideband_17(int32_t *src, int32_t *dst) {
    for (int i = 0; i < 17; i++) {
        dst[i] = src[i];
    }
}

void shape_a_debug(int32_t *current, int32_t *history, int32_t *out) {
    int32_t current_sum = 0;
    int32_t current_xor = 0;
    for (int i = 0; i < 512; i++) {
        current_sum += current[i];
        current_xor ^= current[i];
    }

    int32_t history_sum = 0;
    int32_t history_xor = 0;
    for (int i = 0; i < 2048; i++) {
        history_sum += history[i];
        history_xor ^= history[i];
    }

    out[0] = current_sum;
    out[1] = current_xor;
    out[2] = current[0];
    out[3] = current[511];
    out[4] = history_sum;
    out[5] = history_xor;
    out[6] = history[0];
    out[7] = history[2047];
}

void sideband_debug(int32_t *sideband, int32_t *out) {
    int32_t sum = 0;
    int32_t xors = 0;
    for (int i = 0; i < 17; i++) {
        sum += sideband[i];
        xors ^= sideband[i];
    }

    out[0] = sum;
    out[1] = xors;
    out[2] = sideband[0];
    out[3] = sideband[16];
}

void sideband_debug_and_forward(int32_t *sideband, int32_t *out, int32_t *forward) {
    int32_t sum = 0;
    int32_t xors = 0;
    for (int i = 0; i < 17; i++) {
        int32_t value = sideband[i];
        sum += value;
        xors ^= value;
        forward[i] = value + 17;
    }

    out[0] = sum;
    out[1] = xors;
    out[2] = sideband[0];
    out[3] = sideband[16];
}

}
