#include <stdint.h>

extern "C" {

void producer_fill(int32_t *buf, int32_t batch, int32_t len) {
    for (int32_t i = 0; i < len; i++) {
        buf[i] = batch * len + i;
    }
}

void consumer_add(int32_t *input, int32_t *output, int32_t constant, int32_t len) {
    for (int32_t i = 0; i < len; i++) {
        output[i] = input[i] + constant;
    }
}

}
