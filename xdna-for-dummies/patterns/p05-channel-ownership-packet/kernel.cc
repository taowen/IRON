#include <stdint.h>

extern "C" {

void add_constant(int32_t *input, int32_t *output, int32_t constant, int32_t len) {
    for (int32_t i = 0; i < len; i++) {
        output[i] = input[i] + constant;
    }
}

}
