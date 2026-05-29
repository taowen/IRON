#include <stdint.h>

extern "C" {

void transform(int32_t *input, int32_t *output, int32_t len) {
    for (int32_t i = 0; i < len; i++) {
        output[i] = input[i] * 3 + 7;
    }
}

}
