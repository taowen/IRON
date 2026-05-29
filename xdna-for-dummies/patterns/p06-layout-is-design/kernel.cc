#include <stdint.h>

extern "C" {

void sum_array(int32_t *input, int32_t *output, int32_t len) {
    int32_t sum = 0;
    for (int32_t i = 0; i < len; i++) {
        sum += input[i];
    }
    output[0] = sum;
}

}
