#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void identity_bf16(bfloat16 *input, bfloat16 *output, int32_t size) {
    constexpr int N = 16;
    int vector_chunks = size / N;
    for (int i = 0; i < vector_chunks; i++) {
        ::aie::vector<bfloat16, N> v = ::aie::load_v<N>(input + i * N);
        ::aie::store_v(output + i * N, v);
    }
    int remaining = size % N;
    if (remaining > 0) {
        int start = vector_chunks * N;
        for (int i = 0; i < remaining; i++) {
            output[start + i] = input[start + i];
        }
    }
}

}
