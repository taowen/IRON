#include <stdint.h>

// Demonstrates two layers of dynamic parameters:
// 1. RTP: tile reads a scalar offset from RTP buffer (gated by runtime-start lock)
// 2. Descriptor: host configures BD buffer_offset to select which slice of input to send

extern "C" {

void add_rtp_offset(int32_t *input, int32_t *output, int32_t *rtp, int32_t len) {
    int32_t offset = rtp[0];
    for (int32_t i = 0; i < len; i++) {
        output[i] = input[i] + offset;
    }
}

}
