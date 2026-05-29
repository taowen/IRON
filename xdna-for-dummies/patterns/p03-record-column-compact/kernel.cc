#include <stdint.h>

extern "C" {

// Each producer emits a record: [header | payload...]
// header encodes (tile_id << 16) | payload_dwords
// payload = tile_id * 100 + lane_index
void emit_record(int32_t *record, int32_t tile_id, int32_t payload_len) {
    record[0] = (tile_id << 16) | payload_len;
    for (int32_t i = 0; i < payload_len; i++) {
        record[1 + i] = tile_id * 100 + i;
    }
}

}
