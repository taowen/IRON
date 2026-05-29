#include <stdint.h>

// Demonstrates stable ABI handoff between two operators.
// Op A outputs a fixed-format record: [header | payload]
// Op B receives the same format, processes payload, outputs its own record.
// The ABI = header encoding + payload layout + fixed size.

extern "C" {

// Record format: 1 dword header + 16 dwords payload = 17 dwords total
// Header encodes: operator_id (high 8 bits) | payload_count (low 8 bits)

void op_a_emit_record(int32_t *input, int32_t *record, int32_t payload_len) {
    // Header: op_id=0xA0, payload_count=payload_len
    record[0] = (0xA0 << 8) | payload_len;
    // Payload: input + 10
    for (int32_t i = 0; i < payload_len; i++) {
        record[1 + i] = input[i] + 10;
    }
}

void op_b_consume_record(int32_t *record_in, int32_t *record_out, int32_t payload_len) {
    // Verify ABI: check header from op A
    int32_t header_in = record_in[0];
    int32_t op_id = (header_in >> 8) & 0xFF;
    // Write output record with own header
    record_out[0] = (0xB0 << 8) | payload_len;
    // Payload: input_payload * 2 (regardless of which op produced it)
    for (int32_t i = 0; i < payload_len; i++) {
        record_out[1 + i] = record_in[1 + i] * 2;
    }
    // Embed source op_id in last slot for traceability
    record_out[payload_len] = op_id;
}

}
