#include <stdint.h>

// Online weighted-mean merge: the real pattern from attention.
// Each block produces a carrier = (block_max, block_sum, weighted_value).
// Merge maintains running (max, sum, value) and rescales on each new block.
// This is the minimal sufficient statistic for online softmax merge.

extern "C" {

// Carrier: 3 dwords = (block_max, block_sum, block_weighted_value)
// block_max = max score in this block
// block_sum = sum of exp(score - block_max) weights (Q12 fixed point)
// block_weighted_value = sum(weight[i] * value[i]) (Q12 fixed point)

// Takes a single buffer [scores|values] and splits internally
void compute_block_carrier(int32_t *block_pair, int32_t *carrier, int32_t len) {
    int32_t *scores = block_pair;
    int32_t *values = block_pair + len;
    // Find block max
    int32_t block_max = scores[0];
    for (int32_t i = 1; i < len; i++) {
        if (scores[i] > block_max) {
            block_max = scores[i];
        }
    }

    // Compute weights = exp(score - block_max) approximated as max(0, 4096 - (block_max - score) * 64)
    // and weighted sum + block_sum
    int32_t block_sum = 0;
    int32_t weighted_value = 0;
    for (int32_t i = 0; i < len; i++) {
        int32_t delta = block_max - scores[i];
        int32_t weight = 4096 - delta * 64;
        if (weight < 0) weight = 0;
        block_sum += weight;
        weighted_value += weight * values[i];
    }

    carrier[0] = block_max;
    carrier[1] = block_sum;
    carrier[2] = weighted_value;
}

// Running state: 3 dwords = (running_max, running_sum, running_weighted_value)
void init_merge_state(int32_t *state) {
    state[0] = -2147483647;  // running_max = -INF
    state[1] = 0;            // running_sum = 0
    state[2] = 0;            // running_weighted_value = 0
}

// Online merge: rescale running state when new block has different max
void online_merge(int32_t *carrier, int32_t *state) {
    int32_t block_max = carrier[0];
    int32_t block_sum = carrier[1];
    int32_t block_wv = carrier[2];

    int32_t running_max = state[0];
    int32_t running_sum = state[1];
    int32_t running_wv = state[2];

    if (running_sum == 0) {
        // First block: just copy
        state[0] = block_max;
        state[1] = block_sum;
        state[2] = block_wv;
        return;
    }

    // Determine new max and scale factors
    int32_t new_max;
    int32_t old_scale;   // scale for running state (Q12)
    int32_t new_scale;   // scale for new block (Q12)

    if (block_max >= running_max) {
        new_max = block_max;
        int32_t delta = block_max - running_max;
        old_scale = 4096 - delta * 64;
        if (old_scale < 0) old_scale = 0;
        new_scale = 4096;
    } else {
        new_max = running_max;
        int32_t delta = running_max - block_max;
        old_scale = 4096;
        new_scale = 4096 - delta * 64;
        if (new_scale < 0) new_scale = 0;
    }

    // Rescale and merge
    state[0] = new_max;
    state[1] = (running_sum * old_scale + block_sum * new_scale) / 4096;
    state[2] = (running_wv * old_scale + block_wv * new_scale) / 4096;
}

// Final output: weighted_value / sum (the weighted mean)
void finalize_state(int32_t *state, int32_t *output) {
    if (state[1] == 0) {
        output[0] = 0;
    } else {
        output[0] = state[2] / state[1];
    }
    output[1] = state[0];  // final max (for debugging)
    output[2] = state[1];  // final sum (for debugging)
}

}
