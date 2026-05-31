#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lm_config.hpp"
#include "models/qwen3/qwen3_npu.hpp"
#include "nlohmann/json.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

namespace {

uint32_t get_u32(const nlohmann::json& json, const char* key, uint32_t fallback = 0) {
    if (!json.contains(key) || json[key].is_null()) {
        return fallback;
    }
    return json[key].get<uint32_t>();
}

float get_f32(const nlohmann::json& json, const char* key, float fallback = 0.0f) {
    if (!json.contains(key) || json[key].is_null()) {
        return fallback;
    }
    return json[key].get<float>();
}

std::string get_string(const nlohmann::json& json, const char* key, std::string fallback = "") {
    if (!json.contains(key) || json[key].is_null()) {
        return fallback;
    }
    return json[key].get<std::string>();
}

LM_Config load_config(const std::string& model_path, const std::string& repo_root) {
    std::ifstream input(model_path + "/config.json");
    if (!input) {
        throw std::runtime_error("failed to open config.json under " + model_path);
    }

    nlohmann::json json = nlohmann::json::parse(input);
    LM_Config config;
    config.model_path = model_path;
    config.model_name = std::filesystem::path(model_path).filename().string();
    config.model_type = get_string(json, "model_type");
    config.head_dim = get_u32(json, "head_dim");
    config.hidden_size = get_u32(json, "hidden_size");
    config.hidden_act = get_string(json, "hidden_act");
    config.intermediate_size = get_u32(json, "intermediate_size");
    config.num_attention_heads = get_u32(json, "num_attention_heads");
    config.num_hidden_layers = get_u32(json, "num_hidden_layers");
    config.num_key_value_heads = get_u32(json, "num_key_value_heads");
    config.pretraining_tp = get_u32(json, "pretraining_tp");
    config.rms_norm_eps = get_f32(json, "rms_norm_eps");
    config.rope_theta = get_f32(json, "rope_theta");
    config.vocab_size = get_u32(json, "vocab_size");
    config.sliding_window = get_u32(json, "sliding_window");
    config.sliding_window_pattern = get_u32(json, "sliding_window_pattern");
    config.addr_qk = get_u32(json, "addr_qk");
    config.addr_kv = get_u32(json, "addr_kv");
    config.addr_l_begin_mha = get_u32(json, "addr_l_begin_mha");
    config.addr_l_end_mha = get_u32(json, "addr_l_end_mha");
    config.addr_kk = get_u32(json, "addr_kk");
    config.flm_version = get_string(json, "flm_version");
    config.exec_path = repo_root + "/src";
    config.is_vlm = false;
    config.is_audio = false;
    config._json_config = std::move(json);
    return config;
}

struct Args {
    std::string model_path = "/var/home/taowen/flm/models/Qwen3-8B-NPU2";
    std::string repo_root = "/var/home/taowen/projects/MyLM";
    int max_l = 256;
    int token_id = 9707;
    int top_k = 5;
    int layers = 36;
    int cache_layer = 0;
    std::string dump_prefix;
    bool dump_all_caches = false;
    bool preemption = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) {
                throw std::runtime_error("missing value for " + arg);
            }
            return argv[i];
        };
        if (arg == "--model") {
            args.model_path = next();
        } else if (arg == "--repo-root") {
            args.repo_root = next();
        } else if (arg == "--max-l") {
            args.max_l = std::stoi(next());
        } else if (arg == "--token-id") {
            args.token_id = std::stoi(next());
        } else if (arg == "--top-k") {
            args.top_k = std::stoi(next());
        } else if (arg == "--layers") {
            args.layers = std::stoi(next());
        } else if (arg == "--cache-layer") {
            args.cache_layer = std::stoi(next());
        } else if (arg == "--dump-prefix") {
            args.dump_prefix = next();
        } else if (arg == "--dump-all-caches") {
            args.dump_all_caches = true;
        } else if (arg == "--preemption") {
            args.preemption = true;
        } else {
            throw std::runtime_error("unknown arg: " + arg);
        }
    }
    return args;
}

std::vector<std::pair<float, int>> top_tokens(buffer<bf16>& logits, int top_k) {
    if (top_k <= 0 || static_cast<size_t>(top_k) > logits.size()) {
        throw std::runtime_error("bad top-k");
    }
    std::vector<std::pair<float, int>> values;
    values.reserve(logits.size());
    for (size_t idx = 0; idx < logits.size(); ++idx) {
        values.emplace_back(static_cast<float>(logits[idx]), static_cast<int>(idx));
    }
    std::partial_sort(
        values.begin(),
        values.begin() + top_k,
        values.end(),
        [](const std::pair<float, int>& left, const std::pair<float, int>& right) {
            return left.first > right.first;
        }
    );
    values.resize(top_k);
    return values;
}

void print_bf16_head(const char* name, buffer<bf16> values, int count) {
    std::cout << name << "_size=" << values.size() << "\n";
    std::cout << name << "_head=";
    for (int idx = 0; idx < count && static_cast<size_t>(idx) < values.size(); ++idx) {
        if (idx != 0) {
            std::cout << ",";
        }
        std::cout << static_cast<float>(values[idx]);
    }
    std::cout << "\n";
}

void write_bf16_dump(const std::string& path, buffer<bf16> values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open dump path: " + path);
    }
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(bf16))
    );
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        args.repo_root = std::filesystem::absolute(args.repo_root).string();
        args.model_path = std::filesystem::absolute(args.model_path).string();

        LM_Config config = load_config(args.model_path, args.repo_root);
        if (args.layers <= 0 || args.layers > static_cast<int>(config.num_hidden_layers)) {
            throw std::runtime_error("bad layer count");
        }
        if (args.cache_layer < 0 || args.cache_layer >= args.layers) {
            throw std::runtime_error("cache layer must be inside executed layer range");
        }
        config.num_hidden_layers = static_cast<uint32_t>(args.layers);

        std::cerr << "model=" << config.model_name
                  << " layers=" << config.num_hidden_layers
                  << " hidden=" << config.hidden_size
                  << " token=" << args.token_id << "\n";

        xrt::device device(0);
        npu_xclbin_manager npu(device_npu2, &device, args.preemption);
        qwen3_npu engine(config, &npu, args.max_l);

        Q4NX q4nx(args.model_path);
        engine.load_weights(q4nx);
        engine.clear_context();
        std::cerr << "context_before=" << engine.get_current_context_length() << "\n";

        auto start = std::chrono::steady_clock::now();
        buffer<bf16> logits = engine.forward(args.token_id);
        auto end = std::chrono::steady_clock::now();
        std::cerr << "context_after=" << engine.get_current_context_length() << "\n";
        double seconds = std::chrono::duration<double>(end - start).count();

        std::cout << "logits_size=" << logits.size() << "\n";
        std::cout << "elapsed_seconds=" << seconds << "\n";
        for (const std::pair<float, int>& item : top_tokens(logits, args.top_k)) {
            std::cout << "top token=" << item.second << " logit=" << item.first << "\n";
        }

        buffer<bf16> current_k = engine.get_k_cache(args.cache_layer, 0);
        buffer<bf16> current_v = engine.get_v_cache(args.cache_layer, 0);
        print_bf16_head("cache_k0", current_k, 8);
        print_bf16_head("cache_v0", current_v, 8);

        if (!args.dump_prefix.empty()) {
            write_bf16_dump(args.dump_prefix + ".logits.bf16", logits);
            write_bf16_dump(args.dump_prefix + ".cache_k0.bf16", current_k);
            write_bf16_dump(args.dump_prefix + ".cache_v0.bf16", current_v);
            if (args.dump_all_caches) {
                for (int layer = 0; layer < args.layers; ++layer) {
                    buffer<bf16> layer_k = engine.get_k_cache(layer, 0);
                    buffer<bf16> layer_v = engine.get_v_cache(layer, 0);
                    std::string layer_prefix =
                        args.dump_prefix + ".layer" + (layer < 10 ? "0" : "") + std::to_string(layer);
                    write_bf16_dump(layer_prefix + ".cache_k0.bf16", layer_k);
                    write_bf16_dump(layer_prefix + ".cache_v0.bf16", layer_v);
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
