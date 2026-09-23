#include "core/arena.h"
#include "core/device.h"
#include "ninfer/targets/qwen3_vision/vision.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::targets::qwen3_vision;

Weight make_dense_weight(const void* data, std::int32_t n, std::int32_t k) {
    Weight w{};
    w.payload       = data;
    w.payload_bytes = static_cast<std::size_t>(n) * k * sizeof(std::uint16_t);
    w.qdata         = data;
    w.qtype         = QType::BF16_CTRL;
    w.layout        = QuantLayout::Contiguous;
    w.shape[0]      = n;
    w.shape[1]      = k;
    w.n             = n;
    w.k             = k;
    w.ndim          = 2;
    return w;
}

int test_config_constants() {
    static_assert(VisionTowerConfig::layers == 27);
    static_assert(VisionTowerConfig::hidden == 1152);
    static_assert(VisionTowerConfig::intermediate == 4304);
    static_assert(VisionTowerConfig::heads == 16);
    static_assert(VisionTowerConfig::head_dim == 72);
    static_assert(VisionTowerConfig::patch_dim == 1536);
    static_assert(VisionTowerConfig::merge == 2);
    static_assert(VisionTowerConfig::merge_unit == 4);
    static_assert(VisionTowerConfig::merger_hidden == 4608);
    static_assert(VisionTowerConfig::position_embeddings == 2304);
    static_assert(VisionTowerConfig::rotary_dim == 72);
    static_assert(VisionTowerConfig::rope_theta == 10000.0f);
    static_assert(VisionTowerConfig::norm_epsilon == 1.0e-6f);

    const float expected_scale = 1.0f / std::sqrt(72.0f);
    if (std::abs(VisionTowerConfig::attention_scale - expected_scale) > 1e-6f) {
        std::cerr << "VisionTowerConfig::attention_scale mismatch: "
                  << VisionTowerConfig::attention_scale << " vs " << expected_scale << "\n";
        return 1;
    }

    std::cout << "PASS: test_config_constants\n";
    return 0;
}

int test_workspace_planning() {
    // 1. workspace_bytes calculation
    const std::size_t bytes_1tok = Encoder::workspace_bytes(4, 1);
    const std::size_t bytes_4tok = Encoder::workspace_bytes(16, 4);
    if (bytes_1tok == 0 || bytes_4tok == 0 || bytes_4tok <= bytes_1tok) {
        std::cerr << "Invalid workspace_bytes ordering: " << bytes_1tok << " vs " << bytes_4tok << "\n";
        return 1;
    }

    // 2. Reject non 4:1 patch-to-token relation
    try {
        (void)Encoder::workspace_bytes(5, 1);
        std::cerr << "Failed to reject invalid patch/token relation (5 patches, 1 token)\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    // 3. Plan workspace across different output_hidden targets
    for (std::int32_t out_h : {2048, 2560, 3584, 4096}) {
        const auto plan = Encoder::plan_workspace(1024, 64 * 1024 * 1024, out_h);
        if (plan.max_merged_tokens != 1024 || plan.general_capacity_bytes != 64 * 1024 * 1024) {
            std::cerr << "Plan field mismatch for out_h=" << out_h << "\n";
            return 1;
        }
        if (plan.encode_peak_bytes == 0 || plan.handoff_capacity_bytes == 0 ||
            plan.capacity_bytes < plan.encode_peak_bytes ||
            plan.capacity_bytes < plan.handoff_offset_bytes + plan.handoff_capacity_bytes) {
            std::cerr << "Plan capacity envelope mismatch for out_h=" << out_h << "\n";
            return 1;
        }
    }

    // 4. Reject invalid parameters
    try {
        (void)Encoder::plan_workspace(0, 1024, 2560);
        std::cerr << "Failed to reject max_merged_tokens=0\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    try {
        (void)Encoder::plan_workspace(1024, 0, 2560);
        std::cerr << "Failed to reject general_capacity_bytes=0\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    try {
        (void)Encoder::plan_workspace(1024, 1024, 0);
        std::cerr << "Failed to reject output_hidden=0\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::cout << "PASS: test_workspace_planning\n";
    return 0;
}

int test_output_binding() {
    const std::int32_t out_h = 2560;
    const auto plan = Encoder::plan_workspace(64, 16 * 1024 * 1024, out_h);

    DeviceBuffer buffer(plan.capacity_bytes);
    DeviceSpan backing{buffer.p, buffer.bytes};

    // Valid binding
    const Tensor out = Encoder::bind_output(backing, plan, 16, out_h);
    if (out.dtype != DType::BF16 || out.ne[0] != out_h || out.ne[1] != 16 ||
        out.ne[2] != 1 || out.ne[3] != 1 || out.data == nullptr) {
        std::cerr << "bind_output returned invalid tensor descriptor\n";
        return 1;
    }

    // Rejection: exceed max_merged_tokens
    try {
        (void)Encoder::bind_output(backing, plan, 65, out_h);
        std::cerr << "Failed to reject merged_tokens > max_merged_tokens\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    // Rejection: insufficient backing
    DeviceSpan tiny_backing{buffer.p, plan.capacity_bytes - 1};
    try {
        (void)Encoder::bind_output(tiny_backing, plan, 16, out_h);
        std::cerr << "Failed to reject insufficient backing\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::cout << "PASS: test_output_binding\n";
    return 0;
}

int test_weights_validation() {
    DeviceContext device(0);

    // Empty/zero-initialized weights must throw invalid_argument
    Weights empty_weights{};
    try {
        Encoder encoder(device, empty_weights);
        std::cerr << "Failed to reject uninitialized weights in Encoder constructor\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::cout << "PASS: test_weights_validation\n";
    return 0;
}

int test_encoder_execution_synthetic() {
    DeviceContext device(0);

    constexpr std::int32_t out_h = 2560;

    // Allocate synthetic mock weights on GPU
    DeviceBuffer patch_w_buf(VisionTowerConfig::hidden * VisionTowerConfig::patch_dim * 2);
    DeviceBuffer patch_b_buf(VisionTowerConfig::hidden * 2);
    DeviceBuffer pos_embed_buf(VisionTowerConfig::position_embeddings * VisionTowerConfig::hidden * 2);

    CUDA_CHECK(cudaMemset(patch_w_buf.p, 0, patch_w_buf.bytes));
    CUDA_CHECK(cudaMemset(patch_b_buf.p, 0, patch_b_buf.bytes));
    CUDA_CHECK(cudaMemset(pos_embed_buf.p, 0, pos_embed_buf.bytes));

    Weight patch_w = make_dense_weight(patch_w_buf.p, VisionTowerConfig::hidden, VisionTowerConfig::patch_dim);
    Tensor patch_b(patch_b_buf.p, DType::BF16, {VisionTowerConfig::hidden});
    Tensor pos_embed(pos_embed_buf.p, DType::BF16, {VisionTowerConfig::position_embeddings, VisionTowerConfig::hidden});

    struct LayerStorage {
        DeviceBuffer norm1_w{VisionTowerConfig::hidden * 2};
        DeviceBuffer norm1_b{VisionTowerConfig::hidden * 2};
        DeviceBuffer qkv_w{3 * VisionTowerConfig::hidden * VisionTowerConfig::hidden * 2};
        DeviceBuffer qkv_b{3 * VisionTowerConfig::hidden * 2};
        DeviceBuffer proj_w{VisionTowerConfig::hidden * VisionTowerConfig::hidden * 2};
        DeviceBuffer proj_b{VisionTowerConfig::hidden * 2};
        DeviceBuffer norm2_w{VisionTowerConfig::hidden * 2};
        DeviceBuffer norm2_b{VisionTowerConfig::hidden * 2};
        DeviceBuffer fc1_w{VisionTowerConfig::intermediate * VisionTowerConfig::hidden * 2};
        DeviceBuffer fc1_b{VisionTowerConfig::intermediate * 2};
        DeviceBuffer fc2_w{VisionTowerConfig::hidden * VisionTowerConfig::intermediate * 2};
        DeviceBuffer fc2_b{VisionTowerConfig::hidden * 2};
    };

    std::vector<LayerStorage> layer_storage(VisionTowerConfig::layers);

    Weights weights{};
    weights.patch_embed      = &patch_w;
    weights.patch_embed_bias = &patch_b;
    weights.position_embed   = &pos_embed;
    weights.output_hidden    = out_h;

    for (std::size_t i = 0; i < VisionTowerConfig::layers; ++i) {
        auto& s = layer_storage[i];
        CUDA_CHECK(cudaMemset(s.norm1_w.p, 0, s.norm1_w.bytes));
        CUDA_CHECK(cudaMemset(s.norm1_b.p, 0, s.norm1_b.bytes));
        CUDA_CHECK(cudaMemset(s.qkv_w.p, 0, s.qkv_w.bytes));
        CUDA_CHECK(cudaMemset(s.qkv_b.p, 0, s.qkv_b.bytes));
        CUDA_CHECK(cudaMemset(s.proj_w.p, 0, s.proj_w.bytes));
        CUDA_CHECK(cudaMemset(s.proj_b.p, 0, s.proj_b.bytes));
        CUDA_CHECK(cudaMemset(s.norm2_w.p, 0, s.norm2_w.bytes));
        CUDA_CHECK(cudaMemset(s.norm2_b.p, 0, s.norm2_b.bytes));
        CUDA_CHECK(cudaMemset(s.fc1_w.p, 0, s.fc1_w.bytes));
        CUDA_CHECK(cudaMemset(s.fc1_b.p, 0, s.fc1_b.bytes));
        CUDA_CHECK(cudaMemset(s.fc2_w.p, 0, s.fc2_w.bytes));
        CUDA_CHECK(cudaMemset(s.fc2_b.p, 0, s.fc2_b.bytes));

        auto& lw = weights.layers[i];
        lw.norm1_weight    = new Tensor(s.norm1_w.p, DType::BF16, {VisionTowerConfig::hidden});
        lw.norm1_bias      = new Tensor(s.norm1_b.p, DType::BF16, {VisionTowerConfig::hidden});
        lw.qkv             = new Weight(make_dense_weight(s.qkv_w.p, 3 * VisionTowerConfig::hidden, VisionTowerConfig::hidden));
        lw.qkv_bias        = new Tensor(s.qkv_b.p, DType::BF16, {3 * VisionTowerConfig::hidden});
        lw.projection      = new Weight(make_dense_weight(s.proj_w.p, VisionTowerConfig::hidden, VisionTowerConfig::hidden));
        lw.projection_bias = new Tensor(s.proj_b.p, DType::BF16, {VisionTowerConfig::hidden});
        lw.norm2_weight    = new Tensor(s.norm2_w.p, DType::BF16, {VisionTowerConfig::hidden});
        lw.norm2_bias      = new Tensor(s.norm2_b.p, DType::BF16, {VisionTowerConfig::hidden});
        lw.fc1             = new Weight(make_dense_weight(s.fc1_w.p, VisionTowerConfig::intermediate, VisionTowerConfig::hidden));
        lw.fc1_bias        = new Tensor(s.fc1_b.p, DType::BF16, {VisionTowerConfig::intermediate});
        lw.fc2             = new Weight(make_dense_weight(s.fc2_w.p, VisionTowerConfig::hidden, VisionTowerConfig::intermediate));
        lw.fc2_bias        = new Tensor(s.fc2_b.p, DType::BF16, {VisionTowerConfig::hidden});
    }

    DeviceBuffer merger_norm_w(VisionTowerConfig::hidden * 2);
    DeviceBuffer merger_norm_b(VisionTowerConfig::hidden * 2);
    DeviceBuffer merger_fc1_w(VisionTowerConfig::merger_hidden * VisionTowerConfig::merger_hidden * 2);
    DeviceBuffer merger_fc1_b(VisionTowerConfig::merger_hidden * 2);
    DeviceBuffer merger_fc2_w(out_h * VisionTowerConfig::merger_hidden * 2);
    DeviceBuffer merger_fc2_b(out_h * 2);

    CUDA_CHECK(cudaMemset(merger_norm_w.p, 0, merger_norm_w.bytes));
    CUDA_CHECK(cudaMemset(merger_norm_b.p, 0, merger_norm_b.bytes));
    CUDA_CHECK(cudaMemset(merger_fc1_w.p, 0, merger_fc1_w.bytes));
    CUDA_CHECK(cudaMemset(merger_fc1_b.p, 0, merger_fc1_b.bytes));
    CUDA_CHECK(cudaMemset(merger_fc2_w.p, 0, merger_fc2_w.bytes));
    CUDA_CHECK(cudaMemset(merger_fc2_b.p, 0, merger_fc2_b.bytes));

    Tensor m_norm_w(merger_norm_w.p, DType::BF16, {VisionTowerConfig::hidden});
    Tensor m_norm_b(merger_norm_b.p, DType::BF16, {VisionTowerConfig::hidden});
    Weight m_fc1_w = make_dense_weight(merger_fc1_w.p, VisionTowerConfig::merger_hidden, VisionTowerConfig::merger_hidden);
    Tensor m_fc1_b(merger_fc1_b.p, DType::BF16, {VisionTowerConfig::merger_hidden});
    Weight m_fc2_w = make_dense_weight(merger_fc2_w.p, out_h, VisionTowerConfig::merger_hidden);
    Tensor m_fc2_b(merger_fc2_b.p, DType::BF16, {out_h});

    weights.merger.norm_weight = &m_norm_w;
    weights.merger.norm_bias   = &m_norm_b;
    weights.merger.fc1         = &m_fc1_w;
    weights.merger.fc1_bias    = &m_fc1_b;
    weights.merger.fc2         = &m_fc2_w;
    weights.merger.fc2_bias    = &m_fc2_b;

    Encoder encoder(device, weights);

    const auto plan = Encoder::plan_workspace(4, 16 * 1024 * 1024, out_h);
    DeviceBuffer workspace_buf(plan.capacity_bytes);
    DeviceSpan backing{workspace_buf.p, workspace_buf.bytes};

    Tensor output = Encoder::bind_output(backing, plan, 1, out_h);

    // Create 1-token (4-patch) input item
    std::vector<std::uint16_t> patches(4 * VisionTowerConfig::patch_dim, 0x3c00); // 1.0f in BF16
    std::vector<std::int32_t> pos_ids = {0, 0, 0, 1, 1, 0, 1, 1}; // 4 patches * 2 coords
    std::vector<std::int32_t> pos_table_idx(4 * 4, 0);
    std::vector<float> pos_table_weights(4 * 4, 0.25f);

    EncodeItemView item{
        .patches                = patches,
        .patch_count            = 4,
        .merged_count           = 1,
        .segment_length         = 4,
        .position_ids           = pos_ids,
        .position_table_indices = pos_table_idx,
        .position_table_weights = pos_table_weights,
    };

    encoder.encode(item, output, backing, plan);
    device.synchronize();

    std::vector<std::uint16_t> host_out(out_h);
    CUDA_CHECK(cudaMemcpy(host_out.data(), output.data, out_h * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    // Cleanup dynamic layer weight pointers
    for (std::size_t i = 0; i < VisionTowerConfig::layers; ++i) {
        delete weights.layers[i].norm1_weight;
        delete weights.layers[i].norm1_bias;
        delete weights.layers[i].qkv;
        delete weights.layers[i].qkv_bias;
        delete weights.layers[i].projection;
        delete weights.layers[i].projection_bias;
        delete weights.layers[i].norm2_weight;
        delete weights.layers[i].norm2_bias;
        delete weights.layers[i].fc1;
        delete weights.layers[i].fc1_bias;
        delete weights.layers[i].fc2;
        delete weights.layers[i].fc2_bias;
    }

    std::cout << "PASS: test_encoder_execution_synthetic (output size: " << host_out.size() << ")\n";
    return 0;
}

} // namespace

int main() {
    std::cout << "Starting Qwen3 Vision Encoder Tests...\n";
    try {
        if (test_config_constants() != 0) return 1;
        if (test_workspace_planning() != 0) return 1;
        if (test_output_binding() != 0) return 1;
        if (test_weights_validation() != 0) return 1;
        if (test_encoder_execution_synthetic() != 0) return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal exception in test_vision_encoder: " << ex.what() << "\n";
        return 1;
    }

    std::cout << "OK Qwen3 Vision Encoder Tests\n";
    return 0;
}
