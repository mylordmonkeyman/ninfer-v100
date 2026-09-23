#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include "core/device.h"
#include "core/layout.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include <ninfer/targets/qwen3_vision/vision.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

::ninfer::targets::qwen3_vision::Weights adapt_vision_weights(const LoadedModelData& weights) {
    if (!weights.vision) {
        throw std::invalid_argument("Vision execution was requested without materialized weights");
    }
    const auto& vision = *weights.vision;
    ::ninfer::targets::qwen3_vision::Weights out;
    out.patch_embed      = &vision.common.patch_embedding;
    out.patch_embed_bias = &vision.common.patch_embedding_bias;
    out.position_embed   = &vision.common.position_embedding;
    for (std::size_t i = 0; i < out.layers.size(); ++i) {
        const auto& src     = vision.common.layers[i];
        auto& dst           = out.layers[i];
        dst.norm1_weight    = &src.norm1_weight;
        dst.norm1_bias      = &src.norm1_bias;
        dst.qkv             = &src.qkv;
        dst.qkv_bias        = &src.qkv_bias;
        dst.projection      = &src.output;
        dst.projection_bias = &src.output_bias;
        dst.norm2_weight    = &src.norm2_weight;
        dst.norm2_bias      = &src.norm2_bias;
        dst.fc1             = &src.fc1;
        dst.fc1_bias        = &src.fc1_bias;
        dst.fc2             = &src.fc2;
        dst.fc2_bias        = &src.fc2_bias;
    }
    out.merger.norm_weight = &vision.common.merger_norm_weight;
    out.merger.norm_bias   = &vision.common.merger_norm_bias;
    out.merger.fc1         = &vision.common.merger_fc1;
    out.merger.fc1_bias    = &vision.common.merger_fc1_bias;
    out.merger.fc2         = &vision.merger_fc2;
    out.merger.fc2_bias    = &vision.merger_fc2_bias;
    out.output_hidden      = VisionConfig::output_hidden;
    return out;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a * b;
}

} // namespace

VisionContext::VisionContext(DeviceContext& ctx, const LoadedModelData& weights)
    : encoder_(ctx, adapt_vision_weights(weights)) {}

std::size_t VisionContext::workspace_bytes(const qwen3_6::VisionItemControl& item) {
    return ::ninfer::targets::qwen3_vision::Encoder::workspace_bytes(item.patch_count, item.merged_count);
}

std::size_t VisionContext::workspace_bytes(std::size_t patches, std::size_t merged_tokens) {
    return ::ninfer::targets::qwen3_vision::Encoder::workspace_bytes(patches, merged_tokens);
}

VisionWorkspacePlan VisionContext::plan_workspace(std::uint32_t max_merged_tokens,
                                                  std::size_t general_capacity_bytes) {
    return ::ninfer::targets::qwen3_vision::Encoder::plan_workspace(max_merged_tokens, general_capacity_bytes,
                                                                   VisionConfig::output_hidden);
}

Tensor VisionContext::bind_output(DeviceSpan backing, const VisionWorkspacePlan& plan,
                                  std::size_t merged_tokens) {
    return ::ninfer::targets::qwen3_vision::Encoder::bind_output(backing, plan, merged_tokens,
                                                                VisionConfig::output_hidden);
}

void VisionContext::encode(const VisionItemView& item, Tensor& output, DeviceSpan backing,
                           const VisionWorkspacePlan& plan) const {
    if (item.control == nullptr) { throw std::invalid_argument("Vision item control is null"); }
    const qwen3_6::VisionItemControl& control = *item.control;
    const ::ninfer::targets::qwen3_vision::EncodeItemView item_view{
        .patches                = item.patches,
        .patch_count            = control.patch_count,
        .merged_count           = control.merged_count,
        .segment_length         = control.segment_length,
        .position_ids           = control.position_ids,
        .position_table_indices = control.position_table_indices,
        .position_table_weights = control.position_table_weights,
    };
    encoder_.encode(item_view, output, backing, plan);
}

VisionPrefillSession::VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                                           DeviceSpan workspace,
                                           const VisionWorkspacePlan& workspace_plan,
                                           qwen3_6::PreparedPromptData& prompt,
                                           const VisionPrefillPlan& plan,
                                           std::size_t& handoff_peak_bytes)
    : device_(device), workspace_(workspace), workspace_plan_(workspace_plan), prompt_(prompt),
      plan_(plan), handoff_peak_bytes_(handoff_peak_bytes), context_(device, model) {
    if (plan_.control == nullptr || plan_.control->items.empty() || plan_.uses.empty()) {
        throw std::invalid_argument("Vision prefill plan has no suffix item spans");
    }
    if (workspace_.data == nullptr || workspace_.bytes < workspace_plan_.capacity_bytes ||
        plan_.max_merged_count == 0 || plan_.max_merged_count > workspace_plan_.max_merged_tokens) {
        throw std::invalid_argument("Vision prefill workspace plan is invalid");
    }
    std::uint32_t previous_end = 0;
    std::optional<std::uint32_t> previous_item;
    for (const VisionUseSpan& use : plan_.uses) {
        if (use.begin >= use.end || use.begin < previous_end ||
            use.end > prompt_.token_ids.size()) {
            throw std::invalid_argument("Vision suffix item spans are invalid or unordered");
        }
        if (use.control_index >= plan_.control->items.size() ||
            use.prepared_item_index >= prompt_.vision_items.size() ||
            use.prepared_item_index >= prompt_.media_payloads.size() ||
            plan_.control->prepared_item_begin + use.control_index != use.prepared_item_index ||
            (previous_item && use.prepared_item_index <= *previous_item)) {
            throw std::invalid_argument("Vision suffix item indices are invalid or unordered");
        }
        const qwen3_6::VisionItemControl& control = plan_.control->items[use.control_index];
        const qwen3_6::VisionItem& source         = prompt_.vision_items[use.prepared_item_index];
        if (control.scatter_indices.empty() ||
            use.end != static_cast<std::uint32_t>(control.scatter_indices.back()) + 1U ||
            (use.begin != static_cast<std::uint32_t>(control.scatter_indices.front()) &&
             use.begin + 1U != static_cast<std::uint32_t>(control.scatter_indices.front())) ||
            source.modality != control.modality || source.grid.temporal != control.grid.temporal ||
            source.grid.height != control.grid.height || source.grid.width != control.grid.width ||
            source.patch_begin != control.patch_begin ||
            source.patch_count != control.patch_count) {
            throw std::invalid_argument("Vision suffix plan does not describe the prepared item");
        }
        if (control.merged_count > plan_.max_merged_count) {
            throw std::invalid_argument("Vision suffix item exceeds its request workspace extent");
        }
        const Tensor output =
            VisionContext::bind_output(workspace_, workspace_plan_, control.merged_count);
        const std::size_t patch_elements = checked_mul(
            control.patch_count, static_cast<std::size_t>(VisionScheduleConfig::patch_dim),
            "item patch elements");
        const auto& payload = prompt_.media_payloads[use.prepared_item_index];
        if (output.bytes() > workspace_plan_.handoff_capacity_bytes || !payload ||
            payload->patch_elements != patch_elements) {
            throw std::invalid_argument("Vision suffix item storage has an invalid shape");
        }
        previous_end  = use.end;
        previous_item = use.prepared_item_index;
    }
    if (plan_.max_merged_count != 0 &&
        std::none_of(plan_.control->items.begin(), plan_.control->items.end(),
                     [&](const qwen3_6::VisionItemControl& item) {
                         return item.merged_count == plan_.max_merged_count;
                     })) {
        throw std::invalid_argument("Vision request workspace extent has no matching suffix item");
    }
    encoded_payloads_pending_release_.reserve(plan_.uses.size());
    timers_.reserve(plan_.uses.size());
}

VisionChunk VisionPrefillSession::prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length) {
    if (nominal_length == 0 || begin >= prompt_.token_ids.size()) {
        throw std::invalid_argument("Vision chunk range is empty or outside the prompt");
    }
    const std::uint64_t nominal_end64 =
        static_cast<std::uint64_t>(begin) + static_cast<std::uint64_t>(nominal_length);
    std::uint32_t end = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(nominal_end64, prompt_.token_ids.size()));

    while (next_use_ < plan_.uses.size() && plan_.uses[next_use_].end <= begin) { ++next_use_; }
    const VisionUseSpan* active = nullptr;
    if (next_use_ < plan_.uses.size() && plan_.uses[next_use_].begin < end) {
        active = &plan_.uses[next_use_];
        if (next_use_ + 1U < plan_.uses.size()) {
            end = std::min(end, plan_.uses[next_use_ + 1U].begin);
        }
    }
    if (end <= begin) { throw std::logic_error("Vision chunk cap made no forward progress"); }
    if (active == nullptr) {
        return VisionChunk{static_cast<std::int32_t>(end - begin), nullptr, {}};
    }
    const qwen3_6::VisionItemControl& control = plan_.control->items[active->control_index];
    Tensor output = VisionContext::bind_output(workspace_, workspace_plan_, control.merged_count);

    if (!active_item_ || *active_item_ != active->prepared_item_index) {
        const auto& payload = prompt_.media_payloads[active->prepared_item_index];
        timers_.emplace_back(device_);
        timers_.back().start();
        context_.encode(VisionItemView{payload->span(), &control}, output, workspace_,
                        workspace_plan_);
        timers_.back().record_stop();
        active_item_          = active->prepared_item_index;
        active_handoff_bytes_ = output.bytes();
        handoff_peak_bytes_   = std::max(handoff_peak_bytes_, active_handoff_bytes_);
        encoded_payloads_pending_release_.push_back(active->prepared_item_index);
    }
    return VisionChunk{static_cast<std::int32_t>(end - begin), &control, output};
}

void VisionPrefillSession::release_encoded_media_payloads() noexcept {
    for (const std::uint32_t item_index : encoded_payloads_pending_release_) {
        if (item_index >= prompt_.media_payloads.size()) { std::terminate(); }
        prompt_.media_payloads[item_index].reset();
    }
    encoded_payloads_pending_release_.clear();
}

void VisionPrefillSession::retire_handoff() noexcept {
    active_item_.reset();
    active_handoff_bytes_ = 0;
}

double VisionPrefillSession::elapsed_seconds() const {
    double milliseconds = 0.0;
    for (const CudaEventTimer& timer : timers_) { milliseconds += timer.elapsed_ms(); }
    return milliseconds / 1000.0;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
