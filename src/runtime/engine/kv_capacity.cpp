#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

void validate_curve(const SequenceCapacityCurve& curve) {
    if (curve.main_page_tokens == 0 || curve.minimum_main_page_groups == 0 ||
        curve.minimum_main_page_groups > curve.maximum_main_page_groups ||
        curve.minimum_device_reservation_bytes == 0) {
        throw std::invalid_argument("sequence capacity curve is invalid");
    }
    if (curve.minimum_main_page_groups < curve.maximum_main_page_groups &&
        curve.bytes_per_additional_main_page_group == 0) {
        throw std::invalid_argument("expandable sequence capacity curve has zero byte stride");
    }
}

std::uint32_t explicit_page_groups(const KvCapacityPolicy& policy,
                                   const SequenceCapacityCurve& curve) {
    if (policy.explicit_tokens == 0) {
        throw std::invalid_argument("explicit KV capacity must be positive");
    }
    const std::uint64_t pages =
        1ULL + (static_cast<std::uint64_t>(policy.explicit_tokens) - 1ULL) / curve.main_page_tokens;
    if (pages > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("explicit KV page count exceeds uint32");
    }
    return static_cast<std::uint32_t>(pages);
}

} // namespace

std::size_t SequenceCapacityCurve::reservation_bytes(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument(
            "Main KV page count " + std::to_string(main_page_groups) +
            " is outside the target capacity curve [" + std::to_string(minimum_main_page_groups) +
            ", " + std::to_string(maximum_main_page_groups) + "] (" +
            std::to_string(main_page_tokens) + " tokens per group, so " +
            std::to_string(static_cast<std::uint64_t>(minimum_main_page_groups) * main_page_tokens) +
            " to " +
            std::to_string(static_cast<std::uint64_t>(maximum_main_page_groups) * main_page_tokens) +
            " tokens)");
    }
    const std::size_t additional =
        static_cast<std::size_t>(main_page_groups - minimum_main_page_groups);
    return checked_add(minimum_device_reservation_bytes,
                       checked_mul(additional, bytes_per_additional_main_page_group,
                                   "sequence capacity increment overflows size_t"),
                       "sequence capacity reservation overflows size_t");
}

std::uint32_t SequenceCapacityCurve::resolved_tokens(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument(
            "Main KV page count " + std::to_string(main_page_groups) +
            " is outside the target capacity curve [" + std::to_string(minimum_main_page_groups) +
            ", " + std::to_string(maximum_main_page_groups) + "] (" +
            std::to_string(main_page_tokens) + " tokens per group, so " +
            std::to_string(static_cast<std::uint64_t>(minimum_main_page_groups) * main_page_tokens) +
            " to " +
            std::to_string(static_cast<std::uint64_t>(maximum_main_page_groups) * main_page_tokens) +
            " tokens)");
    }
    const std::uint64_t tokens = static_cast<std::uint64_t>(main_page_groups) * main_page_tokens;
    if (tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("resolved KV token capacity exceeds uint32");
    }
    return static_cast<std::uint32_t>(tokens);
}

KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                         const SequenceCapacityCurve& curve,
                                         std::size_t available_runtime_bytes) {
    validate_curve(curve);

    std::uint32_t pages         = curve.minimum_main_page_groups;
    std::size_t capacity_budget = available_runtime_bytes;
    const std::size_t required_slack =
        std::max(policy.automatic_headroom_bytes, policy.slack_floor_bytes);

    switch (policy.mode) {
    case KvCapacityMode::Explicit:
        if (policy.automatic_headroom_bytes != 0) {
            throw std::invalid_argument("explicit KV capacity must not carry automatic headroom");
        }
        pages = explicit_page_groups(policy, curve);
        if (policy.slack_floor_bytes > 0) {
            if (available_runtime_bytes < policy.slack_floor_bytes) {
                throw std::invalid_argument(
                    "explicit KV capacity requires at least " +
                    std::to_string(policy.slack_floor_bytes) +
                    " bytes of slack floor, but only " +
                    std::to_string(available_runtime_bytes) + " bytes are available after weights");
            }
            capacity_budget -= policy.slack_floor_bytes;
        }
        break;
    case KvCapacityMode::Automatic:
        if (available_runtime_bytes < required_slack) {
            throw std::invalid_argument(
                "automatic KV capacity requires at least " +
                std::to_string(required_slack) + " bytes of headroom/slack floor, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights");
        }
        capacity_budget -= required_slack;
        if (capacity_budget < curve.minimum_device_reservation_bytes) {
            throw std::invalid_argument(
                "minimum Engine runtime reservation requires " +
                std::to_string(curve.minimum_device_reservation_bytes) + " bytes in addition to " +
                std::to_string(required_slack) +
                " bytes of automatic headroom/slack floor, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights");
        }
        if (curve.minimum_main_page_groups < curve.maximum_main_page_groups) {
            const std::size_t additional =
                (capacity_budget - curve.minimum_device_reservation_bytes) /
                curve.bytes_per_additional_main_page_group;
            const std::uint64_t candidate =
                static_cast<std::uint64_t>(curve.minimum_main_page_groups) + additional;
            pages = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(candidate, curve.maximum_main_page_groups));
        }
        break;
    default:
        throw std::invalid_argument("unknown KV capacity policy");
    }

    const std::size_t reservation = curve.reservation_bytes(pages);
    if (reservation > capacity_budget) {
        if (policy.mode == KvCapacityMode::Explicit && policy.slack_floor_bytes > 0) {
            throw std::invalid_argument(
                "requested Engine runtime reservation requires " + std::to_string(reservation) +
                " bytes, which violates the required slack floor of " +
                std::to_string(policy.slack_floor_bytes) +
                " bytes (available after weights: " + std::to_string(available_runtime_bytes) +
                " bytes)");
        }
        throw std::invalid_argument("requested Engine runtime reservation requires " +
                                    std::to_string(reservation) + " bytes, but only " +
                                    std::to_string(capacity_budget) +
                                    " bytes are available for runtime capacity");
    }

    return KvCapacityResolution{
        .mode                                 = policy.mode,
        .main_page_groups                     = pages,
        .maximum_main_page_groups             = curve.maximum_main_page_groups,
        .resolved_tokens                      = curve.resolved_tokens(pages),
        .requested_concurrency                = 0,
        .effective_concurrency                = 0,
        .unbacked_concurrency_ratio           = 1.0,
        .minimum_runtime_reservation_bytes    = curve.minimum_device_reservation_bytes,
        .bytes_per_additional_main_page_group = curve.bytes_per_additional_main_page_group,
        .runtime_reservation_bytes            = reservation,
        .available_after_weights_bytes        = available_runtime_bytes,
        .available_after_startup_bytes        = 0,
        .automatic_headroom_bytes             = policy.automatic_headroom_bytes,
        .slack_floor_bytes                    = policy.slack_floor_bytes,
        .planned_slack_bytes                  = available_runtime_bytes - reservation,
    };
}

} // namespace ninfer::runtime
