#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <memory>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Inference-owner state. Route observations are merged once per layer call;
// no worker thread or per-token cache lock touches the heat table.
class FlashNextExpertHeat {
public:
    FlashNextExpertHeat(double decay = 1.0, unsigned interval = 1)
        : decay_(decay), interval_(interval) {
        if (!std::isfinite(decay) || decay <= 0 || decay > 1 || !interval)
            throw std::invalid_argument("expert heat requires decay in (0,1] and a positive interval");
    }
    void observe(unsigned layer, std::span<const std::int32_t> ids) {
        if (layer >= 48) throw std::invalid_argument("invalid heat layer");
        for (int id : ids)
            if (id < 0 || id >= 512) throw std::invalid_argument("invalid heat expert");
        for (int id : ids) ++pending_[layer][id];
        if (++calls_[layer] == interval_) {
            for (unsigned expert = 0; expert < 512; ++expert) {
                heat_[layer][expert] = decay_ * heat_[layer][expert] + pending_[layer][expert];
                pending_[layer][expert] = 0;
            }
            calls_[layer] = 0;
        }
    }
    void set_prior(const std::array<std::array<int,512>,48>& ranking, double weight) {
        if (!std::isfinite(weight) || weight<=0)
            throw std::invalid_argument("profile prior weight must be positive and finite");
        auto prior=std::make_unique<std::array<std::array<double,512>,48>>();
        for (unsigned layer=0;layer<48;++layer) {
            std::array<bool,512> seen{};
            for (unsigned rank=0;rank<512;++rank) {
                const int id=ranking[layer][rank];
                if (id<0 || id>=512 || seen[id]) throw std::invalid_argument("invalid prior ranking");
                seen[id]=true;
                (*prior)[layer][id]=weight*(512-rank)/512.0;
            }
        }
        prior_=std::move(prior);
    }
    [[nodiscard]] double measured(unsigned layer, unsigned expert) const {
        return heat_.at(layer).at(expert) + pending_.at(layer).at(expert);
    }
    [[nodiscard]] double score(unsigned layer, unsigned expert) const {
        return measured(layer,expert) + (prior_ ? prior_->at(layer).at(expert) : 0);
    }
    void reset() { heat_ = {}; pending_ = {}; calls_ = {}; }
private:
    std::array<std::array<double, 512>, 48> heat_{};
    std::array<std::array<std::uint64_t, 512>, 48> pending_{};
    std::array<unsigned, 48> calls_{};
    std::unique_ptr<std::array<std::array<double,512>,48>> prior_;
    double decay_;
    unsigned interval_;
};
} // namespace ninfer::targets::qwen3_8_flash_next::detail
