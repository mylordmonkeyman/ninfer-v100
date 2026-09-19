#pragma once

#include "ninfer/types.h"
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::runtime {

class CompiledOutputConstraint;

// Compatibility shell for the newer structured-output runtime. The V100 branch does not expose
// structured-output request options, so this state is never constructed in normal serving.
class OutputConstraintState {
public:
    OutputConstraintState(std::shared_ptr<const CompiledOutputConstraint>, bool) {}
    [[nodiscard]] std::span<const std::int32_t> next_mask() const noexcept { return mask_; }
    [[nodiscard]] OutputConstraintState fork() const { return *this; }
    [[nodiscard]] bool try_accept(TokenId) noexcept { return true; }
    [[nodiscard]] bool terminated() const noexcept { return false; }
    void accept(std::span<const TokenId>) noexcept {}
private:
    std::vector<std::int32_t> mask_;
};

} // namespace ninfer::runtime
