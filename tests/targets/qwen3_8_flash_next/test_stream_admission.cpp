#include "targets/qwen3_8_flash_next/impl/stream_admission.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <span>
#include <string_view>

using ninfer::targets::qwen3_8_flash_next::detail::
    flash_next_stream_prefill_admit_candidate;
using ninfer::targets::qwen3_8_flash_next::detail::
    flash_next_stream_prefill_admit_hot;

int main() {
    assert(!flash_next_stream_prefill_admit_hot(nullptr));
    assert(!flash_next_stream_prefill_admit_hot(""));
    assert(!flash_next_stream_prefill_admit_hot("off"));
    assert(flash_next_stream_prefill_admit_hot("hot"));
    for (const char* invalid:{"yes","true","HOT","1","x"}) {
        bool rejected=false;
        try { (void)flash_next_stream_prefill_admit_hot(invalid); }
        catch(const std::invalid_argument& e) {
            rejected=std::string_view(e.what()).find("NINFER_V100_PREFILL_STREAM_ADMIT")!=
                std::string_view::npos;
        }
        assert(rejected);
    }
    std::array<std::size_t,512> n{};
    assert(flash_next_stream_prefill_admit_candidate(n)==-1);
    n[45]=23; n[33]=20; n[511]=23;
    assert(flash_next_stream_prefill_admit_candidate(n)==45);
    n[8]=24;
    assert(flash_next_stream_prefill_admit_candidate(n)==8);
    n[8]=0;
    assert(flash_next_stream_prefill_admit_candidate(n)==45);
    n.fill(1);
    assert(flash_next_stream_prefill_admit_candidate(n)==0);
    bool rejected=false;
    try {
        (void)flash_next_stream_prefill_admit_candidate(
            std::span<const std::size_t>(n).first(511));
    } catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
}
