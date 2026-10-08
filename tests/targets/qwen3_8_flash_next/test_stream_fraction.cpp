#include "targets/qwen3_8_flash_next/impl/stream_fraction.h"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string_view>

using ninfer::targets::qwen3_8_flash_next::detail::flash_next_parse_expert_stream_fraction;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_parse_prefill_stream_min_routes;

int main() {
    const char* name="NINFER_V100_PREFILL_EXPERT_STREAM_FRACTION";
    assert(flash_next_parse_expert_stream_fraction(nullptr,name)==1.0);
    assert(flash_next_parse_expert_stream_fraction("",name)==1.0);
    assert(flash_next_parse_expert_stream_fraction("1",name)==1.0);
    assert(flash_next_parse_expert_stream_fraction("0.9",name)==0.9);
    assert(flash_next_parse_expert_stream_fraction("0.75",name)==0.75);
    for (const char* invalid:{"0","-0.2","1.01","nan","inf","bad","0.5extra","1e1000"}) {
        bool rejected=false;
        try { (void)flash_next_parse_expert_stream_fraction(invalid,name); }
        catch (const std::invalid_argument& e) {
            rejected=std::string_view(e.what()).find(name)!=std::string_view::npos;
        }
        assert(rejected);
    }
    assert(flash_next_parse_prefill_stream_min_routes(nullptr) == 0);
    assert(flash_next_parse_prefill_stream_min_routes("") == 0);
    assert(flash_next_parse_prefill_stream_min_routes("0") == 0);
    assert(flash_next_parse_prefill_stream_min_routes("1") == 1);
    assert(flash_next_parse_prefill_stream_min_routes("10") == 10);
    assert(flash_next_parse_prefill_stream_min_routes("12") == 12);
    assert(flash_next_parse_prefill_stream_min_routes("14") == 14);
    assert(flash_next_parse_prefill_stream_min_routes("4096") == 4096);
    for (const char* invalid : {"-1", "+1", " 8", "8 ", "8extra",
                                "0.5", "4097", "999999999999999", "0x10"}) {
        bool rejected = false;
        try { (void)flash_next_parse_prefill_stream_min_routes(invalid); }
        catch (const std::invalid_argument& e) {
            rejected = std::string_view(e.what()).find(
                "NINFER_V100_PREFILL_EXPERT_STREAM_MIN_ROUTES") != std::string_view::npos;
        }
        assert(rejected);
    }
}
