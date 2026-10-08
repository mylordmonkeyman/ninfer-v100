#include "targets/qwen3_8_flash_next/impl/stream_order.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

using ninfer::targets::qwen3_8_flash_next::detail::flash_next_stream_expert_order;

int main() {
    std::array<std::size_t,512> routes{};
    routes[4] = 3;
    routes[17] = 11;
    routes[200] = 11;
    routes[511] = 1;
    const auto id = flash_next_stream_expert_order(routes,false);
    assert(id.size == 4);
    assert((std::vector<unsigned>(id.ids.begin(),id.ids.begin()+id.size) ==
            std::vector<unsigned>{4,17,200,511}));
    const auto prioritized = flash_next_stream_expert_order(routes,true);
    assert(prioritized.size == 4);
    // Decreasing routed-token work; equal work uses expert ID as stable tiebreak.
    assert((std::vector<unsigned>(prioritized.ids.begin(),
                                  prioritized.ids.begin()+prioritized.size) ==
            std::vector<unsigned>{17,200,4,511}));
    for(unsigned expert: prioritized.ids)
        assert(expert < 512);
    routes.fill(0);
    assert(flash_next_stream_expert_order(routes,true).size==0);
    routes.fill(1);
    const auto all = flash_next_stream_expert_order(routes,true);
    assert(all.size == 512);
    for(unsigned i=0;i<512;++i) assert(all.ids[i]==i);
    bool rejected=false;
    try {
        flash_next_stream_expert_order(
            std::span<const std::size_t>(routes).first(511),true);
    } catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected);
}
