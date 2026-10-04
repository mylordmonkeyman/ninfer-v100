#include "targets/qwen3_8_flash_next/impl/expert_heat.h"
#include <array>
#include <stdexcept>
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while (false)
using namespace ninfer::targets::qwen3_8_flash_next::detail;
int main() {
    FlashNextExpertHeat count;
    const std::array<std::int32_t, 3> routes{1, 1, 2};
    count.observe(0, routes); count.observe(0, routes);
    CHECK(count.score(0,1) == 4 && count.score(0,2) == 2);
    CHECK(count.score(1,1) == 0);
    FlashNextExpertHeat decay(.5, 2);
    decay.observe(0, routes); CHECK(decay.score(0,1) == 2);
    decay.observe(0, routes); CHECK(decay.score(0,1) == 4);
    decay.observe(0, {}); decay.observe(0, {});
    CHECK(decay.score(0,1) == 2);
    decay.reset(); CHECK(decay.score(0,1) == 0);
    std::array<std::array<int,512>,48> ranking{};
    for (auto& row:ranking) for (int id=0;id<512;++id) row[id]=id;
    std::swap(ranking[0][0],ranking[0][9]);
    FlashNextExpertHeat prior;
    prior.set_prior(ranking,512);
    CHECK(prior.score(0,9)==512 && prior.measured(0,9)==0);
    prior.observe(0,std::array<std::int32_t,1>{0});
    CHECK(prior.measured(0,0)==1 && prior.score(0,0)==504);
    prior.reset(); CHECK(prior.score(0,9)==512 && prior.measured(0,0)==0);
    bool rejected = false;
    try { count.observe(0, std::array<std::int32_t,2>{1,512}); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected && count.score(0,1) == 4);
}
