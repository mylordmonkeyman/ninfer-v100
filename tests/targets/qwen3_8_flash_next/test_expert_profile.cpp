#include "targets/qwen3_8_flash_next/impl/expert_profile.h"
#include <iostream>
using namespace ninfer::targets::qwen3_8_flash_next::detail;
void require(bool ok) { if (!ok) throw std::runtime_error("profile test failed"); }
template<class F> void rejects(F f) {
    bool rejected=false; try { f(); } catch (const std::exception&) { rejected=true; }
    require(rejected);
}
int main() { try {
    nlohmann::json j={{"magic","NINFER_V100_EXPERT_PROFILE"},{"version",2},
        {"layers",48},{"experts_per_layer",512},{"model_id","model"},{"weights_id","weights"}};
    j["ranking"]=nlohmann::json::array();
    for (int layer=0;layer<48;++layer) {
        std::vector<int> row; for (int i=0;i<512;++i) row.push_back((i+layer)%512);
        j["ranking"].push_back(row);
    }
    auto p=FlashNextExpertProfile::parse(j,"model","weights");
    require(p.ranking[47][0]==47 && p.ranking[47][511]==46);
    rejects([&]{FlashNextExpertProfile::parse(j,"other","weights");});
    rejects([&]{FlashNextExpertProfile::parse(j,"model","other");});
    rejects([&]{FlashNextExpertProfile::parse(j,"","");});
    auto bad=j; bad["ranking"][0][1]=0;
    rejects([&]{FlashNextExpertProfile::parse(bad,"model","weights");});
    bad=j; bad["ranking"][0][1]=512;
    rejects([&]{FlashNextExpertProfile::parse(bad,"model","weights");});
    bad=j; bad["ranking"][0][1]=1.5;
    rejects([&]{FlashNextExpertProfile::parse(bad,"model","weights");});
    bad=j; bad["ranking"].erase(0);
    rejects([&]{FlashNextExpertProfile::parse(bad,"model","weights");});
    bad=j; bad["version"]=1;
    rejects([&]{FlashNextExpertProfile::parse(bad,"model","weights");});
    rejects([&]{FlashNextExpertProfile::load("/nonexistent-v100-profile","model","weights");});
    FlashNextExpertProfile::Heat heat{};
    FlashNextExpertProfile::Residents resident{};
    resident[0][10]=true; heat[0][11]=10; heat[0][12]=10;
    std::swap(p.ranking[0][11],p.ranking[0][12]);
    auto learned=p.learned(heat,resident);
    require(learned.ranking[0][0]==10 && learned.ranking[0][1]==12 &&
        learned.ranking[0][2]==11 && learned.ranking[0][3]==0);
    const auto directory=std::filesystem::temp_directory_path()/"ninfer-v100-profile-host-test";
    std::filesystem::create_directory(directory);
    const auto path=directory/"profile.json";
    learned.save(path,heat);
    p.save(path,heat); // Atomic replacement, not append.
    const auto restored=FlashNextExpertProfile::load(path.c_str(),"model","weights");
    require(restored.ranking==p.ranking && !std::filesystem::exists(path.string()+".tmp"));
    rejects([&]{learned.save(directory/"absent"/"profile.json",heat);});
    heat[0][0]=std::numeric_limits<double>::infinity();
    rejects([&]{(void)p.learned(heat,resident);});
    std::filesystem::remove_all(directory);
    std::cout<<"PASS: identity, schema, rankings, learned ordering, atomic save/reload\n";
    return 0;
} catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
