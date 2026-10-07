#include "targets/qwen3_8_flash_next/impl/telemetry/lifecycle_telemetry.h"
#include <array>
using namespace v100_compare;
int main() {
    if (!level()) return 0;
    RoundLink link("ninfer");
    if (linked_round_id != link.id()) return 1;
    { RoundLink nested("strata"); if (linked_round_id != nested.id()) return 1; }
    if (linked_round_id != link.id()) return 1;
    const std::array<int32_t, 3> candidates{101,202,303}, inputs{99,101,202};
    for (const char* engine : {"ninfer", "strata"}) {
        LifecycleEvent v{.engine=engine, .round_id=link.id(), .sequence_trace_id="fixture:sequence",
            .event="verified", .first_token_index=42, .proposed_drafts=3, .accepted_drafts=2,
            .verified_tokens=3, .token_semantics="verified_output_candidates", .proposal_source="mtp"};
        v.tokens(std::span<const int32_t>(candidates)); v.emit();
        LifecycleEvent c{.engine=engine, .round_id=link.id(), .sequence_trace_id="fixture:sequence",
            .event="commit_returned", .first_token_index=42};
        if (std::string_view(engine)=="ninfer") {
            c.token_semantics="output_tokens"; c.tokens(std::span<const int32_t>(candidates.data(),2));
        } else {
            c.token_semantics="input_state_prefix"; c.tokens(std::span<const int32_t>(inputs));
        }
        c.emit();
        if (std::string_view(engine)=="strata") {
            LifecycleEvent e{.engine=engine, .round_id=link.id(), .sequence_trace_id="fixture:sequence",
                .event="emitted", .first_token_index=42, .token_semantics="output_tokens"};
            e.tokens(std::span<const int32_t>(candidates.data(),1)); e.emit();
        }
    }
    return 0;
}
