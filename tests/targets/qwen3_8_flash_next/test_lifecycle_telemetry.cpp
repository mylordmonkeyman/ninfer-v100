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
        {
            DraftCall draft(engine, "fixture:sequence", 42);
            draft.complete(std::span<const int32_t>(candidates));
        }
        { DraftCall failed(engine, "fixture:sequence", 45, link.id()); }
        Round native(engine, link.id(), "verify");
        InputContext context;
        context.input_columns = 4;
        context.spans.push_back({0,4,42,std::nullopt,std::nullopt});
        if (level() >= 2) context.input_token_ids = std::vector<std::int64_t>{99,101,202,303};
        native.context(std::move(context));
        const std::array<int32_t,4> sampled{101,202,303,404};
        native.sampled_tokens(std::span<const int32_t>(sampled));
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
    link.finish();
    if (!linked_round_id.empty() || link.id().empty()) return 1;
    return 0;
}
