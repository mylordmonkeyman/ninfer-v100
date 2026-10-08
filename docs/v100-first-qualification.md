# First V100 telemetry qualification

Run **V100 Telemetry First Hardware Qualification** on branch
`perf/v100-strata-derived`. The workflow never cancels
another job. It uses the existing `v100` runner and CUDA 12.8, refuses to proceed
if GPU0 has less than 28,000 MiB free, and does not stop unrelated processes.
It uses the already registered `v100-strata-compare.yml` workflow slot, so the
Actions menu may still show its default-branch name, **V100 NInfer vs Strata
Comparison**. Select `perf/v100-strata-derived` in **Run workflow**; this branch
executes the qualification job described here, not the earlier comparison.
The assistant can also launch it through the previously used opt-in commit
mechanism: a commit changing this workflow file on that branch must contain
`[v100:telemetry-qualify]`. Ordinary commits cannot allocate the hardware job.

This is the first hardware gate, not the comprehensive attribution campaign or
an optimization screen. The user's October 7 instruction to reach actual V100
testing is served by qualifying the instrumentation already implemented while
reporting remaining coverage explicitly. The shared comprehensive specification
still governs the later attribution matrix.

The single job builds NInfer and frozen Strata
`98cbf622124ab25f89529ff0f56eebbf9fdfd5fd` once in separate build directories.
It verifies that `/opt/ai/strata` remains at installed baseline
`ad5206fba4914b4ab0ffb5e1d17bae4cd77ba778`. Existing model and tokenizer files
are reused. Strata's copied configuration selects the newly built executable,
GPU0 and telemetry level; it disables prompt reuse and profile saving without
editing the installed configuration. Other native Strata settings are retained.
NInfer uses the existing validated static resident profile, 156 cache slots,
4K context, FP8 KV storage, default graph behavior, native MTP draft window 1
and disabled prefix reuse. These settings are fixed across telemetry levels.
They are not a claim that the engines have matched quantization or MTP policy.

Known-work NInfer GPU cache and stream fixtures run at levels 0/1/2 before full
model loading. A skip or error fails the job. Each engine then runs levels 0/1/2,
alternating engine order, with five warmups and three measured requests per
process (revised after run 37713168658 showed substantial Strata cache-warmup
drift). The identical greedy, seeded prompt requests 64 output tokens. Strata's
OpenAI frontend reads `chat_template_kwargs.enable_thinking=false`, not the
top-level `enable_thinking=false` field; both are included for Strata and
answer content is required so an accidental reasoning-only test cannot pass.
Only
one request is in flight. Models remain loaded for warmups/repeats; only this
job's launched process groups are stopped between cells.

The observer preserves actual request payloads/responses, monotonic start/end,
content/reasoning bytes and native log offsets. It labels these as HTTP observer
boundaries, not invented native request IDs or emitted model-token counts.
Native round, lifecycle and draft records are validated and preserved. System
sampling runs at 200 ms. It observes the server PID; Strata child CPU/memory
joining remains a gap, while GPU samples describe GPU0.

The report checks exact content/reasoning/tool payload equality within each
engine across telemetry levels and requires repeatable level-0 output. It
separately reports preexisting baseline nondeterminism, hashes absent from
baseline observations and first-to-last measured latency trends, without
relaxing the exact-output gate. Measurements with >5% sequential request drift
remain unqualified even if the nominal median overhead is small. It
reports median HTTP-wall overhead and flags level-1 overhead at or above 2%,
or sample ranges above 5%, as unqualified. It does not compare numerical outputs
across quantizations, measure pure decode overhead, infer a GPU critical path,
or qualify an optimization. Overhead findings require interpretation rather
than an automatic policy change.

The artifact `v100-telemetry-qualification-<run>-<attempt>` contains frozen source
SHAs, CMake caches, installed configuration identity, environment/GPU/NUMA data,
GPU fixture logs, executable/profile/config hashes, request records, native logs,
system samples and summaries. Partial failures are also uploaded. The NInfer
artifact path/stat is recorded; full model/tokenizer content identities and
teacher-forced input traces remain required for the comprehensive manifest.

Next, use this evidence to fix actual instrumentation/build failures, qualify
level-1 overhead, and finish native request correlation and deferred GPU
interval/launch coverage. Then run the comprehensive attribution matrix frozen
by the shared specification. Do not start Phase 18/SV8 or an optimization A/B
from this initial report.
