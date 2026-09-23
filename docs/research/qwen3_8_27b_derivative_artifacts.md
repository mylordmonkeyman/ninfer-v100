# Qwen3.8-27B derivative qualification

The selected [OrcaRouter NVFP4 checkpoint](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4)
now has a separate registered artifact, converter and Windows launcher entry. The implementation
contract and source pin are maintained in the [artifact reference](../maintainer/qwen3.8-27b-artifact.md#14-orcarouter-nvfp4-derivative).
The [performance reference](../performance.md#orcarouter-nvfp4-integration-probe) records the initial
serving measurements and manual review findings. Upstream
[issue #231](https://github.com/Neroued/ninfer/issues/231) motivated this investigation; the source
selected by the user is the NVFP4 release, rather than the issue's BF16/FP8 examples.

## Remaining decision

The user reports strong quality with this model in Unsloth Studio and, after testing the NInfer
integration, reports that it behaves as expected. Its exact Studio revision, matched runtime
settings and successful tasks have not been recorded, so controlled equivalence is not established.
The engine integration preserves the selected source's BF16 endpoints and frontend semantics;
its existing MTP/Vision and companion conversions retain NInfer's documented arithmetic profiles.

Qualification should next replay the same substantial coding and isolated lab-analysis tasks in
both engines, with matching reasoning/sampling settings. Review the actual analysis, execute the
proposed tests, and check long multi-tool sessions and continuation behavior. The initial queue
repair probe already produced concrete coding/test mistakes, despite valid APIs and useful
speculative acceptance. Neither lower refusal nor higher acceptance establishes task correctness.

MTP and DFlash2 remain optional. Compare ordinary decoding and each backend on the matched tasks
before choosing a default. DFlash2's companion was trained for the canonical target; feature
compatibility allows execution but does not prove equivalent behavior on this derivative.
