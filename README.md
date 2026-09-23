# NInfer — workstation fork

> Specialized single-GPU inference for long-running agents and structured application workloads.

This is [igorls/ninfer](https://github.com/igorls/ninfer), a fork of
[Neroued/ninfer](https://github.com/Neroued/ninfer). It builds on upstream's from-scratch C++/CUDA
engine and explicitly registered `.ninfer` artifacts. The fork focuses on native Windows operation,
the NVIDIA RTX PRO 6000 Blackwell workstation, and Qwen3.8-Flash-Next, while retaining the upstream
Qwen 27B and 35B-A3B execution packages and Linux build path.

The matching Flash-Next artifact is published at
[igorls/Qwen3.8-Flash-Next-mixed-NInfer](https://huggingface.co/igorls/Qwen3.8-Flash-Next-mixed-NInfer).

The intended product is a dependable local inference service: efficient prefill and decode,
correct continuation reuse across long agent sessions, constrained JSON responses for applications,
and operational visibility into latency, memory pressure and failures. Performance changes must
preserve model semantics and improve the workload they claim to improve.

## What this fork adds

- **Native Windows builds and a Windows Supervisor.** A tray application and browser dashboard
  manage engine startup, shutdown and restart, inspect health and GPU memory, edit configuration,
  control the desktop memory reserve, and switch between explicitly configured model artifacts.
  The dashboard also exposes request/client activity and prefix-reuse behavior.
- **Qwen3.8-Flash-Next execution.** A separate target for QSA attention, Gated DeltaNet,
  hyper-connections, sparse MoE and mapped PLE, with text, Vision and MTP execution through
  the public Engine. Its mixed artifact uses NVFP4, FP8 and INT4 PLE storage.
- **Flash-Next performance and state work.** CUDA Graph decode, chunked prefill, selected-block
  attention, target-specific MoE kernels, FP8 KV and optional BF16 recurrent state, and checkpoint
  reservations that retain the rewrite needed for thinking/tool-history prefix reuse under pressure.
- **Native structured output.** JSON-object mode and a supported JSON Schema subset, constrained
  during generation through XGrammar. OpenAI Chat Completions, Responses and Anthropic Messages
  translate their documented formats into the same engine contract.
- **Application integration and diagnostics.** Streaming, tool-call parsing, reasoning controls,
  API-key authentication, request JSONL logs, client attribution and speculative/reuse telemetry.
  NInfer returns tool calls to clients; it does not execute them.

These capabilities are implemented in this branch. Application-specific quality qualification is
separate: valid JSON and fast inference do not establish correct legal analysis or reliable behavior
on every agent workload.

## Direction and boundaries

The priorities are sustained agent-session reliability, measured prefill/decode improvements on our
hardware, complete supported API behavior, and evaluation through real application workflows.
Upstream correctness fixes and useful kernels are integrated selectively, preserving this fork's
Windows and Flash-Next paths. New speculative backends or checkpoint families require their own
integration and qualification; their presence upstream does not make them available here.

The engine remains specialized: one GPU and one resident model per Engine, with one to eight active
requests configured at startup and bounded FIFO admission. Supervisor model switching replaces the
resident engine; it does not provide simultaneous model residency. Multi-GPU/distributed serving,
request preemption and arbitrary checkpoint loading are outside the current implementation.

## Models and hardware

The build targets **`sm_120a` only**. This fork's primary workstation is the **RTX PRO 6000
Blackwell 96 GB**. Upstream's published measurements use the **RTX 5090**; those numbers should
not be presented as measurements of this fork or of Flash-Next.

| Model | Registered weight profile | Artifact / reference |
|---|---|---|
| Qwen3.8-Flash-Next | `mixed-nvfp4-fp8-ple-int4` | [Published artifact on Hugging Face](https://huggingface.co/igorls/Qwen3.8-Flash-Next-mixed-NInfer) — [artifact contract](docs/maintainer/qwen3.8-flash-next-artifact.md), [converter](tools/convert/qwen3_8_flash_next/) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` — [upstream artifact](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.8-27B OrcaRouter Uncensored | `nvfp4` | Separate `qwen3.8-27b-orcarouter` identity, preserving BF16 embeddings/output head — [source and conversion](docs/maintainer/qwen3.8-27b-artifact.md#14-orcarouter-nvfp4-derivative) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` — [upstream artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` — [upstream artifact](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` — [upstream artifact](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` — [upstream artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Artifacts bind an exact model and weight profile and embed the required tokenizer/template resources.
They are not interchangeable GGUF or Transformers checkpoints. Flash-Next needs substantial host
memory as well as VRAM: its large PLE table is file-mapped on the host. An MoE's active parameter
count alone does not determine its storage requirements or execution cost.

Capabilities and memory policies differ by target. Flash-Next currently accepts BF16 or FP8 KV and
up to four MTP drafts; the Qwen3.6-family targets have additional KV formats and Device/Host cache
tiering. The 35B-A3B package has DFlash support. Qwen3.8-27B artifacts with companion weights can
select DFlash2 with `--spec dflash2 --draft-tokens 7 --lm-head-draft`; native Windows qualification
passes on the RTX PRO 6000, including concurrent requests, Host prefix restore, Vision and
structured output. See the [measurements and limits](docs/performance.md) before choosing between
DFlash2 and MTP. Consult the target references rather than transferring options between models unchanged.

## Build on Windows

Use a 64-bit Visual Studio C++ build environment and a CUDA-compatible host compiler. The example
below matches the local **Visual Studio 2026, CUDA 13.3 and CMake 4.3** toolchain. Use CUDA 13.3
for this Windows build: the compiled Flash-Next package requires the CCCL header
`cub/device/device_topk.cuh`, which CUDA 13.1 does not provide, even though it passes the project's
CUDA version check. This also applies when serving a 27B model because all execution packages are
compiled. CMake must be at least 3.28; the selected Visual Studio generator may require a newer
version. Not every compiler/CUDA combination is qualified.

Start with a text-only build, which does not need FFmpeg or libcurl:

```powershell
git clone https://github.com/igorls/ninfer.git
cd ninfer

cmake -S . -B build-win -G "Visual Studio 18 2026" -A x64 -DNINFER_BUILD_MEDIA=OFF
cmake --build build-win --config Release -j
```

Binaries for the CLI and HTTP engine are under `build-win/apps/Release/`; Supervisor is under
`build-win/apps/ninfer-supervisor/Release/`. Tests and benchmarks are opt-in through
`BUILD_TESTING` and `NINFER_BUILD_BENCHMARKS`.

For images/video, configure with `-DNINFER_BUILD_MEDIA=ON` and provide FFmpeg development libraries
(`libavformat >= 60`, `libavcodec >= 60`, `libavutil >= 58`, `libswscale >= 7`) and
`libcurl >= 7.85`. CMake accepts `FFMPEG_ROOT` and `CURL_ROOT` for Windows dependency locations.
Make the matching runtime DLLs available on `PATH`. A text-only build rejects Vision requests;
`--vision` alone does not add the missing media dependencies.

### Start a server

For an initial run, obtain the published 27B artifact with the Hugging Face CLI:

```powershell
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer `
  --revision 204e3d92c30d9d05f3300d2f52e443ad1edf6ddf --local-dir models

.\build-win\apps\Release\ninfer-serve.exe .\models\qwen3_8_27b_nvfp4.ninfer `
  --host 127.0.0.1 --port 8010 --model-id qwen3.8-27b `
  --max-context 32768 --kv-capacity 65536 --max-concurrency 2 `
  --kv-dtype fp8 --spec mtp --draft-tokens 3
```

The pinned 27B revision matches this fork's [NVFP4 model card](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md).
It provides the target artifact used by the MTP example above. For DFlash2, use an artifact
containing the complete companion bundle described in [the DFlash2 reference](docs/maintainer/qwen3.8-27b-dflash2.md).

For Flash-Next, read the [release model card](https://huggingface.co/igorls/Qwen3.8-Flash-Next-mixed-NInfer/blob/main/README.md)
for the compatible engine revision, memory requirements, quantization details and Qwen license.
The engineering preview is a 113.30 GB download and does not establish application quality.
On a workstation with sufficient device and host memory:

```powershell
hf download igorls/Qwen3.8-Flash-Next-mixed-NInfer `
  qwen3_8_flash_next_mixed.ninfer artifact-manifest.json SHA256SUMS `
  LICENSE NOTICE.md LICENSE-APACHE-2.0.txt README.md `
  --local-dir models

.\build-win\apps\Release\ninfer-serve.exe .\models\qwen3_8_flash_next_mixed.ninfer `
  --host 127.0.0.1 --port 8010 --model-id qwen3.8-flash-next `
  --max-context 32768 --kv-capacity 65536 --max-concurrency 2 `
  --kv-dtype fp8 --gdn-state-dtype bf16 --spec mtp --draft-tokens 4 `
  --preserve-thinking
```

Run one server at a time on that port. These are bounded starting profiles, not maximum allocations
or throughput recommendations. `--max-context` is a per-request logical limit; `--kv-capacity` is
the shared pool for active requests and retained prefixes. Choose context, concurrency, cache
capacity and desktop reserve together for the artifact and available memory. Use
`--api-key-file PATH` for authenticated serving and `--request-log-jsonl PATH` for request logs.

Send a request to the 27B example:

```powershell
$body = @{
  model = "qwen3.8-27b"
  messages = @(@{ role = "user"; content = "Return a JSON object with a short greeting." })
  max_tokens = 128
  reasoning_effort = "none"
  response_format = @{ type = "json_object" }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod http://127.0.0.1:8010/v1/chat/completions `
  -Method Post -ContentType "application/json" -Body $body
```

Set `model` to `qwen3.8-flash-next` for the Flash-Next example. See [HTTP serving](docs/serving.md)
for schemas, reasoning controls, authentication and protocol-specific fields. Unsupported schema
features return an explicit error; this is a documented subset, not arbitrary grammar support or
full Ollama API compatibility.

### Windows Supervisor

Edit a copy of the [example configuration](apps/ninfer-supervisor/supervisor.example.json) with
your executable, artifact, working directory and API-key paths. With the Visual Studio build above,
the executable path must include `apps/Release/ninfer-serve.exe`. Install the Windows app:

```powershell
.\scripts\windows\install.ps1 -ConfigPath .\supervisor.local.json
```

This installs binaries and runtime DLLs under `%LOCALAPPDATA%\Programs\NInfer`, adds a
**NInfer** Start menu entry and **Settings → Apps → Installed apps** entry, and enables startup
at sign-in. Configuration and logs live under `%LOCALAPPDATA%\NInfer`. The app is launched
by Windows Explorer and keeps running when the terminal, editor or coding agent closes.
The tray's **Start at login** setting controls subsequent sign-in startup. Signing out ends
this user-session app; it does not run as a system service before sign-in.

The installer copies the supplied configuration on first install, resolves artifact paths and
uses the installed engine executable. Models stay in their existing directories. Later installs
update the binaries and preserve the installed configuration and tray preferences. Build the
Release applications first; see [Windows app operations](docs/windows-app.md) for updates and removal.

The example dashboard listens at `http://127.0.0.1:8099`. Supervisor owns the child engine's lifecycle;
stop a manually launched server before managing the same port through Supervisor. An optional
explicit model catalog stores each artifact's own arguments. Supervisor is a Windows application;
the CLI and HTTP engine can also run directly.

The example catalog includes **27B Interactive — DFlash2** (7 draft tokens) and
**27B Concurrent — MTP** (5 draft tokens). Both use the combined DFlash2 artifact,
FP8 KV, eight lanes and a 32K shared KV pool; that pool is shared across requests,
not 32K per lane. These text presets match the performance qualification setup.
Select a preset in the dashboard model launcher and switch to load it. DFlash2
requires the [companion conversion](docs/maintainer/qwen3.8-27b-dflash2.md).

Under **Settings → Model features**, select the speculative backend, draft-token count
and optimized draft head. Backend choices follow the artifact identity; MTP allows
up to 5 drafts (4 for Flash-Next), and DFlash/DFlash2 up to 15. Turning speculation
off clears its dependent options. Save and restart applies the settings and keeps
them in the selected preset. Engine startup still validates companion weights.

## Linux build

The upstream build path remains available with a C++20 compiler, CUDA 13.1+, CMake 3.28+, Ninja,
`pkg-config`, and the media development libraries listed above:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8010 --max-context 32768 --kv-capacity 65536 \
  --max-concurrency 2 --kv-dtype fp8 --spec mtp --draft-tokens 3
```

Use `-DNINFER_BUILD_MEDIA=OFF` for a text-only build. This fork's recent workstation qualification
is on Windows; retaining the Linux build path is not a claim that every fork change has been
requalified on Linux. There is no installed C++ SDK or downloadable binary distribution;
the Windows app installer packages a local Release build.

## Measurement and application evaluation

[Performance](docs/performance.md) contains inherited RTX 5090 results and explicitly labeled
Flash-Next RTX PRO 6000 profiling work. Interpret each result with its model, hardware, cache state,
concurrency and speculative configuration. Operator speedups, aggregate decode throughput and
end-to-end request latency answer different questions.

Synthetic and short-source tests protect specific regressions. Production qualification must also
exercise representative complete application workflows and manually review their delivered
answers against the sources. Structured-output conformance is one part of that qualification.

## Documentation

- [Documentation index](docs/README.md), [CLI](docs/cli.md), [HTTP serving](docs/serving.md)
- [Flash-Next model and state semantics](docs/maintainer/qwen3.8-flash-next-model.md)
- [Flash-Next artifact and source provenance](docs/maintainer/qwen3.8-flash-next-artifact.md)
- [Engine architecture](docs/maintainer/engine-architecture.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Performance](docs/performance.md), [perplexity](docs/perplexity.md), [evaluation tooling](eval/README.md)
- [Tests](tests/README.md), [CLI examples](examples/cli/), [contributing](CONTRIBUTING.md)

Run each executable with `--help` for the current option contract.

## Upstream and licensing

The original inference engine and its published Qwen artifacts are the work of
[Neroued/ninfer](https://github.com/Neroued/ninfer) and its contributors. This repository maintains
the workstation and application-serving changes described above; upstream benchmarks and model
cards retain their own provenance.

Engine source is licensed under [Apache-2.0](LICENSE). Model weights are separate artifacts with
their own source-model terms; consult the linked model cards and artifact provenance, particularly
for Flash-Next. Vendored dependencies retain their license files under `third_party/`.
