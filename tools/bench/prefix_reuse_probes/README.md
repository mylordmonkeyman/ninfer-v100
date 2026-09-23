# Prefix-reuse probes (client-side acceptance)

All scripts hit `http://localhost:8010/v1/chat/completions` and read the API key from the
`NINFER_KEY` environment variable (`P:/models/ninfer-api-key.txt`). Run from any directory:

    NINFER_KEY=$(tr -d '\r\n' < P:/models/ninfer-api-key.txt) python -u <script>

| Script | Shape | Expected on a healthy engine |
|---|---|---|
| `reuse_probe.py` | 4 turns, 4 tools, thinking off | turns 2-4 cached ~ previous prompt |
| `reuse_probe9.py` | same, 9 tools | same |
| `reuse_probe9_think.py` | same, 9 tools, `enable_thinking: true` | same (dead on the 2026-09-07 02:03 build) |
| `toolloop_probe.py` | tool call, echoed tool_calls + tool result, thinking on | turns 2-4 reuse |
| `acceptance.py` | saturates the pool with 20 conversations, then new-after-saturation, evicted-returns, 3 interleaved newcomers | OVERALL: PASS (PHASE 4 6/6) |
| `depth.py` | A,A,B,A,A on whatever pool state exists (run after acceptance.py for the saturated case) | all three `yes` |
| `interleave_ctl.py` | A,B,A,B vs the same pair sequential | 6/6 and 6/6 |

A fresh pool passes the last three regardless; only the saturated pool discriminates.
Text-only probes cannot see thinking-mode or tool-call render regressions; run the
thinking-on and tool-loop probes before deploying any serve-layer change.

| `thinking_toggle_probe.py` | one conversation, thinking off,off,on,on,off | off->on loses the whole prefix (history re-renders with empty think blocks); on->off keeps all but the tail. Keep `enable_thinking` constant within a conversation. |
