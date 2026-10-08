# V100 automatic prefill streaming A/B — run 37769284560

Commit 9293cdc169b8341505aaa8167a67737f4cec7a00. Five sequential cases, same resident model and mixed quantization, 5 warmups and 3 measured requests per prompt length, 64-token greedy output.

| Policy | 585-token long HTTP median | 63-token short HTTP median |
|---|---:|---:|
| Static | 13.804 s | 6.414 s |
| LRU | 12.182 s | 4.899 s |
| LRU + always stream | 9.730 s | 5.234 s |
| LRU + auto256 | **9.647 s** | **4.079 s** |

Auto256 long native prefill 6.758 s; short native prefill 0.887 s. Long prefill streamed experts; short prefill used CPU fallback without expert-weight transfers. Repeat static baseline within 2%. Relative to static, auto256 reduced long wall time 30.1% and short wall time 36.4%. These are measurements, not quality qualification. Greedy output hashes can vary and are not by themselves evidence of task failure.

Next gate: compare static, LRU and auto256 on 12 objective short/long tasks, two repetitions each, recording token counts, raw outputs and task-level strict/lenient accuracy; do not change defaults until accuracy is reviewed. Do not modify resident model or installed Strata. Avoid overlapping GPU jobs.
