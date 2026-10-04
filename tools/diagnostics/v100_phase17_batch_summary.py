"""Summarize Phase 17 paired batching evidence; never auto-promote a runtime policy."""
import json
import math
from pathlib import Path
import statistics
import sys


def analyze(path, expected_tokens, expected_samples, require_eviction):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith('{'):
            row = json.loads(line)
            if row.get('phase17', '').startswith('batch_'):
                rows.append(row)
    results = []
    for stage in ('cold_active', 'warm_active', 'warm_fixed'):
        checks = [r for r in rows if r['phase17'] == 'batch_qualification_correctness'
                  and r['stage'] == stage]
        if len(checks) != 1 or not checks[0]['exact_logits_match'] or not checks[0]['finite_logits']:
            raise ValueError(f'{path.name}/{stage}: missing exact/finite correctness evidence')
        if checks[0]['tokens'] != expected_tokens or checks[0]['committed_frontier'] != expected_tokens:
            raise ValueError(f'{path.name}/{stage}: wrong correctness prefix or state frontier')
        data = [r for r in rows if r['phase17'] == 'batch_qualification' and r['stage'] == stage]
        if len(data) != 2*expected_samples:
            raise ValueError(f'{path.name}/{stage}: incomplete paired observations')
        paired_gains = []
        for sample in range(expected_samples):
            pair = [r for r in data if r['sample'] == sample]
            if len(pair) != 2 or {r['mode'] for r in pair} != {'scalar', 'batched'}:
                raise ValueError(f'{path.name}/{stage}: invalid sample {sample}')
            for r in pair:
                expected_order = (int(r['mode'] == 'batched') if sample % 2 == 0
                                  else int(r['mode'] == 'scalar'))
                if r['order'] != expected_order or r['tokens'] != expected_tokens:
                    raise ValueError('benchmark order or prefix mismatch')
                if not math.isfinite(r['seconds']) or r['seconds'] <= 0:
                    raise ValueError('invalid timing')
                if not math.isclose(r['tokens_per_s'], expected_tokens/r['seconds'], rel_tol=1e-8):
                    raise ValueError('rate inconsistent with measured wall')
                if not r['hits'] or not r['misses'] or r['maximum_outstanding'] > 4:
                    raise ValueError('hybrid branch/queue evidence missing')
                if stage == 'warm_fixed':
                    if r['admitted'] or r['evicted']:
                        raise ValueError('frozen residency changed')
                elif not r['admitted'] or (require_eviction and not r['evicted']):
                    raise ValueError('active fills/required eviction not exercised')
            scalar = next(r for r in pair if r['mode'] == 'scalar')
            batched = next(r for r in pair if r['mode'] == 'batched')
            paired_gains.append(100*(batched['tokens_per_s']/scalar['tokens_per_s']-1))
        result = dict(workload=path.stem, stage=stage, tokens=expected_tokens,
                      samples_per_mode=expected_samples,
                      paired_gain_percent_mean=statistics.mean(paired_gains),
                      paired_gain_percent_min=min(paired_gains),
                      paired_gain_percent_max=max(paired_gains))
        for mode in ('scalar', 'batched'):
            selected = [r for r in data if r['mode'] == mode]
            hits = sum(r['hits'] for r in selected)
            misses = sum(r['misses'] for r in selected)
            result[mode] = dict(
                tokens_per_s_mean=statistics.mean(r['tokens_per_s'] for r in selected),
                tokens_per_s_with_drain_mean=statistics.mean(r['tokens_per_s_with_drain'] for r in selected),
                hit_coverage=hits/(hits+misses),
                admissions=sum(r['admitted'] for r in selected),
                evictions=sum(r['evicted'] for r in selected),
                fill_bytes=sum(r['fill_bytes'] for r in selected),
                hit_launches_per_token=sum(r['hit_kernel_launches'] for r in selected)
                    /(expected_tokens*expected_samples))
        results.append(result)
    return results


def main():
    root = Path(sys.argv[1])
    results = analyze(root/'derived-capacity.log', 512, 4, False)
    results += analyze(root/'eviction-stress.log', 256, 2, True)
    (root/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    print('| Cache workload | Stage | Scalar tok/s | Batched tok/s | Paired mean gain | Scalar/batched hit coverage |')
    print('|---|---|---:|---:|---:|---:|')
    for r in results:
        s, b = r['scalar'], r['batched']
        print(f"| {r['workload']} | {r['stage']} | {s['tokens_per_s_mean']:.3f} | "
              f"{b['tokens_per_s_mean']:.3f} | {r['paired_gain_percent_mean']:.1f}% | "
              f"{s['hit_coverage']:.1%}/{b['hit_coverage']:.1%} |")
    print('\nExact parity uses a shared frozen Ready set after active cache churn. '
          'Active-placement differences are diagnostics; timings exclude logit downloads. '
          'Cache fill tails are reported separately. These are teacher-forced harness rates. '
          'Production defaults remain unchanged pending evidence review. No Phase 18 work.')


if __name__ == '__main__':
    main()
