"""Bounded same-input CPU/stream comparison for the failing production prompt."""
import argparse
import json
import math
import os
from pathlib import Path

from v100_expert_profile import atomic_json
from v100_sv2_serve import run_server


def summarize(text):
    records = [json.loads(line.split('v100.stream_compare=', 1)[1])
               for line in text.splitlines() if line.startswith('v100.stream_compare=')]
    if len(records) != 48*8:
        raise ValueError('need eight real nonresident samples from each of 48 layers')
    keys = {(r['layer'], r['sample']) for r in records}
    if keys != {(layer, sample) for layer in range(48) for sample in range(8)}:
        raise ValueError('missing or duplicate layer/sample records')
    tokens = {r['tokens'] for r in records}
    if len(tokens) != 1 or not 1024 <= next(iter(tokens)) <= 2048:
        raise ValueError('samples do not share the single large production prefill')
    metrics = ('cpu_reference_nrmse', 'cpu_reference_cosine', 'maximum_error',
               'avx2_reference_nrmse','gpu_avx2_nrmse')
    if any(r['nonfinite'] or any(not math.isfinite(r[k]) for k in metrics) for r in records):
        raise ValueError('nonfinite same-input route comparison')
    if any(r['isolated_replay_differences'] for r in records):
        raise ValueError('grouped/ring production output differs from isolated same-input GPU replay')
    if any(not r['avx2_available'] for r in records):
        raise ValueError('V100 host did not exercise the production AVX2 control')
    return dict(schema=1, scope='sampled_real_production_inputs', qualified=False,
                prefill_tokens=next(iter(tokens)), samples=len(records),
                isolated_gpu_replay_exact=True,
                cpu_reference_nrmse_max=max(r['cpu_reference_nrmse'] for r in records),
                cpu_reference_cosine_min=min(r['cpu_reference_cosine'] for r in records),
                avx2_reference_nrmse_max=max(r['avx2_reference_nrmse'] for r in records),
                gpu_avx2_nrmse_max=max(r['gpu_avx2_nrmse'] for r in records),
                maximum_error=max(r['maximum_error'] for r in records),
                # Existing focused CPU-profile criterion: report, never relax or
                # substitute it for mathematical/model/production qualification.
                cpu_profile_criterion_failures=sum(r['cpu_reference_nrmse'] > .002 or
                    r['cpu_reference_cosine'] < .99999 for r in records),
                records=records, limitations=[
                    'eight nonresident routes per layer; unsampled corruption is not excluded',
                    'CPU reference retains BF16 activation boundary; this is precision-profile evidence',
                    'isolated replay shares the GPU arithmetic but changes group size and staging lifetime',
                    'diagnostic downloads/replay/CPU work invalidate throughput measurements',
                    'does not qualify production streaming or alter the failed exact response gate'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('executable','artifact','profile','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['NINFER_V100_STREAM_COMPARE']='1'
    observation=run_server(args.executable,args.artifact,args.profile,args.output,
                           'stream',False,0,prefill_screen=True)
    atomic_json(args.output/'observation.json',observation)
    report=summarize((args.output/'stream-mtp0-0.log').read_text())
    atomic_json(args.output/'comparison.json',report)
    summary={k:v for k,v in report.items() if k not in ('records','limitations')}
    (args.output/'report.txt').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
