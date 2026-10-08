"""Frozen first-hardware qualification; never a cross-quant performance verdict."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

from validate_records import read_records as read_rounds
from lifecycle import read_records as read_lifecycle, summarize as summarize_lifecycle
from draft import read_records as read_drafts, summarize as summarize_drafts

PROMPT = ' '.join(f'Record {i}: the solar station stores daylight energy and supplies the village after sunset.'
                  for i in range(32)) + ' Explain its battery storage in detail.'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def api(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base+path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def stop(proc):
    if proc.poll() is not None: return
    os.killpg(proc.pid, signal.SIGTERM)
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); return sock.getsockname()[1]


def strata_config(original, binary, log, level, run_id):
    cfg = json.loads(original.read_text())
    # Preserve model, tokenizer, native flags and installed working directory.
    # Override the engine executable and instrumentation only in a NEW config.
    cfg['exe'] = str(binary.resolve()); cfg['log'] = str(log.resolve())
    cfg['lazy_load'] = False
    cfg['gpu'] = 0
    # The qualification must not train or overwrite the installed profile.
    cfg.pop('expert_profile_save', None); cfg.pop('expert_profile_save_every', None)
    flags = list(cfg.get('args') or [])
    for option in ('--expert-profile-save', '--expert-profile-save-every'):
        while option in flags:
            at = flags.index(option); del flags[at:at+2]
    cfg['args'] = flags
    cfg['env'] = dict(cfg.get('env') or {}, V100_COMPARE_TELEMETRY_LEVEL=str(level),
                      V100_COMPARE_RUN_ID=run_id, CUDA_VISIBLE_DEVICES='0')
    return cfg


def run_cell(a, engine, level):
    run_id = f'qualification-{engine}-l{level}'
    folder = a.output/run_id; folder.mkdir()
    p = free_port(); base = f'http://127.0.0.1:{p}'
    env = os.environ.copy()
    env.update(V100_COMPARE_TELEMETRY_LEVEL=str(level), V100_COMPARE_RUN_ID=run_id,
               CUDA_VISIBLE_DEVICES='0', STRATA_REQUEST_LINES='1')
    if engine == 'ninfer':
        # All non-telemetry flags are identical across levels.
        env.update(NINFER_V100_TELEMETRY='0', NINFER_FLASH_NEXT_STAGE_LEDGER='0',
                   NINFER_V100_EXPERT_PROFILE=str(a.profile.resolve()), NINFER_V100_EXPERT_POLICY='static',
                   NINFER_FLASH_NEXT_EXPERT_CACHE='1', NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS='156',
                   NINFER_V100_PLE_IO='mmap')
        cmd = [str(a.ninfer.resolve()), str(a.artifact.absolute()), '--host', '127.0.0.1', '--port', str(p),
               '--max-context', '4096', '--kv-capacity', '4096', '--max-concurrency', '1',
               '--prefill-chunk', '2048', '--kv-dtype', 'fp8', '--no-prefix-reuse', '--no-thinking',
               '--spec', 'mtp', '--draft-tokens', '1', '--lm-head-draft',
               '--request-log-jsonl', str((folder/'native-requests.jsonl').resolve())]
        native_log = folder/'server.log'
    else:
        native_log = folder/'native-engine.log'
        cfg = strata_config(a.strata_config, a.strata_binary, native_log, level, run_id)
        # Disable conversation reuse for fresh repeated prompts, preserving all
        # other installed native arguments. This is fixed across levels.
        flags = cfg.get('args')
        if not isinstance(flags, list): raise ValueError('Strata config args must be an explicit list')
        if '--prompt-cache' in flags:
            index = flags.index('--prompt-cache')
            flags[index+1] = '0'
        else: flags.extend(['--prompt-cache', '0'])
        if cfg.get('cwd') and not Path(cfg['cwd']).is_absolute():
            raise ValueError('Strata cwd must be absolute to preserve installed asset resolution')
        config = folder/'config.json'; config.write_text(json.dumps(cfg, indent=2)+'\n')
        # Preserve the venv interpreter path: resolving its symlink selects the
        # base Python and loses the installed tokenizer/server dependencies.
        cmd = [str(a.strata_python.absolute()), str(a.strata_server.resolve()), '--engine', 'strata',
               '--config', str(config.resolve()), '--host', '127.0.0.1', '--port', str(p)]
    (folder/'launch.json').write_text(json.dumps(dict(command=cmd,
        environment={k: v for k, v in env.items() if k.startswith(('NINFER_', 'V100_', 'CUDA_', 'STRATA_'))}), indent=2)+'\n')
    sampler = None
    rows = []
    with (folder/'server.log').open('w') as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic()+1200
            while True:
                if proc.poll() is not None: raise RuntimeError(f'{run_id}: exited before ready; see {folder}')
                try:
                    models = api(base, '/v1/models')
                    model = models['data'][0]['id']; break
                except (OSError, ValueError, KeyError, IndexError):
                    if time.monotonic() > deadline: raise TimeoutError(f'{run_id}: loading timed out')
                    time.sleep(1)
            api(base, '/v1/models')  # availability check only; never counts as a request sample
            sampler = subprocess.Popen([sys.executable, str(Path(__file__).with_name('sample_system.py')),
                '--pid', str(proc.pid), '--engine', engine, '--run-id', run_id, '--level', str(level),
                '--gpu-uuid', a.gpu_uuid, '--interval-ms', '200', '--output', str(folder/'system.jsonl')])
            with (folder/'requests.jsonl').open('w') as output:
                for index in range(a.warmups+a.repeats):
                    payload = dict(model=model, messages=[dict(role='user', content=PROMPT)],
                        max_tokens=64, temperature=0, top_p=1, seed=42, enable_thinking=False)
                    begin_offset = native_log.stat().st_size if native_log.exists() else 0
                    start = time.monotonic_ns()
                    response = api(base, '/v1/chat/completions', payload)
                    end = time.monotonic_ns()
                    choice = response['choices'][0]
                    if choice.get('finish_reason') not in ('length', 'stop'):
                        raise ValueError(f'{run_id}: invalid finish reason')
                    message = choice['message']
                    content = message.get('content') or ''
                    reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
                    if not content and not reasoning: raise ValueError(f'{run_id}: empty response')
                    identity = dict(content=content, reasoning=reasoning, tool_calls=message.get('tool_calls'))
                    row = dict(schema='ninfer-strata-v100-qualification-request-v1', engine=engine,
                        run_id=run_id, level=level, observer_request_id=f'{run_id}:{index}',
                        attribution='single_inflight_http_observer', native_request_id=None,
                        phase='warmup' if index < a.warmups else 'measured',
                        started_monotonic_ns=start, completed_monotonic_ns=end, wall_us=(end-start)/1000,
                        request_payload=payload, response=response, output_content_bytes=len(content.encode()),
                        output_reasoning_bytes=len(reasoning.encode()),
                        output_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                        native_log_start_offset=begin_offset,
                        native_log_end_offset=native_log.stat().st_size if native_log.exists() else 0)
                    output.write(json.dumps(row, allow_nan=False)+'\n'); output.flush(); rows.append(row)
        finally:
            stop(proc)
            if sampler:
                sampler.terminate(); sampler.wait(timeout=10)
    text = native_log.read_text() if native_log.exists() else ''
    rounds = read_rounds(native_log) if 'ninfer-strata-v100-telemetry-v1' in text else []
    lifecycle = read_lifecycle(native_log) if native_log.exists() else []
    drafts = read_drafts(native_log) if native_log.exists() else []
    if level == 0 and (rounds or lifecycle or drafts): raise ValueError(f'{run_id}: level 0 emitted telemetry')
    if level and not rounds: raise ValueError(f'{run_id}: no native round telemetry')
    if any(r['status'] != 'ok' for r in rounds): raise ValueError(f'{run_id}: failed execution telemetry')
    if lifecycle:
        (folder/'lifecycle-summary.json').write_text(json.dumps(summarize_lifecycle(lifecycle, rounds), indent=2)+'\n')
    if drafts:
        (folder/'draft-summary.json').write_text(json.dumps(summarize_drafts(drafts, lifecycle), indent=2)+'\n')
    return rows


def report(rows):
    results = []
    fidelity = True
    for engine in ('ninfer', 'strata'):
        cells = {level: [r for r in rows if r['engine'] == engine and r['level'] == level and r['phase'] == 'measured']
                 for level in (0, 1, 2)}
        hashes = {level: {r['output_sha256'] for r in cell} for level, cell in cells.items()}
        reference = hashes[0]
        stable = len(reference) == 1
        equivalent = stable and all(hashes[level] == reference for level in (1, 2))
        fidelity &= equivalent
        medians = {level: statistics.median(r['wall_us'] for r in cell) for level, cell in cells.items()}
        spread = {level: (max(r['wall_us'] for r in cell)-min(r['wall_us'] for r in cell))/medians[level]
                  for level, cell in cells.items()}
        overhead = medians[1]/medians[0]-1
        # A baseline can be nondeterministic *before* telemetry is enabled.
        # Never call that telemetry-induced drift or silently accept it as fidelity.
        novel = {level: len(hashes[level] - reference) for level in (1, 2)}
        failure_reason = (None if equivalent else
                          'baseline_nondeterministic' if not stable else
                          'telemetry_level_output_mismatch')
        traces = {level: [r['wall_us'] for r in cell] for level, cell in cells.items()}
        trend = {level: cell[-1]['wall_us'] / cell[0]['wall_us'] - 1
                 for level, cell in cells.items()}
        drift = any(abs(value) > .05 for value in trend.values())
        results.append(dict(engine=engine, baseline_output_stable=stable,
            observed_http_outputs_equal_across_levels=equivalent,
            output_fidelity_failure_reason=failure_reason,
            unique_output_hashes_per_level={level: len(values) for level, values in hashes.items()},
            new_output_hashes_vs_baseline=novel,
            median_http_wall_us=medians,
            measured_http_wall_us=traces,
            first_to_last_measured_wall_change_fraction=trend,
            measured_request_drift_exceeds_5pct=drift,
            relative_sample_range=spread, level_1_overhead_fraction=overhead,
            overhead_qualified=(overhead < .02 and all(value <= .05 for value in spread.values())
                                and not drift)))
    return dict(schema='ninfer-strata-v100-first-qualification-v1', software_output_fidelity_passed=fidelity,
        comprehensive_attribution_ready=False, engines=results,
        limitations=['Output fidelity is exact HTTP content/reasoning/tool payload equality within each engine, not cross-quant equality.',
                     'This is fresh-prompt HTTP wall overhead, not isolated decode throughput or teacher-forced equality.',
                     'Baseline nondeterminism and cache-warmup drift are reported separately; neither is a fidelity pass or an isolated overhead estimate.',
                     'System sampler observes the server PID; Strata native child CPU/memory sampling is not yet joined.',
                     'Observer log offsets bracket requests; asynchronous log flushing may cross a boundary.',
                     'Native full request IDs, teacher-forced input traces, GPU intervals/launches and complete MTP costs remain coverage gaps.',
                     'No optimization is qualified by this report.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ninfer', 'artifact', 'profile', 'strata-python', 'strata-server', 'strata-config', 'strata-binary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    a = parser.parse_args()
    if a.warmups < 1 or a.repeats < 3: parser.error('at least one warmup and three measured repeats required')
    for key, value in vars(a).items():
        if isinstance(value, Path) and key != 'output' and not value.exists(): parser.error(f'missing {key}: {value}')
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema='ninfer-strata-v100-first-qualification-manifest-v1', prompt=PROMPT,
        warmups=a.warmups, repeats=a.repeats, levels=[0,1,2], mode='native_mtp_single_inflight',
        args={key: str(value) for key, value in vars(a).items()},
        ninfer_executable_sha256=digest(a.ninfer), strata_executable_sha256=digest(a.strata_binary),
        strata_config_sha256=digest(a.strata_config), resident_profile_sha256=digest(a.profile),
        model_identity='workflow records artifact identity; no cross-quant numerical comparison')
    manifest_path = a.output/'manifest.json'; manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    (a.output/'manifest.sha256').write_text(digest(manifest_path)+'\n')
    rows = []
    for level in (0, 1, 2):
        # Build once, alternate engines and release only our own process group.
        for engine in (('ninfer','strata') if level != 1 else ('strata','ninfer')):
            rows.extend(run_cell(a, engine, level))
    summary = report(rows)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))
    if not summary['software_output_fidelity_passed']: raise SystemExit('telemetry off/on output fidelity failed')


if __name__ == '__main__': main()
