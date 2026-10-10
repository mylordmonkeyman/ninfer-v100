"""Protect observed phase accounting and metadata-only inventory."""
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ab_v100
from structural_v100 import inventory, native_work, summarize, residency_arms, residency_observation, prefill_breakdown
from test_schema import fixture


class StructuralTests(unittest.TestCase):
    def test_prefill_breakdown_separates_request_windows_and_verify(self):
        import gzip
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = b'initialization noise\n'
            requests = []
            for index, duration in enumerate((1000, 2000)):
                begin = len(data)
                def chunk(phase, ms):
                    return dict(kind='prefill_stage_ledger', tokens=10,
                        context=dict(phase=phase), total_chunk_ms=ms, residual_ms=1,
                        stages=[dict(stage='PLE injection', interval_ms=ms-1)])
                data += (json.dumps(chunk('prefill', duration))+'\n').encode()
                data += (json.dumps(chunk('verify', 999999))+'\n').encode()
                requests.append(dict(index=index, phase='measured',
                    native_log_start_offset=begin, native_log_end_offset=len(data),
                    response=dict(usage=dict(prompt_tokens=10)), native_timing=dict(prefill=3),
                    wall_seconds=4, output_sha256='fixture', native_speculative={}))
            with gzip.open(folder/'server.log.gz','wb') as out:
                out.write(data)
            (folder/'requests.jsonl').write_text('\n'.join(map(json.dumps, requests)))
            rows = prefill_breakdown(folder)
            self.assertEqual([r['ledger_seconds'] for r in rows], [1, 2])
            self.assertEqual([r['native_minus_ledger_seconds'] for r in rows], [2, 1])
            self.assertAlmostEqual(rows[1]['categories_seconds']['ple_inclusive'], 1.999)
            requests[1]['response']['usage']['prompt_tokens'] = 11
            (folder/'requests.jsonl').write_text('\n'.join(map(json.dumps, requests)))
            with self.assertRaisesRegex(ValueError, 'complete request prompt'):
                prefill_breakdown(folder)

    def test_residency_comparison_preserves_gemm_selectors_and_validates_observed_work(self):
        arms = residency_arms()
        self.assertEqual([arm for _, arm, _ in arms], ['static64', 'adaptive156'])
        for _, _, flags in arms:
            self.assertEqual(flags['NINFER_V100_PREFILL_EXPERT_GEMM'], 'fp16')
            self.assertEqual(flags['NINFER_V100_PREFILL_RESIDENT_GEMM'], '1')
            self.assertEqual(flags['NINFER_V100_PREFILL_CPU_STREAM_OVERLAP'], '1')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row = dict(response=dict(usage=dict(prompt_tokens=7111, completion_tokens=128,
                prompt_tokens_details=dict(cached_tokens=0)), choices=[dict(finish_reason='length')]),
                native_timing=dict(prefill=32.6, decode=7.9))
            requests = folder/'requests.jsonl'
            requests.write_text(json.dumps(row)+'\n')
            (folder/'server.log').write_text('phase13.cache.slots_per_layer=64\n'
                'phase13.cache.bytes=8494252032\nv100.profile.seeded=3072\n')
            (folder/'system.jsonl').write_text(json.dumps(dict(kind='system_sample',
                gpu=dict(metrics=dict(memory_used_bytes=19_000_000_000))))+'\n')
            observed = residency_observation(folder, 128)
            self.assertEqual(observed['actual_cache']['slots_per_layer'], 64)
            self.assertEqual(observed['sampled_peak_gpu_bytes'], 19_000_000_000)
            row['response']['usage']['prompt_tokens_details']['cached_tokens'] = 7111
            requests.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'fixed frontend workload'):
                residency_observation(folder, 128)
            row['response']['usage']['prompt_tokens_details']['cached_tokens'] = 0
            requests.write_text(json.dumps(row)+'\n')
            (folder/'server.log').write_text('phase13.cache.slots_per_layer=64\n'
                'phase13.cache.bytes=8494252032\nv100.profile.seeded=3060\n')
            with self.assertRaisesRegex(ValueError, 'actual cache capacity'):
                residency_observation(folder, 128)

    def test_directory_and_native_format_inventory_without_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pack = root/'pack'; pack.mkdir()
            (pack/'native_experts.txt').write_text('# native v4\n0 12 7 0 3000 100 200 300 shard.gguf\n')
            cfg = root/'config.json'; cfg.write_text(json.dumps(dict(args=['--pack', str(pack)])))
            model = root/'model.ninfer'
            directory = dict(identity=dict(model_id='fixture', weights_id='q4'), objects=[
                dict(name='expert', kind='tensor', format='nvfp4', bytes=1234)])
            data = json.dumps(directory).encode()
            model.write_bytes(struct.pack('<8sQ', b'NINFER\x00\x02', len(data))+data)
            a = SimpleNamespace(artifact=model, strata_config=cfg, strata_server=root/'server.py', output=root)
            inventory(a)
            r = json.loads((root/'representation.json').read_text())
            self.assertEqual(r['ninfer']['bytes_by_format'], {'nvfp4':1234})
            self.assertEqual(r['strata']['native_rows'][0]['compact_expert_bytes'], 3000)
            self.assertEqual(r['strata']['native_rows'][0]['down_type'], 7)

    def test_native_intervals_stay_separate_and_cache_occupancy_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            record = fixture(); record['layers'][0]['host_us'] = {'moe':10, 'cpu_expert':8}
            (folder/'server.log').write_text(json.dumps(record)+'\n'+json.dumps(dict(
                kind='prefill_stage_ledger', tokens=2048, total_chunk_ms=12, stages=[]))+'\n'+json.dumps(dict(
                kind='expert_stream_timing', experts=2, routes=37, copy_ms=1, compute_wait_ms=2, kernel_interval_ms=5))+'\n'+json.dumps(dict(
                kind='expert_resident_gemm', experts=2, routes=43))+'\n'+json.dumps(dict(
                kind='early_cpu_stream', eligible=True, cpu_routes=7, cpu_ms=5,
                stream_submit_ms=8, overlap_ms=4))+'\n'+json.dumps(dict(
                kind='early_cpu_stream', eligible=False, cpu_routes=3, cpu_ms=2,
                stream_submit_ms=0, overlap_ms=0))+'\n')
            r = native_work(folder, 'ninfer')
            self.assertEqual(r['phases'][record['phase']]['host_spans_us'], {'moe':10, 'cpu_expert':8})
            self.assertEqual(r['ninfer_prefill_chunks'][0]['tokens'], 2048)
            self.assertEqual(r['expert_stream_timing'][0]['routes'], 37)
            self.assertEqual(r['expert_resident_gemm'][0], dict(kind='expert_resident_gemm', experts=2, routes=43))
            self.assertEqual(r['early_cpu_stream'][0]['overlap_ms'],4)
            self.assertTrue(r['early_cpu_stream'][0]['eligible'])
            self.assertEqual(r['early_cpu_stream'][1]['cpu_routes'],3)
            self.assertFalse(r['early_cpu_stream'][1]['eligible'])
            self.assertEqual(r['early_cpu_stream'][1]['overlap_ms'],0)
            self.assertEqual(r['expert_stream_timing'][0]['kernel_interval_ms'], 5)

    def test_real_http_schedule_and_native_phase_timing_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); engine = root/'engine'
            engine.write_text('''#!/usr/bin/env python3
import json,os,sys
from http.server import BaseHTTPRequestHandler,HTTPServer
def arg(k):return sys.argv[sys.argv.index(k)+1]
class H(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def reply(self,data):
  body=json.dumps(data).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def do_GET(self):self.reply({'data':[{'id':'fixture'}]})
 def do_POST(self):
  request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  print(json.dumps({'observed_level':os.environ['V100_COMPARE_TELEMETRY_LEVEL'],'prompt':request['messages']}),flush=True)
  with open(arg('--request-log-jsonl'),'a') as f:f.write(json.dumps({'event':'request_done','timings_seconds':{'prefill':1,'decode':2},'speculative':{'accepted_tokens':3}})+'\\n')
  self.reply({'choices':[{'finish_reason':'length','message':{'content':'fixture'}}],'usage':{'prompt_tokens':10,'completion_tokens':4}})
HTTPServer(('127.0.0.1',int(arg('--port'))),H).serve_forever()
''')
            engine.chmod(0o755)
            a = SimpleNamespace(output=root, ninfer=engine, artifact=root/'model.ninfer', profile=root/'profile',
                gpu_uuid='fake', telemetry_level=0, ninfer_flags=[], launch_prefix=[], long_prompt='fixed long',
                request_schedule=[('cold','long'),('measured','long')], warmups=0, repeats=1,max_output_tokens=4)
            real_popen = subprocess.Popen
            def launch(cmd, **kwargs):
                if any(str(x).endswith('sample_system.py') for x in cmd): return real_popen(['true'])
                return real_popen(cmd, **kwargs)
            with patch('ab_v100.subprocess.Popen', side_effect=launch):
                ab_v100.run_case(a,'ninfer','fixture',{})
            folder = root/'ninfer-fixture'
            rows = [json.loads(l) for l in (folder/'requests.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows], ['cold','measured'])
            self.assertEqual(rows[0]['native_timing'], {'prefill':1,'decode':2})
            self.assertEqual(rows[0]['native_speculative']['accepted_tokens'],3)
            self.assertIn('"observed_level": "0"', (folder/'server.log').read_text())
            cells=[dict(mode='timing',engine='ninfer',arm='adaptive156',folder=str(folder))]
            s=summarize(cells)
            self.assertEqual(s['ninfer/adaptive156/cold']['samples'][0]['native_timing']['decode'],2)


if __name__ == '__main__': unittest.main()
