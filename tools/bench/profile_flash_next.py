"""Root-only Flash-Next requests for phase profiling against an isolated server.

Use a new label per invocation to prevent prefix reuse. Snapshot the same context
file for every run; the model, runtime settings and profiler are managed externally.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
p.add_argument('--output', required=True)
p.add_argument('--tokens', type=int, default=256)
p.add_argument('--repeat', type=int, default=1)
p.add_argument('--url', default='http://127.0.0.1:8011')
p.add_argument('--context-file', type=Path, required=True)
args = p.parse_args()
context = args.context_file.read_text(encoding='utf8')
rows = []
for iteration in range(args.repeat):
    for length in ['short', 'long']:
        label = f'{args.label}/{iteration}/{length}'
        content = label + '\n'
        if length == 'long':
            content += 'Use this engine source as background context:\n' + context + '\n'
        content += 'Write a detailed Python implementation of an LRU cache with expiration, thread safety, and unit tests. Explain each design decision. Continue until the implementation and tests are complete.'
        body = {'model':'qwen3.8-flash-next', 'messages':[{'role':'user','content':content}],
                'temperature':0, 'seed':42, 'presence_penalty':0, 'reasoning_effort':'none',
                'max_tokens':args.tokens, 'stream':False}
        request = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
                  data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        started = time.time()
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        usage = result['usage']
        row = {'label':label, 'start_utc':started, 'elapsed_seconds':time.time()-started, 'usage':usage,
               'finish_reason':result['choices'][0]['finish_reason']}
        rows.append(row)
        print(json.dumps(row), flush=True)
        Path(args.output).write_text(json.dumps(rows, indent=2))
        assert usage['prompt_tokens_details']['cached_tokens'] == 0, 'Root workload unexpectedly reused a prefix'
        assert usage['completion_tokens'] == args.tokens, 'Output stopped before the measured token budget'
