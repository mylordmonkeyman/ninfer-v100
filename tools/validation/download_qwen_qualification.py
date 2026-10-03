"""Fetch one published NInfer test artifact, preserving pinned metadata and checking its hash."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import urllib.request

repo, filename, destination, evidence = sys.argv[1:]
destination, evidence = pathlib.Path(destination), pathlib.Path(evidence)
destination.mkdir(parents=True, exist_ok=True)
evidence.mkdir(parents=True, exist_ok=True)
def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)
info = fetch(f'https://huggingface.co/api/models/{repo}')
revision = info['sha']
base = f'https://huggingface.co/{repo}/resolve/{revision}'
manifest = fetch(base + '/artifact-manifest.json')
artifact = manifest['artifact']
if artifact['filename'] != filename:
    raise SystemExit('Published artifact filename differs from the declared test artifact')
size, expected = artifact['bytes'], artifact['sha256']
final = destination / filename
part = destination / (filename + '.' + expected[:16] + '.part')
metadata = {'repository': repo, 'revision': revision, 'manifest': manifest, 'path': str(final)}
(evidence/'artifact.json').write_text(json.dumps(metadata, indent=2)+'\n')
def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()
if final.exists():
    if final.stat().st_size != size or digest(final) != expected:
        raise SystemExit(f'Existing qualification artifact differs from the published hash: {final}')
    print(f'PASS: existing verified artifact {final}', flush=True)
else:
    partial = part.stat().st_size if part.exists() else 0
    if partial > size:
        raise SystemExit('Partial download exceeds the published artifact size')
    free = shutil.disk_usage(destination).free
    if free < size - partial + (5 << 30):
        raise SystemExit(f'Insufficient disk space: free={free} needed={size-partial+(5<<30)}')
    print(f'Download {repo}@{revision}: {size} bytes; free={free}', flush=True)
    subprocess.run(['curl','--fail','--location','--retry','3','--continue-at','-',
                    '--output',str(part),base+'/'+filename+'?download=true'], check=True)
    if part.stat().st_size != size or digest(part) != expected:
        raise SystemExit('Downloaded model size/hash differs from the published manifest')
    part.rename(final)
    print(f'PASS: published artifact size and SHA256 {final}', flush=True)
