"""Summarize a Nsight Systems SQLite export by NInfer generate/prefill/decode ranges.

Assign GPU activities by their start timestamp inside a CPU phase range. Graph
kernel correlation IDs identify the last launch in each decode round as the target;
earlier launches are MTP drafts (the current Flash-Next schedule). Percentages in
the output use summed kernel time; GPU union time avoids double counting overlap.
"""
import bisect
import collections
import json
import sqlite3
import sys
from pathlib import Path

db = Path(sys.argv[1])
c = sqlite3.connect(db)
names = dict(c.execute('select id,value from StringIds'))
ranges = list(c.execute("select n.start,n.end,coalesce(n.text,s.value) from NVTX_EVENTS n left join StringIds s on s.id=n.textId where coalesce(n.text,s.value) in ('generate','prefill','decode') and n.end is not null order by n.start"))
generations = [(a,b) for a,b,n in ranges if n == 'generate']
phases = [(a,b,n) for a,b,n in ranges if n != 'generate']
starts = [a for a,b,n in phases]
launches = collections.defaultdict(list)
for a,correlation in c.execute("select r.start,r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like 'cudaGraphLaunch%' order by r.start"):
    index = bisect.bisect_right(starts,a)-1
    if index >= 0 and a < phases[index][1] and phases[index][2] == 'decode':
        launches[index].append(correlation)
graph_role = {correlation: ('target' if j == len(ids)-1 else 'draft')
              for ids in launches.values() for j,correlation in enumerate(ids)}
result = {}
for i, (gstart,gend) in enumerate(generations):
    phase_rows = [(a,b,n) for a,b,n in phases if a>=gstart and b<=gend]
    for name in ['prefill','decode']:
        spans = [(a,b) for a,b,n in phase_rows if n==name]
        result[f'{i}-{name}'] = {'wall_ms':sum(b-a for a,b in spans)/1e6,'ranges':len(spans),
                               'kernels':collections.defaultdict(lambda:[0,0,0]), 'kernel_spans':[],
                               'memcpy_ms':0,'memcpy_bytes':0, 'api':collections.defaultdict(lambda:[0,0]),
                               'graph_role_kernel_ms':collections.defaultdict(float)}

def bucket(a,b):
    index = bisect.bisect_right(starts,a)-1
    if index<0: return None
    pstart,pend,phase=phases[index]
    if a>=pend: return None
    for i,(gstart,gend) in enumerate(generations):
        if gstart<=a<gend: return result[f'{i}-{phase}']

for a,b,name,gridx,blockx,reg,shared,correlation in c.execute('select start,end,demangledName,gridX,blockX,registersPerThread,dynamicSharedMemory,correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
    r=bucket(a,b)
    if r is None: continue
    k=r['kernels'][names[name]]
    k[0]+=b-a;k[1]+=1;k[2]=max(k[2],b-a)
    r['kernel_spans'].append((a,b))
    r['graph_role_kernel_ms'][graph_role.get(correlation,'eager')]+=(b-a)/1e6
for a,b,size in c.execute('select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY'):
    r=bucket(a,b)
    if r is not None: r['memcpy_ms']+=(b-a)/1e6; r['memcpy_bytes']+=size
for a,b,name in c.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
    r=bucket(a,b)
    if r is not None: r['api'][names[name]][0]+=(b-a)/1e6; r['api'][names[name]][1]+=1
for key,r in result.items():
    union=0; end=0
    for a,b in r.pop('kernel_spans'):
        union+=max(0,b-max(a,end));end=max(end,b)
    r['gpu_kernel_union_ms']=union/1e6
    total=sum(v[0] for v in r['kernels'].values())
    r['kernel_total_ms']=total/1e6
    r['kernel_count']=sum(v[1] for v in r['kernels'].values())
    r['kernels']=[{'name':n,'ms':v[0]/1e6,'percent':100*v[0]/total,'count':v[1], 'mean_us':v[0]/v[1]/1e3} for n,v in sorted(r['kernels'].items(),key=lambda x:-x[1][0])]
    r['api']=sorted(r['api'].items(),key=lambda x:-x[1][0])[:8]
db.with_suffix('.analysis.json').write_text(json.dumps(result,indent=2))
if any(key.endswith('-decode') and r['ranges'] and not r['kernel_count']
       for key,r in result.items()):
    raise SystemExit('Decode graph kernel nodes are missing: capture from process start with --trace=cuda-sw,nvtx --cuda-graph-trace=node')
for key,r in result.items():
    print(key,json.dumps({k:v for k,v in r.items() if k!='kernels'}))
    for kernel in r['kernels'][:10]: print(json.dumps(kernel))
