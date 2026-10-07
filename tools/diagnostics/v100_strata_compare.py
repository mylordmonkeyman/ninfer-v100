#!/usr/bin/env python3
import argparse,hashlib,json,math,os,re,signal,socket,statistics,struct,subprocess,threading,time,urllib.request
from pathlib import Path

PROMPT=' '.join(f'Record {i}: the solar station stores energy during daylight and supplies the village after sunset.' for i in range(64))+' Explain how its battery storage works in detail.'
STRATA_RE=re.compile(r'\[strata\] request prompt (\d+) cached (\d+) output (\d+) prompt_read ([\d.]+) ms total ([\d.]+) ms prefill ([\d.]+) tok/s decode ([\d.]+) tok/s')
STRATA_DRAFT_RE=re.compile(r'drafts accepted (\\d+) of (\\d+)')
STRATA_SUFFIX_RE=re.compile(r'suffix drafts: \\d+ windows, (\\d+) of (\\d+) drafts accepted')

def strata_profile_to_ninfer(strata_path,template_path,output_path):
    blob=strata_path.read_bytes()
    if len(blob)<24 or blob[:4]!=b'STRP':
        raise ValueError(f'{strata_path}: invalid Strata expert profile')
    version,layers,experts,slots,count=struct.unpack_from('<5I',blob,4)
    if (version,layers,experts)!=(1,48,512) or count>48*512 or len(blob)<24+4*count:
        raise ValueError(f'{strata_path}: unsupported Strata expert profile geometry')
    ranking=[[] for _ in range(48)];seen=[set() for _ in range(48)]
    for i in range(count):
        layer,expert=struct.unpack_from('<HH',blob,24+4*i)
        if layer>=48 or expert>=512 or expert in seen[layer]:
            raise ValueError(f'{strata_path}: invalid or duplicate ranked expert pair')
        seen[layer].add(expert);ranking[layer].append(expert)
    for layer in range(48):
        ranking[layer].extend(expert for expert in range(512) if expert not in seen[layer])
        if len(ranking[layer])!=512:
            raise ValueError(f'{strata_path}: incomplete converted layer ranking')
    profile=json.loads(template_path.read_text())
    if profile.get('magic')!='NINFER_V100_EXPERT_PROFILE' or profile.get('version')!=2:
        raise ValueError(f'{template_path}: invalid NInfer profile identity template')
    profile['ranking']=ranking
    profile['source']='strata_profile_transfer'
    profile['strata_profile_sha256']=hashlib.sha256(blob).hexdigest()
    profile['strata_profile_ranked_pairs']=count
    profile['strata_profile_declared_slots']=slots
    for key in ('frequency','training_trace_sha256','evaluation_trace_sha256'):
        profile.pop(key,None)
    output_path.write_text(json.dumps(profile,indent=2)+'\n')
    return output_path

def port():
    s=socket.socket();s.bind(('127.0.0.1',0));p=s.getsockname()[1];s.close();return p

def request(base,path,body=None,timeout=900):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)

def wait_ready(proc,base,log):
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        if proc.poll() is not None: raise RuntimeError(f'process exited {proc.returncode}; see {log}')
        try:
            models=request(base,'/v1/models',timeout=5)
            if models.get('data'): return models['data'][0]['id']
        except Exception: pass
        time.sleep(1)
    raise TimeoutError(f'readiness timeout; see {log}')

def stop(proc):
    if proc.poll() is not None:return
    proc.terminate()
    try:proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill();proc.wait(10)

def monitor(out,stop_event):
    fields='timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem'
    with out.open('w') as f:
        while not stop_event.is_set():
            subprocess.run(['nvidia-smi','--id=0',f'--query-gpu={fields}','--format=csv,noheader,nounits'],stdout=f,stderr=subprocess.DEVNULL,text=True)
            f.flush();stop_event.wait(1)

def common_payload(model,run):
    messages=[{'role':'user','content':f'Benchmark run {run}. '+PROMPT}]
    return dict(model=model,messages=messages,max_tokens=128,temperature=0,seed=42,enable_thinking=False)

def run_ninfer(a,out,run,draft_tokens):
    mode=f'd{draft_tokens}'; p=port();base=f'http://127.0.0.1:{p}'
    log=out/f'ninfer-{mode}-{run}.log'; reqlog=out/f'ninfer-{mode}-{run}-requests.jsonl'
    env=os.environ.copy(); env.update({
      'NINFER_V100_EXPERT_PROFILE':str(a.profile.resolve()),'NINFER_V100_EXPERT_POLICY':'decay',
      'NINFER_V100_EXPERT_DECAY':'0.9','NINFER_V100_EXPERT_DECAY_INTERVAL':'32',
      'NINFER_FLASH_NEXT_EXPERT_CACHE':'1','NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS':'156',
      'NINFER_FLASH_NEXT_EXPERT_CACHE_SERIAL':'0','NINFER_FLASH_NEXT_EXPERT_CACHE_PREFILL':'1',
      'NINFER_FLASH_NEXT_EXPERT_CACHE_BATCHED_DECODE':'1','NINFER_FLASH_NEXT_EXPERT_CACHE_GROUPED_PREFILL':'1',
      'NINFER_V100_DEVICE_ROUTE_COMBINE':'1','NINFER_V100_PREFILL_EXPERT_POLICY':'cpu-cache',
      'NINFER_V100_CPU_EXPERT_GROUP':'1','NINFER_V100_ROUTE_HANDOFF':'0','NINFER_V100_PLE_IO':'mmap',
      'NINFER_V100_QSA_SCORE_MMA':'0','NINFER_FLASH_NEXT_QSA_PREFILL_MMA':'0',
      'NINFER_V100_TELEMETRY':'0','NINFER_FLASH_NEXT_STAGE_LEDGER':'0',
      'NINFER_FLASH_NEXT_CPU_EXPERT_FP32_INTERMEDIATE':'0','NINFER_FLASH_NEXT_MOE_SHARED_FP32_INTERMEDIATE':'0'})
    cmd=[str(a.ninfer.resolve()),str(a.artifact.absolute()),'--host','127.0.0.1','--port',str(p),
         '--max-context','4096','--kv-capacity','4096','--max-concurrency','1','--prefill-chunk','2048',
         '--kv-dtype','fp8','--device-state-slots','2','--host-state-slots','2','--host-kv-mib','256',
         '--max-private-continuations','2','--max-shared-prefixes','2','--no-cuda-graph','--no-qsa-prefill-mma',
         '--no-thinking','--request-log-jsonl',str(reqlog)]
    if draft_tokens:
        cmd += ['--spec','mtp','--draft-tokens',str(draft_tokens),'--lm-head-draft']
    monstop=threading.Event(); t=threading.Thread(target=monitor,args=(out/f'ninfer-{mode}-{run}-gpu.csv',monstop));t.start()
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            model=wait_ready(proc,base,log); start=time.monotonic()
            resp=request(base,'/v1/chat/completions',common_payload(model,run)); wall=time.monotonic()-start
        finally: stop(proc);monstop.set();t.join()
    events=[json.loads(x) for x in reqlog.read_text().splitlines() if x.strip()]
    done=[x for x in events if x.get('event')=='request_done'][-1]
    r=done['result']; tm=done['timings_seconds']; fresh=r['prompt_tokens']-r['prefix_cache_hit_tokens']
    spec=done.get('speculative') or {}; dec=done.get('engine_timing',{}).get('decode',{})
    message=((resp.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
    return dict(engine=f'ninfer-d{draft_tokens}',run=run,draft_window=draft_tokens,
        prompt_tokens=r['prompt_tokens'],cached=r['prefix_cache_hit_tokens'],fresh=fresh,
        output=r['completion_tokens'],prefill_tps=fresh/tm['prefill'],
        decode_tps=r['completion_tokens']/tm['decode'],ttft_s=tm['ttft'],
        prefill_s=tm['prefill'],decode_s=tm['decode'],wall_s=wall,
        drafted=spec.get('drafted_tokens',0),accepted=spec.get('accepted_tokens',0),
        accepted_per_position=spec.get('accepted_per_position',[]),
        speculative_rounds=spec.get('rounds',0),target_rounds=dec.get('rounds'),
        host_exposed_s=dec.get('host_exposed_seconds'),
        device_wait_exposed_s=dec.get('device_wait_exposed_seconds'),
        response_sha256=hashlib.sha256(message.encode()).hexdigest())

def run_strata(a,out,run):
    p=port();base=f'http://127.0.0.1:{p}';log=out/f'strata-{run}.log'
    env=os.environ.copy();env['STRATA_REQUEST_LINES']='1';env['CUDA_VISIBLE_DEVICES']='0'
    cfg=json.loads(a.strata_config.read_text())
    cfg['log']=str((out/f'strata-engine-{run}.log').resolve())
    run_cfg=out/f'strata-config-{run}.json';run_cfg.write_text(json.dumps(cfg,indent=2))
    cmd=[str(a.strata_python),str(a.strata_server),'--engine','strata','--config',str(run_cfg),
         '--host','127.0.0.1','--port',str(p)]
    monstop=threading.Event();t=threading.Thread(target=monitor,args=(out/f'strata-{run}-gpu.csv',monstop));t.start()
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            model=wait_ready(proc,base,log);start=time.monotonic()
            resp=request(base,'/v1/chat/completions',common_payload(model,run));wall=time.monotonic()-start
        finally:stop(proc);monstop.set();t.join()
    text=log.read_text(errors='replace');matches=list(STRATA_RE.finditer(text))
    if not matches: raise RuntimeError(f'no Strata request metric line; see {log}')
    m=matches[-1];prompt,cached,output=map(int,m.group(1,2,3))
    read_ms=float(m.group(4));total_ms=float(m.group(5));pref=float(m.group(6));dec=float(m.group(7))
    dm=list(STRATA_DRAFT_RE.finditer(text));sm=list(STRATA_SUFFIX_RE.finditer(text))
    accepted=drafted=0
    if dm: accepted,drafted=map(int,dm[-1].group(1,2))
    suffix_accepted=suffix_drafted=0
    if sm: suffix_accepted,suffix_drafted=map(int,sm[-1].group(1,2))
    message=((resp.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
    return dict(engine='strata',run=run,draft_window=4,prompt_tokens=prompt,cached=cached,
        fresh=prompt-cached,output=output,prefill_tps=pref,decode_tps=dec,
        ttft_s=read_ms/1000.0,prefill_s=read_ms/1000.0,decode_s=(total_ms-read_ms)/1000.0,
        wall_s=wall,drafted=drafted+suffix_drafted,accepted=accepted+suffix_accepted,
        accepted_main=accepted,drafted_main=drafted,suffix_accepted=suffix_accepted,
        suffix_drafted=suffix_drafted,response_sha256=hashlib.sha256(message.encode()).hexdigest())

def summary(rows):
    out={}
    for window in range(5):
        eng=f'ninfer-d{window}';rr=[x for x in rows if x['engine']==eng]
        def stat(key):
            vals=[x[key] for x in rr if x.get(key) is not None]
            return {'median':statistics.median(vals),'min':min(vals),'max':max(vals)} if vals else None
        drafted=sum(x.get('drafted',0) for x in rr);accepted=sum(x.get('accepted',0) for x in rr)
        out[eng]={key:stat(key) for key in ('prefill_tps','decode_tps','ttft_s','wall_s',
                    'speculative_rounds','target_rounds','host_exposed_s','device_wait_exposed_s')}
        out[eng]['drafted_total']=drafted;out[eng]['accepted_total']=accepted
        out[eng]['acceptance']=accepted/drafted if drafted else None
        out[eng]['accepted_per_position']=[
            sum((x.get('accepted_per_position') or [0]*window)[p] for x in rr)
            for p in range(window)] if window else []
    sr=[x for x in rows if x['engine']=='strata']
    def sstat(key):
        vals=[x[key] for x in sr if x.get(key) is not None]
        return {'median':statistics.median(vals),'min':min(vals),'max':max(vals)} if vals else None
    out['strata']={key:sstat(key) for key in ('prefill_tps','decode_tps','ttft_s','wall_s')}
    sd=sum(x.get('drafted',0) for x in sr);sa=sum(x.get('accepted',0) for x in sr)
    out['strata']['drafted_total']=sd;out['strata']['accepted_total']=sa
    out['strata']['acceptance']=sa/sd if sd else None
    parity={}
    for run in sorted({x['run'] for x in rows if x['engine'].startswith('ninfer-d')}):
        rr=[x for x in rows if x['run']==run and x['engine'].startswith('ninfer-d')]
        parity[str(run)]={'all_equal':len({x['response_sha256'] for x in rr})==1,
                          'hashes':{x['engine']:x['response_sha256'] for x in rr}}
    out['ninfer_response_parity_by_run']=parity
    best=max(range(5),key=lambda w:out[f'ninfer-d{w}']['decode_tps']['median'])
    out['best_window']=best
    out['strata_over_best_ninfer_decode']=out['strata']['decode_tps']['median']/out[f'ninfer-d{best}']['decode_tps']['median']
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ninfer',type=Path,required=True)
    ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--profile',type=Path,required=True)
    ap.add_argument('--strata-python',type=Path,required=True);ap.add_argument('--strata-server',type=Path,required=True)
    ap.add_argument('--strata-config',type=Path,required=True);ap.add_argument('--strata-profile',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--repeats',type=int,default=3)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for p in (a.ninfer,a.artifact,a.profile,a.strata_python,a.strata_server,a.strata_config,a.strata_profile):
        if not p.exists(): raise FileNotFoundError(p)
    a.profile=strata_profile_to_ninfer(a.strata_profile,a.profile,a.output/'strata-derived-ninfer-profile.json')
    rows=[];arms=[0,1,2,3,4,'strata']
    for i in range(a.repeats):
        order=arms if i%2==0 else list(reversed(arms))
        for arm in order:
            row=run_strata(a,a.output,i) if arm=='strata' else run_ninfer(a,a.output,i,arm)
            rows.append(row);(a.output/'observations.json').write_text(json.dumps(rows,indent=2))
    s=summary(rows);(a.output/'summary.json').write_text(json.dumps(s,indent=2))
    lines=['NInfer MTP draft-window same-host screen (equal 156-slot cache)','',
        'window  prefill tok/s  decode tok/s  drafted  accepted  accept%  rounds']
    for w in range(5):
        x=s[f'ninfer-d{w}'];acc='n/a' if x['acceptance'] is None else f"{100*x['acceptance']:.1f}%"
        rounds='n/a' if x['speculative_rounds'] is None else f"{x['speculative_rounds']['median']:.0f}"
        lines.append(f"{w:>6}  {x['prefill_tps']['median']:>13.2f}  {x['decode_tps']['median']:>12.2f}  {x['drafted_total']:>7}  {x['accepted_total']:>8}  {acc:>7}  {rounds:>6}")
    x=s['strata'];acc='n/a' if x['acceptance'] is None else f"{100*x['acceptance']:.1f}%"
    lines += ['',f"Strata prefill/decode: {x['prefill_tps']['median']:.2f} / {x['decode_tps']['median']:.2f} tok/s; draft acceptance {acc}",
              f"Best NInfer draft window: {s['best_window']}; Strata/best-NInfer decode: {s['strata_over_best_ninfer_decode']:.2f}x",
              f"NInfer exact response parity by repetition: {all(x['all_equal'] for x in s['ninfer_response_parity_by_run'].values())}",
              '', 'Important: Strata uses a different quantization/container format; this diagnoses engine-path bottlenecks, not quantization-normalized speed.']
    (a.output/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
