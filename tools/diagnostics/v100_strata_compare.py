#!/usr/bin/env python3
import argparse,json,math,os,re,signal,socket,statistics,subprocess,threading,time,urllib.request
from pathlib import Path

PROMPT=' '.join(f'Record {i}: the solar station stores energy during daylight and supplies the village after sunset.' for i in range(64))+' Explain how its battery storage works in detail.'
STRATA_RE=re.compile(r'\[strata\] request prompt (\d+) cached (\d+) output (\d+) prompt_read ([\d.]+) ms total ([\d.]+) ms prefill ([\d.]+) tok/s decode ([\d.]+) tok/s')

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

def run_ninfer(a,out,run,grouped,diagnostic=False):
    mode='grouped' if grouped else 'single'; p=port();base=f'http://127.0.0.1:{p}'; log=out/f'ninfer-{mode}-{run}{"-diag" if diagnostic else ""}.log'; reqlog=out/f'ninfer-{mode}-{run}{"-diag" if diagnostic else ""}-requests.jsonl'
    env=os.environ.copy(); env.update({
      'NINFER_V100_EXPERT_PROFILE':str(a.profile.resolve()),'NINFER_V100_EXPERT_POLICY':'static',
      'NINFER_FLASH_NEXT_EXPERT_CACHE':'1','NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS':'64',
      'NINFER_FLASH_NEXT_EXPERT_CACHE_SERIAL':'0','NINFER_FLASH_NEXT_EXPERT_CACHE_PREFILL':'1',
      'NINFER_FLASH_NEXT_EXPERT_CACHE_BATCHED_DECODE':'0','NINFER_FLASH_NEXT_EXPERT_CACHE_GROUPED_PREFILL':'1',
      'NINFER_V100_DEVICE_ROUTE_COMBINE':'1','NINFER_V100_PREFILL_EXPERT_POLICY':'cpu-cache',
      'NINFER_V100_CPU_EXPERT_GROUP':'1' if grouped else '0','NINFER_V100_ROUTE_HANDOFF':'0','NINFER_V100_PLE_IO':'mmap',
      'NINFER_V100_QSA_SCORE_MMA':'0','NINFER_FLASH_NEXT_QSA_PREFILL_MMA':'0',
      'NINFER_V100_TELEMETRY':'1' if diagnostic else '0','NINFER_FLASH_NEXT_STAGE_LEDGER':'1' if diagnostic else '0',
      'NINFER_FLASH_NEXT_CPU_EXPERT_FP32_INTERMEDIATE':'0','NINFER_FLASH_NEXT_MOE_SHARED_FP32_INTERMEDIATE':'0'})
    cmd=[str(a.ninfer.resolve()),str(a.artifact.absolute()),'--host','127.0.0.1','--port',str(p),'--max-context','4096','--kv-capacity','4096','--max-concurrency','1','--prefill-chunk','2048','--kv-dtype','fp8','--device-state-slots','2','--host-state-slots','2','--host-kv-mib','256','--max-private-continuations','2','--max-shared-prefixes','2','--no-cuda-graph','--no-qsa-prefill-mma','--no-thinking','--spec','mtp','--draft-tokens','3','--lm-head-draft','--request-log-jsonl',str(reqlog)]
    monstop=threading.Event(); t=threading.Thread(target=monitor,args=(out/f'ninfer-{mode}-{run}{"-diag" if diagnostic else ""}-gpu.csv',monstop));t.start()
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            model=wait_ready(proc,base,log); start=time.monotonic(); resp=request(base,'/v1/chat/completions',common_payload(model,run)); wall=time.monotonic()-start
        finally: stop(proc);monstop.set();t.join()
    events=[json.loads(x) for x in reqlog.read_text().splitlines() if x.strip()]
    done=[x for x in events if x.get('event')=='request_done'][-1]; r=done['result']; tm=done['timings_seconds']; fresh=r['prompt_tokens']-r['prefix_cache_hit_tokens']
    return dict(engine=f'ninfer-{mode}',run=run,prompt_tokens=r['prompt_tokens'],cached=r['prefix_cache_hit_tokens'],fresh=fresh,output=r['completion_tokens'],prefill_tps=fresh/tm['prefill'],decode_tps=r['completion_tokens']/tm['decode'],ttft_s=tm['ttft'],prefill_s=tm['prefill'],decode_s=tm['decode'],wall_s=wall,drafted=done.get('speculative',{}).get('drafted_tokens'),accepted=done.get('speculative',{}).get('accepted_tokens'),diagnostic=diagnostic)

def run_strata(a,out,run):
    p=port();base=f'http://127.0.0.1:{p}';log=out/f'strata-{run}.log';env=os.environ.copy();env['STRATA_REQUEST_LINES']='1';env['CUDA_VISIBLE_DEVICES']='0'
    # /opt/ai/strata is intentionally mounted read-only in the runner. Rewrite only the
    # top-level log path into the writable results directory for this benchmark run.
    cfg=json.loads(a.strata_config.read_text())
    cfg['log']=str((out/f'strata-engine-{run}.log').resolve())
    run_cfg=out/f'strata-config-{run}.json';run_cfg.write_text(json.dumps(cfg,indent=2))
    # The self-hosted runner container blocks set_mempolicy(2), so run Strata under the container's native NUMA policy.
    # NInfer is measured under the same container policy; this keeps the same-host comparison valid.
    cmd=[str(a.strata_python),str(a.strata_server),'--engine','strata','--config',str(run_cfg),'--host','127.0.0.1','--port',str(p)]
    monstop=threading.Event();t=threading.Thread(target=monitor,args=(out/f'strata-{run}-gpu.csv',monstop));t.start()
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            model=wait_ready(proc,base,log);start=time.monotonic();resp=request(base,'/v1/chat/completions',common_payload(model,run));wall=time.monotonic()-start
        finally:stop(proc);monstop.set();t.join()
    text=log.read_text(errors='replace');matches=list(STRATA_RE.finditer(text))
    if not matches: raise RuntimeError(f'no Strata request metric line; see {log}')
    m=matches[-1];prompt,cached,output=map(int,m.group(1,2,3));read_ms=float(m.group(4));total_ms=float(m.group(5));pref=float(m.group(6));dec=float(m.group(7))
    return dict(engine='strata',run=run,prompt_tokens=prompt,cached=cached,fresh=prompt-cached,output=output,prefill_tps=pref,decode_tps=dec,ttft_s=read_ms/1000.0,prefill_s=read_ms/1000.0,decode_s=(total_ms-read_ms)/1000.0,wall_s=wall)

def summary(rows):
    out={}
    for eng in ('ninfer-single','ninfer-grouped','strata'):
        rr=[x for x in rows if x['engine']==eng and not x.get('diagnostic')]
        out[eng]={}
        for key in ('prefill_tps','decode_tps','ttft_s','wall_s'):
            vals=[x[key] for x in rr];out[eng][key]={'median':statistics.median(vals),'min':min(vals),'max':max(vals)}
    best_pref=max(out['ninfer-single']['prefill_tps']['median'],out['ninfer-grouped']['prefill_tps']['median']);best_dec=max(out['ninfer-single']['decode_tps']['median'],out['ninfer-grouped']['decode_tps']['median']);out['ratios']={'strata_over_best_ninfer_prefill':out['strata']['prefill_tps']['median']/best_pref,'strata_over_best_ninfer_decode':out['strata']['decode_tps']['median']/best_dec}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ninfer',type=Path,required=True);ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--strata-python',type=Path,required=True);ap.add_argument('--strata-server',type=Path,required=True);ap.add_argument('--strata-config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--repeats',type=int,default=3);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for p in (a.ninfer,a.artifact,a.profile,a.strata_python,a.strata_server,a.strata_config):
        if not p.exists(): raise FileNotFoundError(p)
    rows=[]
    for i in range(a.repeats):
        order=('ninfer-single','ninfer-grouped','strata') if i%2==0 else ('strata','ninfer-grouped','ninfer-single')
        for eng in order:
            row=(run_ninfer(a,a.output,i,False) if eng=='ninfer-single' else (run_ninfer(a,a.output,i,True) if eng=='ninfer-grouped' else run_strata(a,a.output,i)));rows.append(row);(a.output/'observations.json').write_text(json.dumps(rows,indent=2))
    # One untimed NInfer diagnostic pass captures stage/telemetry evidence without contaminating timed rows.
    rows.append(run_ninfer(a,a.output,999,True,True));(a.output/'observations.json').write_text(json.dumps(rows,indent=2))
    s=summary(rows);(a.output/'summary.json').write_text(json.dumps(s,indent=2))
    lines=['NInfer vs Strata-V100 same-host screen','',f"NInfer single prefill/decode: {s['ninfer-single']['prefill_tps']['median']:.2f} / {s['ninfer-single']['decode_tps']['median']:.2f} tok/s",f"NInfer grouped prefill/decode: {s['ninfer-grouped']['prefill_tps']['median']:.2f} / {s['ninfer-grouped']['decode_tps']['median']:.2f} tok/s",f"Strata prefill/decode: {s['strata']['prefill_tps']['median']:.2f} / {s['strata']['decode_tps']['median']:.2f} tok/s",f"Strata/best-NInfer prefill: {s['ratios']['strata_over_best_ninfer_prefill']:.2f}x",f"Strata/best-NInfer decode: {s['ratios']['strata_over_best_ninfer_decode']:.2f}x",'', 'Important: NInfer and Strata use different quantization/container formats; this screen diagnoses engine-path bottlenecks, not quantization-normalized speed.']
    (a.output/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
