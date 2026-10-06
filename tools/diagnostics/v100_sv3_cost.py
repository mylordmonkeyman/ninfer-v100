#!/usr/bin/env python3
"""Collect the bounded SV3 per-expert CPU versus staged-GPU cost calibration."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

EXPECTED_ROUTES=(1,2,4,8,16,32,64)
REPEATS=7
EXPERT_BYTES=2_765_056

def collect(text: str) -> dict:
    records=[json.loads(line.split("sv3.cost=",1)[1])
             for line in text.splitlines() if line.startswith("sv3.cost=")]
    if len(records)!=len(EXPECTED_ROUTES)*REPEATS:
        raise ValueError("incomplete SV3 cost records")
    rows=[]
    for routes in EXPECTED_ROUTES:
        selected=[r for r in records if r.get("routes")==routes]
        if sorted(r.get("sample") for r in selected)!=list(range(REPEATS)):
            raise ValueError("missing or duplicate SV3 cost samples")
        for row in selected:
            if (row.get("expert_h2d_bytes")!=EXPERT_BYTES or
                row.get("cpu_result_h2d_bytes")!=routes*2560*4 or
                any(not math.isfinite(row.get(k,math.nan)) or row[k]<=0
                    for k in ("cpu_us","gpu_us"))):
                raise ValueError("invalid SV3 cost accounting or timing")
        cpu=[r["cpu_us"] for r in selected]
        gpu=[r["gpu_us"] for r in selected]
        rows.append({"routes":routes,
            "cpu_us_median":statistics.median(cpu),
            "cpu_us_min":min(cpu),"cpu_us_max":max(cpu),
            "gpu_us_median":statistics.median(gpu),
            "gpu_us_min":min(gpu),"gpu_us_max":max(gpu),
            "gpu_change_percent":(statistics.median(gpu)/statistics.median(cpu)-1)*100,
            "separate_ranges":max(gpu)<min(cpu) or max(cpu)<min(gpu)})
    crossover=next((r["routes"] for r in rows
                    if r["gpu_us_max"]<r["cpu_us_min"]),None)
    return {"schema":1,"milestone":"SV3","scope":"warm_per_expert_isolated_cost",
            "qualified":False,"automatic_route_threshold":None,
            "observed_separate_range_crossover":crossover,"rows":rows,
            "limitations":[
                "warm isolated canonical synthetic expert cost; no overlap with other layer work",
                "CPU timing includes production 32-worker execution and miss-result H2D",
                "GPU timing includes host packing, expert H2D and grouped execution",
                "shared route/control transfer and whole-model effects are excluded",
                "collector never changes the runtime auto policy or defaults"]}

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--executable",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env["NINFER_V100_SV3_COST"]="1"
    process=subprocess.run([str(args.executable.resolve())],env=env,text=True,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800)
    (args.output/"cost.log").write_text(process.stdout)
    if process.returncode: raise RuntimeError("SV3 cost executable failed")
    report=collect(process.stdout)
    (args.output/"cost.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["SV3 isolated canonical per-expert cost; no policy promotion.","",
           "| Routes | CPU median us (range) | GPU median us (range) | GPU change |",
           "|---:|---:|---:|---:|"]
    for r in report["rows"]:
        lines.append(f'| {r["routes"]} | {r["cpu_us_median"]:.2f} '
                     f'({r["cpu_us_min"]:.2f}–{r["cpu_us_max"]:.2f}) | '
                     f'{r["gpu_us_median"]:.2f} ({r["gpu_us_min"]:.2f}–{r["gpu_us_max"]:.2f}) | '
                     f'{r["gpu_change_percent"]:+.2f}% |')
    (args.output/"cost.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__": main()
