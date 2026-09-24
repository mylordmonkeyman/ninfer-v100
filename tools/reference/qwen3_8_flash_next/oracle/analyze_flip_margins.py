#!/usr/bin/env python3
"""Phase 11 flip-margin analysis.

Loads the FP32 oracle logits of a full Phase 11 run (via the oracle
manifest) and the per-position trace of a completed physical run
(candidate/oracle top-1, KL, relative NLL delta, max logit error), and
reports:

  - at FLIP positions (candidate top-1 != oracle top-1): the oracle's
    own top-1 margin (logit[top1] - logit[top2]), the oracle rank of the
    token the candidate selected, and the fraction of flips whose
    margin is below that position's measured max logit error;
  - at NON-FLIP positions: the baseline margin distribution.

If flips concentrate where the oracle's top-1 margin is below the
candidate's measured logit error and the candidate's choice is usually
one of the oracle's top few tokens, the flips are margin-limited
consequences of accumulated drift (the documented error-analysis
mechanism), not a systematic bias in the candidate. A flip at a
position with a large oracle margin that the measured error cannot
explain would be a defect signature.

Pure standard library (array + heapq + multiprocessing): the runner's
python3 has no third-party packages. The reported values are
numerically identical to the numpy definitions (argmax = first max,
partition(-k)[-k] = k-th largest value with multiplicity, percentile
= linear interpolation).
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import statistics
import sys
from array import array
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
MARGIN_THRESHOLDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
RANK_BOUNDS = (2, 3, 5, 10, 50, 100)
# Unambiguous-tier gate (v2): positions where the oracle's own top-1
# margin is this wide must agree on top-1 and stay under the KL limit.
# Calibrated 2026-09-24: max measured flip margin 12.3132 (63% headroom).
TIER_MARGIN = 20.0
TIER_KL_LIMIT = 1e-2


def percentile(values: list[float], p: float) -> float:
    """numpy.percentile with the default 'linear' interpolation."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def _analyze_chunk(rows):
    """rows: list of (index, record, root, vocabulary, cand, orac_top1,
                       mxe, kl)."""
    flip_margins: list[float] = []
    flip_margins10: list[float] = []
    flip_ranks: list[int] = []
    flip_explainable = 0
    flip_explainable10 = 0
    flip_explainable2 = 0
    nonflip_margins: list[float] = []
    flip_details: list[tuple] = []
    tier: list[tuple] = []
    mismatched_top1 = 0
    errors: list[str] = []

    for index, record, root, vocabulary, cand, orac_top1, mxe, kl in rows:
        if record["position"] != index:
            errors.append(f"manifest position {index} is not contiguous")
            continue
        entry = next(
            (t for t in record["tensors"] if t["name"] == "logits"), None)
        if entry is None or entry["dtype"] != "FP32":
            errors.append(f"manifest position {index} lacks FP32 logits")
            continue
        path = root / entry["file"]
        expected_bytes = int(entry["shape"][0]) * 4
        if not path.is_file() or path.stat().st_size != expected_bytes:
            errors.append(f"oracle logits file is invalid: {path}")
            continue
        with open(path, "rb") as handle:
            logits = array("f")
            logits.frombytes(handle.read())
        if len(logits) != vocabulary:
            errors.append(f"position {index} has a different vocabulary")
            continue

        top1_value = max(logits)
        top1 = logits.index(top1_value)
        if top1 != orac_top1:
            mismatched_top1 += 1
            continue
        rest = logits[:top1] + logits[top1 + 1:]
        margin = top1_value - max(rest)
        if cand == orac_top1:
            nonflip_margins.append(margin)
        else:
            flip_margins.append(margin)
            top10 = heapq.nlargest(10, logits)[-1]
            margin10 = top1_value - top10
            flip_margins10.append(margin10)
            rank = 1 + sum(1 for x in logits if x > logits[cand])
            flip_ranks.append(rank)
            if margin <= mxe:
                flip_explainable += 1
            if margin10 <= mxe:
                flip_explainable10 += 1
            if margin < 2.0 * mxe:
                flip_explainable2 += 1
            flip_details.append((index, margin, mxe, rank, cand, orac_top1))
        if margin >= TIER_MARGIN:
            tier.append((index, margin, kl, cand != orac_top1))

    return {
        "flip_margins": flip_margins,
        "flip_margins10": flip_margins10,
        "flip_ranks": flip_ranks,
        "flip_explainable": flip_explainable,
        "flip_explainable10": flip_explainable10,
        "flip_explainable2": flip_explainable2,
        "nonflip_margins": nonflip_margins,
        "tier": tier,
        "flip_details": flip_details,
        "mismatched_top1": mismatched_top1,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="oracle manifest.json (4096-position run)")
    parser.add_argument("--trace", required=True,
                        help="per-position trace CSV from a physical run")
    parser.add_argument("--out", default=None,
                        help="optional path to write the report")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel position workers (default: "
                        "min(16, cpu count))")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.manifest).parent

    trace = {}
    with open(args.trace, newline="") as handle:
        for row in csv.DictReader(handle):
            trace[int(row["position"])] = (
                int(row["candidate_top1"]),
                int(row["oracle_top1"]),
                float(row["kl"]),
                float(row["relative_nll_delta"]),
                float(row["max_logit_error"]),
            )

    positions = manifest["positions"]
    if len(positions) != len(trace):
        raise SystemExit(
            f"trace has {len(trace)} positions, manifest has {len(positions)}")

    vocabulary = -1
    for record in positions:
        entry = next(
            (t for t in record["tensors"] if t["name"] == "logits"), None)
        if entry is None or entry["dtype"] != "FP32":
            raise SystemExit(f"manifest position {record['position']} "
                             "lacks FP32 logits")
        shape0 = int(entry["shape"][0])
        if vocabulary < 0:
            vocabulary = shape0
        elif shape0 != vocabulary:
            raise SystemExit("manifest positions have different vocabularies")

    rows = []
    for index, record in enumerate(positions):
        if index not in trace:
            raise SystemExit(f"trace lacks position {index}")
        cand, orac_top1, kl, _nll, mxe = trace[index]
        rows.append(
            (index, record, root, vocabulary, cand, orac_top1, mxe, kl))

    workers = args.workers or min(16, os.cpu_count() or 1)
    workers = max(1, min(workers, len(rows)))
    chunks = [rows[i::workers] for i in range(workers)]
    if workers == 1:
        results = [_analyze_chunk(chunks[0])]
    else:
        with Pool(workers) as pool:
            results = pool.map(_analyze_chunk, chunks)

    flip_margins: list[float] = []
    flip_margins10: list[float] = []
    flip_ranks: list[int] = []
    flip_explainable = 0
    flip_explainable10 = 0
    flip_explainable2 = 0
    nonflip_margins: list[float] = []
    flip_details: list[tuple] = []
    tier: list[tuple] = []
    mismatched_top1 = 0
    for result in results:
        if result["errors"]:
            raise SystemExit(result["errors"][0])
        flip_margins.extend(result["flip_margins"])
        flip_margins10.extend(result["flip_margins10"])
        flip_ranks.extend(result["flip_ranks"])
        flip_explainable += result["flip_explainable"]
        flip_explainable10 += result["flip_explainable10"]
        flip_explainable2 += result["flip_explainable2"]
        nonflip_margins.extend(result["nonflip_margins"])
        flip_details.extend(result["flip_details"])
        tier.extend(result["tier"])
        mismatched_top1 += result["mismatched_top1"]

    report: list[str] = []
    emit = report.append
    emit(f"phase11.flip_margins.positions={len(positions)}")
    emit(f"phase11.flip_margins.vocabulary={vocabulary}")
    emit(f"phase11.flip_margins.manifest_top1_mismatches={mismatched_top1}")
    if mismatched_top1:
        raise SystemExit(
            f"manifest top-1 disagrees with the recorded trace for "
            f"{mismatched_top1} positions; the trace is not from this oracle")
    if not flip_margins or not nonflip_margins:
        raise SystemExit(
            f"trace has no flips or no non-flips (flips={len(flip_margins)}, "
            f"non-flips={len(nonflip_margins)})")

    flips = len(flip_margins)
    nonflips = len(nonflip_margins)
    emit(f"phase11.flip_margins.flips={flips} non_flips={nonflips}")

    def distribution(values: list[float], label: str) -> None:
        emit(f"== {label} (n={len(values)}) ==")
        emit("oracle top-1 margin: "
             + " ".join(f"p{p}={percentile(values, p):.4f}"
                        for p in PERCENTILES)
             + f" mean={statistics.fmean(values):.4f}")
        for threshold in MARGIN_THRESHOLDS:
            below = sum(1 for value in values if value < threshold)
            emit(f"margin < {threshold:g}: {below} "
                 f"({100.0 * below / len(values):.1f}%)")

    distribution(flip_margins, "flips")
    distribution(nonflip_margins, "non-flips")

    emit("== flip candidate rank in oracle logits ==")
    for bound in RANK_BOUNDS:
        within = sum(1 for rank in flip_ranks if rank <= bound)
        emit(f"rank <= {bound}: {within} ({100.0 * within / len(flip_ranks):.1f}%)")
    rank_counts = Counter(flip_ranks)
    emit("rank histogram (rank: count): "
         + ", ".join(f"{rank}:{count}" for rank, count
                     in sorted(rank_counts.items())[:20]))
    emit(f"max rank observed: {max(flip_ranks)}")
    emit("== flip explainability by measured max logit error ==")
    emit(f"margin(top1-top2) <= mxe: {flip_explainable}/{flips} "
         f"({100.0 * flip_explainable / flips:.1f}%)")
    emit(f"margin(top1-top10) <= mxe: {flip_explainable10}/{flips} "
         f"({100.0 * flip_explainable10 / flips:.1f}%)")
    emit(f"margin < 2*mxe (necessary condition for any flip): "
         f"{flip_explainable2}/{flips} "
         f"({100.0 * flip_explainable2 / flips:.1f}%)")

    emit("== per-flip detail ==")
    anomalies = [d for d in flip_details if d[1] > d[2]]
    emit(f"anomalies (margin > mxe): {len(anomalies)}")
    for pos, margin, mxe, rank, cand, orac in sorted(
            anomalies, key=lambda d: d[1] - d[2], reverse=True):
        emit(f"  pos={pos} margin={margin:.4f} mxe={mxe:.4f} "
             f"gap={margin - mxe:.4f} rank={rank} "
             f"cand={cand} oracle={orac}")
    emit("top-15 flips by oracle top-1 margin:")
    for pos, margin, mxe, rank, cand, orac in sorted(
            flip_details, key=lambda d: d[1], reverse=True)[:15]:
        emit(f"  pos={pos} margin={margin:.4f} mxe={mxe:.4f} "
             f"rank={rank} cand={cand} oracle={orac}")
    emit(f"max flip margin: {max(d[1] for d in flip_details):.4f}")
    emit(f"== unambiguous tier (oracle margin >= {TIER_MARGIN:g}) ==")
    tier_flips = [t for t in tier if t[3]]
    tier_kls = [t[2] for t in tier]
    emit(f"positions: {len(tier)} flips: {len(tier_flips)}")
    if tier_kls:
        emit("per-position KL: "
             + " ".join(f"p{p}={percentile(tier_kls, p):.6f}"
                        for p in (50, 90, 95, 99))
             + f" max={max(tier_kls):.6f} "
             f"mean={statistics.fmean(tier_kls):.6f}")
    tier_kl_violations = [t for t in tier
                          if not t[3] and t[2] > TIER_KL_LIMIT]
    emit(f"KL > {TIER_KL_LIMIT:g}: {len(tier_kl_violations)}")
    for pos, margin, kl, _flipped in sorted(tier_flips):
        emit(f"  FLIP pos={pos} margin={margin:.4f} kl={kl:.6f}")
    for pos, margin, kl, _flipped in sorted(tier_kl_violations):
        emit(f"  KLVIOL pos={pos} margin={margin:.4f} kl={kl:.6f}")
    tier_verdict = ("pass" if not tier_flips and not tier_kl_violations
                    else "fail")
    emit(f"phase11.flip_margins.tier_top1="
         f"{len(tier) - len(tier_flips)}/{len(tier)} "
         f"tier_kl_max="
         f"{max(tier_kls) if tier_kls else 0.0:.6f} "
         f"tier_gate={tier_verdict}")

    text = "\n".join(report) + "\n"
    sys.stdout.write(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
