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
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def percentile(values: np.ndarray, p: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, p))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="oracle manifest.json (4096-position run)")
    parser.add_argument("--trace", required=True,
                        help="per-position trace CSV from a physical run")
    parser.add_argument("--out", default=None,
                        help="optional path to write the report")
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

    flip_margins: list[float] = []
    flip_margins10: list[float] = []
    flip_ranks: list[int] = []
    flip_explainable = 0
    flip_explainable10 = 0
    nonflip_margins: list[float] = []
    mismatched_top1 = 0
    vocabulary = -1

    for index, record in enumerate(positions):
        if record["position"] != index:
            raise SystemExit(f"manifest position {index} is not contiguous")
        logits_entry = next(
            (t for t in record["tensors"] if t["name"] == "logits"), None)
        if logits_entry is None or logits_entry["dtype"] != "FP32":
            raise SystemExit(f"manifest position {index} lacks FP32 logits")
        path = root / logits_entry["file"]
        expected_bytes = int(logits_entry["shape"][0]) * 4
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise SystemExit(f"oracle logits file is invalid: {path}")
        with open(path, "rb") as handle:
            logits = np.frombuffer(handle.read(), dtype=np.float32)
        if vocabulary < 0:
            vocabulary = logits.size
        elif logits.size != vocabulary:
            raise SystemExit(f"position {index} has a different vocabulary")

        cand, orac_top1, _kl, _nll, mxe = trace[index]
        top1 = int(np.argmax(logits))
        if top1 != orac_top1:
            mismatched_top1 += 1
            continue
        top2 = float(np.partition(logits, -2)[-2])
        margin = float(logits[top1] - top2)
        top10 = float(np.partition(logits, -10)[-10])
        margin10 = float(logits[top1] - top10)
        if cand == orac_top1:
            nonflip_margins.append(margin)
        else:
            flip_margins.append(margin)
            flip_margins10.append(margin10)
            cand_rank = 1 + int(np.count_nonzero(logits > logits[cand]))
            flip_ranks.append(cand_rank)
            if margin <= mxe:
                flip_explainable += 1
            if margin10 <= mxe:
                flip_explainable10 += 1

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
        arr = np.asarray(values, dtype=np.float64)
        emit(f"== {label} (n={arr.size}) ==")
        emit(f"oracle top-1 margin: "
             + " ".join(f"p{p}={percentile(arr, p):.4f}"
                        for p in (1, 5, 10, 25, 50, 75, 90))
             + f" mean={arr.mean():.4f}")
        for threshold in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
            below = int(np.count_nonzero(arr < threshold))
            emit(f"margin < {threshold:g}: {below} "
                 f"({100.0 * below / arr.size:.1f}%)")

    distribution(flip_margins, "flips")
    distribution(nonflip_margins, "non-flips")

    emit("== flip candidate rank in oracle logits ==")
    ranks = np.asarray(flip_ranks, dtype=np.int64)
    rank_counts = Counter(ranks.tolist())
    for bound in (2, 3, 5, 10, 50, 100):
        within = int(np.count_nonzero(ranks <= bound))
        emit(f"rank <= {bound}: {within} ({100.0 * within / ranks.size:.1f}%)")
    emit("rank histogram (rank: count): "
         + ", ".join(f"{rank}:{count}" for rank, count
                     in sorted(rank_counts.items())[:20]))
    emit(f"max rank observed: {int(ranks.max())}")
    emit("== flip explainability by measured max logit error ==")
    emit(f"margin(top1-top2) <= mxe: {flip_explainable}/{flips} "
         f"({100.0 * flip_explainable / flips:.1f}%)")
    emit(f"margin(top1-top10) <= mxe: {flip_explainable10}/{flips} "
         f"({100.0 * flip_explainable10 / flips:.1f}%)")

    text = "\n".join(report) + "\n"
    sys.stdout.write(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
