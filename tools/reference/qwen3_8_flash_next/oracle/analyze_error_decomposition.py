#!/usr/bin/env python3
"""Analyze Phase 11 full-layer error decomposition logs.

Parses ``phase11.error_decomposition.`` lines emitted by
``ninfer_qwen3_8_flash_next_vertical_slice_real_test`` and reports, per
layer/stage, how much of the observed NRMSE is explained by the rounding
floor (the oracle value re-rounded to the stage's storage dtype) versus
residual implementation error. A stage whose floor ratio (observed NRMSE
divided by floor NRMSE) stays near 1 across positions is at its rounding
floor; a stage with a ratio well above 2 has real implementation error
beyond rounding.

Floor ratio conventions emitted by the test:
  - a positive finite ratio for BF16 stages;
  - ``inf`` for FP32/integer stages with nonzero NRMSE (zero floor, any
    deviation is an implementation error);
  - ``-1`` for clean FP32/integer stages (zero error, zero floor).
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

LINE_RE = re.compile(
    r"^(?:\d+:\s*)?phase11\.error_decomposition\.position=(\d+)"
    r" stage=(\S+) dtype=(\w+)"
    r" nrmse=(\S+) floor_nrmse=(\S+) residual_nrmse=(\S+)"
    r" floor_ratio=(\S+)\s*$"
)
LAYER_RE = re.compile(r"^L(\d{2})_(.*)$")

OVER_2X = 2.0
OVER_10X = 10.0


def parse_log(text):
    records = []
    for line in text.splitlines():
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        position, stage, dtype, nrmse, floor, residual, ratio = match.groups()
        layer_match = LAYER_RE.match(stage)
        layer = int(layer_match.group(1)) if layer_match else -1
        suffix = layer_match.group(2) if layer_match else stage
        records.append(
            {
                "position": int(position),
                "layer": layer,
                "stage": stage,
                "suffix": suffix,
                "dtype": dtype,
                "nrmse": float(nrmse),
                "floor": float(floor),
                "residual": float(residual),
                "ratio": float(ratio),
            }
        )
    return records


def over_floor(record, threshold):
    return record["ratio"] >= threshold


def mean(values):
    return sum(values) / len(values) if values else 0.0


def fmt(value, digits=6):
    if value == -1.0:
        return "-1"
    if math.isinf(value):
        return "inf"
    return format(value, f".{digits}g")


def layer_span(layers):
    if not layers or (len(layers) == 1 and layers[0] == -1):
        return "top"
    if len(layers) == 1:
        return f"L{layers[0]:02d}"
    return f"L{min(layers):02d}-L{max(layers):02d} ({len(layers)})"


def stage_type_rollup(records):
    by_suffix = defaultdict(list)
    for record in records:
        by_suffix[record["suffix"]].append(record)
    rows = []
    for suffix, group in by_suffix.items():
        nrmse = [r["nrmse"] for r in group]
        floor = [r["floor"] for r in group]
        residual = [r["residual"] for r in group]
        rows.append(
            {
                "suffix": suffix,
                "cells": len(group),
                "layers": layer_span(sorted({r["layer"] for r in group})),
                "dtype": "/".join(sorted({r["dtype"] for r in group})),
                "mean_nrmse": mean(nrmse),
                "max_nrmse": max(nrmse),
                "mean_floor": mean(floor),
                "mean_residual": mean(residual),
                "frac_over_2x": sum(over_floor(r, OVER_2X) for r in group) / len(group),
                "frac_over_10x": sum(over_floor(r, OVER_10X) for r in group) / len(group),
                "max_ratio": max(r["ratio"] for r in group),
            }
        )
    rows.sort(key=lambda row: (-row["mean_residual"], -row["max_nrmse"], row["suffix"]))
    return rows


def layer_stage_cells(records):
    by_cell = defaultdict(list)
    for record in records:
        by_cell[(record["layer"], record["suffix"])].append(record)
    rows = []
    for (layer, suffix), group in by_cell.items():
        nrmse = [r["nrmse"] for r in group]
        residual = [r["residual"] for r in group]
        rows.append(
            {
                "layer": layer,
                "stage": f"L{layer:02d}_{suffix}" if layer >= 0 else suffix,
                "positions": len(group),
                "dtype": "/".join(sorted({r["dtype"] for r in group})),
                "mean_nrmse": mean(nrmse),
                "max_nrmse": max(nrmse),
                "mean_floor": mean([r["floor"] for r in group]),
                "mean_residual": mean(residual),
                "frac_over_2x": sum(over_floor(r, OVER_2X) for r in group) / len(group),
                "frac_over_10x": sum(over_floor(r, OVER_10X) for r in group) / len(group),
                "worst_position": max(group, key=lambda r: r["residual"])["position"],
                "worst_ratio": max(r["ratio"] for r in group),
            }
        )
    rows.sort(key=lambda row: (-row["mean_residual"], -row["max_nrmse"], row["stage"]))
    return rows


def layer_rollup(records):
    by_layer = defaultdict(list)
    for record in records:
        by_layer[record["layer"]].append(record)
    rows = []
    for layer, group in by_layer.items():
        nrmse = [r["nrmse"] for r in group]
        worst = max(group, key=lambda r: r["residual"])
        rows.append(
            {
                "layer": layer,
                "cells": len(group),
                "mean_nrmse": mean(nrmse),
                "max_nrmse": max(nrmse),
                "frac_over_2x": sum(over_floor(r, OVER_2X) for r in group) / len(group),
                "worst_stage": worst["stage"],
                "worst_residual": worst["residual"],
                "worst_position": worst["position"],
            }
        )
    rows.sort(key=lambda row: (row["layer"] != -1, row["layer"]))
    return rows


def position_rollup(records):
    by_position = defaultdict(list)
    for record in records:
        by_position[record["position"]].append(record)
    rows = []
    for position, group in by_position.items():
        worst = max(group, key=lambda r: r["residual"])
        rows.append(
            {
                "position": position,
                "cells": len(group),
                "over_2x": sum(over_floor(r, OVER_2X) for r in group),
                "over_10x": sum(over_floor(r, OVER_10X) for r in group),
                "worst_stage": worst["stage"],
                "worst_residual": worst["residual"],
                "worst_ratio": worst["ratio"],
            }
        )
    rows.sort(key=lambda row: row["position"])
    return rows


def print_table(title, header, rows, format_row):
    print(f"== {title} ==")
    print(" | ".join(header))
    for row in rows:
        print(" | ".join(format_row(row)))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Phase 11 full-layer error decomposition logs"
    )
    parser.add_argument("--log", required=True, help="vertical-slice test log")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory to write CSV summaries into",
    )
    args = parser.parse_args()

    records = parse_log(Path(args.log).read_text(encoding="utf-8", errors="replace"))
    if not records:
        raise SystemExit("no phase11.error_decomposition lines found in log")

    positions = sorted({r["position"] for r in records})
    suffixes = sorted({r["suffix"] for r in records})
    dtypes = sorted({r["dtype"] for r in records})
    print(f"error_decomposition positions={positions[0]}..{positions[-1]} "
          f"count={len(records)} distinct_stages={len(suffixes)} dtypes={dtypes}")
    print()

    stage_types = stage_type_rollup(records)
    print_table(
        "STAGE-TYPE ROLLUP (all layers, sorted by mean residual NRMSE)",
        ["stage", "cells", "layers", "dtype", "mean_nrmse", "max_nrmse",
         "mean_floor", "mean_residual", "frac>2x", "frac>10x", "max_ratio"],
        stage_types,
        lambda r: [
            r["suffix"], str(r["cells"]), r["layers"], r["dtype"],
            fmt(r["mean_nrmse"]), fmt(r["max_nrmse"]), fmt(r["mean_floor"]),
            fmt(r["mean_residual"]), fmt(r["frac_over_2x"], 3),
            fmt(r["frac_over_10x"], 3), fmt(r["max_ratio"]),
        ],
    )

    cells = layer_stage_cells(records)
    print_table(
        "WORST (LAYER, STAGE) CELLS BY MEAN RESIDUAL NRMSE (top 40)",
        ["stage", "positions", "dtype", "mean_nrmse", "max_nrmse", "mean_floor",
         "mean_residual", "frac>2x", "worst_position", "worst_ratio"],
        cells[:40],
        lambda r: [
            r["stage"], str(r["positions"]), r["dtype"],
            fmt(r["mean_nrmse"]), fmt(r["max_nrmse"]), fmt(r["mean_floor"]),
            fmt(r["mean_residual"]), fmt(r["frac_over_2x"], 3),
            str(r["worst_position"]), fmt(r["worst_ratio"]),
        ],
    )

    finite_over_2x = [
        record
        for record in records
        if over_floor(record, OVER_2X) and not math.isinf(record["ratio"])
    ]
    finite_over_2x.sort(
        key=lambda r: (-r["ratio"], -r["nrmse"], r["position"], r["stage"])
    )
    print_table(
        "TOP FINITE FLOOR-RATIO CELLS (ratio >= 2, top 40)",
        ["stage", "position", "dtype", "nrmse", "floor", "residual", "ratio"],
        finite_over_2x[:40],
        lambda r: [
            r["stage"], str(r["position"]), r["dtype"],
            fmt(r["nrmse"]), fmt(r["floor"]), fmt(r["residual"]), fmt(r["ratio"]),
        ],
    )

    print_table(
        "PER-POSITION SUMMARY",
        ["position", "cells", "over_2x", "over_10x", "worst_stage",
         "worst_residual", "worst_ratio"],
        position_rollup(records),
        lambda r: [
            str(r["position"]), str(r["cells"]), str(r["over_2x"]),
            str(r["over_10x"]), r["worst_stage"], fmt(r["worst_residual"]),
            fmt(r["worst_ratio"]),
        ],
    )

    print_table(
        "PER-LAYER ROLLUP",
        ["layer", "cells", "mean_nrmse", "max_nrmse", "frac>2x", "worst_stage",
         "worst_residual", "worst_position"],
        layer_rollup(records),
        lambda r: [
            f"L{r['layer']:02d}" if r["layer"] >= 0 else "top",
            str(r["cells"]), fmt(r["mean_nrmse"]), fmt(r["max_nrmse"]),
            fmt(r["frac_over_2x"], 3), r["worst_stage"],
            fmt(r["worst_residual"]), str(r["worst_position"]),
        ],
    )

    router_flips = [
        record
        for record in records
        if record["dtype"] == "i32" and record["nrmse"] > 0.0
    ]
    print(f"== ROUTER ID FLIPS (i32 stages with nonzero NRMSE) ==")
    if not router_flips:
        print("none")
    for record in sorted(router_flips, key=lambda r: (r["position"], r["stage"])):
        print(f"  position={record['position']} stage={record['stage']} "
              f"nrmse={fmt(record['nrmse'])}")
    print()

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "error_decomposition_cells.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["position", "layer", "stage", "suffix", "dtype",
                 "nrmse", "floor_nrmse", "residual_nrmse", "floor_ratio"]
            )
            for record in sorted(
                records, key=lambda r: (r["position"], r["layer"], r["stage"])
            ):
                writer.writerow(
                    [record["position"], record["layer"], record["stage"],
                     record["suffix"], record["dtype"],
                     repr(record["nrmse"]), repr(record["floor"]),
                     repr(record["residual"]), repr(record["ratio"])]
                )
        with (out_dir / "error_decomposition_stage_types.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["suffix", "cells", "layers", "dtype", "mean_nrmse", "max_nrmse",
                 "mean_floor_nrmse", "mean_residual_nrmse", "frac_over_2x",
                 "frac_over_10x", "max_ratio"]
            )
            for row in stage_types:
                writer.writerow(
                    [row["suffix"], row["cells"], row["layers"], row["dtype"],
                     repr(row["mean_nrmse"]), repr(row["max_nrmse"]),
                     repr(row["mean_floor"]), repr(row["mean_residual"]),
                     repr(row["frac_over_2x"]), repr(row["frac_over_10x"]),
                     repr(row["max_ratio"])]
                )
        with (out_dir / "error_decomposition_layer_stage_cells.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["stage", "positions", "dtype", "mean_nrmse", "max_nrmse",
                 "mean_floor_nrmse", "mean_residual_nrmse", "frac_over_2x",
                 "frac_over_10x", "worst_position", "worst_ratio"]
            )
            for row in cells:
                writer.writerow(
                    [row["stage"], row["positions"], row["dtype"],
                     repr(row["mean_nrmse"]), repr(row["max_nrmse"]),
                     repr(row["mean_floor"]), repr(row["mean_residual"]),
                     repr(row["frac_over_2x"]), repr(row["frac_over_10x"]),
                     row["worst_position"], repr(row["worst_ratio"])]
                )
        with (out_dir / "error_decomposition_positions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["position", "cells", "over_2x", "over_10x", "worst_stage",
                 "worst_residual_nrmse", "worst_ratio"]
            )
            for row in position_rollup(records):
                writer.writerow(
                    [row["position"], row["cells"], row["over_2x"],
                     row["over_10x"], row["worst_stage"],
                     repr(row["worst_residual"]), repr(row["worst_ratio"])]
                )
        with (out_dir / "error_decomposition_layers.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["layer", "cells", "mean_nrmse", "max_nrmse", "frac_over_2x",
                 "worst_stage", "worst_residual_nrmse", "worst_position"]
            )
            for row in layer_rollup(records):
                writer.writerow(
                    [row["layer"], row["cells"], repr(row["mean_nrmse"]),
                     repr(row["max_nrmse"]), repr(row["frac_over_2x"]),
                     row["worst_stage"], repr(row["worst_residual"]),
                     row["worst_position"]]
                )
        print(f"Wrote CSV summaries to {out_dir}")


if __name__ == "__main__":
    main()
