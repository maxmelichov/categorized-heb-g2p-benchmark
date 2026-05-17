# /// script
# requires-python = ">=3.12"
# dependencies = ["jiwer>=4", "pandas>=2"]
# ///
"""
Flag benchmark rows where renikud / renikud_phonikud / renikud_ctc agree (low pairwise CER)
but all disagree strongly with `gt` (high CER vs gt). That pattern often indicates a bad
`word_indices` slice (models see the same wrong span; reference is for a different word).

From repo root (`renikud/`), use ``--no-project`` so ``uv`` does not install the main
``renikud`` package (macOS often fails on ``xformers``); the script uses its PEP 723 deps only:

    uv run --no-project categorized-heb-g2p-benchmark/src/check_word_index_suspicion_cer.py

From inside `categorized-heb-g2p-benchmark/`:

    uv run --no-project src/check_word_index_suspicion_cer.py

(If your default ``uv`` environment already works, you can omit ``--no-project`` here.)

Requires `data/predictions.csv` (see `src/predict.py` / `src/benchmark.py`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from jiwer import cer

MODELS = ("renikud", "renikud_phonikud", "renikud_ctc")
REQUIRED_COLS = ("gt",) + MODELS


def _as_str(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v)


def row_cer(reference: str, hypothesis: str) -> float:
    """jiwer CER with empty-safe strings."""
    ref = _as_str(reference)
    hyp = _as_str(hypothesis)
    return float(cer(ref, hyp))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "predictions.csv",
        help="CSV with gt + renikud columns",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "word_index_suspicion_cer_rows.csv",
    )
    ap.add_argument(
        "--out-rows",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "word_index_suspicion_cer_row_numbers.txt",
        help="1-based CSV line numbers (same convention as word_index_suspicion_rows.csv)",
    )
    ap.add_argument(
        "--max-pairwise-cer",
        type=float,
        default=0.05,
        help="Flag if max pairwise CER among the three Renikud outputs is <= this",
    )
    ap.add_argument(
        "--min-gt-cer",
        type=float,
        default=0.25,
        help="Flag if min(CER(gt, m)) over the three models is >= this",
    )
    ap.add_argument(
        "--min-cer-gap",
        type=float,
        default=0.12,
        help="Also require min_gt_cer - max_pairwise_cer >= this (gt much worse than cross-model agreement)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {args.predictions}: {missing}")

    cer_gt_r: list[float] = []
    cer_gt_rp: list[float] = []
    cer_gt_rc: list[float] = []
    cer_r_rp: list[float] = []
    cer_r_rc: list[float] = []
    cer_rp_rc: list[float] = []
    max_pair: list[float] = []
    min_gt: list[float] = []
    gap: list[float] = []

    r, rp, rc = MODELS

    for _, row in df.iterrows():
        gt = row["gt"]
        sr, srp, src = row[r], row[rp], row[rc]

        c_gr = row_cer(gt, sr)
        c_grp = row_cer(gt, srp)
        c_grc = row_cer(gt, src)
        c_rrp = row_cer(sr, srp)
        c_rrc = row_cer(sr, src)
        c_rprc = row_cer(srp, src)

        mp = max(c_rrp, c_rrc, c_rprc)
        mg = min(c_gr, c_grp, c_grc)

        cer_gt_r.append(c_gr)
        cer_gt_rp.append(c_grp)
        cer_gt_rc.append(c_grc)
        cer_r_rp.append(c_rrp)
        cer_r_rc.append(c_rrc)
        cer_rp_rc.append(c_rprc)
        max_pair.append(mp)
        min_gt.append(mg)
        gap.append(mg - mp)

    out = df.copy()
    out["cer_gt_r"] = cer_gt_r
    out["cer_gt_rp"] = cer_gt_rp
    out["cer_gt_rc"] = cer_gt_rc
    out["cer_r_rp"] = cer_r_rp
    out["cer_r_rc"] = cer_r_rc
    out["cer_rp_rc"] = cer_rp_rc
    out["max_pairwise_cer"] = max_pair
    out["min_gt_cer"] = min_gt
    out["cer_gt_minus_pairwise"] = gap

    mask = (
        (out["max_pairwise_cer"] <= args.max_pairwise_cer)
        & (out["min_gt_cer"] >= args.min_gt_cer)
        & (out["cer_gt_minus_pairwise"] >= args.min_cer_gap)
    )
    flagged = out.loc[mask].copy()
    flagged.insert(0, "df_row_0based", flagged.index)
    flagged.insert(0, "csv_line_1based", flagged.index + 2)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(args.out_csv, index=False)
    args.out_rows.write_text(
        "\n".join(str(int(x)) for x in flagged["csv_line_1based"].tolist()) + "\n",
        encoding="utf-8",
    )

    print(f"Read {len(df)} rows from {args.predictions}")
    print(
        f"Flagged {len(flagged)} rows "
        f"(max_pairwise_cer<={args.max_pairwise_cer}, "
        f"min_gt_cer>={args.min_gt_cer}, "
        f"gap>={args.min_cer_gap})"
    )
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_rows}")


if __name__ == "__main__":
    main()
