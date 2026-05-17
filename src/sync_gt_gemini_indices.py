# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2"]
# ///
"""
Ensure ``data/gt_gemini.csv`` uses the same ``index=`` keys as ``data/gt.csv`` for each
``(Category, Text)`` row. Gemini IPA strings are kept in **label order** (1st target,
2nd target, …); only the index numbers are replaced with gold.

Check only (exit 1 if mismatches):

    uv run --no-project src/sync_gt_gemini_indices.py --check

Rewrite gt_gemini Label column in place:

    uv run --no-project src/sync_gt_gemini_indices.py --fix

From repo root:

    uv run --no-project categorized-heb-g2p-benchmark/src/sync_gt_gemini_indices.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

KEY_COLS = ("Category", "Text")
LABEL_COL = "Label"


def parse_label_parts(label: str) -> list[tuple[int, str]]:
    parts: list[tuple[int, str]] = []
    for part in str(label).strip().split():
        if "=" not in part:
            continue
        idx_str, ipa = part.split("=", 1)
        parts.append((int(idx_str), ipa))
    return parts


def indices_only(parts: list[tuple[int, str]]) -> list[int]:
    return [i for i, _ in parts]


def relabel_with_gt_indices(
    gt_parts: list[tuple[int, str]], gem_parts: list[tuple[int, str]]
) -> str:
    if len(gt_parts) != len(gem_parts):
        raise ValueError(
            f"target count mismatch: gt has {len(gt_parts)}, gemini has {len(gem_parts)}"
        )
    return " ".join(f"{gt_idx}={gem_ipa}" for (gt_idx, _), (_, gem_ipa) in zip(gt_parts, gem_parts))


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "gt.csv")
    ap.add_argument(
        "--gt-gemini",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "gt_gemini.csv",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Report mismatches; exit 1 if any")
    mode.add_argument("--fix", action="store_true", help="Rewrite gt_gemini indices to match gt")
    args = ap.parse_args()

    gt_df = load_csv(args.gt)
    gm_df = load_csv(args.gt_gemini)

    for col in (*KEY_COLS, LABEL_COL):
        if col not in gt_df.columns:
            sys.exit(f"{args.gt} missing column {col!r}")
        if col not in gm_df.columns:
            sys.exit(f"{args.gt_gemini} missing column {col!r}")

    gt_by_key = {
        (str(r["Category"]).strip(), str(r["Text"]).strip()): str(r[LABEL_COL])
        for _, r in gt_df.iterrows()
    }
    gm_keys = [(str(r["Category"]).strip(), str(r["Text"]).strip()) for _, r in gm_df.iterrows()]

    missing_in_gem = set(gt_by_key) - set(gm_keys)
    missing_in_gt = set(gm_keys) - set(gt_by_key)
    if missing_in_gem or missing_in_gt:
        print(f"Row key mismatches: {len(missing_in_gem)} only in gt, {len(missing_in_gt)} only in gemini")
        for k in list(missing_in_gem)[:5]:
            print(f"  only in gt: {k}")
        for k in list(missing_in_gt)[:5]:
            print(f"  only in gemini: {k}")
        sys.exit(1)

    index_mismatches: list[tuple[tuple[str, str], list[int], list[int], str, str]] = []
    count_mismatches: list[tuple[tuple[str, str], str, str]] = []
    fixes: dict[tuple[str, str], str] = {}

    for _, row in gm_df.iterrows():
        key = (str(row["Category"]).strip(), str(row["Text"]).strip())
        gt_label = gt_by_key[key]
        gem_label = str(row[LABEL_COL])
        gt_parts = parse_label_parts(gt_label)
        gem_parts = parse_label_parts(gem_label)
        gt_idx = indices_only(gt_parts)
        gem_idx = indices_only(gem_parts)

        if gt_idx != gem_idx:
            index_mismatches.append((key, gt_idx, gem_idx, gt_label, gem_label))

        if len(gt_parts) != len(gem_parts):
            count_mismatches.append((key, gt_label, gem_label))
            continue

        new_label = relabel_with_gt_indices(gt_parts, gem_parts)
        if new_label != gem_label.strip():
            fixes[key] = new_label

    print(f"Rows: {len(gm_df)}")
    print(f"Index mismatches (same row, different numbers): {len(index_mismatches)}")
    print(f"Target-count mismatches (cannot auto-fix): {len(count_mismatches)}")
    print(f"Labels to rewrite with --fix: {len(fixes)}")

    if index_mismatches:
        print("\nFirst index mismatches:")
        for key, gt_idx, gem_idx, gt_lab, gem_lab in index_mismatches[:10]:
            print(f"  [{key[0]}] {key[1][:50]}")
            print(f"    gt:     {gt_idx}  {gt_lab}")
            print(f"    gemini: {gem_idx}  {gem_lab}")

    if count_mismatches:
        print("\nTarget-count mismatches (manual edit required):")
        for key, gt_lab, gem_lab in count_mismatches[:5]:
            print(f"  {key[1][:50]}")
            print(f"    gt:     {gt_lab}")
            print(f"    gemini: {gem_lab}")

    if args.check:
        if index_mismatches or count_mismatches:
            sys.exit(1)
        print("OK: all indices and target counts match.")
        return

    if count_mismatches:
        sys.exit("Cannot --fix while target-count mismatches exist.")

    if not fixes:
        print("Nothing to fix.")
        return

    def apply_fix(row: pd.Series) -> str:
        key = (str(row["Category"]).strip(), str(row["Text"]).strip())
        return fixes.get(key, row[LABEL_COL])

    out = gm_df.copy()
    out[LABEL_COL] = out.apply(apply_fix, axis=1)
    out.to_csv(args.gt_gemini, index=False, lineterminator="\n")
    print(f"Wrote {len(fixes)} updated labels to {args.gt_gemini}")


if __name__ == "__main__":
    main()
