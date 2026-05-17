# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2"]
# ///
"""
Refresh ``ilspeech-v2-test`` rows in ``data/gt.csv`` and ``data/gt_gemini.csv`` from
``ilspeech-v2/metadata_test.csv``.

    cd categorized-heb-g2p-benchmark
    uv run --no-project src/sync_ilspeech_v2_gt.py

Optional:

    uv run --no-project src/sync_ilspeech_v2_gt.py --metadata ../ilspeech-v2/metadata_test.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

CATEGORY = "ilspeech-v2-test"
IPA_EDGE = ".,?!'\"״…"
# Prosodic clitics only (e.g. lˈe in לעשר), not full words like bˈen.
CLITIC_IPA = frozenset({"lˈe", "bˈe", "kˈe", "mˈe", "vˈe", "ʃˈe", "hˈe", "wˈe", "uˈl", "ʔˈe"})


def strip_ipa_edges(token: str) -> str:
    while token and token[0] in IPA_EDGE:
        token = token[1:]
    while token and token[-1] in IPA_EDGE:
        token = token[:-1]
    return token


def ipa_tokens(ipa_sentence: str) -> list[str]:
    return [strip_ipa_edges(w) for w in ipa_sentence.split() if w]


def hebrew_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def merge_score(left: str, right: str) -> int:
    if left in CLITIC_IPA:
        return 2
    return 0


def align_ipa_to_hebrew(ipa: list[str], n_hebrew: int) -> list[str]:
    if len(ipa) == n_hebrew:
        return ipa
    need = len(ipa) - n_hebrew
    if need < 1:
        raise ValueError(f"too few IPA tokens ({len(ipa)}) for {n_hebrew} Hebrew words")

    def search(tokens: list[str], merges_left: int) -> list[str] | None:
        if len(tokens) == n_hebrew:
            return tokens
        if merges_left == 0:
            return None
        options: list[tuple[int, int, list[str]]] = []
        for i in range(len(tokens) - 1):
            merged = tokens[:i] + [tokens[i] + tokens[i + 1]] + tokens[i + 2 :]
            if len(merged) < n_hebrew:
                continue
            if len(merged) == n_hebrew:
                options.append((merge_score(tokens[i], tokens[i + 1]), i, merged))
                continue
            if merges_left > 1:
                child = search(merged, merges_left - 1)
                if child is not None:
                    options.append((merge_score(tokens[i], tokens[i + 1]), i, child))
        if not options:
            return None
        options.sort(key=lambda x: (-x[0], x[1]))
        return options[0][2]

    out = search(ipa, need)
    if out is None:
        raise ValueError(f"cannot align {len(ipa)} IPA tokens to {n_hebrew} Hebrew words")
    return out


def build_label(ipa_sentence: str, hebrew_text: str) -> str:
    ipa = ipa_tokens(ipa_sentence)
    he = hebrew_tokens(hebrew_text)
    aligned = align_ipa_to_hebrew(ipa, len(he))
    if len(aligned) != len(he):
        raise ValueError("alignment length mismatch after merge")
    return " ".join(f"{i}={tok}" for i, tok in enumerate(aligned))


def load_metadata(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_no}: expected id|ipa|hebrew, got {len(parts)} fields")
        _id, ipa, hebrew = parts
        label = build_label(ipa, hebrew)
        rows.append(
            {
                "Category": CATEGORY,
                "Text": hebrew,
                "Label": label,
                "_id": _id,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("../ilspeech-v2/metadata_test.csv"),
        help="ILSpeech v2 test metadata (id|ipa|hebrew per line)",
    )
    parser.add_argument("--gt", type=Path, default=Path("data/gt.csv"))
    parser.add_argument("--gt-gemini", type=Path, default=Path("data/gt_gemini.csv"))
    args = parser.parse_args()

    metadata_path = args.metadata.resolve()
    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata: {metadata_path}")

    ilspeech_rows = load_metadata(metadata_path)
    print(f"Loaded {len(ilspeech_rows)} utterances from {metadata_path}")

    # --- gt.csv (preserve extra columns) ---
    gt_path = args.gt
    gt_df = pd.read_csv(gt_path)
    before = len(gt_df)
    gt_other = gt_df[gt_df["Category"] != CATEGORY].copy()
    gt_extra_cols = [c for c in gt_df.columns if c not in ("Category", "Text", "Label")]

    gt_new = pd.DataFrame(
        {
            "Category": [r["Category"] for r in ilspeech_rows],
            "Text": [r["Text"] for r in ilspeech_rows],
            "Label": [r["Label"] for r in ilspeech_rows],
        }
    )
    for col in gt_extra_cols:
        gt_new[col] = ""

    gt_out = pd.concat([gt_other, gt_new], ignore_index=True)
    gt_out.to_csv(gt_path, index=False, quoting=csv.QUOTE_MINIMAL)
    removed = before - len(gt_other)
    print(f"Updated {gt_path}: removed {removed} old {CATEGORY} rows, wrote {len(gt_new)} new ({len(gt_out)} total)")

    # --- gt_gemini.csv: keep existing Gemini labels where Text matches, else gold IPA ---
    gem_path = args.gt_gemini
    gem_df = pd.read_csv(gem_path)
    gem_before = len(gem_df)
    gem_other = gem_df[gem_df["Category"] != CATEGORY].copy()
    old_gem = gem_df[gem_df["Category"] == CATEGORY]
    old_lookup = {
        str(row["Text"]).strip(): str(row["Label"]).strip()
        for _, row in old_gem.iterrows()
        if pd.notna(row.get("Text")) and pd.notna(row.get("Label"))
    }

    gem_labels: list[str] = []
    reused = 0
    for row in ilspeech_rows:
        text = row["Text"]
        if text in old_lookup:
            gem_labels.append(old_lookup[text])
            reused += 1
        else:
            gem_labels.append(row["Label"])

    gem_new = pd.DataFrame(
        {
            "Category": [r["Category"] for r in ilspeech_rows],
            "Text": [r["Text"] for r in ilspeech_rows],
            "Label": gem_labels,
        }
    )
    gem_out = pd.concat([gem_other, gem_new], ignore_index=True)
    gem_out.to_csv(gem_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(
        f"Updated {gem_path}: removed {gem_before - len(gem_other)} old rows, "
        f"wrote {len(gem_new)} ({reused} Gemini labels reused by Text, "
        f"{len(gem_new) - reused} filled from gold IPA)"
    )


if __name__ == "__main__":
    main()
