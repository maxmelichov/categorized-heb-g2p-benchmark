"""
Generate data/predictions.csv for benchmark.py.

From repo root ``renikud/``, use the benchmark package directory once:

    cd categorized-heb-g2p-benchmark

Do not run ``cd categorized-heb-g2p-benchmark`` again if your shell is already there.

Gold labels come from ``--gt`` (default ``data/gt.csv``). Gemini reference IPA for the same
``(Category, Text)`` rows and **same word indices as gold** comes from ``--gt-gemini``
(default ``data/gt_gemini.csv``); each row's ``Label`` uses the same ``index=ipa`` format as gold.
Pass ``--gt-gemini ""`` to omit the ``gemini`` column. Rows with no matching Gemini row get an
empty ``gemini`` string (after joining per-target slots with spaces).

Renikud (PyTorch checkpoints, recommended — two classifiers + Phonikud ONNX):

    uv run src/predict.py \\
      --renikud-checkpoint ../renikud_classifier_300M_model/knesset-classifier-vox/knesset-classifier-vox/checkpoint-best \\
      --renikud-checkpoint-phonikud ../renikud_classifier_300M_model/knesset-classifier-phonikud/checkpoint-best \\
      --phonikud-onnx phonikud.onnx

HF weights: https://huggingface.co/notmax123/renikud_exp/tree/main/knesset-classifier-phonikud

Optional locals (auto-used when present): ``renikud-ctc/outputs/knesset-ctc-vox/checkpoint-best``,
``phonikud-byt5/outputs/knesset-byt5-ASR/checkpoint-best`` → columns ``renikud_ctc``, ``phonikud_byt5``.
Pass ``""`` for those flags to disable.

Renikud (ONNX only):

    uv run src/predict.py --renikud-onnx renikud.onnx --phonikud-onnx phonikud.onnx

Use phonikud FP32 weights (e.g. phonikud-1.0.onnx ~308MB), not int8.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd
from phonikud import phonemize
from phonikud_onnx import Phonikud
from renikud_onnx import G2P
from tqdm import tqdm

HEBREW_RE = re.compile(r"[\u05d0-\u05ff]+")
PHONEME_RE = re.compile(r"[abdefhijklmnopstuvwzɡʁʃʒʔˈχ]+")
GERSHAYIM = "\u05f4"  # ״ Hebrew acronym mark — gold IPA omits it


def normalize_renikud_acronym_ipa(pred: str) -> str:
    """Gold acronym IPA omits orthographic / typographic quote marks between letters."""
    for ch in (
        GERSHAYIM,  # ״
        '"',
        "\u201c",
        "\u201d",  # “ ”
        "\u2033",  # ″ double prime
        "\u00ab",
        "\u00bb",  # « »
    ):
        pred = pred.replace(ch, "")
    return pred


def postprocess_renikud_classifier_ipa(pred: str, category: str) -> str:
    """Same cleanup for every Hebrew classifier column (vox + phonikud-trained checkpoints)."""
    if str(category).strip().lower() == "acronyms":
        pred = normalize_renikud_acronym_ipa(pred)
    return pred


def normalize_sentence_for_g2p(sentence: str) -> str:
    """Hyphen in דז'ה-וו splits words and breaks Phonikud alignment; Renikud benefits too."""
    return sentence.replace("דז'ה-וו", "דז'הוו").replace("דז׳ה-וו", "דז׳הוו")


def strip_hebrew_acronym_quotes(sentence: str) -> str:
    """Drop orthographic acronym quotes inside Hebrew tokens (מנכ\"ל → מנכל, דו\"ח → דוח).

    Leaves ג'/ז'/צ' digraph apostrophes intact (those are not between two Hebrew letters).
    """
    parts: list[str] = []
    for m in re.finditer(r"\S+|\s+", sentence):
        chunk = m.group(0)
        if not chunk.strip():
            parts.append(chunk)
            continue
        w = chunk
        while True:
            prev = w
            w = re.sub(r"([\u05d0-\u05ff])\u05f4([\u05d0-\u05ff])", r"\1\2", w)
            w = re.sub(r'([\u05d0-\u05ff])"([\u05d0-\u05ff])', r"\1\2", w)
            if w == prev:
                break
        parts.append(w)
    return "".join(parts)


def renikud_repo_src() -> Path:
    # categorized-heb-g2p-benchmark/src/predict.py -> repo root
    return Path(__file__).resolve().parent.parent.parent / "renikud_classifier_300M_model" / "src"


def default_renikud_checkpoint_phonikud() -> str:
    p = (
        Path(__file__).resolve().parent.parent.parent
        / "renikud_classifier_300M_model"
        / "knesset-classifier-phonikud"
        / "checkpoint-best"
    )
    return str(p) if p.is_dir() and (p / "model.safetensors").exists() else ""


def extract_hebrew_words(sentence: str) -> list[str]:
    return HEBREW_RE.findall(sentence)


def extract_target_phonemes(phonemized_sentence: str, word_index: int) -> str:
    """Extract phonemes for the word at 0-based word_index (Renikud ONNX path)."""
    tokens = PHONEME_RE.findall(phonemized_sentence)
    if word_index < 0 or word_index >= len(tokens):
        return ""
    return tokens[word_index]


def parse_targets(gt_raw: str) -> dict[int, str]:
    targets: dict[int, str] = {}
    for part in str(gt_raw).strip().split():
        if "=" not in part:
            continue
        idx_str, phonemes = part.split("=", 1)
        targets[int(idx_str)] = phonemes
    return targets


def phonikud_words(sentence: str, phonikud_model: Phonikud) -> list[str] | None:
    vocalized = phonikud_model.add_diacritics(sentence)
    full = phonemize(vocalized)
    parts = full.split()
    n_he = len(list(re.finditer(r"\S+", sentence)))
    if len(parts) != n_he:
        return None
    return parts


def strip_ipa_edges(s: str) -> str:
    # Keep in sync with acronym cleanup (quotes often hug IPA at word boundaries).
    edge = ".,?!'\"\u05f4\u201c\u201d\u2033…"
    s = s.strip()
    while s and s[0] in edge:
        s = s[1:]
    while s and s[-1] in edge:
        s = s[:-1]
    return s


def resolve_word_index(idx: int, n_words: int) -> int:
    """GT indices are usually 0-based; some rows use 1-based (idx == n_words for last word)."""
    if idx < 0:
        return idx
    if idx < n_words:
        return idx
    if n_words > 0 and idx == n_words:
        return n_words - 1
    if idx - 1 >= 0 and idx - 1 < n_words:
        return idx - 1
    return idx


def gemini_slice_for_targets(
    targets: dict[int, str], targets_gem: dict[int, str]
) -> str:
    """Same word-index keys as gold ``targets``; IPA strings from Gemini ``Label``."""
    return " ".join(targets_gem.get(k, "") for k in targets)


def preds_from_word_list(words_pred: list[str], targets: dict[int, str]) -> str:
    n = len(words_pred)
    keys_order = list(targets.keys())
    out = []
    for k in keys_order:
        j = resolve_word_index(k, n)
        chunk = words_pred[j] if 0 <= j < n else ""
        out.append(strip_ipa_edges(chunk))
    return " ".join(out)


PREDICTION_COLUMN_ORDER = (
    "gemini",
    "renikud",
    "renikud_phonikud",
    "renikud_ctc",
    "phonikud_byt5",
    "phonikud",
)


def renikud_ctc_src() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "renikud-ctc" / "src"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _ctc_checkpoint_has_weights(p: Path) -> bool:
    return p.is_dir() and (
        (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
    )


def default_renikud_ctc_checkpoint() -> str:
    """Local training export (no Hub download)."""
    for p in (_repo_root() / "renikud-ctc" / "outputs" / "knesset-ctc-vox" / "checkpoint-best",):
        if _ctc_checkpoint_has_weights(p):
            return str(p)
    return ""


def default_phonikud_byt5_model() -> str:
    """Local training export: ``phonikud-byt5/outputs/knesset-byt5-ASR/checkpoint-best``."""
    for p in (
        _repo_root()
        / "phonikud-byt5"
        / "outputs"
        / "knesset-byt5-ASR"
        / "checkpoint-best",
    ):
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
    return ""


def load_ctc_bundle(checkpoint_dir: Path, device):
    """Load Renikud CTC; drop ``sys.modules`` shims so classifier imports stay isolated."""
    import torch

    ctc_src = renikud_ctc_src()
    if not ctc_src.is_dir():
        raise SystemExit(f"Missing renikud-ctc checkout (expected {ctc_src})")
    sys.path.insert(0, str(ctc_src))
    try:
        import constants as ctc_constants  # noqa: PLC0415
        import infer as ctc_infer  # noqa: PLC0415
        import model as ctc_model  # noqa: PLC0415
        import tokenization as ctc_tokenization  # noqa: PLC0415

        max_len = int(ctc_constants.MAX_LEN)
        tok = ctc_tokenization.load_encoder_tokenizer()
        m = ctc_model.HebrewG2PCTC()
        ctc_infer.load_checkpoint_into_model(m, str(checkpoint_dir))
        m.to(device).eval()
        decode_fn = ctc_tokenization.decode_ctc
    finally:
        sys.path.remove(str(ctc_src))
    for name in ("constants", "tokenization", "model", "infer"):
        sys.modules.pop(name, None)
    return {
        "model": m,
        "tokenizer": tok,
        "device": device,
        "max_len": max_len,
        "decode_ctc": decode_fn,
    }


def ctc_phonemize_words(sentence: str, bundle: dict) -> list[str]:
    import torch

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    max_len = bundle["max_len"]
    decode_ctc_fn = bundle["decode_ctc"]

    def one(text: str) -> str:
        enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
        ilen = int(out["input_lengths"][0])
        ids = out["logits"][0].argmax(dim=-1)[:ilen].tolist()
        return decode_ctc_fn(ids)

    n_words = len(list(re.finditer(r"\S+", sentence)))
    full = one(sentence)
    parts = full.split()
    if len(parts) == n_words:
        return parts
    words_out: list[str] = []
    for m in re.finditer(r"\S+", sentence):
        ipa = one(m.group(0)).strip().replace(" ", "")
        words_out.append(ipa)
    return words_out


def load_byt5_bundle(model_dir: Path, device):
    from transformers import ByT5Tokenizer, T5ForConditionalGeneration

    tok = ByT5Tokenizer.from_pretrained(str(model_dir))
    m = T5ForConditionalGeneration.from_pretrained(str(model_dir))
    m.to(device).eval()
    return {"model": m, "tokenizer": tok, "device": device}


def byt5_phonemize_words(sentence: str, bundle: dict, max_gen: int = 512) -> list[str]:
    import torch

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]

    def gen_one(text: str, mlen: int) -> str:
        inp = tokenizer(
            text,
            max_length=mlen,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_length=mlen,
                num_beams=2,
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    n_words = len(list(re.finditer(r"\S+", sentence)))
    pred = gen_one(sentence, max_gen)
    parts = pred.split()
    if len(parts) == n_words:
        return parts
    words_out: list[str] = []
    for m in re.finditer(r"\S+", sentence):
        w = m.group(0)
        pw = gen_one(w, min(256, max_gen)).strip()
        chunk = pw.split()[0] if pw.split() else pw
        words_out.append(chunk)
    return words_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="data/gt.csv")
    parser.add_argument(
        "--gt-gemini",
        default="data/gt_gemini.csv",
        help="Optional CSV (Category, Text, Label) with Gemini IPA per same slices as --gt. "
        "Empty string to skip gemini column.",
    )
    parser.add_argument(
        "--renikud-checkpoint",
        default="",
        help="Renikud classifier dir with model.safetensors (e.g. .../checkpoint-best)",
    )
    parser.add_argument(
        "--renikud-checkpoint-phonikud",
        default=default_renikud_checkpoint_phonikud(),
        help="Second classifier (Knesset+Phonikud); CSV column renikud_phonikud. Default: repo weight if present.",
    )
    parser.add_argument(
        "--renikud-onnx",
        default="renikud.onnx",
        help="Used when --renikud-checkpoint is not set",
    )
    parser.add_argument("--phonikud-onnx", default="phonikud.onnx")
    parser.add_argument(
        "--renikud-ctc-checkpoint",
        default=default_renikud_ctc_checkpoint(),
        help="Renikud CTC checkpoint-best dir → column renikud_ctc. Empty to skip.",
    )
    parser.add_argument(
        "--phonikud-byt5-model",
        default=default_phonikud_byt5_model(),
        help="Phonikud ByT5 model dir (config + weights) → column phonikud_byt5. Empty to skip.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        metavar="NAME",
        help="Run only these Category values from --gt (repeatable). Merges into --output when set.",
    )
    parser.add_argument(
        "--output",
        default="data/predictions.csv",
        help="Predictions CSV path (default: data/predictions.csv).",
    )
    args = parser.parse_args()

    gt_path = Path(args.gt)
    if not gt_path.exists():
        raise SystemExit(f"Missing --gt: {gt_path}")

    df = pd.read_csv(gt_path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"category": "Category", "sentence": "Text", "gt_raw": "Label"})
    text_col = "Text" if "Text" in df.columns else "sentence"
    label_col = "Label" if "Label" in df.columns else "gt_raw"
    cat_col = "Category" if "Category" in df.columns else "category"
    df = df[[cat_col, text_col, label_col]].dropna(subset=[text_col, label_col])
    df[label_col] = df[label_col].astype(str)
    df = df[df[label_col].str.strip().astype(bool)]

    category_filter = {c.strip() for c in args.category if str(c).strip()}
    if category_filter:
        df = df[df[cat_col].astype(str).str.strip().isin(category_filter)].copy()
        if df.empty:
            raise SystemExit(f"No rows in {gt_path} for --category {sorted(category_filter)}")
        print(f"Category filter: {sorted(category_filter)} ({len(df)} rows)")

    gem_lookup: dict[tuple[str, str], str] = {}
    gem_arg = str(args.gt_gemini or "").strip()
    if gem_arg:
        gem_path = Path(gem_arg)
        if not gem_path.is_file():
            raise SystemExit(f"Missing --gt-gemini: {gem_path}")
        df_g = pd.read_csv(gem_path)
        df_g.columns = [str(c).strip() for c in df_g.columns]
        df_g = df_g.rename(columns={"category": "Category", "sentence": "Text", "gt_raw": "Label"})
        g_cat = "Category" if "Category" in df_g.columns else "category"
        g_text = "Text" if "Text" in df_g.columns else "sentence"
        g_lab = "Label" if "Label" in df_g.columns else "gt_raw"
        if g_lab not in df_g.columns:
            raise SystemExit(f"{gem_path} needs a Label column (or gt_raw)")
        for _, grow in df_g.iterrows():
            if pd.isna(grow.get(g_text)) or pd.isna(grow.get(g_lab)):
                continue
            key = (str(grow[g_cat]).strip(), str(grow[g_text]).strip())
            gem_lookup[key] = str(grow[g_lab]).strip()

    phonikud_model = Phonikud(args.phonikud_onnx)

    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ctc_bundle = None
    ctc_arg = str(args.renikud_ctc_checkpoint or "").strip()
    if ctc_arg:
        ctc_path = Path(ctc_arg)
        if not ctc_path.is_dir():
            raise SystemExit(f"CTC checkpoint not found: {ctc_path}")
        ctc_bundle = load_ctc_bundle(ctc_path, device)

    byt5_bundle = None
    byt5_arg = str(args.phonikud_byt5_model or "").strip()
    if byt5_arg:
        byt5_path = Path(byt5_arg)
        if not byt5_path.is_dir():
            raise SystemExit(f"ByT5 model dir not found: {byt5_path}")
        byt5_bundle = load_byt5_bundle(byt5_path, device)

    torch_specs: list[tuple[str, Path]] = []
    if args.renikud_checkpoint:
        ckpt_r = Path(args.renikud_checkpoint)
        if not ckpt_r.is_dir():
            raise SystemExit(f"Checkpoint not found: {ckpt_r}")
        torch_specs.append(("renikud", ckpt_r))
    phonikud_ckpt_arg = str(args.renikud_checkpoint_phonikud or "").strip()
    if phonikud_ckpt_arg:
        ckpt_p = Path(phonikud_ckpt_arg)
        if not ckpt_p.is_dir():
            raise SystemExit(
                f"Checkpoint not found (--renikud-checkpoint-phonikud): {ckpt_p}"
            )
        torch_specs.append(("renikud_phonikud", ckpt_p))

    renikud_onnx = None
    renikud_torch_named: list[tuple[str, tuple]] = []
    if torch_specs:
        sys.path.insert(0, str(renikud_repo_src()))

        from constants import MAX_LEN
        from infer import load_checkpoint, phonemize_words
        from model import HebrewG2PClassifier
        from tokenization import load_tokenizer

        tokenizer = load_tokenizer()
        seen_paths: set[str] = set()
        for col_name, ckpt in torch_specs:
            r = ckpt.resolve()
            key = str(r)
            if key in seen_paths:
                raise SystemExit(f"Duplicate checkpoint path for {col_name}: {ckpt}")
            seen_paths.add(key)
            model = HebrewG2PClassifier()
            load_checkpoint(model, str(ckpt))
            model.to(device).eval()
            renikud_torch_named.append((col_name, (model, tokenizer, device, MAX_LEN)))

    if not args.renikud_checkpoint:
        renikud_onnx = G2P(args.renikud_onnx)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        category = row[cat_col]
        sentence = str(row[text_col]).strip()
        sentence_g2p = normalize_sentence_for_g2p(sentence)
        targets = parse_targets(row[label_col])
        if not targets:
            continue

        gkey = (str(category).strip(), sentence)
        gem_label = gem_lookup.get(gkey, "") if gem_lookup else ""
        targets_gem = parse_targets(gem_label) if gem_label else {}
        gem_phonemes = gemini_slice_for_targets(targets, targets_gem)

        row_models: dict[str, str] = {}
        for col_name, bundle in renikud_torch_named:
            model, tokenizer, device, max_len = bundle
            ctx = sentence_g2p
            if col_name == "renikud_phonikud" and str(category).strip().lower() == "acronyms":
                ctx = strip_hebrew_acronym_quotes(sentence_g2p)
            rw = phonemize_words(
                ctx,
                model,
                tokenizer,
                device,
                max_len,
                restrict_geresh_loanwords=(col_name == "renikud_phonikud"),
            )
            pred = postprocess_renikud_classifier_ipa(
                preds_from_word_list(rw, targets), category
            )
            row_models[col_name] = pred

        if ctc_bundle:
            rw_ctc = ctc_phonemize_words(sentence_g2p, ctc_bundle)
            row_models["renikud_ctc"] = postprocess_renikud_classifier_ipa(
                preds_from_word_list(rw_ctc, targets), category
            )

        if byt5_bundle:
            rw_bt = byt5_phonemize_words(sentence_g2p, byt5_bundle)
            row_models["phonikud_byt5"] = postprocess_renikud_classifier_ipa(
                preds_from_word_list(rw_bt, targets), category
            )

        if renikud_onnx:
            renikud_full = renikud_onnx.phonemize(sentence_g2p)
            renikud_pred = postprocess_renikud_classifier_ipa(
                " ".join(extract_target_phonemes(renikud_full, i) for i in targets),
                category,
            )
            row_models["renikud"] = renikud_pred

        pk_list = phonikud_words(sentence_g2p, phonikud_model)
        if pk_list is None:
            phonikud_pred = ""
        else:
            phonikud_pred = preds_from_word_list(pk_list, targets)

        word_indices = " ".join(str(i) for i in targets)
        gt_phonemes = " ".join(targets.values())
        rec = {
            "category": category,
            "sentence": sentence,
            "word_indices": word_indices,
            "gt": gt_phonemes,
            "gemini": gem_phonemes,
        }
        for key in PREDICTION_COLUMN_ORDER:
            if key == "gemini":
                continue
            if key == "phonikud":
                rec[key] = phonikud_pred
            elif key in row_models:
                rec[key] = row_models[key]
        rows.append(rec)

    out = pd.DataFrame(rows)
    ordered_cols = [
        c
        for c in ["category", "sentence", "word_indices", "gt", *PREDICTION_COLUMN_ORDER]
        if c in out.columns
    ]
    out = out[ordered_cols]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if category_filter and out_path.is_file():
        prev = pd.read_csv(out_path)
        n_replaced = int(prev["category"].astype(str).str.strip().isin(category_filter).sum())
        prev = prev[~prev["category"].astype(str).str.strip().isin(category_filter)]
        n_new = len(out)
        out = pd.concat([prev, out], ignore_index=True)
        print(
            f"Merged into {out_path}: replaced {n_replaced} rows, "
            f"added {n_new} for {sorted(category_filter)}, kept {len(prev)} other rows"
        )

    out.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved {len(out)} rows to {out_path}")
    counts = out["category"].value_counts().sort_index()
    print("\nRows per category (evaluation rows with labels):")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")
    print(f"  TOTAL: {len(out)}")


if __name__ == "__main__":
    main()
