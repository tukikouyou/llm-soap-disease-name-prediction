#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute metrics for two outputs: Free-text (G) vs Structured (H),
given an Excel where:
  B = free-text input            (not strictly required for metrics)
  C = structured-text input      (used as entity list for consistency)
  E = gold label (diagnosis)
  F = ICD code                    (optional, not used here)
  G = output for free-text       (candidates + explanation)
  H = output for structured-text (candidates + explanation)

Outputs:
  - <out>_summary.csv
  - <out>_per_case_FreeText.csv
  - <out>_per_case_Structured.csv
  - <out>_reproducibility.csv  (pairwise Jaccard between FreeText and Structured)

Usage example:
  python compute_topk_metrics.py \
    --input "/path/your.xlsx" --sheet "Sheet1" \
    --col-entities C --col-gold E --col-free-out G --col-struct-out H \
    --out-prefix "/path/run1"
"""
import argparse
import json
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------- helpers: column letters -> index ----------------------
def col_letter_to_idx(letter: str) -> int:
    """
    Convert Excel-style column letters (A,B,...,Z,AA,AB,...) to 0-based index.
    """
    s = letter.strip().upper()
    if not s.isalpha():
        raise ValueError(f"Invalid column letter: {letter}")
    val = 0
    for ch in s:
        val = val * 26 + (ord(ch) - ord('A') + 1)
    return val - 1  # 0-based

# ---------------------- Normalization & Synonyms ----------------------
def zenkaku_to_hankaku(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKC", s)

def normalize_label(s: str) -> str:
    """Approximate canonicalization for Japanese medical labels."""
    s = zenkaku_to_hankaku(str(s)).lower()
    s = re.sub(r"[\s　\t\r\n]+", "", s)
    s = re.sub(r"[（）\(\)［\]【】\[\]、,.:：;；\-–—•*★☆＋+→←<>＜＞]", "", s)
    s = s.replace("ウイルス性", "").replace("細菌性", "")
    s = s.replace("急性", "").replace("慢性", "")
    return s

BUILTIN_SYNONYMS = {
    "咳喘息": {"咳嗽型喘息", "咳嗽喘息", "coughvariantasthma", "cva"},
    "急性上気道炎": {"感冒", "風邪", "かぜ", "上気道炎", "かぜ症候群"},
    "インフルエンザ": {"influenza", "flu"},
    "気管支喘息": {"喘息"},
    "急性気管支炎": {"気管支炎"},
    "肺炎": {"肺炎球菌性肺炎", "viralpneumonia", "bacterialpneumonia"},
    "咽頭炎": {"咽頭痛", "咽頭炎症"},
    "副鼻腔炎": {"蓄膿症", "副鼻腔炎症"},
}

def build_synonym_canon(user_synonyms: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
    syn_map: Dict[str, str] = {}
    src = dict(BUILTIN_SYNONYMS)
    if user_synonyms:
        for k, vals in user_synonyms.items():
            cur = set(src.get(k, set()))
            cur.update(vals if isinstance(vals, (list, set, tuple)) else [vals])
            src[k] = cur
    for k, vals in src.items():
        can = normalize_label(k)
        syn_map[can] = can
        for v in vals:
            syn_map[normalize_label(v)] = can
    return syn_map

def canonize_label(s: str, syn2canon: Dict[str, str]) -> str:
    ns = normalize_label(s)
    return syn2canon.get(ns, ns)

# ---------------------- parsing utils ----------------------
def split_entities(cell: str) -> List[str]:
    """
    Split the structured cell (column C) into entity tokens.
    Primary split is by newline, with a fallback split on punctuation and whitespace.
    """
    if not isinstance(cell, str) or not cell.strip():
        return []
    txt = zenkaku_to_hankaku(cell)
    # primary: newline
    raw = [t.strip() for t in txt.split("\n") if t.strip()]
    # if very few tokens, fallback broader split
    if len(raw) <= 1:
        raw = re.split(r"[,\u3001、;；|｜/\s]+", txt)
        raw = [t.strip() for t in raw if t.strip()]
    # filter trivial tokens
    toks = []
    for t in raw:
        if len(t) < 2:
            continue
        if re.fullmatch(r"[\d\.\-/%]+", t):
            continue
        if re.fullmatch(r"[A-Za-z]{1}", t):
            continue
        toks.append(t)
    # de-dup preserve order
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def extract_top_diagnoses(text: str, topk: int = 3) -> List[str]:
    """Extract up to topk diagnoses from narrative outputs (G/H)."""
    if not isinstance(text, str) or not text.strip():
        return []
    t = zenkaku_to_hankaku(text)
    # numbered lines
    lines = re.findall(r"(?:^|\n)\s*(?:\d+|[①-⑳]|[一二三四五六七八九十])[\.\）\)]\s*(.+)", t)
    if lines:
        preds = [l.strip().splitlines()[0] for l in lines]
    else:
        # bullets
        preds = [b.strip().splitlines()[0] for b in re.findall(r"(?:^|\n)[\-・\*]\s*(.+)", t)]
    cleaned = []
    for p in preds:
        p = re.sub(r"[（(].*?[)）]", "", p)  # drop parenthetical notes
        p = re.sub(r"[。．.]+$", "", p).strip()
        if p:
            cleaned.append(p)
    return cleaned[:topk]

def recognize_tokens_in_text(text: str, dict_tokens: Set[str]) -> Set[str]:
    """Which dictionary tokens appear in the explanation text (substring match)."""
    if not isinstance(text, str) or not text.strip():
        return set()
    ht = zenkaku_to_hankaku(text)
    hits = set()
    for tok in dict_tokens:
        if tok and tok in ht:
            hits.add(tok)
    return hits

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return float("nan")
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# ---------------------- core ----------------------
def compute_for_two_outputs(
    df: pd.DataFrame,
    col_entities: int,
    col_gold: int,
    col_free_out: int,    # G
    col_struct_out: int,  # H
    topk: int = 3,
    synonyms_json: Optional[str] = None,
    out_prefix: Optional[str] = None,
):
    # synonyms
    user_syns = None
    if synonyms_json:
        with open(synonyms_json, "r", encoding="utf-8") as f:
            user_syns = json.load(f)
    syn2canon = build_synonym_canon(user_syns)

    # gold
    gold = df.iloc[:, col_gold].astype(str).apply(lambda x: canonize_label(x, syn2canon))

    # entities (from C)
    ent_list = df.iloc[:, col_entities].apply(split_entities)
    # dictionary (union)
    dict_tokens: Set[str] = set()
    for lst in ent_list:
        dict_tokens.update(lst)

    def compute_one(output_col: int, label: str):
        preds_top3 = df.iloc[:, output_col].apply(lambda x: extract_top_diagnoses(x, topk))
        top1_hits, top3_hits, consistency_vals, expl_sets = [], [], [], []
        for i in range(len(df)):
            preds = preds_top3.iloc[i] if isinstance(preds_top3.iloc[i], list) else []
            g = gold.iloc[i]
            # Top-1/Top-3
            if preds:
                top1 = int(canonize_label(preds[0], syn2canon) == g)
                top3 = int(g in [canonize_label(p, syn2canon) for p in preds[:topk]])
            else:
                top1, top3 = 0, 0
            top1_hits.append(top1)
            top3_hits.append(top3)
            # Consistency
            expl_tokens = recognize_tokens_in_text(df.iloc[i, output_col], dict_tokens)
            expl_sets.append(expl_tokens)
            ents = set(ent_list.iloc[i])
            if len(expl_tokens) == 0:
                consistency_vals.append(np.nan)
            else:
                consistency_vals.append(len(expl_tokens & ents) / len(expl_tokens))
        detail = pd.DataFrame({
            "preds": preds_top3,
            "Top1_hit": top1_hits,
            "Top3_hit": top3_hits,
            "entities": ent_list,
            "expl_tokens": expl_sets,
            "Consistency": consistency_vals,
        })
        summary = {
            "View": label,
            "N": len(df),
            "Top1": float(np.mean(top1_hits)) if len(df) else float("nan"),
            "Top3": float(np.mean(top3_hits)) if len(df) else float("nan"),
            "Consistency": float(np.nanmean(consistency_vals)) if len(df) else float("nan"),
        }
        return detail, summary

    # compute both
    detail_free, sum_free = compute_one(col_free_out, "FreeText(G)")
    detail_struct, sum_struct = compute_one(col_struct_out, "Structured(H)")

    # reproducibility (Jaccard between FreeText vs Structured explanations)
    j_vals = []
    for i in range(len(df)):
        a = set(detail_free.loc[i, "expl_tokens"])
        b = set(detail_struct.loc[i, "expl_tokens"])
        j_vals.append(jaccard(a, b))
    repro = pd.DataFrame([{
        "OutputA": "FreeText(G)",
        "OutputB": "Structured(H)",
        "Jaccard": float(np.nanmean(j_vals)) if len(j_vals) else float("nan")
    }])

    summary_df = pd.DataFrame([sum_free, sum_struct])

    # save
    if out_prefix:
        summary_df.to_csv(f"{out_prefix}_summary.csv", index=False, encoding="utf-8-sig")
        repro.to_csv(f"{out_prefix}_reproducibility.csv", index=False, encoding="utf-8-sig")
        detail_free.to_csv(f"{out_prefix}_per_case_FreeText.csv", index=False, encoding="utf-8-sig")
        detail_struct.to_csv(f"{out_prefix}_per_case_Structured.csv", index=False, encoding="utf-8-sig")
    return summary_df, detail_free, detail_struct, repro

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser(description="Compute metrics for FreeText(G) vs Structured(H) using column LETTERS.")
    ap.add_argument("--input", required=True, help="Excel path (.xlsx)")
    ap.add_argument("--sheet", default=None, help="Sheet name (default: first)")
    ap.add_argument("--header", choices=["none", "row0"], default="none",
                    help="Whether the first row is header. Default none (no header).")
    ap.add_argument("--col-entities", default="C", help="Column letter for structured entities (default C)")
    ap.add_argument("--col-gold",     default="E", help="Column letter for gold label (default E)")
    ap.add_argument("--col-free-out", default="G", help="Column letter for free-text output (default G)")
    ap.add_argument("--col-struct-out", default="H", help="Column letter for structured-text output (default H)")
    ap.add_argument("--topk", type=int, default=3, help="Top-k for coverage (default 3)")
    ap.add_argument("--synonyms-json", default=None, help="Optional synonyms JSON (canon -> [synonyms])")
    ap.add_argument("--out-prefix", default=None, help="Output prefix for CSVs")
    args = ap.parse_args()

    xls = pd.ExcelFile(args.input)
    sheet_name = args.sheet if args.sheet else xls.sheet_names[0]
    header_opt = 0 if args.header == "row0" else None
    df = pd.read_excel(args.input, sheet_name=sheet_name, header=header_opt)

    c_entities   = col_letter_to_idx(args.col_entities)
    c_gold       = col_letter_to_idx(args.col_gold)
    c_free_out   = col_letter_to_idx(args.col_free_out)
    c_struct_out = col_letter_to_idx(args.col_struct_out)

    summary_df, detail_free, detail_struct, repro = compute_for_two_outputs(
        df=df,
        col_entities=c_entities,
        col_gold=c_gold,
        col_free_out=c_free_out,
        col_struct_out=c_struct_out,
        topk=args.topk,
        synonyms_json=args.synonyms_json,
        out_prefix=args.out_prefix
    )

    pd.set_option("display.max_colwidth", 120)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== Reproducibility (Jaccard FreeText vs Structured) ===")
    print(repro.to_string(index=False))

if __name__ == "__main__":
    main()
