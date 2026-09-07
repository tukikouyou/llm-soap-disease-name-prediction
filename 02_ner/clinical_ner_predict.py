#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clinical Japanese NER prediction script.

Purpose
-------
This script is for applying an already trained NER model to new data.
It intentionally does not contain training / validation / plotting logic.

Supported inputs:
  1. CSV / TSV / XLSX table, usually with columns:
       患者番号, オーダ日付, 内容, 病名
  2. Plain text file
  3. Label Studio JSON tasks for pre-annotation

Typical table prediction, PowerShell:

  python clinical_ner_predict.py predict_table ^
    --model_dir trained_clinical_ner_xlmr ^
    --input_table test.xlsx ^
    --text_col 内容 ^
    --patient_col 患者番号 ^
    --date_col オーダ日付 ^
    --disease_col 病名 ^
    --output_xlsx ner_predictions.xlsx ^
    --output_jsonl ner_predictions.jsonl ^
    --labelstudio_json labelstudio_predictions.json
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


LABEL_STUDIO_DISPLAY_LABELS: Dict[str, str] = {
    "SYMPTOM": "症状・徴候　例：咳嗽",
    "NEGATION": "否定詞　例：なし",
    "TIME": "具体的な時間　例：2024年3月",
    "DURATION": "持続時間　例：2週間",
    "SEVERITY": "症状の重症度　例：ひどい",
    "TEST": "診断・検査　例：X線",
    "TEST_FINDING": "検査値・検査結果　例：CRP",
    "ANATOMY": "部位　例：肺",
    "DISEASE": "疾患・障害　例：肺炎",
    "MEDICATION": "薬剤名",
    "CLINICAL_EVENT": "臨床イベント　例：手術",
    "VALUE": "数値　例：38.5",
    "UNIT": "単位　例：mg/dL",
    "DOSAGE": "投与量",
    "ROUTE": "投与方法　例：経鼻",
    "IMAGING_FINDING": "画像所見　例：結節影",
    "FREQUENCY": "頻度",
    "PERSONAL_HISTORY": "個人歴　例：喫煙歴あり",
}


def make_json_safe(x: Any) -> Any:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def clean_clinical_text(text: Any) -> str:
    """Normalize common Excel/XML line break artifacts for prediction."""
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except Exception:
        pass
    s = str(text)
    s = s.replace("_x000D_", "\n")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_training_metadata(model_dir: str | Path) -> Dict[str, Any]:
    path = Path(model_dir) / "training_metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_for_prediction(model_dir: str, device: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("This script requires a fast tokenizer because word_ids are used.")

    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return tokenizer, model, id2label, device


def iter_char_windows(text: str, chunk_size: int, stride: int):
    if not text:
        return
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if stride < 0 or stride >= chunk_size:
        raise ValueError("stride must satisfy 0 <= stride < chunk_size.")
    step = chunk_size - stride
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield start, text[start:end]
        if end == len(text):
            break
        start += step


def predict_char_tags(
    text: str,
    tokenizer: Any,
    model: Any,
    id2label: Dict[int, str],
    device: str,
    chunk_size: int = 450,
    stride: int = 80,
    max_length: int = 512,
) -> Tuple[List[str], List[float]]:
    """
    Predict one BIO tag per character using sliding windows.
    For overlapping windows, keep the prediction with higher max softmax score.
    """
    text = text or ""
    if not text:
        return [], []

    best_label_ids = np.zeros(len(text), dtype=np.int64)
    best_scores = np.zeros(len(text), dtype=np.float32)

    for global_start, chunk in iter_char_windows(text, chunk_size=chunk_size, stride=stride):
        chars = list(chunk)
        encoded = tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids = encoded.word_ids(batch_index=0)
        model_inputs = {
            k: v.to(device)
            for k, v in encoded.items()
            if k in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.no_grad():
            logits = model(**model_inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        used_word_ids = set()
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in used_word_ids:
                continue
            used_word_ids.add(word_id)
            if word_id >= len(chars):
                continue
            pos = global_start + word_id
            if pos >= len(text):
                continue
            label_id = int(np.argmax(probs[token_idx]))
            score = float(np.max(probs[token_idx]))
            if score >= best_scores[pos]:
                best_scores[pos] = score
                best_label_ids[pos] = label_id

    tags = [id2label[int(i)] for i in best_label_ids]
    scores = [float(x) for x in best_scores]
    return tags, scores


def bio_tags_to_spans(
    text: str,
    tags: Sequence[str],
    scores: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    start: Optional[int] = None
    current_label: Optional[str] = None
    current_scores: List[float] = []

    def close(end: int) -> None:
        nonlocal start, current_label, current_scores
        if start is not None and current_label is not None and start < end:
            spans.append({
                "start": start,
                "end": end,
                "text": text[start:end],
                "label": current_label,
                "confidence": float(np.mean(current_scores)) if current_scores else None,
            })
        start = None
        current_label = None
        current_scores = []

    for i, tag in enumerate(tags):
        score = scores[i] if scores is not None and i < len(scores) else None
        if tag == "O" or tag is None or "-" not in tag:
            close(i)
            continue
        prefix, lab = tag.split("-", 1)
        if prefix == "B" or current_label != lab:
            close(i)
            start = i
            current_label = lab
            current_scores = [score] if score is not None else []
        else:
            if score is not None:
                current_scores.append(score)
    close(len(tags))
    return spans


def predict_text(
    text: str,
    tokenizer: Any,
    model: Any,
    id2label: Dict[int, str],
    device: str,
    chunk_size: int,
    stride: int,
    max_length: int,
    min_confidence: Optional[float] = None,
) -> List[Dict[str, Any]]:
    tags, scores = predict_char_tags(
        text=text,
        tokenizer=tokenizer,
        model=model,
        id2label=id2label,
        device=device,
        chunk_size=chunk_size,
        stride=stride,
        max_length=max_length,
    )
    spans = bio_tags_to_spans(text, tags, scores)
    if min_confidence is not None:
        spans = [sp for sp in spans if sp.get("confidence") is None or sp["confidence"] >= min_confidence]
    return spans


def spans_to_labelstudio_result(
    spans: Sequence[Dict[str, Any]],
    from_name: str = "label",
    to_name: str = "text",
    use_original_label_names: bool = True,
) -> List[Dict[str, Any]]:
    results = []
    for sp in spans:
        label = str(sp["label"])
        if use_original_label_names:
            label = LABEL_STUDIO_DISPLAY_LABELS.get(label, label)
        results.append({
            "id": uuid.uuid4().hex[:10],
            "from_name": from_name,
            "to_name": to_name,
            "type": "labels",
            "value": {
                "start": int(sp["start"]),
                "end": int(sp["end"]),
                "text": sp["text"],
                "labels": [label],
            },
            "score": sp.get("confidence"),
        })
    return results


def resolve_prediction_params(args: argparse.Namespace) -> Tuple[int, int, int]:
    meta = load_training_metadata(args.model_dir)
    chunk_size = args.chunk_size or int(meta.get("chunk_size", 450))
    stride = args.stride or int(meta.get("stride", 80))
    max_length = args.max_length or int(meta.get("max_length", 512))
    return chunk_size, stride, max_length


def predict_table_command(args: argparse.Namespace) -> None:
    chunk_size, stride, max_length = resolve_prediction_params(args)
    tokenizer, model, id2label, device = load_model_for_prediction(args.model_dir, args.device)
    df = read_table(args.input_table)
    if args.text_col not in df.columns:
        raise ValueError(f"text_col={args.text_col!r} not found. Available columns: {list(df.columns)}")

    meta_cols = [c for c in [args.patient_col, args.date_col, args.disease_col] if c and c in df.columns]
    if args.extra_meta_cols:
        for c in args.extra_meta_cols:
            if c in df.columns and c not in meta_cols:
                meta_cols.append(c)

    jsonl_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    labelstudio_tasks: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        text = clean_clinical_text(row[args.text_col])
        spans = predict_text(
            text=text,
            tokenizer=tokenizer,
            model=model,
            id2label=id2label,
            device=device,
            chunk_size=chunk_size,
            stride=stride,
            max_length=max_length,
            min_confidence=args.min_confidence,
        )
        metadata = {c: make_json_safe(row[c]) for c in meta_cols}
        record = {
            "row_index": int(idx),
            "metadata": metadata,
            "text": text,
            "entities": spans,
        }
        jsonl_rows.append(record)
        for sp in spans:
            long_rows.append({"row_index": int(idx), **metadata, **sp})

        if args.labelstudio_json:
            labelstudio_tasks.append({
                "data": {"text": text, **metadata},
                "predictions": [{
                    "model_version": Path(args.model_dir).name,
                    "score": None,
                    "result": spans_to_labelstudio_result(
                        spans,
                        from_name=args.from_name,
                        to_name=args.to_name,
                        use_original_label_names=not args.labelstudio_short_labels,
                    ),
                }],
            })

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, jsonl_rows)
    if args.output_xlsx:
        pd.DataFrame(long_rows).to_excel(args.output_xlsx, index=False)
    if args.labelstudio_json:
        with open(args.labelstudio_json, "w", encoding="utf-8") as f:
            json.dump(labelstudio_tasks, f, ensure_ascii=False, indent=2)

    print(f"Predicted rows: {len(df)}")
    print(f"Extracted entity spans: {len(long_rows)}")
    print(f"chunk_size={chunk_size}, stride={stride}, max_length={max_length}")
    if args.output_xlsx:
        print(f"Excel saved to: {args.output_xlsx}")
    if args.output_jsonl:
        print(f"JSONL saved to: {args.output_jsonl}")
    if args.labelstudio_json:
        print(f"Label Studio prediction JSON saved to: {args.labelstudio_json}")


def predict_text_file_command(args: argparse.Namespace) -> None:
    chunk_size, stride, max_length = resolve_prediction_params(args)
    tokenizer, model, id2label, device = load_model_for_prediction(args.model_dir, args.device)
    with open(args.input_txt, "r", encoding="utf-8") as f:
        text = clean_clinical_text(f.read())
    spans = predict_text(
        text=text,
        tokenizer=tokenizer,
        model=model,
        id2label=id2label,
        device=device,
        chunk_size=chunk_size,
        stride=stride,
        max_length=max_length,
        min_confidence=args.min_confidence,
    )
    result = {"text": text, "entities": spans}
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON saved to: {args.output_json}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def predict_labelstudio_command(args: argparse.Namespace) -> None:
    chunk_size, stride, max_length = resolve_prediction_params(args)
    tokenizer, model, id2label, device = load_model_for_prediction(args.model_dir, args.device)
    with open(args.input_json, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("Label Studio input JSON must be a list of tasks.")

    output_tasks = []
    total_spans = 0
    for task in tasks:
        data = task.get("data") or {}
        text = clean_clinical_text(data.get(args.text_key, "") or data.get("text", ""))
        spans = predict_text(
            text=text,
            tokenizer=tokenizer,
            model=model,
            id2label=id2label,
            device=device,
            chunk_size=chunk_size,
            stride=stride,
            max_length=max_length,
            min_confidence=args.min_confidence,
        )
        total_spans += len(spans)
        new_task = dict(task)
        new_task["predictions"] = [{
            "model_version": Path(args.model_dir).name,
            "score": None,
            "result": spans_to_labelstudio_result(
                spans,
                from_name=args.from_name,
                to_name=args.to_name,
                use_original_label_names=not args.labelstudio_short_labels,
            ),
        }]
        output_tasks.append(new_task)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_tasks, f, ensure_ascii=False, indent=2)
    print(f"Predicted Label Studio tasks: {len(output_tasks)}")
    print(f"Extracted entity spans: {total_spans}")
    print(f"Saved to: {args.output_json}")


def add_prediction_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model_dir", required=True)
    p.add_argument("--max_length", type=int, default=None, help="Default: read from training_metadata.json or 512")
    p.add_argument("--chunk_size", type=int, default=None, help="Default: read from training_metadata.json or 450")
    p.add_argument("--stride", type=int, default=None, help="Default: read from training_metadata.json or 80")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p.add_argument("--min_confidence", type=float, default=None, help="Optional threshold for exported entity spans")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clinical Japanese NER prediction")
    sub = parser.add_subparsers(dest="command", required=True)

    p_table = sub.add_parser("predict_table", help="Predict entities for CSV/TSV/XLSX table")
    add_prediction_common_args(p_table)
    p_table.add_argument("--input_table", required=True)
    p_table.add_argument("--text_col", default="内容")
    p_table.add_argument("--patient_col", default="患者番号")
    p_table.add_argument("--date_col", default="オーダ日付")
    p_table.add_argument("--disease_col", default="病名")
    p_table.add_argument("--extra_meta_cols", nargs="*", default=None)
    p_table.add_argument("--output_xlsx", default="ner_predictions.xlsx")
    p_table.add_argument("--output_jsonl", default="ner_predictions.jsonl")
    p_table.add_argument("--labelstudio_json", default=None)
    p_table.add_argument("--from_name", default="label")
    p_table.add_argument("--to_name", default="text")
    p_table.add_argument("--labelstudio_short_labels", action="store_true")
    p_table.set_defaults(func=predict_table_command)

    p_text = sub.add_parser("predict_text", help="Predict entities for a plain UTF-8 text file")
    add_prediction_common_args(p_text)
    p_text.add_argument("--input_txt", required=True)
    p_text.add_argument("--output_json", default=None)
    p_text.set_defaults(func=predict_text_file_command)

    p_ls = sub.add_parser("predict_labelstudio", help="Add model predictions to Label Studio JSON tasks")
    add_prediction_common_args(p_ls)
    p_ls.add_argument("--input_json", required=True)
    p_ls.add_argument("--output_json", required=True)
    p_ls.add_argument("--text_key", default="text")
    p_ls.add_argument("--from_name", default="label")
    p_ls.add_argument("--to_name", default="text")
    p_ls.add_argument("--labelstudio_short_labels", action="store_true")
    p_ls.set_defaults(func=predict_labelstudio_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
