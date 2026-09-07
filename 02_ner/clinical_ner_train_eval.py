#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clinical Japanese NER training, validation, and figure generation.

Purpose
-------
This script is for model development:
  1. Load one or more Label Studio JSON exports.
  2. Merge annotations.
  3. Split documents into training / validation sets.
  4. Fine-tune a Hugging Face token-classification model.
  5. Evaluate on the held-out validation set.
  6. Export common figures and CSV/JSON reports.

Typical command, PowerShell:

  python clinical_ner_train_eval.py train ^
    --train_json project-10-at-2025-08-19-16-32-0033594e.json project-13-at-2025-10-03-16-39-1e3606a8.json ^
    --output_dir trained_clinical_ner_xlmr ^
    --num_train_docs 350 ^
    --epochs 10 ^
    --batch_size 8 ^
    --fp16 ^
    --tensorboard

Outputs
-------
Inside output_dir:
  - trained model files
  - training_metadata.json
  - validation_predictions.jsonl
  - validation_entity_spans.xlsx
  - figures/01_train_loss.png
  - figures/02_eval_loss.png
  - figures/03_eval_metrics.png
  - figures/04_per_label_f1.png
  - figures/05_confusion_matrix_entity_type.png
  - figures/06_precision_recall_micro.png
  - figures/07_roc_micro.png
  - figures/per_label_report.csv
  - figures/eval_summary.json
  - figures/pr_roc_summary.json
  - runs/   if --tensorboard is enabled
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence as HFSequence, Value
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


# =============================================================================
# Label normalization
# =============================================================================

LABEL_MAP: Dict[str, str] = {
    "症状・徴候": "SYMPTOM",
    "否定詞": "NEGATION",
    "具体的な時間": "TIME",
    "持続時間": "DURATION",
    "症状の重症度": "SEVERITY",
    "診断・検査": "TEST",
    "検査値・検査結果": "TEST_FINDING",
    "部位": "ANATOMY",
    "疾患・障害": "DISEASE",
    "薬剤名": "MEDICATION",
    "臨床イベント": "CLINICAL_EVENT",
    "数値": "VALUE",
    "単位": "UNIT",
    "投与量": "DOSAGE",
    "投与方法": "ROUTE",
    "画像所見": "IMAGING_FINDING",
    "頻度": "FREQUENCY",
    "個人歴": "PERSONAL_HISTORY",
}

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


def normalize_label(label: str) -> str:
    """Convert a Label Studio label to a stable short label name."""
    if label is None:
        return "UNKNOWN"
    s = str(label).replace("\u3000", " ").strip()
    base = re.split(r"\s*例[:：]", s, maxsplit=1)[0].strip()
    key = base.split()[0] if base.split() else base
    return LABEL_MAP.get(key, key)


def make_json_safe(x: Any) -> Any:
    """Convert pandas/numpy/datetime values into JSON-safe values."""
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


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =============================================================================
# Label Studio loading
# =============================================================================

@dataclass
class DocumentExample:
    doc_id: str
    text: str
    spans: List[Tuple[int, int, str]]
    meta: Dict[str, Any]


def select_annotation(item: Dict[str, Any], mode: str = "latest") -> Optional[Dict[str, Any]]:
    annotations = item.get("annotations") or []
    annotations = [a for a in annotations if not a.get("was_cancelled", False)]
    if not annotations:
        return None
    if mode == "first":
        return annotations[0]
    return sorted(annotations, key=lambda a: a.get("updated_at") or a.get("created_at") or "")[-1]


def extract_spans_from_labelstudio_item(
    item: Dict[str, Any],
    annotation_mode: str = "latest",
    check_offsets: bool = True,
) -> List[Tuple[int, int, str]]:
    text = item.get("data", {}).get("text", "") or ""
    ann = select_annotation(item, mode=annotation_mode)
    if ann is None:
        return []

    spans: List[Tuple[int, int, str]] = []
    for r in ann.get("result") or []:
        if r.get("type") != "labels":
            continue
        v = r.get("value") or {}
        start = v.get("start")
        end = v.get("end")
        labels = v.get("labels") or []
        if start is None or end is None or not labels:
            continue

        s = int(start)
        e = int(end)
        if s < 0 or e > len(text) or s >= e:
            warnings.warn(
                f"Invalid span skipped: task={item.get('id')} start={s} end={e} len={len(text)}"
            )
            continue

        if check_offsets and v.get("text") is not None:
            label_text = str(v.get("text"))
            actual_text = text[s:e]
            if actual_text != label_text:
                warnings.warn(
                    "Offset mismatch, but span is kept because Label Studio offsets are treated as source of truth. "
                    f"task={item.get('id')} span=({s},{e}) label_text={label_text!r} actual={actual_text!r}"
                )

        spans.append((s, e, normalize_label(labels[0])))
    return spans


def load_labelstudio_documents(
    paths: Sequence[str],
    annotation_mode: str = "latest",
    check_offsets: bool = True,
    keep_empty_span_docs: bool = False,
) -> List[DocumentExample]:
    docs: List[DocumentExample] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"Label Studio export must be a list of tasks: {path}")

        for i, item in enumerate(raw):
            data = item.get("data") or {}
            text = data.get("text", "") or ""
            if not text:
                continue
            spans = extract_spans_from_labelstudio_item(
                item,
                annotation_mode=annotation_mode,
                check_offsets=check_offsets,
            )
            if not keep_empty_span_docs and not spans:
                continue
            task_id = item.get("id")
            doc_id = str(task_id or f"{Path(path).stem}_{i}")
            meta = {
                "source_file": Path(path).name,
                "task_id": task_id,
                "patient_id": data.get("patient_id"),
                "date": data.get("date"),
            }
            docs.append(DocumentExample(doc_id=doc_id, text=text, spans=spans, meta=meta))
    return docs


# =============================================================================
# BIO conversion and chunking
# =============================================================================

def spans_to_char_bio(text: str, spans: Sequence[Tuple[int, int, str]]) -> List[str]:
    tags = ["O"] * len(text)
    skipped = 0
    for s, e, lab in sorted(spans, key=lambda x: (x[0], x[1])):
        s = max(0, int(s))
        e = min(len(text), int(e))
        if s >= e:
            continue
        if any(tags[pos] != "O" for pos in range(s, e)):
            skipped += 1
            continue
        tags[s] = f"B-{lab}"
        for pos in range(s + 1, e):
            tags[pos] = f"I-{lab}"
    if skipped:
        warnings.warn(f"Skipped {skipped} overlapping spans.")
    return tags


def make_bio_label_list(docs: Sequence[DocumentExample]) -> List[str]:
    base_labels = sorted({lab for d in docs for _, _, lab in d.spans})
    return ["O"] + [f"B-{b}" for b in base_labels] + [f"I-{b}" for b in base_labels]


def split_document_to_chunks(
    doc: DocumentExample,
    chunk_size: int = 450,
    stride: int = 80,
) -> List[Dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if stride < 0 or stride >= chunk_size:
        raise ValueError("stride must satisfy 0 <= stride < chunk_size.")

    text = doc.text
    tags = spans_to_char_bio(text, doc.spans)
    rows: List[Dict[str, Any]] = []
    step = chunk_size - stride
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        rows.append({
            "tokens": list(text[start:end]),
            "ner_tags": tags[start:end],
            "doc_id": doc.doc_id,
            "chunk_start": start,
            "chunk_end": end,
        })
        if end == len(text):
            break
        start += step
    return rows


def build_hf_dataset_from_docs(
    docs: Sequence[DocumentExample],
    label_list: Sequence[str],
    chunk_size: int,
    stride: int,
) -> Dataset:
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        rows.extend(split_document_to_chunks(doc, chunk_size=chunk_size, stride=stride))
    if not rows:
        raise ValueError("No rows were generated. Check annotations and chunk settings.")

    features = Features({
        "tokens": HFSequence(Value("string")),
        "ner_tags": HFSequence(ClassLabel(names=list(label_list))),
        "doc_id": Value("string"),
        "chunk_start": Value("int32"),
        "chunk_end": Value("int32"),
    })
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0].keys()}, features=features)


# =============================================================================
# Train / validation split
# =============================================================================

def split_docs_train_val_ratio(
    docs: Sequence[DocumentExample],
    test_size: float,
    seed: int,
) -> Tuple[List[DocumentExample], List[DocumentExample]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    indices = list(range(len(docs)))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)
    return [docs[i] for i in train_idx], [docs[i] for i in val_idx]


def split_docs_train_val_count(
    docs: Sequence[DocumentExample],
    num_train_docs: int,
    num_val_docs: Optional[int],
    seed: int,
    max_docs: Optional[int] = None,
) -> Tuple[List[DocumentExample], List[DocumentExample]]:
    if num_train_docs <= 0:
        raise ValueError("--num_train_docs must be positive.")
    docs_list = list(docs)
    rng = random.Random(seed)
    rng.shuffle(docs_list)

    if max_docs is not None:
        if max_docs <= 0:
            raise ValueError("--max_docs must be positive.")
        docs_list = docs_list[:max_docs]

    total = len(docs_list)
    if num_train_docs >= total:
        raise ValueError(
            f"--num_train_docs={num_train_docs} is too large. Available annotated documents: {total}."
        )
    if num_val_docs is None:
        num_val_docs = total - num_train_docs
    else:
        if num_val_docs <= 0:
            raise ValueError("--num_val_docs must be positive when specified.")
        if num_train_docs + num_val_docs > total:
            raise ValueError(
                f"Requested {num_train_docs}+{num_val_docs} documents, but only {total} are available."
            )
    train_docs = docs_list[:num_train_docs]
    val_docs = docs_list[num_train_docs:num_train_docs + num_val_docs]
    if not train_docs or not val_docs:
        raise ValueError("Train or validation split is empty.")
    return train_docs, val_docs


def save_split_manifest(output_dir: str | Path, train_docs: Sequence[DocumentExample], val_docs: Sequence[DocumentExample]) -> None:
    rows = []
    for split_name, docs in [("train", train_docs), ("validation", val_docs)]:
        for d in docs:
            rows.append({
                "split": split_name,
                "doc_id": d.doc_id,
                "patient_id": d.meta.get("patient_id"),
                "date": d.meta.get("date"),
                "source_file": d.meta.get("source_file"),
                "task_id": d.meta.get("task_id"),
                "text_len": len(d.text),
                "span_count": len(d.spans),
            })
    pd.DataFrame(rows).to_csv(Path(output_dir) / "split_manifest.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# Tokenization and training metrics
# =============================================================================

def tokenize_and_align_dataset(
    dataset_dict: DatasetDict,
    tokenizer: Any,
    label_list: Sequence[str],
    max_length: int = 512,
    label_all_tokens: bool = False,
) -> DatasetDict:
    label2id = {label: i for i, label in enumerate(label_list)}

    def to_id(tag_or_id: Any) -> int:
        if isinstance(tag_or_id, str):
            return label2id[tag_or_id]
        return int(tag_or_id)

    def tokenize_and_align_labels(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        all_label_ids = []
        for i, char_labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(to_id(char_labels[word_id]))
                else:
                    label_ids.append(to_id(char_labels[word_id]) if label_all_tokens else -100)
                previous_word_id = word_id
            all_label_ids.append(label_ids)
        tokenized["labels"] = all_label_ids
        return tokenized

    remove_cols = [
        c for c in dataset_dict["train"].column_names
        if c in {"tokens", "ner_tags", "doc_id", "chunk_start", "chunk_end"}
    ]
    return dataset_dict.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=128,
        remove_columns=remove_cols,
        desc="Tokenize and align labels",
    )


def build_compute_metrics(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        true_predictions: List[List[str]] = []
        true_labels: List[List[str]] = []
        for pred_seq, label_seq in zip(predictions, labels):
            pred_labels: List[str] = []
            gold_labels: List[str] = []
            for pred_id, gold_id in zip(pred_seq, label_seq):
                if int(gold_id) == -100:
                    continue
                pred_labels.append(id2label[int(pred_id)])
                gold_labels.append(id2label[int(gold_id)])
            true_predictions.append(pred_labels)
            true_labels.append(gold_labels)
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }
    return compute_metrics


def make_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    report_to: Any = ["tensorboard"] if args.tensorboard else "none"
    kwargs: Dict[str, Any] = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=report_to,
        logging_dir=str(Path(args.output_dir) / "runs"),
        push_to_hub=False,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


# =============================================================================
# Prediction utilities for validation figures
# =============================================================================

def iter_char_windows(text: str, chunk_size: int, stride: int) -> Iterable[Tuple[int, str]]:
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


def predict_char_tags_and_probs(
    text: str,
    tokenizer: Any,
    model: Any,
    id2label: Dict[int, str],
    device: str,
    chunk_size: int,
    stride: int,
    max_length: int,
) -> Tuple[List[str], np.ndarray]:
    num_labels = len(id2label)
    text = text or ""
    if not text:
        return [], np.zeros((0, num_labels), dtype=np.float32)

    best_scores = np.zeros(len(text), dtype=np.float32)
    best_probs = np.zeros((len(text), num_labels), dtype=np.float32)

    for global_start, chunk in iter_char_windows(text, chunk_size, stride):
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
            score = float(np.max(probs[token_idx]))
            if score >= best_scores[pos]:
                best_scores[pos] = score
                best_probs[pos, :] = probs[token_idx, :]

    pred_ids = np.argmax(best_probs, axis=1)
    return [id2label[int(i)] for i in pred_ids], best_probs


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


# =============================================================================
# Figure and report generation
# =============================================================================

def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(model_dir: str | Path, fig_dir: Path) -> None:
    state_path = Path(model_dir) / "trainer_state.json"
    if not state_path.exists():
        print(f"trainer_state.json not found, skip training curves: {state_path}")
        return
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    logs = pd.DataFrame(state.get("log_history", []))
    if logs.empty:
        print("No log_history found in trainer_state.json.")
        return

    if "loss" in logs.columns:
        train_logs = logs[logs["loss"].notna()].copy()
        if not train_logs.empty:
            x = train_logs["step"] if "step" in train_logs.columns else range(len(train_logs))
            plt.figure(figsize=(8, 5))
            plt.plot(x, train_logs["loss"], marker="o", linewidth=1)
            plt.xlabel("Training step")
            plt.ylabel("Training loss")
            plt.title("Training loss curve")
            plt.grid(True, alpha=0.3)
            save_fig(fig_dir / "01_train_loss.png")
            train_logs.to_csv(fig_dir / "train_loss_log.csv", index=False, encoding="utf-8-sig")

    if "eval_loss" in logs.columns:
        eval_logs = logs[logs["eval_loss"].notna()].copy()
        if not eval_logs.empty:
            x = eval_logs["epoch"] if "epoch" in eval_logs.columns else range(len(eval_logs))
            plt.figure(figsize=(8, 5))
            plt.plot(x, eval_logs["eval_loss"], marker="o", linewidth=1)
            plt.xlabel("Epoch")
            plt.ylabel("Validation loss")
            plt.title("Validation loss curve")
            plt.grid(True, alpha=0.3)
            save_fig(fig_dir / "02_eval_loss.png")
            eval_logs.to_csv(fig_dir / "eval_log.csv", index=False, encoding="utf-8-sig")

            metric_cols = [
                c for c in ["eval_precision", "eval_recall", "eval_f1", "eval_accuracy"]
                if c in eval_logs.columns
            ]
            if metric_cols:
                plt.figure(figsize=(8, 5))
                for c in metric_cols:
                    plt.plot(x, eval_logs[c], marker="o", linewidth=1, label=c.replace("eval_", ""))
                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.title("Validation metrics by epoch")
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                plt.legend()
                save_fig(fig_dir / "03_eval_metrics.png")


def simplify_bio_tag(tag: str) -> str:
    if tag == "O":
        return "O"
    return tag.split("-", 1)[1] if "-" in tag else tag


def plot_per_label_f1(y_true_docs, y_pred_docs, fig_dir: Path) -> pd.DataFrame:
    report = classification_report(y_true_docs, y_pred_docs, output_dict=True, zero_division=0)
    rows = []
    for label, values in report.items():
        if label in {"micro avg", "macro avg", "weighted avg"}:
            continue
        if not isinstance(values, dict):
            continue
        rows.append({
            "label": label,
            "precision": values.get("precision", 0),
            "recall": values.get("recall", 0),
            "f1": values.get("f1-score", 0),
            "support": values.get("support", 0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("f1", ascending=True)
    df.to_csv(fig_dir / "per_label_report.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, max(5, 0.35 * len(df))))
    plt.barh(df["label"], df["f1"])
    plt.xlabel("F1-score")
    plt.ylabel("Entity label")
    plt.title("Per-label F1-score")
    plt.xlim(0, 1)
    plt.grid(True, axis="x", alpha=0.3)
    save_fig(fig_dir / "04_per_label_f1.png")
    return df


def plot_entity_type_confusion_matrix(y_true_docs, y_pred_docs, fig_dir: Path) -> None:
    true_types = []
    pred_types = []
    for gold_seq, pred_seq in zip(y_true_docs, y_pred_docs):
        for g, p in zip(gold_seq, pred_seq):
            true_types.append(simplify_bio_tag(g))
            pred_types.append(simplify_bio_tag(p))
    labels = sorted(set(true_types) | set(pred_types))
    labels = ["O"] + [x for x in labels if x != "O"]
    cm = confusion_matrix(true_types, pred_types, labels=labels)
    cm_norm = cm.astype(float)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sum, out=np.zeros_like(cm_norm), where=row_sum != 0)

    plt.figure(figsize=(max(8, 0.5 * len(labels)), max(6, 0.5 * len(labels))))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title("Normalized confusion matrix by entity type")
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    save_fig(fig_dir / "05_confusion_matrix_entity_type.png")

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        fig_dir / "confusion_matrix_entity_type_raw.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(
        fig_dir / "confusion_matrix_entity_type_normalized.csv", encoding="utf-8-sig"
    )


def plot_pr_and_roc(y_true_ids: np.ndarray, y_prob: np.ndarray, id2label: Dict[int, str], fig_dir: Path) -> None:
    if len(y_true_ids) == 0:
        print("No validation labels for PR/ROC.")
        return
    entity_ids = [i for i, lab in id2label.items() if lab != "O"]
    if not entity_ids:
        print("No entity labels found for PR/ROC.")
        return

    all_classes = list(range(len(id2label)))
    y_true_bin = label_binarize(y_true_ids, classes=all_classes)
    y_true_entity = y_true_bin[:, entity_ids]
    y_score_entity = y_prob[:, entity_ids]
    if y_true_entity.sum() == 0:
        print("No positive entity labels in validation set for PR/ROC.")
        return

    precision, recall, _ = precision_recall_curve(y_true_entity.ravel(), y_score_entity.ravel())
    ap = average_precision_score(y_true_entity, y_score_entity, average="micro")
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Micro-average Precision-Recall curve, AP = {ap:.3f}")
    plt.grid(True, alpha=0.3)
    save_fig(fig_dir / "06_precision_recall_micro.png")

    fpr, tpr, _ = roc_curve(y_true_entity.ravel(), y_score_entity.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=1, label=f"micro-average AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Micro-average ROC curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    save_fig(fig_dir / "07_roc_micro.png")

    save_json(fig_dir / "pr_roc_summary.json", {
        "micro_average_precision": float(ap),
        "micro_roc_auc": float(roc_auc),
    })


def evaluate_and_plot_validation(
    model_dir: str | Path,
    val_docs: Sequence[DocumentExample],
    fig_dir: str | Path,
    output_dir: str | Path,
    chunk_size: int,
    stride: int,
    max_length: int,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    fig_dir = ensure_dir(fig_dir)
    output_dir = ensure_dir(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    y_true_docs: List[List[str]] = []
    y_pred_docs: List[List[str]] = []
    all_true_ids: List[int] = []
    all_prob_mats: List[np.ndarray] = []
    jsonl_rows: List[Dict[str, Any]] = []
    span_rows: List[Dict[str, Any]] = []

    for i, doc in enumerate(val_docs, start=1):
        gold_tags = spans_to_char_bio(doc.text, doc.spans)
        pred_tags, prob_mat = predict_char_tags_and_probs(
            text=doc.text,
            tokenizer=tokenizer,
            model=model,
            id2label=id2label,
            device=device,
            chunk_size=chunk_size,
            stride=stride,
            max_length=max_length,
        )
        min_len = min(len(gold_tags), len(pred_tags), len(prob_mat))
        gold_tags = gold_tags[:min_len]
        pred_tags = pred_tags[:min_len]
        prob_mat = prob_mat[:min_len]

        y_true_docs.append(gold_tags)
        y_pred_docs.append(pred_tags)
        all_true_ids.extend([label2id[t] for t in gold_tags])
        all_prob_mats.append(prob_mat)

        char_scores = prob_mat.max(axis=1).tolist() if len(prob_mat) else []
        pred_spans = bio_tags_to_spans(doc.text, pred_tags, char_scores)
        jsonl_rows.append({
            "doc_id": doc.doc_id,
            "metadata": {k: make_json_safe(v) for k, v in doc.meta.items()},
            "text": doc.text,
            "predicted_entities": pred_spans,
        })
        for sp in pred_spans:
            span_rows.append({"doc_id": doc.doc_id, **doc.meta, **sp})

        if i % 10 == 0:
            print(f"Validated documents: {i}/{len(val_docs)}")

    y_true_ids = np.array(all_true_ids, dtype=np.int64)
    y_prob = np.vstack(all_prob_mats) if all_prob_mats else np.zeros((0, len(id2label)))

    summary = {
        "seqeval_precision": float(precision_score(y_true_docs, y_pred_docs)),
        "seqeval_recall": float(recall_score(y_true_docs, y_pred_docs)),
        "seqeval_f1": float(f1_score(y_true_docs, y_pred_docs)),
        "seqeval_accuracy": float(accuracy_score(y_true_docs, y_pred_docs)),
        "num_validation_documents": len(val_docs),
    }
    save_json(fig_dir / "eval_summary.json", summary)
    with open(fig_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_true_docs, y_pred_docs, digits=4))

    write_jsonl(output_dir / "validation_predictions.jsonl", jsonl_rows)
    pd.DataFrame(span_rows).to_excel(output_dir / "validation_entity_spans.xlsx", index=False)

    plot_training_curves(model_dir, fig_dir)
    plot_per_label_f1(y_true_docs, y_pred_docs, fig_dir)
    plot_entity_type_confusion_matrix(y_true_docs, y_pred_docs, fig_dir)
    plot_pr_and_roc(y_true_ids, y_prob, id2label, fig_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# =============================================================================
# Train command
# =============================================================================

def prepare_train_val_docs(args: argparse.Namespace) -> Tuple[List[DocumentExample], List[DocumentExample], List[DocumentExample], int]:
    docs_all = []
    for path in args.train_json:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            docs_all.extend(raw)
    docs_loaded_total = len(docs_all)

    docs = load_labelstudio_documents(
        args.train_json,
        annotation_mode=args.annotation_mode,
        check_offsets=True,
        keep_empty_span_docs=args.keep_empty_span_docs,
    )
    if len(docs) < 2:
        raise ValueError(
            f"Need at least two documents for train/validation split. Loaded usable docs={len(docs)}."
        )

    if args.num_train_docs is not None:
        train_docs, val_docs = split_docs_train_val_count(
            docs=docs,
            num_train_docs=args.num_train_docs,
            num_val_docs=args.num_val_docs,
            seed=args.seed,
            max_docs=args.max_docs,
        )
    else:
        docs_for_split = list(docs)
        if args.max_docs is not None:
            rng = random.Random(args.seed)
            rng.shuffle(docs_for_split)
            docs_for_split = docs_for_split[:args.max_docs]
        train_docs, val_docs = split_docs_train_val_ratio(
            docs_for_split,
            test_size=args.test_size,
            seed=args.seed,
        )
    return docs, train_docs, val_docs, docs_loaded_total


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    docs, train_docs, val_docs, docs_loaded_total = prepare_train_val_docs(args)
    label_list = make_bio_label_list(docs)
    if len(label_list) <= 1:
        raise ValueError("No entity labels found. Check Label Studio annotations.")

    print("========== Data split ==========")
    print(f"Loaded Label Studio tasks       : {docs_loaded_total}")
    print(f"Usable documents                : {len(docs)}")
    print(f"Train documents                 : {len(train_docs)}")
    print(f"Validation documents            : {len(val_docs)}")
    print(f"Labels                          : {label_list}")
    print("================================")

    save_split_manifest(args.output_dir, train_docs, val_docs)

    train_ds = build_hf_dataset_from_docs(train_docs, label_list, args.chunk_size, args.stride)
    val_ds = build_hf_dataset_from_docs(val_docs, label_list, args.chunk_size, args.stride)
    ds = DatasetDict({"train": train_ds, "validation": val_ds})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("This script requires a fast tokenizer because word_ids are used.")

    tokenized_ds = tokenize_and_align_dataset(
        ds,
        tokenizer=tokenizer,
        label_list=label_list,
        max_length=args.max_length,
        label_all_tokens=args.label_all_tokens,
    )

    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in id2label.items()}
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, config=config)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = make_training_arguments(args)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(id2label),
    )
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "script": "clinical_ner_train_eval.py",
        "model_name": args.model_name,
        "train_json": [str(p) for p in args.train_json],
        "label_list": label_list,
        "label_map": LABEL_MAP,
        "chunk_size": args.chunk_size,
        "stride": args.stride,
        "max_length": args.max_length,
        "seed": args.seed,
        "documents_loaded_total": docs_loaded_total,
        "usable_documents": len(docs),
        "num_train_documents": len(train_docs),
        "num_validation_documents": len(val_docs),
        "num_train_chunks": len(train_ds),
        "num_validation_chunks": len(val_ds),
        "num_train_docs_arg": args.num_train_docs,
        "num_val_docs_arg": args.num_val_docs,
        "max_docs_arg": args.max_docs,
        "test_size_arg": args.test_size,
        "keep_empty_span_docs": args.keep_empty_span_docs,
        "tensorboard": args.tensorboard,
        "metrics_from_trainer": {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
        },
    }
    save_json(Path(args.output_dir) / "training_metadata.json", meta)

    fig_dir = Path(args.fig_dir) if args.fig_dir else Path(args.output_dir) / "figures"
    summary = evaluate_and_plot_validation(
        model_dir=args.output_dir,
        val_docs=val_docs,
        fig_dir=fig_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        stride=args.stride,
        max_length=args.max_length,
        device=args.device,
    )
    meta["metrics_from_validation_prediction"] = summary
    save_json(Path(args.output_dir) / "training_metadata.json", meta)

    print("Training, validation, and figure generation finished.")
    print(f"Model directory: {args.output_dir}")
    print(f"Figure directory: {fig_dir}")
    if args.tensorboard:
        print(f"TensorBoard: tensorboard --logdir {Path(args.output_dir) / 'runs'}")


# =============================================================================
# Plot-only command
# =============================================================================

def plot_command(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    metadata_path = model_dir / "training_metadata.json"
    meta: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    train_json = args.train_json or meta.get("train_json")
    if not train_json:
        raise ValueError("--train_json is required for plot command when training_metadata.json has no train_json.")

    args.train_json = train_json
    args.output_dir = str(model_dir)
    args.chunk_size = args.chunk_size or int(meta.get("chunk_size", 450))
    args.stride = args.stride or int(meta.get("stride", 80))
    args.max_length = args.max_length or int(meta.get("max_length", 512))
    args.keep_empty_span_docs = bool(meta.get("keep_empty_span_docs", False))

    docs, _train_docs, val_docs, _docs_loaded_total = prepare_train_val_docs(args)
    fig_dir = Path(args.fig_dir) if args.fig_dir else model_dir / "figures"
    evaluate_and_plot_validation(
        model_dir=model_dir,
        val_docs=val_docs,
        fig_dir=fig_dir,
        output_dir=model_dir,
        chunk_size=args.chunk_size,
        stride=args.stride,
        max_length=args.max_length,
        device=args.device,
    )
    print(f"Plot-only evaluation finished. Figure directory: {fig_dir}")


# =============================================================================
# CLI
# =============================================================================

def add_common_split_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--annotation_mode", choices=["latest", "first"], default="latest")
    p.add_argument("--num_train_docs", type=int, default=None)
    p.add_argument("--num_val_docs", type=int, default=None)
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_empty_span_docs", action="store_true", help="Include documents with no spans as all-O training examples.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clinical Japanese NER training, validation, and figures")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train model, validate, and export figures")
    p_train.add_argument("--train_json", nargs="+", required=True)
    p_train.add_argument("--model_name", default="xlm-roberta-base")
    p_train.add_argument("--output_dir", default="trained_clinical_ner")
    p_train.add_argument("--fig_dir", default=None, help="Default: output_dir/figures")
    p_train.add_argument("--epochs", type=float, default=10)
    p_train.add_argument("--batch_size", type=int, default=8)
    p_train.add_argument("--eval_batch_size", type=int, default=None)
    p_train.add_argument("--learning_rate", type=float, default=2e-5)
    p_train.add_argument("--weight_decay", type=float, default=0.01)
    p_train.add_argument("--max_length", type=int, default=512)
    p_train.add_argument("--chunk_size", type=int, default=450)
    p_train.add_argument("--stride", type=int, default=80)
    p_train.add_argument("--label_all_tokens", action="store_true")
    p_train.add_argument("--fp16", action="store_true")
    p_train.add_argument("--tensorboard", action="store_true")
    p_train.add_argument("--logging_steps", type=int, default=20)
    p_train.add_argument("--save_total_limit", type=int, default=2)
    p_train.add_argument("--device", default=None, choices=["cpu", "cuda"])
    add_common_split_args(p_train)
    p_train.set_defaults(func=train_command)

    p_plot = sub.add_parser("plot", help="Regenerate validation figures from an existing trained model")
    p_plot.add_argument("--model_dir", required=True)
    p_plot.add_argument("--train_json", nargs="+", default=None)
    p_plot.add_argument("--fig_dir", default=None, help="Default: model_dir/figures")
    p_plot.add_argument("--chunk_size", type=int, default=None)
    p_plot.add_argument("--stride", type=int, default=None)
    p_plot.add_argument("--max_length", type=int, default=None)
    p_plot.add_argument("--device", default=None, choices=["cpu", "cuda"])
    add_common_split_args(p_plot)
    p_plot.set_defaults(func=plot_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
