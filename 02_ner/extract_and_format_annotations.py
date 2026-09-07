# Convert a Label Studio JSON export into Excel tables of annotated spans.
#
# Requires: pip install pandas openpyxl
#
# Usage:
#   python extract_and_format_annotations.py --input <label_studio_export.json> --out-dir . [--patient-key patient_id]

import argparse
import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd


def parse_dt(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def format_text(text: str) -> str:
    text = re.sub(r"[、,，]+", "　", text)
    text = re.sub(r"[。．]+", "\n", text)
    text = re.sub(r"(<[^>]+>)", r"\n\1\n", text)
    text = re.sub(r"(?<=\S):(?=\S)", ": ", text)
    return text.strip()


def pick_latest_annotation_block(anns_blocks):
    if not anns_blocks:
        return None
    candidates = [a for a in anns_blocks if a.get("result")]
    if not candidates:
        return None
    latest = max(
        candidates,
        key=lambda a: (parse_dt(a.get("updated_at")) or
                       parse_dt(a.get("created_at")) or
                       datetime.min)
    )
    return latest


def robust_sort_key(span):
    s = int(span["value"]["start"])
    e = int(span["value"]["end"])
    return (s, e, -(e - s))


def pick_patient_id(item: dict, preferred_key: str | None = None) -> str | None:
    """Best-effort lookup of a patient_id in data, meta or the note text."""
    data = item.get("data", {}) or {}
    meta = item.get("meta", {}) or {}

    if preferred_key:
        v = data.get(preferred_key)
        if v is not None and str(v).strip():
            return str(v).strip()

    # Key spellings seen across exports.
    candidates = [
        "patient_id", "PatientID", "patientId", "pid", "PID",
        "患者ID", "患者番号",
        "case_id", "CaseID",
        "subject_id", "SUBJECT_ID",
        "study_id", "StudyID"
    ]
    for k in candidates:
        v = data.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()

    # Fall back to meta.
    for k in candidates:
        v = meta.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()

    # Last resort: pull the ID out of the note text.
    text = data.get("text") or ""
    m = re.search(r"(?:患者ID|患者番号|Patient(?:\s*|_)?ID|PID)\s*[:：]?\s*([A-Za-z0-9._\-]+)", text)
    if m:
        return m.group(1)

    return None


def main(input_path: Path, out_dir: Path, patient_key: str | None):
    data = json.loads(input_path.read_text(encoding="utf-8"))

    detailed_records = []

    for item in data:
        full_text = item.get("data", {}).get("text", "")
        if not full_text:
            continue

        ann_block = pick_latest_annotation_block(item.get("annotations", []))
        if not ann_block:
            continue

        annotations = ann_block.get("result", [])

        label_spans = [
            a for a in annotations
            if a.get("type") == "labels"
            and isinstance(a.get("value"), dict)
            and a["value"].get("start") is not None
            and a["value"].get("end") is not None
        ]
        sorted_ann = sorted(label_spans, key=robust_sort_key)

        # Fall back to a text hash when the export carries no item id.
        doc_id = item.get("id")
        if doc_id is None:
            doc_id = abs(hash(full_text))

        patient_id = pick_patient_id(item, preferred_key=patient_key)

        for ann in sorted_ann:
            s = int(ann["value"]["start"])
            e = int(ann["value"]["end"])
            if not (0 <= s <= e <= len(full_text)):
                continue
            labels = ann["value"].get("labels") or ann.get("labels") or []
            label = labels[0] if labels else None
            seg = full_text[s:e]

            detailed_records.append({
                "patient_id": patient_id,
                "document_id": doc_id,
                "start": s,
                "end": e,
                "label": label,
                "extracted_text": seg,
                "formatted_text": format_text(seg),
                "document": full_text
            })

    # Span-level table: one row per annotated span.
    detailed_df = pd.DataFrame(detailed_records)
    if not detailed_df.empty:
        detailed_df = detailed_df.sort_values(["patient_id", "document_id", "start", "end"])

    out1 = out_dir / "annotated_segments_with_position.xlsx"
    detailed_df.to_excel(out1, index=False)

    # Document-level table: all spans of a document joined together.
    per_doc_records = []
    if not detailed_df.empty:
        for (doc_id), group in detailed_df.groupby("document_id", sort=False):
            group = group.sort_values(["start", "end"])
            original_text = group["document"].iloc[0]
            extracted = "\n".join(group["extracted_text"])
            formatted = format_text(extracted)

            # A document normally maps to a single patient; join if several were found.
            pid_vals = sorted({str(v) for v in group["patient_id"].dropna().astype(str)}) if "patient_id" in group else []
            pid_joined = ";".join(pid_vals) if pid_vals else None

            per_doc_records.append({
                "patient_id": pid_joined,
                "document_id": doc_id,
                "document": original_text,
                "extracted_annotated": extracted,
                "formatted_annotated": formatted
            })

    final_df = pd.DataFrame(per_doc_records)
    out2 = out_dir / "per_document_extracted_ordered_detailed.xlsx"
    final_df.to_excel(out2, index=False)

    print("Saved:")
    print(f"- {out1}")
    print(f"- {out2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to the Label Studio JSON export.")
    parser.add_argument("--out-dir", default=Path("."), type=Path,
                        help="Output directory (default: current directory).")
    parser.add_argument("--patient-key", default=None, type=str,
                        help="Name of the data field holding patient_id (optional).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    main(args.input, args.out_dir, args.patient_key)
