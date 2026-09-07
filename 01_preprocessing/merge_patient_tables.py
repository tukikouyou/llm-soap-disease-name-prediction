#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge 3 tables by patient ID.
- table1 is the base: 患者番号 is the key, and its `document` / `ICD` columns get filled in.
- table2 supplies 内容, written into table1's `document`.
- table3 supplies ICDコード, written into table1's `ICD`.

USAGE (CLI):
    python merge_patient_tables.py \
        --t1 path/to/table1.xlsx \
        --t2 path/to/table2.xlsx \
        --t3 path/to/table3.xlsx \
        --out merged_output.xlsx

Optional args for custom column or sheet names are available; run with -h to see.
"""

import argparse
import pandas as pd
import re
from pathlib import Path

def only_digits(x: str) -> str:
    """Keep only the digits of an ID-like cell, so that '1E+08', '1000003010.0' and
    '  1000003010  ' all collapse to the same key.
    """
    if x is None:
        return None
    s = str(x)
    digits = re.findall(r'\d+', s)
    if not digits:
        return None
    return ''.join(digits)

def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).map(only_digits)

def agg_unique_join(series: pd.Series, sep=' | ') -> str:
    vals = [str(v) for v in series.dropna().astype(str) if str(v).strip() != '']
    if not vals:
        return None
    # Keep the order of first occurrence.
    seen = set()
    ordered = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return sep.join(ordered)

def main():
    ap = argparse.ArgumentParser(description='Merge 3 medical tables by 患者番号/患者ID.')
    ap.add_argument('--t1', required=True, help='Path to table1 (has 患者番号, document, ICD).')
    ap.add_argument('--t2', required=True, help='Path to table2 (has 患者番号, 内容).')
    ap.add_argument('--t3', required=True, help='Path to table3 (has 患者ID, ICDコード).')
    ap.add_argument('--sheet1', default=0, help='Sheet for table1 (name or index).')
    ap.add_argument('--sheet2', default=0, help='Sheet for table2 (name or index).')
    ap.add_argument('--sheet3', default=0, help='Sheet for table3 (name or index).')
    ap.add_argument('--col1_id', default='患者番号', help='Key column in table1.')
    ap.add_argument('--col1_doc', default='document', help='Target column in table1 for 内容.')
    ap.add_argument('--col1_icd', default='ICD', help='Target column in table1 for ICDコード.')
    ap.add_argument('--col2_id', default='患者番号', help='Key column in table2.')
    ap.add_argument('--col2_content', default='内容', help='Content column in table2 to map to document.')
    ap.add_argument('--col3_id', default='患者ID', help='Key column in table3.')
    ap.add_argument('--col3_icd', default='ICDコード', help='ICD column in table3 to map to ICD.')
    ap.add_argument('--out', default='merged_output.xlsx', help='Output Excel path.')
    ap.add_argument('--content_strategy', choices=['first', 'concat'], default='concat',
                    help="If multiple rows in table2 per 患者番号: 'first' keeps first; 'concat' joins unique values with ' | '.")
    args = ap.parse_args()

    # Read all as strings to prevent Excel auto-format issues
    t1 = pd.read_excel(args.t1, sheet_name=args.sheet1, dtype=str)
    t2 = pd.read_excel(args.t2, sheet_name=args.sheet2, dtype=str)
    t3 = pd.read_excel(args.t3, sheet_name=args.sheet3, dtype=str)

    t1['_key'] = normalize_id_series(t1[args.col1_id])
    t2['_key'] = normalize_id_series(t2[args.col2_id])
    t3['_key'] = normalize_id_series(t3[args.col3_id])

    # Collapse table2 to one row per key.
    if args.content_strategy == 'first':
        t2_agg = (t2
                  .dropna(subset=['_key'])
                  .sort_index()
                  .drop_duplicates(subset=['_key'], keep='first')
                  .rename(columns={args.col2_content: '_document'})[['_key', '_document']])
    else:
        t2_agg = (t2
                  .dropna(subset=['_key'])
                  .groupby('_key', as_index=False)[args.col2_content]
                  .apply(lambda s: agg_unique_join(s))
                  .rename(columns={args.col2_content: '_document'}))

    # One ICD per key: keep the first non-null occurrence.
    t3_agg = (t3
              .dropna(subset=['_key'])
              .sort_index()
              .drop_duplicates(subset=['_key'], keep='first')
              .rename(columns={args.col3_icd: '_ICD'})[['_key', '_ICD']])

    merged = (t1
              .merge(t2_agg, on='_key', how='left')
              .merge(t3_agg, on='_key', how='left'))

    if args.col1_doc not in merged.columns:
        merged[args.col1_doc] = None
    if args.col1_icd not in merged.columns:
        merged[args.col1_icd] = None

    def fillcol(df, target, source):
        """Fill only where the target is empty, so existing values are never overwritten."""
        mask_empty = df[target].isna() | (df[target].astype(str).str.strip() == '')
        df.loc[mask_empty, target] = df.loc[mask_empty, source]

    fillcol(merged, args.col1_doc, '_document')
    fillcol(merged, args.col1_icd, '_ICD')

    merged = merged.drop(columns=['_document', '_ICD'])
    merged = merged.drop(columns=['_key'])

    out_path = Path(args.out)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        merged.to_excel(writer, index=False, sheet_name='result')

    print(f'Saved: {out_path.resolve()}')
    print(f'Rows: {len(merged)} | Columns: {list(merged.columns)}')

if __name__ == '__main__':
    main()
