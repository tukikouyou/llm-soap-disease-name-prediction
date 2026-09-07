#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM disease-name prediction: CANDIDATE-CONSTRAINED SELECTION task, DIRECT prompting.

Task format : selection of up to three disease names from the fixed set of 55
              candidate labels (Supplementary Table 8).
Prompt      : direct - the 55 candidates are presented as one flat list and the
              model selects from them, with no ICD category step.

Both input representations of every record are processed in one run:
  - column "document" : unstructured S/O text
  - column "label"    : NER-structured summary (see ../02_ner)

The ICD-guided counterpart of this script is run_constrained_choice_icd_guided.py.

The API key is read from the OPENAI_API_KEY environment variable.
Set OPENAI_BASE_URL to use an OpenAI-compatible endpoint.

Usage:
  python run_constrained_choice_direct.py --input data/records.xlsx --model gpt-4.1 --output out.xlsx
"""

import argparse
import os
import time
from collections import defaultdict

import openai
import pandas as pd
from tqdm import tqdm

# ========= command line arguments =========
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True,
                    help="Excel file with the S/O records. Must contain the "
                         "columns 'document' (unstructured S/O text) and "
                         "'label' (NER-structured summary).")
parser.add_argument("--output", required=True, help="Output Excel file.")
parser.add_argument("--model", default="gpt-4.1", help="Model name, e.g. gpt-4.1 or gpt-5.")
parser.add_argument("--sleep", type=float, default=1.0,
                    help="Seconds to wait between API calls.")
args = parser.parse_args()

# The API key must be provided through the environment, never hard-coded.
client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL") or None,
)

df = pd.read_excel(args.input, dtype=str)

# ========= candidate labels =========
# ICD-10 categories, kept for reference; the direct prompt uses the flat list below.
ICD_CLASSES = [
    ("D381", "肺腫瘍"),
    ("J439", "肺気腫（含ブラ/嚢胞）"),
    ("J459", "喘息"),
    ("J82",  "好酸球性肺炎/好酸球関連"),
    ("J841", "びまん性間質性肺疾患（線維化系/IIP）"),
]

# The 55 candidate disease labels (Supplementary Table 8).
DISEASE_CLASSES = [
    "びまん性間質性肺炎","アレルギー性肺炎","ブラ性肺気腫","リンパ球性間質性肺炎","単純性肺好酸球増加症",
    "右肺中葉肺腫瘍","右肺腫瘍","呼吸細気管支炎関連性間質性肺疾患","咳喘息","喘息性気管支喘息","喘息性気管支炎",
    "多発性肺腫瘍","多発気腫性肺のう胞","多発肺腫瘍","好酸球増加性喘息","好酸球性気管支炎","好酸球性肺炎",
    "巨大気腫性肺のう胞","巨大気腫性肺のう胞感染","急性好酸球性肺炎","急性間質性肺炎","慢性好酸球性肺炎",
    "慢性肺気腫","気管支喘息","気管支喘息合併妊娠","気管支腺腫","気腫合併肺線維症","気腫性肺のう胞","気腫性肺嚢胞",
    "炎症後肺線維症","特発性器質化肺炎","特発性肺線維症","特発性肺線維症の急性増悪","特発性間質性肺炎",
    "特発性間質性肺炎の急性増悪","特発性非特異性間質性肺炎","肺好酸球増加症","肺好酸球増加症の再発","肺気腫",
    "肺線維症","肺線維症の急性増悪","肺腫瘍","若年性肺気腫","転移性肺腫瘍","通常型間質性肺炎","運動誘発性喘息",
    "遷延性肺好酸球増加症","重症気管支喘息","閉塞性肺気腫","間質性肺線維症","難治性喘息","難治性気管支喘息",
    "非特異性間質性肺炎","非特異性間質性肺炎の増悪","非特異性間質性肺炎の急性増悪",
]

# ========= prompt construction =========
def build_classification_prompt(input_text: str) -> str:
    choices = "\n- " + "\n- ".join(DISEASE_CLASSES)
    prompt = f"""あなたは呼吸器専門医です。以下の情報をもとに、考えられる呼吸器疾患の病名を次の選択肢の中から３つ選んでください。

【カルテ情報】
{input_text}

【選択肢】
{choices}
（複数該当する場合は、もっとも代表的なものを3つ選んでください。）"""
    return prompt

# ========= LLM inference for one record =========
def GPT_predict_single_input(input_text: str) -> str:
    """Return the raw model text so that it can be written straight into the output table."""
    try:
        prompt = build_classification_prompt(input_text)
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "あなたは呼吸器疾患の医者です。与えられた情報から、病名を正確に分類してください。"},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"エラー: {e}"

# ========= run both input representations =========
#   "document" = unstructured S/O text, "label" = NER-structured summary
for col in ["document", "label"]:
    output_col = f"GPT_病名_{col}"
    results = []
    print(f"Processing: {col} -> {output_col} ...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {col}"):
        result = GPT_predict_single_input(row[col])
        results.append(result)
        time.sleep(args.sleep)
    df[output_col] = results


# ========= save =========
df.to_excel(args.output, index=False)
print(f"Saved: {args.output}")