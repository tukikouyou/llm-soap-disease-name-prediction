#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM disease-name prediction: FREE-FORM task, DIRECT prompting.

Task format : free-form (no candidate list; up to three respiratory diagnoses)
Prompt      : direct - the model is asked for disease names straight from the
              case text, with no ICD category step and no output template.

Both input representations of every record are processed in one run:
  - column "document" : unstructured S/O text
  - column "label"    : NER-structured summary (see ../02_ner)

The ICD-guided counterpart of this script is run_free_text_icd_guided.py.

The API key is read from the OPENAI_API_KEY environment variable.
Set OPENAI_BASE_URL to use an OpenAI-compatible endpoint.

Usage:
  python run_free_text_direct.py --input data/records.xlsx --model gpt-4.1 --output out.xlsx
"""

import argparse
import os
import time

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

# ========= LLM inference for one record =========
def GPT_predict_single_input(input_text):
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content":f"""あなたは呼吸器疾患の医者です。与えられた情報から、考えられる呼吸器疾患の病名を最大3つまで予測してください。"""},
                {"role": "user", "content": input_text},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error:", e)
        return "エラー: " + str(e)


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
