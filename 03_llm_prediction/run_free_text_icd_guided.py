#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM disease-name prediction: FREE-FORM task.

The model is given the Subjective/Objective (S/O) part of a Japanese
respiratory SOAP note and must propose up to three respiratory disease
names, ranked by likelihood, without any predefined candidate list.

Two input representations are processed for every record:
  - column "document" : unstructured S/O text
  - column "label"    : NER-structured summary (entities predicted by the
                        fine-tuned XLM-RoBERTa model, see ../02_ner)

Two prompt strategies were compared in the study. This file contains the
ICD-guided prompt (the model is asked to attach an ICD-10 code to each
predicted disease name). The direct-prompt condition used the same script
with SYSTEM_PROMPT / USER_TEMPLATE replaced by the plain instruction that
contains no ICD-related requirement.

The API key is read from the OPENAI_API_KEY environment variable.
Set OPENAI_BASE_URL to use an OpenAI-compatible endpoint.

Usage:
  set OPENAI_API_KEY=...
  python run_free_text_prediction.py --input records.xlsx ^
      --model gpt-4.1 --output GPT4.1_free_text.xlsx
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
parser.add_argument("--output", required=True,
                    help="Output Excel file.")
parser.add_argument("--model", default="gpt-4.1",
                    help="Model name, e.g. gpt-4.1 or gpt-5.")
parser.add_argument("--sleep", type=float, default=1.0,
                    help="Seconds to wait between API calls.")
args = parser.parse_args()

# The API key must be provided through the environment, never hard-coded.
client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL") or None,
)

df = pd.read_excel(args.input, dtype=str)

SYSTEM_PROMPT = (
    "あなたは臨床経験が豊富な呼吸器内科医です。"
    "必ず与えられた初診情報から、呼吸器領域を中心に鑑別診断を挙げます。"
    "指定フォーマットで回答してください。"
)

USER_TEMPLATE = """以下の初診S/Oから、呼吸器疾患のみの診断候補を最大3件、尤度の高い順に提案してください。

【カルテ情報】
{snippet}

出力形式（厳守）：
病気名　所属ICD1: <コード or N/A> <名称>（XX％） 
理由：<主要根拠>

病気名　所属ICD2: <コード or N/A> <名称>（XX％） 
理由：<主要根拠>

病気名　所属ICD3: <コード or N/A> <名称>（XX％） 
理由：<主要根拠>

注意：
- 病名を予測し最大3つ、確率の高い順に列挙してください。
- まず病名を記載し、次にその病名に対応するICD-10コードと名称を記載。
- ICD-10コードが不明な場合は <コード or N/A> の部分を N/A とする（名称は疾患群名で可）。
- ％は推定確率、簡潔な根拠（症状・所見・既往・検査の示唆など）を1行で。

"""


# ========= LLM inference for one record =========
def GPT_predict_single_input(input_text):
    try:
        payload =USER_TEMPLATE.format(snippet=input_text)
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
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
