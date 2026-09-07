#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Llama-3.3-70B disease-name prediction: FREE-FORM task, ICD-GUIDED prompting.

Task format : free-form (no candidate list; up to three respiratory diagnoses)
Prompt      : ICD-guided - each predicted name must be reported together with
              its ICD-10 code, a probability and a one-line rationale.

Both input representations of every record are processed in one run:
  - column "document" : unstructured S/O text
  - column "label"    : NER-structured summary (see ../02_ner)

The other prompt strategy for this task format is run_free_text_direct_llama.py.
The GPT-4.1 / GPT-5 counterpart of this script is run_free_text_icd_guided.py.

Inference is local (transformers, fp16, greedy decoding); no API key is needed.

Usage:
  python run_free_text_icd_guided_llama.py --input data/records.xlsx \
      --model_path /path/to/hf-llama3.3-70b --output out.xlsx

Originally run as hanfeng_tuilun_putong_V2.py.
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import time
from tqdm import tqdm
import os
from collections import defaultdict

# ========= command line arguments =========
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True,
                    help="Excel file with the S/O records. Must contain the "
                         "columns 'document' (unstructured S/O text) and "
                         "'label' (NER-structured summary).")
parser.add_argument("--output", required=True, help="Output Excel file.")
parser.add_argument("--model_path", required=True,
                    help="Local directory holding the Llama-3.3-70B-Instruct weights.")
args = parser.parse_args()

MODEL_PATH = Path(args.model_path)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
model.eval()


SYSTEM_PROMPT = (
    "あなたは臨床経験が豊富な呼吸器内科医です。"
    "必ず与えられた初診情報から、呼吸器領域を中心に鑑別診断を挙げます。"
    "指定フォーマットで回答してください。"
)

USER_TEMPLATE = """以下の初診S/Oから、呼吸器疾患の診断候補を最大3件、尤度の高い順に提案してください。

【カルテ情報】
{snippet}

出力形式（厳守）：
病気名　所属ICD1: <コード or N/A> <名称>（XX％） 理由：<主要根拠>

病気名　所属ICD2: <コード or N/A> <名称>（XX％） 理由：<主要根拠>

病気名　所属ICD3: <コード or N/A> <名称>（XX％） 理由：<主要根拠>

注意：
- 病名を予測し最大3つ、確率の高い順に列挙してください。
- まず病名を記載し、次にその病名に対応するICD-10コードと名称を記載。
- ICD-10コードが不明な場合は <コード or N/A> の部分を N/A とする（名称は疾患群名で可）。
- ％は推定確率、簡潔な根拠（症状・所見・既往・検査の示唆など）を1行で。

"""


# ========= LLM inference for one record =========
def LLaMA_predict_single_input(input_text):
    try:
        payload =USER_TEMPLATE.format(snippet=input_text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        print("Error:", e)
        return "エラー: " + str(e)

input_excel_path = args.input
df = pd.read_excel(input_excel_path, dtype=str)

# Run both input representations: "document" = unstructured S/O text, "label" = NER-structured summary.
for col in ["document", "label"]:
    output_col = f"LLaMA_病名_{col}"
    results = []

    print(f"Processing: {col} -> {output_col} ...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {col}"):
        result = LLaMA_predict_single_input(row[col])
        results.append(result)

    df[output_col] = results

output_excel_path = args.output
df.to_excel(output_excel_path, index=False)
print("Saved:", output_excel_path)

