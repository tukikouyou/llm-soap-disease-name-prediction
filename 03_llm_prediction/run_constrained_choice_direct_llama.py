#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Llama-3.3-70B disease-name prediction: CONSTRAINED-CHOICE task, DIRECT prompting.

Task format : constrained choice - three names picked from the fixed list of
              55 respiratory disease labels.
Prompt      : direct - the candidate list is given without any ICD grouping.

Both input representations of every record are processed in one run:
  - column "document" : unstructured S/O text
  - column "label"    : NER-structured summary (see ../02_ner)

The other prompt strategy for this task format is run_constrained_choice_icd_guided_llama.py.
The GPT-4.1 / GPT-5 counterpart of this script is run_constrained_choice_direct.py.

Inference is local (transformers, fp16, greedy decoding); no API key is needed.

Usage:
  python run_constrained_choice_direct_llama.py --input data/records.xlsx \
      --model_path /path/to/hf-llama3.3-70b --output out.xlsx

Originally run as hanfeng_tuilun.py.
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

def build_prompt(input_text):
    choices = "\n- " + "\n- ".join(DISEASE_CLASSES)
    prompt = f"""あなたは呼吸器専門医です。以下の情報をもとに、考えられる呼吸器疾患の病名を次の選択肢の中から３つ選んでください。

【カルテ情報】
{input_text}

【選択肢】
{choices}
（複数該当する場合は、もっとも代表的なものを3つ選んでください。）"""
    return prompt

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,    local_files_only=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
model.eval()

def llama_predict(input_text):
    try:
        messages = [
            {"role": "system", "content": "あなたは呼吸器疾患の医者です。与えられた情報から、病名を正確に分類してください。"},
            {"role": "user", "content": build_prompt(input_text)}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return output_text
    except Exception as e:
        return f"エラー: {e}"

input_excel_path = args.input
df = pd.read_excel(input_excel_path, dtype=str)

# Run both input representations: "document" = unstructured S/O text, "label" = NER-structured summary.
for col in ["document", "label"]:
    output_col = f"LLaMA_病名_{col}"
    results = []
    print(f"Processing with LLaMA: {col} -> {output_col} ...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {col}"):
        result = llama_predict(row[col])
        results.append(result)
    df[output_col] = results

output_excel_path = args.output
df.to_excel(output_excel_path, index=False)
print("Saved:", output_excel_path)

