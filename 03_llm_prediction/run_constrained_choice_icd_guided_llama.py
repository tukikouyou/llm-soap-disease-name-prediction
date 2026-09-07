#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Llama-3.3-70B disease-name prediction: CONSTRAINED-CHOICE task, ICD-GUIDED prompting.

Task format : constrained choice - names picked from the fixed list of 55
              respiratory disease labels.
Prompt      : ICD-guided - the model first selects up to three of the five
              ICD-10 categories, then picks sub-labels under those categories.

Both input representations of every record are processed in one run:
  - column "document" : unstructured S/O text
  - column "label"    : NER-structured summary (see ../02_ner)

The other prompt strategy for this task format is run_constrained_choice_direct_llama.py.
The GPT-4.1 / GPT-5 counterpart of this script is run_constrained_choice_icd_guided.py.

Inference is local (transformers, fp16, greedy decoding); no API key is needed.

Usage:
  python run_constrained_choice_icd_guided_llama.py --input data/records.xlsx \
      --model_path /path/to/hf-llama3.3-70b --output out.xlsx

Originally run as hanfeng_tuilun_V2.py.
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

# The 5 ICD-10 categories (code, name).
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

# Mapping from each of the 55 labels to its ICD-10 category.
DX_TO_ICD = {
    # D381 lung tumour
    "肺腫瘍":"D381","右肺腫瘍":"D381","右肺中葉肺腫瘍":"D381","多発肺腫瘍":"D381",
    "多発性肺腫瘍":"D381","転移性肺腫瘍":"D381","気管支腺腫":"D381",

    # J439 emphysema / bulla
    "肺気腫":"J439","慢性肺気腫":"J439","若年性肺気腫":"J439","閉塞性肺気腫":"J439","ブラ性肺気腫":"J439",
    "気腫性肺嚢胞":"J439","気腫性肺のう胞":"J439","多発気腫性肺のう胞":"J439",
    "巨大気腫性肺のう胞":"J439","巨大気腫性肺のう胞感染":"J439","気腫合併肺線維症":"J439",

    # J459 asthma
    "気管支喘息":"J459","咳喘息":"J459","重症気管支喘息":"J459","難治性気管支喘息":"J459","難治性喘息":"J459",
    "運動誘発性喘息":"J459","好酸球増加性喘息":"J459","気管支喘息合併妊娠":"J459","喘息性気管支喘息":"J459",
    "喘息性気管支炎":"J459",

    # J82 eosinophil-related
    "好酸球性肺炎":"J82","急性好酸球性肺炎":"J82","慢性好酸球性肺炎":"J82","好酸球性気管支炎":"J82",
    "肺好酸球増加症":"J82","単純性肺好酸球増加症":"J82","遷延性肺好酸球増加症":"J82","肺好酸球増加症の再発":"J82",

    # J841 interstitial lung disease / fibrosis
    "特発性間質性肺炎":"J841","特発性間質性肺炎の急性増悪":"J841","特発性肺線維症":"J841",
    "特発性肺線維症の急性増悪":"J841","非特異性間質性肺炎":"J841","非特異性間質性肺炎の増悪":"J841",
    "非特異性間質性肺炎の急性増悪":"J841","呼吸細気管支炎関連性間質性肺疾患":"J841","リンパ球性間質性肺炎":"J841",
    "急性間質性肺炎":"J841","間質性肺線維症":"J841","炎症後肺線維症":"J841","肺線維症":"J841",
    "肺線維症の急性増悪":"J841","びまん性間質性肺炎":"J841","通常型間質性肺炎":"J841",
    "特発性器質化肺炎":"J841","アレルギー性肺炎":"J841",
}

# Group the labels by ICD category; used only to render the prompt.
ICD_TO_DX = defaultdict(list)
for dx in DISEASE_CLASSES:
    icd = DX_TO_ICD.get(dx)
    if icd:
        ICD_TO_DX[icd].append(dx)
        
        
def build_prompt(input_text)-> str:
    icd_choices = "\n".join([f"- {code}: {name}" for code, name in ICD_CLASSES])

    # One line per ICD category, its labels separated by "/".
    dx_grouped_lines = []
    for code, name in ICD_CLASSES:
        dxs = ICD_TO_DX.get(code, [])
        if dxs:
            dx_text = " / ".join(dxs)
            dx_grouped_lines.append(f"{code}（{name}）: {dx_text}")
    dx_choices = "\n".join(dx_grouped_lines)

    prompt = f"""あなたは呼吸器専門医です。以下の初診S/Oから、
まず 5つのICD-10大分類の中から **最大3つ** を選び、
つづいて **選んだ各ICDに対応する小分類（下記の候補）** から **最大3つ** を選んでください。
（ICDと小分類は必ず対応させてください。小分類は、そのICDに属するもののみを選んでください）

【カルテ情報】
{input_text}

【ICD-10大分類の候補（先にここから最大3つ選ぶ）】
{icd_choices}

【各ICDに属する小分類（上のICDを選んでから、この中から最大3つ選ぶ）】
{dx_choices}

出力形式（厳守）：
ICD1: <コード> <名称>（XX％）
小分類:
- <小分類1>（xx％） 理由：〜〜〜
- <小分類2>（xx％） 理由：〜〜〜
- <小分類3>（xx％） 理由：〜〜〜

ICD2: <コード> <名称>（XX％）
小分類:
- <小分類1>（xx％） 理由：〜〜〜
...

ICD3: <コード> <名称>（XX％）
小分類:
- <小分類1>（xx％） 理由：〜〜〜
- <小分類2>（xx％） 理由：〜〜〜
- <小分類3>（xx％） 理由：〜〜〜

注意：
- まずICD（大分類）を最大3つ、確率の高い順に列挙してください。
- 各ICDごとに、そのICD配下の小分類のみを最大3つ挙げてください。
- ％は推定確率、簡潔な根拠（症状・所見・既往・検査の示唆など）を1行で。
"""
    return prompt

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,local_files_only=True)

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
            {"role": "system", "content": "あなたは呼吸器疾患の医者です。必ず与えられた初診情報から、まずICD大分類を最大3つ、ついで各ICD配下の小分類を最大3つ、指定フォーマットで回答してください。"},
            {"role": "user", "content": build_prompt(input_text)}
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

