"""Clean and normalize first-visit SOAP notes, merging S and O per patient/date into JSON."""
import pandas as pd
import json
import re
import unicodedata
from collections import defaultdict

HEISEI_RE = re.compile(r"平成(\d+)年")

def convert_heisei_to_seireki(text):
    match = HEISEI_RE.search(text)
    if match:
        heisei_year = int(match.group(1))
        seireki_year = 1988 + heisei_year  # Heisei 1 = 1989
        text = HEISEI_RE.sub(f"{seireki_year}年", text)
    return text

def normalize_numbers(text):
    text = re.sub(r"(\d+)％", r"\1%", text)
    text = re.sub(r"(\d+)～(\d+)", r"\1-\2", text)
    text = re.sub(r"(\d+)℃", r"\1°C", text)
    return text

def normalize_units(text):
    text = re.sub(r"mg/dl", "mg/dL", text, flags=re.IGNORECASE)
    text = re.sub(r"mEq/ｌ", "mEq/L", text, flags=re.IGNORECASE)
    text = re.sub(r"U/ｌ", "U/L", text, flags=re.IGNORECASE)
    return text

def normalize_fullwidth_to_halfwidth(text):
    return unicodedata.normalize("NFKC", text)

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = convert_heisei_to_seireki(text)
    text = normalize_numbers(text)
    text = normalize_units(text)
    text = normalize_fullwidth_to_halfwidth(text)

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Turn sentence-final periods into "。" while leaving decimal points intact.
    text = re.sub(r"(?<!\d)\.(?!\d)", "。", text)

    return text


csv_file = r"data/200初診SO.csv"
df = pd.read_csv(csv_file, dtype=str)

df["内容"] = df["内容"].fillna("")

merged_data = defaultdict(lambda: {"S": "", "O": ""})

for _, row in df.iterrows():
    patient_id = row["患者番号"]
    date = row["オーダ日付"]
    key = (patient_id, date)

    if "S)" in str(row["項目名"]):
        merged_data[key]["S"] += " " + clean_text(row["内容"])
    elif "O)" in str(row["項目名"]):
        merged_data[key]["O"] += " " + clean_text(row["内容"])

output_data = []
for (patient_id, date), soap in merged_data.items():
    merged_text = f"{soap['S']} {soap['O']}".strip()
    merged_text = re.sub(r"\s+", " ", merged_text)

    output_data.append({
        "patient_id": patient_id,
        "date": date,
        "text": merged_text
    })

output_file = "cleaned_soap_frist_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
