"""Link each extracted document back to a patient ID by fuzzy-matching the note text."""
import pandas as pd
from difflib import SequenceMatcher

table1 = pd.read_csv(r"data/S_O_統合結果.csv", dtype=str, encoding="utf-8")
table2 = pd.read_excel(r"data/per_document_extracted_ordered_detailed.xlsx", dtype=str)

# Strip whitespace and unify colons so that only content differences remain.
table1['内容_clean'] = table1['内容'].str.replace(r'\s+', '', regex=True).str.replace('：', ':')
table2['document_clean'] = table2['document'].str.replace(r'\s+', '', regex=True)

def match_patient_fuzzy(row, threshold=0.85):
    """Return the patient ID of the most similar note, or None if none clears the threshold."""
    best_score = 0
    best_id = None
    for _, row2 in table1.iterrows():
        score = SequenceMatcher(None, row2['内容_clean'], row['document_clean']).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_id = row2['患者番号']
    return best_id

table2['患者番号'] = table2.apply(lambda row: match_patient_fuzzy(row), axis=1)

table2.to_excel("document_with_patient_fuzzy.xlsx", index=False)
