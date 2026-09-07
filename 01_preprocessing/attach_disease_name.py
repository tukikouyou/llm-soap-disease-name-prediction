"""Attach diagnosis names to each merged S/O document via the patient ID."""
import pandas as pd

table1 = pd.read_csv(r"data/01_病名.csv", dtype=str, encoding="CP932")
table2 = pd.read_excel("S_O_統合結果.xlsx", dtype=str)

table1 = table1.rename(columns={"患者ID": "患者番号"})

# A patient can carry several diagnoses; collapse them into one field.
病名dict = (
    table1.groupby("患者番号")["病名"]
    .apply(lambda x: "／".join(x.dropna().unique()))
    .to_dict()
)

table2["病名"] = table2["患者番号"].map(病名dict)

table2.to_excel("S_O_統合結果+病名.xlsx", index=False)
