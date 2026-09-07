"""Reshape the raw SOAP export into one row per document: patient ID, date and merged S/O text."""
import pandas as pd

df = pd.read_csv(r"data/03_SOAP.csv", dtype=str, encoding="CP932")
df['オーダ日付'] = pd.to_datetime(df['オーダ日付'], errors='coerce')

df_s = df[df['項目名'] == 'S)'][['患者番号', 'オーダ日付', '内容']].rename(columns={'内容': 'S内容'})
df_o = df[df['項目名'] == 'O)'][['患者番号', 'オーダ日付', '内容']].rename(columns={'内容': 'O内容'})

# Inner join keeps only visits that have both an S and an O entry.
merged = pd.merge(df_s, df_o, on=['患者番号', 'オーダ日付'])
merged['内容'] = merged['S内容'] + '\n' + merged['O内容']

final_df = merged[['患者番号', 'オーダ日付', '内容']]
final_df.to_excel("S_O_統合結果.xlsx", index=False)
