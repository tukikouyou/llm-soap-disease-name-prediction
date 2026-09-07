"""Pivot the S) and O) rows of a visit into a single S+O text column."""
import pandas as pd

df = pd.read_excel(r"data/200初診SO.xlsx")
df = df[["患者番号", "オーダ日付", "項目名", "内容"]]

pivot_df = df.pivot_table(index=["患者番号", "オーダ日付"],
                          columns="項目名",
                          values="内容",
                          aggfunc="first").reset_index()

pivot_df["S+O"] = pivot_df["S)"].fillna('') + "\n" + pivot_df["O)"].fillna('')

pivot_df.to_excel(r"data/200初診SO_merged.xlsx", index=False)
