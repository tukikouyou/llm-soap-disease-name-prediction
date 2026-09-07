#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GEE repeated-measures logistic regression for the LLM disease-name prediction results.

Every evaluation record is predicted 24 times (3 models x 2 task formats x 2 input
representations x 2 prompt strategies), so the prediction records are not independent.
Inference therefore uses generalised estimating equations with the case document as the
clustering unit:

  family / link       binomial, logit
  working correlation exchangeable
  covariance          robust sandwich estimator
  clustering unit     document_id (primary), patient_id (sensitivity analysis)

Both outcomes are fitted with the same multivariable main-effects model:

  top1_correct ~ task_format + input_type + prompt_strategy + visit_type + model
  top3_correct ~ task_format + input_type + prompt_strategy + visit_type + model

Reference levels are free-form prediction, unstructured S/O text, direct prompting,
follow-up records and GPT-4.1, so an odds ratio above 1 means higher odds of a correct
prediction than the reference level.

Input
-----
A long-format CSV with one row per prediction record and the columns
document_id, patient_id, model, visit_type, prompt_strategy, task_format, input_type,
top1_correct, top3_correct. This file is derived from the clinical records and is not
part of this repository; see the Data availability statement.

The condition columns must use these level names, the first of each pair being the
reference level:

  task_format      free_text | constrained_choice
  input_type       unstructured_text | ner_structured_summary
  prompt_strategy  direct | icd_guided
  visit_type       follow_up | initial_visit
  model            GPT-4.1 | GPT-5 | Llama-3.3-70B

Outputs (written to --outdir)
-----------------------------
  gee_primary_document_cluster_all.csv     both outcomes, clustered by document_id
  gee_top1_document_cluster.csv            Top-1 only
  gee_top3_document_cluster.csv            Top-3 only
  gee_sensitivity_patient_cluster_all.csv  clustered by patient_id
  gee_model_summary.csv                    model settings and cluster sizes

Usage:
  python run_gee_analysis.py --input data/00_long_format_llm_records.csv --outdir gee_out
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--input", required=True,
                    help="Long-format CSV of prediction records (one row per prediction).")
parser.add_argument("--outdir", default="gee_repeated_measures",
                    help="Directory for the CSV outputs.")
args = parser.parse_args()

INPUT_CSV = args.input
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_CSV)

# Standardize outcome names and formula. Reference levels are deliberately chosen to make ORs match the manuscript comparison table.
FORMULA = (
    'Y ~ C(task_format, Treatment(reference="free_text"))'
    ' + C(input_type, Treatment(reference="unstructured_text"))'
    ' + C(prompt_strategy, Treatment(reference="direct"))'
    ' + C(visit_type, Treatment(reference="follow_up"))'
    ' + C(model, Treatment(reference="GPT-4.1"))'
)

TERM_LABELS = {
    'C(task_format, Treatment(reference="free_text"))[T.constrained_choice]': 'constrained choice vs free text',
    'C(input_type, Treatment(reference="unstructured_text"))[T.ner_structured_summary]': 'NER-structured vs unstructured',
    'C(prompt_strategy, Treatment(reference="direct"))[T.icd_guided]': 'ICD-guided prompt vs direct prompt',
    'C(visit_type, Treatment(reference="follow_up"))[T.initial_visit]': 'initial visit vs follow-up',
    'C(model, Treatment(reference="GPT-4.1"))[T.GPT-5]': 'GPT-5 vs GPT-4.1',
    'C(model, Treatment(reference="GPT-4.1"))[T.Llama-3.3-70B]': 'Llama vs GPT-4.1',
}


def fit_gee(data: pd.DataFrame, outcome: str, group_col: str) -> tuple[pd.DataFrame, object]:
    d = data.copy()
    d["Y"] = d[outcome].astype(int)
    gee = smf.gee(
        FORMULA,
        groups=group_col,
        data=d,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable(),
    )
    res = gee.fit(maxiter=100)
    ci = res.conf_int()
    rows = []
    for term, factor in TERM_LABELS.items():
        coef = float(res.params[term])
        se = float(res.bse[term])
        z = float(res.tvalues[term])
        p = float(res.pvalues[term])
        low, high = map(float, ci.loc[term])
        rows.append(
            {
                "cluster_by": group_col,
                "outcome": "Top-1 correct" if outcome == "top1_correct" else "Top-3 correct",
                "factor": factor,
                "reference": factor.split(" vs ")[1] if " vs " in factor else "",
                "comparison": factor.split(" vs ")[0] if " vs " in factor else factor,
                "coef_log_odds": coef,
                "robust_se": se,
                "z_value": z,
                "odds_ratio": float(np.exp(coef)),
                "ci95_low": float(np.exp(low)),
                "ci95_high": float(np.exp(high)),
                "p_value": p,
                "n_observations": int(res.nobs),
                "n_clusters": int(d[group_col].nunique()),
                "cluster_size_min": int(d.groupby(group_col).size().min()),
                "cluster_size_max": int(d.groupby(group_col).size().max()),
                "working_corr_exchangeable": float(res.cov_struct.dep_params),
            }
        )
    return pd.DataFrame(rows), res

# Primary model: clustered by document/case ID, as requested.
primary_parts = []
model_summaries = []
for outcome in ["top1_correct", "top3_correct"]:
    table, res = fit_gee(df, outcome, "document_id")
    primary_parts.append(table)
    model_summaries.append(
        {
            "cluster_by": "document_id",
            "outcome": table["outcome"].iloc[0],
            "n_observations": int(res.nobs),
            "n_clusters": int(df["document_id"].nunique()),
            "cluster_size": "24 per document",
            "working_corr_exchangeable": float(res.cov_struct.dep_params),
            "covariance": "robust sandwich covariance",
            "family": "Binomial logit",
        }
    )
primary = pd.concat(primary_parts, ignore_index=True)

# Sensitivity model: clustered by patient ID, useful because some patients have multiple documents.
sensitivity_parts = []
for outcome in ["top1_correct", "top3_correct"]:
    table, res = fit_gee(df, outcome, "patient_id")
    sensitivity_parts.append(table)
    model_summaries.append(
        {
            "cluster_by": "patient_id",
            "outcome": table["outcome"].iloc[0],
            "n_observations": int(res.nobs),
            "n_clusters": int(df["patient_id"].nunique()),
            "cluster_size": f"{int(df.groupby('patient_id').size().min())}–{int(df.groupby('patient_id').size().max())}",
            "working_corr_exchangeable": float(res.cov_struct.dep_params),
            "covariance": "robust sandwich covariance",
            "family": "Binomial logit",
        }
    )
sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
summary_df = pd.DataFrame(model_summaries)

# Add formatted columns for readability.
def p_fmt(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

for t in [primary, sensitivity]:
    t["OR_95CI"] = t.apply(lambda r: f"{r['odds_ratio']:.2f} ({r['ci95_low']:.2f}–{r['ci95_high']:.2f})", axis=1)
    t["p_formatted"] = t["p_value"].map(p_fmt)
    t["interpretation"] = t.apply(
        lambda r: "significantly higher odds of a correct prediction"
        if (r["p_value"] < 0.05 and r["odds_ratio"] > 1)
        else ("significantly lower odds of a correct prediction"
              if (r["p_value"] < 0.05 and r["odds_ratio"] < 1) else "no significant difference"),
        axis=1,
    )

# Save CSV outputs.
primary.to_csv(os.path.join(OUTDIR, "gee_primary_document_cluster_all.csv"), index=False, encoding="utf-8-sig")
sensitivity.to_csv(os.path.join(OUTDIR, "gee_sensitivity_patient_cluster_all.csv"), index=False, encoding="utf-8-sig")
summary_df.to_csv(os.path.join(OUTDIR, "gee_model_summary.csv"), index=False, encoding="utf-8-sig")
primary[primary["outcome"] == "Top-1 correct"].to_csv(os.path.join(OUTDIR, "gee_top1_document_cluster.csv"), index=False, encoding="utf-8-sig")
primary[primary["outcome"] == "Top-3 correct"].to_csv(os.path.join(OUTDIR, "gee_top3_document_cluster.csv"), index=False, encoding="utf-8-sig")

print("Saved to", OUTDIR)
for fn in sorted(os.listdir(OUTDIR)):
    print("  ", fn)
