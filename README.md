# Design factors influencing LLM disease-name prediction from Japanese respiratory SOAP notes

Code accompanying the manuscript:

> **Design factors influencing large language model disease name prediction from Japanese respiratory SOAP notes**
> Feng Han, Minghui Tang, Ziheng Zhang, Hidenori Kitai, Kenji Hirata, Katsuhiko Ogasawara, Satoshi Konno, Kohsuke Kudo

This is the code repository for that study: data preprocessing, NER model training and
inference, LLM disease-name prediction, the supervised reference baseline, evaluation, and
statistical analysis.

**This README describes only what the code does and how it is run.** The study rationale,
prompt design, matching rules, hyperparameter choices, results and their interpretation are
described in the paper and its Supplementary Information; please refer to those for
anything beyond the mechanics below.

---

## Study design

400 real-world Japanese respiratory SOAP notes (200 initial-visit, 200 follow-up) were
predicted under every combination of three factors, on three LLMs:

| Factor | Levels |
|---|---|
| Model | GPT-4.1, GPT-5, Llama-3.3-70B |
| Task format | free-form prediction, selection among 55 candidate disease names |
| Prompt strategy | direct prompting, ICD-guided prompting |
| Input representation | unstructured S/O text, NER-structured summary |

3 x 2 x 2 x 2 = **24 conditions per record**, 400 records = **9,600 predictions**.
Documentation stage (initial-visit vs. follow-up) is a property of the record, not a
crossed factor.

---

## Pipeline

Each stage is a directory. All scripts read their inputs from a local `data/` directory
that is not part of this repository.

| Stage | Input | Output |
|---|---|---|
| `01_preprocessing/` | Raw hospital CSV exports (`03_SOAP.csv`, `01_病名.csv`; CP932) | Per-document table: `患者番号`, `オーダ日付`, merged S/O text, reference disease name, ICD code (`.xlsx`) |
| `02_ner/` | Label Studio JSON annotation exports; the per-document table | Fine-tuned XLM-RoBERTa model directory; `ner_predictions.xlsx` with the NER-structured summary column |
| `03_llm_prediction/` | `.xlsx` with a `document` column (unstructured S/O text) and a `label` column (NER-structured summary) | The same table plus two prediction columns, one per input representation |
| `04_baseline/` | Training records outside the evaluation set; the 400 evaluation records | Per-case predictions and summary metrics (`.csv`) |
| `05_evaluation/` | An LLM output table with gold and predicted columns | Top-1 Accuracy, Top-3 Inclusion Rate, Jaccard overlap (`.csv`) |
| `06_statistics/` | Long-format CSV, one row per prediction record | GEE odds ratios with 95% CI and p values (`.csv`, `.md`) |


### 01_preprocessing

Run in roughly this order:
`split_soap_csv.py`, `csv_to_txt.py`, `split_txt_chunks.py`, `nfkc_normalize.py`,
`fuzzy_dedup.py`, `clean_and_normalize_soap.py`, `merge_s_and_o.py`,
`build_document_table.py`, `attach_disease_name.py`, `match_document_to_patient.py`,
`merge_patient_tables.py`.

Identifying information (names, addresses, contact details, patient IDs and exact dates)
was removed or excluded before any downstream step.

### 02_ner

```bash
python 02_ner/clinical_ner_train_eval.py train --train_json export_1.json export_2.json --output_dir trained_clinical_ner_xlmr --num_train_docs 350 --epochs 40 --batch_size 8 --fp16
```

```bash
python 02_ner/clinical_ner_predict.py predict_table --model_dir trained_clinical_ner_xlmr --input_table data/records.xlsx --text_col 内容 --patient_col 患者番号 --date_col オーダ日付 --disease_col 病名 --output_xlsx ner_predictions.xlsx
```

Base model `xlm-roberta-base`; training settings and the model-selection criterion are
reported in the paper.

### 03_llm_prediction — the eight scripts and the 24 conditions

Each script implements one (task format x prompt strategy) cell for one backend, and each
run processes **both** input representations of every record: the `document` column
(unstructured S/O text) and the `label` column (NER-structured summary from `02_ner`).

| Script | Model(s) | Task format | Prompt strategy | Conditions |
|---|---|---|---|---|
| `run_free_text_direct.py` | GPT-4.1, GPT-5 via `--model` | free-form | direct | 4 |
| `run_free_text_icd_guided.py` | GPT-4.1, GPT-5 via `--model` | free-form | ICD-guided | 4 |
| `run_constrained_choice_direct.py` | GPT-4.1, GPT-5 via `--model` | selection (55) | direct | 4 |
| `run_constrained_choice_icd_guided.py` | GPT-4.1, GPT-5 via `--model` | selection (55) | ICD-guided | 4 |
| `run_free_text_direct_llama.py` | Llama-3.3-70B | free-form | direct | 2 |
| `run_free_text_icd_guided_llama.py` | Llama-3.3-70B | free-form | ICD-guided | 2 |
| `run_constrained_choice_direct_llama.py` | Llama-3.3-70B | selection (55) | direct | 2 |
| `run_constrained_choice_icd_guided_llama.py` | Llama-3.3-70B | selection (55) | ICD-guided | 2 |

Conditions = models x input representations. The four OpenAI scripts contribute
4 x 2 x 2 = 16 and the four Llama scripts 4 x 1 x 2 = 8, giving all **24 conditions**.

The 55 candidate disease names (Supplementary Table 8) are defined in the two selection
scripts; the ICD-guided one additionally holds the label-to-ICD-10 mapping and the five ICD
categories used for the category-selection step. The prompts themselves are in the scripts
and are reproduced in the paper.

GPT-4.1 and GPT-5 (OpenAI API) — the key is read from the environment:

```bash
export OPENAI_API_KEY=...
```

```bash
python 03_llm_prediction/run_constrained_choice_icd_guided.py --input data/records.xlsx --model gpt-4.1 --output GPT4.1_choice_icd.xlsx
```

Llama-3.3-70B — local inference (transformers, fp16, `device_map="auto"`, greedy decoding);
no API key:

```bash
python 03_llm_prediction/run_constrained_choice_icd_guided_llama.py --input data/records.xlsx --model_path /path/to/hf-llama3.3-70b --output Llama_choice_icd.xlsx
```

### 04_baseline

`baseline_tfidf_sgd.py` — character-level TF-IDF + SGD classifier trained on in-hospital
respiratory records outside the evaluation dataset under a patient-disjoint split, and
evaluated on the same 400 records. A non-equivalent supervised reference, not a matched
comparator; see the paper.

### 05_evaluation

```bash
python 05_evaluation/compute_topk_metrics.py --input results.xlsx --col-gold E --col-free-out G --col-struct-out H --col-entities C --out-prefix run1
```

Top-1 Accuracy and Top-3 Inclusion Rate for both input representations, plus their Jaccard
overlap. The clinical-equivalence matching rules are specified in the paper.

### 06_statistics

```bash
python 06_statistics/run_gee_analysis.py --input data/00_long_format_llm_records.csv --outdir gee_out
```

GEE logistic regression, binomial family with logit link, exchangeable working correlation,
robust sandwich covariance, clustered by `document_id` (sensitivity analysis by
`patient_id`). 

---

## Requirements

Python 3.11.9.

```bash
pip install -r requirements.txt
```

NER fine-tuning and Llama-3.3-70B inference were performed on the same GPU-equipped system, with sufficient memory to accommodate the 70B model weights in fp16. NER fine-tuning used Hugging Face Transformers 5.9.0 and PyTorch 2.6.0 (CUDA 12.4), with dependencies specified in `02_ner/requirements_clinical_ner.txt`.

---

## Data availability

The clinical data analysed in this study are **not** publicly available. They are
real-world electronic medical records containing sensitive patient information and are
subject to institutional and ethical restrictions.

Accordingly, this repository contains **code only**. 

 Enquiries about data or model access
should be directed to the corresponding author; access remains subject to approval by
Hokkaido University Hospital and to the applicable ethical and legal restrictions.

Scripts expect their inputs under a local `data/` directory.

---

## Ethics

Approved by the Institutional Review Board of Hokkaido University Hospital (approval number
024-0206), with an opt-out consent procedure.

## Funding

Supported by AMED under Grant Number JP266f0137006.

## License

MIT License; see `LICENSE`.
