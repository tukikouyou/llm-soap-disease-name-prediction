import os
import re
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


# ======================================
# 0. File paths
# ======================================
TRAIN_FILE = r"data/training_data_dedup.xlsx"
TEST_INIT_FILE = r"data/baseline_initial_visit_free.xlsx"
TEST_FOLLOW_FILE = r"data/baseline_followup_free.xlsx"

OUT_DIR = r"baseline_free_text_results_sgd"
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================
# 1. Text cleaning
# ======================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("_x000D_", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


LABEL_NORMALIZATION = {
    # Add entries here to merge variant spellings of a label.
}

def normalize_label(x):
    x = str(x).strip()
    return LABEL_NORMALIZATION.get(x, x)


# ======================================
# 2. Data loading
# ======================================
def load_train_data(path):
    df = pd.read_excel(path)

    required_cols = ["患者番号", "内容", "病名"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Training file is missing required columns: {missing}")

    df = df.copy()
    df["text"] = df["内容"].apply(clean_text)
    df["label"] = df["病名"].apply(normalize_label)

    df = df[["患者番号", "text", "label"]].copy()
    df = df[(df["text"] != "") & (df["label"] != "")].copy()

    # Drop rows that repeat the same text with the same label.
    df = df.drop_duplicates(subset=["text", "label"]).copy()
    return df


def load_test_data(path, visit_type):
    df = pd.read_excel(path)

    required_cols = ["document_id", "document", "患者番号", "病名"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{visit_type} test file is missing required columns: {missing}")

    df = df.copy()
    df["visit_type"] = visit_type
    df["text"] = df["document"].apply(clean_text)
    df["label"] = df["病名"].apply(normalize_label)

    keep_cols = ["document_id", "患者番号", "visit_type", "text", "label"]
    df = df[keep_cols].copy()
    df = df[(df["text"] != "") & (df["label"] != "")].copy()
    df = df.drop_duplicates(subset=["document_id"]).copy()

    return df


# ======================================
# 3. Train/test overlap check
# ======================================
def check_exact_text_overlap(train_df, test_df):
    train_texts = set(train_df["text"].tolist())
    test_texts = set(test_df["text"].tolist())

    overlap = train_texts & test_texts
    print(f"\n[INFO] texts identical in train and test: {len(overlap)}")

    overlap_df = pd.DataFrame({"overlap_text": list(overlap)})
    overlap_df.to_csv(
        os.path.join(OUT_DIR, "exact_text_overlap.csv"),
        index=False,
        encoding="utf-8-sig"
    )


# ======================================
# 4. Top-k
# ======================================
def top_k_accuracy_from_scores(scores, classes, y_true, k=3):
    k = min(k, scores.shape[1])
    topk_idx = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    topk_labels = classes[topk_idx]

    hits = []
    for true_label, pred_labels in zip(y_true, topk_labels):
        hits.append(true_label in pred_labels)
    return float(np.mean(hits))


# ======================================
# 5. Label coverage check and export
# ======================================
def check_label_coverage_and_export(train_df, test_df):
    train_labels = set(train_df["label"].unique())
    test_labels = set(test_df["label"].unique())

    unseen = sorted(test_labels - train_labels)
    if unseen:
        print("\n[WARNING] the following test labels never occur in the training set:")
        for x in unseen:
            print(" -", x)
    else:
        print("\n[INFO] every test label occurs in the training set.")

    label_count_df = train_df["label"].value_counts().reset_index()
    label_count_df.columns = ["label", "train_count"]
    label_count_df.to_csv(
        os.path.join(OUT_DIR, "train_label_counts.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    test_support = (
        test_df[["label"]]
        .drop_duplicates()
        .merge(label_count_df, on="label", how="left")
        .fillna({"train_count": 0})
        .sort_values("train_count")
    )
    test_support["train_count"] = test_support["train_count"].astype(int)
    test_support.to_csv(
        os.path.join(OUT_DIR, "test_label_support_in_train.csv"),
        index=False,
        encoding="utf-8-sig"
    )


# ======================================
# 6. majority baseline
# ======================================
def compute_majority_baseline(train_df, init_df, follow_df, all_df):
    majority_label = train_df["label"].value_counts().idxmax()
    print(f"\n[INFO] most frequent training label: {majority_label}")

    rows = []
    for name, df_ in [
        ("initial_free", init_df),
        ("followup_free", follow_df),
        ("all_free", all_df),
    ]:
        top1 = (df_["label"] == majority_label).mean()
        rows.append({
            "test_name": name,
            "majority_label": majority_label,
            "top1": float(top1)
        })

    majority_df = pd.DataFrame(rows)
    majority_df.to_csv(
        os.path.join(OUT_DIR, "majority_baseline.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    print("\n===== Majority baseline =====")
    print(majority_df)


# ======================================
# 7. Model
# ======================================
def build_model():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=200000,
            sublinear_tf=True,
            dtype=np.float32
        )),
        ("clf", SGDClassifier(
            loss="log_loss",      # required for predict_proba
            penalty="l2",
            alpha=1e-5,
            max_iter=50,
            tol=1e-3,
            random_state=42
        ))
    ])
    return model


# ======================================
# 8. Evaluation on one test set
# ======================================
def evaluate_on_test(model, test_df, output_name):
    X_test = test_df["text"].astype(str)
    y_test = test_df["label"].astype(str)

    proba = model.predict_proba(X_test)
    classes = model.named_steps["clf"].classes_

    # Top-1
    top1_idx = np.argmax(proba, axis=1)
    pred_top1 = classes[top1_idx]
    top1 = accuracy_score(y_test, pred_top1)

    # Top-3
    top3 = top_k_accuracy_from_scores(proba, classes, y_test.values, k=3)

    pred_series = pd.Series(pred_top1)
    pred_series.value_counts().head(20).to_csv(
        os.path.join(OUT_DIR, f"predicted_top1_distribution_{output_name}.csv"),
        encoding="utf-8-sig"
    )

    # Case-level output.
    top3_idx = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
    top3_labels = classes[top3_idx]
    top3_scores = np.take_along_axis(proba, top3_idx, axis=1)

    rows = []
    for i in range(len(test_df)):
        rows.append({
            "document_id": test_df.iloc[i]["document_id"],
            "患者番号": test_df.iloc[i]["患者番号"],
            "visit_type": test_df.iloc[i]["visit_type"],
            "gold_label": y_test.iloc[i],
            "pred_top1": pred_top1[i],
            "top1_correct": int(pred_top1[i] == y_test.iloc[i]),
            "top3_correct": int(y_test.iloc[i] in top3_labels[i]),
            "rank1_label": top3_labels[i][0] if len(top3_labels[i]) > 0 else None,
            "rank1_score": float(top3_scores[i][0]) if len(top3_scores[i]) > 0 else None,
            "rank2_label": top3_labels[i][1] if len(top3_labels[i]) > 1 else None,
            "rank2_score": float(top3_scores[i][1]) if len(top3_scores[i]) > 1 else None,
            "rank3_label": top3_labels[i][2] if len(top3_labels[i]) > 2 else None,
            "rank3_score": float(top3_scores[i][2]) if len(top3_scores[i]) > 2 else None,
            "text_used": test_df.iloc[i]["text"]
        })

    pred_df = pd.DataFrame(rows)

    summary = {
        "test_name": output_name,
        "n_cases": int(len(test_df)),
        "top1": float(top1),
        "top3": float(top3)
    }

    pred_df.to_csv(
        os.path.join(OUT_DIR, f"case_predictions_{output_name}.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    with open(os.path.join(OUT_DIR, f"metrics_{output_name}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n===== {output_name} =====")
    print(summary)
    print("\nTop-20 most frequent Top-1 predictions:")
    print(pred_series.value_counts().head(20))

    return pred_df, summary


# ======================================
# 9. Main
# ======================================
if __name__ == "__main__":
    train_df = load_train_data(TRAIN_FILE)
    init_df = load_test_data(TEST_INIT_FILE, visit_type="初診")
    follow_df = load_test_data(TEST_FOLLOW_FILE, visit_type="非初診")
    test_all_df = pd.concat([init_df, follow_df], ignore_index=True)

    print("===== dataset sizes =====")
    print("Train:", len(train_df))
    print("Initial test:", len(init_df))
    print("Follow-up test:", len(follow_df))
    print("All test:", len(test_all_df))

    print("\n===== label counts =====")
    print("Train label count:", train_df["label"].nunique())
    print("Initial test label count:", init_df["label"].nunique())
    print("Follow-up test label count:", follow_df["label"].nunique())

    check_label_coverage_and_export(train_df, test_all_df)
    check_exact_text_overlap(train_df, test_all_df)
    compute_majority_baseline(train_df, init_df, follow_df, test_all_df)

    X_train = train_df["text"].astype(str)
    y_train = train_df["label"].astype(str)

    model = build_model()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    train_top1 = (train_pred == y_train).mean()
    print(f"\n[INFO] train top1 = {train_top1:.6f}")

    pd.DataFrame({
        "text": X_train,
        "gold_label": y_train,
        "pred_top1": train_pred,
        "top1_correct": (train_pred == y_train).astype(int)
    }).to_csv(
        os.path.join(OUT_DIR, "train_predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pred_init, summary_init = evaluate_on_test(model, init_df, "initial_free")
    pred_follow, summary_follow = evaluate_on_test(model, follow_df, "followup_free")
    pred_all, summary_all = evaluate_on_test(model, test_all_df, "all_free")

    summary_df = pd.DataFrame([
        {
            "test_name": "initial_free",
            "n_cases": summary_init["n_cases"],
            "top1": summary_init["top1"],
            "top3": summary_init["top3"]
        },
        {
            "test_name": "followup_free",
            "n_cases": summary_follow["n_cases"],
            "top1": summary_follow["top1"],
            "top3": summary_follow["top3"]
        },
        {
            "test_name": "all_free",
            "n_cases": summary_all["n_cases"],
            "top1": summary_all["top1"],
            "top3": summary_all["top3"]
        }
    ])
    summary_df.to_csv(
        os.path.join(OUT_DIR, "summary_all_tests.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== final results =====")
    print(summary_df)
    print(f"\nOutput directory: {OUT_DIR}")