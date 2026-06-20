"""
preprocess_sms.py
=================
Dataset 1 — "A Balanced Dataset for Spam and Smishing Detection using LLMs"
(Munoz & Islam, 2025, Mendeley Data). Columns: LABEL, TEXT, URL, EMAIL, PHONE.

Pipeline (Chapter 3 §3.2.1, Chapter 4 §4.3.4(1) of the thesis):
  1.  Load and inspect structure / first rows               (Figures D.1, D.2)
  2.  Raw visualisation: label pie+bar, URL/EMAIL/PHONE,
      message-length distribution                           (Figures C.1–C.3)
  3.  Null check, duplicate removal (2,169 expected)        (Figures D.3, D.4)
  4.  Binary label standardisation (ham=0; spam,smishing=1) (Table 4.6)
      Note: both "spam" and "smishing" are merged into the "scam" (1) class.
  5.  Text cleaning: strip URLs / emails / phones /
      non-alphabetic chars, lowercase, stopword removal
  6.  Feature engineering: msg_len, binary URL/EMAIL/PHONE  (Figures D.5–D.7)
  7.  TF-IDF (200 n-grams, ngram_range=(1,2)) + numeric
      indicators -> TruncatedSVD to 50 dense dimensions     (Table 4.7)
  8.  C-SMOTE streaming oversampling (k=5, reservoir>=100,
      ADWIN delta=0.002) -> exactly 6,794 ham vs 6,794 scam (Figure 4.1)
  9.  Save cleaned CSV + dense balanced feature matrix.

Run:  python preprocess_sms.py
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from c_smote import csmote_balance_stream

# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #
DATA_PATH = "data/sms_spam_smishing.csv"   # <- path to the Mendeley CSV
FIG_DIR = "figures/sms"
OUT_DIR = "processed"
RANDOM_STATE = 42

TFIDF_MAX_FEATURES = 200      # Table 4.7 (SMS vocabulary size)
TFIDF_NGRAM_RANGE = (1, 2)    # unigrams + bigrams
SVD_COMPONENTS = 50           # dense dimensions fed to C-SMOTE / ARF
TARGET_COUNT = 6_794          # exact per-class target after C-SMOTE balancing

sns.set_style("whitegrid")
PALETTE = {"teal": "#3BAEA0", "orange": "#E8745B", "purple": "#7B5EA7"}

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------- #
# NLTK stopwords (downloaded once)
# --------------------------------------------------------------------- #
import nltk

for pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Lowercase; strip URLs, e-mail addresses, phone numbers,
    non-alphabetic characters; remove English stopwords."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def main() -> None:
    # =================================================================== #
    # 1. LOAD + STRUCTURE (Figures D.1, D.2)
    # =================================================================== #
    df = pd.read_csv(DATA_PATH)
    print("SMS Dataset:")
    print(df.info())
    print(df.head(), "\n")

    # =================================================================== #
    # 2. RAW DATA VISUALISATION (Appendix C)
    # =================================================================== #
    # --- Figure C.1: pie + bar of the three-class label distribution ---
    counts = df["LABEL"].str.lower().value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
              colors=[PALETTE["purple"], PALETTE["teal"], PALETTE["orange"]])
    ax[0].set_title("Label")
    bars = ax[1].bar(counts.index, counts.values,
                     color=[PALETTE["purple"], PALETTE["teal"], PALETTE["orange"]])
    ax[1].bar_label(bars, fmt="%d", fontweight="bold")
    ax[1].set_title("Message Count by Label")
    ax[1].set_xlabel("Label"); ax[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/C1_label_pie_bar.png", dpi=150)
    plt.close(fig)

    # --- Figure C.2: URL / EMAIL / PHONE presence per class ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax_i, col in zip(axes, ["URL", "EMAIL", "PHONE"]):
        ct = pd.crosstab(df["LABEL"].str.lower(), df[col])
        ct.plot(kind="bar", ax=ax_i, color=["#BDBDBD", PALETTE["purple"]], rot=0)
        ax_i.set_title(f"{col} by Label"); ax_i.set_xlabel("Label"); ax_i.set_ylabel("Count")
        ax_i.legend(title=col, labels=["No", "Yes"])
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/C2_url_email_phone.png", dpi=150)
    plt.close(fig)

    # --- Figure C.3: message length distribution by class ---
    df["msg_len"] = df["TEXT"].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lbl, color in zip(["ham", "spam", "smishing"],
                          ["#8FA8FF", "#7BE0AD", "#8B97A8"]):
        sns.histplot(df.loc[df["LABEL"].str.lower() == lbl, "msg_len"],
                     bins=60, color=color, label=lbl, ax=ax, alpha=0.75)
    ax.set_title("Message Length Distribution by Class")
    ax.set_xlabel("Message Length (characters)"); ax.set_ylabel("Count")
    ax.legend(title="Class")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/C3_msg_length_by_class.png", dpi=150)
    plt.close(fig)

    # =================================================================== #
    # 3. CLEANING: nulls + duplicates (Figures D.3, D.4)
    # =================================================================== #
    print("Total null values per column:")
    print(df.isnull().sum(), "\n")
    df = df.dropna(subset=["TEXT"])               # drop records missing TEXT

    n_dupes = df.duplicated().sum()
    print(f"Duplicate rows: {n_dupes}")           # 2,169 expected
    df = df.drop_duplicates().reset_index(drop=True)

    # =================================================================== #
    # 4. BINARY LABEL STANDARDISATION (Table 4.6)
    #    ham -> 0 (legitimate)   |   spam + smishing -> 1 (scam)
    #    Both "spam" and "smishing" are merged into a single "scam" class.
    # =================================================================== #
    label_lower = df["LABEL"].str.lower().str.strip()
    df["label"] = label_lower.isin(["spam", "smishing"]).astype(int)

    # Yes/No indicator columns -> binary 1/0
    for col in ["URL", "EMAIL", "PHONE"]:
        df[col] = (df[col].astype(str).str.strip().str.lower()
                   .isin(["yes", "y", "1", "true"])).astype(int)

    # =================================================================== #
    # 5. TEXT CLEANING + 6. FEATURE ENGINEERING (Figures D.5–D.7)
    # =================================================================== #
    df["clean_text"] = df["TEXT"].apply(clean_text)
    df["msg_len"] = df["TEXT"].astype(str).str.len()
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    clean_cols = ["label", "clean_text", "URL", "EMAIL", "PHONE", "msg_len"]
    sms_clean = df[clean_cols]
    print("\nSMS structure after cleaning:")
    print(sms_clean.info())
    print("\nSMS first five rows after cleaning:")
    print(sms_clean.head())

    n_legit = int((sms_clean["label"] == 0).sum())
    n_scam = int((sms_clean["label"] == 1).sum())
    print(f"\nAfter cleaning: {len(sms_clean):,} messages "
          f"({n_legit:,} legitimate | {n_scam:,} scam)")

    # --- Figure D.7: length by binary label + binary label count ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for lbl, color, name in [(0, PALETTE["teal"], "Legitimate / Ham (0)"),
                             (1, PALETTE["orange"], "Scam (1)")]:
        sns.histplot(sms_clean.loc[sms_clean["label"] == lbl, "msg_len"],
                     bins=60, color=color, label=name, ax=ax[0], alpha=0.8)
    ax[0].set_title("Message Length by Binary Label")
    ax[0].set_xlabel("Message Length (characters)"); ax[0].legend(title="Label")
    bars = ax[1].bar(["Legitimate (0)", "Scam (1)"], [n_legit, n_scam],
                     color=[PALETTE["teal"], PALETTE["orange"]])
    ax[1].bar_label(bars, fmt="{:,.0f}", fontweight="bold")
    ax[1].set_title("Binary Label Count"); ax[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/D7_binary_label_after_cleaning.png", dpi=150)
    plt.close(fig)

    # =================================================================== #
    # 7. TF-IDF (200 n-grams) + indicators -> TruncatedSVD (50 dims)
    # =================================================================== #
    # NOTE (Section 4.3.5): in the streaming experiment the vocabulary is
    # learned once from an early portion of the stream and then frozen.
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                            ngram_range=TFIDF_NGRAM_RANGE)
    X_text = tfidf.fit_transform(sms_clean["clean_text"])

    from scipy.sparse import csr_matrix, hstack
    X_num = csr_matrix(sms_clean[["URL", "EMAIL", "PHONE", "msg_len"]]
                       .to_numpy(dtype=np.float64))
    X_sparse = hstack([X_text, X_num]).tocsr()

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    X_dense = svd.fit_transform(X_sparse)        # (n, 50) dense matrix
    y = sms_clean["label"].to_numpy()

    # =================================================================== #
    # 8. C-SMOTE STREAMING OVERSAMPLING (k=5, reservoir>=100, delta=0.002)
    #    Fixed target: exactly TARGET_COUNT per class (6,794 vs 6,794).
    #    Both ham and scam are trimmed / oversampled to reach TARGET_COUNT.
    # =================================================================== #
    print("\nApplying C-SMOTE in the streaming pipeline ...")
    X_bal, y_bal, sampler = csmote_balance_stream(
        X_dense, y, minority_label=1,
        k_neighbors=5, min_size_minority=100, adwin_delta=0.002,
        chunk=2_000, random_state=RANDOM_STATE,
        target_count=TARGET_COUNT,
    )
    before = np.bincount(y, minlength=2)
    after = np.bincount(y_bal, minlength=2)
    print(f"Before C-SMOTE -> legit: {before[0]:,} | scam: {before[1]:,}")
    print(f"After  C-SMOTE -> legit: {after[0]:,} | scam: {after[1]:,} "
          f"(synthetic: {sampler.n_synthetic:,}, drifts: {sampler.n_drifts})")

    # --- Figure 4.1: before vs after C-SMOTE ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for a, cnt, title in [
        (ax[0], before, "Before C-SMOTE\n(SMS Dataset)"),
        (ax[1], after,  f"After C-SMOTE\n(SMS Dataset)\n{TARGET_COUNT:,} vs {TARGET_COUNT:,}"),
    ]:
        bars = a.bar(["Legitimate / Ham (0)", "Spam + Smishing / Scam (1)"], cnt,
                     color=[PALETTE["teal"], PALETTE["orange"]])
        a.bar_label(bars, fmt="{:,.0f}", fontweight="bold")
        a.set_title(title); a.set_xlabel("Label"); a.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/Fig4_1_before_after_csmote.png", dpi=150)
    plt.close(fig)

    # =================================================================== #
    # 9. SAVE OUTPUTS
    # =================================================================== #
    sms_clean.to_csv(f"{OUT_DIR}/sms_cleaned.csv", index=False)
    np.savez_compressed(f"{OUT_DIR}/sms_stream_balanced.npz",
                        X=X_bal, y=y_bal)
    print(f"\nSaved: {OUT_DIR}/sms_cleaned.csv, "
          f"{OUT_DIR}/sms_stream_balanced.npz, figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()