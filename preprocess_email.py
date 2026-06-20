import os
import re
from collections import Counter

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
DATA_PATH = "data/phishing_email.csv"      # <- path to the Kaggle CSV
FIG_DIR = "figures/email"
OUT_DIR = "processed"
RANDOM_STATE = 42

TFIDF_MAX_FEATURES = 300      # Table 4.7 (emails are longer than SMS)
TFIDF_NGRAM_RANGE = (1, 2)    # bigrams capture phrases like "click here"
SVD_COMPONENTS = 50

sns.set_style("whitegrid")
PALETTE = {"teal": "#3BAEA0", "orange": "#E8745B"}

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------- #
# NLTK stopwords
# --------------------------------------------------------------------- #
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def main() -> None:
    # =================================================================== #
    # 1. LOAD + STRUCTURE (Figures D.17, D.18)
    # =================================================================== #
    df = pd.read_csv(DATA_PATH)
    print("Email Dataset:")
    print(df.info())
    print(df.head(), "\n")

    # =================================================================== #
    # 2. RAW DATA VISUALISATION (Appendix C)
    # =================================================================== #
    # --- Figure C.10: pie + bar of Email Type ---
    counts = df["Email Type"].value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
              colors=[PALETTE["orange"], PALETTE["teal"]])
    ax[0].set_title("Email Type")
    bars = ax[1].bar(counts.index, counts.values,
                     color=[PALETTE["orange"], PALETTE["teal"]])
    ax[1].bar_label(bars, fmt="{:,.0f}", fontweight="bold")
    ax[1].set_title("Email Count by Type")
    ax[1].set_xlabel("Email Type"); ax[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/C10_email_type_pie_bar.png", dpi=150)
    plt.close(fig)

    # --- Figure C.11: body length histogram by type ---
    df["body_len"] = df["Email Text"].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for lbl, color in [("Safe Email", PALETTE["orange"]),
                       ("Phishing Email", PALETTE["teal"])]:
        sns.histplot(df.loc[df["Email Type"] == lbl, "body_len"].clip(upper=4000),
                     bins=60, color=color, label=lbl, ax=ax, alpha=0.8)
    ax.set_title("Email Body Length Distribution by Type")
    ax.set_xlabel("Email Body Length (characters)"); ax.set_ylabel("Count")
    ax.legend(title="Type")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/C11_body_length_hist.png", dpi=150)
    plt.close(fig)

    # --- Figure C.12: top-10 most frequent words per class ---
    def top_words(series: pd.Series, n: int = 10) -> Counter:
        c: Counter = Counter()
        for txt in series.dropna().astype(str):
            c.update(t for t in NON_ALPHA_RE.sub(" ", txt.lower()).split()
                     if t not in STOPWORDS and len(t) > 2)
        return Counter(dict(c.most_common(n)))

    safe_top = top_words(df.loc[df["Email Type"] == "Safe Email", "Email Text"])
    phish_top = top_words(df.loc[df["Email Type"] == "Phishing Email", "Email Text"])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for a, top, title, color in [(ax[0], safe_top, "Safe Email", PALETTE["orange"]),
                                 (ax[1], phish_top, "Phishing Email", PALETTE["teal"])]:
        words, freqs = zip(*top.most_common(10))
        bars = a.barh(list(words)[::-1], list(freqs)[::-1], color=color)
        a.bar_label(bars, fmt="{:,.0f}", fontsize=8)
        a.set_title(title); a.set_xlabel("Frequency")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/C12_top10_words.png", dpi=150)
    plt.close(fig)

    # =================================================================== #
    # 3. NULLS + DUPLICATES (Figures D.19, D.20)
    # =================================================================== #
    print("Total null values per column:")
    print(df.isnull().sum())
    n_missing = df["Email Text"].isnull().sum()
    print(f"\nRemoving {n_missing} records with missing email body text")  # 16 expected
    df = df.dropna(subset=["Email Text"]).reset_index(drop=True)

    n_dupes = df.duplicated(subset=["Email Text"]).sum()
    print(f"Duplicate rows: {n_dupes}")            # 0 expected
    df = df.drop_duplicates(subset=["Email Text"]).reset_index(drop=True)

    # =================================================================== #
    # 4. BINARY LABEL STANDARDISATION (Table 4.6)
    #    Safe Email -> 0   |   Phishing Email -> 1
    # =================================================================== #
    df["label_binary"] = (df["Email Type"].str.strip()
                          == "Phishing Email").astype(int)

    # =================================================================== #
    # 5. TEXT CLEANING + FEATURES (Figures D.21–D.23)
    # =================================================================== #
    df["cleaned_text"] = df["Email Text"].apply(clean_text)
    df["body_len"] = df["Email Text"].astype(str).str.len()
    df["word_count"] = df["cleaned_text"].str.split().str.len()
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)

    print("\nEmail structure after cleaning:")
    print(df.info())
    print("\nEmail first five rows after cleaning:")
    print(df[["Email Type", "label_binary", "cleaned_text",
              "body_len", "word_count"]].head())

    n_safe = int((df["label_binary"] == 0).sum())
    n_phish = int((df["label_binary"] == 1).sum())
    print(f"\nAfter cleaning: {len(df):,} emails "
          f"({n_safe:,} safe | {n_phish:,} phishing)")

    # --- Figure D.23: class distribution after cleaning ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for lbl, color, name in [(0, PALETTE["teal"], "Safe Email (0)"),
                             (1, PALETTE["orange"], "Phishing Email (1)")]:
        sns.histplot(df.loc[df["label_binary"] == lbl, "body_len"].clip(upper=10000),
                     bins=60, color=color, label=name, ax=ax[0], alpha=0.8)
    ax[0].set_title("Body Length by Binary Label")
    ax[0].set_xlabel("Email Body Length (characters)"); ax[0].legend(title="Label")
    bars = ax[1].bar(["Safe (0)", "Phishing (1)"], [n_safe, n_phish],
                     color=[PALETTE["teal"], PALETTE["orange"]])
    ax[1].bar_label(bars, fmt="{:,.0f}", fontweight="bold")
    ax[1].set_title("Binary Label Count"); ax[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/D23_class_after_cleaning.png", dpi=150)
    plt.close(fig)

    # =================================================================== #
    # 6. TF-IDF (300 n-grams) + numeric features -> TruncatedSVD (50)
    # =================================================================== #
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                            ngram_range=TFIDF_NGRAM_RANGE)
    X_text = tfidf.fit_transform(df["cleaned_text"])

    from scipy.sparse import csr_matrix, hstack
    X_num = csr_matrix(df[["body_len", "word_count"]].to_numpy(dtype=np.float64))
    X_sparse = hstack([X_text, X_num]).tocsr()

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    X_dense = svd.fit_transform(X_sparse)
    y = df["label_binary"].to_numpy()

    # =================================================================== #
    # 7. C-SMOTE STREAMING OVERSAMPLING (k=5, reservoir>=100, delta=0.002)
    # =================================================================== #
    print("\nApplying C-SMOTE in the streaming pipeline ...")
    X_bal, y_bal, sampler = csmote_balance_stream(
        X_dense, y, minority_label=1,
        k_neighbors=5, min_size_minority=100, adwin_delta=0.002,
        chunk=2_000, random_state=RANDOM_STATE,
    )
    before = np.bincount(y, minlength=2)
    after = np.bincount(y_bal, minlength=2)
    print(f"Before C-SMOTE -> safe: {before[0]:,} | phishing: {before[1]:,}")
    print(f"After  C-SMOTE -> safe: {after[0]:,} | phishing: {after[1]:,} "
          f"(synthetic: {sampler.n_synthetic:,}, drifts: {sampler.n_drifts})")

    # --- Figure 4.3: before vs after C-SMOTE ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for a, cnt, title in [(ax[0], before, "Before C-SMOTE\n(Email Dataset)"),
                          (ax[1], after, "After C-SMOTE\n(Email Dataset)")]:
        bars = a.bar(["Legitimate / Safe (0)", "Fraud / Scam (1)"], cnt,
                     color=[PALETTE["teal"], PALETTE["orange"]])
        a.bar_label(bars, fmt="{:,.0f}", fontweight="bold")
        a.set_title(title); a.set_xlabel("Label"); a.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/Fig4_3_before_after_csmote.png", dpi=150)
    plt.close(fig)

    # =================================================================== #
    # 8. SAVE OUTPUTS
    # =================================================================== #
    df[["label_binary", "cleaned_text", "body_len", "word_count"]].to_csv(
        f"{OUT_DIR}/email_cleaned.csv", index=False)
    np.savez_compressed(f"{OUT_DIR}/email_stream_balanced.npz",
                        X=X_bal, y=y_bal)
    print(f"\nSaved: {OUT_DIR}/email_cleaned.csv, "
          f"{OUT_DIR}/email_stream_balanced.npz, figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
