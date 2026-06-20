"""
preprocess_financial.py
=======================
Dataset 2 — "Financial Transactions Dataset for Fraud Detection"
(Kumar, 2025, Kaggle). Five million synthetic bank transactions,
Jan 2023 – Jan 2024, fraud rate ~3.6%.

Pipeline (Chapter 3 §3.2.1, Chapter 4 §4.3.4(2), Tables 4.3 / 4.4):
  1.  Load and inspect structure / first rows               (Figures D.8, D.9)
  2.  Raw visualisation: is_fraud bar, transaction-type
      pie+bar, amount histogram, risk-score boxplots,
      fraud rate by type                                    (Figures C.4–C.9)
  3.  Column selection — keep the nine evidence-based
      columns (Table 4.3), drop the rest (Table 4.4)
  4.  Null check (fraud_type nulls expected), duplicates    (Figures D.12, D.13)
  5.  Outlier removal on `amount` using the IQR rule
      (workflow Figure 3.1: "IQR on financial")
  6.  Binary label standardisation (is_fraud -> 0/1)        (Table 4.6)
  7.  CHRONOLOGICAL SORT by timestamp (drift simulation);
      temporal features hour / day-of-week / month;
      ordinal encoding of transaction_type;
      StandardScaler on continuous features                 (Figures D.14–D.16)
      (timestamp itself is NOT scaled — kept only for stream
       ordering and the false-drift calendar comparison)
  8.  C-SMOTE streaming oversampling (k=5, reservoir>=100,
      ADWIN delta=0.002): 179,553 -> ~majority count        (Figure 4.2)
  9.  Save cleaned parquet/CSV + balanced feature matrix.

NOTE ON RUNTIME: this dataset has 5,000,000 rows. The streaming C-SMOTE
loop is pure Python and may take a long while on the full file. Set
SAMPLE_FRAC < 1.0 for a quick pilot run, and 1.0 for the final PSM run.

Run:  python preprocess_financial.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from c_smote import csmote_balance_stream

# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #
DATA_PATH = "data/financial_transactions.csv"   # <- path to the Kaggle CSV
FIG_DIR = "figures/financial"
OUT_DIR = "processed"
RANDOM_STATE = 42
SAMPLE_FRAC = 1.0          # set e.g. 0.05 for a quick pilot run
APPLY_IQR_OUTLIERS = True  # IQR rule on `amount` (Figure 3.1 workflow)

# Table 4.3 — nine selected columns
SELECTED = [
    "transaction_id",            # record identifier (not a model feature)
    "timestamp",                 # stream ordering + false-drift analysis
    "amount",                    # numeric model feature
    "transaction_type",          # categorical -> ordinal encoded
    "velocity_score",            # numeric model feature
    "spending_deviation_score",  # numeric model feature
    "geo_anomaly_score",         # numeric model feature
    "is_fraud",                  # binary target label
    "fraud_type",                # post-hoc analysis label (not a feature)
]

# Table 4.3 — ordinal mapping of the four transaction categories
TXN_TYPE_ORDINAL = {"deposit": 0, "payment": 1, "transfer": 2, "withdrawal": 3}

CONTINUOUS = ["amount", "velocity_score",
              "spending_deviation_score", "geo_anomaly_score"]

sns.set_style("whitegrid")
PALETTE = {"teal": "#3BAEA0", "orange": "#E8745B", "blue": "#5B8DEF",
           "green": "#58C9A4", "slate": "#5D6D7E", "gold": "#F0B429"}

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def iqr_outlier_mask(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Boolean mask of rows that lie INSIDE the IQR fences."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return series.between(lo, hi)


def risk_score_boxplots(df: pd.DataFrame, path: str, title_suffix: str = "") -> None:
    """Figures C.7 / D.16 — boxplots of the three behavioural risk scores."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    score_cols = [("spending_deviation_score", "Spending Deviation Score"),
                  ("velocity_score", "Velocity Score"),
                  ("geo_anomaly_score", "Geo Anomaly Score")]
    for ax, (col, title) in zip(axes, score_cols):
        sns.boxplot(data=df, x="is_fraud", y=col, hue="is_fraud", ax=ax,
                    palette=[PALETTE["teal"], PALETTE["orange"]], legend=False)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Legitimate", "Fraudulent"])
        ax.set_title(title + title_suffix); ax.set_xlabel(""); ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    # =================================================================== #
    # 1. LOAD + STRUCTURE (Figures D.8, D.9)
    # =================================================================== #
    df = pd.read_csv(DATA_PATH)
    if SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
        print(f"[pilot mode] using {len(df):,} rows ({SAMPLE_FRAC:.0%})")
    print("Financial Dataset:")
    print(df.info())
    print(df.head(), "\n")

    df["is_fraud"] = df["is_fraud"].astype(int)

    # =================================================================== #
    # 2. RAW DATA VISUALISATION (Appendix C)
    # =================================================================== #
    # --- Figure C.4: is_fraud distribution before resampling ---
    counts = df["is_fraud"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values,
                  color=[PALETTE["blue"], PALETTE["orange"]])
    ax.bar_label(bars, fmt="{:,.0f}", fontweight="bold")
    ax.set_title("Distribution of is_fraud (Before C-SMOTE)")
    ax.set_xlabel("is_fraud"); ax.set_ylabel("Count")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/C4_is_fraud_bar.png", dpi=150)
    plt.close(fig)

    # --- Figures C.5 / C.8: transaction type pie + bar ---
    tcounts = df["transaction_type"].str.lower().value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax[0].pie(tcounts.values, labels=tcounts.index, autopct="%1.1f%%",
              colors=[PALETTE["blue"], PALETTE["green"],
                      PALETTE["slate"], PALETTE["gold"]],
              wedgeprops=dict(width=0.55))
    ax[0].set_title("Transaction Type")
    bars = ax[1].bar(tcounts.index, tcounts.values,
                     color=[PALETTE["blue"], PALETTE["green"],
                            PALETTE["slate"], PALETTE["gold"]])
    ax[1].bar_label(bars, fmt="{:,.0f}", fontweight="bold")
    ax[1].set_title("Count by Transaction Type")
    ax[1].set_xlabel("Transaction Type"); ax[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/C5_transaction_type_pie_bar.png", dpi=150)
    plt.close(fig)

    # --- Figure C.6: transaction amount distribution by class ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df.loc[df["is_fraud"] == 0, "amount"], bins=80,
                 color="#8FA8FF", label="Legitimate", ax=ax, alpha=0.85)
    sns.histplot(df.loc[df["is_fraud"] == 1, "amount"], bins=80,
                 color=PALETTE["orange"], label="Fraud", ax=ax, alpha=0.85)
    ax.set_title("Transaction Amount Distribution by Class")
    ax.set_xlabel("Transaction Amount (USD)"); ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/C6_amount_histogram.png", dpi=150)
    plt.close(fig)

    # --- Figure C.7: behavioural risk score boxplots (raw) ---
    risk_score_boxplots(df, f"{FIG_DIR}/C7_risk_score_boxplots.png")

    # --- Figure C.9: fraud rate by transaction type ---
    rate = (df.groupby(df["transaction_type"].str.lower())["is_fraud"]
              .mean().sort_values(ascending=False) * 100)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(rate.index, rate.values,
                  color=["#7B5EA7", PALETTE["teal"], PALETTE["orange"], "#F4D58D"])
    ax.bar_label(bars, fmt="%.1f%%", fontweight="bold")
    ax.set_title("Fraud Rate by Transaction Type")
    ax.set_xlabel("Transaction Type"); ax.set_ylabel("Fraud Rate (%)")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/C9_fraud_rate_by_type.png", dpi=150)
    plt.close(fig)

    # =================================================================== #
    # 3. COLUMN SELECTION (Tables 4.3 / 4.4)
    # =================================================================== #
    df = df[[c for c in SELECTED if c in df.columns]].copy()
    print(f"Columns retained ({len(df.columns)}): {list(df.columns)}\n")

    # =================================================================== #
    # 4. NULLS + DUPLICATES (Figures D.12, D.13)
    # =================================================================== #
    print("Total null values per column:")
    print(df.isnull().sum())
    # fraud_type nulls are EXPECTED for legitimate records and harmless,
    # because fraud_type is excluded from model features (Table 4.3).
    model_cols = [c for c in df.columns if c != "fraud_type"]
    df = df.dropna(subset=model_cols)

    n_dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {n_dupes}")        # 0 expected
    df = df.drop_duplicates().reset_index(drop=True)

    # =================================================================== #
    # 5. IQR OUTLIER REMOVAL on `amount` (workflow Figure 3.1)
    # =================================================================== #
    if APPLY_IQR_OUTLIERS:
        before_n = len(df)
        df = df[iqr_outlier_mask(df["amount"])].reset_index(drop=True)
        print(f"IQR outlier removal on 'amount': {before_n - len(df):,} rows removed")

    # =================================================================== #
    # 6. BINARY LABEL (Table 4.6) — is_fraud already 0/1
    # =================================================================== #
    df["is_fraud"] = df["is_fraud"].astype(int)

    # =================================================================== #
    # 7. CHRONOLOGICAL SORT + TEMPORAL FEATURES + ENCODING + SCALING
    # =================================================================== #
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)   # stream order

    # Temporal features extracted from timestamp (Section 4.3.4(2))
    df["txn_hour"] = df["timestamp"].dt.hour.astype(float)
    df["txn_dow"] = df["timestamp"].dt.dayofweek.astype(float)
    df["txn_month"] = df["timestamp"].dt.month.astype(float)

    # Ordinal encoding of transaction_type (Table 4.3, Figure D.14/D.15)
    df["transaction_type_enc"] = (df["transaction_type"].str.lower()
                                  .map(TXN_TYPE_ORDINAL).astype(int))

    # StandardScaler on continuous features BEFORE C-SMOTE, so no single
    # feature's numerical scale biases synthetic sample generation
    # (Section 3.2.1). The timestamp itself is never scaled.
    scaler = StandardScaler()
    df[CONTINUOUS + ["txn_hour", "txn_dow", "txn_month"]] = scaler.fit_transform(
        df[CONTINUOUS + ["txn_hour", "txn_dow", "txn_month"]]
    )

    print("\nFinancial structure after cleaning:")
    print(df.info())
    print("\nFinancial first five rows after cleaning:")
    feature_cols = ["amount", "spending_deviation_score", "velocity_score",
                    "geo_anomaly_score", "txn_hour", "txn_dow", "txn_month",
                    "transaction_type_enc"]
    print(df[feature_cols + ["is_fraud"]].head())

    # --- Figure D.16: risk-score boxplots after cleaning ---
    risk_score_boxplots(df, f"{FIG_DIR}/D16_risk_scores_after_cleaning.png",
                        title_suffix="")

    # =================================================================== #
    # 8. C-SMOTE STREAMING OVERSAMPLING (k=5, reservoir>=100, delta=0.002)
    # =================================================================== #
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["is_fraud"].to_numpy(dtype=np.int64)

    print("\nApplying C-SMOTE in the streaming pipeline "
          "(this may take a while on 5M rows) ...")
    X_bal, y_bal, sampler = csmote_balance_stream(
        X, y, minority_label=1,
        k_neighbors=5, min_size_minority=100, adwin_delta=0.002,
        chunk=250_000, random_state=RANDOM_STATE,
    )
    before = np.bincount(y, minlength=2)
    after = np.bincount(y_bal, minlength=2)
    print(f"Before C-SMOTE -> legit: {before[0]:,} | fraud: {before[1]:,}")
    print(f"After  C-SMOTE -> legit: {after[0]:,} | fraud: {after[1]:,} "
          f"(synthetic: {sampler.n_synthetic:,}, drifts: {sampler.n_drifts})")

    # --- Figure 4.2: before vs after C-SMOTE ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for a, cnt, title in [(ax[0], before, "Before C-SMOTE\n(Financial Dataset)"),
                          (ax[1], after, "After C-SMOTE\n(Financial Dataset)")]:
        bars = a.bar(["Legitimate / Safe (0)", "Fraud / Scam (1)"], cnt,
                     color=[PALETTE["teal"], PALETTE["orange"]])
        a.bar_label(bars, fmt="{:,.0f}", fontweight="bold")
        a.set_title(title); a.set_xlabel("Label"); a.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/Fig4_2_before_after_csmote.png", dpi=150)
    plt.close(fig)

    # =================================================================== #
    # 9. SAVE OUTPUTS
    # =================================================================== #
    df.to_csv(f"{OUT_DIR}/financial_cleaned.csv", index=False)
    np.savez_compressed(f"{OUT_DIR}/financial_stream_balanced.npz",
                        X=X_bal, y=y_bal)
    print(f"\nSaved: {OUT_DIR}/financial_cleaned.csv, "
          f"{OUT_DIR}/financial_stream_balanced.npz, figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
