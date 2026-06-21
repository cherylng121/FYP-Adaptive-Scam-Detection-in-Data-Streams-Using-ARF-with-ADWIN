from collections import deque

import numpy as np

try:
    # River >= 0.21 keeps ADWIN under river.drift
    from river.drift import ADWIN
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'river' package is required for the ADWIN-monitored reservoir.\n"
        "Install it with:  pip install river"
    ) from exc


class CSMOTE:
    """Continuous SMOTE with an ADWIN-monitored rolling reservoir."""

    def __init__(
        self,
        k_neighbors: int = 5,          # canonical SMOTE default (Chawla et al., 2002)
        min_size_minority: int = 100,  # minimum reservoir size before synthesis (Bernardo et al., 2020)
        adwin_delta: float = 0.002,    # classic ADWIN default (Bifet & Gavalda, 2007)
        max_reservoir_size: int = 1000,  # fixed-size rolling reservoir => low memory
        minority_label: int = 1,       # scam / fraud class
        random_state: int = 42,
    ):
        self.k_neighbors = k_neighbors
        self.min_size_minority = min_size_minority
        self.adwin_delta = adwin_delta
        self.max_reservoir_size = max_reservoir_size
        self.minority_label = minority_label
        self.rng = np.random.default_rng(random_state)

        # Rolling reservoir of recent minority-class feature vectors
        self.reservoir: deque = deque(maxlen=max_reservoir_size)

        # ADWIN monitors the label stream; a confirmed drift means the class
        # distribution has shifted -> clear the reservoir.
        self.adwin = ADWIN(delta=adwin_delta)

        # Bookkeeping
        self.n_seen = 0
        self.n_minority = 0
        self.n_majority = 0
        self.n_synthetic = 0
        self.n_drifts = 0

    # ------------------------------------------------------------------ #
    # Streaming interface
    # ------------------------------------------------------------------ #
    def learn_one(self, x: np.ndarray, y: int) -> None:
        """Observe one real record from the stream (prequential order)."""
        self.n_seen += 1
        if y == self.minority_label:
            self.n_minority += 1
            self.reservoir.append(np.asarray(x, dtype=np.float64))
        else:
            self.n_majority += 1

        # Feed the label into ADWIN. A confirmed change in the label
        # distribution is treated as drift of the minority concept.
        self.adwin.update(int(y))
        if self.adwin.drift_detected:
            self.n_drifts += 1
            self.reservoir.clear()  # rebuild from subsequent real instances

    def can_generate(self) -> bool:
        """Synthesis is allowed only once the reservoir is large enough."""
        return len(self.reservoir) >= self.min_size_minority

    # ------------------------------------------------------------------ #
    # Synthetic sample generation (SMOTE interpolation, k = 5)
    # ------------------------------------------------------------------ #
    def generate_one(self) -> np.ndarray:
        """Generate a single synthetic minority instance."""
        return self.generate_batch(1)[0]

    # Maximum rows processed in one vectorised pass inside generate_batch.
    # Peak memory per sub-batch = GEN_BATCH_SIZE * max_reservoir_size * d * 8 bytes.
    # At 4 096 × 1 000 × 8 × 8 B ≈ 250 MB — safe on any modern machine.
    GEN_BATCH_SIZE: int = 4_096

    def generate_batch(self, n: int) -> np.ndarray:
        if not self.can_generate():
            raise RuntimeError(
                f"Reservoir too small ({len(self.reservoir)} < "
                f"{self.min_size_minority}); cannot generate synthetic samples yet."
            )

        # Fast path: small enough to handle in one shot.
        if n <= self.GEN_BATCH_SIZE:
            result = self._generate_batch_core(n)
            self.n_synthetic += n
            return result

        # Slow path: break into sub-batches to cap peak memory usage.
        parts = []
        remaining = n
        while remaining > 0:
            sub_n = min(remaining, self.GEN_BATCH_SIZE)
            parts.append(self._generate_batch_core(sub_n))
            remaining -= sub_n
        self.n_synthetic += n
        return np.vstack(parts)

    def _generate_batch_core(self, n: int) -> np.ndarray:
        R = np.stack(self.reservoir)                     # (m, d)
        m = R.shape[0]
        k = min(self.k_neighbors, m - 1)

        base_idx = self.rng.integers(0, m, size=n)
        base = R[base_idx]                               # (n, d)

        # Pairwise distances base -> reservoir  shape: (n, m)
        # Peak allocation: n * m * d * 8 bytes = 4096 * 1000 * 8 * 8 ≈ 250 MB
        d2 = ((base[:, None, :] - R[None, :, :]) ** 2).sum(axis=2)
        d2[np.arange(n), base_idx] = np.inf              # exclude self

        # k nearest neighbours for each base, choose one at random
        knn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]   # (n, k)
        chosen = knn_idx[np.arange(n), self.rng.integers(0, k, size=n)]
        neighbour = R[chosen]                            # (n, d)

        gap = self.rng.random((n, 1))
        return base + gap * (neighbour - base)


# ---------------------------------------------------------------------- #
# Convenience wrapper used by the three preprocessing scripts
# ---------------------------------------------------------------------- #
def csmote_balance_stream(
    X: np.ndarray,
    y: np.ndarray,
    minority_label: int = 1,
    k_neighbors: int = 5,
    min_size_minority: int = 100,
    adwin_delta: float = 0.002,
    chunk: int = 50_000,
    random_state: int = 42,
    verbose: bool = True,
    target_count: int = None,
):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    majority_label = 1 - minority_label   # assumes binary {0, 1}

    sampler = CSMOTE(
        k_neighbors=k_neighbors,
        min_size_minority=min_size_minority,
        adwin_delta=adwin_delta,
        minority_label=minority_label,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)

    synth_blocks = []
    seen_min, seen_maj = 0, 0
    n = len(y)

    # ------------------------------------------------------------------ #
    # Streaming pass: observe every real record and accumulate synthetics
    # ------------------------------------------------------------------ #
    for start in range(0, n, chunk):
        end = min(start + chunk, n)

        # ---- observe real records one at a time (streaming protocol) ----
        for i in range(start, end):
            sampler.learn_one(X[i], y[i])
        seen_min = sampler.n_minority
        seen_maj = sampler.n_majority

        # ---- decide how many synthetics are needed so far ----
        if target_count is not None:
            # Top the minority class up toward target_count progressively.
            # Do not overshoot: cap the running synthetic total at
            # max(0, target_count - seen_min).
            needed_total = max(0, target_count - seen_min)
            deficit = needed_total - sampler.n_synthetic
        else:
            # Original behaviour: match the majority count dynamically.
            deficit = (seen_maj - seen_min) - sampler.n_synthetic

        if deficit > 0 and sampler.can_generate():
            synth_blocks.append(sampler.generate_batch(int(deficit)))

        if verbose:
            print(
                f"  [C-SMOTE] processed {end:>9,}/{n:,} | "
                f"minority {seen_min:,} | majority {seen_maj:,} | "
                f"synthetic {sampler.n_synthetic:,} | drifts {sampler.n_drifts}"
            )

    # ------------------------------------------------------------------ #
    # Assemble the minority side (real + synthetic)
    # ------------------------------------------------------------------ #
    min_mask = (y == minority_label)
    X_min_real = X[min_mask]
    y_min_real = y[min_mask]

    if synth_blocks:
        X_syn = np.vstack(synth_blocks)
        y_syn = np.full(len(X_syn), minority_label, dtype=np.int64)
        X_min_all = np.vstack([X_min_real, X_syn])
        y_min_all = np.concatenate([y_min_real, y_syn])
    else:
        X_min_all = X_min_real
        y_min_all = y_min_real

    # ------------------------------------------------------------------ #
    # Assemble the majority side
    # ------------------------------------------------------------------ #
    maj_mask = (y == majority_label)
    X_maj_all = X[maj_mask]
    y_maj_all = y[maj_mask]

    # ------------------------------------------------------------------ #
    # Trim both sides to exactly target_count (if specified)
    # ------------------------------------------------------------------ #
    if target_count is not None:
        # --- minority: trim if oversampled beyond target ---
        if len(y_min_all) > target_count:
            # Keep all real records first; trim from synthetics only.
            n_real_min = len(X_min_real)
            n_keep_syn = target_count - n_real_min
            if n_keep_syn < 0:
                # Even real minority exceeds target; sub-sample real records.
                idx = rng.choice(n_real_min, size=target_count, replace=False)
                X_min_all = X_min_real[idx]
                y_min_all = y_min_real[idx]
            else:
                idx_syn = rng.choice(len(X_syn), size=n_keep_syn, replace=False)
                X_min_all = np.vstack([X_min_real, X_syn[idx_syn]])
                y_min_all = np.concatenate(
                    [y_min_real, np.full(n_keep_syn, minority_label, dtype=np.int64)]
                )
        elif len(y_min_all) < target_count:
            # Still short after streaming pass — generate the remainder now.
            extra = target_count - len(y_min_all)
            if sampler.can_generate():
                X_extra = sampler.generate_batch(int(extra))
                y_extra = np.full(extra, minority_label, dtype=np.int64)
                X_min_all = np.vstack([X_min_all, X_extra])
                y_min_all = np.concatenate([y_min_all, y_extra])
                if verbose:
                    print(f"  [C-SMOTE] top-up: generated {extra:,} extra "
                          f"synthetic samples to reach target {target_count:,}")

        # --- majority: sub-sample randomly to target_count if too large ---
        if len(y_maj_all) > target_count:
            idx = rng.choice(len(y_maj_all), size=target_count, replace=False)
            X_maj_all = X_maj_all[idx]
            y_maj_all = y_maj_all[idx]
        elif len(y_maj_all) < target_count:
            # Majority is smaller than target — oversample with replacement.
            extra = target_count - len(y_maj_all)
            idx = rng.choice(len(y_maj_all), size=extra, replace=True)
            X_maj_all = np.vstack([X_maj_all, X_maj_all[idx]])
            y_maj_all = np.concatenate(
                [y_maj_all, np.full(extra, majority_label, dtype=np.int64)]
            )
            if verbose:
                print(f"  [C-SMOTE] majority top-up: duplicated {extra:,} "
                      f"majority samples to reach target {target_count:,}")

    # ------------------------------------------------------------------ #
    # Combine and shuffle
    # ------------------------------------------------------------------ #
    X_bal = np.vstack([X_maj_all, X_min_all])
    y_bal = np.concatenate([y_maj_all, y_min_all])

    # Shuffle so the returned array is not majority-then-minority ordered.
    shuffle_idx = rng.permutation(len(y_bal))
    X_bal = X_bal[shuffle_idx]
    y_bal = y_bal[shuffle_idx]

    return X_bal, y_bal, sampler