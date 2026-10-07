"""Statistics for the time-aware results: confidence intervals, paired comparisons, Friedman / Nemenyi.

Why the intervals are built this way
------------------------------------
Samples from the same batch are near-identical "twins" (notebook 01), so a bootstrap that resamples samples
as if they were independent gives intervals that are far too narrow. The **primary** interval is therefore a
*two-level* bootstrap: resample the test batches with replacement (between-batch variability) and, inside every
drawn batch, resample its samples (within-batch variability). The **conditional** interval resamples only the
samples of the fixed test batches and is reported as secondary. All comparisons are *paired*: every model is
scored on exactly the same resampled samples.

Pre-declared verdict rule for a difference A - B (declared before looking at any result)
  clear         the two-level 95% interval excludes 0
  suggestive    only the conditional interval excludes 0
  inconclusive  neither excludes 0

With only 7 (P1) or 9 (P3) test batches an exact paired test cannot reach p < 0.05 after correcting for many
comparisons (the smallest possible two-sided p is 2/2^7 = 0.016 for 7 batches), so effect sizes with intervals,
not p-values, are the main evidence. p-values are reported uncorrected and with the Holm adjustment.
"""
import numpy as np
from scipy import stats

N_CLASSES = 6


def fast_macro_f1(y_true, y_pred):
    """Macro-F1 over the gases present in ``y_true`` (identical to ``evaluate.compute_metrics``), vectorised."""
    conf = np.bincount((y_true - 1) * N_CLASSES + (y_pred - 1), minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    tp = np.diag(conf).astype(float)
    support, predicted = conf.sum(axis=1), conf.sum(axis=0)
    prec = tp / np.maximum(predicted, 1)
    rec = tp / np.maximum(support, 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return float(f1[support > 0].mean())


def per_batch_f1(y, pred, batch, batches):
    """Macro-F1 of one prediction vector on each of the given batches."""
    return np.array([fast_macro_f1(y[batch == b], pred[batch == b]) for b in batches])


def bootstrap_f1(y, preds, batch, batches, n_boot=2000, seed=42):
    """Paired within-batch bootstrap. Returns ``F`` of shape (n_boot, n_batches, n_models) and the model names."""
    names = list(preds)
    P = {n: np.asarray(preds[n]) for n in names}
    rng = np.random.RandomState(seed)
    idx = {b: np.where(batch == b)[0] for b in batches}
    F = np.empty((n_boot, len(batches), len(names)))
    for r in range(n_boot):
        for bi, b in enumerate(batches):
            ids = idx[b][rng.randint(0, len(idx[b]), len(idx[b]))]       # same resample for every model -> paired
            yt = y[ids]
            for mi, n in enumerate(names):
                F[r, bi, mi] = fast_macro_f1(yt, P[n][ids])
    return F, names


def two_level_means(F, seed=7):
    """Mean over test batches when the batches themselves are also resampled with replacement. Shape (n_boot, n_models)."""
    R, B, _ = F.shape
    rng = np.random.RandomState(seed)
    draw = rng.randint(0, B, size=(R, B))
    return F[np.arange(R)[:, None], draw, :].mean(axis=1)


def percentile_ci(values, level=0.95):
    a = (1 - level) / 2 * 100
    return float(np.percentile(values, a)), float(np.percentile(values, 100 - a))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def verdict(ci_two_level, ci_conditional):
    if ci_two_level[0] > 0 or ci_two_level[1] < 0:
        return "clear"
    if ci_conditional[0] > 0 or ci_conditional[1] < 0:
        return "suggestive"
    return "inconclusive"


def compare(F, names, a, b, point_by_batch):
    """Paired comparison A - B from bootstrap array ``F`` and the observed per-batch macro-F1 ``point_by_batch``."""
    ia, ib = names.index(a), names.index(b)
    D = F[:, :, ia] - F[:, :, ib]
    d_batch = point_by_batch[a] - point_by_batch[b]
    cond = percentile_ci(D.mean(axis=1)); two = percentile_ci(two_level_means(D[:, :, None])[:, 0])
    try:
        p = float(stats.wilcoxon(d_batch, method="exact").pvalue) if np.any(d_batch != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"A": a, "B": b, "diff": float(d_batch.mean()), "ci_two_level": two, "ci_conditional": cond,
            "batches_A_better": int((d_batch > 0).sum()), "batches_B_better": int((d_batch < 0).sum()), "n_batches": len(d_batch),
            "wilcoxon_p": p, "verdict": verdict(two, cond)}


def mcnemar_exact(y, pred_a, pred_b):
    """Exact McNemar test on per-sample correctness (pooled samples; treats samples as independent, so optimistic)."""
    ca, cb = pred_a == y, pred_b == y
    b_only, a_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(stats.binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return {"A_correct_B_wrong": b_only, "A_wrong_B_correct": a_only, "p": p}


def friedman_nemenyi(M, names, alpha=0.05):
    """Friedman test over blocks (rows = test batches, columns = models) and the Nemenyi critical difference.

    Higher score = better; rank 1 is best. Returns the average ranks, the Friedman statistic / p-value and the
    critical difference: two models whose average ranks differ by less than ``cd`` are not significantly different.
    """
    M = np.asarray(M, dtype=float); N, k = M.shape
    ranks = np.vstack([stats.rankdata(-row) for row in M])
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * N)))
    return {"names": list(names), "avg_rank": ranks.mean(axis=0), "chi2": float(chi2), "p": float(p), "cd": cd, "n_blocks": N}
