"""Stable FP64 metrics for comparing the Phase 11 logit distributions."""

import numpy as np


def compare_logits(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("logit shapes differ")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate contains nonfinite logits")
    a = reference.astype(np.float64)
    b = candidate.astype(np.float64)
    a_logsum = float(np.logaddexp.reduce(a))
    b_logsum = float(np.logaddexp.reduce(b))
    p = np.exp(a - a_logsum)
    kl = float(np.sum(p * ((a - a_logsum) - (b - b_logsum))))
    top = int(np.argmax(a))
    return {"kl": max(kl, 0.0), "oracle_top1": top,
            "candidate_top1": int(np.argmax(b)),
            "top1_agree": bool(np.argmax(b) == top),
            "relative_nll_delta": abs(float(b_logsum - b[top]) -
                                      float(a_logsum - a[top])) /
                                  max(abs(float(a_logsum - a[top])), 1e-12)}
