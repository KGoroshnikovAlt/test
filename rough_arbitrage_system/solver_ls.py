"""
solver_ls.py — non-Markovian Longstaff–Schwartz on path signatures.

We approximate the continuation value of an open position by

    C_n(ω) ≈ ⟨ α_n*, S^L(Z)_{0,t_n}(ω) ⟩,

where α_n* are regression weights fit offline by backward induction. At every
exercise date n the algorithm regresses the realised payoff if held to the
optimal future stopping time onto the truncated signature observed at n.

Online: stop iff  payoff_n  >=  C_n  (and never below LS_PAYOFF_THRESHOLD).

The implementation deliberately avoids machine-learning frameworks; ridge
regression with a small regulariser is sufficient and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from config import Config


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge regression with column of ones already inside X."""
    XtX = X.T @ X
    XtX += lam * np.eye(XtX.shape[0])
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


@dataclass
class LongstaffSchwartz:
    """Offline fit + online inference of continuation-value weights."""
    cfg: Config
    weights: List[np.ndarray] | None = None     # one weight vector per exercise step

    # --- offline training ---------------------------------------------------
    def fit(
        self,
        signatures: np.ndarray,  # (M paths, N steps, F features)
        payoffs: np.ndarray,     # (M paths, N steps)  immediate exercise value
    ) -> "LongstaffSchwartz":
        """Backward induction.

        At step n we observe the signature S_n and the payoff P_n. The realised
        cash-flow of the optimal policy from n onwards is computed by walking
        backwards: at the terminal step the payoff is mechanically taken; at
        earlier steps we compare immediate payoff against the regression's
        estimate of future-optimal payoff.
        """
        M, N, F = signatures.shape
        assert payoffs.shape == (M, N), "payoffs and signatures must align"

        cashflow = payoffs[:, -1].astype(float).copy()
        weights: List[np.ndarray] = [np.zeros(F + 1)] * N
        weights[-1] = np.zeros(F + 1)             # terminal step: forced stop

        for n in range(N - 2, -1, -1):
            X = np.hstack([np.ones((M, 1)), signatures[:, n, :]])
            y = cashflow
            beta = _ridge_fit(X, y, self.cfg.LS_REG_LAMBDA)
            weights[n] = beta
            continuation = X @ beta
            exercise = payoffs[:, n]
            stop_now = exercise >= continuation
            cashflow = np.where(stop_now, exercise, cashflow)

        self.weights = weights
        return self

    # --- online inference ---------------------------------------------------
    def continuation_value(self, step: int, signature: np.ndarray) -> float:
        """Return ⟨ α_n*, S^L(Z)_{0,t} ⟩ at exercise date `step`."""
        if self.weights is None:
            return 0.0
        step = max(0, min(step, len(self.weights) - 1))
        beta = self.weights[step]
        x = np.concatenate([[1.0], signature.ravel()])
        if x.shape[0] != beta.shape[0]:
            # signature length mismatch — fall back to threshold-only policy
            return 0.0
        return float(beta @ x)

    def should_stop(self, step: int, signature: np.ndarray, payoff: float) -> bool:
        cv = self.continuation_value(step, signature)
        return payoff >= max(cv, self.cfg.LS_PAYOFF_THRESHOLD)


# --------------------------------------------------------------------------- #
# Helper to manufacture training material from a single Z series              #
# --------------------------------------------------------------------------- #
def build_training_episodes(
    z: np.ndarray,
    payoff_fn: Callable[[np.ndarray, int], float],
    signature_fn: Callable[[np.ndarray], np.ndarray],
    cfg: Config,
    episode_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the historical Z series into overlapping episodes for LS training.

    Each episode of length `episode_len` is sampled every `stride` minutes; at
    every step the signature of the trailing WINDOW_SIZE points is computed and
    the payoff function evaluates the mark-to-market PnL of an open position.
    """
    z = np.asarray(z, dtype=float)
    W = cfg.WINDOW_SIZE
    episodes_sig: list[np.ndarray] = []
    episodes_pay: list[np.ndarray] = []
    for start in range(0, len(z) - W - episode_len + 1, stride):
        sigs = np.empty((episode_len, _signature_length(cfg)))
        pays = np.empty(episode_len)
        for n in range(episode_len):
            window = z[start + n: start + n + W]
            sigs[n] = signature_fn(window)
            pays[n] = payoff_fn(z, start + n + W - 1)
        episodes_sig.append(sigs)
        episodes_pay.append(pays)
    if not episodes_sig:
        F = _signature_length(cfg)
        return np.zeros((0, episode_len, F)), np.zeros((0, episode_len))
    return np.stack(episodes_sig), np.stack(episodes_pay)


def _signature_length(cfg: Config) -> int:
    from signature_engine import signature_dim
    return signature_dim(d=2, L=cfg.SIGNATURE_ORDER, include_constant=False)
