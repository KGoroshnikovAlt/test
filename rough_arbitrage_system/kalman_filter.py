"""
kalman_filter.py — online estimator of the dynamic hedge ratio.

State:   theta_t = [beta_t, alpha_t]^T,  theta_t = theta_{t-1} + w_t,  w ~ N(0, Q)
Obs.:    y_t = H_t theta_t + v_t,        v ~ N(0, R),  H_t = [x_t, 1]

The filter returns the cleaned spread  Z_t = y_t - beta_t x_t - alpha_t.
"""
from __future__ import annotations

import numpy as np

from config import Config


class KalmanSpread:
    """Online Kalman filter for the regression y = beta * x + alpha."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # state mean and covariance
        self.theta = np.array([cfg.KALMAN_BETA0, cfg.KALMAN_ALPHA0], dtype=float)
        self.P = np.eye(2) * cfg.KALMAN_P0
        # state noise covariance and observation noise variance
        self.Q = np.diag([cfg.KALMAN_Q_BETA, cfg.KALMAN_Q_ALPHA])
        self.R = float(cfg.KALMAN_R)

    def step(self, y: float, x: float) -> tuple[float, float, float]:
        """Update with one new observation pair and return (Z, beta, alpha).

        Z is the cleaned residual; beta and alpha are the posterior means.
        """
        # --- predict ---
        # theta is a random walk so the predicted mean is unchanged; covariance
        # gains the process-noise term.
        P_pred = self.P + self.Q

        # --- update ---
        H = np.array([x, 1.0])                         # 1 x 2 observation matrix
        y_hat = float(H @ self.theta)                  # predicted observation
        innovation = y - y_hat                         # residual
        S = float(H @ P_pred @ H.T) + self.R           # innovation variance
        K = (P_pred @ H) / S                           # Kalman gain (2,)

        self.theta = self.theta + K * innovation
        # Joseph form would be numerically safer; for 2x2 the simple form is fine.
        self.P = P_pred - np.outer(K, H) @ P_pred

        beta, alpha = float(self.theta[0]), float(self.theta[1])
        Z = y - beta * x - alpha
        return Z, beta, alpha


def run_kalman_series(y: np.ndarray, x: np.ndarray, cfg: Config) -> dict:
    """Run the filter over full arrays and return the resulting time series."""
    n = len(y)
    Z = np.empty(n)
    beta = np.empty(n)
    alpha = np.empty(n)
    kf = KalmanSpread(cfg)
    for t in range(n):
        Z[t], beta[t], alpha[t] = kf.step(float(y[t]), float(x[t]))
    return {"Z": Z, "beta": beta, "alpha": alpha}
