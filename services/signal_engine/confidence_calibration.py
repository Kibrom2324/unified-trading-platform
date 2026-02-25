"""Confidence calibration using Platt scaling (logistic regression on raw scores).

Transforms raw model confidence scores into calibrated probabilities.
Prevents overconfident predictions from dominating the ensemble.

Fit on validation set predictions vs actual outcomes, then apply to live scores.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger("confidence_calibration")


class PlattScaler:
    """Platt scaling — fits a sigmoid to map raw scores → calibrated probabilities.

    P(correct | score) = 1 / (1 + exp(A*score + B))

    Fit with Newton's method on validation data (no sklearn dependency).
    """

    def __init__(self) -> None:
        self.A: float = 0.0
        self.B: float = 0.0
        self._fitted = False

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 100,
        min_step: float = 1e-10,
    ) -> None:
        """Fit Platt scaling parameters A, B using Newton's method.

        Args:
            scores: Raw model confidence scores (N,)
            labels: Binary labels 1=correct, 0=incorrect (N,)
        """
        n = len(scores)
        if n == 0:
            logger.warning("platt_fit_empty")
            return

        # Target probabilities with Bayesian correction
        n_pos = float(np.sum(labels > 0))
        n_neg = float(n - n_pos)
        t_pos = (n_pos + 1) / (n_pos + 2)
        t_neg = 1.0 / (n_neg + 2)
        targets = np.where(labels > 0, t_pos, t_neg)

        A = 0.0
        B = math.log((n_neg + 1) / (n_pos + 1))

        for iteration in range(max_iter):
            p = 1.0 / (1.0 + np.exp(A * scores + B))
            p = np.clip(p, 1e-15, 1 - 1e-15)

            d1_A = float(np.sum(scores * (p - targets)))
            d1_B = float(np.sum(p - targets))
            d2_AA = float(np.sum(scores * scores * p * (1 - p)))
            d2_BB = float(np.sum(p * (1 - p)))
            d2_AB = float(np.sum(scores * p * (1 - p)))

            det = d2_AA * d2_BB - d2_AB * d2_AB
            if abs(det) < 1e-15:
                break

            dA = -(d2_BB * d1_A - d2_AB * d1_B) / det
            dB = -(d2_AA * d1_B - d2_AB * d1_A) / det

            A += dA
            B += dB

            if abs(dA) < min_step and abs(dB) < min_step:
                break

        self.A = A
        self.B = B
        self._fitted = True
        logger.info("platt_scaler_fitted", A=round(A, 6), B=round(B, 6), samples=n, iterations=iteration + 1)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply calibration to raw scores.

        Returns calibrated probabilities in [0, 1].
        """
        if not self._fitted:
            return scores  # Pass-through if not fitted
        calibrated = 1.0 / (1.0 + np.exp(self.A * scores + self.B))
        return np.clip(calibrated, 0.0, 1.0)

    def transform_single(self, score: float) -> float:
        """Calibrate a single score."""
        if not self._fitted:
            return score
        return float(1.0 / (1.0 + math.exp(self.A * score + self.B)))

    def save(self, path: str | Path) -> None:
        """Save calibration parameters."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"A": self.A, "B": self.B, "fitted": self._fitted}, f)

    def load(self, path: str | Path) -> None:
        """Load calibration parameters."""
        with open(path) as f:
            data = json.load(f)
        self.A = data["A"]
        self.B = data["B"]
        self._fitted = data.get("fitted", True)
        logger.info("platt_scaler_loaded", A=self.A, B=self.B)
