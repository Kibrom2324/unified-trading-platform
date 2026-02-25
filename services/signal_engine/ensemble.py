"""Signal Ensemble — combines TFT predictions + Lean alpha signals into one scored signal.

Weighting scheme (all configurable via env):
  TFT direction  : weight 0.5
  RSI-14 signal  : weight 0.2
  EMA cross      : weight 0.2
  MACD signal    : weight 0.1

Output: ensemble_direction (long/short/neutral), ensemble_confidence (0–1),
        signal_score (weighted vote sum, -1 to +1)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

# Signal weights
TFT_WEIGHT = float(os.getenv("TFT_WEIGHT", "0.5"))
RSI_WEIGHT = float(os.getenv("RSI_WEIGHT", "0.2"))
EMA_WEIGHT = float(os.getenv("EMA_WEIGHT", "0.2"))
MACD_WEIGHT = float(os.getenv("MACD_WEIGHT", "0.1"))

# How long to hold an alpha signal as "active" (seconds)
# EMA/MACD signals are sparse; we carry them for N bars
SIGNAL_TTL_SECONDS = int(os.getenv("ALPHA_SIGNAL_TTL_SECONDS", "300"))  # 5 minutes


@dataclass
class AlphaSignal:
    indicator: str
    direction: str  # "UP" or "DOWN"
    weight: float
    ts: datetime
    indicator_value: Optional[float] = None


@dataclass
class TftSignal:
    direction: str      # "long" / "short" / "neutral"
    confidence: float   # 0–1
    q50: float
    ts: datetime


@dataclass
class EnsembleResult:
    ticker: str
    ts: datetime
    tft_direction: Optional[str]
    tft_confidence: float
    ensemble_direction: str     # "long" / "short" / "neutral"
    ensemble_confidence: float  # 0–1
    signal_score: float         # -1 to +1 (negative = short, positive = long)
    active_alphas: list[str]    # which alphas contributed
    regime: str = "unknown"     # bull / bear / sideways (Phase 3)


class SignalEnsemble:
    """Combines TFT + Lean alpha signals per symbol into a single scored signal.

    Call flow:
        1. `update_tft(symbol, tft_signal)` — when new TFT prediction arrives
        2. `update_alpha(symbol, alpha_signal)` — when RSI/EMA/MACD fires
        3. `compute(symbol)` — returns EnsembleResult or None
    """

    def __init__(self) -> None:
        # Latest TFT signal per symbol
        self._tft: dict[str, TftSignal] = {}
        # Active alpha signals per symbol (list, TTL-pruned)
        self._alphas: dict[str, list[AlphaSignal]] = {}

    def update_tft(self, symbol: str, signal: TftSignal) -> None:
        self._tft[symbol] = signal

    def update_alpha(self, symbol: str, signal: AlphaSignal) -> None:
        self._alphas.setdefault(symbol, []).append(signal)

    def _prune_alphas(self, symbol: str) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=SIGNAL_TTL_SECONDS)
        self._alphas[symbol] = [
            s for s in self._alphas.get(symbol, [])
            if s.ts >= cutoff
        ]

    def compute(self, symbol: str) -> Optional[EnsembleResult]:
        """Compute ensemble signal for symbol. Returns None if no TFT signal present."""
        tft = self._tft.get(symbol)
        if tft is None:
            return None

        self._prune_alphas(symbol)

        # TFT contribution
        if tft.direction == "long":
            tft_vote = TFT_WEIGHT
        elif tft.direction == "short":
            tft_vote = -TFT_WEIGHT
        else:
            tft_vote = 0.0

        # Scale TFT by confidence
        tft_vote *= tft.confidence

        # Alpha contributions (take most recent per indicator)
        alpha_votes: dict[str, float] = {}
        active_alphas: list[str] = []
        latest_by_indicator: dict[str, AlphaSignal] = {}

        for a in self._alphas.get(symbol, []):
            if a.indicator not in latest_by_indicator or a.ts > latest_by_indicator[a.indicator].ts:
                latest_by_indicator[a.indicator] = a

        for indicator, a in latest_by_indicator.items():
            vote = a.weight if a.direction == "UP" else -a.weight
            alpha_votes[indicator] = vote
            active_alphas.append(indicator)

        total_alpha_vote = sum(alpha_votes.values())
        signal_score = tft_vote + total_alpha_vote

        # Normalize to -1..+1 (max possible weight = TFT_WEIGHT + RSI+EMA+MACD weights)
        max_score = TFT_WEIGHT + RSI_WEIGHT + EMA_WEIGHT + MACD_WEIGHT
        if max_score > 0:
            norm_score = signal_score / max_score
            # Clamp to exactly [-1, 1] (guard against float precision edge cases)
            norm_score = max(-1.0, min(1.0, norm_score))
        else:
            norm_score = 0.0

        # Direction
        if norm_score > 0.15:
            direction = "long"
        elif norm_score < -0.15:
            direction = "short"
        else:
            direction = "neutral"

        # Ensemble confidence: absolute value of normalized score, boosted by TFT confidence
        raw_confidence = abs(norm_score) * 0.7 + tft.confidence * 0.3
        confidence = max(0.0, min(1.0, raw_confidence))

        return EnsembleResult(
            ticker=symbol,
            ts=datetime.now(timezone.utc),
            tft_direction=tft.direction,
            tft_confidence=tft.confidence,
            ensemble_direction=direction,
            ensemble_confidence=confidence,
            signal_score=norm_score,
            active_alphas=active_alphas,
        )
