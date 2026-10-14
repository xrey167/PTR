"""Replay backtest of the self-improvement cycle: is the outcome model predictive at all?

Pure Python, no torch. Nothing here trains or serves a model.
"""

from ptr_training.replay.backtest import (
    BacktestReport,
    ExtrapolationAbort,
    HistoryError,
    HistoryPool,
    NotPredictive,
    Outcome,
    ReplaySimulator,
    backtest,
    choose_winner,
)

__all__ = [
    "BacktestReport",
    "ExtrapolationAbort",
    "HistoryError",
    "HistoryPool",
    "NotPredictive",
    "Outcome",
    "ReplaySimulator",
    "backtest",
    "choose_winner",
]
