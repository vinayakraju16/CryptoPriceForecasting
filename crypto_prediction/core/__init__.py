"""Dependency-free core: datasets, indicators, models, backtests and signals."""

from .dataset import DataError, Bar, Series, discover_datasets, load_series, validate_series
from .timeseries import MinMax, make_windows, sequential_split
from .metrics import mae, rmse, mape, smape, r2, directional_accuracy
from .indicators import sma, ema, rsi, macd, bollinger, volatility, max_drawdown, sharpe, value_at_risk
from .models import MODEL_REGISTRY, Forecaster, build_model, auto_select, feature_matrix
from .backtest import BacktestResult, walk_forward, strategy_returns
from .signals import Prediction, build_prediction, screen_universe, position_size, recommend
from .sentiment import score_text, aggregate_daily_sentiment, SentimentResult
from .data_source import fetch_and_merge, merge_series, PROVIDERS

__all__ = [
    "DataError", "Bar", "Series", "discover_datasets", "load_series", "validate_series",
    "MinMax", "make_windows", "sequential_split",
    "mae", "rmse", "mape", "smape", "r2", "directional_accuracy",
    "sma", "ema", "rsi", "macd", "bollinger", "volatility", "max_drawdown", "sharpe", "value_at_risk",
    "MODEL_REGISTRY", "Forecaster", "build_model", "auto_select", "feature_matrix",
    "BacktestResult", "walk_forward", "strategy_returns",
    "Prediction", "build_prediction", "screen_universe", "position_size", "recommend",
    "score_text", "aggregate_daily_sentiment", "SentimentResult",
    "fetch_and_merge", "merge_series", "PROVIDERS",
]
