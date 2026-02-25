from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AppHttpConfig(BaseModel):
    host: str
    data_ingestion_port: int
    feature_engineering_port: int
    prediction_port: int
    risk_port: int
    execution_port: int
    watchtower_port: int
    orchestrator_port: int
    night_trainer_port: int
    morning_deployer_port: int
    trade_watchdog_port: int
    backtester_port: int
    health_report_port: int


class AppConfig(BaseModel):
    environment: str
    correlation_header: str
    http: AppHttpConfig
    execution_approval_file: str
    execution_default_qty: int = 1      # Default quantity for paper trades
    execution_dry_run: bool = True      # Safety: when True, ALL trades blocked
    # Trading logic parameters
    max_trade_notional_usd: float = 1000.0  # Max USD per trade
    max_qty_per_trade: int = 100            # Max shares per trade
    min_qty_per_trade: int = 1              # Min shares per trade
    check_position_before_sell: bool = True  # Check position before SELL
    use_bracket_orders: bool = False
    take_profit_pct: float = 0.03
    stop_loss_pct: float = 0.015
    allow_add_to_position: bool = False
    max_orders_per_minute: int = 5
    max_orders_per_symbol_per_day: int = 3
    max_open_orders_total: int = 20
    cooldown_seconds_per_symbol: int = 900
    max_total_exposure_usd: float = 50000.0
    max_exposure_per_symbol_usd: float = 10000.0
    max_holding_minutes: int = 60
    exit_check_interval_seconds: int = 60
    symbols: list[str]


class TraderDecisionAgentConfig(BaseModel):
    port: int = 8011
    min_confidence: float = 0.65
    buy_return_threshold: float = 0.005
    sell_return_threshold: float = 0.005


class KafkaGroupIds(BaseModel):
    feature_engineering: str
    prediction: str
    risk: str
    execution: str
    watchtower: str
    trader_decision: str


class KafkaTopics(BaseModel):
    predictions_tft: str = "predictions.tft"
    trade_decisions: str = "trade.decisions"


class KafkaConfig(BaseModel):
    bootstrap_servers: list[str]
    schema_registry_url: str
    auto_offset_reset: str
    group_ids: KafkaGroupIds
    topics: KafkaTopics


class LimitsConfig(BaseModel):
    """Risk limits configuration - matches config/limits.yaml schema."""
    # Position limits (can use either pct or absolute)
    max_position_pct: float = 0.02
    max_position: int = 1000  # Legacy/absolute fallback
    max_notional: int = 100000  # Max notional value per position
    max_daily_loss_pct: float = 0.05
    max_loss_per_trade: int = 1000  # Legacy/absolute fallback
    max_volatility: float = 0.10
    min_confidence: float = 0.3
    max_correlation_spike: float = 0.95
    kill_switch_loss_pct: float = 0.10


class PolygonConfig(BaseModel):
    base_url: str = "https://api.polygon.io"


class AlpacaConfig(BaseModel):
    base_url: str = "https://data.alpaca.markets"


class YFinanceConfig(BaseModel):
    """YFinance config (FREE - no API key required)."""
    rate_limit_per_minute: int = 30


class MarketDataConfig(BaseModel):
    provider: str = "yfinance"  # polygon, alpaca, or yfinance (free)
    timeframe: str = "1D"
    backfill_minutes: int = 60
    backfill_days: int = 30
    polling_interval_seconds: int = 60
    rate_limit_per_minute: int = 5
    request_timeout_seconds: float = 30.0
    polygon: PolygonConfig = Field(default_factory=PolygonConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    yfinance: YFinanceConfig = Field(default_factory=YFinanceConfig)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    json_format: bool = Field(default=True, alias="json")


class DataConfig(BaseModel):
    """Configuration for historical data paths."""
    data_root: str = "/home/kironix/train"
    polygon_minute_dir: str = "poligon/Minute Aggregates"
    polygon_day_dir: str = "poligon/Day Aggregates"
    reddit_submissions_dir: str = "reddit/submissions"
    reddit_comments_dir: str = "reddit/comments"
    start_date: str | None = None
    end_date: str | None = None


class TrainingConfig(BaseModel):
    """Configuration for TFT model training."""
    lookback_window: int = 120
    horizons: list[int] = Field(default_factory=lambda: [30, 45, 60])
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    hidden_size: int = 64
    num_attention_heads: int = 4
    dropout: float = 0.1
    model_output_dir: str = "models"
    onnx_model_path: str = "models/tft_model.onnx"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "awet_tft"


class LLMConfig(BaseModel):
    provider: str = "local"
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama-3.1-70b-instruct-q4_k_m"
    timeout_seconds: int = 60
    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.9
    concurrency_limit: int = 4
    # LLM Tracing settings
    trace_enabled: bool = True
    trace_preview_chars: int = 800
    trace_log_level: str = "INFO"
    trace_store_db: bool = True


class Settings(BaseModel):
    app: AppConfig
    trader_decision_agent: TraderDecisionAgentConfig = Field(default_factory=TraderDecisionAgentConfig)
    kafka: KafkaConfig
    limits: LimitsConfig
    logging: LoggingConfig
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_set(data: dict[str, Any], keys: list[str], value: Any) -> None:
    target = data
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    for key, value in os.environ.items():
        if "__" not in key:
            continue
        keys = [k.lower() for k in key.split("__")]
        _deep_set(data, keys, value)

    if "KAFKA_BOOTSTRAP_SERVERS" in os.environ:
        servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"].split(",")
        _deep_set(data, ["kafka", "bootstrap_servers"], servers)
    if "SCHEMA_REGISTRY_URL" in os.environ:
        _deep_set(data, ["kafka", "schema_registry_url"], os.environ["SCHEMA_REGISTRY_URL"])
    if "LOG_LEVEL" in os.environ:
        _deep_set(data, ["logging", "level"], os.environ["LOG_LEVEL"])
    if "LLM_BASE_URL" in os.environ:
        _deep_set(data, ["llm", "base_url"], os.environ["LLM_BASE_URL"])
    if "LLM_MODEL" in os.environ:
        _deep_set(data, ["llm", "model"], os.environ["LLM_MODEL"])
    if "LLM_TIMEOUT_SECONDS" in os.environ:
        _deep_set(data, ["llm", "timeout_seconds"], os.environ["LLM_TIMEOUT_SECONDS"])
    if "LLM_MAX_TOKENS" in os.environ:
        _deep_set(data, ["llm", "max_tokens"], os.environ["LLM_MAX_TOKENS"])

    return data


def load_settings(config_dir: str = "config") -> Settings:
    # Load .env from repo root (handles relative paths)
    from pathlib import Path
    
    # Try multiple paths for .env
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent / ".env",
    ]
    
    env_loaded = False
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            env_loaded = True
            break
    
    if not env_loaded:
        import sys
        print(
            "⚠️  WARNING: No .env file found. "
            "Alpaca credentials may not be available.",
            file=sys.stderr,
        )
    
    # Check for critical Alpaca vars and warn if missing
    _check_alpaca_env()
    
    base = {}
    base.update(_load_yaml(Path(config_dir) / "app.yaml"))
    base.update(_load_yaml(Path(config_dir) / "kafka.yaml"))
    base.update(_load_yaml(Path(config_dir) / "limits.yaml"))
    base.update(_load_yaml(Path(config_dir) / "logging.yaml"))
    base.update(_load_yaml(Path(config_dir) / "market_data.yaml"))
    base.update(_load_yaml(Path(config_dir) / "llm.yaml"))
    merged = _apply_env_overrides(base)
    return Settings.model_validate(merged)


def _check_alpaca_env() -> None:
    """Check for Alpaca environment variables and warn if missing."""
    import sys
    
    missing = []
    if not os.getenv("ALPACA_API_KEY"):
        missing.append("ALPACA_API_KEY")
    if not os.getenv("ALPACA_SECRET_KEY"):
        missing.append("ALPACA_SECRET_KEY")
    
    if missing:
        print(
            f"⚠️  WARNING: Missing Alpaca credentials: {', '.join(missing)}. "
            "Paper trading will use simulated orders.",
            file=sys.stderr,
        )
    
    # Warn if using live endpoint
    base_url = os.getenv("ALPACA_BASE_URL", "")
    if base_url and "paper-api" not in base_url and "api.alpaca.markets" in base_url:
        print(
            f"🚨 DANGER: ALPACA_BASE_URL appears to be a LIVE endpoint: {base_url}. "
            "AlpacaClient will refuse to start.",
            file=sys.stderr,
        )
