"""
config_manager.py

Configuration management system for UltraTrader.
Loads and validates YAML configurations with environment variable overrides.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    max_steps: int = 500
    observation_dim: int = 200
    action_dim: int = 2
    exposure_max: float = 0.6
    initial_equity: float = 1.0
    commission: float = 0.0005
    slippage_base: float = 0.0005
    slippage_impact: float = 1.0
    reward_weights: Dict[str, float] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """SAC training configuration."""
    learning_rate: float = 3e-4
    buffer_size: int = 200000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.995
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto_0.2"
    total_timesteps: int = 100000
    eval_freq: int = 2000
    checkpoint_freq: int = 5000
    n_eval_episodes: int = 5
    device: str = "auto"
    l2_reg: float = 0.0001
    gradient_clip: float = 10.0
    policy_network: Dict[str, Any] = field(default_factory=dict)
    value_network: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskManagementConfig:
    """Risk management configuration."""
    var: Dict[str, Any] = field(default_factory=dict)
    position_sizing: Dict[str, Any] = field(default_factory=dict)
    stop_loss: Dict[str, Any] = field(default_factory=dict)
    take_profit: Dict[str, Any] = field(default_factory=dict)
    dynamic_adjustment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    console: bool = True
    file: bool = True
    tensorboard: bool = True
    mlflow: bool = True
    log_dir: str = "logs"
    log_file: str = "ultratrader.log"
    max_size_mb: int = 100
    backup_count: int = 5
    track_every_n_steps: int = 100


class ConfigManager:
    """
    Centralized configuration management for UltraTrader.

    Loads configuration from YAML files and environment variables.
    Provides type-safe access to configuration values.

    Usage:
        config = ConfigManager.load("config/default_config.yaml")
        lr = config.training.learning_rate
        max_dd = config.environment.risk_limits["max_drawdown"]
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self.raw_config = config_dict
        self._parse_config()

    def _parse_config(self):
        """Parse raw config dictionary into typed dataclasses."""
        # Environment
        env_cfg = self.raw_config.get("environment", {})
        self.environment = EnvironmentConfig(
            max_steps=env_cfg.get("max_steps", 500),
            observation_dim=env_cfg.get("observation_dim", 200),
            action_dim=env_cfg.get("action_dim", 2),
            exposure_max=env_cfg.get("exposure_max", 0.6),
            initial_equity=env_cfg.get("initial_equity", 1.0),
            commission=env_cfg.get("commission", 0.0005),
            slippage_base=env_cfg.get("slippage_base", 0.0005),
            slippage_impact=env_cfg.get("slippage_impact", 1.0),
            reward_weights=env_cfg.get("reward_weights", {}),
            risk_limits=env_cfg.get("risk_limits", {}),
        )

        # Training
        train_cfg = self.raw_config.get("training", {})
        self.training = TrainingConfig(
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            buffer_size=train_cfg.get("buffer_size", 200000),
            batch_size=train_cfg.get("batch_size", 256),
            tau=train_cfg.get("tau", 0.005),
            gamma=train_cfg.get("gamma", 0.995),
            train_freq=train_cfg.get("train_freq", 1),
            gradient_steps=train_cfg.get("gradient_steps", 1),
            ent_coef=train_cfg.get("ent_coef", "auto_0.2"),
            total_timesteps=train_cfg.get("total_timesteps", 100000),
            eval_freq=train_cfg.get("eval_freq", 2000),
            checkpoint_freq=train_cfg.get("checkpoint_freq", 5000),
            n_eval_episodes=train_cfg.get("n_eval_episodes", 5),
            device=train_cfg.get("device", "auto"),
            l2_reg=train_cfg.get("l2_reg", 0.0001),
            gradient_clip=train_cfg.get("gradient_clip", 10.0),
            policy_network=train_cfg.get("policy_network", {}),
            value_network=train_cfg.get("value_network", {}),
        )

        # Risk Management
        risk_cfg = self.raw_config.get("risk_management", {})
        self.risk_management = RiskManagementConfig(
            var=risk_cfg.get("var", {}),
            position_sizing=risk_cfg.get("position_sizing", {}),
            stop_loss=risk_cfg.get("stop_loss", {}),
            take_profit=risk_cfg.get("take_profit", {}),
            dynamic_adjustment=risk_cfg.get("dynamic_adjustment", {}),
        )

        # Logging
        log_cfg = self.raw_config.get("logging", {})
        self.logging = LoggingConfig(
            level=log_cfg.get("level", "INFO"),
            format=log_cfg.get("format", "json"),
            console=log_cfg.get("console", True),
            file=log_cfg.get("file", True),
            tensorboard=log_cfg.get("tensorboard", True),
            mlflow=log_cfg.get("mlflow", True),
            log_dir=log_cfg.get("log_dir", "logs"),
            log_file=log_cfg.get("log_file", "ultratrader.log"),
            max_size_mb=log_cfg.get("max_size_mb", 100),
            backup_count=log_cfg.get("backup_count", 5),
            track_every_n_steps=log_cfg.get("track_every_n_steps", 100),
        )

        # Store other sections as-is
        self.market_simulator = self.raw_config.get("market_simulator", {})
        self.expert_ensemble = self.raw_config.get("expert_ensemble", {})
        self.evaluation = self.raw_config.get("evaluation", {})
        self.backtesting = self.raw_config.get("backtesting", {})
        self.mlflow = self.raw_config.get("mlflow", {})
        self.data = self.raw_config.get("data", {})
        self.versioning = self.raw_config.get("versioning", {})
        self.deployment = self.raw_config.get("deployment", {})

    @classmethod
    def load(cls, config_path: str | Path) -> ConfigManager:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ConfigManager instance with loaded configuration
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Apply environment variable overrides
        config_dict = cls._apply_env_overrides(config_dict)

        return cls(config_dict)

    @staticmethod
    def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.

        Environment variables should be prefixed with ULTRA_
        Example: ULTRA_TRAINING_LEARNING_RATE=0.001
        """
        # Training overrides
        if "ULTRA_TRAINING_LEARNING_RATE" in os.environ:
            config.setdefault("training", {})["learning_rate"] = float(
                os.environ["ULTRA_TRAINING_LEARNING_RATE"]
            )

        if "ULTRA_TRAINING_TIMESTEPS" in os.environ:
            config.setdefault("training", {})["total_timesteps"] = int(
                os.environ["ULTRA_TRAINING_TIMESTEPS"]
            )

        if "ULTRA_TRAINING_DEVICE" in os.environ:
            config.setdefault("training", {})["device"] = os.environ[
                "ULTRA_TRAINING_DEVICE"
            ]

        # Environment overrides
        if "ULTRA_ENV_MAX_STEPS" in os.environ:
            config.setdefault("environment", {})["max_steps"] = int(
                os.environ["ULTRA_ENV_MAX_STEPS"]
            )

        # Logging overrides
        if "ULTRA_LOG_LEVEL" in os.environ:
            config.setdefault("logging", {})["level"] = os.environ["ULTRA_LOG_LEVEL"]

        return config

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot-notation path.

        Args:
            key_path: Dot-separated path to config value (e.g., "training.learning_rate")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self.raw_config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Return raw configuration dictionary."""
        return self.raw_config.copy()

    def __repr__(self) -> str:
        return f"ConfigManager(sections={list(self.raw_config.keys())})"


# Global config instance (lazy loaded)
_global_config: ConfigManager | None = None


def get_config() -> ConfigManager:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        # Load default config
        default_path = Path(__file__).parent / "config" / "default_config.yaml"
        _global_config = ConfigManager.load(default_path)
    return _global_config


def set_config(config: ConfigManager):
    """Set global configuration instance."""
    global _global_config
    _global_config = config
