"""
logger.py

Production-grade structured logging system for UltraTrader.

Features:
- JSON and text logging formats
- Multiple output destinations (console, file, tensorboard)
- Rotating file handlers
- Performance tracking
- Error tracking with context
- Integration with MLflow
"""

from __future__ import annotations

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add custom fields
        if hasattr(record, "custom_fields"):
            log_data.update(record.custom_fields)

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """
    Custom text formatter with colors for console output.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.RESET)

        # Format timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Format message
        message = record.getMessage()

        # Build log line
        log_line = f"{color}[{timestamp}] {levelname:8s}{self.RESET} {message}"

        # Add location info for DEBUG level
        if levelname == "DEBUG":
            location = f"{record.module}:{record.funcName}:{record.lineno}"
            log_line += f" ({location})"

        # Add exception if present
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            log_line += f"\n{exc_text}"

        return log_line


class UltraLogger:
    """
    Production-grade logger for UltraTrader.

    Features:
    - Multiple output destinations
    - Structured JSON logging
    - Rotating file handlers
    - Context-aware logging
    - Performance metrics tracking
    """

    def __init__(
        self,
        name: str = "UltraTrader",
        log_dir: str | Path = "logs",
        log_file: str = "ultratrader.log",
        level: str = "INFO",
        format_type: str = "json",
        max_size_mb: int = 100,
        backup_count: int = 5,
        console_output: bool = True,
        file_output: bool = True,
    ):
        """
        Initialize logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            log_file: Log file name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_type: Format type ("json" or "text")
            max_size_mb: Max size of each log file in MB
            backup_count: Number of backup files to keep
            console_output: Enable console logging
            file_output: Enable file logging
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.format_type = format_type

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []  # Clear existing handlers

        # Add console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))

            if format_type == "json":
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(TextFormatter())

            self.logger.addHandler(console_handler)

        # Add file handler
        if file_output:
            log_path = self.log_dir / self.log_file
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=backup_count,
            )
            file_handler.setLevel(getattr(logging, level.upper()))

            if format_type == "json":
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )

            self.logger.addHandler(file_handler)

        # Performance tracking
        self.metrics = {}
        self.start_times = {}

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)

    def _log(self, level: int, message: str, exc_info: bool = False, **kwargs):
        """
        Internal logging method with custom fields.

        Args:
            level: Logging level
            message: Log message
            exc_info: Include exception info
            **kwargs: Additional fields to include in log
        """
        extra = {"custom_fields": kwargs} if kwargs else {}
        self.logger.log(level, message, exc_info=exc_info, extra=extra)

    def log_metric(self, name: str, value: float, step: int | None = None):
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        self.metrics[name] = value
        msg = f"Metric: {name} = {value:.6f}"
        if step is not None:
            msg += f" (step {step})"
        self.info(msg, metric_name=name, metric_value=value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None):
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def start_timer(self, name: str):
        """
        Start a timer for performance tracking.

        Args:
            name: Timer name
        """
        self.start_times[name] = datetime.utcnow()
        self.debug(f"Timer started: {name}")

    def stop_timer(self, name: str) -> float:
        """
        Stop a timer and log the elapsed time.

        Args:
            name: Timer name

        Returns:
            Elapsed time in seconds
        """
        if name not in self.start_times:
            self.warning(f"Timer '{name}' was not started")
            return 0.0

        start = self.start_times.pop(name)
        elapsed = (datetime.utcnow() - start).total_seconds()

        self.info(
            f"Timer stopped: {name}",
            timer_name=name,
            elapsed_seconds=elapsed,
        )

        return elapsed

    def log_training_step(
        self,
        step: int,
        loss: float,
        metrics: Dict[str, float] | None = None,
    ):
        """
        Log training step information.

        Args:
            step: Training step number
            loss: Loss value
            metrics: Optional additional metrics
        """
        log_data = {
            "step": step,
            "loss": loss,
        }

        if metrics:
            log_data.update(metrics)

        self.info(f"Training step {step}: loss={loss:.6f}", **log_data)

    def log_evaluation(
        self,
        episode: int,
        reward: float,
        equity: float,
        metrics: Dict[str, float] | None = None,
    ):
        """
        Log evaluation episode information.

        Args:
            episode: Episode number
            reward: Episode reward
            equity: Final equity
            metrics: Optional additional metrics
        """
        log_data = {
            "episode": episode,
            "reward": reward,
            "equity": equity,
        }

        if metrics:
            log_data.update(metrics)

        self.info(
            f"Evaluation {episode}: reward={reward:.3f}, equity={equity:.4f}",
            **log_data,
        )

    def log_model_save(self, path: str, metrics: Dict[str, float] | None = None):
        """
        Log model save event.

        Args:
            path: Model save path
            metrics: Optional metrics at save time
        """
        log_data = {"model_path": path}
        if metrics:
            log_data.update(metrics)

        self.info(f"Model saved: {path}", **log_data)

    def log_risk_alert(self, alert_type: str, message: str, severity: str = "WARNING"):
        """
        Log risk management alert.

        Args:
            alert_type: Type of alert (e.g., "drawdown", "var_limit")
            message: Alert message
            severity: Alert severity
        """
        log_level = getattr(logging, severity.upper(), logging.WARNING)

        self._log(
            log_level,
            f"RISK ALERT [{alert_type}]: {message}",
            alert_type=alert_type,
            severity=severity,
        )

    def log_trade(
        self,
        direction: str,
        size: float,
        price: float,
        pnl: float | None = None,
    ):
        """
        Log trade execution.

        Args:
            direction: Trade direction ("BUY" or "SELL")
            size: Position size
            price: Execution price
            pnl: Realized PnL if closing position
        """
        log_data = {
            "trade_direction": direction,
            "trade_size": size,
            "trade_price": price,
        }

        if pnl is not None:
            log_data["pnl"] = pnl

        msg = f"TRADE: {direction} {size:.4f} @ {price:.4f}"
        if pnl is not None:
            msg += f" | PnL: {pnl:.4f}"

        self.info(msg, **log_data)

    def get_metrics(self) -> Dict[str, float]:
        """Get all logged metrics."""
        return self.metrics.copy()

    def clear_metrics(self):
        """Clear all logged metrics."""
        self.metrics.clear()


# Global logger instance
_global_logger: UltraLogger | None = None


def get_logger() -> UltraLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = UltraLogger()
    return _global_logger


def setup_logger(
    name: str = "UltraTrader",
    log_dir: str | Path = "logs",
    level: str = "INFO",
    format_type: str = "json",
    **kwargs,
) -> UltraLogger:
    """
    Setup and return configured logger.

    Args:
        name: Logger name
        log_dir: Log directory
        level: Logging level
        format_type: Format type ("json" or "text")
        **kwargs: Additional arguments for UltraLogger

    Returns:
        Configured UltraLogger instance
    """
    global _global_logger
    _global_logger = UltraLogger(
        name=name,
        log_dir=log_dir,
        level=level,
        format_type=format_type,
        **kwargs,
    )
    return _global_logger
