"""
Logging configuration for the arthropod classification pipeline.

Provides centralized logging setup with console and file handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.config import config


def setup_logging(
    name: Optional[str] = None,
    log_file: Optional[Path] = None,
    level: Optional[str] = None,
    console: Optional[bool] = None
) -> logging.Logger:
    """
    Setup logging with both console and file handlers.

    Args:
        name: Logger name. If None, uses root logger.
        log_file: Path to log file. If None, uses config or creates default.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). If None, uses config.
        console: Whether to log to console. If None, uses config.

    Returns:
        Configured logger instance

    Example:
        >>> from src.utils.logging_config import setup_logging
        >>> logger = setup_logging('my_module')
        >>> logger.info('Processing started')
    """
    # Get logger
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Get configuration
    if level is None:
        level = config.get('logging.level', 'INFO')
    if console is None:
        console = config.get('logging.console', True)
    if log_file is None:
        log_file_str = config.get('logging.file', './logs/pipeline.log')
        log_file = Path(log_file_str)

    # Create log directory
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Format
    log_format = config.get(
        'logging.format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    formatter = logging.Formatter(log_format)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized: level={level}, file={log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug('Debug message')
    """
    logger = logging.getLogger(name)

    # Setup if not already configured
    if not logger.handlers:
        setup_logging(name)

    return logger


class ProgressLogger:
    """
    Progress logger for long-running operations.

    Logs progress at regular intervals to avoid log spam.

    Example:
        >>> progress = ProgressLogger(total=1000, name='processing')
        >>> for i in range(1000):
        ...     progress.update()
        ...     # do work
        >>> progress.finish()
    """

    def __init__(
        self,
        total: int,
        name: str = 'operation',
        log_interval: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize progress logger.

        Args:
            total: Total number of items to process
            name: Operation name for logging
            log_interval: Log every N items
            logger: Logger instance. If None, creates one.
        """
        self.total = total
        self.name = name
        self.log_interval = log_interval
        self.logger = logger or get_logger(__name__)

        self.current = 0
        self.start_time = datetime.now()

        self.logger.info(f"Starting {name}: {total} items")

    def update(self, n: int = 1):
        """
        Update progress.

        Args:
            n: Number of items processed (default: 1)
        """
        self.current += n

        # Log at intervals
        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            percent = (self.current / self.total * 100) if self.total > 0 else 0

            self.logger.info(
                f"{self.name}: {self.current}/{self.total} ({percent:.1f}%) - "
                f"{rate:.1f} items/s"
            )

    def finish(self):
        """Log completion."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"{self.name} complete: {self.current} items in {elapsed:.1f}s "
            f"({self.current / elapsed:.1f} items/s)"
        )


class LogContext:
    """
    Context manager for logging sections.

    Example:
        >>> with LogContext('Processing images'):
        ...     # do work
        ...     pass
        # Logs: "Processing images - Started"
        # Logs: "Processing images - Completed in X.Xs"
    """

    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        """
        Initialize log context.

        Args:
            operation: Operation name
            logger: Logger instance. If None, creates one.
        """
        self.operation = operation
        self.logger = logger or get_logger(__name__)
        self.start_time = None

    def __enter__(self):
        """Enter context."""
        self.start_time = datetime.now()
        self.logger.info(f"{self.operation} - Started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info(f"{self.operation} - Completed in {elapsed:.1f}s")
        else:
            self.logger.error(
                f"{self.operation} - Failed after {elapsed:.1f}s: {exc_val}"
            )

        # Don't suppress exceptions
        return False


# Create logs directory on import
logs_dir = Path(config.get('logging.file', './logs/pipeline.log')).parent
logs_dir.mkdir(parents=True, exist_ok=True)
