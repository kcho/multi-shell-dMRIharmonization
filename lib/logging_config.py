#!/usr/bin/env python

"""
Logging configuration module using rich for colored console output.
Falls back to standard logging if rich is not available.
"""

import logging
import sys

# Try to import rich, fall back to standard logging if not available
try:
    from rich.logging import RichHandler
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Create logger
logger = logging.getLogger('dMRIharmonization')

# Configure handler based on availability
if HAS_RICH:
    console = Console()
    handler = RichHandler(console=console, rich_tracebacks=True)
    handler.setFormatter(logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]",
    ))
else:
    # Fall back to standard logging with colored output via stdlib
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt='[%(levelname)s] %(message)s',
        datefmt='[%X]',
    ))

# Add handler to logger
if not logger.handlers:
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Convenience functions for different log levels
def log_info(*args, **kwargs):
    """Log info level message - accepts multiple arguments like print()"""
    sep = kwargs.get('sep', ' ')
    message = sep.join(str(arg) for arg in args)
    logger.info(message)

def log_debug(*args, **kwargs):
    """Log debug level message - accepts multiple arguments like print()"""
    sep = kwargs.get('sep', ' ')
    message = sep.join(str(arg) for arg in args)
    logger.debug(message)

def log_warning(*args, **kwargs):
    """Log warning level message - accepts multiple arguments like print()"""
    sep = kwargs.get('sep', ' ')
    message = sep.join(str(arg) for arg in args)
    logger.warning(message)

def log_error(*args, **kwargs):
    """Log error level message - accepts multiple arguments like print()"""
    sep = kwargs.get('sep', ' ')
    message = sep.join(str(arg) for arg in args)
    logger.error(message)

def log_critical(*args, **kwargs):
    """Log critical level message - accepts multiple arguments like print()"""
    sep = kwargs.get('sep', ' ')
    message = sep.join(str(arg) for arg in args)
    logger.critical(message)
