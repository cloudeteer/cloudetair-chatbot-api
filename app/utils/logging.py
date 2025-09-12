"""
----------------------------------------------------------------
# Logging configuration and utilities.
----------------------------------------------------------------
# 
# This module provides functions to set up logging for the 
# application, allowing for different logging levels and formats.
# It also includes a utility function to retrieve a logger 
# instance by name.
#
# Usage:
# from app.utils.logging import setup_logging, get_logger
# setup_logging(level="DEBUG")
# logger = get_logger(__name__)
# logger.info("This is an info message")
#
# Example log format:
# 2023-10-01 12:00:00,000 - app.utils.logging - INFO - This is 
# an info message
#
----------------------------------------------------------------
"""


import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("app").setLevel(getattr(logging, level.upper()))


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name for the logger (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
