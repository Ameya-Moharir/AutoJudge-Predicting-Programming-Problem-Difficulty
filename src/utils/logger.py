"""
Logging utilities for AutoJudge
"""
import logging
import os
from pathlib import Path
from datetime import datetime

class Logger:
    """Logger setup and management"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name, log_file=None, level=logging.INFO):
        """
        Get or create a logger instance
        
        Args:
            name: Logger name
            log_file: Path to log file (optional)
            level: Logging level
        
        Returns:
            Logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler (if log_file provided)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def setup_default_logger(cls):
        """Setup default logger for the application"""
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"autojudge_{timestamp}.log"
        
        return cls.get_logger("autojudge", str(log_file))


# Default logger instance
logger = Logger.setup_default_logger()
