import logging
from pathlib import Path
from datetime import datetime


def setup_logger(model_name: str, output_dir: str = "logs") -> logging.Logger:
    """
    Sets up a logger that writes to both console and a file.

    Args:
        model_name: Name of the model being evaluated (used in log file name)
        output_dir: Directory to store log files

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a formatted timestamp
    timestamp = datetime.now().strftime("%Y%m%d")

    # Create log file name
    log_file = log_dir / f"{model_name.replace('/', '_')}_{timestamp}.log"

    # Create logger
    logger = logging.getLogger(f"evaluation_{model_name}")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
