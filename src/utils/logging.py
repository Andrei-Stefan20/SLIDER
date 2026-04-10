"""Centralised logger factory for the SLIDERS project."""

import logging

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a single StreamHandler.

    Calling this function multiple times with the same ``name`` always
    returns the same :class:`logging.Logger` instance and never adds
    duplicate handlers.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level.  Defaults to ``logging.INFO``.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger
