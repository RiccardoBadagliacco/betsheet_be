import logging
import sys


def setup_logging(level: int = logging.INFO):
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.setLevel(level)
    root.addHandler(handler)
