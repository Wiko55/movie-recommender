import logging
import sys
import time
from functools import wraps

# Konfiguracja loggera (żeby działało i w Dockerze, i w konsoli)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # czas
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def measure_execution_time(func):
    """
    Dekorator mierzący czas wykonania funkcji.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        msg = (
            f"⏱️ Funkcja '{func.__name__}' wykonała się w: {execution_time:.4f} sekundy."
        )

        logger.info(msg)

        return result

    return wrapper
