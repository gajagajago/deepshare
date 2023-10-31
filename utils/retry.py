import functools
import logging
import random
import time

_logger = logging.getLogger('retry_test')
_logger.setLevel(logging.INFO)


def retry(total=5, sleep=1, retry_exception=()):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for cnt in range(total):
                _logger.info(f"trying {func.__name__}() [{cnt+1}/{total}]")

                try:
                    result = func(*args, **kwargs)
                    _logger.info(f"in retry(), {func.__name__}() returned '{result}'")
                    return result
                except retry_exception as e:
                    _logger.info(f"in retry(), {func.__name__}() raised retry_exception '{e}'")
                    pass
                except Exception as e:
                    _logger.info(f"in retry(), {func.__name__}() raised {e}")
                    raise e

                time.sleep(sleep)
            _logger.info(f"{func.__name__} finally has been failed")
        return wrapper
    return decorator