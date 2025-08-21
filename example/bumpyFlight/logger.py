# logger.py
import datetime
import inspect
import os
import sys


class Logger:
    """
    "Level - Timestamp : Message"
    """

    def __init__(self, name=None):
        self.name = name or "app"

    def _log(self, level_code, message):
        # Get caller's frame (1 level up from this function)
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename)
        line_number = frame.f_lineno

        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # Format according to company standard
        log_message = f"{level_code} - {timestamp} : {message}"

        # Print to stdout
        print(log_message)

    def debug(self, message):
        self._log("D", message)

    def info(self, message):
        self._log("Info", message)

    def warning(self, message):
        self._log("W", message)

    def error(self, message):
        self._log("E", message)

    def critical(self, message):
        self._log("C", message)


def get_logger(name=None):
    """Returns a logger with the given name."""
    return Logger(name)
