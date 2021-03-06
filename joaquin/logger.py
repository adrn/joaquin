# Standard library
import logging
import sys

# Third-party
from astropy.logger import StreamHandler

__all__ = ['logger']


class JoaquinHandler(StreamHandler):
    def emit(self, record):
        record.origin = 'joaquin'
        super().emit(record)


class JoaquinLogger(logging.getLoggerClass()):
    def _set_defaults(self):
        """Reset logger to its initial state"""

        # Remove all previous handlers
        for handler in self.handlers[:]:
            self.removeHandler(handler)

        # Set default level
        self.setLevel(logging.INFO)

        # Set up the stdout handler
        sh = JoaquinHandler(sys.stdout)
        self.addHandler(sh)


logging.setLoggerClass(JoaquinLogger)
logger = logging.getLogger('joaquin')
logger._set_defaults()
logger.setLevel(20)  # default
