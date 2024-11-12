"""
simple logging setup
"""

import logging
import coloredlogs


# Create a logger object.
logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "%(asctime)s - \033[95m%(filename)s\033[0m : \033[38;5;208m%(lineno)d\033[0m : \033[94m%(funcName)s\033[0m - %(levelname)s - %(message)s"
)

# By default the install() function installs a handler on the root logger,
# this means that log messages from your code and log messages from the
# libraries that you use will all show up on the terminal.
coloredlogs.install(level="DEBUG", fmt=LOG_FORMAT, datefmt='%H:%M:%S')
