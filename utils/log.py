import os
import logging
import datetime

class Logger(object):
    # Mapping between string levels and logging module levels
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self, root_path, log_name, level='info', fmt='%{asctime}s - %(levelname)s - %(message)s)'):
        """
        Initialize the logger.
        :param root_path: Root directory where logs will be stored
        :param log_name: Name of the log file (without extension)
        :param level: Logging level (default: 'info')
        :param format: Log message format
        """
        # Root path for log storage
        self.root_path = root_path

        # Log file name
        self.log_name = log_name

        # Log format
        self.fmt = fmt

        # Create a logger instance
        self.logger = logging.getLogger(log_name)

        # Set logging level
        self.logger.setLevel(self.level_relations.get(level))

    def get_logger(self):
        """
        Create and return a configured logger instance
        """
        # Create log directory if it does not exist
        path = os.path.join(self.root_path, 'log')
        os.makedirs(path, exist_ok=True)

        # Define log file path
        file_name = os.path.join(path, self.log_name + '.log')

        # Create file handler (append mode, UTF-8 encoding)
        file_handler = logging.FileHandler(file_name, encoding = 'utf-8', mode = 'a')

        # Set formatter for the handler
        formatter = logging.Formatter(fmt=self.fmt)
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)

        return self.logger