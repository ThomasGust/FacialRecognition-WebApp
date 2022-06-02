import logging
import sys
import traceback
from typing import List


class LoggerFactory:
    def __init__(self, logger_name: str):
        self.logger = self.create_logger(logger_name=logger_name)

    def create_formatter(self, format_pattern: str):
        format_pattern = (
            format_pattern
            or "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        return logging.Formatter(format_pattern)

    def get_console_handler(self, formatter, level=logging.INFO, stream=sys.stdout):
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        return console_handler

    def get_file_handler(
        self, formatter, level=logging.INFO, file_path: str = "data/app.log"
    ):
        file_handler = logging.FileHandler(filename=file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        return file_handler

    def create_logger(
        self,
        logger_name,
        level=logging.DEBUG,
        format_pattern: str = None,
        file_path: str = "data/app.log",
    ):
        logger = logging.getLogger(logger_name)
        formatter = self.create_formatter(format_pattern=format_pattern)
        console_handler = self.get_console_handler(
            formatter=formatter, level=logging.INFO
        )
        file_handler = self.get_file_handler(
            formatter=formatter, level=logging.INFO, file_path=file_path
        )
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(level)
        logger.propagate = False
        return logger

    def get_logger(self):
        return self.logger

    def create_custom_logger(
        self, logger_name: str, handlers: List, propagate_error: bool = False
    ):
        logger = logging.getLogger(logger_name)
        for handler in handlers:
            logger.addHandler(handler)
        logger.propagate = propagate_error
        return logger

    def uncaught_exception_hook(self, type, value, tb):
        tb_message = traceback.extract_tb(tb).format()
        tb_message = "\n".join(tb_message)
        err_message = "Uncaught Exception raised! \n{}: {}\nMessage: {}".format(
            type, value, tb_message
        )
        self.logger.critical(err_message)