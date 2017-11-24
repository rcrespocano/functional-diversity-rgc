# -*- coding: utf-8 -*-
"""Helper utilities for logging."""

import logging
import os


def create_logger(path, file_name='log.log'):
    if not os.path.exists(path):
        os.makedirs(path)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(message)s',
                        handlers=[logging.FileHandler(path + file_name),
                                  logging.StreamHandler()])


def get_logger(name):
    return logging.getLogger(name)
