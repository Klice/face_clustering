import json
import sys

import logging


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def read_data(file):
    with open(file, 'r') as openfile:
        return json.load(openfile)
