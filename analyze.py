import json
import logging
from pathlib import Path
import sys
from deepface import DeepFace

from config import DATA_FILE

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def rec_img(img):
    logger.info(f"Analyzing: {img}")
    try:
        embedding_objs = DeepFace.represent(img_path=img)
        return embedding_objs[0]["embedding"]
    except ValueError:
        logger.warn(f"No face detected: {img}")
        return None


def get_images(dir):
    files = Path(dir).glob('*')
    return [file for file in files]


def analyze_save(file, dir):
    images = get_images(dir)
    res = {str(i): rec_img(str(i)) for i in images}
    with open(file, "w") as outfile:
        json.dump(res, outfile)


if __name__ == "__main__":
    analyze_save(DATA_FILE, "images")
