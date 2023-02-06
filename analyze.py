import json
import logging
from pathlib import Path
import sys
from deepface import DeepFace

from config import DATA_FILE, FILE_MASK, IMAGES_PATH
from utils import read_data

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]


def rec_img(img):
    logger.info(f"Analyzing: {img}")
    try:
        embedding_objs = DeepFace.represent(img_path=img, model_name=models[0], detector_backend=detectors[4])
        return embedding_objs[0]["embedding"]
    except ValueError:
        logger.warn(f"No face detected: {img}")
        return None


def get_images(dir):
    files = Path(dir).glob(FILE_MASK)
    return [file for file in files]


def analyze_save(file, dir):
    data = read_data(file)
    images = get_images(dir)
    res = {}
    for i in images:
        if str(i) in data:
            res[str(i)] = data[str(i)]
        else:
            res[str(i)] = rec_img(str(i))
    with open(file, "w") as outfile:
        json.dump(res, outfile)


if __name__ == "__main__":
    analyze_save(DATA_FILE, IMAGES_PATH)
