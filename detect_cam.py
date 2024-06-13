import configparser
import os
import sys
from datetime import datetime
import time
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

import crud
import models
import utils
import logging
from database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

config = configparser.ConfigParser()
cfg = config.read('config.ini')

if not cfg:
    raise Exception('Config file(config.ini) not found')

module_id = config.get('system', 'module_id')
image_path = config.get('system', 'image_path')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(logging.FileHandler("detect_cam.log"))
logger.setLevel(logging.DEBUG)


def run(stream_url: str, width: int, height: int, num_threads: int) -> None:
  cap = cv2.VideoCapture(stream_url)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Initialize the object detection model
  base_options = core.BaseOptions(file_name="efficientdet_lite0.tflite", use_coral=False, num_threads=num_threads)
  detection_options = processor.DetectionOptions(max_results=1, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  success, image = cap.read()
  if not success:
    sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')
  cap.release()

  image = cv2.flip(image, 1)

  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  input_tensor = vision.TensorImage.create_from_array(rgb_image)
  logger.debug("start detecting")
  # estimate detecting time
  start_time = time.time_ns()
  detection_result = detector.detect(input_tensor)
  end_time = time.time_ns()
  elapsed_time_ms = (end_time - start_time) / 1e6
  logger.debug(f"detect completed in {elapsed_time_ms} milliseconds")
  results = utils.process_result(detection_result)

  file_suffix = ""

  detection, confidence = None, None
  if results:
    logger.info(results)
    detection = results[0][0]
    confidence = results[0][1]
    file_suffix = f"_{detection}_{confidence}".replace(" ", "_")
  else:
    logger.info("No detection")

  image_dir = image_path + datetime.now().strftime('%Y/%m/%d')
  if not os.path.exists(image_dir):
    os.makedirs(image_dir)

  file_name = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_suffix}.jpg"
  file_abs_path = f"{image_dir}/{file_name}"
  crud.insert_detection_result(db=SessionLocal(),
                               module_id=module_id,
                               image_path=file_abs_path,
                               detection=detection,
                               confidence=confidence)
  cv2.imwrite(file_abs_path, image)
  cv2.destroyAllWindows()


def main(stream_url: str):
  width = 320
  height = 240
  num_threads = 4
  run(stream_url, width, height, num_threads)


if __name__ == '__main__':
  main("http://192.168.219.116:81/stream")