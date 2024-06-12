import sys

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


def run(stream_url: str, width: int, height: int, num_threads: int) -> None:
  # Start capturing video input from the camera
  cap = cv2.VideoCapture(stream_url)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Initialize the object detection model
  base_options = core.BaseOptions(file_name="efficientdet_lite0.tflite", use_coral=False, num_threads=num_threads)
  detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  success, image = cap.read()
  if not success:
    sys.exit(
        'ERROR: Unable to read from webcam. Please verify your webcam settings.'
    )
  cap.release()

  image = cv2.flip(image, 1)

  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  input_tensor = vision.TensorImage.create_from_array(rgb_image)
  print("start detecting")
  detection_result = detector.detect(input_tensor)
  print("end detecting")
  results = utils.process_result(detection_result)
  print(results)
  cv2.imwrite("image.jpg", image)
  cv2.destroyAllWindows()


def main(stream_url: str):
  width = 320
  height = 240
  num_threads = 4
  run(stream_url, width, height, num_threads)


if __name__ == '__main__':
  main("http://192.168.219.116:81/stream")