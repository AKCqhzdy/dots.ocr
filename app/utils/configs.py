from pathlib import Path

# Number of concurrent jobs that can run.
NUM_WORKERS = 16

# The max number of concurrent OCR inference requests that can be sent to the ocr model.
# Increasing this may improve GPU utilization to some extent but at the cost of the model
# server memory usage.
CONCURRENT_OCR_INFERENCE_TASK_LIMIT = 16

# The max number of concurrent picture description requests that can be sent to the internVL
# model. Increasing this may improve GPU utilization to some extent but at the cost of the
# model server memory usage.
CONCURRENT_DESCRIBE_PICTURE_TASK_LIMIT = 16

# The max number of concurrent OCR tasks that can be run. Increasing this may improve overall
# resource overlapping, but at the cost of memory for buffering the extracted images from docs,
# i.e, around CONCURRENT_OCR_TASK_LIMIT images (one for each page) can be buffered in memory.
CONCURRENT_OCR_TASK_LIMIT = 64

# DPI of images extracted from documents that are used for OCR.
DPI = 200

# The max number of jobs that can be queued. 0 means unlimited.
JOB_QUEUE_MAX_SIZE = 0

# The number of OCR inference tasks that can be queued. Increase this may improve resource
# overlapping, but at the cost of memory for buffering the extracted images from docs,
OCR_INFERENCE_TASK_QUEUE_MAX_SIZE = 2 * CONCURRENT_OCR_INFERENCE_TASK_LIMIT

# The number of OCR inference tasks that can be queued. Increase this may improve resource
# overlapping, but at the cost of memory for buffering the picture blocks identified from
# the documents.
DESCRIBE_PICTURE_TASK_QUEUE_MAX_SIZE = 24

OCR_INFERENCE_HOST = "localhost"
OCR_INFERENCE_PORT = 8000
OCR_HEALTH_CHECK_URL = f"http://{OCR_INFERENCE_HOST}:{OCR_INFERENCE_PORT}/health"

INTERN_VL_HOST = "internvl3-5"
INTERN_VL_PORT = 6008

# TODO(tatiana): need to check the timeout semantics in OpenAI API.
# Exclude queuing time from the timeout.
API_TIMEOUT = 60

TASK_RETRY_COUNT = 3

# If the number of failed tasks is greater than this threshold, the job will be considered failed.
TASK_FAIL_THRESHOLD = 0.1

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
