import logging
import os
from logging.handlers import TimedRotatingFileHandler

logs_dir = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(logs_dir):
    os.mkdir(logs_dir)

logs_path = logs_dir

logger = logging.getLogger("logs_camera_record")
formatter = logging.Formatter('%(asctime)s | %(name)s |  %(levelname)s: %(message)s')
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

file_handler = TimedRotatingFileHandler(
    os.path.join(logs_path, 'processor.log'),
    when='midnight',
    interval=1,
    backupCount=1,
    encoding='utf-8'
)

file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)