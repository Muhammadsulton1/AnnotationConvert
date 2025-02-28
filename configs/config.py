import json


class Config:
    with open("/zmq_server/configs/main_conf.json", "r") as config_file:
        CONFIG = json.load(config_file)

    REDIS_HOST = CONFIG["redis"]["host"]
    REDIS_PORT = CONFIG["redis"]["port"]
    REDIS_USERNAME = CONFIG["redis"]["username"]
    REDIS_PASSWORD = CONFIG["redis"]["password"]
    ZMQ_PORT = CONFIG["zmq"]["port"]
    VIDEO_SOURCE = CONFIG["video_source"]
    FRAME_DELAY = CONFIG["frame_delay"]
