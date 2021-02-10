import os
import pickle

HOST = "3.8.238.85"

class PATHS:
    UUID = os.path.join(os.path.dirname(__file__),'resources/uuid.ssg')
    IMG_PATH = os.path.join(os.path.dirname(__file__),'resources/image_capture.png')
    CSV_PATH = os.path.join(os.path.dirname(__file__),'resources/object_paths.csv')
    DETECTIONS = os.path.join(os.path.dirname(__file__),'resources/detections.csv')
    GATES = os.path.join(os.path.dirname(__file__),'resources/gates.ssg')


class DEVICE:
    try:
        with open(PATHS.UUID, 'rb') as f:
            UUID = pickle.load(f)
    except:
        UUID = None


class SOCKET:
    HOST = HOST
    PORT = 5001


class MQTT:
    HOST = HOST
    PORT = 1884
    TOPIC = [("cycle/live", 2)]
    CLIENT_ID = DEVICE.UUID

