import os

HOST = "192.168.0.13"

class PATHS:
    UUID = os.path.join(os.path.dirname(__file__),'resources/uuid.ssg')
    IMG_PATH = os.path.join(os.path.dirname(__file__),'resources/image_capture.png')
    CSV_PATH = os.path.join(os.path.dirname(__file__),'resources/object_paths.csv')
    DETECTIONS = os.path.join(os.path.dirname(__file__),'resources/detections.csv')
    GATES = os.path.join(os.path.dirname(__file__),'resources/gates.ssg')


class SOCKET:
    HOST = HOST
    PORT = 5001


class MQTT:
    HOST = HOST
    PORT = 1883
    CLIENT_ID = 1#<UUID HERE>
    TOPIC = [("cycle/live", 2)]


