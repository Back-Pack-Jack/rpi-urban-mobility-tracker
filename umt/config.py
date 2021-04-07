import os
import pickle

HOST = "192.168.1.236"
HOST_NAME = "172-31-22-149.eu-west-2.compute.internal"

class PATHS:
    UUID = os.path.join(os.path.dirname(__file__),'resources/uuid.ssg')
    IMG_PATH = os.path.join(os.path.dirname(__file__),'resources/image_capture.png')
    CSV_PATH = os.path.join(os.path.dirname(__file__),'resources/object_paths.csv')
    DETECTIONS = os.path.join(os.path.dirname(__file__),'resources/detections.csv')
    GATES = os.path.join(os.path.dirname(__file__),'resources/gates.ssg')
    
    SERVER_CERT = os.path.join(os.path.dirname(__file__),'resources/server.crt')
    CLIENT_CERT = os.path.join(os.path.dirname(__file__),'resources/client.crt')
    CLIENT_KEY = os.path.join(os.path.dirname(__file__),'resources/client.key')


class DEVICE:
    try:
        with open(PATHS.UUID, 'rb') as f:
            UUID = pickle.load(f)
    except:
        UUID = None


class SOCKET:
    HOST = HOST
    PORT = 5001
    HOST_NAME = HOST_NAME
    SERVER_CERT = PATHS.SERVER_CERT
    CLIENT_CERT = PATHS.CLIENT_CERT
    CLIENT_KEY = PATHS.CLIENT_KEY
    


class MQTT:
    HOST = HOST
    PORT = 1883
    TOPIC = [("cycle/uuid", 2), ("cycle/img", 2), ("cycle/gps", 2), ("cycle/bme680", 2), ("cycle/sensiron", 2), ("cycle/count", 2), ("cycle/newdevice", 2)]
    CLIENT_ID = DEVICE.UUID

