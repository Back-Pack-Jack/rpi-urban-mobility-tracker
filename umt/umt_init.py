import uuid
import pickle
import mqtt
import logging

logging.basicConfig(level=logging.WARNING)  # Global logging configuration
logger = logging.getLogger("init - umt_init")  # Logger for this module
logger.setLevel(logging.INFO) # Debugging for this file.


# Device looks to find it's UUID no. if it doesn't exist it generates one, communicates it to the server and saves it to 'uuid.ssg'
def initialize_device():
    try:
        with open('uuid.ssg', 'rb') as f:
            UUID = pickle.load(f)
            logger.info("Loaded UUID")
    except FileNotFoundError:
        UUID = str(uuid.uuid4())
        UUIDb = bytearray(UUID, 'utf8')
        mqtt.init_UUID(UUID)
        with open("uuid.ssg", "wb") as f:
            pickle.dump(UUID, f)
