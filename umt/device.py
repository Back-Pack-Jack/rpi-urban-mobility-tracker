import os.path
from os import path
import uuid



class Device:


    def __init__(self):
        self.DEVICE_SETTINGS = os.path.join(os.path.dirname(__file__),'Settings.SSG')
        self.UUID = None
        self.GATES = None
        self.SERVER_CERT = None
        self.CLIENT_CERT = None
        self.CLIENT_KEY = None
        self.HOST = "3.8.238.85"
        self.HOST_NAME = "172-31-22-149.eu-west-2.compute.internal"
        self.TOPIC = [("cycle/live", 2)]


    def create_uuid(self):
        self.UUID = str(uuid.uuid4())
        print(self.UUID)

    def create_gates(self):
        pass


    def load_socket(self):
        HOST = self.HOST
        PORT = 5001
        HOST_NAME = self.HOST_NAME
        SERVER_CERT = self.SERVER_CERT
        CLIENT_CERT = self.CLIENT_CERT
        CLIENT_KEY = self.CLIENT_KEY



    def load_MQTT(self):
        HOST = self.HOST
        PORT = 1884
        TOPIC = self.TOPIC
        CLIENT_ID = self.UUID



    def load(self):
        # Loads a file that contains all specific and general information that the device requires. 
        if path.exists(self.DEVICE_SETTINGS):
            # Existing device settings
            pass
        else:
            # Create new device
            self.create_uuid()
            self.create_gates()
            self.load_socket()
            self.load_MQTT()
            print('File does not exist.')

        

Device = Device()
Device.load()