import os.path
from os import path
import uuid
import pickle
import OpenSSL.crypto 

# Device Settings
# -- This module is responsible for loading and setting the configuration of each device,
# -- based on a combination of both static and abstract variables. Device is loaded from
# -- settings.ssg file. If no file exists one is created and the variables stored.

class Device:


    def __init__(self):
        self.DEVICE_SETTINGS = os.path.join(os.path.dirname(__file__),'settings.ssg')
        self.UUID = None
        self.GATES = None
        self.CERTS = {'SERVER_CERT' : 'resources/server.crt', 
                    'CLIENT_CERT' : 'resources/client.crt', 
                    'CLIENT_KEY' : 'resources/client.key' 
                    }
        self.HOST = "3.8.238.85"
        self.HOST_NAME = "172-31-22-149.eu-west-2.compute.internal"
        self.TOPIC = [("cycle/live", 2)]


    def create_uuid(self):
        self.UUID = str(uuid.uuid4())
        print('UUID : ', self.UUID)


    def load_socket(self):
        HOST = self.HOST
        PORT = 5001
        HOST_NAME = self.HOST_NAME
        SERVER_CERT = self.CERTS['SERVER_CERT']
        CLIENT_CERT = self.CERTS['CLIENT_CERT']
        CLIENT_KEY = self.CERTS['CLIENT_KEY']


    def load_MQTT(self):
        HOST = self.HOST
        PORT = 1884
        TOPIC = self.TOPIC
        CLIENT_ID = self.UUID


    def load_certs(self):
        pass


    def load(self):
        # Loads a file that contains all specific and general information that the device requires. 
        if path.exists(self.DEVICE_SETTINGS):
            # Existing device settings
            with open (self.DEVICE_SETTINGS, 'rb') as f:
                dev_pickle = pickle.load(f)
            print(f"Loaded exisiting configuration : {dev_pickle}")
        else:
            # Create new device
            print('Creating new device...')
            self.create_uuid()
            self.load_socket()
            self.load_MQTT()
            self.save()
            return True
            

    def save(self):
        settings = {
            "UUID" : self.UUID,
            "GATES" : self.GATES,
            "CERTS" : self.CERTS,
            "TOPIC" : self.TOPIC 
        }
        dev_pickle = pickle.dumps(settings)
        with open (self.DEVICE_SETTINGS, 'wb') as f:
            f.write(dev_pickle)
        print("Device saved.")
