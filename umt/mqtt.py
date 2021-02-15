import os
import logging
import signal
import sys
import json
from time import sleep
import os
import subprocess
import time
import paho.mqtt.client as mqtt 
import paho.mqtt.publish as publish
import threading
import hashlib
from config import MQTT as MQT
from config import PATHS

# Initialize Logging
logging.basicConfig(filename='app.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d, %H:%M:%S',
                            level=logging.INFO)  # Global logging configuration

logger = logging.getLogger("MQTT (mqtt.py) - ")  # Logger for this module
    
# Global Variables
BROKER_HOST = MQT.HOST
BROKER_PORT = MQT.PORT
CLIENT_ID = MQT.CLIENT_ID
TOPIC = MQT.TOPIC
DATA_BLOCK_SIZE = 2000

process = None
client = None  # MQTT client instance. See init_mqtt()
logger = logging.getLogger("mqtt.MQTT_Client")


"""
MQTT Related Functions and Callbacks
"""
def on_connect( client, user_data, flags, connection_result_code):                              
    """on_connect is called when our program connects to the MQTT Broker.
    Always subscribe to topics in an on_connect() callback.
    This way if a connection is lost, the automatic
    re-connection will also results in the re-subscription occurring."""

    if connection_result_code == 0:                                                            
        # 0 = successful connection
        logger.info("Connected to MQTT Broker")
    else:
        # connack_string() gives us a user friendly string for a connection code.
        logger.error("Failed to connect to MQTT Broker: " + mqtt.connack_string(connection_result_code))

    # Subscribe to the topic for LED level changes.
    client.subscribe(TOPIC)                                                             



def on_disconnect( client, user_data, disconnection_result_code):                               
    """Called disconnects from MQTT Broker."""
    logger.error("Disconnected from MQTT Broker")



def on_message( client, user_data, msg): # Callback called when a message is received on a subscribed topic.                                                    
    logger.debug("Received message for topic {}: {}".format( msg.topic, msg.payload))
    msg_dec = msg.payload.decode("utf-8") # Writes the decoded msg to an object


def init_UUID(device):  # When the device initialises it sends a MQTT message with it's UUID back the server
    logger.info("Sending UUID to server")
    CLIENT_ID = device
    publish.single("cycle/init", device, hostname=BROKER_HOST, port=BROKER_PORT)
    
    

def on_publish(client, user_data, connection_result_code):
    logger.info("Message Published")
    pass


def signal_handler( sig, frame): # Capture Control+C and disconnect from Broker.
    logger.info("You pressed Control + C. Shutting down, please wait...")
    client.disconnect() # Graceful disconnection.
    sys.exit(0)



def init_mqtt():

    logger.info("Creating an instance of MQTT_Client")
    
    global client

    logger.info("Initialising Client")
    client = mqtt.Client(
        client_id=CLIENT_ID,
        clean_session=False)

    # Route Paho logging to Python logging.
    client.enable_logger()

    # Setup callbacks
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.on_publish = on_publish
    
    # Connect to Broker.
    while True:
        try:
            client.connect(BROKER_HOST, BROKER_PORT)
            break
        except:
            logger.exception("Couldn't connect to broker. Retrying...")
            time.sleep(60)
            
    client.loop_start()

def main():
   # signal.signal(signal.SIGINT, signal_handler)  # Capture Control + C
    logger.info("Listening for messages on topic '" + str(TOPIC) + "'. Press Control + C to exit.")

    init_mqtt()
    #signal.pause()

if __name__ == "__main__":
    main()