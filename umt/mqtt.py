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
import threading
import hashlib

# Initialize Logging
logging.basicConfig(level=logging.WARNING)  # Global logging configuration
logger = logging.getLogger("mqtt.MQTT_Client")  # Logger for this module
logger.setLevel(logging.INFO) # Debugging for this file.


    
# Global Variables
BROKER_HOST = "192.168.1.236"
BROKER_PORT = 1883
CLIENT_ID = "cycle_1"
TOPIC = [("cycle/live", 2)]
DATA_BLOCK_SIZE = 2000
process = None
client = None  # MQTT client instance. See init_mqtt()
logger = logging.getLogger("mqtt.MQTT_Client")
logger.info("Creating an instance of MQTT_Client")
'''
def set_barrier(data):

    livestream = None
    rfid = None
    
    if "rfid" in data:
        rfid = data["rfid"]
        
        if isinstance(rfid, str):
            rfid_signal(rfid)
            logger.info("Request for barrier approved")
        else:
            logger.info("Request for unknown barrier status of '{}'. We'll null the value.".format(rfid))      
    
    elif "livestream" in data:
        livestream = data["livestream"]
        
        if isinstance(livestream, str):
                Live_Stream.begin(livestream)
                logger.info("Request for live stream {}.".format(livestream))
        else:
            logger.info("Request for unknown live status '{}'. We'll null the value.".format(livestream))
            livestream = None # off.
    
    else:
        logger.info("Message '{}' did not contain property.".format(data))
    
'''
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



def on_message( client, user_data, msg):                                                         
    """Callback called when a message is received on a subscribed topic."""
    logger.debug("Received message for topic {}: {}".format( msg.topic, msg.payload))

    data = None

    try:
        data = json.loads(msg.payload.decode("UTF-8"))                                         
    except json.JSONDecodeError as e:
        logger.error("JSON Decode Error: " + msg.payload.decode("UTF-8"))
    
    topics = []
    for a_topic in TOPIC:
        topics.append(a_topic[0])

    if msg.topic in topics: 
        print(topics)                                                                    
        #set_barrier(data)                                                                    
        logger.info(data)

    else:
        logger.error("Unhandled message topic {} with payload ") #+ str(msg.topic, msg.payload)

def on_publish(client, user_data, connection_result_code):
    logger.info("Message Published")
    pass

def signal_handler( sig, frame):
    """Capture Control+C and disconnect from Broker."""

    logger.info("You pressed Control + C. Shutting down, please wait...")

    client.disconnect() # Graceful disconnection.
    sys.exit(0)



def init_mqtt():
    global client

    # Our MQTT Client. See PAHO documentation for all configurable options.
    # "clean_session=True" means we don"t want Broker to retain QoS 1 and 2 messages
    # for us when we"re offline. You"ll see the "{"session present": 0}" logged when
    # connected.
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
    client.connect(BROKER_HOST, BROKER_PORT)
    client.loop_start()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Capture Control + C
    logger.info("Listening for messages on topic '" + str(TOPIC) + "'. Press Control + C to exit.")

    init_mqtt()
    signal.pause()