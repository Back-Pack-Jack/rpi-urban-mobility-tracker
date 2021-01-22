"""
File: chapter04/mqtt_led.py

A full life-cycle Python + MQTT program to control an LED.

Dependencies:
  pip3 install paho-mqtt gpiozero pigpio

Built and tested with Python 3.7 on Raspberry Pi 4 Model B
"""
import logging
import signal
import sys
import json
from time import sleep
import os
import subprocess
import time
import paho.mqtt.client as mqtt                                                                # (1)


# Initialize Logging
logging.basicConfig(level=logging.WARNING)  # Global logging configuration
logger = logging.getLogger("main")  # Logger for this module
logger.setLevel(logging.INFO) # Debugging for this file

# Global Variables
BROKER_HOST = "192.168.1.236"                                                                       # (2)
BROKER_PORT = 1883
CLIENT_ID = "Cycle"                                                                         # (3)
TOPIC = "live"                                                                                   # (4)
process = None
client = None  # MQTT client instance. See init_mqtt()                                          # (5)
led = None     # PWMLED Instance. See init_led()

def spawn_stream(live):
    global process
    try:
        if live == 6:
            process = subprocess.Popen(['ffmpeg', '-f', 'v4l2', '-framerate', '25', '-video_size', '640x480', '-i', '/dev/video0', '-f', 'mpegts', '-codec:v', 'mpeg1video', '-s', '640x480', '-b:v', '1000k', '-bf', '0', 'http://192.168.1.236:8081/supersecret'], shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            time.sleep(1)
        else:
            kill = process.communicate(input=b'q')[0]
            time.sleep(1)
    except:
        print('Stream already dead')

"""
GPIO Related Functions
"""
def init_live():
    """Create and initialise an LED Object"""
    global live
    live = None
   # led.off()

def set_live_status(data):                                                                       # (6)

    live = None # 0 or 1
    
    if "live" in data:
        live = data["live"]
        
        if isinstance(live, int):
            spawn_stream(live)
        else:
            logger.info("Request for unknown Live status of '{}'. We'll null the value.".format(live))
            live = None # 0% = Led off.
    else:
        logger.info("Message '{}' did not contain property 'live'.".format(data))
        

"""
MQTT Related Functions and Callbacks
"""
def on_connect(client, user_data, flags, connection_result_code):                              # (7)
    """on_connect is called when our program connects to the MQTT Broker.
       Always subscribe to topics in an on_connect() callback.
       This way if a connection is lost, the automatic
       re-connection will also results in the re-subscription occurring."""

    if connection_result_code == 0:                                                            # (8)
        # 0 = successful connection
        logger.info("Connected to MQTT Broker")
    else:
        # connack_string() gives us a user friendly string for a connection code.
        logger.error("Failed to connect to MQTT Broker: " + mqtt.connack_string(connection_result_code))

    # Subscribe to the topic for LED level changes.
    client.subscribe(TOPIC, qos=2)                                                             # (9)



def on_disconnect(client, user_data, disconnection_result_code):                               # (10)
    """Called disconnects from MQTT Broker."""
    logger.error("Disconnected from MQTT Broker")



def on_message(client, userdata, msg):                                                         # (11)
    """Callback called when a message is received on a subscribed topic."""
    logger.debug("Received message for topic {}: {}".format( msg.topic, msg.payload))

    data = None

    try:
        data = json.loads(msg.payload.decode("UTF-8"))                                         # (12)
    except json.JSONDecodeError as e:
        logger.error("JSON Decode Error: " + msg.payload.decode("UTF-8"))

    if msg.topic == TOPIC:                                                                     # (13)
        set_live_status(data)                                                                    # (14)
        logger.info(data)

    else:
        logger.error("Unhandled message topic {} with payload " + str(msg.topic, msg.payload))



def signal_handler(sig, frame):
    """Capture Control+C and disconnect from Broker."""
    #global led_state

    logger.info("You pressed Control + C. Shutting down, please wait...")

    client.disconnect() # Graceful disconnection.
    live = None
    sys.exit(0)



def init_mqtt():
    global client

    # Our MQTT Client. See PAHO documentation for all configurable options.
    # "clean_session=True" means we don"t want Broker to retain QoS 1 and 2 messages
    # for us when we"re offline. You"ll see the "{"session present": 0}" logged when
    # connected.
    client = mqtt.Client(                                                                      # (15)
        client_id=CLIENT_ID,
        clean_session=False)

    # Route Paho logging to Python logging.
    client.enable_logger()                                                                     # (16)

    # Setup callbacks
    client.on_connect = on_connect                                                             # (17)
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # Connect to Broker.
    client.connect(BROKER_HOST, BROKER_PORT)                                                   # (18)



# Initialise Module
init_live()
init_mqtt()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Capture Control + C                        # (19)
    logger.info("Listening for messages on topic '" + TOPIC + "'. Press Control + C to exit.")

    client.loop_start()                                                                        # (20)
    signal.pause()
