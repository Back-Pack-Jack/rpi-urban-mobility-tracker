import multiprocessing
import subprocess
import logging
import signal
import os
from sys import platform
from umt_init import UMTinit

logging.basicConfig(level=logging.WARNING)  # Global logging configuration
logger = logging.getLogger("umt - run")  # Logger for this module
logger.setLevel(logging.INFO) # Debugging for this file.


MAIN = os.path.join(os.path.dirname(__file__),'umt_main.py')
COUNTER = os.path.join(os.path.dirname(__file__),'umt_counter.py')
MQTT = os.path.join(os.path.dirname(__file__),'mqtt.py')


init = UMTinit()
init.initialize_device() # From umt_init.py the device initializes i.e. checks if a UUID exists, sends it's GPS location
logger.info('Initializing Device')
init.initialize_picture()
logger.info('Initializing Picture')
init.initialize_zones()
logger.info('Initializing Zones')
logger.info('Running all scripts')

def signal_handler(sig, frame): # Capture Control+C and disconnect from Broker.
    logger.info("You pressed Control + C. Shutting down, please wait...")
    p1.send_signal(signal.SIGINT)
    p2.send_signal(signal.SIGINT)
    p3.send_signal(signal.SIGINT)

signal.signal(signal.SIGINT, signal_handler)  # Capture Control + C

p1 = subprocess.Popen(['python', MAIN])
p2 = subprocess.Popen(['python', COUNTER])
p3 = subprocess.Popen(['python', MQTT])
