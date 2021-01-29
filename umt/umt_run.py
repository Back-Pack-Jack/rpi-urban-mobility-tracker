import multiprocessing
import subprocess
import logging
import signal
from sys import platform
from umt_init import UMTinit

logging.basicConfig(level=logging.WARNING)  # Global logging configuration
logger = logging.getLogger("umt - run")  # Logger for this module
logger.setLevel(logging.INFO) # Debugging for this file.

if platform == 'linux' or platform == 'linux2':
    MAIN = 'umt/umt_main.py'
    COUNTER = 'umt/umt_counter.py'
    MQTT = 'umt/mqtt.py'
if platform == 'darwin':
    MAIN = 'umt_main.py'
    COUNTER = 'umt_counter.py'
    MQTT = 'mqtt.py'

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
'''
p1 = subprocess.Popen(['python', MAIN])
p2 = subprocess.Popen(['python', COUNTER])
p3 = subprocess.Popen(['python', MQTT])
'''