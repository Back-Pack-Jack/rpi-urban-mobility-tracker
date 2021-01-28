import multiprocessing
import subprocess
import logging
import signal

#umt init
from umt_init import initialize_device
from umt_init import initialize_picture
from umt_init import initialize_zones

logging.basicConfig(level=logging.WARNING)  # Global logging configuration
logger = logging.getLogger("umt - run")  # Logger for this module
logger.setLevel(logging.INFO) # Debugging for this file.


initialize_device() # From umt_init.py the device initializes i.e. checks if a UUID exists, sends it's GPS location
logger.info('Initializing Device')
initialize_picture()
logger.info('Initializing Picture')
initialize_zones()
logger.info('Initializing Zones')

logger.info('Running all scripts')

def signal_handler(sig, frame): # Capture Control+C and disconnect from Broker.
    logger.info("You pressed Control + C. Shutting down, please wait...")
    p1.send_signal(signal.SIGINT)
    p2.send_signal(signal.SIGINT)
    p3.send_signal(signal.SIGINT)

signal.signal(signal.SIGINT, signal_handler)  # Capture Control + C

p1 = subprocess.Popen(['python', 'umt_main.py'])
p2 = subprocess.Popen(['python', 'umt_counter.py'])
p3 = subprocess.Popen(['python', 'mqtt.py'])



