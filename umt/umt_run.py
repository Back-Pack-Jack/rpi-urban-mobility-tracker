import multiprocessing
import subprocess
import threading
import logging
import signal
import os
from sys import platform
from umt_init import UMTinit
from umt_counter import main as countermain
from mqtt import init_mqtt as mqttmain
from umt_main.py import main as umtmain

logging.basicConfig(level=logging.WARNING)  # Global logging configuration
logger = logging.getLogger("umt - run")  # Logger for this module
logger.setLevel(logging.INFO) # Debugging for this file.


MAIN = os.path.join(os.path.dirname(__file__),'umt_main.py')
'''
COUNTER = os.path.join(os.path.dirname(__file__),'umt_counter.py')
MQTT = os.path.join(os.path.dirname(__file__),'mqtt.py')
'''

init = UMTinit()
init.initialize_device() # From umt_init.py the device initializes i.e. checks if a UUID exists, sends it's GPS location
logger.info('Initializing Device')
init.initialize_picture()
logger.info('Initializing Picture')
init.initialize_zones()
logger.info('Initializing Zones')
logger.info('Running all scripts')



t3 = threading.Thread(target=umtmain).start()

t1 = threading.Thread(target=countermain).start()

t2 = threading.Thread(target=mqttmain).start()
