import multiprocessing
import subprocess
import threading
import logging
import signal
import os
from sys import platform
from umt_init import UMTinit
#from umt_counter import main as countermain
from mqtt import main as mqttmain
'''
logging.basicConfig(filename='app.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d, %H:%M:%S',
                            level=logging.INFO)  # Global logging configuration

logger = logging.getLogger("Run (umt_run.py) - ")  # Logger for this module
'''

MAIN = os.path.join(os.path.dirname(__file__),'umt_main.py')

init = UMTinit()
init.initialize_device() # From umt_init.py the device initializes i.e. checks if a UUID exists
#logger.info('Initializing Device')
init.initialize_picture()
#logger.info('Initializing Picture')
init.initialize_zones()
#logger.info('Initializing Zones')
#logger.info('Running all scripts')



p1 = subprocess.Popen(['python', MAIN])

#t1 = threading.Thread(target=countermain).start()

t2 = threading.Thread(target=mqttmain).start()
 