import multiprocessing

#umt main
import umt_main

#mqtt
import mqtt

#umt counter
import umt_counter

#umt init
from umt.umt_init import initialize_device

initialize_device() # From umt_init.py the device initializes i.e. checks if a UUID exists, sends it's GPS location

def main():
    for umt in ('umt_main', 'mqtt', 'umt_counter'):
        p = multiprocessing.Process(target=lambda: __import__(umt))
        p.start

if __name__ == '__main__':
    main()