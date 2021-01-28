import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from tkinter import Tk, Canvas, mainloop, PhotoImage, Label
from PIL import ImageTk, Image 
import mqtt
import pickle
from mqtt import init_mqtt
import bz2
import sys
import client_sock
import pickle
import schedule
import time
import logging

logging.basicConfig(level=logging.WARNING)  # Global logging configuration
logger = logging.getLogger("umt - umt_counter")  # Logger for this module
logger.setLevel(logging.INFO) # Debugging for this file.

with open('uuid.ssg', 'rb') as f:
            UUID = pickle.load(f)
            logger.info("Loaded UUID")

# --- Initialises the MQTT client to send a message to the server

IMG_PATH = 'image_capture.png'
CSV_PATH = 'object_paths.csv' 
DEVICE = UUID

gates = []
detections = []

# load object paths
df = pd.read_csv(CSV_PATH, header=None, names=['frame', 'time', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'class','score'])
df.shape

#df.sample(5)

#  compute detection centroids
df['cx'] = df['bb_left'] + (0.5 * df['bb_width'])
df['cy'] = df['bb_top']  + (0.5 * df['bb_height'])

try:
        with open('gates.ssg', 'rb') as f: 
            gates = pickle.load(f)
except:
    logger.info('Gate load error')


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def cross(s1, s2):
    a, b = s1
    c, d = s2
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d) 


# now lets cycle throught each objects trajectory and determine if it has crossed either of the gates
def crossed_gates():
    for n, obj_path in df.groupby(by='id'):
        
        # cycle through each time step of trajectory in ascending order
        for i, row in obj_path.sort_values(by='time', ascending=True).iterrows():
        
            # get position at current time
            xy_t0 = tuple(row[['cx', 'cy']].values)
            
            # get position at most recent historic time step
            xy_t1 = obj_path[ obj_path['frame'] < row['frame']].sort_values(by='frame', ascending=False)
            
            # if a previous time step is found, let's check if it crosses any of the gates
            if xy_t1.shape[0]>0:
                timecat = list(tuple(xy_t1[['time', 'class']].values[0]))
                xy_t1 = tuple(xy_t1[['cx', 'cy']].values[0])
                
                # cycle through gates
                for g, gate in enumerate(gates):
                    if cross(gates[g], [xy_t0, xy_t1]):
                        timecat.insert(0, g)
                        timecat.insert(0, DEVICE)
                        detections.insert(0, timecat)
                        
# --- Pickle the detection list to a byte file --------
def count():
    crossed_gates()
    transfer_file = pickle.dumps(detections)
    filename = "detections.ssg"
    client_sock.sendFile(filename, DEVICE)
    logger.info("Transfer of detection information to server complete!")

def main():
    schedule.every(15).minutes.do(count)

    while 1:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    main()


