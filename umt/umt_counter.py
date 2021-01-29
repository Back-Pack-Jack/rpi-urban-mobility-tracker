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
from sys import platform
import os

logging.basicConfig(level=logging.WARNING)  # Global logging configuration
logger = logging.getLogger("UMT - Counter")  # Logger for this module
logger.setLevel(logging.INFO) # Debugging for this file.

# --- Sets platform directories --------------------------------
if platform == 'linux' or platform == 'linux2':
    UUID = 'umt/uuid.ssg'
    IMG_PATH = 'umt/image_capture.png'
    CSV_PATH = 'umt/object_paths.csv' 
    DETECTIONS = 'umt/detections.ssg'
    GATES = 'umt/gates.ssg'
if platform == 'darwin':
    UUID = 'rpi-urban-mobility-tracker/umt/uuid.ssg'
    IMG_PATH = 'rpi-urban-mobility-tracker/umt/image_capture.png'
    CSV_PATH = 'rpi-urban-mobility-tracker/umt/object_paths.csv' 
    DETECTIONS = 'rpi-urban-mobility-tracker/umt/detections.ssg'
    GATES = 'rpi-urban-mobility-tracker/umt/gates.ssg'


with open(UUID, 'rb') as f:
            UUID = pickle.load(f)
            logger.info("Loaded UUID")

# --- Initialises the MQTT client to send a message to the server
DEVICE = UUID

gates = []
detections = []

# load object paths
df = pd.read_csv(CSV_PATH, header=None, names=['frame', 'time', 'class', 'id', 'age', 'obj_t_since_last_update', 'obj_hits', 'bb_left', 'bb_top', 'bb_width', 'bb_height'])
df.shape

#  compute detection centroids
df['cx'] = df['bb_left'] + (0.5 * df['bb_width'])
df['cy'] = df['bb_top']  + (0.5 * df['bb_height'])

try:
        with open(GATES, 'rb') as f: 
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


#--- Looks for Outstanding detection files, if a file exists it's opened and
#--- any new detections are appended to the end and the file saved.
#--- The client attempts to transfer the file to the server returning a true
#--- or false dependant upon the success of a server connection.
def sendFile():
    try:
        with open(DETECTIONS, 'rb') as f:
            previous_detections = pickle.load(f)
            detections.insert(len(detections), previous_detections)
            logger.info("Outstanding detections found. Inserted Outstanding Detections into File")
            with open(DETECTIONS, 'wb') as f:
                pickle.dump(detections, f)
            sent = client_sock.sendFile(DETECTIONS, DEVICE)
            return sent
    except FileNotFoundError:
        logger.info('No Outstanding Detections Found. Dumping detections to file.')
        with open(DETECTIONS, 'wb') as f:
            pickle.dump(detections, f)
        sent = client_sock.sendFile(DETECTIONS, DEVICE)
        return sent
    
                        
# --- Pickle the detection list to a byte file --------
def count():
    
    #--- Runs the algorithm to determine whether anybody has crossed the gates
    crossed_gates()
    sent = sendFile()
    
    #--- If the file has been sent the existing detection file is deleted and if
    #--- not the file is retained to be appended to next time the counter runs.
    if not sent:
        logger.info('Unable to send File, will retry after detections are calculated')
        return
    else:
        os.remove(DETECTIONS)
        logger.info('File Sent to Server')

def main():
    schedule.every(15).minutes.do(count)

    while 1:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    main()


