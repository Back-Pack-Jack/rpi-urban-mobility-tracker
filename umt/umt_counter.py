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
import os.path
from os import path
from config import PATHS, DEVICE

logname = os.path.join(os.path.dirname(__file__),"{}".format(DEVICE.UUID))
'''
logging.basicConfig(filename='app.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d, %H:%M:%S',
                            level=logging.INFO)  # Global logging configuration

logger = logging.getLogger("Counter (umt_counter.py) - ")  # Logger for this module
'''
# --- Sets platform directories --------------------------------

gates = []
detections = []

try:
        with open(PATHS.GATES, 'rb') as f: 
            gates = pickle.load(f)
except:
    #logger.info('Gate load error')
    pass


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def cross(s1, s2):
    a, b = s1
    c, d = s2
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d) 


# now lets cycle throught each objects trajectory and determine if it has crossed either of the gates
def crossed_gates():
    #  compute detection centroids
    df['cx'] = df['bb_left'] + (0.5 * df['bb_width'])
    df['cy'] = df['bb_top']  + (0.5 * df['bb_height'])

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
                        timecat.insert(0, DEVICE.UUID)
                        detections.insert(0, timecat)


def confirmDetectionContents():
    if detections == []:
        if path.exists(PATHS.DETECTIONS):
            os.remove(PATHS.DETECTIONS)
        #logger.info('No Detections Found')
        return False
    else:
        with open(PATHS.DETECTIONS, 'wb') as f:
            pickle.dump(detections, f)
        sent = client_sock.sendFile(PATHS.DETECTIONS, DEVICE.UUID)
        #logger.info('Detections Found')
        return sent

#--- Looks for Outstanding detection files, if a file exists it's opened and
#--- any new detections are appended to the end and the file saved.
#--- The client attempts to transfer the file to the server returning a true
#--- or false dependant upon the success of a server connection.
def sendFile():
    try:
        with open(PATHS.DETECTIONS, 'rb') as f:
            #logger.info("Outstanding detections found. Inserted Outstanding Detections into File")
            previous_detections = pickle.load(f)
            for previous_detection in previous_detections:
                detections.insert(len(detections), previous_detection)
            #logger.info(detections)
            confDet = confirmDetectionContents()
            return confDet
    except FileNotFoundError:
        #logger.info('No Outstanding Detections Found. Dumping detections to file.')
        confDet = confirmDetectionContents()
        return confDet

# --- Looks for 'object_paths.csv' and loads them into 'df' returning 'True' if the path
# --- exists and 'False' if not.
def readObjPaths():
    global df
    if(path.exists(PATHS.CSV_PATH)):
        #logger.info('Loading CSV paths into pandas')
        df = pd.read_csv(PATHS.CSV_PATH, header=None, names=['frame', 'time', 'class', 'id', 'age', 'obj_t_since_last_update', 'obj_hits', 'bb_left', 'bb_top', 'bb_width', 'bb_height'])
        df.shape
        os.remove(PATHS.CSV_PATH)
        return True
    else:
        #logger.info('No CSV path file to send')
        return False
    
                        
# --- Pickle the detection list to a byte file --------
def count():
    
    readyTosend = readObjPaths()
    
    if readyTosend:
        crossed_gates() # Runs the algorithm to determine whether anybody has crossed the gates
        sent = sendFile()
        # If the file has been sent the existing detection file is deleted and if
        # not the file is retained to be appended to next time the counter runs.
        if not sent:
            #logger.info('Unable to send File, will retry after detections are calculated / No detections to send')
            return
        else:
            os.remove(PATHS.DETECTIONS)
            #logger.info('File Sent to Server')
    else:
        #logger.info("object_paths.csv - Not yet generated, will retry once scheduled time has elapsed.")
        pass


def main():
    
    schedule.every(30).seconds.do(count)
    
    while 1:
        schedule.run_pending()
        time.sleep(1)

    

if __name__ == '__main__':
    main()
