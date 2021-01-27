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

with open('rpi-urban-mobility-tracker/umt/uuid.ssg', 'rb') as f:
            UUID = pickle.load(f)
            logger.info("Loaded UUID")

# --- Initialises the MQTT client to send a message to the server

IMG_PATH = 'rpi-urban-mobility-tracker/umt/highway02_frame000010.png'
CSV_PATH = 'rpi-urban-mobility-tracker/umt/object_paths_highway02_pednet.txt'
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

my_window = Tk() # Defines Tkinter Window
background_image = ImageTk.PhotoImage(Image.open(IMG_PATH)) # Creates an img object
cwidth = background_image.width()
cheight = background_image.height()
C = Canvas(my_window,width=cwidth,height=cheight,background='white')

# --- Window for drawing gates --------------------------
def draw_line(event):
    global click_number
    global x1,y1
    if click_number==0:
        x1=event.x
        y1=event.y
        click_number=1
    else:
        x2=event.x
        y2=event.y
        line = C.create_line(x1,y1,x2,y2,fill='orange',width=5, dash=(4,2))
        gate = [(x1,y1), (x2,y2)]
        gates.append(gate)
        C.tag_raise(line)
        click_number=0


def create_gates():    
    C.pack(side='top',fill='both',expand='yes')
    C.create_image(0,0, image=background_image, anchor='nw')

    C.bind('<Button-1>',draw_line)

    global click_number
    click_number=0
    my_window.mainloop()


def display_gates():
    # load helper image
    pil_img = Image.open(IMG_PATH)

    # let's validate that these gates are in the right place with a plot
    fig, ax = plt.subplots(figsize=(15,15))
    plt.imshow(pil_img)

    print(gates)

    for g, gate in enumerate(gates):
        x1, y1 = gate[0]
        x2, y2 = gate[1]
        x, y = [x1, x2], [y1, y2]
        plt.plot(x, y, color='orange', linewidth=5)

        plt.text(
            sum(x)/2,
            sum(y)/2 - 10, 
            f"gate: #{g}", 
            color='orange', 
            horizontalalignment='center',
            fontsize=10)
            
    ax.set_aspect('equal')
    plt.show() # Show plot lines in screen


try:
    with open('gates.ssg', 'rb') as f:
        gates = pickle.load(f)
except FileNotFoundError:
    create_gates()
    display_gates()
    with open("gates.ssg", "wb") as f:
        pickle.dump(gates, f)



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
    filename = "rpi-urban-mobility-tracker/detections.ssg"
    client_sock.sendFile(filename, DEVICE)
    logger.info("Transfer of detection information to server complete!")

schedule.every(5).minutes.do(count)

while 1:
    schedule.run_pending()
    time.sleep(1)


