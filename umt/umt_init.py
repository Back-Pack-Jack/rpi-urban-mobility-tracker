import uuid
import pickle
from mqtt import init_UUID
import logging
from picamera import PiCamera
from time import sleep
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from tkinter import Tk, Canvas, mainloop, PhotoImage, Label
from PIL import ImageTk, Image 
import pickle
from sys import platform
from config import PATHS, DEVICE
'''
logging.basicConfig(filename='app.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d, %H:%M:%S',
                            level=logging.INFO)  # Global logging configuration

logger = logging.getLogger("Initialise (umt_init.py) - ")  # Logger for this module
'''
class UMTinit:

    def __init__(self):
        self.DEV_UUID = PATHS.UUID
        self.IMG_PATH = PATHS.IMG_PATH
        self.GATES = PATHS.GATES
        self.gates = []

    # Device looks to find it's UUID no. if it doesn't exist it generates one, communicates it to the server and saves it to 'uuid.ssg'
    def initialize_device(self):
        try:
            with open(self.DEV_UUID, 'rb') as f:
                self.DEV_UUID = pickle.load(f)
                #logger.info("Loaded UUID")
        except FileNotFoundError:
            device = str(uuid.uuid4())
            init_UUID(device)
            with open(self.DEV_UUID, "wb") as f:
                pickle.dump(device, f)

    def take_picture(self):
        camera = PiCamera()
        camera.start_preview()
        sleep(2)
        camera.capture(self.IMG_PATH)
        camera.stop_preview()

    def initialize_picture(self):
        try:
            f = open(self.IMG_PATH)
            f.close()
        except IOError:
            print("No image found, taking a new one.")
            self.take_picture()

    # --- Creating Gates --------------------------

    def draw_line(self, event):
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
            self.gates.append(gate)
            C.tag_raise(line)
            click_number=0


    def create_gates(self):    
        C.pack(side='top',fill='both',expand='yes')
        C.create_image(0,0, image=background_image, anchor='nw')

        C.bind('<Button-1>',self.draw_line)

        global click_number
        click_number=0
        my_window.mainloop()


    def display_gates(self):
        # load helper image
        pil_img = Image.open(self.IMG_PATH)

        # let's validate that these gates are in the right place with a plot
        fig, ax = plt.subplots(figsize=(15,15))
        plt.imshow(pil_img)


        for g, gate in enumerate(self.gates):
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

    def initialize_zones(self):
        global my_window
        global background_image
        global C
        
        try:
            with open(self.GATES, 'rb') as f:
                self.gates = pickle.load(f)
        except FileNotFoundError:
            my_window = Tk() # Defines Tkinter Window
            background_image = ImageTk.PhotoImage(Image.open(self.IMG_PATH)) # Creates an img object
            cwidth = background_image.width()
            cheight = background_image.height()
            C = Canvas(my_window,width=cwidth,height=cheight,background='white')
            self.create_gates()
            self.display_gates()
            with open(self.GATES, "wb") as f:
                pickle.dump(self.gates, f)
        
