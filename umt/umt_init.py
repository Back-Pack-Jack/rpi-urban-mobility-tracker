import uuid
import pickle
from mqtt import init_UUID
import logging
#rom picamera import PiCamera
from time import sleep
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from tkinter import Tk, Canvas, mainloop, PhotoImage, Label
from PIL import ImageTk, Image 
import pickle
from sys import platform

logging.basicConfig(level=logging.WARNING)  # Global logging configuration
logger = logging.getLogger("init - umt_init")  # Logger for this module
logger.setLevel(logging.INFO) # Debugging for this file.

class UMTinit:

    def __init__(self):
        if platform == 'linux' or platform == 'linux2':
            self.UUID = 'uuid.ssg'
            self.IMG_PATH = 'image_capture.png'
            self.GATES = 'gates.ssg'
        if platform == 'darwin':
            self.UUID = 'rpi-urban-mobility-tracker/umt/uuid.ssg'
            self.IMG_PATH = 'rpi-urban-mobility-tracker/umt/image_capture.png'
            self.GATES = 'rpi-urban-mobility-tracker/umt/gates.ssg'

    # Device looks to find it's UUID no. if it doesn't exist it generates one, communicates it to the server and saves it to 'uuid.ssg'
    def initialize_device(self):
        try:
            with open(self.UUID, 'rb') as f:
                self.UUID = pickle.load(f)
                logger.info("Loaded UUID")
        except FileNotFoundError:
            self.UUID = str(uuid.uuid4())
            init_UUID(self.UUID)
            with open(self.UUID, "wb") as f:
                pickle.dump(self.UUID, f)

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
            take_picture()

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
            gates.append(gate)
            C.tag_raise(line)
            click_number=0


    def create_gates(self):    
        C.pack(side='top',fill='both',expand='yes')
        C.create_image(0,0, image=background_image, anchor='nw')

        C.bind('<Button-1>',draw_line)

        global click_number
        click_number=0
        my_window.mainloop()


    def display_gates(self):
        # load helper image
        pil_img = Image.open(self.IMG_PATH)

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

    def initialize_zones(self):
        global my_window
        global background_image
        global C
        
        try:
            with open(self.GATES, 'rb') as f:
                gates = pickle.load(f)
        except FileNotFoundError:
            my_window = Tk() # Defines Tkinter Window
            background_image = ImageTk.PhotoImage(Image.open(self.IMG_PATH)) # Creates an img object
            cwidth = background_image.width()
            cheight = background_image.height()
            C = Canvas(my_window,width=cwidth,height=cheight,background='white')
            gates = []
            create_gates()
            display_gates()
            with open(self.GATES, "wb") as f:
                pickle.dump(gates, f)
        
