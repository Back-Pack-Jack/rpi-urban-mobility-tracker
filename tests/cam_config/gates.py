#import camera
from tkinter import Tk, Canvas, mainloop, PhotoImage, Label
from PIL import ImageTk, Image 
import cv2
import matplotlib.pyplot as plt
import numpy

class Gates:


    def __init__(self):
        self.gates = []
        self.my_window = None
        self.background_image = None
        self.bg_array = None
        self.C = None

    '''
    def take_pic(self):
        frame = camera.take_picture()
        self.bg_array = frame
        self.background_image = Image.fromarray(frame)
        #self.background_image.show()
    '''
 
    def createWindow(self):
        
        self.my_window = Tk() # Defines Tkinter Window
        self.background_image = ImageTk.PhotoImage(self.background_image) # Creates an img object
        cwidth = self.background_image.width()
        cheight = self.background_image.height()
        self.C = Canvas(self.my_window,width=cwidth,height=cheight,background='white')


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
            line = self.C.create_line(x1,y1,x2,y2,fill='red',width=3, dash=(6,2))
            gate = [(x1,y1), (x2,y2)]
            self.gates.append(gate)
            self.C.tag_raise(line)
            click_number=0


    def create_gates(self):    
        self.C.pack(side='top',fill='both',expand='yes')
        self.C.create_image(0,0, image=self.background_image, anchor='nw')

        self.C.bind('<Button-1>',self.draw_line)

        global click_number
        click_number=0
        self.my_window.mainloop()


    def display_gates(self):
        # load helper image
        pil_img = self.bg_array

        # let's validate that these gates are in the right place with a plot
        fig, ax = plt.subplots(figsize=(10,5))
        plt.imshow(pil_img)

        for g, gate in enumerate(self.gates):
            x1, y1 = gate[0]
            x2, y2 = gate[1]
            x, y = [x1, x2], [y1, y2]
            plt.plot(x, y, color='yellow', linewidth=3)

            plt.text(
                sum(x)/2,
                sum(y)/2 - 10, 
                f"gate: #{g}", 
                color='black', 
                horizontalalignment='center',
                fontsize=10)
                
        ax.set_aspect('equal')
        plt.show() # Show plot lines in screen
        plt.waitforbuttonpress(timeout=-1)


    def newDevice(self):
        #self.take_pic()
        self.createWindow()
        self.create_gates()
        #self.display_gates()

