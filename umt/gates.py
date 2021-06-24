from tkinter import Tk, Canvas, mainloop, PhotoImage, Label
from PIL import ImageTk, Image 
import cv2
import time
import matplotlib.pyplot as plt
import pickle

class Gates:

    def __init__(self):
        self.gates = []
        self.my_window = None
        self.background_image = None
        self.C = None
        self.frame = None
        self.click_number = None
        self.x1 = None
        self.y1 = None
        self.path = './gates.ssg'

    def captureFrame(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        now = time.time()
        while time.time() < now + 2:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        self.frame = frame
        cap.release()
        cv2.destroyAllWindows()

    # --- Creating Gates --------------------------

    def createWindow(self):
        self.my_window = Tk() # Defines Tkinter Window
        background_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        background_image = Image.fromarray(background_image) # Creates an img object
        self.background_image = ImageTk.PhotoImage(background_image)
        cwidth, cheight = background_image.size
        self.C = Canvas(self.my_window, width=cwidth, height=cheight, background='white')


    def draw_line(self, event):
        if self.click_number == 0:
            self.x1 = event.x
            self.y1 = event.y
            self.click_number = 1
        else:
            x2 = event.x
            y2 = event.y
            line = self.C.create_line(self.x1, self.y1, x2, y2, fill='orange', width=5, dash=(4,2))
            gate = [(self.x1,self.y1), (x2,y2)]
            self.gates.append(gate)
            self.C.tag_raise(line)
            self.click_number=0


    def create_gates(self):
        self.C.pack(side='top',fill='both',expand='yes')
        self.C.create_image(0,0, image=self.background_image, anchor='nw')

        self.C.bind('<Button-1>',self.draw_line)

        self.click_number=0
        self.my_window.mainloop()


    def display_gates(self):
        fig, ax = plt.subplots(figsize=(15,15))
        frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        plt.imshow(frame)


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


    def save_gates(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.gates, f)
            print("Gates Saved")


    def load_gates(self):
        try:
            with open(self.path, 'rb') as f:
                self.gates = pickle.load(f)
                print(f"Gates Loaded : {self.gates}")
        except FileNotFoundError:
            print("Gates Don't Exists")


    def newDevice(self):
        self.captureFrame()
        self.createWindow()
        self.create_gates()
        self.display_gates()
        self.save_gates()

gates = Gates()
gates.newDevice()