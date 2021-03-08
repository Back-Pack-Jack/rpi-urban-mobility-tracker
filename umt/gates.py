from camera import Camera
from tkinter import Tk, Canvas, mainloop, PhotoImage, Label
from PIL import ImageTk, Image 


class Gates:


    def __init__(self):
        self.gates = []
        self.my_window = None
        self.background_image = None
        self.C = None

    def test(self):
        cam = Camera()
        x = cam.take_picture()

    # --- Creating Gates --------------------------

    def createWindow(self):
        self.test()
        my_window = Tk() # Defines Tkinter Window
        '''
        background_image = ImageTk.PhotoImage(Image.open(self.background_image)) # Creates an img object
        cwidth = background_image.width()
        cheight = background_image.height()
        C = Canvas(my_window,width=cwidth,height=cheight,background='white')
        '''

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

    def newDevice(self):
        self.createWindow()
        self.create_gates()
        self.display_gates()

Gates = Gates()
Gates.createWindow()
