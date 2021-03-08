import cv2

# Change to encasulated function 
class Camera:


    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    
    def start_cam(self, win_desc):

        if not self.cam.isOpened():
            raise IOError("Cannot open camera")

        while True:
            ret, frame = self.cam.read()
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.imshow(win_desc, frame)
            k = cv2.waitKey(1)

            if k%256 == 32:
                # SPACE pressed
                _, frame = self.cam.read()
                break

        self.cam.release()
        cv2.destroyAllWindows()
        return(frame)


    def preview(self):

        win_desc = "Preview"
        self.start_cam(win_desc)


    def take_picture(self):

        win_desc = "Take Picture"
        picture = self.start_cam(win_desc)
        return picture
