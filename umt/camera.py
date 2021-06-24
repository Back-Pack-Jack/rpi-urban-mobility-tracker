import cv2
import time

def take_picture():

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        raise IOError("Cannot open camera")

    time.sleep(2)

    ret, frame = cam.read()
    time.sleep(2)
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    cam.release()
    #cv2.destroyAllWindows()
    return(frame)
