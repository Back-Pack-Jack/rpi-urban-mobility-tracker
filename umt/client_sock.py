import socket
import tqdm
import os
import sys
import pickle
import logging
from sys import platform
import time
import logging
from config import SOCKET, PATHS

logging.basicConfig(level=logging.WARNING)  # Global logging configuration
logger = logging.getLogger("UMT - Client Socket")  # Logger for this module
logger.setLevel(logging.INFO) # Debugging for this file.

# --- Server Network Information
HOST = SOCKET.HOST 
PORT = SOCKET.PORT 
DETECTIONS = PATHS.DETECTIONS
SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096 # send 4096 bytes each time step


# --- Attempts to connect to the server, if the devices fails it'll re-try every 60
# --- seconds until a succesful connection is made.
def connectToServer(host, port):
    sent = False
    for i in range(3):
        try:
            global s
            s = socket.socket() # create the client socket
            s.connect((host, port))
            logger.info(f"[+] Connecting to {host}:{port}")
            sent = True
            break
        except:
            logger.info('Cannot connect to server. Retrying...')
            time.sleep(5)
    logger.info('Failed to connect to server.')
    return sent


def sendFile(filename, device):

    sent = connectToServer(HOST, PORT)

    if not sent:
        return sent

    filesize = sys.getsizeof(filename) # get the file size


    # send the filename and filesize
    s.send(f"{device}{SEPARATOR}{filesize}".encode())

    # start sending the file
    progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)

    with open(filename, "rb") as f:
        for _ in progress:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                # file transmitting is done
                break
            # we use sendall to assure transimission in 
            # busy networks
            s.sendall(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))
    # close the socket
    s.close()
    return sent


 