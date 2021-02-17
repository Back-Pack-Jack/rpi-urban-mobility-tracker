import socket
import tqdm
import os
import sys
import pickle
import logging
from sys import platform
import time
import ssl
import logging
from config import SOCKET, PATHS
'''
logging.basicConfig(filename='app.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d, %H:%M:%S',
                            level=logging.INFO)  # Global logging configuration

logger = logging.getLogger("Socket (client_sock.py) - ")  # Logger for this module
'''
# --- Server Network Information
HOST = SOCKET.HOST 
HOST_NAME = SOCKET.HOST_NAME
PORT = SOCKET.PORT
SERVER_CERT = SOCKET.SERVER_CERT
CLIENT_CERT = SOCKET.CLIENT_CERT
CLIENT_KEY = SOCKET.CLIENT_KEY
DETECTIONS = PATHS.DETECTIONS
SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096 # send 4096 bytes each time step


# --- Attempts to connect to the server, if the devices fails it'll re-try every 60
# --- seconds until a succesful connection is made.
def connectToServer(host, port):
    sent = False
    global conn
    for i in range(3):
        try:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=SERVER_CERT)
            context.load_cert_chain(certfile=CLIENT_CERT, keyfile=CLIENT_KEY)

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # create the client socket
            conn = context.wrap_socket(s, server_side=False, server_hostname=HOST_NAME)
            conn.connect((HOST, PORT))

            #logger.info(f"[+] Connecting to {host}:{port}")
            sent = True
            break
        except:
            #logger.info('Cannot connect to server. Retrying...')
            time.sleep(5)
    #if not sent:
        #logger.info('Failed to connect to server.')
    return sent


def sendFile(filename, device):

    sent = connectToServer(HOST, PORT)

    if not sent:
        return sent

    filesize = sys.getsizeof(filename) # get the file size


    # send the filename and filesize
    conn.send(f"{device}{SEPARATOR}{filesize}".encode())
    #logger.info(f"{device}{SEPARATOR}{filesize}".encode())

    # start sending the file
    progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=BUFFER_SIZE)

    with open(filename, "rb") as f:
        for _ in progress:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                # file transmitting is done
                break
            # we use sendall to assure transimission in 
            # busy networks
            conn.sendall(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))
            time.sleep(0.1)
    # close the socket
    conn.shutdown(socket.SHUT_WR)
    time.sleep(7)
    conn.close()
    return sent

 