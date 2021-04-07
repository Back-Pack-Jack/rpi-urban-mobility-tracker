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
from struct import pack

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
DEVICE = "TEST"
SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096 # send 4096 bytes each time step


# --- Attempts to connect to the server, if the devices fails it'll re-try every 60
# --- seconds until a succesful connection is made.
def connectToServer():

    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=SERVER_CERT)
    context.load_cert_chain(certfile=CLIENT_CERT, keyfile=CLIENT_KEY)

    for i in range(3):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn = context.wrap_socket(s, server_side=False, server_hostname=HOST_NAME)
            conn.connect((HOST, PORT))
            print(f"[+] Connecting to {HOST}:{PORT}")
            return True, conn
        except:
            print("[+] Cannot connect to server. Retrying...")
            time.sleep(5)

    print(f"[+] Failed to connect to {HOST}:{PORT}")
    return False, None


def sendPacket(packet):
    ID = packet["ID"]
    TIME = packet["TIME"]
    TOPIC = packet["TOPIC"]

    def pickledPacket(packet):
        pickledPacket = pickle.dumps(packet)
        return pickledPacket
    
    connected, conn = connectToServer()
    if not connected:
        return

    packet = pickledPacket(packet)
    packetsize = sys.getsizeof(packet)

    conn.send(f"{TIME}{SEPARATOR}{ID}{SEPARATOR}{packetsize}{SEPARATOR}{TOPIC}".encode())
    
    length = pack('>Q', len(packet))
    conn.sendall(length)
    conn.sendall(packet)
    ack = conn.recv(1)
    
    print(f"[+] Data transfer successfully completed.")

    conn.shutdown(socket.SHUT_WR)
    conn.close()


