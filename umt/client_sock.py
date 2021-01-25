import socket
import tqdm
import os
import sys

def sendFile(filename, device):
    SEPARATOR = "<SEPARATOR>"
    BUFFER_SIZE = 4096 # send 4096 bytes each time step

    host = "192.168.1.236" # the ip address or hostname of the server, the receiver
    port = 5001 # the port, let's use 5001

    s = socket.socket() # create the client socket
    print(f"[+] Connecting to {host}:{port}")

    s.connect((host, port))
    print("[+] Connected.")

    #filename = "rpi-urban-mobility-tracker/detections.ssg" # the name of file we want to send, make sure it exists
    #filesize = os.path.getsize(filename) # get the file size
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