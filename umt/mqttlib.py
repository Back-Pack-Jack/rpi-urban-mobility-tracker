#import camera
from datetime import datetime
#from client_sock import sendPacket
#from device import Device

# UUID (static) (ID)
# SEND, RECEIEVE, TRANSFER (varying) - inc. (send / recv / transfer (S/R/T)) Handled in sockets (sockets should be threaded)
# ACTION - DO SOMETHING (ACT)
# Time / Date (static) (TD)
# Authentication Key >> MSG --> RESPONSE

# Functions
# Live Stream
# Share large data packets
# Share data between devices

# PROTOCOL
# ID<:>TD<:>SRT<:>ACT<:>DATA

# READ PACKETS AND FORM ACTIONS
# DETERMINE WHETHER IT'S A SEND OR RCV ACTION SEND(PERFORM ACTION --> SEND INFORMATION) RECV(RECV INFORMATION --> PERFORM ACTION)

def hello():
    return("hello")


def goodbye():
    return("goodbye")

class IOT_HANDLER:

    class PROTOCOL:
        def __init__(self, msg):
            self.MSG = msg
            self.ID = None
            self.TD = None
            self.SRT = None
            self.TRS = {"S":"R", "T":"T", "R":"S"}
            self.ACT = None
            self.DATA = None
            self.DISPATCH = {
                "PRINT GREETING" : [hello, goodbye, hello],
                "PRINT GOODBYE" : [goodbye]
                }
            self.SEPARATOR = "<:>"

        def read_packet(self):
            #msg = dict(_.split("=") for _ in self.MSG.split(self.SEPARATOR))
            self.ID = msg["ID"]
            self.TD = msg["TD"]
            self.SRT = msg["SRT"]
            self.ACT = msg["ACT"]
            self.DATA = msg["DATA"]

        def create_packet(self):
            self.MSG = {
                "ID" : "RETURN_SENDER",
                "TD" : datetime.now().strftime("%d/%m/%y %H:%M:%S"),
                "SRT" : self.TRS[self.SRT],
                "ACT" : self.ACT,
                "DATA" : self.DATA
                }
            print(self.MSG)

        def perform_action(self):
            if self.SRT == "S":
                self.DATA = self._perform(self.ACT)
                self.create_packet()
        
        # TO DO - Thread dispatch functions and pool workers
        def _perform(self, ACT): 
            data_d = dict()
            data_l = list()
            for action in self.DISPATCH[ACT]:
                data_d[ACT] = f"{action.__name__} = {action()}"
                data_l.append(data_d.copy())
            return data_l

        def _repackage(self, MSG, DATA):
            pass

        def handle(self):
            self.read_packet()
            self.perform_action()


MSG = {
        "ID" : "SENDER",
        "TD" : datetime.now().strftime("%d/%m/%y %H:%M:%S"),
        "SRT" : "SRT",
        "ACT" : "ACT",
        "DATA" : "DATA"
        }    

x = IOT_HANDLER.PROTOCOL(MSG)
x.handle()

'''
class MSG_HANDLER:

    def __init__(self, msg):

        self.id = _id
        self.function = _function
        self.timedate = _timedate
        
        self.timenow = datetime.now().strftime("%d/%m/%y %H:%M:%S")

        self.query = msg
        self.snd_rcv = _sr
        self.data = None
        self.time = None
        self.dispatcher = {
            "cycle/newdevice" : [camera.take_picture, device.UUID],
            "cycle/img" : [camera.take_picture]
            }

    def perform_request(self, func_list):
        
        


        def form_data():
            if self.snd_rcv == "S":
                for f in func_list:
                    try:
                        self.data = f()
                    except Exception:
                        self.data = f
            elif self.snd_rcv == "R":
                self.data = None

        
        def snd_packet():  
                form_data()       
                packet = create_packet()
                sendPacket(packet)
            
        snd_packet()


    # Reads in an MQTT topic and performs an action based upon pre-determined dispatcher functions,
    # returning the data back to the create_response function as data.
    def handle_request(self):
        self.perform_request(self.dispatcher[self.query]) 
        

'''