import camera
from datetime import datetime
from client_sock import sendPacket
from device import Device

class MSG_HANDLER:
    def __init__(self, _id, msg, _sr):
        device = Device()
        self.id = _id
        self.query = msg
        self.snd_rcv = _sr
        self.data = None
        self.time = None
        self.dispatcher = {
            "cycle/newdevice" : [camera.take_picture, device.UUID],
            "cycle/img" : [camera.take_picture]
            }

    def perform_request(self, func_list):
        
        def create_packet():
            now = datetime.now()
            self.time = now.strftime("%d/%m/%y %H:%M:%S")
            packet = {
                        "ID" : self.id,
                        "TIME" : self.time,
                        "TOPIC" : self.query,
                        "DATA" : self.data
                    }
            return(packet)


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
        

        