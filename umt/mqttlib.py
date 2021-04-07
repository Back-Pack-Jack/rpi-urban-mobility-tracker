import camera
from datetime import datetime
from client_sock import sendPacket
from device import Device

class MSG_HANDLER:
    def __init__(self, _id, msg):
        device = Device()
        self.id = _id
        #self.query = msg.topic
        self.query = msg
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

        for f in func_list:
            try:
                self.data = f()
            except Exception:
                self.data = f
            packet = create_packet()
            sendPacket(packet)

    # Reads in an MQTT topic and performs an action based upon pre-determined dispatcher functions,
    # returning the data back to the create_response function as data.
    def handle_request(self):
        self.perform_request(self.dispatcher[self.query]) 
        

        