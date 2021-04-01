import camera
from datetime import datetime

'''
packet = {
            "ID" : CLIENT_ID,
            "TIME" : "TBC",
            "TOPIC": _TOPIC,
            "DESC" : _DESC,
            "DATA" : _DATA
        }
'''


class MSG_HANDLER:
    def __init__(self, _id, msg):
        self.id = _id
        #self.query = msg.topic
        self.query = msg
        self.data = None
        self.time = None
        self.dispatcher = {
            "cycle/img" : [camera.take_picture],
            "cycle/gates" : [camera.take_picture]
            }
        self.packet = None

    def perform_request(self, func_list):
        for f in func_list:
            return f()

    # Reads in an MQTT topic and performs an action based upon pre-determined dispatcher functions,
    # returning the data back to the create_response function as data.
    def create_response(self):
        self.data = self.perform_request(self.dispatcher[self.query]) 
        now = datetime.now()
        self.time = now.strftime("%d/%m/%y %H:%M:%S")


    def create_packet(self):
        packet = {
                    "ID" : self.id,
                    "TIME" : self.time,
                    "TOPIC" : self.query,
                    "DATA" : self.data
                }
        
        print(packet)

    def handle(self):
        self.create_response()

x = MSG_HANDLER('TEST', 'cycle/img')
x.create_packet()

        



    












'''
        img_bytes = pickle.dumps(img)

        packet = packet(msg.topic, "TBD" ,img)

        publish.single("server/img", packet, retain=True, client_id=CLIENT_ID, hostname=BROKER_HOST, port=BROKER_PORT, qos=0)
        print("cycle img published to server/img")

    if msg.topic == "cycle/gates" and msg == CLIENT_ID:
        device.GATES = msg.payload
'''