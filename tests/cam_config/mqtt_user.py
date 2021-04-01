import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import logging
import time
import cv2
from gates import Gates
import pickle
from PIL import ImageTk, Image 


# --- Initialise Logging
logging.basicConfig(    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d, %H:%M:%S',
                        level=logging.INFO)  # Global logging configuration

logger = logging.getLogger("MQTT - ")  # Logger for this module

BROKER_HOST = '192.168.1.236'
BROKER_PORT = 1883
CLIENT_ID = "SERVER"
TOPIC = [("server/location", 2), ("server/img", 2), ("server/count", 2), ("server/sensor", 2)]
TOPIC_DESC = [top[0] for top in TOPIC]
DATA_BLOCK_SIZE = 2000

def msg_handler(msg):

    def packet(_TOPIC, _DESC, _DATA):
        packet = {
            "ID" : CLIENT_ID,
            "TIME" : "TBC",
            "TOPIC": _TOPIC,
            "DESC" : _DESC,
            "DATA" : _DATA
        }
        pack = pickle.dumps(packet)
        return pack
    
    def un_packet(packet):
        pack = pickle.loads(packet)
        print(pack)

    un_packet(msg.payload)

    '''
    if msg.topic == "server/img":
        print('incoming msg')
        _Gates = Gates()
        img_bytes = pickle.loads(msg.payload)
        rgb_img_array = cv2.cvtColor(img_bytes, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_img_array)
        _Gates.bg_array = rgb_img_array
        _Gates.background_image = img
        _Gates.newDevice()
        publish.single("server/img", img_bytes, retain=True, client_id=CLIENT_ID, hostname=BROKER_HOST, port=BROKER_PORT, qos=0)
        print(_Gates.gates)
    '''
    '''
        print(jpg_original)
        with open('test.jpg', 'wb') as f_output:
            f_output.write(jpg_original)
    '''


def on_connect(client, user_data, flags, connection_result_code):                              
    client.subscribe(TOPIC)
    logger.info("Connected to MQTT Broker, Subscribed to {}".format(TOPIC_DESC))


def on_disconnect(client, user_data, disconnection_result_code):                               

    logger.info("Disconnected from MQTT Broker")


def on_message(client, user_data, msg):                                      

    #msg_dec = msg.payload.decode("utf-8") # Writes the decoded msg to an object
    msg_handler(msg)

    
def on_publish(client, user_data, connection_result_code):
    #logger.info("Message Published")
    pass


def signal_handler(sig, frame): # Capture Control+C and disconnect from Broker.
    #logger.info("You pressed Control + C. Shutting down, please wait...")
    client.disconnect() # Graceful disconnection.
    sys.exit(0)


def init_mqtt():

    #logger.info("Creating an instance of MQTT_Client")
    
    global client

    #logger.info("Initialising Client")
    client = mqtt.Client(
        client_id=CLIENT_ID,
        clean_session=False)

    # Route Paho logging to Python logging.
    client.enable_logger()

    # Setup callbacks
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.on_publish = on_publish
    
    # Connect to Broker.
    while True:
        try:
            client.connect(BROKER_HOST, BROKER_PORT)
            break
        except:
            logger.exception("Couldn't connect to broker. Retrying...")
            time.sleep(5)
            
    client.loop_forever()


def main():
   # signal.signal(signal.SIGINT, signal_handler)  # Capture Control + C
    #logger.info("Listening for messages on topic '" + str(TOPIC) + "'. Press Control + C to exit.")

    init_mqtt()
    #signal.pause()

if __name__ == "__main__":
    main()