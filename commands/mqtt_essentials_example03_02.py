from command import *
import paho.mqtt.client as mqtt
import os.path
import time
import json


# Replace /Users/gaston/certificates with the path
# in which you saved the certificate authoritity file,
# the client certificate file and the client key
'''
certificates_path = "/Users/gaston/certificates"
ca_certificate = os.path.join(certificates_path, "ca.crt")
client_certificate = os.path.join(certificates_path, "device001.crt")
'''
# Replace localhost with the IP for the Mosquitto
# or other MQTT server
mqtt_server_host = "localhost"
mqtt_server_port = 1883
mqtt_keepalive = 60


class AGX_Cam:
    def __init__(self, name):
        self.name = name
        self.min_altitude = 0
        self.max_altitude = 30

    def print_with_name_prefix(self, message):
        print("{}: {}".format(self.name, message))

    def take_off(self):
        self.print_with_name_prefix("Taking off")

    def land(self):
        self.print_with_name_prefix("Landing")

    def land_in_safe_place(self):
        self.print_with_name_prefix("Landing in a safe place")

    def move_up(self):
        self.print_with_name_prefix("Moving up")

    def move_down(self):
        self.print_with_name_prefix("Moving down")

    def move_forward(self):
        self.print_with_name_prefix("Moving forward")

    def move_back(self):
        self.print_with_name_prefix("Moving back")

    def move_left(self):
        self.print_with_name_prefix("Moving left")

    def move_right(self):
        self.print_with_name_prefix("Moving right")

    def rotate_right(self, degrees):
        self.print_with_name_prefix("Rotating right {} degrees".format(degrees))

    def rotate_left(self, degrees):
        self.print_with_name_prefix("Rotating left {} degrees".format(degrees))

    def set_max_altitude(self, feet):
        self.max_altitude = feet
        self.print_with_name_prefix("Setting maximum altitude to {} feet".format(feet))

    def set_min_altitude(self, feet):
        self.min_altitude = feet
        self.print_with_name_prefix("Setting minimum altitude to {} feet".format(feet))

    def take_picture(self, resolution):
        self.resolution_y, self.resolution_x = resolution
        self.print_with_name_prefix(f"Taking a picture with resolution {self.resolution_y} : {self.resolution_x}")


class AGX_CamCommandProcessor:
    commands_topic = ""
    processed_commands_topic = ""
    active_instance = None

    def __init__(self, name, AGX_Cam):
        self.name = name
        self.AGX_Cam = AGX_Cam
        AGX_CamCommandProcessor.commands_topic = "commands/{}".format(self.name)
        AGX_CamCommandProcessor.processed_commands_topic = "processedcommands/{}".format(self.name)
        self.client = mqtt.Client(protocol=mqtt.MQTTv311)
        AGX_CamCommandProcessor.active_instance = self
        self.client.on_connect = AGX_CamCommandProcessor.on_connect
        self.client.on_message = AGX_CamCommandProcessor.on_message
        '''
        self.client.tls_set(ca_certs = ca_certificate,
            certfile=client_certificate,
            keyfile=client_key)
        '''
        self.client.connect(host=mqtt_server_host,
                            port=mqtt_server_port,
                            keepalive=mqtt_keepalive)

    @staticmethod
    def on_connect(client, userdata, flags, rc):
        print("Connected to the MQTT server")
        client.subscribe(
            AGX_CamCommandProcessor.commands_topic, 
            qos=2)
        client.publish(
            topic=AGX_CamCommandProcessor.processed_commands_topic,
            payload="{} is listening to messages".format(AGX_CamCommandProcessor.active_instance.name))

    @staticmethod
    def on_message(client, userdata, msg):
        if msg.topic == AGX_CamCommandProcessor.commands_topic:
            payload_string = msg.payload.decode('utf-8')
            print("I've received the following message: {0}".format(payload_string))
            try:
                message_dictionary = json.loads(payload_string)
                if COMMAND_KEY in message_dictionary:
                    command = message_dictionary[COMMAND_KEY]
                    AGX_Cam = AGX_CamCommandProcessor.active_instance.AGX_Cam
                    is_command_processed = False
                    if command == CMD_TAKE_PICTURE:
                        resolution = message_dictionary[KEY_RESOLUTION]
                        AGX_Cam.take_picture(resolution)
                        is_command_processed = True
                    if is_command_processed:
                        AGX_CamCommandProcessor.active_instance.publish_response_message(
                            message_dictionary)
                    else:
                        print("The message includes an unknown command.")
            except ValueError:
                # msg is not a dictionary
                # No JSON object could be decoded
                print("The message doesn't include a valid command.")

    def publish_response_message(self, message):
        response_message = json.dumps({
            SUCCESFULLY_PROCESSED_COMMAND_KEY:
                message[COMMAND_KEY]})
        result = self.client.publish(
            topic=self.__class__.processed_commands_topic,
            payload=response_message)
        return result

    def process_commands(self):
        self.client.loop()


if __name__ == "__main__":
    AGX_Cam = AGX_Cam("AGX_Cam_01")
    AGX_Cam_command_processor = AGX_CamCommandProcessor("AGX_Cam_01", AGX_Cam)
    while True:
        # Process messages and the commands every 1 second
        AGX_Cam_command_processor.process_commands()
        print("Command process")
        time.sleep(1)
