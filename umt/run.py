import argparse
from device import Device
from mqttlib import MSG_HANDLER


def main():

    # Load Device Settings 
    # --- Device is loaded from settings.ssg file. If no file
    # --- exists one is created and the variables stored.
    device = Device()
    newDevice = device.load()
    if newDevice:
        message = MSG_HANDLER(device.UUID, 'cycle/newdevice')
        message.handle_request()
        


    '''
    if device.GATES == None:
        # Create gates server side.
        device.save()
        pass 
    '''
    '''
    # parse arguments
    parser = argparse.ArgumentParser(description='--- Raspbery Pi Urban Mobility Tracker ---')
    parser.add_argument('-modelpath', dest='model_path', type=str, required=False, help='specify path of a custom detection model')
    parser.add_argument('-labelmap', dest='label_map_path', default=DEFAULT_LABEL_MAP_PATH, type=str, required=False, help='specify the label map text file')
    parser.add_argument('-imageseq', dest='image_path', type=str, required=False, help='specify an image sequence')
    parser.add_argument('-video', dest='video_path', type=str, required=False, help='specify video file')
    parser.add_argument('-camera', dest='camera', default=False, action='store_true', help='specify this when using the rpi camera as the input')
    parser.add_argument('-threshold', dest='threshold', type=float, default=0.5, required=False, help='specify a custom inference threshold')
    parser.add_argument('-tpu', dest='tpu', required=False, default=False, action='store_true', help='add this when using a coral usb accelerator')
    parser.add_argument('-nframes', dest='nframes', type=int, required=False, default=10, help='specify nunber of frames to process')
    parser.add_argument('-display', dest='live_view', required=False, default=False, action='store_true', help='add this flag to view a live display. note, that this will greatly slow down the fps rate.')
    parser.add_argument('-save', dest='save_frames', required=False, default=False, action='store_true', help='add this flag if you want to persist the image output. note, that this will greatly slow down the fps rate.')
    args = parser.parse_args()
    
    # basic checks
    if args.model_path: assert args.label_map_path, "when specifying a custom model, you must also specify a label map path using: '-labelmap <path to labelmap.txt>'"
    if args.model_path: assert os.path.exists(args.model_path)==True, "can't find the specified model..."
    if args.label_map_path: assert os.path.exists(args.label_map_path)==True, "can't find the specified label map..."
    if args.video_path: assert os.path.exists(args.video_path)==True, "can't find the specified video file..."
    '''

if __name__ == '__main__':
    main()
