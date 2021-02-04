#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import os
import time
import argparse
import signal
import sys
import datetime

import cv2
import numpy as np

# deep sort
from deep_sort.tracker import Tracker
from deep_sort import preprocessing
from deep_sort import nn_matching
 
# umt utils
from umt_utils import parse_label_map
from umt_utils import initialize_detector
from umt_utils import initialize_img_source
from umt_utils import generate_detections

if sys.platform == 'linux' or platform == 'linux2':
    TRACKER_OUTPUT_TEXT_FILE = 'umt/object_paths.csv'
if sys.platform == 'darwin':
    TRACKER_OUTPUT_TEXT_FILE = 'object_paths.csv'

#--- CONSTANTS ----------------------------------------------------------------+

LABEL_PATH = "models/pednet/model/labels.txt"
DEFAULT_LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), LABEL_PATH)
TRACKER_OUTPUT_TEXT_FILE = 'object_paths.csv'

# deep sort related
MAX_COSINE_DIST = 0.4
NN_BUDGET = None
NMS_MAX_OVERLAP = 1.0

#--- FILES --------------------------------------------------------------------+
global tracked_list
tracked_list = []

#--- MAIN ---------------------------------------------------------------------+

def signal_handler(sig, frame):
    print('You pressed ctrl + c')
    print(tracked_list)
    with open(TRACKER_OUTPUT_TEXT_FILE, 'w') as out_file:
        for x in tracked_list:
            print(x, file=out_file)

    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    print('Running. Press Ctrl + C to exit.')
    global tracked_list
    threshold = 0.2

    print('> INITIALIZING UMT...')
    print('   > THRESHOLD:',threshold)

	# parse label map
    labels = parse_label_map(DEFAULT_LABEL_MAP_PATH)
    
    # initialize detector
    interpreter = initialize_detector()
 
 	# initialize deep sort tracker   
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DIST, NN_BUDGET)
    tracker = Tracker(metric) 

    # initialize image source
    img_generator = initialize_img_source()


    # main tracking loop
    print('\n> TRACKING...')
    #with open(TRACKER_OUTPUT_TEXT_FILE, 'w') as out_file:

    for i, pil_img in enumerate(img_generator()):
    
        f_time = int(time.time())
        print('> FRAME:', i)
        
        # add header to trajectory file
        '''
        if i == 0:
            header = (f'frame_num, rpi_time, obj_class, obj_id, obj_age,'
                'obj_t_since_last_update, obj_hits,'
                'xmin, ymin, xmax, ymax')

            tracked_list.append(header)
        '''
        # get detections
        detections = generate_detections(pil_img, interpreter, threshold)
        
        # proceed to updating state
        if len(detections) == 0: print('> no detections...')
        else:
        
            # update tracker
            tracker.predict()
            tracker.update(detections)
            
            # save object locations
            if len(tracker.tracks) > 0:
                for track in tracker.tracks:
                    bbox = track.to_tlbr()
                    class_name = labels[track.get_class()]
                    row = (f'{i},{f_time},{class_name},'
                        f'{track.track_id},{int(track.age)},'
                        f'{int(track.time_since_update)},{str(track.hits)},'
                        f'{int(bbox[0])},{int(bbox[1])},'
                        f'{int(bbox[2])},{int(bbox[3])}')
                    tracked_list.append(row)
            
        if int(time.strftime("%M")) == 29 | int(time.strftime("%M")) == 0:
            print("send data")
            
    cv2.destroyAllWindows()         
    pass


#--- MAIN ---------------------------------------------------------------------+

if __name__ == '__main__':
    main()
    
     
#--- END ----------------------------------------------------------------------+
