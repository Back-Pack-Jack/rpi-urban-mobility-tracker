#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import os
import time
import argparse
import signal
import sys
import cv2
import numpy as np
import dlib

# deep sort
from deep_sort.tracker import Tracker
from deep_sort import preprocessing
from deep_sort import nn_matching
 
# umt utils
from umt_utils import parse_label_map
from umt_utils import initialize_detector
from umt_utils import initialize_img_source
from umt_utils import generate_detections

from utils.centroidtracker import CentroidTracker

from umt_utils import visualize_boxes_and_labels_on_image_array

from utils.color_recognition_module import color_recognition_api
from utils import label_map_util
from utils import visualization_utils as vis_util

from utils import optical_flow
from utils.optical_flow import App

from PIL import Image

#config
from config import PATHS

#counter


TRACKER_OUTPUT_TEXT_FILE = PATHS.CSV_PATH
COLORS = (np.random.rand(32, 3) * 255).astype(int)

#--- CONSTANTS ----------------------------------------------------------------+

LABEL_PATH = "models/tpu/mobilenet_ssd_v2_coco_quant/coco_labels.txt"
LABEL_PATH_PB = "resources/mscoco_label_map.pbtxt"
DEFAULT_LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), LABEL_PATH)
DEFAULT_LABEL_MAP_PATH_PB = os.path.join(os.path.dirname(__file__), LABEL_PATH_PB)
NUM_CLASSES = 90

# deep sort related
MAX_COSINE_DIST = 0.4
NN_BUDGET = None
NMS_MAX_OVERLAP = 1.0

#--- FILES --------------------------------------------------------------------+
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
    #signal.signal(signal.SIGINT, signal_handler)
    #print('Running. Press Ctrl + C to exit.')
    global tracked_list
    threshold = 0.5

    print('> INITIALIZING UMT...')
    print('   > THRESHOLD:',threshold)

	# parse label map
    labels = parse_label_map(DEFAULT_LABEL_MAP_PATH)

    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(DEFAULT_LABEL_MAP_PATH_PB)
    categories = label_map_util.convert_label_map_to_categories(label_map,
            max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
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

    total_passed_vehicle = 0  # using it to count vehicles
    speed = 'waiting...'
    direction = 'waiting...'
    size = 'waiting...'
    color = 'waiting...'

    #for i, pil_img in enumerate(img_generator()):

    VID = 'resources/input_video.mp4'
    cap = cv2.VideoCapture(VID)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    ret, frame = cap.read()
    time.sleep(2)
    #optical_flow.init_opt_flow(frame)
    optiflow = App()

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

    while cap.isOpened():
        # read and return frame from camera
        ret, frame = cap.read() 

        # if the video stream ends then terminate the loop
        if not ret:
            print ('end of the video file...')
            break

        print('> FRAME:', cap.get(1))

        # create an image object from an array and pass it generate detections
        # to return a deep sort object and a dictionary containing detection info 
        pil_img = Image.fromarray(frame)
        detections, det_info = generate_detections(pil_img, interpreter, threshold)

        #flow_frame, flow_mask = optical_flow.opt_flow(frame)
        optiflow.frame = frame
        optiflow.run()
        
    
        f_time = int(time.time())


        input_frame = np.array(pil_img)
        (counter, csv_line) = \
                    visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    input_frame,
                    det_info['bboxes'],
                    det_info['classes'],
                    det_info['scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    )
        
        # Saves tracked to file every x frames
        if len(tracked_list) >= 1000:
            with open(TRACKER_OUTPUT_TEXT_FILE, 'a') as out_file:
                for x in tracked_list:
                    print(x, file=out_file)
            print('dumped tracked to list')
            tracked_list = []

        # proceed to updating state
        if len(detections) == 0: print('> no detections...')
        else:
            trackers = []
        
            # input_frame = np array
            # pil_img = Pillow Image

            im_width, im_height = pil_img.size
            # update tracker
            tracker.predict()
            tracker.update(detections)
            
            def get_bbox_xy(bbox):
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2])
                ymax = int(bbox[3])
                return xmin, ymin, xmax, ymax

            def calc_centroid(xmin, ymin, xmax, ymax):
                cx = xmin + (0.5 * xmax)
                cy = ymin + (0.5 * ymax)
                return cx, cy

            # save object locations
            if len(tracker.tracks) > 0:
                for track in tracker.tracks:
                    bbox = track.to_tlbr()
                    xmin, ymin, xmax, ymax = get_bbox_xy(bbox)
                    cx, cy = calc_centroid(xmin, ymin, xmax, ymax)
                    detected_vehicle_image = input_frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                    try:
                        predicted_color = color_recognition_api.color_recognition(detected_vehicle_image)
                    except IndexError:
                        predicted_color = 'N/A'
                    class_name = labels[track.get_class()]
                    row = (f'{cap.get(1)},{f_time},{class_name},'
                        f'{track.track_id},{int(track.age)},'
                        f'{int(track.time_since_update)},{str(track.hits)},'
                        f'{xmin},{ymin},'
                        f'{xmax},{ymax}')
                    #print(track.track_id, top, bottom, left, right)
                    #print(track.track_id, predicted_color)
                    #print(detected_vehicle_image)
                    tracked_list.append(row)
            

            # only for live display
            if True:
            
            	# convert pil image to cv2
                cv2_img = frame
            
            	# cycle through actively tracked objects
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    
                    # draw detections and label
                    bbox = track.to_tlbr()
                    class_name = labels[track.class_name]
                    colour = COLORS[int(track.track_id) % len(COLORS)].tolist()

                    cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colour, 2)
                    cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(str(class_name))+len(str(track.track_id))+len(str(predicted_color)))*17, int(bbox[1])), colour, -1)
                    cv2.putText(cv2_img, str(class_name) + "-" + str(track.track_id) + '-' + str(predicted_color), (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

                # live view
                if True:
                    cv2.imshow("Vehicle Detection", cv2_img)
                    #cv2.imshow("Vehicle Detection", input_frame)
                    #input_frame = cv2.add(cv2_img, flow_mask) # Old Optical Flow
                    #cv2.imshow("Vehicle Detection", optiflow.vis) # Optical Flow
                    cv2.waitKey(1)

    cv2.destroyAllWindows()         
    pass


#--- MAIN ---------------------------------------------------------------------+

if __name__ == '__main__':
    main()
    
     
#--- END ----------------------------------------------------------------------+
