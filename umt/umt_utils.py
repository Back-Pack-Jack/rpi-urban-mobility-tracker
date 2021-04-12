import os
from time import sleep
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np

import cv2
from scipy.spatial.distance import cosine

import imutils
from imutils.video import VideoStream

# deep sort
from deep_sort import generate_detections as gd
from deep_sort.detection import Detection
from deep_sort.preprocessing import non_max_suppression

# constants
nms_max_overlap = 1.0

# initialize an instance of the deep-sort tracker
w_path = os.path.join(os.path.dirname(__file__), 'deep_sort/mars-small128.pb')
encoder = gd.create_box_encoder(w_path, batch_size=1)
    

def camera_frame_gen():

    # initialize the video stream and allow the camera sensor to warmup
    print("> starting video stream...")
    vs = VideoStream(src=0).start()
    sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # pull frame from video stream
        frame = vs.read()

        # array to PIL image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield Image.fromarray(frame)

    pass


def initialize_img_source():
        
    # track objects from camera source
    return camera_frame_gen


def initialize_detector():
                
    TPU_PATH = 'models/tpu/mobilenet_ssd_v2_coco_quant/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite' 
    model_path = os.path.join(os.path.dirname(__file__), TPU_PATH)
        
    _, *device = model_path.split('@')
    edgetpu_shared_lib = 'libedgetpu.so.1'
    interpreter = tflite.Interpreter(
            model_path,
            experimental_delegates=[
                tflite.load_delegate(edgetpu_shared_lib,
                    {'device': device[0]} if device else {})
            ])
    interpreter.allocate_tensors()

    return interpreter


def generate_detections(pil_img_obj, interpreter, threshold):
    
    # resize image to match model input dimensions
    img = pil_img_obj.resize((interpreter.get_input_details()[0]['shape'][2], 
                              interpreter.get_input_details()[0]['shape'][1]))

    # add n dim
    input_data = np.expand_dims(img, axis=0)

    # infer image
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()

    # collect results
    bboxes = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[1]['index']) + 1).astype(np.int32)
    scores = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[2]['index']))
    num = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[3]['index']))
    
    # keep detections above specified threshold
    keep_idx = np.less(scores[np.greater(scores, threshold)], 1)
    bboxes  = bboxes[:keep_idx.shape[0]][keep_idx]
    classes = classes[:keep_idx.shape[0]][keep_idx]
    scores = scores[:keep_idx.shape[0]][keep_idx]
    
    # keep detections of specified classes
    #
    #
	#...
	

    # denormalize bounding box dimensions
    if len(keep_idx) > 0:
        bboxes[:,0] = bboxes[:,0] * pil_img_obj.size[1]
        bboxes[:,1] = bboxes[:,1] * pil_img_obj.size[0]
        bboxes[:,2] = bboxes[:,2] * pil_img_obj.size[1]
        bboxes[:,3] = bboxes[:,3] * pil_img_obj.size[0]
    
    # convert bboxes from [ymin, xmin, ymax, xmax] -> [xmin, ymin, width, height]
    for box in bboxes:
        xmin = int(box[1])
        ymin = int(box[0])
        w = int(box[3]) - xmin
        h = int(box[2]) - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, w, h
		
    # generate features for deepsort
    features = encoder(np.array(pil_img_obj), bboxes)

    # munge into deep sort detection objects
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, classes, features)]

    # run non-maximum suppression
    # borrowed from: https://github.com/nwojke/deep_sort/blob/master/deep_sort_app.py#L174
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    
    return detections


def parse_label_map(DEFAULT_LABEL_MAP_PATH):
    labels = {}
    for i, row in enumerate(open(DEFAULT_LABEL_MAP_PATH)):
        labels[i] = row.replace('\n','')
    return labels
