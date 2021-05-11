import time

class OpticalFlow:

    def __init__(self):
        self.track = None
        self.frame = None
        self._optiflow_list = []
        self.group_frame = None
        self.temp_list = []
        self.track_len = 10
        self.vehicle_type = None
        self.vehicle_colour = None
        self.pasttwoframes = list
        self.pasttwocentroids = []
        self.grouped_id = []
        self.time = int(time.time())

    def append_list(self):
        if self.group_frame == None:
            self.group_frame = self.frame
        if self.group_frame != self.frame:
            self._optiflow_list.append(self.temp_list)
            self.temp_list = []
            self.group_frame = self.frame
            self.pasttwoframes = self._optiflow_list[-2:]
            self.group_by_id()
            self.extract_last_two_centroids()
        if len(self._optiflow_list) > self.track_len:
            del(self._optiflow_list[0])

    def append_record(self):
        self.append_list()
        cx, cy = self._calc_centroid(self.track.to_tlbr())
        rec = [self.frame, self.track.track_id, self.vehicle_type, self.vehicle_colour, self.time, [cx, cy]]
        self.temp_list.append(rec)

    def group_by_id(self): 
        if len(self.pasttwoframes) != 1:
            group = [lists for list in self._optiflow_list for lists in list]
            unique_id = set([list[1] for list in group])
            self.grouped_id = [[list[1:] for list in group if list[1] == value] for value in unique_id]

    def extract_last_two_centroids(self):
        temp_list = [lists[-2:] for lists in self.grouped_id]
        self.pasttwocentroids = temp_list

    def _calc_centroid(self, bbox):
        xmin, ymin, width, height = self._clac_cent_coord(bbox)
        cx = xmin + (0.5 * width)
        cy = ymin + (0.5 * height)
        return cx, cy

    def _clac_cent_coord(self, bbox):
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        width = int(bbox[2]- bbox[0])
        height = int(bbox[3]-bbox[1])
        return xmin, ymin, width, height