# SIPEC
# JIN QIUHAN, MARKUS MARKS
# SCRIPT FOR SMOOTHING SEGMENTATION RESULTS OVER TIME
import pickle
import time

import cv2
import numpy as np
from shapely.geometry import Polygon
from tensorflow.keras import backend as K

from SwissKnife.segmentation import SegModel
from SwissKnife.utils import masks_to_coms, setGPU


class MaskMatcher:
    """MaskMatcher class to match segmentation masks of temporally adjacent frames.

    This class implements extensions of greedy masks matching algorithms.
    """

    def __init__(self, max_ids=4):
        self.ids = None
        self.max_ids = max_ids
        self.hist_length = 0
        pass

    def bbox_to_polygon(self, bbox):
        y1, x1, y2, x2 = bbox
        return Polygon([(y1, x1), (y1, x2), (y2, x2), (y2, x1)])

    def iou(self, bbox1, bbox2):
        """Calculate the intersection over union (IoU) for two bounding boxes.

        Parameters
        ----------
        bbox1 : bounding box
            First bounding box for IoU calculation.

        bbox2 : bounding box
            Second bounding box for IoU calculation.

        Returns
        -------
        float
            IoU value.
        """
        p1 = self.bbox_to_polygon(bbox1)
        p2 = self.bbox_to_polygon(bbox2)
        a = p1.intersection(p2).area
        c = p1.area + p2.area
        if c == 0:
            return 0
        return a / c

    def bbox_match(self, bboxes_cur, bboxes_pre):
        """Bounding box greedy matching algorithm.

        Find all current bboxes which are identical to any in the previous frame.
        Condition: a bbox_cur intersects with one and only bbox_pre &&
        it doesn't intersect with any other bbox_cur.

        Parameters
        ----------
        bboxes_cur : bounding boxes
            Bounding boxes from current frame.

        bboxes_pre : bounding boxes
            Bounding boxes from previous frame.

        Returns
        -------
        dict
            Dictionary containing the mapping of indices from matching bounding boxes of current frame to previous frame.
        """
        mapping_dict = {}
        for idx_cur, bbox_cur in enumerate(bboxes_cur):
            mapping_dict[idx_cur] = [0, 0]
            for idx_pre, bbox_pre in enumerate(bboxes_pre):
                overlap = self.iou(bbox_cur, bbox_pre)
                # TODO: fix ugly hack, just make nxn matrix
                if overlap > mapping_dict[idx_cur][0]:
                    mapping_vals = np.array(list(mapping_dict.values()))
                    if int(idx_pre) in mapping_vals[:, 1].astype("int"):
                        idx = np.where(mapping_vals[:, 1] == int(idx_pre))[0][0]
                        if overlap > mapping_vals[idx][0]:
                            mapping_dict[idx_cur] = [overlap, int(idx_pre)]
                            if not idx_cur == idx:
                                mapping_dict[idx] = [0, 0]
                        else:
                            continue
                    else:
                        mapping_dict[idx_cur] = [overlap, int(idx_pre)]
        return mapping_dict

    def assign_ids(
        self, frame_cur, _model, _classes, _threshold, bboxes_cur, bboxes_pre, ids_pre
    ):
        """Identify all bboxes in the current frame.
        Call IdNet for unmatched bboxes.
        In case of repeated ids, label the less probable ones as "Wrong".
        Output: a list of tuples, each tuple is (ID, confidence level).
        """
        # ids_list[0]: --> list
        # [('Paul', 0.85876614), ('Alan', 0.8630158)]
        ids_list_pre = ids_pre.copy()
        ids_list_cur = [0] * len(bboxes_cur)
        mapping = self.bbox_match(bboxes_cur, bboxes_pre)
        # matched bboxes found
        if len(mapping) > 0:
            for pair in mapping:
                ids_list_cur[pair[0]] = ids_list_pre[pair[1]]
        return ids_list_cur

    def euclidean_dist(self, com1, com2):
        """TODO: Fill in description"""
        res = 0
        for i,_ in enumerate(com1):
            el1 = com1[i]
            el2 = com2[i]
            res += np.square(el1 - el2)
        return np.sqrt(res)

    def match_masks(self, bboxes_cur, bboxes_pre):
        """TODO: Fill in description"""

        mapping_matrix = np.zeros((self.max_ids, self.max_ids))
        for idx_cur, el_cur in enumerate(bboxes_cur):
            for prev_boxes_idxs, prev_boxes_els in enumerate(reversed(bboxes_pre)):
                for idx_prev, el_prev in enumerate(prev_boxes_els["boxes"]):
                    overlap = self.iou(el_cur, el_prev)
                    mapping_matrix[idx_prev][idx_cur] += overlap**2 / (
                        prev_boxes_idxs**1.5 + 1
                    )
        for i in range(self.max_ids):
            mx = np.max(mapping_matrix)
            coords = np.where(mapping_matrix == mx)
            if mx == 0:
                # two masks merge and one leftover ids arises, pick only max
                for col in range(len(mapping_matrix)):
                    if np.sum(mapping_matrix[:, col]) == 0:
                        for row in range(len(mapping_matrix)):
                            if np.sum(mapping_matrix[row, :]) == 0:
                                mapping_matrix[row, col] = -1
            else:
                # set rows and cols away from the max value to 0
                for row in range(len(mapping_matrix)):
                    if not row == coords[0][0]:
                        mapping_matrix[row, coords[1][0]] = 0
                for col in range(len(mapping_matrix)):
                    if not col == coords[1][0]:
                        mapping_matrix[coords[0][0], col] = 0
                mapping_matrix[coords[0][0], coords[1][0]] = -mapping_matrix[
                    coords[0][0], coords[1][0]
                ]
        mapping_matrix = -mapping_matrix

        mapp = {}
        for i in range(len(mapping_matrix)):
            col = mapping_matrix[:, i]
            ma = np.max(col)
            if ma == 0:
                continue
            mapp[i] = [
                mapping_matrix[np.argmax(col), i] / np.max(mapping_matrix),
                np.argmax(col),
            ]
        return mapp

    def match_ids(self, mapping, nums):
        """TODO: Fill in description"""
        if self.ids is None or len(mapping) == 0:
            self.ids = np.zeros((self.max_ids,)).astype("int")
        else:
            new_ids = np.zeros((self.max_ids,)).astype("int")
            is_zero = self.ids == new_ids
            left_nums = set(range(1, nums + 1))
            if is_zero.all():
                for num_idx, num in enumerate(list(left_nums)):
                    new_ids[num_idx] = num
            else:
                # TODO: fix, hacky
                for map_id in mapping.keys():
                    # TODO: catch somehow cleaner
                    try:
                        new_id = self.ids[mapping[map_id][1]]
                        if new_id == 0:
                            new_id = list(left_nums)[0]
                        new_ids[map_id] = new_id
                        if new_id in left_nums:
                            left_nums.remove(new_id)
                    except IndexError:
                        continue
                if len(left_nums) > 0:
                    for num in list(left_nums):
                        for el_idx, el in enumerate(new_ids):
                            if el == 0:
                                new_ids[el_idx] = num
                                break
            self.ids = new_ids
        return self.ids

    def map(self, mapping, arr):
        """TODO: Fill in description"""

        inverse_mapping = {}
        for map in mapping:
            inverse_mapping[mapping[map][1]] = map

        new_arr = {}
        leftovers = list(range(0, len(arr)))
        leftovers_new = list(
            range(0, max(max(inverse_mapping.keys()) + 1, max(leftovers)))
        )
        for el,_ in inverse_mapping.items():
            try:
                new_arr[el] = arr[inverse_mapping[el]]
                leftovers.remove(inverse_mapping[el])
                leftovers_new.remove(el)
            except IndexError:
                try:
                    new_arr[el] = np.zeros(arr[0].shape)
                    leftovers_new.remove(el)
                except AttributeError:
                    new_arr[el] = 0
                    leftovers_new.remove(el)

        for el in leftovers:
            try:
                new_arr[leftovers_new[0]] = arr[el]
                leftovers_new.remove(leftovers_new[0])
            except IndexError:
                new_arr[len(new_arr)] = arr[el]

        for el in leftovers_new:
            try:
                new_arr[el] = np.zeros(arr[0].shape)
            except AttributeError:
                new_arr[el] = 0

        new_arr_list = []
        for i in range(0, max(new_arr.keys()) + 1):
            new_arr_list.append(new_arr[i])

        return new_arr_list


def main():
    """TODO: Fill in description"""
    start_time = time.process_time()

    # Config 1: GPU config
    setGPU(K, "0")
    # Config 2: inference config
    # config = InferenceConfigPrimate()

    species = "mouse"
    SegNet = SegModel(species=species)
    SegNet.set_inference(
        model_path="/media/nexus/storage4/swissknife_results/segmentation/mouse_/mouse20200519T1254/mask_rcnn_mouse_0050.h5"
    )
    # IdNet = load_model("./IDnet_20180124T115800-20180124T122800b_%T1_recognitionNet.h5")
    maskmatcher = MaskMatcher()

    print("Load models:", time.process_time() - start_time, "seconds")

    # Load test video and process each frame as it is loaded
    start_time = time.process_time()

    cap = cv2.VideoCapture("test.avi")

    # Initiation
    smooth_ids_list = []
    single_coms = {
        "Paul": [(0, 0)],
        "Max": [(0, 0)],
        "Charles": [(0, 0)],
        "Alan": [(0, 0)],
    }
    count = 0
    success, frame_pre = cap.read()

    # Process first frame
    img_pre, masks_pre, bboxes_pre = SegNet.detect_image(frame_pre, verbose=0)
    if len(masks_pre.shape) == 2:
        masks_pre = np.expand_dims(masks_pre, axis=-1)
    #coms_pre = masks_to_coms(masks_pre)

    print("First frame:", time.process_time() - start_time, "seconds")
    start_time = time.process_time()

    # Process from second frame
    # assign_ids for smooth identification
    while success:
        success, frame_cur = cap.read()
        count += 1
        print("Processing frame", count)
        img_cur, masks_cur, bboxes_cur = SegNet.detect_image(frame_cur, verbose=0)
        if len(masks_cur.shape) == 2:
            masks_cur = np.expand_dims(masks_cur, axis=-1)
        coms_cur = masks_to_coms(masks_cur)
        # ids_cur = maskmatcher.assign_ids(
        #     img_cur, IdNet, classes_invert, threshold, bboxes_cur, bboxes_pre, ids_pre
        # )
        ids_cur = maskmatcher.match_masks(bboxes_cur, bboxes_pre)
        smooth_ids_list.append(ids_cur)
        # put coms and ids in dict
        for _idx, _id in enumerate(ids_cur):
            if _id[0] in single_coms.keys():
                single_coms[_id[0]].append(coms_cur[_idx])
        # bboxes and ids are looped
        bboxes_pre = bboxes_cur.copy()
        #ids_pre = ids_cur.copy()
        if count == 300:  # test 10 sec
            break

    cap.release()
    print(len(smooth_ids_list))
    print(smooth_ids_list[0])
    print(smooth_ids_list[1])
    with open("./smooth_ids_180124.pkl", "wb") as f:
        pickle.dump(smooth_ids_list, f)

    print(time.process_time() - start_time, "seconds")


if __name__ == "__main__":
    main()
