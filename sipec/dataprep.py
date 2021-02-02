# SIPEC
# MARKUS MARKS
# DATA PREPARATION FOR DATA USED IN SIPEC PAPER
import json
import random
import sys


import os
from argparse import ArgumentParser
import skimage
import skimage.io
import numpy as np
import pickle
from glob import glob

from scipy.ndimage import center_of_mass
from tqdm import tqdm

import pandas as pd
from scipy import misc

from keras import backend as K

# from sipec.segmentation import mold_image, MouseConfig, SegModel
from sipec.dataloader import create_dataset
from sipec.poseestimation import (
    heatmaps_for_image_whole,
    bbox_mask,
    heatmap_mask,
    dilate_mask,
)

# from sipec.segmentation import SegModel, mold_image
from sipec.utils import setGPU, loadVideo

## adapted from matterport Mask_RCNN implementation
from sipec.mrcnn import utils


# adapted from mrcnn (Waleed Abdulla, (c) 2017 Matterport, Inc.)


class Dataset(utils.Dataset):
    def __init__(self, species):
        super(Dataset, self).__init__()
        self.species = species

    def load(self, dataset_dir, subset, annotations):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class(self.species, 1, self.species)

        # Train or validation dataset?

        annotations = [a for a in annotations if a["regions"]]

        non_cv_split = subset in ["train", "val"]
        if non_cv_split:
            dataset_dir = os.path.join(dataset_dir, subset)
        frames = glob(dataset_dir + "/*.png")
        frames = [el.split("/")[-1] for el in frames]

        # Add images
        for a in annotations:
            if not a["filename"] in frames and non_cv_split:
                continue
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r["shape_attributes"] for r in a["regions"]]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a["filename"])
            # TODO: fixme
            # new_img = np.zeros((1280,1920,3))
            image = skimage.io.imread(image_path).astype("uint8")
            # new_img[96:-96,:,:]=image[:,:,:]
            # image = new_img.astype('uint8')
            height, width = image.shape[:2]

            self.add_image(
                self.species,
                image_id=a["filename"],  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        #         if image_info["source"] != "mouse":
        #             return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros(
            [info["height"], info["width"], len(info["polygons"])], dtype=np.uint8
        )
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == self.species:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


# TODO: cleanup this
def prepareData(
    frames_path, annotations_path, species, fold=None, cv_folds=None, fraction=None
):
    annotations = json.load(open(annotations_path))
    if "_via_img_metadata" in annotations.keys():
        annotations = list(annotations["_via_img_metadata"].values())
    else:
        annotations = list(annotations.values())

    # annotations = annotations[:20]

    if cv_folds == 0:

        # Training dataset
        dataset_train = Dataset(species)
        dataset_train.load(frames_path, "train", annotations)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = Dataset(species)
        dataset_val.load(frames_path, "val", annotations)
        dataset_val.prepare()

    else:

        num_imgs = len(annotations)
        imgs_by_folds = int(float(num_imgs) / float(cv_folds))
        if fold == cv_folds - 1:
            annotations_val = annotations[int(fold * imgs_by_folds) :]
            annotations_train = annotations[: int(fold * imgs_by_folds)]
        else:
            annotations_val = annotations[
                int(fold * imgs_by_folds) : int((fold + 1) * imgs_by_folds)
            ]
            annotations_train = (
                annotations[: int(fold * imgs_by_folds)]
                + annotations[int((fold + 1) * imgs_by_folds) :]
            )

        if fraction:
            num_training_imgs = int(len(annotations_train) * fraction)
            annotations_train = random.sample(annotations_train, num_training_imgs)

        # Training dataset
        dataset_train = Dataset(species)
        dataset_train.load(frames_path, "all", annotations_train)
        dataset_train.prepare()

        # Validation datasetss
        dataset_val = Dataset(species)
        dataset_val.load(frames_path, "all", annotations_val)
        dataset_val.prepare()

    return dataset_train, dataset_val


def get_SIPEC_reproduction_data(name):

    if name == "primate":
        # prepare path, such that roughly 0.2 of frames are in "val" folder, rest in "train" folder
        if cv_folds == 0:
            frames_path = "/media/nexus/storage5/swissknife_data/primate/segmentation_inputs/current/annotated_frames/"
            annotations_path = "/media/nexus/storage5/swissknife_data/primate/segmentation_inputs/current/primate_segmentation.json"
        else:
            frames_path = "/media/nexus/storage5/swissknife_data/primate/segmentation_inputs/current/frames_merged/"
            annotations_path = "/media/nexus/storage5/swissknife_data/primate/segmentation_inputs/current/primate_segmentation.json"
    elif name == "mouse":
        if cv_folds == 0:
            frames_path = "/media/nexus/storage5/swissknife_data/mouse/segmentation_inputs/annotated_frames/"
            annotations_path = "/media/nexus/storage5/swissknife_data/mouse/segmentation_inputs/mouse_top_segmentation.json"
        else:
            frames_path = "/media/nexus/storage5/swissknife_data/mouse/segmentation_inputs/frames/"
            annotations_path = "/media/nexus/storage5/swissknife_data/mouse/segmentation_inputs/mouse_top_segmentation.json"
    elif name == "ineichen":
        frames_path = (
            "/media/nexus/storage5/DeepLab-sipec/ineichen_data/frames_test/"
        )
        annotations_path = (
            "/media/nexus/storage5/DeepLab-sipec/ineichen_data/new_anno_image.json"
        )
    elif name == "jin":
        frames_path = "/home/nexus/github/DeepLab-sipec/jin_data/merged/frames/"
        annotations_path = (
            "/home/nexus/github/DeepLab-sipec/jin_data/merged/annotations.json"
        )
    elif name == "jin_markus":
        frames_path = "/media/nexus/storage5/swissknife_data/primate/segmentation_inputs/markus_jin_merged/annotated_frames"
        annotations_path = "/media/nexus/storage5/swissknife_data/primate/segmentation_inputs/markus_jin_merged/merged_annotations.json"
    else:
        raise NotImplementedError("Dataset not implemented")

    return frames_path, annotations_path


# TODO: cleanup
def get_segmentation_data(
    frames_path=None,
    annotations_path=None,
    name=None,
    fold=None,
    cv_folds=None,
    fraction=None,
):
    #TODO: fix here for existing paths
    print("load data")
    # if name:
    #     frames_path, annotations_path = get_SIPEC_reproduction_data(name)

    # prepare path, such that roughly 0.2 of frames are in "val" folder, rest in "train" folder
    dataset_train, dataset_val = prepareData(
        frames_path,
        annotations_path,
        species=name,
        fold=fold,
        cv_folds=cv_folds,
        fraction=fraction,
    )
    print("data loaded")
    return dataset_train, dataset_val


### IDENTIFICATION PREPROCESSING ###


### POSE ESTIMATION PREPROCESSING ###
def generate_mouse_pose_data():
    # skip frames
    fnames = 1000
    end = 15000

    vids = ["OFT_16", "OFT_23", "OFT_35"]

    all_coms = []
    all_labels = []
    all_video = []

    for vid in vids:

        coms = "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/individual_new_smallmask/coms/OFT_16.npy"
        coms = np.load(coms)
        coms = coms[fnames:end]

        all_coms.append(coms)

        labels = "/media/nexus/storage5/swissknife_data/mouse/dlc_annotations/OFT_16_labels_dlc.npy"
        labels = np.load(labels)

        all_labels.append(labels)

        # fixme: skip here first 5 labels (corners?)
        labels = labels[:, 5:-1, :]
        labels = labels[fnames:end]

        video = loadVideo(
            "/media/nexus/storage5/swissknife_data/mouse/raw_videos/individual_data/OFT_16.mp4"
        )
        video = video[fnames:end]

        all_video.append(video)

    coms = np.vstack(all_coms)
    labels = np.vstack(all_labels)
    video = np.vstack(all_video)

    molded_frames = []
    molded_heatmaps = []
    molded_labels = []

    num_labels = 1000
    window = 64

    # fix here to choice
    indices = np.arange(0, len(video))
    indices = np.random.choice(indices, size=num_labels, replace=False)

    for idx in tqdm(indices):

        frame = video[idx]
        molded_frame = mold_image(frame, config=MouseConfig(), dimension=1024)

        img_shape = frame.shape[:2]

        # sigmas = [0.75, 1.0, 1.25, 1.5, 1.75]
        sigmas = [1.0]

        testdata = []
        for sigma in sigmas:
            data = heatmaps_for_image_whole(
                labels[idx, :, :], img_shape=img_shape, sigma=sigma
            )
            data = np.swapaxes(data, 0, 2)
            data = np.swapaxes(data, 1, 2)
            testdata.append(data)
        test = np.vstack(testdata)
        test = np.swapaxes(test, 0, 1)
        test = np.swapaxes(test, 1, 2)

        molded_maps = []
        _test = np.zeros((img_shape[0], img_shape[1], 3, test.shape[-1]))
        _test[:, :, 0, :] = test
        test = _test
        for i in range(test.shape[-1]):
            molded_maps.append(mold_image(test[:, :, :, i], dimension=1024))

        molded_maps = np.asarray(molded_maps)[:, :, :, 0]

        com = coms[idx]
        masked_img = molded_frame[
            int(com[0]) - window : int(com[0]) + window,
            int(com[1]) - window : int(com[1]) + window,
        ]

        _labels = labels[idx]
        masked_labels = [
            #         _labels[:,0]-np.abs(int(com[0])),
            #         _labels[:,1]-np.abs(int(com[1]))
            _labels[:, 0],
            _labels[:, 1],
        ]
        molded_labels.append(masked_labels)
        masked_maps = []
        for i in range(test.shape[-1]):
            molded_label = molded_maps[i, :, :]
            masked_label = molded_label[
                int(com[0]) - window : int(com[0]) + window,
                int(com[1]) - window : int(com[1]) + window,
            ]
            masked_maps.append(masked_label)

        masked_maps = np.asarray(masked_maps)
        molded_frames.append(masked_img)
        molded_heatmaps.append(masked_maps)

    molded_frames = np.asarray(molded_frames)
    molded_heatmaps = np.asarray(molded_heatmaps)

    molded_frames = np.asarray(molded_frames)
    molded_heatmaps = np.asarray(molded_heatmaps)
    molded_labels = np.asarray(molded_labels)

    X = molded_frames.astype("uint8")
    y = molded_heatmaps
    y = np.swapaxes(y, 1, -1)
    y = np.swapaxes(y, 1, 2)
    y = y.astype("float32")

    molded_labels = np.swapaxes(molded_labels, 1, 2)

    # y = y.reshape((y.shape[0], 128, 128, 5, 12))
    y = y.reshape((y.shape[0], 128, 128, 12))
    np.save(
        "/media/nexus/storage5/swissknife_data/mouse/pose_inputs/"
        "mouse_posedata_masked_X.npy",
        X,
    )
    np.save(
        "/media/nexus/storage5/swissknife_data/mouse/pose_inputs/"
        "mouse_posedata_masked_y.npy",
        y,
    )
    np.save(
        "/media/nexus/storage5/swissknife_data/mouse/pose_inputs/"
        "mouse_posedata_masked_labels.npy",
        molded_labels,
    )


def generate_primate_pose_data():
    # first pose estimate primates

    base_path = "/media/nexus/storage2/primate_data/"
    labels = pd.read_csv(base_path + "camera6_21.06.2019.csv")
    df = pd.read_csv("/media/nexus/storage2/primate_data/camera6_21.06.2019.csv")

    base_path = "/media/nexus/storage3/primate_data/"
    img_path = (
        base_path + df.iloc[0]["fulldirectoryname"] + "/" + df.iloc[0]["fullfilename"]
    )

    labels = {}
    labs = []

    for idx, row in tqdm(enumerate(df.iterrows())):
        row = row[-1]
        img_path = base_path + row["fulldirectoryname"] + "/" + row["fullfilename"]

        #     x = row['label_x']/scale_y
        #     y = new_x_max-row['label_y']/scale_y

        x = row["label_x"]
        y = 1080 - row["label_y"]

        label = row["labelname"]
        ids = row["group"]
        #     arr = np.asarray([[x,1.0] , [y,1.0] , [0.0,0.0]]).astype(np.float32)
        arr = [x, y]
        try:
            labels[row["fullfilename"]].append([arr, label, ids])
        except KeyError:
            labels[row["fullfilename"]] = []
            labels[row["fullfilename"]].append([arr, label, ids])
        labs.append(label)
    classes = list(set(labs))

    # now rearrange all the labels
    # divide by constant
    # scaling = 1.92
    scaling = 1.0

    new_labels = {}
    only_complete = False
    for label in labels:
        # find out groups from image
        groups = []
        for el in labels[label]:
            for keypoints in el:
                groups.append(el[-1])
        groups = set(groups)

        group_keypoints = []
        for group in groups:
            keypoints = {}
            for cl in classes:
                keypoints[cl] = [-1, -1]
            for _class in classes:
                for el in labels[label]:
                    if _class == el[1] and el[2] == group:
                        #                     keypoints.append(el[0]/scaling)
                        #                     keypoints.append(el[0])
                        keypoints[_class] = el[0]
            # exclude incomplete classes
            if only_complete:
                if len(keypoints) == len(classes):
                    keypoints = np.asarray(keypoints)
                    group_keypoints.append(keypoints)
                    new_labels[label] = group_keypoints
            else:
                keypoints = np.asarray(list(keypoints.values()))
                group_keypoints.append(keypoints)
                new_labels[label] = group_keypoints

    model = SegModel(species="primate")
    model.inference_config.DETECTION_MIN_CONFIDENCE = 0.99
    # indoor network
    model.set_inference(model_path="/home/nexus/mask_rcnn_primate_0119.h5")

    window = 128
    score_threshold = 0.5

    X = []
    y = []
    for key_idx, key in tqdm(enumerate(new_labels.keys())):

        keypoints = new_labels[key][0]
        frame = misc.imread(
            "/media/nexus/storage2/primate_data/" + key.split("_img")[0] + "_%T1/" + key
        )
        molded_frame = mold_image(frame, dimension=2048)
        labels = new_labels[key][0]

        testdata = []
        sigmas = [5.0]
        for sigma in sigmas:
            data = heatmaps_for_image_whole(
                labels,
                img_shape=frame.shape,
                sigma=sigma,
                # labels, img_shape=frame.shape, sigma=sigma, threshold=0.001
            )
            data = np.swapaxes(data, 0, 2)
            data = np.swapaxes(data, 1, 2)
            testdata.append(data)
        test = np.vstack(testdata)
        test = np.swapaxes(test, 0, 1)
        test = np.swapaxes(test, 1, 2)

        img_shape = molded_frame.shape
        molded_maps = []
        molded_labels = []

        for i in range(test.shape[-2]):
            molded_maps.append(mold_image(test[:, :, i, :], dimension=2048))
        molded_maps = np.asarray(molded_maps)[:, :, :, 0]

        molded_img, masks, bboxes, scores = model.detect_image(
            molded_frame, verbose=0, mold=False
        )

        ## threshold bboxes
        idx = scores > score_threshold
        bboxes = bboxes[idx]
        masks = masks[:, :, idx]

        idxs = []

        for idx in range(masks.shape[-1]):
            mask = masks[:, :, idx]
            score = heatmap_mask(molded_maps, mask)
            if score:
                idxs.append(idx)

        if len(idxs) > 0:
            box = bboxes[idxs[0]]
            mask = masks[:, :, idxs[0]]

            com = center_of_mass(mask.astype("int"))

            ### mask image
            # channels = []
            # new_mask = dilate_mask(mask, factor=30)
            # for i in range(0, 3):
            #     channels.append(molded_frame[:, :, i] * new_mask)
            #
            # channels = np.asarray(channels)
            # channels = np.moveaxis(channels, 0, -1)
            # molded_frame = channels

            masked_img = molded_frame[
                int(com[0]) - window : int(com[0]) + window,
                int(com[1]) - window : int(com[1]) + window,
            ]

            #         other_imgs = []
            #         for frame in other_frames:
            #             m_frame = frame[int(com[0])-window:int(com[0])+window,
            #                            int(com[1])-window:int(com[1])+window]
            #             other_imgs.append(m_frame)
            #         other_imgs = np.asarray(other_imgs)

            _labels = labels
            masked_labels = [
                #         _labels[:,0]-np.abs(int(com[0])),
                #         _labels[:,1]-np.abs(int(com[1]))
                _labels[:, 0],
                _labels[:, 1],
            ]
            molded_labels.append(masked_labels)

            #     plt.imshow(molded_frame)
            #     plt.scatter(com[1],com[0])

            masked_maps = []
            for i in range(test.shape[-2]):
                molded_label = molded_maps[i, :, :]
                masked_label = molded_label[
                    int(com[0]) - window : int(com[0]) + window,
                    int(com[1]) - window : int(com[1]) + window,
                ]
                masked_maps.append(masked_label)

            masked_maps = np.asarray(masked_maps)
            y.append(masked_maps)
            X.append(masked_img)

        X_new = []
        y_new = []
        for idx, el in enumerate(X):
            if el.shape == (window * 2, window * 2, 3):
                X_new.append(el)
                y_new.append(y[idx])

        np.save(
            "/media/nexus/storage5/swissknife_data/primate/pose_inputs/"
            "pose_estimation_no_threshold_no_masked_X_128.npy",
            np.asarray(X_new),
            allow_pickle=True,
        )
        np.save(
            "/media/nexus/storage5/swissknife_data/primate/pose_inputs/"
            "pose_estimation_no_threshold_no_masked_y_128.npy",
            np.asarray(y_new),
            allow_pickle=True,
        )
        np.save(
            "/media/nexus/storage5/swissknife_data/primate/pose_inputs/"
            "pose_estimation_no_threshold_no_masked_classes_128.npy",
            classes,
            allow_pickle=True,
        )


# primate identification
# TODO: add preperation script here
def get_primate_identification_data(scaled=True):
    basepath = "/media/nexus/storage5/swissknife_data/primate/identification_inputs"

    if scaled:
        with open(basepath + "/Identification_recurrent_scaled_X_1.pkl", "rb") as f:
            X = pickle.load(f)

        with open(basepath + "/Identification_recurrent_scaled_y_1.pkl", "rb") as f:
            y = pickle.load(f)

        with open(
            basepath + "/Identification_recurrent_scaled_videos_1.pkl", "rb"
        ) as f:
            vidlist = pickle.load(f)
    else:
        with open(basepath + "/Identification_recurrent_X.pkl", "rb") as f:
            X = pickle.load(f)

        with open(basepath + "/Identification_recurrent_y.pkl", "rb") as f:
            y = pickle.load(f)

        with open(basepath + "/Identification_recurrent_videos.pkl", "rb") as f:
            vidlist = pickle.load(f)

    return X, y, vidlist


def get_individual_mouse_data():
    x_train = np.load(
        "/media/nexus/storage5/swissknife_data/mouse/identification_inputs/"
        "mouse_identification_x_train.npy"
    )
    y_train = np.load(
        "/media/nexus/storage5/swissknife_data/mouse/identification_inputs/"
        "mouse_identification_y_train.npy"
    )
    x_test = np.load(
        "/media/nexus/storage5/swissknife_data/mouse/identification_inputs/"
        "mouse_identification_x_test.npy"
    )
    y_test = np.load(
        "/media/nexus/storage5/swissknife_data/mouse/identification_inputs/"
        "mouse_identification_y_test.npy"
    )

    return x_train, y_train, x_test, y_test


from scipy.misc import imresize
from skimage import color

# mouse individual (60 mice)
def generate_individual_mouse_data(
    animal_lim=None, cv_folds=5, fold=0, day=1, masking=False
):
    # mouse individual
    # videos = glob(
    #     "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/individual/*.npy"
    # )
    if day == 1:
        videos = [
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal1_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal2_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal3_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal4_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal5_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal6_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal7_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal8_1.npy",
        ]
        masks = [
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal1_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal2_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal3_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal4_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal5_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal6_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal7_1.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal8_1.npy",
        ]
    elif day == 2:
        videos = [
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal1_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal2_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal3_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal4_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal5_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal6_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal7_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal8_2.npy",
        ]
        masks = [
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal1_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal2_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal3_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal4_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal5_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal6_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal7_2.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal8_2.npy",
        ]
    elif day == 3:
        videos = [
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal1_3postswim.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal2_3postswim.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal3_3postswim.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal4_3postswim.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal5_3postswim.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal6_3postswim.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal7_3postswim.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masked_imgs/animal8_3postswim.npy",
        ]
        masks = [
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal1_3.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal2_3.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal3_3.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal4_3.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal5_3.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal6_3.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal7_3.npy",
            "/media/nexus/storage5/swissknife_data/mouse/inference/segmentation/identification/individuals/masks/animal8_3.npy",
        ]
    else:
        raise NotImplementedError
    trans = 1
    ###
    look_back = 5

    # TODO: match with annotation data
    # glob(base_path + "/dlc_annotations/*.npy")

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    _id = 0

    recurrent = False

    # prepare crossval
    all_idxs = list(range(1000, 14000))
    foldsize = int(len(all_idxs) / float(cv_folds))
    indices_val = all_idxs[int(fold * foldsize) : int((fold + 1) * foldsize)]
    for el in indices_val:
        all_idxs.remove(el)
    indices_train = all_idxs

    for idx, video in tqdm(enumerate(videos[:animal_lim])):
        vid = np.load(video, allow_pickle=True)
        if masking:
            vid_masks = np.load(masks[idx], allow_pickle=True)
        if trans:
            vid_new = []
            for el_idx, el in tqdm(enumerate(vid)):
                if masking:
                    mask = vid_masks[el_idx, :, :]
                    # new_mask = dilate_mask(mask, factor=10)
                    new_mask = dilate_mask(mask, factor=20)
                    channels = []
                    for i in range(0, 3):
                        channels.append(el[:, :, i] * new_mask)
                    channels = np.asarray(channels)
                    channels = np.moveaxis(channels, 0, -1)
                    el = channels

                el = color.rgb2gray(el)

                el = imresize(el, 0.5)

                vid_new.append(el)

            vid = np.asarray(vid_new)

        if recurrent:
            vid = create_dataset(vid, 10)

        X_train.append(vid[indices_train])
        label = [_id] * len(vid[indices_train])
        y_train.append(label)

        X_val.append(vid[indices_val])
        label = [_id] * len(vid[indices_val])
        y_val.append(label)
        _id += 1

    y_train = np.hstack(y_train)
    x_train = np.vstack(X_train)
    y_train = y_train.astype(int)

    y_test = np.hstack(y_val)
    x_test = np.vstack(X_val)
    y_test = y_test.astype(int)

    return x_train, y_train, x_test, y_test


parser = ArgumentParser()
parser.add_argument(
    "--operation",
    action="store",
    dest="operation",
    type=str,
    default="train_primate",
    help="standard training options for SIPEC data",
)
parser.add_argument(
    "--gpu",
    action="store",
    dest="gpu",
    type=str,
    default="0",
    help="filename of the video to be processed (has to be a segmented one)",
)


def main():
    args = parser.parse_args()
    operation = args.operation
    gpu_name = args.gpu

    setGPU(K, gpu_name)

    if operation == "pose_primate":
        generate_primate_pose_data()
    if operation == "pose_mouse":
        generate_mouse_pose_data()
    if operation == "identification_mouse_individual":
        generate_individual_mouse_data(animal_lim=20)


if __name__ == "__main__":
    main()
