"""
SIPEC
MARKUS MARKS
DATA PREPARATION FOR DATA USED IN SIPEC PAPER
"""
import json
import os
import pickle
import random
from argparse import ArgumentParser
from glob import glob

import numpy as np
import pandas as pd
import skimage
import skimage.io
from scipy import misc
from scipy.ndimage import center_of_mass
from skimage import color
from sklearn.externals._pilutil import imresize
from tensorflow.keras import backend as K
from tqdm import tqdm

from SwissKnife.dataloader import create_dataset
from SwissKnife.mrcnn import utils

# from SwissKnife.segmentation import SegModel
from SwissKnife.utils import (
    dilate_mask,
    heatmap_mask,
    heatmaps_for_image_whole,
    heatmaps_for_images,
    heatmaps_to_locs,
    loadVideo,
    setGPU,
)


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

            attributes = [r["region_attributes"] for r in a["regions"]]

            self.add_image(
                self.species,
                image_id=a["filename"],  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                annotations=attributes,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
         masks: A bool array of shape [height, width, instance count] with
             one mask per instance.
         class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        # image_info = self.image_info[image_id]
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
            try:
                rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
                mask[rr, cc, i] = 1
            except (IndexError, KeyError):
                print("ERROR skipping image {}".format(image_id))
                pass

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == self.species:
            return info["path"]
        super(self.__class__, self).image_reference(image_id)


# TODO: remove unused code
def merge_annotations(path1, path2, save_path):
    with open(path1) as fh:
        annotations_1 = json.load(fh)
    with open(path2) as fh:
        annotations_2 = json.load(fh)
    annotations_1["_via_img_metadata"].update(annotations_2["_via_img_metadata"])
    with open(save_path, "w") as f:
        json.dump(annotations_1, f)


# TODO: cleanup this
def prepareData(
    frames_path,
    annotations_path,
    species,
    fold=None,
    cv_folds=None,
    fraction=None,
    prepare=True,
):
    annotations = json.load(open(annotations_path))
    if "_via_img_metadata" in annotations.keys():
        annotations = list(annotations["_via_img_metadata"].values())
    else:
        annotations = list(annotations.values())

    # annotations = annotations[:20]
    # TODO: make one/cv_fold
    if cv_folds == 0:
        #
        # Training dataset
        dataset_train = Dataset(species)
        dataset_train.load(frames_path, "train", annotations)
        if prepare:
            dataset_train.prepare()

        # Validation dataset
        dataset_val = Dataset(species)
        dataset_val.load(frames_path, "val", annotations)
        if prepare:
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

        # Validation datasetss
        dataset_val = Dataset(species)
        dataset_val.load(frames_path, "all", annotations_val)

        if prepare:
            dataset_train.prepare()
            dataset_val.prepare()

    return dataset_train, dataset_val


# TODO: remove unused code
def get_SIPEC_reproduction_data(name, cv_folds=0):

    # TODO: Remove hardcoded paths
    if name == "mouse_merged":
        if cv_folds == 0:
            frames_path = "/media/nexus/storage5/swissknife_data/mouse/segmentation_inputs_merged/frames/"
            annotations_path = "/media/nexus/storage5/swissknife_data/mouse/segmentation_inputs_merged/merged_annotations.json"
    elif name == "primate":
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
            "/media/nexus/storage5/DeepLab-SwissKnife/ineichen_data/frames_test/"
        )
        annotations_path = (
            "/media/nexus/storage5/DeepLab-SwissKnife/ineichen_data/new_anno_image.json"
        )
    elif name == "jin":
        frames_path = "/home/nexus/github/DeepLab-SwissKnife/jin_data/merged/frames/"
        annotations_path = (
            "/home/nexus/github/DeepLab-SwissKnife/jin_data/merged/annotations.json"
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
    base_folder=None,
    name=None,
    fold=None,
    cv_folds=None,
    fraction=None,
):
    # TODO: fix here for existing paths
    # print("load data")
    # if not name == 'mouse':
    #     frames_path, annotations_path = get_SIPEC_reproduction_data("primate", cv_folds=5)
    # print('awphnowpaho')

    if base_folder:
        dataset_train = None
        dataset_val = None
        for root, subFolders, files in os.walk(base_folder):
            for file in files:
                if "json" in file:
                    print(file)
                    print(subFolders)
                    print(root)
                    annotations_path = os.path.join(root, file)
                    if dataset_train:
                        annotations = json.load(open(annotations_path))
                        if "_via_img_metadata" in annotations.keys():
                            annotations = list(
                                annotations["_via_img_metadata"].values()
                            )
                        else:
                            annotations = list(annotations.values())

                        dataset_train.load(root + "/", "all", annotations)
                    else:
                        dataset_train, dataset_val = prepareData(
                            root + "/",
                            annotations_path,
                            species=name,
                            fold=fold,
                            cv_folds=cv_folds,
                            fraction=fraction,
                            prepare=False,
                        )

        dataset_train.prepare()
        dataset_val.prepare()
        print("done")

    else:
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


# TODO: remove unused code
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


def get_primate_pose_data():
    X = np.load(
        "/media/nexus/storage5/swissknife_data/primate/pose_inputs/"
        "pose_estimation_no_threshold_no_masked_X_128.npy",
    )
    y = np.load(
        "/media/nexus/storage5/swissknife_data/primate/pose_inputs/"
        "pose_estimation_no_threshold_no_masked_y_128.npy",
    )

    y = np.swapaxes(y, 1, 2)
    y = np.swapaxes(y, 2, 3)

    X = X.astype("uint8")

    y_bac = heatmaps_to_locs(y)
    img_shape = (X.shape[1], X.shape[2])
    sigmas = [16.0, 6.0, 1.0, 1.0, 0.5]
    y = heatmaps_for_images(y_bac, img_shape=img_shape, sigma=sigmas[0], threshold=None)

    split = 25
    x_train = X[split:]
    y_train = y[split:]
    x_test = X[:split]
    y_test = y[:split]

    return x_train, y_train, x_test, y_test


def get_mouse_pose_data(fraction=1.0):
    X = np.load("/home/markus/sipec_data/pose_inputs/" "mouse_posedata_masked_X.npy")

    y = np.load("/home/markus/sipec_data/pose_inputs/" "mouse_posedata_masked_y.npy")

    new_X = []
    for el in X:
        new_X.append(cv2.cvtColor(el, cv2.COLOR_GRAY2RGB).astype("uint8"))
    X = np.asarray(new_X)
    # y = y.astype("uint8")
    y_bac = y[:, :, :, :]

    ########

    y = heatmaps_to_locs(y)
    img_shape = (X.shape[1], X.shape[2])

    sigmas = [5.0, 4.0, 1.0, 1.0, 0.5]

    y = heatmaps_for_images(y, img_shape=img_shape, sigma=sigmas[0], threshold=None)

    split = 50
    x_train = X[split:]
    y_train = y[split:]
    x_test = X[:split]
    y_test = y[:split]

    num_labels = int(len(x_train) * fraction)
    indices = np.arange(0, len(x_train))
    random_idxs = np.random.choice(indices, size=num_labels, replace=False)
    x_train = x_train[random_idxs]
    y_train = y_train[random_idxs]

    return x_train, y_train, x_test, y_test


def get_mouse_pose_dlc_comparison_data(fold):
    asgrey = False

    base_path = "/media/nexus/storage5/swissknife_data/mouse/pose_estimation_comparison_data/OFT/"
    folders = os.walk(base_path + "labeled-data/")

    folders = folders.__next__()[1]

    dlc_path = (
        "/home/nexus/evaluation_results/evaluation-results/iteration-0/Blockcourse1May9-trainset"
        + str(fold)
        + "shuffle1/LabeledImages_DLC_resnet50_Blockcourse1May9shuffle1_1030000_snapshot-1030000/"
    )
    from glob import glob

    dlc_files = glob(dlc_path + "*.png")

    training_files = []
    testing_files = []
    for file in dlc_files:
        suffix = "Training"
        if "Test" in file:
            suffix = "Test"
        if suffix == "Training":
            training_files.append(file.split(suffix + "-")[1])
        else:
            testing_files.append(file.split(suffix + "-")[1])

    Xs = []
    ys = []
    for folder in folders:
        path = base_path + "labeled-data/" + folder + "/CollectedData_BCstudent1.csv"
        X, y = read_DLC_labels(
            base_path=base_path,
            label_file_path=path,
            exclude_labels=["tl", "tr", "bl", "br", "centre"],
            as_gray=asgrey,
            file_list=training_files,
        )
        Xs.append(X)
        ys.append(y)

    x_train_bac = np.concatenate(Xs)
    y_train_bac = np.concatenate(ys)

    Xs = []
    ys = []
    folders = os.walk(base_path + "labeled-data/")
    folders = folders.__next__()[1]
    for folder in folders:
        path = base_path + "labeled-data/" + folder + "/CollectedData_BCstudent1.csv"
        X, y = read_DLC_labels(
            base_path=base_path,
            label_file_path=path,
            exclude_labels=["tl", "tr", "bl", "br", "centre"],
            as_gray=asgrey,
            file_list=testing_files,
        )
        if not X.tostring() == b"":
            Xs.append(X)
            ys.append(y)

    x_test_bac = np.concatenate(Xs)
    y_test_bac = np.concatenate(ys)

    SegNet = SegModel(species="mouse")
    SegNet.inference_config.DETECTION_MIN_CONFIDENCE = 0.001
    SegNet.set_inference(
        model_path="/home/nexus/reviews/mouse_segmentation/mouse20210531T1038/mask_rcnn_mouse_0095.h5"
    )

    sigmas = [6.0, 4.0, 1.0, 1.0, 0.5]
    img_shape = (x_test_bac.shape[1], x_test_bac.shape[2])
    y_train = heatmaps_for_images(
        y_train_bac, img_shape=img_shape, sigma=sigmas[0], threshold=None
    )
    y_test = heatmaps_for_images(
        y_test_bac, img_shape=img_shape, sigma=sigmas[0], threshold=None
    )

    mask_size = 64
    x_train, y_train, _ = segment_images_and_masks(
        x_train_bac, y_train, SegNet=SegNet, mask_size=mask_size
    )
    x_test, y_test, coms_test = segment_images_and_masks(
        x_test_bac, y_test, SegNet=SegNet, mask_size=mask_size
    )

    return x_train, y_train, x_test, y_test, img_shape


def get_mouse_dlc_data():
    # base_path = '/media/nexus/storage5/swissknife_data/mouse/pose_estimation_comparison_data/OFT/'
    # path ='/media/nexus/storage5/swissknife_data/mouse/pose_estimation_comparison_data/OFT/labeled-data/1_01_A_190507114629/CollectedData_BCstudent1.csv',
    # X, y = read_DLC_labels(base_path=base_path, label_file_path=path,
    #                        exclude_labels=['tl', 'tr', 'bl', 'br', 'centre'])
    asgrey = False

    base_path = "/media/nexus/storage5/swissknife_data/mouse/pose_estimation_comparison_data/OFT/"
    folders = os.walk(base_path + "labeled-data/")

    folders = folders.__next__()[1]

    Xs = []
    ys = []
    for folder in folders:
        path = base_path + "labeled-data/" + folder + "/CollectedData_BCstudent1.csv"
        X, y = read_DLC_labels(
            base_path=base_path,
            label_file_path=path,
            exclude_labels=["tl", "tr", "bl", "br", "centre"],
            as_gray=asgrey,
        )
        Xs.append(X)
        ys.append(y)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)

    img_shape = (X.shape[1], X.shape[2])
    y = heatmaps_for_images(y, img_shape=img_shape, sigma=3, threshold=None)

    # mold images
    mold_dimension = 1024
    if asgrey:
        X = np.expand_dims(X, axis=-1)
    X = mold_video(video=X, dimension=mold_dimension)

    resize_factor = 0.25

    im_re = []
    for el in tqdm(X):
        im_re.append(imresize(el, resize_factor))
    X = np.asarray(im_re)

    out_dim = X.shape[2]

    molded_maps = []
    for el in y:
        help = np.moveaxis(el, 2, 0)
        maps = []
        for map in help:
            map = imresize(map, resize_factor)
            new_map = np.zeros((out_dim, out_dim))
            x_start = int((out_dim - map.shape[0]) / 2)
            y_start = int((out_dim - map.shape[1]) / 2)
            new_map[x_start:-x_start, y_start:-y_start] = map
            maps.append(new_map)
        maps = np.moveaxis(np.asarray(maps), 0, 2)
        molded_maps.append(maps)
    y = np.asarray(molded_maps)

    split = 4
    x_train = X[split:]
    y_train = y[split:]
    x_test = X[:split]
    y_test = y[:split]

    return x_train, y_train, x_test, y_test


def get_primate_paths():
    video_train = [
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180124T095000-20180124T103000_%T1_1.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180124T095000-20180124T103000_%T1_2.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180124T095000-20180124T103000_%T1_3.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180124T095000-20180124T103000_%T1_4.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180124T095000-20180124T103000_%T1_5.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180124T115800-20180124T122800b_%T1_1.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180124T115800-20180124T122800b_%T1_2.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180124T113800-20180124T115800_%T1_1.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180124T113800-20180124T115800_%T1_2.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180115T150759-20180115T151259_%T1_1.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180131T135402-20180131T142501_%T1_1.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180131T135402-20180131T142501_%T1_2.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180116T135000-20180116T142000_%T1_1.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180116T135000-20180116T142000_%T1_2.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180116T135000-20180116T142000_%T1_3.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180116T135000-20180116T142000_%T1_4.npy",
        "/media/nexus/storage3/idtracking/idtracking_gui/results/IDresults_20180115T150502-20180115T150902_%T1_1.npy",
    ]

    classes = {
        "Charles": 0,
        "Max": 1,
        "Paul": 2,
        "Alan": 3,
    }

    video_1 = [
        "20180131T135402-20180131T142501_%T1_1",
        "20180131T135402-20180131T142501_%T1_2",
        "20180131T135402-20180131T142501_%T1_3",
        "20180131T135402-20180131T142501_%T1_4",
    ]

    idresults_base = "/media/nexus/storage3/idtracking/idtracking_gui/results/"
    fnames_base = "/media/nexus/storage1/swissknife_data/primate/inference/segmentation_highres_multi/"
    vid_basepath = (
        "/media/nexus/storage1/swissknife_data/primate/raw_videos/2018_merge/"
    )

    return video_train, classes, idresults_base, fnames_base, vid_basepath, video_1


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
