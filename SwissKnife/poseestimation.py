# SIPEC
# MARKUS MARKS
# POSE ESTIMATION
from datetime import datetime
from sklearn.externals._pilutil import imresize
import matplotlib.pyplot as plt
import os

import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from skimage.filters import gaussian
from scipy.ndimage.morphology import binary_dilation
import cv2

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras as keras
import imgaug.augmenters as iaa

from SwissKnife.architectures import posenet as posenet_architecture
from SwissKnife.segmentation import SegModel, mold_video, mold_image
from SwissKnife.mrcnn import utils
from SwissKnife.mrcnn.utils import resize

from SwissKnife.utils import (
    setGPU,
    load_config,
    set_random_seed,
    check_directory,
    get_tensorbaord_callback,
    masks_to_coms,
    apply_all_masks,
    get_callbacks,
)

# adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        imgs,
        masks,
        augmentation,
        batch_size=32,
        dim=(32, 32, 32),
        n_channels=1,
        shuffle=True,
    ):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = np.array(range(0, len(imgs)))
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.imgs = imgs
        self.masks = masks
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image = self.imgs[ID]
            # segmaps = self.masks[ID].astype("bool")
            segmaps = self.masks[ID]

            maps_augment = []
            seq_det = self.augmentation.to_deterministic()

            # segmaps = np.moveaxis(segmaps,2,0)
            segmaps = np.expand_dims(segmaps, axis=0)

            # TODO: adjust for batch
            image_aug, aug_maps = seq_det(image=image, heatmaps=segmaps)

            # for segmap_idx in range(segmaps.shape[-1]):
            #     segmap = segmaps[:, :, segmap_idx]
            #     # segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
            #     image_aug, _ = seq_det(image=image, hea=segmap)
            #     segmap_aug, _ = seq_det(image=segmap, segmentation_maps=segmap)
            #     #                 print(segmap_aug.shape)
            #     # segmap_aug = (
            #     #     segmap_aug.draw()[0][:, :, 0].astype("bool").astype("uint8")
            #     # )
            #     maps_augment.append(segmap_aug)
            # Store sample

            X.append(image_aug)
            # Store class
            y.append(aug_maps[0])

        maps = np.asarray(y)
        # maps = np.moveaxis(maps, 1, 3)
        return np.asarray(X), np.asarray(maps)


def keypoints_in_mask(mask, keypoints):
    for point in keypoints:
        keypoint = point.astype(int)

        res = mask[keypoint[1], keypoint[0]]
        if res == False:
            return False
    return True


def heatmap_to_scatter(heatmaps, threshold=0.6e-9):
    coords = []

    for idx in range(0, heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, idx]
        # heatmap = gaussian(heatmap, sigma=2)
        val = max(heatmap.flatten())
        if val > threshold:
            _coord = np.where(heatmap == val)
            coords.append([_coord[1][0], _coord[0][0]])
        else:
            coords.append([0, 0])

    return np.asarray(coords)


def calculate_rmse(pred, true):
    """Calculate Root Mean Squared Error (RMSE)

    Calculate RMSE between predicted and ground truth landmarks for pose estimation.

    Parameters
    ----------
    pred : np.ndarray
        Coordinates of predicted landmarks of pose estimation network.
    true : np.ndarray
        Coordinates of ground truth landmarks of pose estimation network.

    Returns
    -------
    keras.model
        model
    """
    rmses = []

    for idx, el in enumerate(pred):
        point_pred = pred[idx]
        point_gt = true[idx]
        if point_gt[0] == 0:
            continue
        dist = (point_gt[0] - point_pred[0]) ** 2 + (point_gt[1] - point_pred[1]) ** 2
        rmses.append(np.sqrt(dist))
    return np.nanmean(np.array(rmses))


class rmse_metric(keras.callbacks.Callback):
    def setModel(self, model):
        self.model = model

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]

        rmses = []
        for idx, el in enumerate(X_val):
            _pred = self.model.predict(np.expand_dims(el, axis=0))
            rmse = calculate_rmse(_pred, y_val[idx])
            rmses.append(rmse)
        rmse = np.asarray(rmses).mean()

        self._data.append(
            {
                #             'val_roc': roc_auc_score(y_val, y_predict, average='macro'),
                "calculated_rmse": rmse,
            }
        )
        print("calculated_rmse ::: " + str(rmse))
        return

    def get_data(self):
        return self._data


def dilate_mask(mask, factor=20):
    new_mask = binary_dilation(mask, iterations=factor)

    return new_mask


def bbox_mask(model, img, verbose=0):
    image, window, scale, padding, crop = utils.resize_image(
        img,
        # min_dim=config.IMAGE_MIN_DIM,
        # min_scale=config.IMAGE_MIN_SCALE,
        # max_dim=config.IMAGE_MAX_DIM,
        # mode=config.IMAGE_RESIZE_MODE)
        # TODO: nicer here
        min_dim=2048,
        max_dim=2048,
        mode="square",
    )
    if verbose:
        vid_results = model.detect([image], verbose=1)
    else:
        vid_results = model.detect([image], verbose=0)
    r = vid_results[0]

    return image, r["scores"], r["rois"], r["masks"]


def heatmap_mask(maps, mask):
    ret = False
    for mold in tqdm(maps):
        a = mold * mask
        if a.sum() > 10:
            return True

    return ret


def heatmaps_for_image(labels, window=100, sigma=3):
    heatmaps = []
    for label in labels:
        heatmap = np.zeros((window, window))
        heatmap[int(label[1]), int(label[0])] = 1
        heatmap = gaussian(heatmap, sigma=sigma)
        heatmap[heatmap > 0.001] = 1
        heatmaps.append(heatmap)

    heatmaps = np.asarray(heatmaps)
    heatmaps = np.moveaxis(heatmaps, 0, 2)

    return heatmaps


def heatmaps_for_image_whole(labels, img_shape, sigma=3, threshold=None):
    heatmaps = []
    for label in labels:
        heatmap = np.zeros(img_shape)
        if label[1] > -1:
            heatmap[int(label[1]), int(label[0])] = 1
            heatmap = gaussian(heatmap, sigma=sigma)
            # threshold
            if threshold:
                heatmap[heatmap > threshold] = 1
            else:
                heatmap = heatmap / heatmap.max()
        heatmaps.append(heatmap)
    heatmaps = np.asarray(heatmaps)
    heatmaps = np.moveaxis(heatmaps, 0, 2)

    return heatmaps


class PoseModel:
    def __init__(self, species):
        self.species = species

    def set_inference(self):
        pass


class Metrics(keras.callbacks.Callback):
    def __init__(self, writer=None, unmold=None):
        super(Metrics, self).__init__()
        self.writer = writer

    def setModel(self, model):
        self.model = model

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        rmses = []

        for idx, test_img in tqdm(enumerate(X_val)):
            heatmaps = self.model.predict(np.expand_dims(test_img, axis=0))
            # set upper left to 0
            heatmaps = heatmaps[0, :, :, :]
            heatmaps[:20, :20, :] = 0
            coords_gt = heatmap_to_scatter(y_val[idx])[:-1]
            coords_predict = heatmap_to_scatter(heatmaps)[:-1]
            rmses.append(calculate_rmse(coords_predict, coords_gt))
        rmses = np.asarray(rmses)
        rmse_mean = np.nanmean(rmses)

        self._data.append(
            {
                "rmse": rmse_mean,
            }
        )
        self._data.append(rmse_mean)
        print("rmse ::: ", rmse_mean)
        if self.writer is not None:
            rmse_summary = tf.compat.v1.summary.scalar(
                "rmses", tf.convert_to_tensor(value=rmse_mean)
            )
            # rmse_summary = tf.compat.v1.summary.scalar(
            #     "rmses", tf.convert_to_tensor(self._data)
            # )
            all_summary = tf.compat.v1.summary.merge_all()
            self.writer.add_summary(K.eval(all_summary), batch)
            # self.writer.add_summary(K.eval(all_summary))
            self.writer.flush()
        return

    def get_data(self):
        return self._data


def custom_binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = tf.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
    y_true = tf.cast(y_true, y_pred.dtype)

    return K.mean(
        K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1
    )


class VIZ(keras.callbacks.Callback):
    def setModel(self, model):
        self.model = model

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]

        for id in [1, 2, 3, 4, 5]:
            fig, ax = plt.subplots()

            y_true = y_val[id : id + 1]
            y_predict = self.model.predict(X_val[id : id + 1])
            y_predict[:20, :20, :] = 0
            y_predict = y_predict[0, :, :, :]
            coords_gt = y_true[0, :, :, :]
            coords_gt = heatmap_to_scatter(coords_gt)[:-1]
            # coords_gt = y_true[0, :, :, :]
            coords_predict = heatmap_to_scatter(y_predict)[:-1]
            ax.imshow(X_val[id][:, :, 0], cmap="Greys_r")
            for map_id, map in enumerate(coords_predict):
                true = coords_gt[map_id]
                # plt.scatter(map[1], map[0], c="red")
                ax.scatter(map[1], map[0], s=50)
                # plt.scatter(true[1], true[0], c="blue")
            fig.savefig("viz_primate" + str(id) + ".png")
            # plt.show()
        return


import pandas as pd
import skimage.io


def read_DLC_labels(
    base_path, label_file_path, exclude_labels=[], as_gray=False, file_list=None
):
    frame = pd.read_csv(label_file_path, header=[1, 2])

    # on oliver's dataset exclude arena keypoints for DLC comparison
    kps = frame.columns.values[1:]
    keypoints = []
    for kp in kps:
        if kp[0] in exclude_labels:
            continue
        keypoints.append(kp[0])
    keypoints = np.unique(keypoints)
    coords = ["x", "y"]

    y = []
    X = []
    for id in range(len(frame)):
        frame_part = frame.iloc[id]
        image_path = frame_part[0].split("\\")
        image = image_path[1] + "-" + image_path[2]
        if file_list:
            if not image in file_list:
                continue

        all_pts = []
        for keypoint_id, keypoint in enumerate(keypoints):
            kps = []
            for coord_id, coord in enumerate(coords):
                kps.append(frame_part[keypoint][coord])
            all_pts.append(np.array(kps))
        y.append(np.array(all_pts))
        path = ""
        for el in image_path:
            path += el + "/"
        path = path[:-1]
        path = base_path + path
        image = skimage.io.imread(path, as_gray=as_gray).astype("uint8")
        X.append(image)
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def heatmaps_for_images(labels, img_shape, sigma=3, threshold=None):
    heatmaps = []
    for el in labels:
        maps = heatmaps_for_image_whole(
            img_shape=img_shape, labels=el, sigma=sigma, threshold=threshold
        )
        heatmaps.append(maps)
    heatmaps = np.asarray(heatmaps)

    return heatmaps.astype("float32")


def heatmaps_to_locs(y):
    locs = []
    for maps in y:
        map_locs = []
        for map_id in range(y.shape[-1]):
            map = maps[:, :, map_id]
            loc = np.where(map == map.max())
            map_locs.append([loc[1][0], loc[0][0]])
        locs.append(np.array(map_locs))

    y = np.array(locs)

    return y


def segment_images_and_masks(X, y, SegNet, asgrey=False, mask_size=64):
    ### mold images
    mold_dimension = 1024
    if asgrey:
        X = np.expand_dims(X, axis=-1)
    X = mold_video(video=X, dimension=mold_dimension, n_jobs=1)
    y = mold_video(video=y, dimension=mold_dimension, n_jobs=1)

    masked_X = []
    masked_maps = []
    meta_coms = []
    for img_idx, img in enumerate(X):
        molded_img, masks, boxes, mask_scores = SegNet.detect_image(
            img, verbose=0, mold=False
        )
        coms = masks_to_coms(masks)
        masked_imgs, masked_masks = apply_all_masks(
            masks, coms, molded_img, mask_size=64
        )
        masked_heatmaps, _ = apply_all_masks(masks, coms, y[img_idx], mask_size=64)
        masked_X.append(masked_imgs[0].astype("uint8"))
        masked_maps.append(masked_heatmaps[0].astype("float32"))
        meta_coms.append(coms)
        print("img")

    X = np.asarray(masked_X)
    y = np.asarray(masked_maps)

    return X, y, meta_coms


def vis_locs(X, y):
    plt.imshow(X)
    for i in range(y.shape[1]):
        plt.scatter(y[i, 0], y[i, 1])
    plt.show()


def vis_maps(X, y):
    plt.imshow(X[0])
    for i in range(y.shape[-1]):
        plt.imshow(y[0][:, :, i], alpha=0.1)
    plt.show()


def revert_mold(img, padding, scale, dtype="uint8"):
    unpad = img[padding[0][0] : -padding[0][1], :, :]
    rec = resize(
        unpad, (unpad.shape[0] // scale, unpad.shape[1] // scale), preserve_range=True
    ).astype(dtype)
    return rec


def evaluate_pose_estimation(x_test, y_test, save=False, remold=False):
    rmses = []
    for idx, test_img in tqdm(enumerate(x_test)):
        heatmaps = posenet.predict(np.expand_dims(test_img, axis=0))
        # set upper left to 0
        heatmaps = heatmaps[0, :, :, :]
        heatmaps[:20, :20, :] = 0
        # TODO: dont incorporate rmse if heatmap in upper left corner

        if remold:
            image, window, scale, padding, crop = mold_image(
                x_test_bac[0], dimension=1024, return_all=True
            )

            unmolded_maps = []
            for map_id in range(heatmaps.shape[-1]):
                map = heatmaps[:, :, map_id]
                a = mask_to_original_image(1024, map, coms_test[idx][0], 64)
                a = np.expand_dims(a, axis=-1)
                b = revert_mold(a, padding, scale, dtype="float32")
                unmolded_maps.append(b)
            unmolded_maps = np.array(unmolded_maps)
            unmolded_maps = np.swapaxes(unmolded_maps, 0, -1)
            unmolded_maps = unmolded_maps[0]

            coords_predict = heatmap_to_scatter(unmolded_maps)[:-1]
            coords_gt = y_test_bac[idx]
            rmses.append(calculate_rmse(coords_predict, coords_gt))
        else:
            coords_gt = heatmap_to_scatter(y_test[idx])[:-1]
            coords_predict = heatmap_to_scatter(heatmaps)[:-1]
            rmses.append(calculate_rmse(coords_predict, coords_gt))

        if save:
            pass
            # posenet.save(save)
            # np.save(results_sink + "results" + ".npy", res)

        # posenet.save('./posenet_primate_masked.h5')

    rmses = np.asarray(rmses)
    # overall rmse
    print("overall result RMSE")
    print(str(rmses))
    print("\n")
    print(str(np.nanmean(rmses)))
    res = np.nanmean(rmses)
    np.save("./poseestimation_results_new" + str(fold) + ".npy", res)


def fix_layers(network, with_backbone=True):
    for layer in network.layers:
        layer.trainable = True
        if with_backbone:
            if 'layers' in dir(layer):
                for _layer in layer.layers:
                    _layer.trainable = True
    return network


def treshold_maps(y, threshold=0.9):
    y[y > threshold] = 1
    y[y <= threshold] = 0
    return y


def train_on_data(species, config, results_sink, percentage, fold=90, save=None):
    global posenet

    remold = False
    if species == "primate":
        X = np.load(
            "/media/nexus/storage5/swissknife_data/primate/pose_inputs/"
            # "/home/markus/"
            # "pose_estimation_no_threshold_no_masked_X_128.npy",
            "pose_estimation_no_threshold_masked_X.npy"
        )
        y = np.load(
            "/media/nexus/storage5/swissknife_data/primate/pose_inputs/"
            # "/home/markus/"
            "pose_estimation_no_threshold_no_masked_y_128.npy",
        )

        y = np.swapaxes(y, 1, 2)
        y = np.swapaxes(y, 2, 3)

        # gauss_thresh = 0.925
        # y[y > gauss_thresh] = 1
        # y[y <= gauss_thresh] = 0
        X = X.astype("uint8")

        y_bac = heatmaps_to_locs(y)
        img_shape = (X.shape[1], X.shape[2])
        sigmas = [16.0, 6.0, 1.0, 1.0, 0.5]
        y = heatmaps_for_images(
            y_bac, img_shape=img_shape, sigma=sigmas[0], threshold=None
        )

        split = 25
        x_train = X[split:]
        y_train = y[split:]
        x_test = X[:split]
        y_test = y[:split]

    if species == "mouse":
        X = np.load(
            "/home/markus/sipec_data/pose_inputs/"
            # "/media/nexus/storage5/swissknife_data/mouse/pose_inputs/"
            "mouse_posedata_masked_X.npy"
        )

        y = np.load(
            "/home/markus/sipec_data/pose_inputs/"
            # "/media/nexus/storage5/swissknife_data/mouse/pose_inputs/"
            "mouse_posedata_masked_y.npy"
        )

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

        ########

        # y_std = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0))
        # y = y * (y.max() - y.min()) + y.min()
        bla = y[10, :, :, 1]
        plt.imshow(bla)
        plt.colorbar()
        plt.show()

        bla = y[10, :, :, 1]

        plt.imshow(bla)
        plt.colorbar()
        plt.show()

        split = 50
        x_train = X[split:]
        y_train = y[split:]
        x_test = X[:split]
        y_test = y[:split]

        num_labels = int(len(x_train) * percentage)
        indices = np.arange(0, len(x_train))
        random_idxs = np.random.choice(indices, size=num_labels, replace=False)
        x_train = x_train[random_idxs]
        y_train = y_train[random_idxs]

    if species == "mouse_dlc":
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
            path = (
                base_path + "labeled-data/" + folder + "/CollectedData_BCstudent1.csv"
            )
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
        # y = heatmaps_for_images(y, img_shape=img_shape, sigma=5, threshold=0.66)
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

    if species == "dlc_comparison":
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
            path = (
                base_path + "labeled-data/" + folder + "/CollectedData_BCstudent1.csv"
            )
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
            path = (
                base_path + "labeled-data/" + folder + "/CollectedData_BCstudent1.csv"
            )
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
        # SegNet.set_inference(model_path='/home/markus/sipec_data/networks/mask_rcnn_mouse_0095.h5')
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

        remold = True

    img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    input_shape = (img_rows, img_cols, 3)

    #### primate
    batch_size = 4
    epochs = 250

    # mouse
    batch_size = 4
    epochs = 100

    def run_ai_cumulative_gradient(optimizer):
        import runai.ga.keras
        optim = runai.ga.keras.optimizers.Optimizer(optimizer, steps=8)
        return optim
    #
    adam = tf.keras.optimizers.Adam(lr=0.001)

    posenet = posenet_architecture(input_shape, num_classes=12)
    posenet.compile(
        loss=["binary_crossentropy"],
        optimizer=adam,
        metrics=["mse"],
    )

    if species == "primate":
        sometimes = lambda aug: iaa.Sometimes(0.4, aug)

        often = lambda aug: iaa.Sometimes(1.0, aug)
        medium = lambda aug: iaa.Sometimes(0.4, aug)
        rare = lambda aug: iaa.Sometimes(0.4, aug)
        augmentation_image = iaa.Sequential(
            [
                often(
                    iaa.Affine(
                        scale=(
                            0.6,
                            1.4,
                        ),  # scale images to 80-120% of their size, individually per axis
                        #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-40, 40),  # rotate by -45 to +45 degrees
                    )
                ),
                iaa.Fliplr(0.5, name="Flipper"),
                sometimes(
                    iaa.CoarseDropout(p=0.2, size_percent=0.5, per_channel=False)
                ),
                sometimes(iaa.GaussianBlur(sigma=(0, 1.0))),
                sometimes(
                    iaa.CoarseDropout(p=0.2, size_percent=0.8, per_channel=False)
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.05, size_percent=0.25, per_channel=False)
                ),
            ],
            random_order=True,
        )
    else:
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        often = lambda aug: iaa.Sometimes(0.95, aug)
        medium = lambda aug: iaa.Sometimes(0.05, aug)
        rare = lambda aug: iaa.Sometimes(0.05, aug)
        augmentation_image = iaa.Sequential(
            [
                often(
                    iaa.Affine(
                        #                 scale={("x": (0.75, 1.25), "y": (0.75, 1.25))}, # scale images to 80-120% of their size, individually per axis
                        scale=(
                            0.9,
                            1.1,
                        ),  # scale images to 80-120% of their size, individually per axis
                        #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-180, 180),  # rotate by -45 to +45 degrees
                        # #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    )
                ),
                # iaa.Fliplr(0.5, name="Flipper"),
                # sometimes(iaa.CoarseDropout(p=0.2, size_percent=0.8, per_channel=False)),
                sometimes(
                    iaa.CoarseDropout(p=0.2, size_percent=0.8, per_channel=False)
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.1, size_percent=0.4, per_channel=False)
                ),
                sometimes(iaa.GaussianBlur(sigma=(0, 1.0))),
            ],
            random_order=True,
        )

    def callbacks_logging(path = './logs/'):
        logdir = os.path.join(path, datetime.now().strftime("%Y%m%d-%H%M%S"))
        tf_callback = get_tensorbaord_callback(logdir)
        return tf_callback
    tf_callback = callbacks_logging(path='./logs/posenet/')
    #
    # TODO: model checkpoint callbacks
    my_metrics = Metrics(unmold=img_shape)
    my_metrics.validation_data = (np.asarray(x_test), np.asarray(y_test))
    my_metrics.setModel(posenet)

    viz_cb = VIZ()
    viz_cb.validation_data = (np.asarray(x_test), np.asarray(y_test))
    viz_cb.setModel(posenet)

    # callbacks = [my_metrics, viz_cb]
    # CB_es, CB_lr = get_callbacks(min_lr=1e-9, factor=0.5, patience=30)
    # callbacks = [my_metrics, CB_es, CB_lr]
    # augmentation_image = primate_identification(level=1)

    # callbacks = [my_metrics, tf_callback]
    callbacks = [my_metrics, viz_cb, tf_callback]

    training_generator = DataGenerator(
        x_train, y_train, augmentation=augmentation_image, batch_size=batch_size
    )

    # training_generator.set_sgima(....)

    # epochs = [300,200,400] # primate
    epochs = [250, 200, 2000] # mouse
    #
    # epochs = [400, 100, 400]  # gpu 2 # primate, working well
    lrs = [0.00075, 0.0001, 0.00001]
    steps_per_epoch = [25,50,50]

    for epoch_id, epoch in enumerate(epochs):
        K.set_value(posenet.optimizer.lr, lrs[epoch_id])

        dense_history_1 = posenet.fit(
            training_generator,
            epochs=epoch,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            shuffle=True,
            use_multiprocessing=False,
            steps_per_epoch=steps_per_epoch[epoch_id],
            # workers=40,
        )
    print("first meta done")

    if save:
        posenet.save(results_sink + "posenetNet" + ".h5")
        # np.save(results_sink + "results" + ".npy", res)

    # evaluate_pose_estimation()
    # ...


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
parser.add_argument(
    "--fraction",
    action="store",
    dest="fraction",
    type=float,
    default=1.0,
    help="fraction to use for training",
)
parser.add_argument(
    "--annotations",
    action="store",
    dest="annotations",
    type=str,
    default=None,
    help="path for annotations from VGG annotator",
)
parser.add_argument(
    "--frames",
    action="store",
    dest="frames",
    type=str,
    default=None,
    help="path to folder with annotated frames",
)
parser.add_argument(
    "--fold",
    action="store",
    dest="fold",
    type=int,
    default=None,
    help="fold for crossvalidation",
)


def main():
    args = parser.parse_args()
    operation = args.operation
    gpu_name = args.gpu
    fraction = args.fraction
    annotations = args.annotations
    frames = args.frames
    fold = args.fold

    setGPU(gpu_name)

    config_name = "poseestimation_config"
    config = load_config("../configs/poseestimation/" + config_name)
    set_random_seed(config["random_seed"])

    results_sink = (
        "/media/nexus/storage4/swissknife_results/poseestimation/"
        + config["experiment_name"]
        + "_"
        + str(fraction)
        + "_"
        + datetime.now().strftime("%Y-%m-%d-%H_%M")
        + "/"
    )
    # check_directory(results_sink)
    # with open(results_sink + "config.json", "w") as f:
    #     json.dump(config, f)
    # f.close()

    train_on_data(
        species=operation,
        config=config,
        results_sink=results_sink,
        percentage=fraction,
        fold=fold,
    )


if __name__ == "__main__":
    main()
