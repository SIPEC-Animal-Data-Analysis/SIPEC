# SIPEC
# MARKUS MARKS
# POSE ESTIMATION
import json
from datetime import datetime

## adapted from matterport Mask_RCNN implementation
from SwissKnife.mrcnn import utils

import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from skimage.filters import gaussian
from scipy.ndimage.morphology import binary_dilation
import cv2

from keras import backend as K

import os
import tensorflow as tf

import imgaug.augmenters as iaa

from SwissKnife.architectures import posenet_primate, posenet_mouse

from SwissKnife.utils import (
    setGPU,
    load_config,
    set_random_seed,
    check_directory,
    get_tensorbaord_callback,
)
from SwissKnife.augmentations import primate_identification

import numpy as np
import keras
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


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
            segmaps = self.masks[ID].astype("bool")

            maps_augment = []
            seq_det = self.augmentation.to_deterministic()

            for segmap_idx in range(segmaps.shape[-1]):
                segmap = segmaps[:, :, segmap_idx]
                segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
                image_aug, segmap_aug = seq_det(image=image, segmentation_maps=segmap)
                #                 print(segmap_aug.shape)
                segmap_aug = (
                    segmap_aug.draw()[0][:, :, 0].astype("bool").astype("uint8")
                )
                maps_augment.append(segmap_aug)
            # Store sample
            X.append(image_aug)

            # Store class
            y.append(maps_augment)

        maps = np.asarray(y)
        maps = np.swapaxes(maps, 1, 2)
        maps = np.swapaxes(maps, 2, 3)
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
        heatmap = gaussian(heatmap, sigma=2)
        val = max(heatmap.flatten())
        if val > threshold:
            _coord = np.where(heatmap == val)
            coords.append([_coord[0][0], _coord[1][0]])
        else:
            coords.append([0, 0])

    return np.asarray(coords)


def calculate_rmse(pred, true):
    rmses = []

    for idx, el in enumerate(pred):
        point_gt = pred[idx]
        point_pred = true[idx]
        dist = (point_gt[0] - point_pred[0]) ** 2 + (point_gt[1] - point_pred[1]) ** 2
        rmses.append(np.sqrt(dist))
    return np.array(rmses).mean()


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

        # threshold

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
    def __init__(self, writer=None):
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
            heatmaps[:20, :20, :] = 0
            heatmaps = heatmaps[0, :, :, :]
            coords_gt = heatmap_to_scatter(y_val[idx])[:-1]
            coords_predict = heatmap_to_scatter(heatmaps)[:-1]
            rmses.append(calculate_rmse(coords_predict, coords_gt))
        rmses = np.asarray(rmses)
        rmse_mean = rmses.mean()

        self._data.append(
            {"rmse": rmse_mean,}
        )
        print("rmse ::: ", rmse_mean)
        if self.writer is not None:
            rmse_summary = tf.compat.v1.summary.scalar(
                "rmses", tf.convert_to_tensor(rmse_mean)
            )
            all_summary = tf.compat.v1.summary.merge_all()
            self.writer.add_summary(K.eval(all_summary), batch)
            self.writer.flush()
        return

    def get_data(self):
        return self._data


def custom_binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    #
    # index = tf.where(tf.not_equal(labels, tf.constant(-1, dtype=tf.float32)))
    # logits = tf.gather(logits, index)
    # labels = tf.gather(labels, index)
    # entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
    # print("yo")
    # actives = K.evy_pred

    return K.mean(
        K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1
    )


def train_on_data(species, config, results_sink, percentage):
    global posenet
    if species == "primate":
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

        gauss_thresh = 0.05
        y[y > gauss_thresh] = 1
        y[y <= gauss_thresh] = 0
        X = X.astype("uint8")

        split = 25
        x_train = X[split:]
        y_train = y[split:]
        x_test = X[:split]
        y_test = y[:split]

    if species == "mouse":
        X = np.load(
            "/media/nexus/storage5/swissknife_data/mouse/pose_inputs/"
            "mouse_posedata_masked_X.npy"
        )

        y = np.load(
            "/media/nexus/storage5/swissknife_data/mouse/pose_inputs/"
            "mouse_posedata_masked_y.npy"
        )

        new_X = []
        for el in X:
            new_X.append(cv2.cvtColor(el, cv2.COLOR_GRAY2RGB).astype("uint8"))
        X = np.asarray(new_X)
        # y = y.astype("uint8")
        y = y[:, :, :, :]

        gauss_thresh = 0.5
        y[y > gauss_thresh] = 1
        y[y <= gauss_thresh] = 0

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

    num_classes = y.shape

    img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    input_shape = (img_rows, img_cols, 3)

    adam = keras.optimizers.Adam(lr=0.0001)

    if species == "primate":
        posenet = posenet_primate(input_shape, num_classes=14)
    if species == "mouse":
        posenet = posenet_mouse(input_shape, num_classes=12)
    posenet.compile(
        loss=custom_binary_crossentropy,
        optimizer=adam,
        metrics=["binary_crossentropy"],
    )

    sometimes = lambda aug: iaa.Sometimes(0.8, aug)

    augmentation_image = iaa.Sequential(
        [
            sometimes(iaa.CoarseDropout(p=0.2, size_percent=0.8, per_channel=False)),
            sometimes(iaa.CoarseDropout(p=0.05, size_percent=0.25, per_channel=False)),
            sometimes(iaa.GaussianBlur(sigma=(0, 1.0))),
        ],
        random_order=True,
    )

    #### primate
    batch_size = 1
    epochs = 20

    # mouse
    batch_size = 8
    epochs = 35

    logdir = os.path.join("./logs/posenet/", datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.compat.v1.summary.FileWriter(logdir + "/metrics")
    # file_writer.set_as_desfault()
    tf_callback = get_tensorbaord_callback(logdir)

    my_metrics = Metrics(writer=file_writer)
    my_metrics.setModel(posenet)

    augmentation_image = primate_identification(level=1)

    training_generator = DataGenerator(
        x_train, y_train, augmentation=augmentation_image, batch_size=1
    )

    dense_history_1 = posenet.fit_generator(
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[my_metrics, tf_callback],
        shuffle=True,
        generator=training_generator,
        use_multiprocessing=True,
        workers=40,
    )

    K.set_value(posenet.optimizer.lr, 0.0001)
    epochs = 50

    dense_history_2 = posenet.fit_generator(
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[my_metrics, tf_callback],
        shuffle=True,
        generator=training_generator,
        use_multiprocessing=True,
        workers=40,
    )

    K.set_value(posenet.optimizer.lr, 0.00001)
    epochs = 100

    dense_history_2 = posenet.fit_generator(
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[my_metrics, tf_callback],
        shuffle=True,
        generator=training_generator,
        use_multiprocessing=True,
        workers=40,
    )

    posenet.save(results_sink + "posenetNet" + ".h5")

    # skip tail for now
    rmses = []
    for idx, test_img in tqdm(enumerate(x_test)):
        heatmaps = posenet.predict(np.expand_dims(test_img, axis=0))
        # set upper left to 0
        heatmaps[:20, :20, :] = 0
        heatmaps = heatmaps[0, :, :, :]
        coords_gt = heatmap_to_scatter(y[idx])[:-1]
        coords_predict = heatmap_to_scatter(heatmaps)[:-1]
        rmses.append(calculate_rmse(coords_predict, coords_gt))
    rmses = np.asarray(rmses)
    # overall rmse
    print("overall result RMSE")
    print(str(rmses))
    print("\n")
    print(str(np.mean(rmses)))
    res = np.mean(rmses)
    np.save(results_sink + "results" + ".npy", res)


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
    default=None,
    help="fraction to use for training",
)


def main():
    args = parser.parse_args()
    operation = args.operation
    gpu_name = args.gpu
    fraction = args.fraction

    setGPU(K, gpu_name)

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
    check_directory(results_sink)
    with open(results_sink + "config.json", "w") as f:
        json.dump(config, f)
    f.close()

    if operation == "train_primate":
        train_on_data(species="primate", config=config, results_sink=results_sink)
    if operation == "train_mouse":
        train_on_data(
            species="mouse",
            config=config,
            results_sink=results_sink,
            percentage=fraction,
        )
