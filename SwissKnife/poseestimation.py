"""
SIPEC
MARKUS MARKS
POSE ESTIMATION
"""
import json
import os
from argparse import ArgumentParser
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import backend as K
from tqdm import tqdm

from SwissKnife.architectures import posenet as posenet_architecture
from SwissKnife.augmentations import mouse_poseestimation, primate_poseestimation
from SwissKnife.dataprep import (
    get_mouse_dlc_data,
    get_mouse_pose_data,
    get_mouse_pose_dlc_comparison_data,
    get_primate_pose_data,
)
from SwissKnife.mrcnn.utils import resize
from SwissKnife.segmentation import SegModel, mold_image, mold_video
from SwissKnife.utils import (
    apply_all_masks,
    callbacks_tf_logging,
    heatmap_to_scatter,
    heatmaps_for_images,
    load_config,
    mask_to_original_image,
    masks_to_coms,
    set_random_seed,
    setGPU,
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
        for _, ID in enumerate(list_IDs_temp):
            image = self.imgs[ID]
            # segmaps = self.masks[ID].astype("bool")
            segmaps = self.masks[ID]

            # maps_augment = []
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

    for idx, _ in enumerate(pred):
        point_pred = pred[idx]
        point_gt = true[idx]
        if point_gt[0] == 0:
            continue
        dist = (point_gt[0] - point_pred[0]) ** 2 + (point_gt[1] - point_pred[1]) ** 2
        rmses.append(np.sqrt(dist))
    return np.nanmean(np.array(rmses))


# TODO: remove unused code
class rmse_metric(keras.callbacks.Callback):
    """TODO: Fill in description"""
    def setModel(self, model):
        """TODO: Fill in description"""
        self.model = model

    def on_train_begin(self, logs=None):
        """TODO: Fill in description"""
        self._data = []

    def on_epoch_end(self, batch, logs=None):
        """TODO: Fill in description"""
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
        """TODO: Fill in description"""
        return self._data


class Metrics(keras.callbacks.Callback):
    """TODO: Fill in description"""
    def __init__(self, writer=None, unmold=None):
        """TODO: Fill in description"""
        super(Metrics, self).__init__()
        self.writer = writer

    def setModel(self, model):
        """TODO: Fill in description"""
        self.model = model

    def on_train_begin(self, logs=None):
        """TODO: Fill in description"""
        self._data = []

    def on_epoch_end(self, batch, logs=None):
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
            #rmse_summary = tf.compat.v1.summary.scalar(
            #    "rmses", tf.convert_to_tensor(value=rmse_mean)
            #)
            # rmse_summary = tf.compat.v1.summary.scalar(
            #     "rmses", tf.convert_to_tensor(self._data)
            # )
            all_summary = tf.compat.v1.summary.merge_all()
            self.writer.add_summary(K.eval(all_summary), batch)
            # self.writer.add_summary(K.eval(all_summary))
            self.writer.flush()
        return

    def get_data(self):
        """TODO: Fill in description"""
        return self._data


def custom_binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    """TODO: Fill in description"""
    y_pred = tf.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
    y_true = tf.cast(y_true, y_pred.dtype)

    return K.mean(
        K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1
    )


class callbacks_viz_poseestimation(keras.callbacks.Callback):
    """TODO: Fill in description"""
    def setModel(self, model):
        """TODO: Fill in description"""
        self.model = model

    def on_train_begin(self, logs=None):
        """TODO: Fill in description"""
        self._data = []

    def on_epoch_end(self, batch, logs=None):
        """TODO: Fill in description"""
        X_val, y_val = self.validation_data[0], self.validation_data[1]

        for id in range(1, 3):
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
                #true = coords_gt[map_id]
                # plt.scatter(map[1], map[0], c="red")
                ax.scatter(map[1], map[0], s=50)
                # plt.scatter(true[1], true[0], c="blue")
            #
            # print("plotted")
            fig.savefig("viz_primate" + str(id) + ".png")
            # plt.show()
        return


def read_DLC_data(dlc_path, folder, label_file_path, exclude_labels=[], as_gray=False):
    """TODO: Fill in description"""
    frame = pd.read_csv(label_file_path, header=[1, 2])

    # on oliver's dataset exclude arena keypoints for DLC comparison
    kps = frame.columns.values
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
        if "\\" in frame_part[0]:
            image_path = frame_part[0].split("\\")
        elif "/" in frame_part[0]:
            image_path = frame_part[0].split("/")
        else:
            raise ValueError
        # image = image_path[1] + "-" + image_path[2]
        # if file_list:
        #     if not image in file_list:
        #         continue

        all_pts = []
        for _, keypoint in enumerate(keypoints):
            kps = []
            for _, coord in enumerate(coords):
                try:
                    kps.append(frame_part[keypoint][coord])
                except KeyError:
                    kps.append(np.nan)
            all_pts.append(np.array(kps))
        y.append(np.array(all_pts))
        # path = ""
        # for el in image_path:
        #     path += el + "/"
        # path = path[:-1]
        # path = base_path + path
        path = dlc_path + folder + "/" + image_path[2]
        image = skimage.io.imread(path, as_gray=as_gray).astype("uint8")
        X.append(image)
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def read_dlc_labels_from_folder(dlc_path, exclude_labels=[]):
    """TODO: Fill in description"""
    folders = os.walk(dlc_path)
    folders = folders.__next__()[1]
    asgrey = False

    Xs = []
    ys = []
    for folder in folders:
        path = dlc_path + folder + "/"
        csv_file = glob(path + "*.csv")
        X, y = read_DLC_data(
            dlc_path,
            folder,
            label_file_path=csv_file[0],
            exclude_labels=exclude_labels,
            as_gray=asgrey,
        )
        Xs.append(X)
        ys.append(y)
    Xs = np.concatenate(Xs)
    ys = np.concatenate(ys)

    return Xs, ys


def segment_images_and_masks(X, y, SegNet, asgrey=False, mask_size=64):
    """mold images"""
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


# TODO: remove unused code
def vis_locs(X, y):
    """TODO: Fill in description"""
    plt.imshow(X)
    for i in range(y.shape[1]):
        plt.scatter(y[i, 0], y[i, 1])
    plt.show()


# TODO: remove unused code
def vis_maps(X, y):
    """TODO: Fill in description"""
    plt.imshow(X[0])
    for i in range(y.shape[-1]):
        plt.imshow(y[0][:, :, i], alpha=0.1)
    plt.show()


def revert_mold(img, padding, scale, dtype="uint8"):
    """TODO: Fill in description"""
    unpad = img[padding[0][0] : -padding[0][1], :, :]
    rec = resize(
        unpad, (unpad.shape[0] // scale, unpad.shape[1] // scale), preserve_range=True
    ).astype(dtype)
    return rec


def evaluate_pose_estimation(
    x_test,
    y_test,
    posenet,
    remold=False,
    y_test_orig=None,
    x_test_orig=None,
    coms_test=None,
):
    """TODO: Fill in description"""
    rmses = []
    for idx, test_img in tqdm(enumerate(x_test)):
        heatmaps = posenet.predict(np.expand_dims(test_img, axis=0))
        # set upper left to 0
        heatmaps = heatmaps[0, :, :, :]
        heatmaps[:20, :20, :] = 0
        # TODO: dont incorporate rmse if heatmap in upper left corner

        if remold:
            image, window, scale, padding, crop = mold_image(
                x_test_orig[0], dimension=1024, return_all=True
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
            coords_gt = heatmap_to_scatter(y_test_orig[idx])[:-1]
            rmses.append(calculate_rmse(coords_predict, coords_gt))
        else:
            coords_gt = heatmap_to_scatter(y_test[idx])[:-1]
            coords_predict = heatmap_to_scatter(heatmaps)[:-1]
            rmses.append(calculate_rmse(coords_predict, coords_gt))

    rmses = np.asarray(rmses)
    # overall rmse
    print("overall result RMSE")
    print(str(rmses))
    print("\n")
    print(str(np.nanmean(rmses)))
    res = np.nanmean(rmses)

    return res


# TODO: remove unused code
def treshold_maps(y, threshold=0.9):
    y[y > threshold] = 1
    y[y <= threshold] = 0
    return y


def train_on_data(
    x_train,
    y_train,
    x_test,
    y_test,
    config,
    results_sink,
    segnet_path=None,
    augmentation="primate",
    original_img_shape=None,
    save=None,
):
    remold = False
    y_test_orig = None
    x_test_orig = None
    coms_test = None
    if segnet_path:
        SegNet = SegModel(species="mouse")
        SegNet.inference_config.DETECTION_MIN_CONFIDENCE = 0.001
        SegNet.set_inference(model_path=segnet_path)

        mask_size = 64
        x_test_orig = x_test
        y_test_orig = y_test
        x_train, y_train, _ = segment_images_and_masks(
            x_train, y_train, SegNet=SegNet, mask_size=mask_size
        )
        x_test, y_test, coms_test = segment_images_and_masks(
            x_test, y_test, SegNet=SegNet, mask_size=mask_size
        )
        remold = True

    img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    input_shape = (img_rows, img_cols, 3)

    adam = tf.keras.optimizers.Adam(lr=0.001)
    posenet = posenet_architecture(
        input_shape,
        num_classes=y_train.shape[-1],
        backbone=config["poseestimation_model_backbone"],
    )
    posenet.compile(
        loss=["binary_crossentropy"],
        optimizer=adam,
        metrics=["mse"],
    )

    if config["poseestimation_model_augmentation"] == "primate":
        augmentation_image = primate_poseestimation()
    elif config["poseestimation_model_augmentation"] == "mouse":
        augmentation_image = mouse_poseestimation()
    else:
        raise NotImplementedError

    tf_callback = callbacks_tf_logging(path="./logs/posenet/")

    # TODO: model checkpoint callbacks
    my_metrics = Metrics(unmold=original_img_shape)
    my_metrics.validation_data = (np.asarray(x_test), np.asarray(y_test))
    my_metrics.setModel(posenet)
    viz_cb = callbacks_viz_poseestimation()
    viz_cb.validation_data = (np.asarray(x_test), np.asarray(y_test))
    viz_cb.setModel(posenet)
    callbacks = [my_metrics, viz_cb, tf_callback]

    training_generator = DataGenerator(
        x_train,
        y_train,
        augmentation=augmentation_image,
        batch_size=config["poseestimation_batch_size"],
    )

    for epoch_id, epoch in enumerate(config["poseestimation_model_epochs"]):
        K.set_value(
            posenet.optimizer.lr,
            config["poseestimation_model_learning_rates"][epoch_id],
        )

        dense_history_1 = posenet.fit(
            training_generator,
            epochs=epoch,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            shuffle=True,
            use_multiprocessing=False,
            steps_per_epoch=config["poseestimation_model_steps_per_epochs"][epoch_id],
            # workers=40,
        )

    res = evaluate_pose_estimation(
        x_test,
        y_test,
        posenet,
        remold=remold,
        y_test_orig=y_test_orig,
        x_test_orig=x_test_orig,
        coms_test=coms_test,
    )
    if save:
        posenet.save(results_sink + "posenetNet" + ".h5")
        np.save(results_sink + "poseestimation_results_new.npy", res)


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
    "--dlc_path",
    action="store",
    dest="dlc_path",
    type=str,
    default=None,
    help="path for labeled-data path of deeplabcut labelled data",
)
parser.add_argument(
    "--fold",
    action="store",
    dest="fold",
    type=int,
    default=None,
    help="fold for crossvalidation",
)
parser.add_argument(
    "--results_sink",
    action="store",
    dest="results_sink",
    type=str,
    default=None,
    help="path to results",
)
parser.add_argument(
    "--segnet_path",
    action="store",
    dest="segnet_path",
    type=str,
    default=None,
    help="path to segmentation model",
)
parser.add_argument(
    "--config",
    action="store",
    dest="config",
    type=str,
    default=None,
    help="name of configuration file to use",
)


def main():
    args = parser.parse_args()
    operation = args.operation
    gpu_name = args.gpu
    fraction = args.fraction
    dlc_path = args.dlc_path
    fold = args.fold
    results_sink = args.results_sink
    segnet_path = args.segnet_path
    config = args.config

    setGPU(gpu_name)

    config = load_config("../configs/poseestimation/" + config)
    set_random_seed(config["random_seed"])
    # check_directory(results_sink)
    with open(results_sink + "config.json", "w") as f:
        json.dump(config, f)
    f.close()

    original_img_shape = None
    if operation == "primate":
        x_train, y_train, x_test, y_test = get_primate_pose_data()
    if operation == "mouse":
        x_train, y_train, x_test, y_test = get_mouse_pose_data(fraction=fraction)
    if operation == "dlc_comparison":
        (
            x_train,
            y_train,
            x_test,
            y_test,
            original_img_shape,
        ) = get_mouse_pose_dlc_comparison_data(fold=fold)
    if operation == "mouse_dlc":
        x_train, y_train, x_test, y_test = get_mouse_dlc_data()
    if dlc_path:
        # TODO: integrate exclude labels into cfg
        X, y = read_dlc_labels_from_folder(
            dlc_path, exclude_labels=["tl", "tr", "bl", "br", "centre"]
        )
        split = 4
        x_train = X[split:]
        y_train = y[split:]
        x_test = X[:split]
        y_test = y[:split]

        # TODO: sigmas in config and test
        sigmas = [6.0, 4.0, 1.0, 1.0, 0.5]
        img_shape = (x_train.shape[1], x_train.shape[2])
        y_train = heatmaps_for_images(
            y_train, img_shape=img_shape, sigma=sigmas[0], threshold=None
        )
        y_test = heatmaps_for_images(
            y_test, img_shape=img_shape, sigma=sigmas[0], threshold=None
        )

    train_on_data(
        x_train,
        y_train,
        x_test,
        y_test,
        config=config,
        results_sink=results_sink,
        segnet_path=segnet_path,
        original_img_shape=original_img_shape,
    )


# example usage
# python poseestimation.py --gpu 0 --results_sink /home/markus/posetest/ --dlc_path /home/markus/OFT/labeled-data/ --segnet_path /home/markus/mask_rcnn_mouse_0095.h5 --config poseestimation_config_test
if __name__ == "__main__":
    main()
