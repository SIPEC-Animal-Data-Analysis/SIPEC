# SIPEC
# MARKUS MARKS
# UTILITY FUNCTIONS
import datetime
import random
import sys
from glob import glob
import pandas as pd

from scipy.ndimage import center_of_mass
from skimage.filters import threshold_minimum
from skimage.measure import regionprops
from skimage.transform import rescale

sys.path.append("../")

import os
import pickle
from distutils.version import LooseVersion
import os.path

# import matplotlib.pyplot as plt
import numpy as np
import skimage
import skvideo
import skvideo.io
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, f1_score

from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.utils import multi_gpu_model


def masks_to_coords(masks):
    coords = []
    for i in range(masks.shape[-1]):
        coords.append(np.column_stack(np.where(masks[:, :, i] > 0)).astype("uint16"))
    return coords


def coords_to_masks(coords, dim=2048):
    masks = np.zeros((dim, dim, len(coords)), dtype="uint8")
    for coord_id, coord in enumerate(coords):
        for co in coord:
            masks[co[0], co[1], coord_id] = 1
    return masks


def saveModel(model):
    json_model = model_tt.model.to_json()
    open("model_architecture.json", "w").write(json_model)
    model_tt.model.save_weights("model_weights.h5", overwrite=True)


def clearMemory(model, backend):
    del model
    backend.clear_session()


def get_tensorbaord_callback(path="./logs"):
    # Tensorflow board
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=path, histogram_freq=0, write_graph=True, write_images=True
    )
    return tensorboard_callback


# helper class to keep track of results from different methods
class ResultsTracker:
    # write results as comma seperated lines in a single file
    def __init__(self, path=None):
        self.path = path

    def add_result(self, results):
        if os.path.exists(self.path):
            while not self.file_available():
                pass
        self.write_results(results)

    def write_results(self, results):
        for result in results:
            hs = open(self.path, "a")
            hs.write(result + "\n")
            hs.close()

    def file_available(self):
        try:
            os.rename(self.path, self.path)
            print('Access on file "' + self.path + '" is available!')
            return 1
        except OSError as e:
            print('Access on file "' + self.path + '" is not available!')
            print(str(e))
            return 0


import ast


# TODO: include multi behavior
def load_vgg_labels(annotations, video_length, framerate_video, behavior=None):
    if type(annotations) == "str":
        annotations = pd.read_csv(annotations, error_bad_lines=False, header=1)
    labels = ["none"] * video_length

    if "temporal_segment_start" in annotations.columns:
        for line in annotations.iterrows():
            start = int(line[1]["temporal_segment_start"] * framerate_video)
            end = int(line[1]["temporal_segment_end"] * framerate_video)
            if behavior is not None:
                label = behavior
            else:
                label = ast.literal_eval(line[1]["metadata"])["1"]
            labels[start:end] = [label] * (end - start)
    elif "temporal_coordinates" in annotations.columns:
        for line in annotations.iterrows():
            start = int(
                float(line[1]["temporal_coordinates"][1:-1].split(",")[0])
                * framerate_video
            )
            end = int(
                float(line[1]["temporal_coordinates"][1:-1].split(",")[1])
                * framerate_video
            )
            if behavior is not None:
                label = behavior
            else:
                label = ast.literal_eval(line[1]["metadata"])["1"]
            labels[start:end] = [label] * (end - start)
    else:
        raise NotImplementedError

    return labels


def distance(x, y, x_prev, y_prev):
    return np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)


def calculate_speed(distances):
    x = range(0, len(distances))
    y = distances
    dx = np.diff(x)
    dy = np.diff(y)
    d = dy / dx

    return d


# crop png images for segmentation inputs
def crop_pngs():
    basepath = "/media/nexus/storage1/swissknife_data/primate/segmentation_inputs/annotated_frames/"
    new_path = "/media/nexus/storage1/swissknife_data/primate/segmentation_inputs/annotated_frames_resized/"
    folders = ["train/", "val/"]
    for folder in folders:
        path = basepath + folder
        images = glob(path + "*.png")
        for image in images:
            helper = skimage.io.imread(image)
            plt.figure(figsize=(20, 10))
            plt.imshow(helper)
            new_img = helper[:1024, :, :]
            filename = image.split(folder)[-1]
            skimage.io.imsave(new_path + folder + filename, new_img)


def rescale_img(mask, frame, mask_size=256):
    rectsize = [mask[3] - mask[1], mask[2] - mask[0]]

    rectsize = np.asarray(rectsize)
    scale = mask_size / rectsize.max()

    cutout = frame[mask[0] : mask[0] + rectsize[1], mask[1] : mask[1] + rectsize[0], :]

    img_help = rescale(cutout, scale, multichannel=True)
    padded_img = np.zeros((mask_size, mask_size, 3))

    padded_img[
        int(mask_size / 2 - img_help.shape[0] / 2) : int(
            mask_size / 2 + img_help.shape[0] / 2
        ),
        int(mask_size / 2 - img_help.shape[1] / 2) : int(
            mask_size / 2 + img_help.shape[1] / 2
        ),
        :,
    ] = img_help

    return padded_img


# TODO: make all of these part of segmentation/identification


def set_random_seed(random_seed):
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    my_rnd_seed = np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    #tf.set_random_seed(random_seed)
    #tf.random.set_random_seed(random_seed)


def detect_primate(_img, _model, classes, threshold):
    prediction = _model.predict(np.expand_dims(_img, axis=0))
    if prediction.max() > threshold:
        return classes[np.argmax(prediction)], prediction.max()
    else:
        return "None detected", prediction.max()


def masks_to_coms(masks):
    # calculate center of masses
    coms = []
    for idx in range(0, masks.shape[-1]):
        mask = masks[:, :, idx]
        com = center_of_mass(mask.astype("int"))
        coms.append(com)
    coms = np.asarray(coms)

    return coms


def apply_to_mask(mask, img, com, mask_size):
    masked_img = maskedImg(img, com, mask_size=mask_size)
    masked_mask = maskedImg(mask, com, mask_size=mask_size)

    return masked_img, masked_mask


def apply_all_masks(masks, coms, img, mask_size=128):
    # mask images
    masked_imgs = []
    masked_masks = []
    for idx, com in enumerate(coms):
        mask = masks[:, :, idx]
        masked_img, masked_mask = apply_to_mask(mask, img, com, mask_size=mask_size)
        masked_masks.append(masked_mask)
        masked_imgs.append(masked_img)

    return np.asarray(masked_imgs), np.asarray(masked_masks)


# functions for data processing

### BEHAVIOR PREPROCESSING ###
def startend(df_entry, ms, df):
    start = float(df_entry["temporal_coordinates"][2:-2].split(",")[0]) / ms
    end = float(df_entry["temporal_coordinates"][2:-2].split(",")[1]) / ms
    label = df_entry["metadata"][2:-2].split(":")[-1][1:-1]
    length = float(end - start)

    return int(start), int(end), label, length


#### manual segmentation ###


def extractCOM(image, threshold):
    try:
        try:
            threshold = threshold_minimum(image, nbins=256)
        except RuntimeError:
            threshold = threshold_minimum(image, nbins=768)
    except RuntimeError:
        threshold = threshold_minimum(image, nbins=1024)
    thresh = image > threshold
    labeled_foreground = (thresh).astype(int)
    properties = regionprops(labeled_foreground, thresh)
    center_of_mass = properties[0].centroid
    weighted_center_of_mass = properties[0].weighted_centroid

    return center_of_mass, weighted_center_of_mass


# TODO: fixme / streamline
def extractCOM_only(image):
    properties = regionprops(image)
    center_of_mass = properties[0].centroid
    weighted_center_of_mass = properties[0].weighted_centroid

    return center_of_mass, weighted_center_of_mass


def maskedImg(
    img, center_of_mass, mask_size=74,
):
    if len(img.shape) == 2:
        ret = np.zeros((int(mask_size * 2), int(mask_size * 2)))
    else:
        ret = np.zeros((int(mask_size * 2), int(mask_size * 2), img.shape[-1]))

    cutout = img[
        np.max([0, int(center_of_mass[0] - mask_size)]) : np.min(
            [img.shape[0], int(center_of_mass[0] + mask_size)]
        ),
        np.max([0, int(center_of_mass[1] - mask_size)]) : np.min(
            [img.shape[0], int(center_of_mass[1] + mask_size)]
        ),
    ]

    ret[
        ret.shape[0] - int(cutout.shape[0]) : ret.shape[0] + int(cutout.shape[0]),
        ret.shape[1] - int(cutout.shape[1]) : ret.shape[1] + int(cutout.shape[1]),
    ] = cutout

    return ret.astype("uint8")


### DL Utils
# TODO: own file for DL utils


def plotHistory(history, measure):
    plt.plot(history.history[measure])
    plt.plot(history.history["val_" + measure])
    plt.title("model" + measure)
    plt.ylabel(measure)
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """

    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss

    return focal_loss


def f1(y_true, y_pred):
    y_true = K.cast(y_true, "float")

    #     y_pred = K.round(y_pred)

    y_pred = K.cast(K.greater(K.cast(y_pred, "float"), 0.01), "float")

    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def balanced_acc(y_true, y_pred):
    with sess.as_default():
        return balanced_accuracy_score(y_true.eval(), y_pred.eval())


from sklearn.metrics import classification_report


class Metrics(keras.callbacks.Callback):
    def setModel(self, model):
        self.model = model

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_val = np.where(y_val == 1)[1].astype(int)

        # old
        y_predict = self.model.predict(X_val)
        y_predict = np.argmax(y_predict, axis=-1).astype(int)
        print(classification_report(y_val, y_predict))

        self._data.append(
            {
                # 'val_roc': roc_auc_score(y_val, y_predict, average='macro'),
                "val_balanced_acc": balanced_accuracy_score(y_val, y_predict),
                "val_sklearn_f1": f1_score(y_val, y_predict, average="macro"),
            }
        )
        print("val_balanced_acc ::: " + str(balanced_accuracy_score(y_val, y_predict)))
        print("val_sklearn_f1 ::: " + str(f1_score(y_val, y_predict, average="macro")))
        return

    def get_data(self):
        return self._data


# TODO: maybe somewhere else?
def get_optimizer(optim_name, lr=0.01):
    optim = None
    if optim_name == "adam":
        optim = keras.optimizers.Adam(lr=lr, clipnorm=0.5)
    if optim_name == "sgd":
        optim = keras.optimizers.SGD(lr=lr, clipnorm=0.5, momentum=0.9)
    if optim_name == "rmsprop":
        optim = keras.optimizers.RMSprop(lr=lr)
    return optim


def get_callbacks():
    CB_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", min_delta=0.0001, verbose=True, patience=8, min_lr=1e-7
    )

    CB_es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=8,
        mode="min",
        restore_best_weights=True,
    )

    return CB_es, CB_lr


def train_model(
    model,
    optimizer,
    epochs,
    batch_size,
    data_train,
    data_val=None,
    callbacks=None,
    class_weights=None,
    loss="crossentropy",
    augmentation=None,
    num_gpus=1,
):
    if num_gpus > 1:
        print("This part needs to be fixed!!!")
        #model = multi_gpu_model(model, gpus=num_gpus, cpu_merge=True)
    if loss == "crossentropy":
        # TODO: integrate number of GPUs in config
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=["categorical_crossentropy", "categorical_accuracy"],
        )
    elif loss == "focal_loss":
        model.compile(
            loss=categorical_focal_loss(gamma=3.0, alpha=0.5),
            optimizer=optimizer,
            metrics=["categorical_crossentropy", "categorical_accuracy", f1],
        )
    else:
        raise NotImplementedError

    print(model.summary())

    if augmentation:
        image_gen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=augmentation.augment_image,
        )

        try:
            batch_gen = image_gen.flow(
                data_train[0],
                data_train[1],
                batch_size=batch_size,
                shuffle=True,
                # TODO: implement here
                # TODO: fix seed globallly
                #     sample_weight=train_sample_weights,
                # TODO: check if global seed works here
                # seed=42,
            )
        except ValueError:
            batch_gen = image_gen.flow(
                data_train[0],
                data_train[1],
                batch_size=batch_size,
                shuffle=True,
                # TODO: implement here
                # TODO: fix seed globallly
                #     sample_weight=train_sample_weights,
                # seed=42,
            )
        # TODO: implement me
        # if balanced:
        # training_generator, steps_per_epoch = balanced_batch_generator(x_train, y_train,
        #                                                                sampler=RandomOverSampler(),
        #                                                                batch_size=32,
        #                                                                random_state=42)

        if class_weights is not None:
            training_history = model.fit_generator(
                batch_gen,
                epochs=epochs,
                steps_per_epoch=len(data_train[0]),
                validation_data=(data_val[0], data_val[1]),
                callbacks=callbacks,
                class_weight=class_weights,
                use_multiprocessing=True,
                workers=40,
            )
        else:
            training_history = model.fit_generator(
                batch_gen,
                epochs=epochs,
                # TODO: check here, also multiprocessing
                steps_per_epoch=len(data_train[0]),
                validation_data=(data_val[0], data_val[1]),
                callbacks=callbacks,
                use_multiprocessing=True,
                workers=40,
            )

    else:
        if class_weights is not None:
            training_history = model.fit(
                data_train[0],
                data_train[1],
                epochs=epochs,
                batch_size=batch_size,
                # TODO: here validation split instead ?
                validation_data=(data_val[0], data_val[1]),
                callbacks=callbacks,
                shuffle=True,
                class_weight=class_weights,
            )
        else:
            training_history = model.fit(
                data_train[0],
                data_train[1],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(data_val[0], data_val[1]),
                callbacks=callbacks,
                shuffle=True,
            )

    return model, training_history


def eval_model(
    model, data, results_dict, results_array, filename, dataloader, model_name="",
):
    true_confidence = model.predict(data)
    true_numerical = np.argmax(true_confidence, axis=-1).astype(int)
    # TODO: also save certainty

    true_behavior = dataloader.decode_labels(true_numerical)

    true_numerical = np.expand_dims(true_numerical, axis=-1)
    true_behavior = np.expand_dims(true_behavior, axis=-1)

    true = np.hstack([true_confidence, true_numerical, true_behavior])

    # TODO: generate automatically
    res = results_dict.copy()
    res[model_name + filename] = true

    res_array = results_array.copy()
    for el in true:
        res_array.append(np.hstack([model_name, filename, el]))
    return res, res_array


def save_dict(filename, dict):
    with open(filename, "wb") as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(filename):
    with open(filename, "rb") as handle:
        file = pickle.load(handle)
    return file


def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print("experiment already exists")
        raise ValueError


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def check_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


### set gpu backend
def setGPU(backend, GPU):
    # Outdated syntax from tf1
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.visible_device_list = GPU
    #https://www.tensorflow.org/guide/migrate
    tf.config.set_visible_devices(GPU, 'GPU')
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[GPU], 'GPU')
    # session = tf.Session(config=config)
    # TODO: Replace the following by tf2 equivalent 
    ##backend.tensorflow_backend.set_session(tf.Session(config=config))


def pathForFile(paths, filename):
    if "labels" in paths[0]:
        filename = filename + "_"
    else:
        filename = filename + "."
    for path in paths:
        if filename in path:
            return path
    return "none"


def loadVideo(path, num_frames=None, greyscale=True):
    # load the video
    if not num_frames is None:
        return skvideo.io.vread(path, as_grey=greyscale, num_frames=num_frames)
    else:
        return skvideo.io.vread(path, as_grey=greyscale)


def load_config(path):
    params = {}
    with open(path) as f:
        for line in f.readlines():
            if "\n" in line:
                line = line.split("\n")[0]
            try:
                params[line.split(" = ")[0]] = int(line.split(" = ")[1])
            except ValueError:
                try:
                    params[line.split(" = ")[0]] = float(line.split(" = ")[1])
                except ValueError:
                    params[line.split(" = ")[0]] = str(line.split(" = ")[1])
    return params


# adapted from maskrcnn
def resize(
    image,
    output_shape,
    order=1,
    mode="constant",
    cval=0,
    clip=True,
    preserve_range=False,
    anti_aliasing=False,
    anti_aliasing_sigma=None,
):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image,
            output_shape,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma,
        )
    else:
        return skimage.transform.resize(
            image,
            output_shape,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
        )


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y : y + min_dim, x : x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop
