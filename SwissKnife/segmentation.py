"""
segmentation.py
====================================
"""

# SIPEC
# MARKUS MARKS
# SEGMENTATION PART
# This code is optimized from the Mask RCNN (Waleed Abdulla, (c) 2017 Matterport, Inc.) repository

import gc
import multiprocessing as mp
import os
import os.path
import random
import warnings
from argparse import ArgumentParser
from time import time

import imgaug.augmenters as iaa
import numpy as np
from joblib import Parallel, delayed

import SwissKnife.mrcnn.model as modellib
from SwissKnife.dataprep import get_segmentation_data
from SwissKnife.mrcnn import utils
## adapted from matterport Mask_RCNN implementation
from SwissKnife.mrcnn.config import Config
from SwissKnife.utils import (check_folder, load_config, save_dict,
                              set_random_seed, setGPU)

warnings.filterwarnings("ignore")

# TODO: fix this import bug here
# from dataprep import get_segmentation_data


# TODO: include validation image that network detects new Ground truth!!
def mold_image(img, config=None, dimension=None, min_dimension=None, return_all=False):
    """
    Args:
        img:
        config:
        dimension:
    """
    if config:
        image, window, scale, padding, crop = utils.resize_image(
            img[:, :, :],
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE,
        )
    elif dimension:
        if min_dimension:
            image, window, scale, padding, crop = utils.resize_image(
                img[:, :, :],
                min_dim=min_dimension,
                max_dim=dimension,
                mode="pad64",
            )
        else:
            image, window, scale, padding, crop = utils.resize_image(
                img[:, :, :],
                min_dim=dimension,
                max_dim=dimension,
                mode="square",
            )
    else:
        return NotImplementedError
    if return_all:
        return image, window, scale, padding, crop
    return image


def mold_video(video, config=None, dimension=None, n_jobs=mp.cpu_count(), min_dimension=None):
    """
    Args:
        video:
        dimension:
        n_jobs:
    """
    results = Parallel(
        n_jobs=n_jobs, max_nbytes=None, backend="multiprocessing", verbose=40
    )(
        delayed(mold_image)(
            image, config=config, dimension=dimension, min_dimension=min_dimension
        )
        for image in video
    )
    return np.asarray(results)


# TODO: batch size in inference
class PrimateConfig(Config):
    """TODO: Fill in description"""
    NAME = "primate"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    BATCH_SIZE = 2
    BACKBONE = "resnet101"
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 10
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.85
    LEARNING_RATE = 0.0025
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 1280
    IMAGE_MAX_DIM = 1280
    IMAGE_SHAPE = [1280, 1280, 3]

    TRAIN_ROIS_PER_IMAGE = 200
    WEIGHT_DECAY = 0.0001

    GRADIENT_CLIP_NORM = 1.0


# TODO: remove unused code
class InferenceConfigPrimate(PrimateConfig):
    """TODO: Fill in description"""
    IMAGE_RESIZE_MODE = "square"
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1

    DETECTION_MIN_CONFIDENCE = 0.8
    IMAGE_MIN_DIM = 1920
    IMAGE_MAX_DIM = 1920
    IMAGE_SHAPE = [1920, 1920, 3]
    # DETECTION_MIN_CONFIDENCE = 0.99
    # IMAGE_MIN_DIM = 4096
    # IMAGE_MAX_DIM = 4096
    # IMAGE_SHAPE = [4096, 4096, 3]


class MouseConfig(Config):
    """TODO: Fill in description"""
    NAME = "mouse"
    BACKBONE = "resnet101"
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.5
    GPU_COUNT = 1

    LEARNING_RATE = 0.001
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_SHAPE = [1024, 1024, 3]
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 6
    TRAIN_ROIS_PER_IMAGE = 128

    WEIGHT_DECAY = 0.0001
    GRADIENT_CLIP_NORM = 1.0


# TODO: remove unused code
class SmallConfig(Config):
    """TODO: Fill in description"""
    NAME = "small"
    BACKBONE = "resnet101"
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 2
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.5
    GPU_COUNT = 1

    LEARNING_RATE = 0.001
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_SHAPE = [512, 512, 3]
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    MAX_GT_INSTANCES = 20

    TRAIN_ROIS_PER_IMAGE = 200
    WEIGHT_DECAY = 0.0001

    GRADIENT_CLIP_NORM = 1.0


# TODO: remove unused code
class InferenceConfigSmall(MouseConfig):
    """TODO: Fill in description"""
    # TODO: test / anpassen
    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1


# TODO: remove unused code
class InferenceConfigMouse(MouseConfig):
    """TODO: Fill in description"""
    # TODO: test / anpassen
    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1


# TODO: remove unused code
class IneichenConfig(Config):
    """TODO: Fill in description"""
    NAME = "mouse"
    BATCH_SIZE = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.95
    GPU_COUNT = 1

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    # TRAIN_BN = True
    MINI_MASK_SHAPE = (56, 56)


# TODO: remove unused code
class InferencIneichenConfig(IneichenConfig):
    """TODO: Fill in description"""
    DETECTION_MIN_CONFIDENCE = 0.99
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1


# TODO: remove unused code
class IneichenConfigSmall(IneichenConfig):
    """TODO: Fill in description"""
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    IMAGE_SHAPE = [320, 320, 3]
    MINI_MASK_SHAPE = (56, 56)


class MaskFilter:
    """
    Return the most important thing about a person.
    Parameters
    ----------
    your_name
        A string indicating the name of the person.
    """

    def __init__(self):
        pass

    def train(self):
        """TODO: Fill in description"""
        pass

    def predict(self):
        """TODO: Fill in description"""
        pass


class SegModel:
    """TODO: Fill in description"""
    # TODO: give confidence as argument
    def __init__(self, species, training_config=None, inference_config=None):
        """Main class for a segmentation model used for training and inference.
        Args:
            species:Species to initialize parameters with. "mouse" and "primate" are available.
        """
        self.species = species
        self.config = training_config
        self.inference_config = inference_config

        self.model_path = None
        self.augmentation = None
        self.model = None

    def train(self, dataset_train, dataset_val):
        """Train the segmentation network.
        Args:
            dataset_train:
            dataset_val:
        """

        # TODO:modulefy me
        if self.config.NAME == "test":
            training_params_dict = [
                [5, "heads", 1.0],
            ]
        else:
            if self.species == "primate":
                if self.load_model_path:
                    training_params_dict = [
                        [150, "all", 10.0],
                    ]
                else:
                    training_params_dict = [
                        [3, "heads", 1.0],
                        [5, "5+", 1.0],
                        [8, "4+", 1.0],
                        [10, "3+", 1.0],
                        [60, "all", 5.0],
                        [100, "all", 10.0],
                        [200, "all", 15.0],
                    ]
            else:
                if self.load_model_path:
                    training_params_dict = [
                        [125, "all", 5.0],
                        [200, "all", 10.0],
                    ]
                else:
                    training_params_dict = [
                        [1, "heads", 1.0],
                        [5, "5+", 1.0],
                        [8, "4+", 1.0],
                        [20, "3+", 1.0],
                        [100, "all", 5.0],
                        [125, "all", 7.5],
                        [200, "all", 10.0],
                    ]

        for training_params in training_params_dict:
            epochs, layers, lr_modifier = training_params

            self.model.train(
                dataset_train,
                dataset_val,
                learning_rate=self.config.LEARNING_RATE / lr_modifier,
                epochs=epochs,
                layers=layers,
                augmentation=self.augmentation,
            )

    def init_augmentation(self):
        """Initializes the augmentation for segmentation network training,
        different default levels are available.
        Args:
            dataset_train:
            dataset_val:
        """
        if self.species in ("mouse", "ineichen"):
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            self.augmentation = iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    #             iaa.Fliplr(0.5), # horizontally flip 50% of all images\
                    # crop images by -5% to 10% of their height/width
                    sometimes(
                        iaa.Affine(
                            # #                 scale={("x": (0.75, 1.25), "y": (0.75, 1.25))}, # scale images to 80-120% of their size, individually per axis
                            scale=(
                                0.9,
                                1.1,
                            ),  # scale images to 80-120% of their size, individually per axis
                            #                 #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                            rotate=(-90, 90),  # rotate by -45 to +45 degrees
                            shear=(-10, 10),  # shear by -16 to +16 degrees
                            #                 order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                            #                 cval=0, # if mode is constant, use a cval between 0 and 255
                            #                 mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                        )
                    ),
                    sometimes(
                        iaa.CoarseDropout(p=0.2, size_percent=0.8, per_channel=False)
                    ),
                    sometimes(
                        iaa.CoarseDropout(p=0.05, size_percent=0.25, per_channel=False)
                    ),
                    sometimes(iaa.GaussianBlur(sigma=(0, 0.5))),
                ],
                random_order=True,
            )

        if self.species == ("primate", "jin"):
            sometimes = lambda aug: iaa.Sometimes(0.2, aug)  # latest run 0.2

            self.augmentation = iaa.Sequential(
                [
                    sometimes(
                        iaa.CoarseDropout(p=0.1, size_percent=0.02, per_channel=False)
                    ),
                    sometimes(
                        iaa.CoarseDropout(p=0.1, size_percent=0.2, per_channel=False)
                    ),
                    sometimes(
                        iaa.CoarseDropout(p=0.1, size_percent=0.8, per_channel=False)
                    ),
                ],
                random_order=True,
            )

    def init_training(self, model_path, load_model_path=None, init_with="coco"):
        """Initialized training of a new or existing segmentation network.
        Args:
            model_path:Path to segmentation network, either existing or desired path for new model.
            init_with:The initializations "imagenet" or "coco" are available for new segmentation models and "last" if retraining existing network.
        """
        self.model_path = model_path
        self.load_model_path = load_model_path
        # Create model in training mode
        self.model = modellib.MaskRCNN(
            mode="training", config=self.config, model_dir=self.model_path
        )

        self.config.display()

        if load_model_path:
            self.model.load_weights(
                load_model_path,
                by_name=True,
            )
        elif init_with == "imagenet":
            self.model.load_weights(self.model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            COCO_MODEL_PATH = os.path.join("./", "mask_rcnn_coco.h5")
            if not os.path.exists(COCO_MODEL_PATH):
                utils.download_trained_weights(COCO_MODEL_PATH)
            self.model.load_weights(
                COCO_MODEL_PATH,
                by_name=True,
                exclude=[
                    "mrcnn_class_logits",
                    "mrcnn_bbox_fc",
                    "mrcnn_bbox",
                    "mrcnn_mask",
                ],
            )
        else:
            raise NotImplementedError

    # FIXME: remove hardcoing
    # functioning primate model
    # path = '/home/nexus/mask_rcnn_primate_0119.h5'
    def set_inference(self, model_path=None):
        """Set segmentation model to inference.
        Args:
            model_path:
        """
        if "mask_rcnn" in model_path:
            helper_path = model_path.split("mask_rcnn")[0]
            self.model = modellib.MaskRCNN(
                mode="inference", config=self.inference_config, model_dir=helper_path
            )
        elif "mask_rcnn" not in model_path:
            self.model = modellib.MaskRCNN(
                mode="inference", config=self.inference_config, model_dir=model_path
            )
            model_path = self.model.find_last()
        else:
            return NotImplementedError

        # Recreate the model in inference mode
        self.model.load_weights(model_path, by_name=True)
        return model_path

    def evaluate(self, dataset_val, maskfilter=None):
        """Evaluate segmentation model on a given validation set.
        Args:
            dataset_val:Validation dataset.
            maskfilter:
        """
        image_ids = dataset_val.image_ids

        APs = []
        IOUs = []
        dice = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, _, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
                dataset_val, self.inference_config, image_id, use_mini_mask=False
            )
            r = self.detect_image_original(image, verbose=0)
            if maskfilter:
                r = maskfilter.predict(r)
            # Compute AP
            AP, _, _, _, _ious, dices = utils.compute_ap(
                gt_bbox,
                gt_class_id,
                gt_mask,
                r["rois"],
                r["class_ids"],
                r["scores"],
                r["masks"],
            )
            APs.append(AP)
            # IOUs.append(np.mean(overlaps))
            IOUs.append(_ious)
            dice.append(dices)

        mean_ap = np.mean(APs)
        mean_iou = np.mean(IOUs)
        mean_dice = np.mean(dice)
        print("evaluation done")
        print("mAP: ", mean_ap)
        print("IOUUUUU: ", mean_iou)
        print("dice: ", mean_dice)

        return mean_ap, mean_iou, mean_dice

    # TODO: implement adaptive batch inference
    def detect_image(self, img, mold=True, verbose=1):
        """
        Args:
            img:
            mold:
            verbose:
        """
        if mold:
            # img = mold_image(img, self.inference_config, dimension=2048)
            img = mold_image(img, self.inference_config, dimension=1024)
        result = self.model.detect([img], verbose=verbose)
        return img, result[0]["masks"], result[0]["rois"], result[0]["scores"]

    def detect_image_original(self, img, mold=True, verbose=1):
        """
        Args:
            img:
            mold:
            verbose:
        """
        if mold:
            img = mold_image(img, self.inference_config)
        result = self.model.detect([img], verbose=verbose)
        return result[0]

    # TODO: remove unused code
    def detect_batch(self, img_list, mold=True, verbose=1):
        """
        Args:
            img_list:
            mold:
            verbose:
        """
        if mold:
            img_list = mold_video(img_list, self.inference_config)
        result = self.model.detect(img_list, verbose=verbose)
        return result

    # TODO: remove unused code
    def detect_video(self, video, results_sink=None):

        """
        Args:
            video:
            results_sink:
        """
        videodata = mold_video(video, self.inference_config)

        results = []
        batch_size = self.inference_config.BATCH_SIZE
        batches = int(len(videodata) / batch_size)

        for idx, _ in enumerate(range(batches)):
            start = time()
            data = videodata[idx * batch_size : (idx + 1) * batch_size]
            vid_results = self.model.detect(data, verbose=1)
            results = results + vid_results
            print("time", time() - start)

        if results_sink:
            check_folder(results_sink)
            save_dict(results_sink + "SegResults.pkl", results)
        else:
            return results


# TODO: remove unused code
def evaluate_network(
    model_path,
    species,
    filter_masks=False,
    cv_folds=0,
    pass_fold=None,
    name=None,
    fraction=None,
):
    # load training and val data
    """
    Args:
        model_path:
        species:
        filter_masks:
        cv_folds:
    """
    mean_aps = []
    IOUs = []
    dice = []
    for fold in range(cv_folds + 1):
        if not fold == pass_fold:
            continue
        dataset_train, dataset_val = get_segmentation_data(
            species, cv_folds=cv_folds, fold=fold, fraction=1.0, name="mouse"
        )
        model = SegModel(species)
        model.set_inference(model_path=model_path)
        if filter_masks:
            maskfilter = MaskFilter()
            maskfilter.train(dataset_train)
            mean_ap, mean_iou, mean_dice = model.evaluate(
                dataset_val, maskfilter=maskfilter
            )
        else:
            mean_ap, mean_iou, mean_dice = model.evaluate(dataset_val)
        print("MEAN AP", mean_ap)
        mean_aps.append(mean_ap)
        IOUs.append(mean_iou)
        dice.append(mean_dice)
    print("overall aps", mean_aps)
    print("mAP: ", str(np.mean(np.array(mean_aps))))
    print("IOUUUUU: ", str(np.mean(np.array(IOUs))))
    print("dice: ", str(np.mean(np.array(dice))))

    np.save(
        "./" + name + "res.npy",
        [np.mean(np.array(mean_aps)), np.mean(np.array(IOUs)), np.mean(np.array(dice))],
    )


# TODO: change cv folds to None default
# TODO: seperate training from evaluation
def train_on_data_once(
    model_path,
    species,
    config,
    inference_config=None,
    cv_folds=0,
    frames_path=None,
    load_model_path=None,
    annotations_path=None,
    base_folder=None,
    fold=0,
    fraction=None,
    perform_evaluation=True,
    debug=1,
):
    """Performs training for the segmentation moduel of SIPEC (SIPEC:SegNet).

    Parameters
    ----------
    inference_config
    model_path : str
        Path to model, can be either where a new model should be stored or 
        a path to an existing model to be retrained.
    cv_folds : int
        Number of cross_validation folds, use 0 for a normal train/test split.
    frames_path : str
        Path to the frames used for training.
    annotations_path : str
        Path to the annotations used for training.
    species : str
        Species to perform segmentation on (can be any species, but "mouse" or "primate" have more specialised parameters). 
        If your species is neither "mouse" nor "primate", use "default".
    fold : int
        If cv_folds > 1, fold is the number of fold to be tested on.
    fraction : float
        Factor by which to decimate the training data points.
    perform_evaluation : bool
        Perform subsequent evaluation of the model
    debug : bool
        Debug verbosity.


    Returns
    -------
    model
        SIPEC:SegNet model
    mean_ap
        Mean average precision score achieved by this model
    """
    dataset_train, dataset_val = get_segmentation_data(
        frames_path=frames_path,
        annotations_path=annotations_path,
        base_folder=base_folder,
        name=species,
        cv_folds=cv_folds,
        fold=fold,
        fraction=fraction,
    )
    # initiate mouse model
    model = SegModel(species, training_config=config)
    # initiate training
    model.init_training(
        model_path=model_path, load_model_path=load_model_path, init_with="imagenet"
    )
    model.init_augmentation()
    # start training
    print("training on #NUM images : ", str(len(dataset_train.image_ids)))
    model.train(dataset_train, dataset_val)
    # evaluate model
    if perform_evaluation:
        model = SegModel(species)
        model.inference_config = inference_config
        model.inference_config.__init__()
        model_path = model.set_inference(model_path=model_path)
        mean_ap, mean_iou, mean_dice = model.evaluate(dataset_val)
    if debug:
        helper = model_path.split("mask_rcnn_" + species + "_0")
        epochs = [
            "030",
            "050",
            "095",
        ]
        print(helper)
        print(helper[0] + "mask_rcnn_" + species + "_0" + "001" + ".h5")
        for epoch in epochs:
            model = SegModel(species)
            model.set_inference(
                model_path=helper[0] + "mask_rcnn_" + species + "_0" + epoch + ".h5"
            )
            mean_ap = model.evaluate(dataset_val)
            print(epoch)
            print(mean_ap)

    return model, mean_ap, mean_iou, mean_dice


# TODO: remove unused code
def do_ablation(species, cv_folds, random_seed, fraction):
    """
    Args:
        species:
        cv_folds:
        random_seed:
        fraction:
    """
    experiment_name = "ablation"

    results_path = "~/segmentation/" + species + "_" + experiment_name
    results_fname = (
        results_path
        + "results_array"
        + "_"
        + str(random_seed)
        + "_"
        + str(fraction)
        + ".npy"
    )
    if os.path.isfile(results_fname):
        results = list(np.load(results_fname, allow_pickle=True))
    else:
        results = [["random_seed", "data_fraction", "MEAN_AP"]]

    set_random_seed(random_seed)
    random.seed(random_seed)

    mean_aps = train_on_data(
        species,
        cv_folds=cv_folds,
        fraction=fraction,
        to_file=False,
        experiment=experiment_name,
    )
    results.append([random_seed, fraction, mean_aps])
    # check_folder(results_path)
    np.save(results_fname, results, allow_pickle=True)


# TODO: shorten
def train_on_data(
    species, cv_folds, fraction=None, to_file=True, experiment="", fold=None
):
    # path, where to save trained model
    """
    Args:
        species:
        cv_folds:
        fraction:
        to_file:
        experiment:
        fold:
    """
    #experiment_name = "cv"
    # results_path = "./segmentation_logs_1st_review/" + species + "_" + experiment_name + "/"
    base = "~/"
    results_path = base + "segmentation/" + species + "_" + experiment

    model_path = (
        results_path + str(fraction).replace(".", "_") + "_fold_" + str(fold) + "/"
    )

    print(model_path)

    start = time()

    #model = None
    mean_aps = []
    if cv_folds > 0:
        if fold is not None:
            print("TRAINING on FOLD", str(fold))
            _, mean_ap, mean_iou, mean_dice = train_on_data_once(
                model_path, species, cv_folds=cv_folds, fold=fold, fraction=fraction
            )
            mean_aps.append(mean_ap)
            gc.collect()
        else:
            for fold in range(cv_folds):
                _, mean_ap, mean_iou, mean_dice = train_on_data_once(
                    model_path, species, cv_folds=cv_folds, fold=fold, fraction=fraction
                )
                mean_aps.append(mean_ap)
                gc.collect()
    else:
        _, mean_ap, mean_iou, mean_dice = train_on_data_once(
            model_path, species, cv_folds=cv_folds, fold=0, fraction=fraction
        )
        print("MEAN AP", mean_ap)

    end = time() - start

    print("time : " + str(end))

    if to_file:

        results_fname = (
            results_path
            + "results_array"
            + "_"
            + str(0)
            + "_"
            + str(fraction)
            + "_fold_"
            + str(fold)
            + ".npy"
        )
        if os.path.isfile(results_fname):
            results = list(np.load(results_fname, allow_pickle=True))
        else:
            results = [["random_seed", "data_fraction", "MEAN_AP"]]
        results.append([0, fraction, mean_ap, mean_iou, mean_dice])
        print("mean aps : " + str(mean_ap))
        print("mean_iou : " + str(mean_iou))
        print("mean_dice : " + str(mean_dice))

        # check_folder(results_path)
        np.save(results_fname, results, allow_pickle=True)

        np.save(results_path + "_" + str(fraction) + "_" + str(fold) + "_time.npy", end)

    return mean_aps


# TODO: remove unused code
def train_on_data_path(annotations, frames):
    """
    Args:
        annotations:
        frames:
    """
    pass


def main():
    """TODO: Fill in description"""
    args = parser.parse_args()
    gpu_name = args.gpu
    cv_folds = args.cv_folds
    model_path = args.model_path
    load_model_path = args.load_model_path
    annotations = args.annotations
    frames = args.frames
    operation = args.operation
    #fraction = args.fraction
    #fold = args.fold
    #name = args.name
    base_folder = args.base_folder
    config = args.config
    inference_config = args.inference_config

    random_seed = 42
    setGPU(gpu_name)
    set_random_seed(random_seed)

    config = load_config("../configs/segmentation/" + config)
    cfg = Config()
    for key, value in config.items():
        cfg.__dict__[key] = value
    cfg.__init__()

    inference_config = load_config("../configs/segmentation/" + inference_config)
    inference_cfg = Config()
    for key, value in inference_config.items():
        inference_cfg.__dict__[key] = value
    inference_cfg.__init__()

    train_on_data_once(
        species=operation,
        model_path=model_path,
        config=cfg,
        inference_config=inference_cfg,
        load_model_path=load_model_path,
        annotations_path=annotations,
        base_folder=base_folder,
        frames_path=frames,
        cv_folds=cv_folds,
        debug=0,
    )

    print("done")


parser = ArgumentParser()
parser.add_argument(
    "--cv_folds",
    action="store",
    dest="cv_folds",
    type=int,
    default=0,
    help="folds for cross validation",
)
parser.add_argument(
    "--operation",
    action="store",
    dest="operation",
    type=str,
    default="default",
    help="deprecated - only for reproduction of SIPEC paper results",
)
parser.add_argument(
    "--config",
    action="store",
    dest="config",
    type=str,
    default="training_default",
    help="config to use",
)
parser.add_argument(
    "--inference_config",
    action="store",
    dest="inference_config",
    type=str,
    default="inference_default",
    help="inference config to use",
)
parser.add_argument(
    "--gpu",
    action="store",
    dest="gpu",
    type=str,
    default=None,
    help="number of the gpu to use (can be used to run multiple training processes on the same machine)",
parser.add_argument(
    "--fraction",
    action="store",
    dest="fraction",
    type=float,
    default=None,
    help="fraction to use for training",
)
# TODO: add default path
parser.add_argument(
    "--model_path",
    action="store",
    dest="model_path",
    type=str,
    default=None,
    help="model path for saving",
)
parser.add_argument(
    "--load_model_path",
    action="store",
    dest="load_model_path",
    type=str,
    default=None,
    help="model path for evaluation or training continuation",
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
    "--name",
    action="store",
    dest="name",
    type=str,
    default=None,
    help="path to folder with annotated frames",
)
parser.add_argument(
    "--base_folder",
    action="store",
    dest="base_folder",
    type=str,
    default=None,
    help="path to folders with subfolders of annotated frames and json inside",
)

if __name__ == "__main__":
    main()

# example usage
# python segmentation.py --cv_folds 0 --gpu 0 --frames ./published_data_zenodo/mouse/segmentation_single/annotated_frames --annotations ./published_data_zenodo/mouse/segmentation_single/mouse_top_segmentation.json --model_path ./test_models

# Docker usage
# docker container run -v "/home/tarun/Documents/Work/Neuro_technology/data:/home/user/data" -v "/home/tarun/Documents/Work/Neuro_technology/results:/home/user/results" -v "/home/tarun/Documents/Work/Neuro_technology/SIPEC:/home/user/SIPEC:ro" --runtime=nvidia --rm chadhat/sipec:tf2 segmentation.py --cv_folds 0 --gpu 0 --frames /home/user/data/mouse_segmentation_single/annotated_frames --annotations /home/user/data/mouse_segmentation_single/mouse_top_segmentation.json
