"""
SIPEC
MARKUS MARKS
IDENTIFICATION
"""

import json
import pickle
import random
from argparse import ArgumentParser
from datetime import datetime

import imageio
import matplotlib.pyplot as plt
import numpy as np
import skvideo
from joblib import Parallel, delayed
from skimage.transform import rescale
from sklearn import metrics
from tensorflow.keras import backend as K

from SwissKnife.architectures import idtracker_ai
from SwissKnife.augmentations import mouse_identification, primate_identification
from SwissKnife.dataloader import Dataloader
from SwissKnife.dataprep import (
    generate_individual_mouse_data,
    get_primate_identification_data,
)
from SwissKnife.model import Model
from SwissKnife.segmentation import mold_image
from SwissKnife.utils import (
    Metrics,
    check_directory,
    get_callbacks,
    load_config,
    rescale_img,
    set_random_seed,
    setGPU,
    mask_image,
)

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
vid_basepath = "/media/nexus/storage1/swissknife_data/primate/raw_videos/2018_merge/"


def evaluate_on_data(
    species, network, video=None, config=None, exclude_hard=False, masking=False
):
    if species == "mouse_crossday":
        # x_train, y_train, x_test, y_test = get_individual_mouse_data()

        x_train, y_train, x_test, y_test = generate_individual_mouse_data(
            animal_lim=8, cv_folds=5, fold=0, day=1, masking=masking
        )

        dataloader = Dataloader(
            x_train, y_train, x_test, y_test, look_back=config["look_back"]
        )

        # FIXME: remove?
        dataloader.change_dtype()

        # preproc labels
        print("encoding")
        dataloader.encode_labels()

        dataloader.expand_dims()

        dataloader.create_recurrent_data()
        dataloader.create_flattened_data()

        our_model = Model()
        our_model.load_recognition_model(network)
        res = our_model.predict(dataloader.x_test)

        plt.imshow(dataloader.x_test[10, :, :, 0])
        plt.show()
        # print('max', max(dataloader.x_test[10,:,:,0]).flatten())
        # print('min', min(dataloader.x_test[10, :, :, 0]).flatten())
        metric = metrics.balanced_accuracy_score(
            # res, np.argmax(dataloader.y_test, axis=-1)
            res,
            dataloader.y_test,
        )
        print("Result", str(metric))

    if species == "primate":

        print(video_train)
        print("preparing data")
        X, y, vidlist = get_primate_identification_data(scaled=True)
        num_classes = 4

        print("before ,", str(len(X[0])))

        if exclude_hard:
            hard_list = "/media/nexus/storage5/swissknife_data/primate/identification_inputs/hard_list.pkl"
            with open(hard_list, "rb") as handle:
                excludes = pickle.load(handle)
            print(excludes)
            for vid_idx, elements in enumerate(excludes):
                X[vid_idx] = X[vid_idx][elements]
                y[vid_idx] = y[vid_idx][elements]

        print("after ,", str(len(X[0])))

        results = []
        videos = np.unique(vidlist)
        print(video)
        print(videos)
        idxs_list = []
        for vidname in videos:
            if video not in vidname:
                continue
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            for idx, el in enumerate(vidlist):
                if vidname == el:
                    x_test.append(X[idx])
                    y_test.append(y[idx])
                    idxs_list.append(len(y[idx]))
                else:
                    x_train.append(X[idx])
                    y_train.append(y[idx])
            x_train = np.vstack(x_train)
            y_train = np.hstack(y_train)
            x_test = np.vstack(x_test)
            y_test = np.hstack(y_test)

            dataloader = Dataloader(
                x_train[:, 4, :, :, :],
                y_train,
                x_test[:, 4, :, :, :],
                y_test,
            )

            dataloader.categorize_data(num_classes=num_classes)
            print("data preparation done")

            our_model = Model()
            our_model.load_recognition_model(network)
            res = our_model.predict(dataloader.x_test)
            results.append(
                [
                    "SIPEC_recognition",
                    vidname,
                    metrics.accuracy_score(res, np.argmax(dataloader.y_test, axis=-1)),
                    metrics.f1_score(
                        res, np.argmax(dataloader.y_test, axis=-1), average="macro"
                    ),
                ]
            )
            print(metrics.confusion_matrix(res, np.argmax(dataloader.y_test, axis=-1)))
            print("Mismatches")
            print(vidname)
            print(len(dataloader.x_test))
            equal = res == np.argmax(dataloader.y_test, axis=-1)
            print(np.where(equal == 0))

        print("FINAL results")
        print(results)


def train_on_data(
    species,
    network,
    config,
    results_sink,
    video=None,
    fraction=None,
    cv_folds=None,
    fold=None,
    masking=False,
):
    results = []

    if species == "primate":
        print("preparing data")
        X, y, vidlist = get_primate_identification_data(scaled=True)
        print(vidlist)
        results = []
        _results_sink = results_sink

        results_sink = _results_sink + video + "/"
        check_directory(results_sink)
        num_classes = 4

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for idx, el in enumerate(vidlist):
            if video == el:
                x_test.append(X[idx])
                y_test.append(y[idx])
            else:
                x_train.append(X[idx])
                y_train.append(y[idx])
        x_train = np.vstack(x_train)
        y_train = np.hstack(y_train)
        x_test = np.vstack(x_test)
        y_test = np.hstack(y_test)

        dataloader = Dataloader(
            x_train[:, 4, :, :, :],
            y_train,
            x_test[:, 4, :, :, :],
            y_test,
            config=config,
        )

        dataloader.x_train_recurrent = x_train
        dataloader.y_train_recurrent = y_train
        dataloader.x_test_recurrent = x_test
        dataloader.y_test_recurrent = y_test

        # TODO: doesn't work here
        dataloader.downscale_frames()
        dataloader.change_dtype()

        print("data preparation done")

    if species == "mouse":
        num_classes = 8

        # x_train, y_train, x_test, y_test = get_individual_mouse_data()

        x_train, y_train, x_test, y_test = generate_individual_mouse_data(
            animal_lim=num_classes, cv_folds=cv_folds, fold=fold, masking=masking
        )

        dataloader = Dataloader(x_train, y_train, x_test, y_test, config=config)

        plt.imshow(dataloader.x_test[5, :, :])
        plt.show()

        # FIXME: remove?
        dataloader.change_dtype()

        # dataloader.normalize_data()

        # preproc labels
        print("encoding")
        dataloader.encode_labels()

        dataloader.expand_dims()

        video = "mouse"

        # dataloader.undersample_data()

    if network == "ours" or network == "both":
        our_model = Model()

        class_weights = None
        if config["use_class_weights"]:
            from sklearn.utils import class_weight

            class_weights = class_weight.compute_class_weight(
                "balanced", np.unique(y_train), y_train
            )
            our_model.set_class_weight(class_weights)

        if config["undersample"]:
            print("undersampling")
            dataloader.undersample_data()

        if config["is_test"]:
            # dataloader.undersample_data()
            dataloader.decimate_labels(percentage=0.33)

        dataloader.categorize_data(num_classes=num_classes)

        # todo: change pos
        if species == "mouse":
            dataloader.create_recurrent_data()
            dataloader.create_flattened_data()

        if fraction is not None:
            dataloader.decimate_labels(percentage=fraction)
            print("Training data recuded to: ", str(len(dataloader.x_train)))

        # chose recognition model
        our_model.set_recognition_model(
            architecture=config["recognition_backbone"],
            input_shape=dataloader.get_input_shape(),
            num_classes=num_classes,
        )

        # set optimizer
        our_model.set_optimizer(
            config["recognition_model_optimizer"],
            lr=config["recognition_model_lr"],
        )

        ## Define callbacks
        # use lr scheduler
        if config["recognition_model_use_scheduler"]:
            our_model.scheduler_lr = config["recognition_model_scheduler_lr"]
            our_model.scheduler_factor = config["recognition_model_scheduler_factor"]
            our_model.set_lr_scheduler()
        else:
            # use standard training callback
            CB_es, CB_lr = get_callbacks()
            our_model.add_callbacks([CB_es, CB_lr])

        # add sklearn metrics for tracking in training
        my_metrics = Metrics(validation_data=(dataloader.x_test, dataloader.y_test))
        my_metrics.setModel(our_model.recognition_model)
        our_model.add_callbacks([my_metrics])

        if species == "primate":
            augmentation = primate_identification(level=config["augmentation_level"])
        elif species == "mouse":
            augmentation = mouse_identification(level=config["augmentation_level"])
        else:
            raise NotImplementedError
        if config["recognition_model_augmentation"]:
            our_model.set_augmentation(augmentation)

        our_model.recognition_model_batch_size = config["recognition_model_batch_size"]
        our_model.recognition_model_epochs = config["recognition_model_epochs"]

        if config["train_recognition_model"]:
            # start training of recognition network
            our_model.recognition_model_loss = config["recognition_model_loss"]
            our_model.train_recognition_network(dataloader=dataloader)
            our_model.recognition_model.save(
                results_sink + "IDnet_" + video + "_recognitionNet" + ".h5"
            )

            res = our_model.predict(dataloader.x_test)
            res = np.argmax(res[0], axis=-1)
            results.append(
                [
                    "SIPEC_recognition",
                    video,
                    fraction,
                    metrics.balanced_accuracy_score(
                        res, np.argmax(dataloader.y_test, axis=-1)
                    ),
                    metrics.f1_score(
                        res,
                        np.argmax(dataloader.y_test, axis=-1),
                        average="macro",
                    ),
                ]
            )
            print(results[-1])

        if config["train_sequential_model"]:
            our_model.fix_recognition_layers()
            our_model.remove_classification_layers()

            our_model.sequential_model_loss = config["sequential_model_loss"]

            # TODO: prettify me!
            if species == "primate":
                dataloader.categorize_data(num_classes=num_classes, recurrent=True)

            print("input shape", dataloader.get_input_shape())

            our_model.set_sequential_model(
                architecture=config["sequential_backbone"],
                input_shape=dataloader.get_input_shape(recurrent=True),
                num_classes=num_classes,
            )
            my_metrics.setModel(our_model.sequential_model)
            my_metrics.validation_data = (
                dataloader.x_test_recurrent,
                dataloader.y_test_recurrent,
            )
            our_model.add_callbacks([my_metrics])
            our_model.set_optimizer(
                config["sequential_model_optimizer"],
                lr=config["sequential_model_lr"],
            )
            if config["sequential_model_use_scheduler"]:
                our_model.scheduler_lr = config["sequential_model_scheduler_lr"]
                our_model.scheduler_factor = config["sequential_model_scheduler_factor"]
                our_model.set_lr_scheduler()

            our_model.sequential_model_epochs = config["sequential_model_epochs"]
            our_model.sequential_model_batch_size = config[
                "sequential_model_batch_size"
            ]

            CB_es, CB_lr = get_callbacks()
            # CB_train = [CB_lr, CB_es]
            # our_model.add_callbacks(CB_train)

            our_model.train_sequential_network(dataloader=dataloader)
            print(our_model.sequential_model.summary())

            res = our_model.predict_sequential(dataloader.x_test_recurrent)
            res = np.argmax(res[0], axis=-1)
            results.append(
                [
                    "SIPEC_sequential",
                    video,
                    fraction,
                    metrics.balanced_accuracy_score(
                        res, np.argmax(dataloader.y_test_recurrent, axis=-1)
                    ),
                    metrics.f1_score(
                        res,
                        np.argmax(dataloader.y_test_recurrent, axis=-1),
                        average="macro",
                    ),
                ]
            )
            print(results[-1])
            # our_model.sequential_model.sample_weights(
            #     results_sink + "IDnet_" + video + "_sequentialNet" + ".h5"
            # )

    if network in ("idtracker", "both"):
        dataloader.categorize_data(num_classes=num_classes)

        our_model = Model()
        # Comparison to Idtracker.ai network for animal identification
        idtracker = idtracker_ai(dataloader.get_input_shape(), num_classes)
        our_model.recognition_model = idtracker

        # Supplementary
        # optimizer default SGD, but also adam, test both
        our_model.set_optimizer("sgd", lr=0.0001)

        our_model.recognition_model_epochs = 100
        our_model.recognition_model_batch_size = 64

        CB_es = get_callbacks(patience=10, min_delta=0.05, reduce=False)
        our_model.add_callbacks([CB_es])

        my_metrics = Metrics()
        my_metrics.setModel(our_model.recognition_model)
        my_metrics.validation_data = (dataloader.x_test, dataloader.y_test)
        our_model.add_callbacks([my_metrics])

        our_model.train_recognition_network(dataloader=dataloader)
        # our_model.recognition_model.save('IdTracker_' + vidname + '_recognitionNet' + '.h5')
        res = our_model.predict(dataloader.x_test)
        results.append(
            [
                "IdTracker",
                video,
                fraction,
                metrics.balanced_accuracy_score(
                    res, np.argmax(dataloader.y_test, axis=-1)
                ),
                metrics.f1_score(
                    res, np.argmax(dataloader.y_test, axis=-1), average="macro"
                ),
            ]
        )

    if network == "shuffle":

        res = list(dataloader.y_test)
        random.shuffle(res)
        res = np.asarray(res)
        results.append(
            [
                "shuffle",
                video,
                fraction,
                metrics.balanced_accuracy_score(
                    # res, np.argmax(dataloader.y_test, axis=-1)
                    res,
                    dataloader.y_test,
                ),
                metrics.f1_score(
                    # res, np.argmax(dataloader.y_test, axis=-1), average="macro"
                    res,
                    dataloader.y_test,
                    average="macro",
                ),
            ]
        )

        print(results)

    np.save(
        results_sink + "results_df",
        results,
        allow_pickle=True,
    )


def idresults_to_training_recurrent(
    idresults, fnames_base, video, index, masking=True, mask_size=128, rescaling=False
):
    multi_imgs_x = []
    multi_imgs_y = []

    skipped = 0

    batch_size = 10000

    offset = index * batch_size

    # for 1024 is 128

    for el in idresults.keys():
        print(str(el))
        if el < 100:
            continue
        #     print(el)

        # older 1024, 1024 version
        #     fnames_base = '/media/nexus/storage1/swissknife_data/primate/inference/segmentation/20180115T150502-20180115T150902_%T1/frames/'

        #         frame = fnames_base + 'frames/' + 'frame_' + str(el) + '.npy'

        masks = idresults[el]["masks"]["masks"]
        boxes = idresults[el]["masks"]["rois"]

        for ids in range(0, min(masks.shape[-1], 4)):

            #             # TODO FIX
            #             try:
            #                 image = np.load(frame)
            #             except FileNotFoundError:
            #                 continue
            #             image = idresults[el]['frame']
            # if rescaling:
            if False:
                com = [
                    int(
                        (
                            idresults[el]["masks"]["rois"][ids][0]
                            + idresults[el]["masks"]["rois"][ids][2]
                        )
                        / 2
                    )
                    * 2,
                    int(
                        (
                            idresults[el]["masks"]["rois"][ids][1]
                            + idresults[el]["masks"]["rois"][ids][3]
                        )
                        / 2
                    )
                    * 2,
                ]
                mask = rescale(masks[:, :, ids], 0.5)
            #         mask = rescale(masks[:,:,ids], 1.0)

            else:

                # normal one
                com = [
                    int(
                        (
                            idresults[el]["masks"]["rois"][ids][0]
                            + idresults[el]["masks"]["rois"][ids][2]
                        )
                        / 2
                    ),
                    int(
                        (
                            idresults[el]["masks"]["rois"][ids][1]
                            + idresults[el]["masks"]["rois"][ids][3]
                        )
                        / 2
                    ),
                ]
                mask = masks[:, :, ids]
                mybox = boxes[ids, :]

            #         multi_imgs_y.append(int(idresults[el]['results'][ids][0])-1)

            # TODO: fixme
            if el in (2220, 2395, 9601):
                print(el)
                skipped += 1
                continue

            try:

                if (
                    idresults[el]["results"][ids].split("_")[0] in classes
                    and idresults[el]["results"][ids].split("_")[1] == "easy"
                ):

                    images = []

                    spacing = 3

                    dil_scaling = 5

                    if rescaling:

                        for j in range(-3, 4):
                            print("jjjjjj", j)
                            img = mold_image(video.get_data(offset + el + j * spacing))
                            print("image shape", img.shape)
                            rescaled_img = rescale_img(mybox, img)
                            images.append(rescaled_img)

                        res_images = images

                    elif masking:

                        for j in range(-3, 4):
                            images.append(
                                mask_image(
                                    mold_image(
                                        video.get_data(offset + el + j * spacing)
                                    ),
                                    mask,
                                    dilation_factor=100 * dil_scaling,
                                )
                            )

                        res_images = []

                        for image in images:

                            img = np.zeros((int(2 * mask_size), int(2 * mask_size), 3))
                            img_help = image[
                                max(com[0] - mask_size, 0) : min(
                                    com[0] + mask_size, 2048
                                ),
                                max(com[1] - mask_size, 0) : min(
                                    com[1] + mask_size, 2048
                                ),
                            ]

                            le = int(img_help.shape[0] / 2)
                            ri = img_help.shape[0] - le
                            up = int(img_help.shape[1] / 2)
                            do = img_help.shape[1] - up
                            img[
                                mask_size - le : mask_size + ri,
                                mask_size - up : mask_size + do,
                                :,
                            ] = img_help
                            img = img.astype("uint8")

                            #             break
                            if img.shape != (
                                int(2 * mask_size),
                                int(2 * mask_size),
                                3,
                            ):
                                skipped += 1
                                print(el)
                                continue

                            res_images.append(img)

                    else:
                        for j in range(-3, 4):
                            images.append(
                                mold_image(video.get_data(offset + el + j * spacing))
                            )

                        res_images = []

                        for image in images:

                            img = np.zeros((int(2 * mask_size), int(2 * mask_size), 3))
                            img_help = image[
                                max(com[0] - mask_size, 0) : min(
                                    com[0] + mask_size, 2048
                                ),
                                max(com[1] - mask_size, 0) : min(
                                    com[1] + mask_size, 2048
                                ),
                            ]

                            le = int(img_help.shape[0] / 2)
                            ri = img_help.shape[0] - le
                            up = int(img_help.shape[1] / 2)
                            do = img_help.shape[1] - up
                            img[
                                mask_size - le : mask_size + ri,
                                mask_size - up : mask_size + do,
                                :,
                            ] = img_help
                            img = img.astype("uint8")

                            #             break
                            if img.shape != (
                                int(2 * mask_size),
                                int(2 * mask_size),
                                3,
                            ):
                                skipped += 1
                                print(el)
                                continue

                            res_images.append(img)

                    res_images = np.asarray(res_images)
                    multi_imgs_y.append(
                        classes[idresults[el]["results"][ids].split("_")[0]]
                    )
                    multi_imgs_x.append(res_images)
            except KeyError:
                continue

    X = np.asarray(multi_imgs_x)
    y = np.asarray(multi_imgs_y)

    return X, y


def load_vid(basepath, vid, idx, batch_size=10000):
    videodata = skvideo.io.vread(basepath + vid + ".mp4", as_grey=False)
    videodata = videodata[idx * batch_size : (idx + 1) * batch_size]
    results_list = Parallel(
        n_jobs=20, max_nbytes=None, backend="multiprocessing", verbose=40
    )(delayed(mold_image)(image) for image in videodata)
    results = {}
    for idx, el in enumerate(results_list):
        results[idx] = el

    return results


def vid_to_xy(video):
    video = video.split("/")[-1].split("IDresults_")[-1].split(".np")[0]
    vid = video.split(".npy")[0][:-2]
    #TODO: Check, vidlist is not defined!
    vidlist.append(vid)
    idx = int(video.split(".npy")[0][-1:])
    idx -= 1

    idresults = np.load(
        idresults_base + "IDresults_" + video + ".npy", allow_pickle=True
    ).item()
    pat = vid_basepath + vid + ".mp4"
    print(pat)
    vid = imageio.get_reader(pat, "ffmpeg")
    #     vid = load_vid(vid_basepath,vid,idx)

    _X, _y = idresults_to_training_recurrent(
        idresults, fnames_base + video + "/", vid, idx, mask_size=mask_size
    )

    return [_X, _y]


def main():
    args = parser.parse_args()
    operation = args.operation
    network = args.network
    gpu_name = args.gpu
    config_name = args.config
    video = args.video
    fraction = args.fraction
    cv_folds = args.cv_folds
    fold = args.fold
    nw_path = args.nw_path

    config = load_config("../configs/identification/" + config_name)

    config["use_generator"] = False

    set_random_seed(config["random_seed"])
    setGPU(gpu_name=gpu_name)

    # TODO: Get rid of hardcoded paths 
    if operation == "train_primate_cv":
        results_sink = (
            "/media/nexus/storage5/swissknife_results/identification/"
            + "primate/"
            + config["experiment_name"]
            + "_"
            + network
            + "_CV_"
            + "fraction_"
            + str(fraction)
            + "_"
            + datetime.now().strftime("%Y-%m-%d-%H_%M")
            + "/"
        )
        check_directory(results_sink)
        X, y, vidlist = get_primate_identification_data(scaled=True)
        videos = np.unique(vidlist)
        print(videos)
        for video in videos:
            print("VIDEO")
            print(video)
            train_on_data(
                species="primate",
                network=network,
                config=config,
                results_sink=results_sink,
                video=video,
                fraction=fraction,
            )

    if operation == "train_primate":
        results_sink = (
            "/media/nexus/storage5/swissknife_results/identification/"
            + "primate/"
            + config["experiment_name"]
            + "_"
            + network
            + "_"
            + datetime.now().strftime("%Y-%m-%d-%H_%M")
            + "/"
        )
        check_directory(results_sink)

        train_on_data(
            species="primate",
            network=network,
            config=config,
            results_sink=results_sink,
            video=video,
            fraction=fraction,
            masking=config["masking"],
        )
    if operation == "train_mouse":
        results_sink = (
            # "/media/nexus/storage5/swissknife_results/identification/"
            "/media/nexus/storage5/swissknife_results/identification_masked/"
            + "mouse/"
            + config["experiment_name"]
            + "_"
            + network
            + "_"
            + str(fraction)
            + "_fold_"
            + str(fold)
            + datetime.now().strftime("%Y-%m-%d-%H_%M")
            + "/"
        )
        # check_directory(results_sink)
        train_on_data(
            species="mouse",
            network=network,
            config=config,
            results_sink=results_sink,
            fraction=fraction,
            cv_folds=cv_folds,
            fold=fold,
            masking=config["masking"],
        )
    if operation == "evaluate_primate":
        evaluate_on_data(
            species="primate",
            # network="../results/identification/"
            # network = "/media/nexus/storage5/swissknife_results/identification/primate/identification_full_CV_2020-06-18-00_54/"
            network="/media/nexus/storage5/swissknife_results/identification/old/identification_full_ours_CV_fraction_1.0_2020-07-13-10_33/"
            + video
            + "/IDnet_"
            + video
            + "_recognitionNet.h5",
            video=video,
            exclude_hard=True,
        )

    if operation == "evaluate_mouse_multi":
        evaluate_on_data(
            species="mouse_crossday",
            network=nw_path,
            config=config,
            masking=config["masking"],
        )

    # save config
    with open(results_sink + "config.json", "w") as f:
        json.dump(config, f)
    f.close()

    print("DONE")


parser = ArgumentParser()
parser.add_argument(
    "--config",
    action="store",
    dest="config",
    type=str,
    default="identification_config",
    help="config for specifying training params",
)
parser.add_argument(
    "--video",
    action="store",
    dest="video",
    type=str,
    default=None,
    help="which video to train/infer on",
)
parser.add_argument(
    "--network",
    action="store",
    dest="network",
    type=str,
    default="ours",
    help="which network used for training",
)
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
    default=None,
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

parser.add_argument(
    "--fold",
    action="store",
    dest="fold",
    type=int,
    default=None,
    help="fold for crossvalidation",
)

parser.add_argument(
    "--cv_folds",
    action="store",
    dest="cv_folds",
    type=int,
    default=0,
    help="folds for cross validation",
)

parser.add_argument(
    "--nw_path",
    action="store",
    dest="nw_path",
    type=str,
    default=None,
    help="network used for evaluation",
)

if __name__ == "__main__":
    main()
