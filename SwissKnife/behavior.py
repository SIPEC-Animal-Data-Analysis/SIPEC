# SIPEC
# MARKUS MARKS
# Behavioral Classification
import sys
import os

# from scipy.misc import imresize
from tqdm import tqdm
import pandas as pd
import random
from datetime import datetime

from SwissKnife.architectures import classification_small

from argparse import ArgumentParser
import keras.backend as K
import numpy as np

from sklearn import metrics
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold

from SwissKnife.utils import (
    setGPU,
    Metrics,
    get_callbacks,
    load_vgg_labels,
    loadVideo,
    load_config,
    check_directory,
)
from SwissKnife.dataloader import Dataloader
from SwissKnife.model import Model


def train_behavior(
    dataloader,
    config,
    num_classes,
    encode_labels=True,
    class_weights=None,
    # results_sink=results_sink,
):

    print("data prepared!")

    our_model = Model()

    our_model.recognition_model = classification_small(
        input_shape=dataloader.get_input_shape(), num_classes=num_classes
    )
    our_model.set_class_weight(class_weights)

    our_model.set_optimizer(
        config["recognition_model_optimizer"], lr=config["recognition_model_lr"],
    )
    if config["recognition_model_use_scheduler"]:
        our_model.scheduler_lr = config["recognition_model_scheduler_lr"]
        our_model.scheduler_factor = config["recognition_model_scheduler_factor"]
        our_model.set_lr_scheduler()
    else:
        # use standard training callback
        CB_es, CB_lr = get_callbacks()
        our_model.add_callbacks([CB_es, CB_lr])

    # add sklearn metrics for tracking in training
    my_metrics = Metrics()
    my_metrics.setModel(our_model.recognition_model)
    our_model.add_callbacks([my_metrics])

    if config["train_recognition_model"]:
        our_model.recognition_model_epochs = config["recognition_model_epochs"]
        our_model.recognition_model_batch_size = config["recognition_model_batch_size"]
        print(config["recognition_model_batch_size"])
        our_model.train_recognition_network(dataloader=dataloader)
        print(config)

    if config["train_sequential_model"]:
        # if False:
        if config["recognition_model_fix"]:
            our_model.fix_recognition_layers()
        if config["recognition_model_remove_classification"]:
            our_model.remove_classification_layers()
        our_model.set_sequential_model(
            architecture=config["sequential_backbone"],
            input_shape=dataloader.get_input_shape(recurrent=True),
            num_classes=num_classes,
        )
        my_metrics.setModel(our_model.sequential_model)
        our_model.add_callbacks([my_metrics])
        our_model.set_optimizer(
            config["sequential_model_optimizer"], lr=config["sequential_model_lr"],
        )
        if config["sequential_model_use_scheduler"]:
            our_model.scheduler_lr = config["sequential_model_scheduler_lr"]
            our_model.scheduler_factor = config["sequential_model_scheduler_factor"]
            our_model.set_lr_scheduler()

        our_model.sequential_model_epochs = config["sequential_model_epochs"]
        our_model.sequential_model_batch_size = config["sequential_model_batch_size"]
        print(our_model.sequential_model.summary())
        our_model.train_sequential_network(dataloader=dataloader)

    print(config)

    res = our_model.predict(dataloader.x_test, model="recognition")
    acc = metrics.balanced_accuracy_score(res, np.argmax(dataloader.y_test, axis=-1))
    f1 = metrics.f1_score(res, np.argmax(dataloader.y_test, axis=-1), average="macro")

    corr = pearsonr(res, np.argmax(dataloader.y_test, axis=-1))[0]
    report = metrics.classification_report(res, np.argmax(dataloader.y_test, axis=-1),)
    return [acc, f1, corr], report


def train_primate(config, results_sink, shuffle):

    basepath = "/media/nexus/storage5/swissknife_data/primate/behavior/"

    vids = [
        basepath + "fullvids_20180124T113800-20180124T115800_%T1_0.mp4",
        basepath + "fullvids_20180124T113800-20180124T115800_%T1_1.mp4",
        basepath + "20180116T135000-20180116T142000_social_complete.mp4",
        basepath + "20180124T115800-20180124T122800b_0_complete.mp4",
        basepath + "20180124T115800-20180124T122800b_1_complete.mp4",
    ]
    all_annotations = [
        basepath + "20180124T113800-20180124T115800_0.csv",
        basepath + "20180124T113800-20180124T115800_1.csv",
        basepath + "20180116T135000-20180116T142000_social_complete.csv",
        basepath + "20180124T115800-20180124T122800b_0_complete.csv",
        basepath + "20180124T115800-20180124T122800b_1_complete.csv",
    ]

    all_vids = []
    for vid in vids:
        myvid = loadVideo(vid, greyscale=False)
        if "social" in vid:
            im_re = []
            for el in tqdm(myvid):
                im_re.append(imresize(el, 0.5))
            myvid = np.asarray(im_re)
        all_vids.append(myvid)
    vid = np.vstack(all_vids)

    # #TODO: test here w/without
    new_vid = []
    for el in tqdm(vid):
        new_vid.append(imresize(el, 0.5))
    vid = np.asarray(new_vid)

    labels = []
    labels_idxs = []
    for annot_idx, annotation in enumerate(all_annotations):
        annotation = pd.read_csv(annotation, error_bad_lines=False, header=9)
        annotation = load_vgg_labels(
            annotation, video_length=len(all_vids[annot_idx]), framerate_video=25
        )
        labels = labels + annotation
        labels_idxs.append(annotation)
    idxs = []
    for idx, img in enumerate(vid):
        if max(img.flatten()) == 0:
            # print('black image')
            pass
        else:
            idxs.append(idx)
    idxs = np.asarray(idxs)

    global groups

    groups = (
        [0] * len(labels_idxs[0])
        + [0] * len(labels_idxs[1])
        + [3] * len(labels_idxs[2])
        + [4] * len(labels_idxs[3])
        + [4] * len(labels_idxs[4])
    )

    groups = groups
    vid = vid
    labels = labels

    groups = [groups[i] for i in idxs]
    labels = [labels[i] for i in idxs]
    vid = [vid[i] for i in idxs]
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    vid = np.asarray(vid)

    num_splits = 5
    # TODO: prettify me!
    sss = StratifiedKFold(n_splits=num_splits, random_state=0, shuffle=False)
    print("shuffle")

    y = labels
    print("Classes")
    print(np.unique(y))

    X = list(range(0, len(labels)))
    X = np.asarray(X)
    X = np.expand_dims(X, axis=-1)

    results = []
    reports = []

    for split in range(0, num_splits):

        tr_idx = None
        tt_idx = None
        idx = 0
        print(split)
        for train_index, test_index in sss.split(X, y, groups=groups):
            if idx == split:
                tr_idx = train_index
                tt_idx = test_index
            idx += 1

        y_train = y[tr_idx]
        y_test = y[tt_idx]
        x_train = vid[tr_idx]
        x_test = vid[tt_idx]

        dataloader = Dataloader(
            x_train, y_train, x_test, y_test, look_back=config["look_back"]
        )

        # config_name = 'primate_' + str(1)
        #
        # config = load_config("../configs/behavior/primate/" + config_name)
        config["recognition_model_batch_size"] = 128
        config["backbone"] = "imagenet"
        print(config)

        encode_labels = True
        num_classes = config["num_classes"]

        print("preparing data")
        # TODO: adjust

        dataloader.change_dtype()
        print("dtype changed")

        dataloader.remove_behavior(behavior="walking")

        if config["normalize_data"]:
            dataloader.normalize_data()
        if encode_labels:
            dataloader.encode_labels()
        print("labels encoded")

        if shuffle:

            res = list(dataloader.y_test)
            random.shuffle(res)
            res = np.asarray(res)
            results.append(
                [
                    "shuffle",
                    "bla",
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
            report = metrics.classification_report(res, dataloader.y_test,)

            print(results)
        else:
            class_weights = None
            if config["use_class_weights"]:
                print("calc class weights")
                from sklearn.utils import class_weight

                class_weights = class_weight.compute_class_weight(
                    "balanced", np.unique(dataloader.y_train), dataloader.y_train
                )

            # if config["undersample_data"]:
            print("undersampling data")
            # dataloader.undersample_data()

            print("preparing recurrent data")
            dataloader.create_recurrent_data()
            print("preparing flattened data")
            # dataloader.create_flattened_data()

            print("categorize data")
            dataloader.categorize_data(num_classes, recurrent=True)

            print("data ready")

            # if operation == "train":
            res, report = train_behavior(
                dataloader,
                config,
                num_classes=config["num_classes"],
                class_weights=class_weights,
            )

        print(config)
        print("DONE")
        print(report)
        np.save(
            results_sink + "results.npy", res,
        )
        np.save(
            results_sink + "reports.npy", report,
        )


def main():
    args = parser.parse_args()
    operation = args.operation
    gpu_name = args.gpu
    config_name = args.config_name
    network = args.network
    shuffle = args.shuffle

    setGPU(K, gpu_name)

    results_sink = (
        "./results/primate/behavior"
        + "_"
        + network
        + "_"
        + datetime.now().strftime("%Y-%m-%d-%H_%M")
        + "/"
    )

    check_directory(results_sink)

    if operation("train_primate"):
        config_name = "primate_final"
        config = load_config("../configs/behavior/primate/" + config_name)
        train_primate(config=config, results_sink=results_sink, shuffle=shuffle)
    else:
        config = load_config(config_name)
        train_behavior(config=config, results_sink=results_sink)


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
    "--config_name",
    action="store",
    dest="config_name",
    type=str,
    default="behavior_config_baseline",
    help="behavioral config to use",
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
    "--shuffle", action="store", dest="shuffle", type=bool, default=False,
)

if __name__ == "__main__":
    main()
