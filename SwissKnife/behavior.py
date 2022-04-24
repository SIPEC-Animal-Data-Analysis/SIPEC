# SIPEC
# MARKUS MARKS
# Behavioral Classification

import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.externals._pilutil import imresize
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from SwissKnife.architectures import (
    pretrained_recognition,
)
from SwissKnife.dataloader import Dataloader, DataGenerator
from SwissKnife.model import Model
from SwissKnife.utils import (
    setGPU,
    Metrics,
    load_vgg_labels,
    loadVideo,
    load_config,
    check_directory,
    callbacks_learningRate_plateau,
)


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

    our_model.recognition_model = pretrained_recognition(
        config["backbone"],
        dataloader.get_input_shape(),
        num_classes,
        fix_layers=False,
        skip_layers=True,
    )

    our_model.set_class_weight(class_weights)

    our_model.set_optimizer(
        config["recognition_model_optimizer"],
        lr=config["recognition_model_lr"],
    )
    if config["recognition_model_use_scheduler"]:
        our_model.scheduler_lr = config["recognition_model_scheduler_lr"]
        our_model.scheduler_factor = config["recognition_model_scheduler_factor"]
        our_model.set_lr_scheduler()
    else:
        # use standard training callback
        CB_es, CB_lr = callbacks_learningRate_plateau()
        our_model.add_callbacks([CB_es, CB_lr])

    # add sklearn metrics for tracking in training
    validation_data = (dataloader.x_test, dataloader.y_test)
    my_metrics = Metrics(validation_data)
    my_metrics.setModel(our_model.recognition_model)
    our_model.add_callbacks([my_metrics])

    if config["train_recognition_model"]:
        if dataloader.config["use_generator"]:
            dataloader.training_generator = DataGenerator(
                x_train=dataloader.x_train,
                y_train=dataloader.y_train,
                look_back=dataloader.config["look_back"],
                batch_size=config["recognition_model_batch_size"],
                type="recognition",
            )
            dataloader.validation_generator = DataGenerator(
                x_train=dataloader.x_test,
                y_train=dataloader.y_test,
                look_back=dataloader.config["look_back"],
                batch_size=config["recognition_model_batch_size"],
                type="recognition",
            )
        our_model.recognition_model_epochs = config["recognition_model_epochs"]
        our_model.recognition_model_batch_size = config["recognition_model_batch_size"]
        print()
        our_model.train_recognition_network(dataloader=dataloader)
        print(config)

    res = our_model.predict(dataloader.x_test, model="recognition")
    np.save('predictions_recognition.npy', res)
    print('safed predictions')

    if config["train_sequential_model"]:
        if dataloader.config["use_generator"]:
            dataloader.training_generator = DataGenerator(
                x_train=dataloader.x_train,
                y_train=dataloader.y_train,
                look_back=dataloader.config["look_back"],
                batch_size=32,
                type="sequential",
            )
            dataloader.validation_generator = DataGenerator(
                x_train=dataloader.x_test,
                y_train=dataloader.y_test,
                look_back=dataloader.config["look_back"],
                batch_size=config["recognition_model_batch_size"],
                type="sequential",
            )
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
        our_model.sequential_model_batch_size = config["sequential_model_batch_size"]
        print(our_model.sequential_model.summary())
        our_model.train_sequential_network(dataloader=dataloader)

    print(config)

    print("evaluating")
    res = []
    batches = len(dataloader.x_test)
    batches = int(batches / config["sequential_model_batch_size"])
    test_gt = []
    # TODO: fix -1 to really use all VAL data
    for idx in tqdm(range(batches - 1)):
        if config["train_sequential_model"]:
            eval_batch = []
            for i in range(config["sequential_model_batch_size"]):
                new_idx = (
                    (idx * config["sequential_model_batch_size"])
                    + i
                    + dataloader.look_back
                )
                data = dataloader.x_test[
                    new_idx - dataloader.look_back : new_idx + dataloader.look_back
                ]
                eval_batch.append(data)
                test_gt.append(dataloader.y_test[new_idx])
            eval_batch = np.asarray(eval_batch)
            prediction = our_model.predict(eval_batch, model="sequential")
        else:
            eval_batch = []
            for i in range(config["recognition_model_batch_size"]):
                new_idx = (idx * config["recognition_model_batch_size"]) + i
                data = dataloader.x_test[new_idx]
                eval_batch.append(data)
                test_gt.append(dataloader.y_test[new_idx])
            eval_batch = np.asarray(eval_batch)
            prediction = our_model.predict(eval_batch, model="recognition")
        for idx, el in enumerate(prediction):
            res.append(el)

    res = np.argmax(res, axis=-1).astype(int)
    test_gt = np.asarray(test_gt)

    acc = metrics.balanced_accuracy_score(res, np.argmax(test_gt, axis=-1))
    f1 = metrics.f1_score(res, np.argmax(test_gt, axis=-1), average="macro")
    #
    corr = pearsonr(res, np.argmax(test_gt, axis=-1))[0]
    report = metrics.classification_report(
        res,
        np.argmax(test_gt, axis=-1),
    )

    print(report)
    return our_model, [acc, f1, corr], report


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

        dataloader = Dataloader(x_train, y_train, x_test, y_test, config=config)

        # config_name = 'primate_' + str(1)
        #
        # config = load_config("../configs/behavior/primate/" + config_name)
        print(config)

        num_classes = config["num_classes"]

        print("preparing data")
        # TODO: adjust

        dataloader.change_dtype()
        print("dtype changed")

        dataloader.remove_behavior(behavior="walking")

        if config["normalize_data"]:
            dataloader.normalize_data()
        if config["encode_labels"]:
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
            report = metrics.classification_report(
                res,
                dataloader.y_test,
            )

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
            results_sink + "results.npy",
            res,
        )
        np.save(
            results_sink + "reports.npy",
            report,
        )


def main():
    args = parser.parse_args()
    operation = args.operation
    gpu_name = args.gpu
    config_name = args.config_name
    network = args.network
    shuffle = args.shuffle
    annotations = args.annotations
    video = args.video
    results_sink = args.results_sink
    only_flow = args.only_flow

    setGPU(gpu_name)
    check_directory(results_sink)

    if annotations:
        myvid = loadVideo(video, greyscale=False)
        annotation = pd.read_csv(annotations, error_bad_lines=False, header=9)
        annotation = load_vgg_labels(
            annotation, video_length=len(myvid), framerate_video=25
        )

        # Train test split
        split = int(len(myvid) * 0.8)
        x_train, x_test, y_train, y_test = (
            myvid[:split],
            myvid[split:],
            annotation[:split],
            annotation[split:],
        )

        # load cfg
        config = load_config("../configs/behavior/shared_config")
        beh_config = load_config("../configs/behavior/default")
        config.update(beh_config)
        print(config)

        num_classes = len(np.unique(annotation))
        dataloader = Dataloader(x_train, y_train, x_test, y_test, config)
        dataloader.prepare_data()
        train_behavior(dataloader=dataloader, num_classes=num_classes, config=config)

    elif operation == "train_primate":
        config_name = "primate_final"
        config = load_config("../configs/behavior/primate/" + config_name)
        beh_config = load_config("../configs/behavior/default")
        config.update(beh_config)
        print(config)
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
    default="default",
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
# TODO: check if folder and then load all files in folder, similar for vid files
parser.add_argument(
    "--annotations",
    action="store",
    dest="annotations",
    type=str,
    default=None,
    help="path for annotations from VGG annotator",
)
parser.add_argument(
    "--video",
    action="store",
    dest="video",
    type=str,
    default=None,
    help="path to folder with annotated video",
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
    "--only_flow",
    action="store",
    dest="only_flow",
    type=str,
    default=None,
    help="use_only_flow",
)
parser.add_argument(
    "--shuffle",
    action="store",
    dest="shuffle",
    type=bool,
    default=False,
)

# example usage
# python behavior.py --annotations "/media/nexus/storage5/swissknife_data/primate/behavior/20180124T113800-20180124T115800_0.csv" --video "/media/nexus/storage5/swissknife_data/primate/behavior/fullvids_20180124T113800-20180124T115800_%T1_0.mp4" --gpu 2
if __name__ == "__main__":
    main()
