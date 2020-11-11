# SIPEC
# MARKUS MARKS
# RUN FULL INFERENCE
import os
import sys
import operator

import cv2

from SwissKnife.masksmoothing import MaskMatcher
from SwissKnife.poseestimation import heatmap_to_scatter, custom_binary_crossentropy

sys.path.append("../")

from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from keras.engine.saving import load_model
import keras.backend as K
import keras.losses

# from SwissKnife.poseestimation import heatmap_to_scatter
from SwissKnife.segmentation import SegModel, mold_video
from SwissKnife.utils import (
    setGPU,
    loadVideo,
    masks_to_coms,
    apply_all_masks,
    detect_primate,
    check_directory,
    rescale_img,
    save_dict,
    masks_to_coords,
)


# TODO: save molded imgs?


def full_inference(
    videodata,
    results_sink,
    SegNet=None,
    IdNet=None,
    PoseNet=None,
    BehaveNet=None,
    mask_matching=False,
    id_matching=False,
    output_video=None,
):
    # classes
    classes = {
        "Charles": 0,
        "Max": 1,
        "Paul": 2,
        "Alan": 3,
    }

    maskmatcher = MaskMatcher()
    maskmatcher.max_ids = 6

    # invert classes / to go both ways
    classes_invert = [el for el in classes.keys()]

    # set threshold for detection of primate identities
    threshold = 0.5
    results = []

    if type(videodata) == str:
        length = len(os.listdir(videodata))

    # for idx, el in tqdm(enumerate(videodata)):
    for idx in range(length):
        el = cv2.imread(videodata + "frame%d.jpg" % idx)
        results_per_frame = {}
        molded_img, masks, boxes, mask_scores = SegNet.detect_image(
            el, verbose=0, mold=True
        )
        coms = masks_to_coms(masks)
        # TODO: fixme
        try:
            masked_imgs, masked_masks = apply_all_masks(
                masks, coms, molded_img, mask_size=128
            )
        except ValueError:
            results.append(0)
            continue

        if mask_matching:

            if not idx == 0:
                mapping = maskmatcher.match_masks(
                    boxes[: maskmatcher.max_ids], results[-1]["boxes"]
                )
                print(mapping)
                new_ids = maskmatcher.match_ids(
                    mapping, len(boxes[: maskmatcher.max_ids])
                )
                overlaps = [mapping[el][0] for el in mapping]
                if len(overlaps) < len(boxes):
                    for i in range(len(boxes) - len(overlaps)):
                        overlaps.append(0)
                if max(new_ids) > 0:
                    print("boxes before: ", str(boxes))
                    boxes = maskmatcher.map(mapping, boxes)
                    print("boxes after: ", str(boxes))
                    masks = np.swapaxes(masks, 0, 2)
                    masks = maskmatcher.map(mapping, masks)
                    masks = np.swapaxes(masks, 0, 2)
                    masked_imgs = maskmatcher.map(mapping, masked_imgs)
                    coms = maskmatcher.map(mapping, coms)
                    # overlaps = maskmatcher.map(mapping, overlaps)
                print(new_ids)
                results_per_frame["track_ids"] = new_ids
                results_per_frame["overalps"] = overlaps
                results_per_frame["mapping"] = mapping
            else:
                results_per_frame["track_ids"] = np.zeros(
                    (maskmatcher.max_ids,)
                ).astype("int")
                results_per_frame["overalps"] = np.zeros((maskmatcher.max_ids,)).astype(
                    "float"
                )

        mask_size = 256
        # mask_size = 128
        rescaled_imgs = []
        for box in boxes:
            if box[0] == 0:
                rescaled_imgs.append(np.zeros((mask_size, mask_size, 3)))
            else:
                rescaled_img = rescale_img(box, molded_img, mask_size=mask_size)
                rescaled_imgs.append(rescaled_img)
        rescaled_imgs = np.asarray(rescaled_imgs)

        # resulting_frames[idx] = molded_img.astype("uint8")
        results_per_frame["mask_coords"] = masks_to_coords(masks)
        results_per_frame["mask_scores"] = mask_scores
        results_per_frame["boxes"] = boxes
        results_per_frame["coms"] = np.asarray(coms)
        results_per_frame["masked_imgs"] = np.asarray(masked_imgs).astype("uint8")
        results_per_frame["masked_masks"] = masked_masks.astype("uint8")
        results_per_frame["rescaled_imgs"] = rescaled_imgs.astype("uint8")

        # maskmatch.sort ()

        # append IdNet results
        if IdNet:
            ids = []
            confidences = []
            for img in rescaled_imgs:
                primate, confidence = detect_primate(
                    img, IdNet, classes_invert, threshold
                )
                ids.append(primate)
                confidences.append(confidence)
            results_per_frame["ids"] = ids
            results_per_frame["confidences"] = confidences

        if PoseNet:
            maps = []
            for img in masked_imgs:
                heatmaps = PoseNet.predict(np.expand_dims(img, axis=0))
                heatmaps = heatmaps[0, :, :, :]
                coords_predict = heatmap_to_scatter(heatmaps)
                maps.append(coords_predict)
            results_per_frame["pose_coordinates"] = maps

        results.append(results_per_frame)

    if id_matching:
        for idx, el in tqdm(enumerate(videodata)):
            lookback = 150
            if not (lookback < idx < len(videodata) - lookback):
                results[idx]["smoothed_ids"] = ids
            else:
                corrected_ids = {}
                for i in range(len(ids)):
                    prev_ids = {}
                    # for j in range(-lookback, 0):
                    # TODOL forward backward filter
                    for j in range(-lookback, lookback):
                        try:
                            prev_id = results[idx + j]["track_ids"][i]
                            prev_names = results[idx + j]["ids"]
                            confidences = results[idx + j]["confidences"]
                            try:
                                # prev_ids[prev_names[i]] = prev_ids[prev_names[i]] + confidences[i] * (10/(np.abs(j)+1))
                                prev_ids[prev_names[i]].append(confidences[i])
                            except KeyError:
                                # prev_ids[prev_names[i]] = confidences[prev_id]
                                prev_ids[prev_names[i]] = [confidences[i]]
                        except IndexError:
                            continue
                    if prev_ids == {}:
                        corrected_ids[i] = ids[i]
                    else:
                        for el in prev_ids.keys():
                            prev_ids[el] = np.median(prev_ids[el])
                        # corrected_ids[i] = max(prev_ids.items(), key=operator.itemgetter(1))[0]
                        sorted_x = sorted(prev_ids.items(), key=operator.itemgetter(1))
                        corrected_ids[i] = sorted_x[-1][0]

                results[idx]["smoothed_ids"] = corrected_ids

    np.save(results_sink + "inference_results.npy", results, allow_pickle=True)
    # save_dict(
    #     results_sink + "inference_resulting_masks.pkl", resulting_masks,
    # )
    # save_dict(
    #     results_sink + "/inference_resulting_frames.pkl", resulting_frames,
    # )


def main():
    args = parser.parse_args()
    operation = args.operation
    gpu_name = args.gpu

    if gpu_name:
        setGPU(K, gpu_name)

    if operation == "primate":
        species = "primate"
        SegNet = SegModel(species=species)
        SegNet.inference_config.DETECTION_MIN_CONFIDENCE = 0.99
        # indoor network
        SegNet.set_inference(model_path="/home/nexus/mask_rcnn_primate_0119.h5")
        # all cam network
        # SegNet.set_inference(model_path="/media/nexus/storage5/swissknife_results/networks/mask_rcnn_primate_0400.h5")

        vidbase = "/media/nexus/storage5/swissknife_data/primate/raw_videos_sorted/2018_merge/"

        vidnames = [
            "20180124T115800-20180124T122800b_%T1",
            "20180115T150502-20180115T150902_%T1",
        ]

        for videoname in vidnames:
            results_sink = "/media/nexus/storage5/swissknife_results/full_inference/primate_july_test_no_matching/"
            name_helper = videoname
            IdNet = load_model(
                "/media/nexus/storage5/swissknife_results/identification/primate/identification_full_ours_CV_fraction_1.0_2020-07-13-10_33/"
                + name_helper
                + "/IDnet_"
                # + name_helper
                # + "20180131T135402-20180131T142501_%T1"
                + videoname
                + "_recognitionNet.h5"
            )
            videoname = vidbase + videoname
            # videoname = '../testovideo_short'

            results_sink = results_sink + name_helper + "/"
            # results_sink = "testing_short/"
            check_directory(results_sink)

            # load example video
            print("loading video")
            print(videoname)

            batchsize = 5000

            print("video loaded")
            videodata = "path"

            full_inference(
                # videodata=molded_video,
                videodata=videodata,
                results_sink=results_sink,
                SegNet=SegNet,
                IdNet=IdNet,
            )

    elif operation == "mouse":

        species = "mouse"
        SegNet = SegModel(species=species)
        SegNet.inference_config.DETECTION_MIN_CONFIDENCE = 0.8
        SegNet.set_inference(
            model_path="/media/nexus/storage4/swissknife_results/segmentation/mouse_/mouse20200624T1414/"
            "mask_rcnn_mouse_0040.h5"
        )

        keras.losses.custom_binary_crossentropy = custom_binary_crossentropy
        PoseNet = load_model(
            "/media/nexus/storage4/swissknife_results/poseestimation/poseestimation_full_2020-07-01-21_20/posenetNet.h5",
            custom_objects={"loss": custom_binary_crossentropy},
        )

        results_sink = (
            "/media/nexus/storage4/swissknife_results/full_inference/mouse_test/"
        )
        # check_directory(results_sink)

        videodata = loadVideo(videoname, greyscale=False, num_frames=700)[300:500]
        molded_video = mold_video(videodata, dimension=1024, n_jobs=10)
        full_inference(
            videodata=molded_video,
            results_sink=results_sink,
            SegNet=SegNet,
            # IdNet=IdNet,
            PoseNet=PoseNet,
            id_matching=False,
        )
    else:
        raise NotImplementedError

    print("DONE")


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
    default=None,
    help="filename of the video to be processed (has to be a segmented one)",
)

if __name__ == "__main__":
    main()
