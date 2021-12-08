# SIPEC
# MARKUS MARKS
# RUN FULL INFERENCE
import os
import operator
from argparse import ArgumentParser
from tqdm import tqdm
from tensorflow.keras.models import load_model
import numpy as np
from skimage.registration import optical_flow_tvl1
from sklearn.externals._pilutil import imresize

from SwissKnife.visualization import visualize_full_inference
from SwissKnife.masksmoothing import MaskMatcher
from SwissKnife.poseestimation import heatmap_to_scatter, custom_binary_crossentropy
from SwissKnife.segmentation import SegModel, mold_video, mold_image
from SwissKnife.utils import (
    setGPU,
    loadVideo,
    masks_to_coms,
    apply_all_masks,
    detect_primate,
    check_directory,
    rescale_img,
    masks_to_coords,
    mask_to_original_image,
    load_config,
)
from SwissKnife.model import Model


def full_inference(
    videodata,
    results_sink,
    networks,
    example_frame,
    id_classes,
    use_flow=1,
    downsample_factor=1,
    mask_matching=False,
    id_matching=False,
    mask_size=256,
    lookback_matching=300,
    lookback_behavior=25,
    mold_dimension=1024,
    max_ids=4,
    posenet_resize_factor=None,
    behaviornet_confidence=0.5,
):
    """Performs full inference on a given video using available SIPEC modules.

    Parameters
    ----------
    videodata : np.ndarray
        numpy array of read-in videodata.
    results_sink : str
        Path to where data will be saved.
    networks : dict
        Dictionary containing SIPEC modules to be used for full inference ("SegNet", "PoseNet", "BehaveNet", IdNet")
    mask_matching : bool
        Use greedy-mask-matching
    id_matching : bool
        Correct/smooth SIPEC:IdNet identity using identities based on temporal tracking (greedy-mask-matching)
    mask_size : int
        Mask size used for the cutout of animals.
    lookback : int
        Number of timesteps to look back into the past for id_matching.
    max_ids : int
        Number of maximum ids / maximum number of animals in any FOV.


    Returns
    -------
    list
        Outputs of all the provided SIPEC modules for each video frame.
    """
    maskmatcher = MaskMatcher()
    maskmatcher.max_ids = max_ids
    classes = id_classes
    # invert classes / to go both ways
    classes_invert = [el for el in classes.keys()]

    # set threshold for detection of primate identities
    threshold = 0.5
    results = []
    # lookback_matching = 50 # idtracker length
    maskmatcher.hist_length = lookback_matching

    prev_results = None

    for idx, el in tqdm(enumerate(videodata)):
        # for idx in range(length):
        #     el = cv2.imread(videodata + "frame%d.jpg" % idx)
        if idx == 1253:
            print("yo")
        results_per_frame = {}
        molded_img, masks, boxes, mask_scores = networks["SegNet"].detect_image(
            el, verbose=0, mold=True
        )

        masks = masks[:, :, :max_ids]
        boxes = boxes[:max_ids, :]
        mask_scores = mask_scores[:max_ids]
        coms = masks_to_coms(masks)
        # TODO: fixme
        try:
            masked_imgs, masked_masks = apply_all_masks(
                masks, coms, molded_img, mask_size=mask_size
            )
            masked_imgs = imresize(masked_imgs[0][:, :, 0], downsample_factor)
            masked_imgs = np.expand_dims(masked_imgs, axis=-1)

            v, u = optical_flow_tvl1(
                videodata[idx - 1, :, :, 0], videodata[idx, :, :, 0]
            )
            v = np.expand_dims(v, axis=-1)
            test = mold_image(
                v,
                dimension=mold_dimension,
            )
            flow_image, flow_mask = apply_all_masks(
                masks, coms, test, mask_size=mask_size
            )
            v = imresize(flow_image[0][:, :, 0], downsample_factor)
            flow_imags = np.expand_dims(v, axis=-1)

        except (IndexError, ValueError):
            results.append(0)
            continue

        if mask_matching:

            if not idx == 0:
                try:
                    mapping = maskmatcher.match_masks(
                        boxes[: maskmatcher.max_ids], results[-lookback_matching:]
                    )
                    prev_results = results[-lookback_matching:-1]
                except TypeError:
                    mapping = maskmatcher.match_masks(
                        boxes[: maskmatcher.max_ids], prev_results
                    )
                print(mapping)
                new_ids = maskmatcher.match_ids(
                    mapping, len(boxes[: maskmatcher.max_ids])
                )
                new_ids = [el[1] + 1 for el in mapping.values()]
                overlaps = [mapping[el][0] for el in mapping]
                if len(overlaps) < len(boxes):
                    for i in range(len(boxes) - len(overlaps)):
                        overlaps.append(0)
                if max(new_ids) > 0 and len(boxes) > 0:
                    print("boxes before: ", str(boxes))
                    boxes = maskmatcher.map(mapping, boxes)
                    print("boxes after: ", str(boxes))
                    masks = np.swapaxes(masks, 0, 2)
                    masks = maskmatcher.map(mapping, masks)
                    masks = np.swapaxes(masks, 0, 2)
                    masked_imgs = maskmatcher.map(mapping, masked_imgs)
                    flow_imags = maskmatcher.map(mapping, flow_imags)
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
        results_per_frame["flow_imags"] = np.asarray(flow_imags)
        results_per_frame["masked_masks"] = masked_masks.astype("uint8")
        results_per_frame["rescaled_imgs"] = rescaled_imgs.astype("uint8")
        # maskmatch.sort ()

        if "IdNet" in networks.keys():
            ids = []
            confidences = []
            for img in rescaled_imgs:
                primate, confidence = detect_primate(
                    img, networks["IdNet"], classes_invert, threshold
                )
                ids.append(primate)
                confidences.append(confidence)
            results_per_frame["ids"] = ids
            results_per_frame["confidences"] = confidences

        if "PoseNet" in networks.keys():
            maps = []
            for mask_id, img in enumerate(masked_imgs):
                if posenet_resize_factor:
                    img = imresize(img, posenet_resize_factor).astype("uint8")
                heatmaps = networks["PoseNet"].predict(np.expand_dims(img, axis=0))
                heatmaps = heatmaps[0, :, :, :]
                if posenet_resize_factor:
                    heatmaps = imresize(heatmaps, (1 / posenet_resize_factor)).astype(
                        "uint8"
                    )
                # TODO: merge with section in posestimation.py and make common util fcn
                image, window, scale, padding, crop = mold_image(
                    example_frame, dimension=mold_dimension, return_all=True
                )
                unmolded_maps = []
                for map_id in range(heatmaps.shape[-1]):
                    map = heatmaps[:, :, map_id]
                    a = mask_to_original_image(
                        mold_dimension, map, coms[mask_id], mask_size
                    )
                    a = np.expand_dims(a, axis=-1)
                    # b = revert_mold(a, padding, scale, dtype='float32')
                    unmolded_maps.append(a)
                unmolded_maps = np.array(unmolded_maps)
                unmolded_maps = np.swapaxes(unmolded_maps, 0, -1)
                unmolded_maps = unmolded_maps[0]
                coords_predict = heatmap_to_scatter(unmolded_maps)
                maps.append(coords_predict)
            results_per_frame["pose_coordinates"] = maps

        if "BehaveNet" in networks.keys():
            if idx == 0:
                predictions = np.array([[0.0, 1.0]])
            elif idx < lookback_behavior:
                masked_imgs = np.swapaxes(masked_imgs, 1, 2)
                masked_imgs = np.swapaxes(masked_imgs, 0, 1)
                flow_imags = np.swapaxes(flow_imags, 1, 2)
                flow_imags = np.swapaxes(flow_imags, 0, 1)
                if use_flow == 0:
                    inp = masked_imgs[0]
                elif use_flow == 1:
                    inp = np.stack([flow_imags, masked_imgs], axis=-1)
                else:
                    flow_imags = np.swapaxes(flow_imags, 0, 1)
                    flow_imags = np.swapaxes(flow_imags, 1, 2)
                    inp = flow_imags
                    inp = np.expand_dims(inp, axis=0)
                predictions, _ = networks["BehaveNet"].predict(inp, "recognition")
            else:
                if use_flow == 2:
                    inp = []
                    for el in reversed(results[-lookback_behavior:]):
                        try:
                            inp.append(el["flow_imags"])
                        except TypeError:
                            print("FLOW ERROR")
                            idx = -1
                            while True:
                                try:
                                    inp.append(results[idx]["flow_imags"])
                                    break
                                except TypeError:
                                    idx -= 1
                    inp = np.array(inp)
                    inp = np.expand_dims(inp, axis=0)
                else:
                    raise Exception("Not implemented")
                predictions, _ = networks["BehaveNet"].predict(inp, "sequential")
            results_per_frame["behavior_scores"] = str(predictions)
            results_per_frame["behavior"] = str(predictions.argmax(axis=-1))
            if predictions[0][0] > behaviornet_confidence:
                results_per_frame["behavior_threshold"] = "freezing"
            else:
                results_per_frame["behavior_threshold"] = "none"
        results.append(results_per_frame)

    if id_matching:
        for idx, el in tqdm(enumerate(videodata)):
            if not (lookback_matching < idx < len(videodata) - lookback_matching):
                results[idx]["smoothed_ids"] = ids
            else:
                corrected_ids = {}
                for i in range(len(ids)):
                    prev_ids = {}
                    # for j in range(-lookback, 0):
                    # TODO: forward backward filter
                    for j in range(-lookback_matching, lookback_matching):
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
    return results


def main():
    args = parser.parse_args()
    species = args.species
    gpu_name = args.gpu
    video = args.video
    max_ids = args.max_ids
    segnet_path = args.segnet_path
    posenet_path = args.posenet_path
    behavenet_path_recognition = args.behavenet_path_recognition
    behavenet_path_sequential = args.behavenet_path_sequential
    do_visualization = args.do_visualization
    results_sink = args.results_sink
    output_video_name = args.output_video_name
    config = args.config

    # TODO: somehow nicer catch this
    if not results_sink[-1] == "/":
        results_sink += "/"

    setGPU(gpu_name)
    check_directory(results_sink)

    inference_cfg = load_config("../configs/inference/" + config)
    inference_cfg["max_ids"] = max_ids
    inference_cfg["id_classes"] = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
    }
    # primate: id_classes = {"Charles": 0,"Max": 1,"Paul": 2,"Alan": 3,}

    videodata = loadVideo(
        video,
        greyscale=inference_cfg["greyscale"],
        num_frames=inference_cfg["num_frames"],
    )
    molded_video = mold_video(
        videodata, dimension=inference_cfg["mold_dimension"], n_jobs=20
    )
    # /media/nexus/storage4/swissknife_results/second_submission/mouse20211015T1558/mask_rcnn_mouse_0095.h5
    SegNet = SegModel(species=species)
    SegNet.inference_config.DETECTION_MIN_CONFIDENCE = inference_cfg[
        "segnet_detection_confidence"
    ]
    SegNet.inference_config.IMAGE_MIN_DIM = inference_cfg["mold_dimension"]
    SegNet.inference_config.IMAGE_MAX_DIM = inference_cfg["mold_dimension"]
    SegNet.inference_config.IMAGE_SHAPE = [
        inference_cfg["mold_dimension"],
        inference_cfg["mold_dimension"],
        3,
    ]
    SegNet.set_inference(model_path=segnet_path)

    networks = {"SegNet": SegNet}

    if posenet_path:
        PoseNet = load_model(
            posenet_path,
            custom_objects={"loss": custom_binary_crossentropy},
        )
        networks["PoseNet"] = PoseNet

    if behavenet_path_recognition:
        BehaveNet = Model()
        BehaveNet.load_model(recognition_path=behavenet_path_recognition)
        if behavenet_path_sequential:
            BehaveNet.load_model(sequential_path=behavenet_path_sequential)
        networks["BehaveNet"] = BehaveNet

    results = full_inference(
        videodata=np.array(molded_video),
        results_sink=results_sink,
        networks=networks,
        example_frame=videodata[0],
        id_classes=inference_cfg["id_classes"],
        use_flow=inference_cfg["use_flow"],
        downsample_factor=inference_cfg["downsample_factor"],
        max_ids=max_ids,
        id_matching=inference_cfg["id_matching"],
        mask_matching=inference_cfg["mask_matching"],
        mask_size=inference_cfg["mask_size"],
        lookback_matching=inference_cfg["lookback_matching"],
        lookback_behavior=inference_cfg["lookback_behavior"],
        mold_dimension=inference_cfg["mold_dimension"],
        behaviornet_confidence=inference_cfg["behavenet_detection_confidence"],
    )
    if do_visualization:
        if inference_cfg["greyscale"]:
            videodata = loadVideo(
                video, greyscale=False, num_frames=inference_cfg["num_frames"]
            )
            molded_video = mold_video(
                videodata, dimension=inference_cfg["mold_dimension"], n_jobs=20
            )
        visualize_full_inference(
            results_sink=results_sink,
            networks=networks,
            video=molded_video,
            results=results,
            output_video_name=output_video_name,
            dimension=inference_cfg["mold_dimension"],
            display_coms=inference_cfg["display_coms"],
        )


parser = ArgumentParser()
parser.add_argument(
    "--species",
    action="store",
    dest="species",
    type=str,
    default="primate",
    help="load species specific training params",
)
parser.add_argument(
    "--gpu",
    action="store",
    dest="gpu",
    type=int,
    default=0,
    help="gpu to be used",
)
parser.add_argument(
    "--video",
    action="store",
    dest="video",
    type=str,
    default=None,
    help="filename of the video to be processed (has to be a segmented one)",
)
parser.add_argument(
    "--max_ids",
    action="store",
    dest="max_ids",
    type=int,
    default=4,
    help="maximum number of instances to be detected in the FOV",
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
    "--posenet_path",
    action="store",
    dest="posenet_path",
    type=str,
    default=None,
    help="path to posenet model",
)
parser.add_argument(
    "--behavenet_path_recognition",
    action="store",
    dest="behavenet_path_recognition",
    type=str,
    default=None,
    help="path to behaviornet recognition model",
)
parser.add_argument(
    "--behavenet_path_sequential",
    action="store",
    dest="behavenet_path_sequential",
    type=str,
    default=None,
    help="path to behaviornet sequential model",
)
parser.add_argument(
    "--config",
    action="store",
    dest="config",
    type=str,
    default="default",
    help="config to use",
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
    "--do_visualization",
    action="store",
    dest="do_visualization",
    type=bool,
    default=True,
    help="visualize results",
)
parser.add_argument(
    "--output_video_name",
    action="store",
    dest="output_video_name",
    type=str,
    default="results_video.mp4",
    help="name for visualization video",
)

if __name__ == "__main__":
    main()

# example call
# python full_inference.py --gpu 0 --species mouse --video ./animal5678_day2.avi --segnet_path "./mask_rcnn_mouse_0095.h5" --max_ids 4 --results_sink .//

# Docker usage
# docker container run -v "/home/tarun/Documents/Work/Neuro_technology/data:/home/user/data" -v "/home/tarun/Documents/Work/Neuro_technology/SIPEC:/home/user/SIPEC:ro" --runtime=nvidia --rm chadhat/sipec:tf2 full_inference.py --gpu 0 --species mouse --video /home/user/data/full_inference_and_vis_data/animal5678_day2.avi --segnet_path "/home/user//data/full_inference_and_vis_data/mask_rcnn_mouse_0095.h5" --max_ids 4 --results_sink /home/user/data/test
