# SIPEC
# MARKUS MARKS
# RUN FULL INFERENCE
import os
import operator
from argparse import ArgumentParser
from tqdm import tqdm
from tensorflow.keras.models import load_model
import numpy as np

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
)

# TODO: save molded imgs?


def full_inference(
    videodata,
    results_sink,
    networks,
    example_frame,
    id_classes,
    mask_matching=False,
    id_matching=False,
    mask_size=256,
    lookback=100,
    mold_dimension=1024,
    max_ids=4,
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

    for idx, el in tqdm(enumerate(videodata)):
        # for idx in range(length):
        #     el = cv2.imread(videodata + "frame%d.jpg" % idx)
        if idx == 115:
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
                heatmaps = networks["PoseNet"].predict(np.expand_dims(img, axis=0))
                heatmaps = heatmaps[0, :, :, :]
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

        results.append(results_per_frame)

    if id_matching:
        for idx, el in tqdm(enumerate(videodata)):
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

    return results


def main():
    args = parser.parse_args()
    species = args.species
    gpu_name = args.gpu
    video = args.video
    max_ids = args.max_ids
    segnet_path = args.segnet_path
    posenet_path = args.posenet_path
    do_visualization = args.do_visualization
    results_sink = args.results_sink
    output_video_name = args.output_video_name

    # TODO: somehow nicer catch this
    if not results_sink[-1] == "/":
        results_sink += "/"

    # TODO: put me in cfg file
    inference_cfg = {
        "mold_dimension": 1024,
        "mask_size": 64,
        "lookback": 25,
        "num_frames": 1000,  # have only the first 1000 frames analyzed for testing
        "id_matching": False,
        "mask_matching": True,
        "display_coms": False,
        "id_classes": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
        },
    }
    # primate: id_classes = {"Charles": 0,"Max": 1,"Paul": 2,"Alan": 3,}

    setGPU(gpu_name)
    check_directory(results_sink)

    videodata = loadVideo(
        video, greyscale=False, num_frames=inference_cfg["num_frames"]
    )
    molded_video = mold_video(
        videodata, dimension=inference_cfg["mold_dimension"], n_jobs=20
    )

    SegNet = SegModel(species=species)
    SegNet.inference_config.DETECTION_MIN_CONFIDENCE = 0.99
    SegNet.set_inference(model_path=segnet_path)
    # SegNet.inference_config.DETECTION_MIN_CONFIDENCE = 0.1

    networks = {"SegNet": SegNet}

    if posenet_path:
        PoseNet = load_model(
            posenet_path,
            custom_objects={"loss": custom_binary_crossentropy},
        )
        networks["PoseNet"] = PoseNet

    results = full_inference(
        videodata=molded_video,
        results_sink=results_sink,
        networks=networks,
        example_frame=videodata[0],
        id_classes=inference_cfg["id_classes"],
        max_ids=max_ids,
        id_matching=inference_cfg["id_matching"],
        mask_matching=inference_cfg["mask_matching"],
        mask_size=inference_cfg["mask_size"],
        lookback=inference_cfg["lookback"],
        mold_dimension=inference_cfg["mold_dimension"],
    )
    if do_visualization:
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
