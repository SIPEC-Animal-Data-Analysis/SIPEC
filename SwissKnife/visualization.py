"""
SIPEC
MARKUS MARKS
SCRIPT TO HELP VISUALIZE DATA AND RESULTS
"""
from argparse import ArgumentParser

import cv2
import numpy as np
import skvideo.io
from tqdm import tqdm

from SwissKnife.segmentation import mold_video
from SwissKnife.utils import coords_to_masks, load_config, load_vgg_labels, loadVideo


def visualize_labels_on_video_cv(
    video, labels, framerate_video, out_path, num_frames=None, predictions=None
):
    """TODO: Fill in description"""
    # TODO: change here to matplotlib?
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Error opening video stream or file")

    results = []
    idx = 0
    start_idx = 0
    while cap.isOpened():
        print(idx)
        ret, frame = cap.read()
        if idx == len(labels):
            break
        # if num_frames:
        #     if idx > num_frames:
        #         break
        if ret:
            if idx == 0:
                size = np.asarray(frame).shape
            if predictions is None:
                cv2.putText(
                    frame,
                    labels[idx] + "_" + str(idx),
                    (50, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    lineType=2,
                )
            else:
                cv2.putText(
                    frame,
                    "GT:" + labels[idx],
                    (50, 40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    lineType=2,
                )
                cv2.putText(
                    frame,
                    "Prediction:" + predictions[idx],
                    (50, 55),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    lineType=2,
                )
            results.append(frame)
            idx += 1
        else:
            break

    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    out = cv2.VideoWriter(
        out_path, fourcc, framerate_video, (int(cap.get(3)), int(cap.get(4))), True
    )
    for res in results:
        out.write(res)

    out.release()
    cap.release()
    cv2.destroyAllWindows()


# TODO: remove unused code
def visualize_labels_on_video_skimage_array(
    video, labels, framerate_video, out_path, num_frames=None, predictions=None
):
    # TODO: change here to matplotlib?
    # cap = cv2.VideoCapture(video)
    # if not cap.isOpened():
    #     print("Error opening video stream or file")

    results = []
    idx = 0
    # start_idx = 0
    # while cap.isOpened():
    for idx in tqdm(range(0, len(video))):
        frame = video[idx]
        print(idx)
        # ret, frame = cap.read()
        # if start_idx < 251:
        #     start_idx = start_idx + 1
        #     continue
        if idx == len(labels):
            break
        if num_frames:
            if idx > num_frames:
                break
        # if ret:
        # if idx == 0:
        #    size = np.asarray(frame).shape
        if predictions is None:
            cv2.putText(
                frame,
                labels[idx] + "_" + str(idx),
                (50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 0, 0),
                lineType=2,
            )
        else:
            cv2.putText(
                frame,
                "GT:" + labels[idx],
                (50, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 0, 0),
                lineType=2,
            )
            cv2.putText(
                frame,
                "Prediction:" + predictions[idx],
                (50, 55),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                lineType=2,
            )
        results.append(frame)
        idx = idx + 1

    skvideo.io.vwrite(out_path, results, verbosity=1)

    # fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    # out = cv2.VideoWriter(
    #     out_path, fourcc, framerate_video, (int(cap.get(3)), int(cap.get(4))), True
    # )
    # for res in results:
    #     out.write(res)

    # out.release()
    # cap.release()
    # cv2.destroyAllWindows()


# TODO: remove unused code
def visualize_labels_on_video(video_path, labels_path, outpath):
    """TODO: Fill in description"""
    vid = loadVideo(video_path, greyscale=False)
    framerate_video = 17
    behavior = "freezing"

    labels = load_vgg_labels(
        labels_path,
        video_length=len(vid),
        framerate_video=framerate_video,
        behavior=behavior,
    )

    visualize_labels_on_video_cv(video_path, labels, framerate_video, outpath)


# TODO: remove unused code
def multiply_list(input_list, mul):
    """Mutilpy each element of a list by another object"""
    return [el * mul for el in input_list]


def displayBoxes(
    frame, mask, color=(0, 0, 255), animal_id=None, mask_id=None, alpha=0.5
):
    """TODO: Fill in description"""
    mask_color_labeled = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    cv2.rectangle(frame, (mask[1], mask[0]), (mask[3], mask[2]), color, 3)

    if animal_id:
        cv2.putText(
            frame,
            animal_id,
            (mask[1], mask[0]),
            font,
            0.5,
            mask_color_labeled,
            font_thickness,
            cv2.LINE_AA,
        )

    return frame


def displayScatter(frame, coords, color=(0, 0, 255)):
    """TODO: Fill in description"""
    cv2.circle(frame, (int(coords[0]), int(coords[1])), 3, color, -1)
    return frame


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(image.shape[-1]):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c],
            image[:, :, c],
        )
    return image


colors = [
    (255, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 255),
    (150, 150, 150),
    (0, 150, 0),
    (150, 0, 0),
    (0, 0, 150),
    (150, 0, 150),
    (75, 75, 75),
    (0, 75, 0),
    (75, 0, 0),
    (0, 0, 75),
    (75, 0, 75),
]
posenet_colors = [
    (0, 32, 0),
    (0, 64, 0),
    (0, 127, 0),
    (0, 255, 0),
    (0, 0, 32),
    (0, 0, 64),
    (0, 0, 127),
    (0, 0, 255),
    (32, 0, 0),
    (64, 0, 0),
    (127, 0, 0),
    (255, 0, 0),
    (32, 32, 0),
    (64, 64, 0),
    (127, 127, 0),
    (255, 255, 0),
]


def visualize_full_inference(
    results_sink,
    networks,
    video,
    results,
    output_video_name,
    display_coms=False,
    dimension=1024,
):
    """TODO: Fill in description"""
    resulting_frames = []
    # prev_results = None
    for idx in tqdm(range(0, len(video))):
        frame = video[idx]

        frame = cv2.addWeighted(
            np.zeros(frame.shape, dtype="uint8"), 0.4, frame, 0.6, 0
        )

        # indicate frame
        frame = cv2.putText(
            frame,
            "framenum: " + str(idx),
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            colors[0],
            1,
            cv2.LINE_AA,
        )

        try:
            results[idx]["coms"]
            # prev_results = results[idx]
        except TypeError:
            resulting_frames.append(frame)
            continue

        try:
            frame = cv2.putText(
                frame,
                "behavior: " + results[idx]["behavior_scores"],
                (100, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[0],
                1,
                cv2.LINE_AA,
            )
            frame = cv2.putText(
                frame,
                "behavior: " + results[idx]["behavior"],
                (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[0],
                1,
                cv2.LINE_AA,
            )
            frame = cv2.putText(
                frame,
                "behavior: " + results[idx]["behavior_threshold"],
                (100, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[1],
                1,
                cv2.LINE_AA,
            )
        except (KeyError, TypeError):
            pass

        if "SegNet" in networks.keys():
            coms = results[idx]["coms"]

            # ids = np.zeros([0, 1, 2, 3]).astype("int")
            # ids = results[idx]["track_ids"]

            boxes = results[idx]["boxes"]
            for box_id, box in enumerate(boxes):
                if box[0] == 0:
                    continue
                try:
                    frame = displayBoxes(frame, box, color=colors[box_id])
                except IndexError:
                    continue
            try:
                # masks = results[idx]["masked_masks"]
                masks = coords_to_masks(results[idx]["mask_coords"], dim=dimension)
            except IndexError:
                continue
            print("num masks: ", str(masks.shape[-1]))
            for i in range(masks.shape[-1]):
                mask = masks[:, :, i]
                print(mask)
                frame = apply_mask(frame, mask, color=colors[i])

                if display_coms:
                    for hist in range(10):
                        coms = results[idx - hist]["coms"]
                        sizefactor = int(10.0 * (1.0 - hist / 10))
                        try:
                            frame = displayScatter(
                                frame, coms[i, :], color=colors[i], size=sizefactor
                            )
                        except IndexError:
                            continue

        if "IdNet" in networks.keys():
            # offset = 100
            name_ids = results[idx]["ids"]
            confidences = results[idx]["confidences"]
            mymasks = results[idx]["masked_masks"]
            overlaps = results[idx]["overalps"]
            corrected_ids = results[idx]["smoothed_ids"]

            # corrected_ids = {}
            if len(masks.shape) < 3:
                masks = np.expand_dims(masks, axis=-1)
            for i in range(masks.shape[-1]):
                if i >= 4:
                    print("OUTSIDE!")
                    continue
                try:
                    mask = boxes[i]
                except IndexError:
                    print("OUTSIDE3!")
                    continue

                try:
                    if i < len(coms):
                        frame = cv2.putText(
                            frame,
                            "X Pos " + "%.2f" % coms[i, 0],
                            (mask[1], mask[0] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            colors[0],
                            1,
                            cv2.LINE_AA,
                        )
                        frame = cv2.putText(
                            frame,
                            "Y Pos " + "%.2f" % coms[i, 1],
                            (mask[1] + 5, mask[0] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            colors[0],
                            1,
                            cv2.LINE_AA,
                        )
                    if i < len(confidences):
                        frame = cv2.putText(
                            frame,
                            "Confidence " + "%.2f" % confidences[i],
                            (mask[1], mask[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            colors[0],
                            1,
                            cv2.LINE_AA,
                        )
                    if i < len(name_ids):
                        frame = cv2.putText(
                            frame,
                            "Animal ID " + name_ids[i],
                            (mask[1], mask[0] - 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            colors[0],
                            1,
                            cv2.LINE_AA,
                        )
                    if i < len(name_ids):
                        frame = cv2.putText(
                            frame,
                            "Overlap: " + str(overlaps[i]),
                            (mask[1], mask[0] - 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            colors[0],
                            1,
                            cv2.LINE_AA,
                        )
                    try:
                        if i < len(name_ids):
                            frame = cv2.putText(
                                frame,
                                "corrected Animal ID " + corrected_ids[i],
                                (mask[1], mask[0] - 125),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                colors[0],
                                1,
                                cv2.LINE_AA,
                            )
                    except KeyError:
                        if i < len(name_ids):
                            frame = cv2.putText(
                                frame,
                                "corrected Animal ID " + "none",
                                (mask[1], mask[0] - 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                colors[0],
                                1,
                                cv2.LINE_AA,
                                10,
                            )
                    if i < len(mymasks):
                        masksize = float(mymasks[i].sum()) / float(
                            mymasks[i].shape[0] * mymasks[i].shape[1]
                        )
                        frame = cv2.putText(
                            frame,
                            "Size " + "%.1f" % masksize,
                            (mask[1], mask[0] - 175),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            colors[0],
                            1,
                            cv2.LINE_AA,
                        )
                except (TypeError, IndexError):
                    continue

        else:
            name_ids = results[idx]["track_ids"]
            mymasks = results[idx]["masked_masks"]
            overlaps = results[idx]["overalps"]
            for i in range(masks.shape[-1]):
                if i >= 4:
                    print("OUTSIDE!")
                    continue
                try:
                    mask = boxes[i]
                except IndexError:
                    print("OUTSIDE3!")
                    continue

                try:
                    textsize = 0.5
                    if i < len(coms):
                        frame = cv2.putText(
                            frame,
                            "X Pos " + "%.2f" % coms[i, 0],
                            (mask[1], mask[0] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            textsize,
                            colors[0],
                            1,
                            cv2.LINE_AA,
                        )
                        frame = cv2.putText(
                            frame,
                            "Y Pos " + "%.2f" % coms[i, 1],
                            (mask[1], mask[0] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            textsize,
                            colors[0],
                            1,
                            cv2.LINE_AA,
                        )
                    # if i < len(confidences):
                    #     frame = cv2.putText(
                    #         frame,
                    #         "Confidence " + "%.2f" % confidences[i],
                    #         (mask[1], mask[0] - 50),
                    #         cv2.FONT_HERSHEY_SIMPLEX,
                    #         0.75,
                    #         colors[0],
                    #         1,
                    #         cv2.LINE_AA,
                    #     )
                    # if i < len(name_ids):
                    #     frame = cv2.putText(
                    #         frame,
                    #         "Animal ID " + name_ids[i],
                    #         (mask[1], mask[0] - 75),
                    #         cv2.FONT_HERSHEY_SIMPLEX,
                    #         0.75,
                    #         colors[0],
                    #         1,
                    #         cv2.LINE_AA,
                    #     )
                    if i < len(name_ids):
                        frame = cv2.putText(
                            frame,
                            "movement " + "%.1f" % overlaps[i],
                            (mask[1], mask[0] - 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            textsize,
                            colors[0],
                            1,
                            cv2.LINE_AA,
                        )
                    # try:
                    #     if i < len(name_ids):
                    #         frame = cv2.putText(
                    #             frame,
                    #             "corrected Animal ID " + corrected_ids[i],
                    #             (mask[1], mask[0] - 125),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.75,
                    #             colors[0],
                    #             1,
                    #             cv2.LINE_AA,
                    #         )
                    # except KeyError:
                    #     if i < len(name_ids):
                    #         frame = cv2.putText(
                    #             frame,
                    #             "corrected Animal ID " + "none",
                    #             (mask[1], mask[0] - 150),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.75,
                    #             colors[0],
                    #             1,
                    #             cv2.LINE_AA,
                    #             10,
                    #         )
                    if i < len(mymasks):
                        masksize = float(mymasks[i].sum()) / float(
                            mymasks[i].shape[0] * mymasks[i].shape[1]
                        )
                        frame = cv2.putText(
                            frame,
                            "Size " + "%.1f" % masksize,
                            (mask[1], mask[0] - 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            textsize,
                            colors[0],
                            1,
                            cv2.LINE_AA,
                        )
                except Exception as e:
                    print(e)
                    continue

        if "PoseNet" in networks.keys():
            # TODO: fix hack
            for _, poses in enumerate(results[idx]["pose_coordinates"]):
                try:
                    for pose in poses[:-1]:
                        frame = displayScatter(frame, pose)
                except KeyError:
                    pass

        resulting_frames.append(frame)
    skvideo.io.vwrite(results_sink + output_video_name, resulting_frames, verbosity=1)


def main():
    """The main function block"""
    args = parser.parse_args()
    output_video_name = args.output_video_name
    video = args.video
    results_path = args.results_path
    config = args.config

    viz_cfg = load_config("../configs/inference/" + config)
    viz_cfg["id_classes"] = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
    }
    viz_cfg["networks"] = {"SegNet": None, "PoseNet": None}

    videodata = loadVideo(
        video, greyscale=viz_cfg["greyscale"], num_frames=viz_cfg["num_frames"]
    )
    molded_video = mold_video(videodata, dimension=viz_cfg["mold_dimension"])
    results = np.load(results_path, allow_pickle=True)

    result_dir = ""
    for el in results_path.split("/")[:-1]:
        result_dir += el + "/"

    visualize_full_inference(
        results_sink=result_dir,
        networks=viz_cfg["networks"],
        video=molded_video,
        results=results,
        output_video_name=output_video_name,
        dimension=viz_cfg["mold_dimension"],
        display_coms=viz_cfg["display_coms"],
    )
    print("DONE")


parser = ArgumentParser()

parser.add_argument(
    "--video",
    action="store",
    dest="video",
    type=str,
    default=None,
    help="which video to visualize",
)
parser.add_argument(
    "--output_video_name",
    action="store",
    dest="output_video_name",
    type=str,
    default="results_video.mp4",
    help="name for visualization video",
)
parser.add_argument(
    "--results_path",
    action="store",
    dest="results_path",
    type=str,
    default=None,
    help="path the results file from full_inference",
)
parser.add_argument(
    "--config",
    action="store",
    dest="config",
    type=str,
    default="default",
    help="config to use",
)


if __name__ == "__main__":
    main()
