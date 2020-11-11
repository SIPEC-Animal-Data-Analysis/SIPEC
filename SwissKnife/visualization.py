# SIPEC
# MARKUS MARKS
# SCRIPT TO HELP VISUALIZE DATA AND RESULTS
from argparse import ArgumentParser
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter

import sys

sys.path.append("../")

from SwissKnife.segmentation import mold_video
from SwissKnife.utils import loadVideo, load_vgg_labels, load_dict, coords_to_masks


def visualize_labels_on_video_cv(video, labels, framerate_video, out_path):
    # TODO: change here to matplotlib?
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Error opening video stream or file")

    results = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if idx == 0:
                size = np.asarray(frame).shape
            cv2.putText(
                frame,
                labels[idx],
                (50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
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


def visualize_labels_on_video():
    base_path = "/home/nexus/github/DeepLab-SwissKnife/ineichen_data/data/"
    video_path = base_path + "BMA1_PE2_2018_12_19_084143.576.mp4"
    labels_path = base_path + "video_freezing_tempseg_only_BMA1_PE2.csv"
    outpath = (
        "/home/nexus/github/DeepLab-SwissKnife/ineichen_data/results/testowatz.avi"
    )

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


def displayBoxes(frame, mask, color=(0, 0, 255), animal_id=None, mask_id=None):
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
    # for coord in coords:
    cv2.circle(frame, (int(coords[1]), int(coords[0])), 3, color, -1)
    return frame


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c],
        )
    return image


colors = [
    (255, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 255),
]


def visualize_full_inference(networks, video, results, display_coms=False):
    resulting_frames = []
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

        if "segmentation" in networks:
            coms = results[idx]["coms"]

            # ids = np.zeros([0, 1, 2, 3]).astype("int")
            ids = results[idx]["track_ids"]

            boxes = results[idx]["boxes"]
            for box_id, box in enumerate(boxes):
                if box[0] == 0:
                    continue
                frame = displayBoxes(frame, box, color=colors[box_id])
            masks = coords_to_masks(results[idx]["mask_coords"])
            print("num masks: ", str(masks.shape[-1]))
            for i in range(masks.shape[-1]):
                mask = masks[:, :, i]
                print(mask)
                if i < 4:
                    frame = apply_mask(frame, mask, color=colors[i])
                    pass

                if display_coms:
                    frame = displayScatter(frame, coms[i, :], color=colors[i])

        if "identification" in networks:
            offset = 100
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

        if "poseestimation" in networks:
            # TODO: fix hack
            for pose_id, poses in enumerate(results[idx]["pose_coordinates"]):
                try:
                    poses[:, 0] += int(coms[pose_id][0])
                    poses[:, 1] += int(coms[pose_id][1])
                    poses[:, 0] -= 64
                    poses[:, 1] -= 64
                    for pose in poses[:-1]:
                        frame = displayScatter(frame, pose)
                except KeyError:
                    pass

        resulting_frames.append(frame)

    out_path = "../viz_tests/"
    import skvideo.io

    skvideo.io.vwrite(out_path + "/outputvideo_short_new.mp4", resulting_frames)


def main():
    args = parser.parse_args()
    operation = args.operation
    if operation == "behavior_labels_mouse":
        visualize_labels_on_video()
    if operation == "vis_results_mouse":
        print("load data")
        videoname = (
            "/media/nexus/storage5/swissknife_data/mouse/raw_videos/individual_data/"
            "OFT_38.mp4"
        )
        videodata = loadVideo(videoname, greyscale=False, num_frames=700)[300:500]
        molded_video = mold_video(videodata, dimension=1024)
        results_path = (
            "/media/nexus/storage4/swissknife_results/full_inference/mouse_test/"
        )
        results = np.load(results_path + "inference_results.npy", allow_pickle=True)
        print("data loaded")
        visualize_full_inference(
            networks=["segmentation", "poseestimation"],
            video=molded_video,
            results=results,
        )

    if operation == "vis_results_primate":
        print("load data")
        videoname = (
            "/media/nexus/storage5/swissknife_data/primate/raw_videos_sorted/2018_merge/"
            "20180115T150502-20180115T150902_%T1.mp4"
        )

        # videoname = '../testovideo_short.mp4'

        videodata = loadVideo(videoname, greyscale=False)
        molded_video = mold_video(videodata, dimension=2048, n_jobs=40)
        print("loaded_video")

        results_path = (
            "/media/nexus/storage5/swissknife_results/full_inference/primate_july_test_large/"
            "/20180115T150502-20180115T150902_%T1/"
        )


        results = np.load(results_path + "inference_results.npy", allow_pickle=True)
        # results_masks = load_dict(results_path + "inference_resulting_masks.pkl")
        print("data loaded")
        visualize_full_inference(
            networks=["segmentation", "identification"],
            video=molded_video,
            results=results,
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
    "--operation",
    action="store",
    dest="operation",
    type=str,
    default="train_primate",
    help="standard training options for SIPEC data",
)

if __name__ == "__main__":
    main()
