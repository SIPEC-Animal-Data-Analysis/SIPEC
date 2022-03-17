# SIPEC
# MARKUS MARKS
# Extract cutout videos from multi animal videos

import numpy as np

from tqdm import tqdm
import os


from SwissKnife.utils import dilate_mask
from scipy.ndimage.measurements import center_of_mass
from SwissKnife.utils import apply_to_mask

# from SwissKnife.segmentation import SegModel, mold_video

from joblib import Parallel, delayed


def detect_social(
    mask_1, mask_2, threshold=10, dilation_factor=40,
):
    mask_1 = dilate_mask(mask_1, factor=dilation_factor)
    mask_2 = dilate_mask(mask_2, factor=dilation_factor)

    # event detection simply multiplication of dilated masks
    multi_event = np.logical_and(mask_1.astype(np.bool), mask_2.astype(np.bool))

    if multi_event.sum() > threshold:
        return 1
    else:
        return 0


def detect_social_parallel(frame, masks):
    res = []

    num_masks = masks.shape[-1]
    if num_masks > 1:
        for i in range(0, num_masks):
            for j in range(0, num_masks):
                soc_ev = 0
                if not j == i:
                    soc_ev = detect_social(masks[:, :, i], masks[:, :, j])
                if soc_ev == 1:
                    print("ho")
                    mask = masks[:, :, i].astype(np.int) + masks[:, :, j].astype(np.int)
                    #                     res.append(mask)
                    mask = mask.astype(np.bool)
                    com = center_of_mass(mask)
                    print(com)
                    masked_img, _ = apply_to_mask(mask, frame, com, mask_size=128)
                    #                     masked_img = masked_img[int(com[0])-128:int(com[0])+128, int(com[1])-128:int(com[1])+128]
                    print(_)
                    res.append(masked_img)

    return res


if __name__ == "__main__":

    videos = [
        # "20180115T150502-20180115T150902_%T1",
        # "20180115T150759-20180115T151259_%T1",
        # "20180124T113800-20180124T115800_%T1",
        # "20180131T135402-20180131T142501_%T1",
        "20180124T115800-20180124T122800b_%T1",
        # "20180124T095000-20180124T103000_%T1",
        # "20180116T135000-20180116T142000_%T1"
    ]

    for video in videos:

        print("new vid")
        print(video)

        results = np.load(
            "/media/nexus/storage5/swissknife_results/full_inference/primate_july_test/"
            + video
            + "/inference_results.npy",
            allow_pickle=True,
        )

        results = results[40000:]

        from SwissKnife.utils import coords_to_masks, loadVideo

        results_masks = []

        for el in tqdm(results):
            results_masks.append(coords_to_masks(el["mask_coords"]))

        vidbase = "/media/nexus/storage5/swissknife_data/primate/raw_videos_sorted/2018_merge/"
        # vidnames = [video]
        path = vidbase + videos[0] + ".mp4"
        videodata = loadVideo(path, greyscale=False)[40000:]
        molded_video = mold_video(videodata, dimension=2048, n_jobs=3)

        events = Parallel(
            n_jobs=3, max_nbytes=None, backend="multiprocessing", verbose=40
        )(
            delayed(detect_social_parallel)(frame, masks)
            for frame, masks in zip(molded_video, results_masks)
        )

        new_events = []
        for el in events:
            if not el == []:
                new_events.append(np.asarray(el[0]))

        testvid = new_events
        filename = "../full_vids/fullvids_3_" + video + "_" + "social" + ".mp4"
        print(filename)
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

        if not testvid == []:
            writer = skvideo.io.FFmpegWriter(filename)
            for frame in testvid:
                social_stuff = np.array(frame).astype("uint8")
                print(social_stuff.shape)
                writer.writeFrame(social_stuff)
            writer.close()

        single_frames = {
            0: [],
            1: [],
            2: [],
            3: [],
        }

        for idx, result in tqdm(enumerate(results)):
            coms = results[idx]["coms"]
            ids = results[idx]["track_ids"]
            boxes = results[idx]["boxes"]
            masks = coords_to_masks(results[idx]["mask_coords"])
            name_ids = results[idx]["ids"]
            confidences = results[idx]["confidences"]
            mymasks = results[idx]["masked_masks"]
            overlaps = results[idx]["overalps"]
            corrected_ids = results[idx]["smoothed_ids"]

            for i in range(masks.shape[-1]):
                if i >= 4:
                    print("OUTSIDE!")
                    continue
                try:
                    mask = boxes[i]
                except IndexError:
                    print("OUTSIDE3!")
                    continue

                for el in range(5 - len(overlaps)):
                    overlaps.append(0)
                if i in ids:
                    try:
                        overl = []
                        for el in results[idx - 3 : idx + 3]:
                            overl.append(el["overalps"][i])
                    except IndexError:
                        pass
                    condition = np.array(overl) > 0
                    condition = condition.all()

                    if condition:
                        try:
                            #                     single_frames[corrected_ids[i]].append(results[idx]['masked_imgs'][i])
                            single_frames[i].append(results[idx]["masked_imgs"][i])
                        except (TypeError, IndexError, KeyError):
                            #                     single_frames[corrected_ids[i]].append(np.zeros((128, 128, 3)))
                            continue

        for primate in tqdm(single_frames.keys()):
            try:
                print(primate)

                testvid = np.asarray(single_frames[primate], dtype="uint8")
                print(len(testvid))
                filename = (
                    "../full_vids/fullvids_3_" + video + "_" + str(primate) + ".mp4"
                )
                print(filename)
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass
                writer = skvideo.io.FFmpegWriter(filename)
                for frame in testvid:
                    writer.writeFrame(frame)
                writer.close()

            except AttributeError:
                print("wrong")
                print(primate)
                continue

    ## save indexes of selected frames
