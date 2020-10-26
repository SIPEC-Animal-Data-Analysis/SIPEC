### preprocessing
### keypoint extraction
### mask extraction
### identification

import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

from SwissKnife.utils import extractCOM, maskedImg, loadVideo


def path_for_file(paths, filename):
    for path in paths:
        if filename in path:
            return path
    return "none"


# TODO: segmentation
# TODO: segmented keypoints


class MouseDataset:
    def __init__(
        self, directory,
    ):
        self.directory = directory
        self.crop = 15000
        self.keypoint_classes = None
        self.keypoint_data = None

    def extract_keypoints_from_dlcfile(self, keypoints_file):

        df_annot = pd.DataFrame.from_csv(keypoints_file, header=0, sep=";")

        grand_features = []
        for idx, el in tqdm(enumerate(df_annot.iterrows())):
            if idx == 0 or idx == 1:
                continue
            else:
                #         features.append(el[3])
                features_x = []
                features_y = []
                for idx_2, el_el in enumerate(el[1]):
                    if "x" in df_annot.iloc[1][idx_2]:
                        features_x.append(el_el)
                    elif "y" in df_annot.iloc[1][idx_2]:
                        features_y.append(el_el)
                features_x = np.asarray(features_x)
                features_y = np.asarray(features_y)
            grand_features.append(np.vstack([features_x, features_y]))
        grand_features = np.asarray(grand_features)

        return grand_features

    def create_annotations_from_dlc(
        self, safe=None, keypoints_file=None  # TODO: implement saving
    ):
        # create classes if non-existent in object yet
        if self.keypoint_classes is None:
            df_annot = pd.DataFrame.from_csv(keypoints_file, header=0, sep=";")
            # extract features and classes from annotations file
            self.feature_list = []
            self.keypoint_classes = []
            for idx, el in enumerate(df_annot.iloc[1]):
                if "x" in el or "y" in el:
                    self.feature_list.append(idx)
                if "x" in el:
                    self.keypoint_classes.append(df_annot.iloc[0][idx])

        keypoints = self.extract_keypoints_from_dlcfile(keypoints_file)
        filename = (
            keypoints_file.split("labels/")[-1].split("_labels")[0].split("/")[-1]
        )

        ###
        paths = glob(self.directory + "*.npy")
        images = np.load(path_for_file(paths, filename))

        # cropping to same length
        images = images[: self.crop]
        keypoints = keypoints[: self.crop]

        self.keypoint_data = [images, keypoints]

    def get_keypoints(self):
        # check if keypoints are available
        annotations = glob(self.directory + "dlc_annotations/*.npy")
        return annotations

    def get_segmented_data(self):
        data = glob(self.directory + "segmented/*.npy")
        return data

    def get_raw_videos(self):
        data = glob(self.directory + "raw_videos/*.mp4")
        return data

    def get_masks(self):

        pass

    def get_keypoint_training_data(
        self,
        segmented=False,  # TODO: implement returning everything segmeneted
        stacked=True,
    ):
        annotations = self.get_keypoints()
        data = self.get_segmented_data()

        # TODO: give returning options for different sessions, but for now merge all data

        keypoint_data = []
        video_data = []
        for annot in annotations[:3]:
            # FIXME:
            try:
                filename = annot.split("annotations/")[1].split("_labels")[0]
                keypoints = np.load(annot)[: self.crop]

                # FIXME: for now load full video, but rather do segmented one
                # videos = np.load(path_for_file(data, filename))[:self.crop]

                video_path = path_for_file(self.get_raw_videos(), filename)

                video = loadVideo(video_path)[: self.crop]

                keypoint_data.append(keypoints)
                video_data.append(video)
            except FileNotFoundError:
                pass

        if stacked:
            return (
                np.vstack(np.asarray(keypoint_data)),
                np.vstack(np.asarray(video_data)),
            )
        else:
            return np.asarray(keypoint_data), np.asarray(video_data)

    def get_segmentation_train_data(self):
        pass

    def get_segmentation_test_data(self):
        pass


def main():
    # minimal example
    mouse = MouseDataset("/media/nexus/storage1/swissknife_data/mouse/")
    keypoints, videos = mouse.get_keypoint_training_data()


if __name__ == "__main__":
    main()
