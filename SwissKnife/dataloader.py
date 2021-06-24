# SIPEC
# MARKUS MARKS
# Dataloader
from tensorflow import keras
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.externals._pilutil import imresize
from sklearn.utils import class_weight
from tqdm import tqdm


def create_dataset(dataset, look_back=5, oneD=False):
    """Create a recurrent dataset from array.

    Parameters
    ----------
    dataset : np.ndarray
        numpy array of dataset to make recurrent
    look_back : int
        Number of timesteps to look into the past and future.
    oneD : bool
        Boolean that indicates if the current dataset is one dimensional.

    Returns
    -------
    np.ndarray
        recurrent dataset
    """
    dataX = []
    print("creating recurrency")
    for i in tqdm(range(look_back, len(dataset) - look_back)):
        if oneD:
            a = dataset[i - look_back : i + look_back]
        else:
            a = dataset[i - look_back : i + look_back, :]
        dataX.append(a)
    return np.array(dataX)


class Dataloader:

    # FIXME: for now just pass
    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        config,
        with_dlc=False,
        dlc_train=None,
        dlc_test=None,
    ):

        """
        Args:
            x_train:
            y_train:
            x_test:
            y_test:
            look_back:
            with_dlc:
            dlc_train:
            dlc_test:
        """
        self.with_dlc = with_dlc
        self.dlc_train = dlc_train
        self.dlc_test = dlc_test
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.config = config

        self.look_back = self.config["look_back"]

        self.label_encoder = None

        self.x_train_recurrent = None
        self.x_test_recurrent = None
        self.y_train_recurrent = None
        self.y_test_recurrent = None

    def encode_labels(self):
        self.label_encoder = preprocessing.LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.y_train)
        self.y_test = self.label_encoder.transform(self.y_test)

    # FIXME: nicer all
    def encode_label(self, label):
        """
        Args:
            label:
        """
        return self.label_encoder.transform(label)

    def decode_labels(self, labels):
        """
        Args:
            labels:
        """
        decoded = self.label_encoder.inverse_transform(labels)
        return decoded

    def categorize_data(self, num_classes, recurrent=False):
        """
        Args:
            num_classes:
            recurrent:
        """
        self.y_train = self.y_train.astype(int)
        # TODO: parametrize num behaviors
        self.y_train = keras.utils.to_categorical(
            self.y_train, num_classes=num_classes, dtype="int"
        )

        self.y_test = self.y_test.astype(int)
        self.y_test = keras.utils.to_categorical(
            self.y_test, num_classes=num_classes, dtype="int"
        )

        if recurrent:
            self.y_train_recurrent = self.y_train_recurrent.astype(int)
            # TODO: parametrize num behaviors
            self.y_train_recurrent = keras.utils.to_categorical(
                self.y_train_recurrent, num_classes=num_classes, dtype="int"
            )

            self.y_test_recurrent = self.y_test_recurrent.astype(int)
            self.y_test_recurrent = keras.utils.to_categorical(
                self.y_test_recurrent, num_classes=num_classes, dtype="int"
            )

    def normalize_data(self):
        # TODO: double check this here
        # self.mean = self.x_train[1000:-1000].mean(axis=0)
        # self.std = np.std(self.x_train[1000:-1000], axis=0)
        self.mean = self.x_train.mean(axis=0)
        self.std = np.std(self.x_train, axis=0)
        self.x_train = self.x_train - self.mean
        self.x_train /= self.std
        self.x_test = self.x_test - self.mean
        self.x_test /= self.std

        if not self.dlc_train is None:
            self.mean_dlc = self.dlc_train.mean(axis=0)
            self.std_dlc = self.dlc_train.std(axis=0)
            self.dlc_train -= self.mean_dlc
            self.dlc_test -= self.mean
            self.dlc_train /= self.std_dlc
            self.dlc_test /= self.std_dlc

    def create_dataset(dataset, oneD, look_back=5):
        """
        Args:
            oneD:
            look_back:
        """
        dataX = []
        for i in range(look_back, len(dataset) - look_back):
            if oneD:
                a = dataset[i - look_back : i + look_back]
            else:
                a = dataset[i - look_back : i + look_back, :]
            dataX.append(a)
        return np.array(dataX)

    def create_recurrent_labels(self):

        self.y_train_recurrent = self.y_train[self.look_back : -self.look_back]
        self.y_test_recurrent = self.y_test[self.look_back : -self.look_back]

        self.y_train = self.y_train[self.look_back : -self.look_back]
        self.y_test = self.y_test[self.look_back : -self.look_back]

    def create_recurrent_data(self, oneD=False, recurrent_labels=True):
        """
        Args:
            oneD:
        """
        self.x_train_recurrent = create_dataset(self.x_train, self.look_back, oneD=oneD)
        self.x_test_recurrent = create_dataset(self.x_test, self.look_back, oneD=oneD)

        # also shorten normal data so all same length
        self.x_train = self.x_train[self.look_back : -self.look_back]
        self.x_test = self.x_test[self.look_back : -self.look_back]

        if recurrent_labels:
            self.create_recurrent_labels()

    def create_recurrent_data_dlc(self, recurrent_labels=True):

        self.dlc_train_recurrent = create_dataset(self.dlc_train, self.look_back)
        self.dlc_test_recurrent = create_dataset(self.dlc_test, self.look_back)

        # also shorten normal data so all same length
        self.dlc_train = self.dlc_train[self.look_back : -self.look_back]
        self.dlc_test = self.dlc_test[self.look_back : -self.look_back]

        if recurrent_labels:
            self.create_recurrent_labels()

    # TODO: redo all like this, i.e. gettters instead of changing data
    def expand_dims(self):
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test = np.expand_dims(self.x_test, axis=-1)
        if self.x_test_recurrent is not None:
            self.x_train_recurrent = np.expand_dims(self.x_train_recurrent, axis=-1)
            self.x_test_recurrent = np.expand_dims(self.x_test_recurrent, axis=-1)

    def create_flattened_data(self):
        if self.with_dlc:
            _shape = self.dlc_train.shape
            self.dlc_train_flat = self.dlc_train.reshape(
                (_shape[0], _shape[1] * _shape[2])
            )
            _shape = self.dlc_test.shape
            self.dlc_test_flat = self.dlc_test.reshape(
                (_shape[0], _shape[1] * _shape[2])
            )

            _shape = self.dlc_train_recurrent.shape
            self.dlc_train_recurrent_flat = self.dlc_train_recurrent.reshape(
                (_shape[0], _shape[1] * _shape[2] * _shape[3])
            )
            _shape = self.dlc_test_recurrent.shape
            self.dlc_test_recurrent_flat = self.dlc_test_recurrent.reshape(
                (_shape[0], _shape[1] * _shape[2] * _shape[3])
            )

    def decimate_labels(self, percentage, balanced=False):
        """decimate labels to a given percentate percentage in [0,1] :return:

        Args:
            percentage:
            balanced:
        """
        if balanced:
            # TODO: do w class weights and probability in choice fcn
            raise NotImplementedError
        if self.x_train_recurrent is not None:
            num_labels = int(len(self.x_train_recurrent) * percentage)
            indices = np.arange(0, len(self.x_train_recurrent) - 1)
            random_idxs = np.random.choice(indices, size=num_labels, replace=False)
            self.x_train = self.x_train[random_idxs]
            self.y_train = self.y_train[random_idxs]
            self.x_train_recurrent = self.x_train_recurrent[random_idxs]
            self.y_train_recurrent = self.y_train_recurrent[random_idxs]
        if self.config["train_ours"] or self.config["train_ours_plus_dlc"]:
            num_labels = int(len(self.x_train) * percentage)
            indices = np.arange(0, len(self.x_train))
            random_idxs = np.random.choice(indices, size=num_labels, replace=False)
            self.x_train = self.x_train[random_idxs]
            self.y_train = self.y_train[random_idxs]
        if self.dlc_train is not None:
            num_labels = int(len(self.dlc_train) * percentage)
            indices = np.arange(0, len(self.dlc_train))
            random_idxs = np.random.choice(indices, size=num_labels, replace=False)
            self.dlc_train = self.dlc_train[random_idxs]
            self.dlc_train_flat = self.dlc_train_flat[random_idxs]
            # self.y_train = self.y_train[random_idxs]
        if self.dlc_train_recurrent is not None:
            self.dlc_train_recurrent = self.dlc_train_recurrent[random_idxs]
            self.dlc_train_recurrent_flat = self.dlc_train_recurrent_flat[random_idxs]
            # self.y_train_recurrent = self.y_train_recurrent[random_idxs]

    # old
    def reduce_labels(self, behavior, num_labels):

        """
        Args:
            behavior:
            num_labels:
        """
        idx_behavior = self.y_train == behavior
        idx_behavior = np.asarray(idx_behavior)
        idx_behavior_true = np.where(idx_behavior == 1)[0]

        # select only a subset of labels for the behavior
        selected_labels = np.random.choice(idx_behavior_true, num_labels, replace=False)

        self.y_train = ["none"] * len(idx_behavior)
        self.y_train = np.asarray(self.y_train)
        self.y_train[selected_labels] = behavior

        idx_behavior = self.y_test == behavior
        idx_behavior = np.asarray(idx_behavior)
        idx_behavior_true = np.where(idx_behavior == 1)[0]
        self.y_test = ["none"] * len(idx_behavior)
        self.y_test = np.asarray(self.y_test)
        self.y_test[idx_behavior_true] = behavior

    def remove_behavior(self, behavior):
        """
        Args:
            behavior:
        """
        idx_behavior = self.y_test == behavior
        idx_behavior = np.asarray(idx_behavior)
        self.y_test[idx_behavior] = "none"

        idx_behavior = self.y_train == behavior
        idx_behavior = np.asarray(idx_behavior)
        self.y_train[idx_behavior] = "none"

    def undersample_data(self):
        random_under_sampler = RandomUnderSampler(
            sampling_strategy="majority", random_state=42
        )

        shape = self.x_train.shape
        if len(shape) == 2:
            self.x_train, self.y_train = random_under_sampler.fit_sample(
                self.x_train, self.y_train
            )

        if len(shape) == 3:
            self.x_train = self.x_train.reshape((shape[0], shape[1] * shape[2]))
            self.x_train, self.y_train = random_under_sampler.fit_sample(
                self.x_train, self.y_train
            )
            self.x_train = self.x_train.reshape(
                (self.x_train.shape[0], shape[1], shape[2])
            )
            self.x_train = np.expand_dims(self.x_train, axis=-1)
        if len(shape) == 4:
            self.x_train = self.x_train.reshape(
                (shape[0], shape[1] * shape[2] * shape[3])
            )
            self.x_train, self.y_train = random_under_sampler.fit_sample(
                self.x_train, self.y_train
            )
            self.x_train = self.x_train.reshape(
                (self.x_train.shape[0], shape[1], shape[2], shape[3])
            )
        else:
            raise NotImplementedError

        # TODO: undersample recurrent

    def change_dtype(self):
        self.x_train = np.asarray(self.x_train, dtype="uint8")
        self.x_test = np.asarray(self.x_test, dtype="uint8")

    def get_input_shape(self, recurrent=False):
        """
        Args:
            recurrent:
        """
        if recurrent:
            img_rows, img_cols = (
                self.x_train_recurrent.shape[2],
                self.x_train_recurrent.shape[3],
            )
            input_shape = (
                self.x_train_recurrent.shape[1],
                img_rows,
                img_cols,
                self.x_train_recurrent.shape[4],
            )
            return input_shape
        else:
            if len(self.x_train.shape) == 5:
                img_rows, img_cols = self.x_train.shape[2], self.x_train.shape[3]
                input_shape = (img_rows, img_cols, self.x_train.shape[4])
                return input_shape
            elif len(self.x_train.shape) == 4:
                img_rows, img_cols = self.x_train.shape[1], self.x_train.shape[2]
                input_shape = (img_rows, img_cols, self.x_train.shape[3])
                return input_shape
            else:
                raise NotImplementedError

    # TODO: parallelizs
    def downscale_frames(self, factor=0.5):
        im_re = []
        for el in tqdm(self.x_train):
            im_re.append(imresize(el, factor))
        self.x_train = np.asarray(im_re)
        im_re = []
        for el in tqdm(self.x_test):
            im_re.append(imresize(el, factor))
        self.x_test = np.asarray(im_re)

    def prepare_data(self, downscale=0.5, remove_behaviors=[], flatten=False):
        print("preparing data")
        self.change_dtype()

        for behavior in remove_behaviors:
            self.remove_behavior(behavior=behavior)

        if downscale:
            self.downscale_frames(factor=downscale)
        if self.config["normalize_data"]:
            self.normalize_data()
        if self.config["encode_labels"]:
            self.encode_labels()
        print("labels encoded")
        if self.config["use_class_weights"]:
            print("calc class weights")
            self.class_weights = class_weight.compute_class_weight(
                "balanced", np.unique(self.y_train), self.y_train
            )

        if self.config["undersample_data"]:
            print("undersampling data")
            self.undersample_data()

        print("preparing recurrent data")
        self.create_recurrent_data()
        print("preparing flattened data")
        if flatten:
            self.create_flattened_data()

        print("categorize data")
        self.categorize_data(self.config["num_classes"], recurrent=True)

        print("data ready")
