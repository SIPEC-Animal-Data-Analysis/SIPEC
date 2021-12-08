# SIPEC
# MARKUS MARKS
# Dataloader
from skimage.registration import optical_flow_tvl1
from tensorflow import keras
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.externals._pilutil import imresize
from sklearn.utils import class_weight
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import pickle


# adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        x_train,
        y_train,
        look_back,
        type="recognition",
        batch_size=32,
        shuffle=True,
        temporal_causal=True,
    ):
        # self.dim = dim
        self.batch_size = batch_size
        self.look_back = look_back
        self.list_IDs = np.array(range(self.look_back, len(x_train) - self.look_back))
        # self.n_channels = n_channels
        self.shuffle = shuffle
        # self.augmentation = augmentation
        self.type = type
        self.x_train = x_train
        self.y_train = y_train
        self.dlc_train_flat = None
        self.temporal_causal = temporal_causal

        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            if self.type == "recognition":
                X.append(self.x_train[ID])
            else:
                if self.temporal_causal:
                    X.append(self.x_train[ID - (2 * self.look_back): ID])
                else:
                    X.append(self.x_train[ID - self.look_back: ID + self.look_back])

            y.append(self.y_train[ID])
            # _y = self.y_train[ID - self.look_back: ID + self.look_back]
            # y.append(self.label_encoder.transform(_y))

        return np.asarray(X).astype("float32"), np.asarray(y).astype("int")


def create_dataset(dataset, look_back=5, oneD=False):
    # """Create a recurrent dataset from array.
    # Args:
    #     dataset: Numpy/List of dataset.
    #     look_back: Number of future/past timepoints to add to current timepoint.
    #     oneD: Boolean flag whether data is one dimensional or not.
    # """
    """Summary line.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        dataset
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
        self.num_classes = len(np.unique(y_train))

        self.look_back = self.config["look_back"]

        self.label_encoder = None

        self.x_train_recurrent = None
        self.x_test_recurrent = None
        self.y_train_recurrent = None
        self.y_test_recurrent = None

        self.use_generator = False

    def encode_label(self, label):
        return self.label_encoder.transform(label)

    def encode_labels(self):
        self.label_encoder = preprocessing.LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.y_train)
        self.y_test = self.label_encoder.transform(self.y_test)

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

    def create_recurrent_labels(self, only_test=False):

        if only_test:
            self.y_test_recurrent = self.y_test[self.look_back : -self.look_back]
            self.y_test = self.y_test[self.look_back : -self.look_back]
        else:
            self.y_train_recurrent = self.y_train[self.look_back : -self.look_back]
            self.y_test_recurrent = self.y_test[self.look_back : -self.look_back]

            self.y_train = self.y_train[self.look_back : -self.look_back]
            self.y_test = self.y_test[self.look_back : -self.look_back]

    def create_recurrent_data(self, oneD=False, recurrent_labels=True, only_test=False):
        """
        Args:
            oneD:
        """
        if only_test:
            self.x_test_recurrent = create_dataset(
                self.x_test, self.look_back, oneD=oneD
            )
        else:
            self.x_train_recurrent = create_dataset(
                self.x_train, self.look_back, oneD=oneD
            )
            self.x_test_recurrent = create_dataset(
                self.x_test, self.look_back, oneD=oneD
            )

        # also shorten normal data so all same length
        if only_test:
            self.x_test = self.x_test[self.look_back : -self.look_back]
        else:
            self.x_test = self.x_test[self.look_back : -self.look_back]
            self.x_train = self.x_train[self.look_back : -self.look_back]

        if recurrent_labels:
            self.create_recurrent_labels(only_test=only_test)

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
        if hasattr(self, 'dlc_train_recurrent'):
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
            0.2, sampling_strategy="majority", random_state=42
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
            self.x_train, self.y_train = random_under_sampler.fit_resample(
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

        try:
            rec_shape = self.x_train_recurrent.shape[1]
        except AttributeError:
            rec_shape = int(2 * self.look_back)
        if recurrent:
            input_shape = (
                # new version here:
                rec_shape,
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
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

    def prepare_data(
        self, downscale=0.5, remove_behaviors=[], flatten=False, use_generator=True
    ):
        print("preparing data")
        self.change_dtype()

        for behavior in remove_behaviors:
            self.remove_behavior(behavior=behavior)
        if downscale:
            self.downscale_frames(factor=downscale)
        if self.config["normalize_data"]:
            self.normalize_data()
        if self.config["do_flow"]:
            self.create_flow_data()
        if self.config["encode_labels"]:
            print("test")
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
        if self.config["use_generator"]:
            self.categorize_data(self.num_classes, recurrent=False)
        else:
            print("preparing recurrent data")
            self.create_recurrent_data()
            print("preparing flattened data")
            if flatten:
                self.create_flattened_data()
            print("categorize data")
            self.categorize_data(self.num_classes, recurrent=True)

        print("data ready")

    def flow_single(self, fr1, fr2):
        if len(fr1.shape) > 2:
            fr2 = fr2[:, :, 0]
            fr1 = fr1[:, :, 0]
        v, u = optical_flow_tvl1(fr1, fr2)
        v = np.expand_dims(v, axis=-1)
        u = np.expand_dims(u, axis=-1)
        s = np.stack([u, v], axis=2)[:, :, :, 0]
        return v

    def do_flow(self, videodata, num_cores=(multiprocessing.cpu_count()) * 2):

        flow_data = Parallel(n_jobs=num_cores)(
            delayed(self.flow_single)(videodata[i], videodata[i - 1])
            for i in tqdm(range(1, len(videodata)))
        )
        flow_data = np.array(flow_data)
        flow_data = np.vstack([np.expand_dims(flow_data[0], axis=0), flow_data])

        return flow_data

    def create_flow_data(self):

        flow_data_train = self.do_flow(self.x_train)
        # self.x_train = np.stack([flow_data_train, self.x_train])
        a = np.swapaxes(flow_data_train, 0, -1)
        b = np.swapaxes(self.x_train, 0, -1)
        self.x_train = np.swapaxes(np.vstack([a, b]), 0, -1)
        # self.x_train = self.x_train[:, :, :, 0, :]

        flow_data_test = self.do_flow(self.x_test)
        # self.x_test = np.stack([flow_data_test, self.x_test], axis=-1)
        a = np.swapaxes(flow_data_test, 0, -1)
        b = np.swapaxes(self.x_test, 0, -1)
        self.x_test = np.swapaxes(np.vstack([a, b]), 0, -1)
        # self.x_test = self.x_test[:, :, :, 0, :]

        pass

    def expand_dims(self, axis=-1):

        self.x_train = np.expand_dims(self.x_train, axis=axis)
        self.x_test = np.expand_dims(self.x_test, axis=axis)

    def save(self, path):
        self.x_train = None
        self.x_train_recurrent = None
        self.x_test = None
        self.x_test_recurrent = None
        self.y_train = None
        self.y_train_recurrent = None
        self.y_test = None
        self.y_test_recurrent = None
        pickle.dump(self, open(path, "wb"))
