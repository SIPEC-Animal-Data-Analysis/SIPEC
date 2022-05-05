"""
SIPEC
MARKUS MARKS
MODEL CLASS
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from SwissKnife.architectures import (
    pretrained_recognition,
    recurrent_model_lstm,
    recurrent_model_tcn,
)

# TODO: import these DL utils into this class
from SwissKnife.utils import get_optimizer, train_model


# TODO: presets for parameters
# TODO: make sure modulization is correct -> rather have for each task one inheretance model
# TODO: make param list dict?
class Model:
    """TODO: Fill in description"""
    def __init__(self, config=None):
        """Initialize model with default parameters, which can be updated
        through config.

        Args:
            config: Config for updated hyperparameters.
        """
        # TODO: fix hardcoded here
        self.architecture = ""
        self.callbacks = []
        self.scheduler_factor = 1.1
        self.scheduler_lr = 0.00035
        self.scheduler_lower_lr = 0.0000001
        self.recognition_model_epochs = 20
        self.recognition_model_batch_size = 16
        self.sequential_model_epochs = 20
        self.sequential_model_batch_size = 16
        self.augmentation = None
        self.optim = None
        self.recognition_model = None
        self.recognition_model_history = None
        self.class_weight = None
        self.recognition_model_loss = "crossentropy"
        self.sequential_model_loss = "crossentropy"

        if config:
            self.update_params(config)

    def load_recognition_model(self, path):
        """Loads recognition model.
        Args:
            path: Path to existing recognition model.
        """
        self.recognition_model = load_model(path)

    def set_recognition_model(self, architecture, input_shape, num_classes):
        """Sets recognitions model.
        Args:
            architecture:
            input_shape:
            num_classes:
        """

        if architecture in [
            "densenet",
            "efficientnet",
            "efficientnet4",
            "resnet",
            "xception",
            "inceptionResnet",
            "resnet150",
            "inceptionv3",
            "classification_small",
            "classification_large",
        ]:
            self.recognition_model = pretrained_recognition(
                architecture, input_shape, num_classes, fix_layers=False
            )
        elif architecture == "idtracker":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def set_sequential_model(self, architecture, input_shape, num_classes):
        """
        Args:
            architecture:
            input_shape:
            num_classes:
        """
        if architecture == "tcn":
            self.sequential_model = recurrent_model_tcn(
                self.recognition_model,
                input_shape,
                classes=num_classes,
            )
        if architecture == "lstm":
            self.sequential_model = recurrent_model_lstm(
                self.recognition_model,
                input_shape,
                classes=num_classes,
            )
        # reset callbacks
        self.callbacks = []
        # reset augmentations
        self.augmentation = None

    def train_recognition_network(self, dataloader):
        """
        Args:
            dataloader:
        """
        self.recognition_model, self.recognition_model_history = train_model(
            self.recognition_model,
            self.optim,
            self.recognition_model_epochs,
            self.recognition_model_batch_size,
            dataloader=dataloader,
            callbacks=self.callbacks,
            loss=self.recognition_model_loss,
            # TODO: activate augmentation
            augmentation=self.augmentation,
            class_weights=self.class_weight,
        )

    def train_sequential_network(self, dataloader):
        """
        Args:
            dataloader:
        """
        self.sequential_model, self.sequential_model_history = train_model(
            self.sequential_model,
            self.optim,
            self.sequential_model_epochs,
            self.sequential_model_batch_size,
            dataloader=dataloader,
            callbacks=self.callbacks,
            loss=self.sequential_model_loss,
            # TODO: activate augmentation
            augmentation=self.augmentation,
            class_weights=self.class_weight,
            sequential=True,
        )

    def predict(self, data, model="recognition", threshold=None, default_behavior=1):
        """
        Args:
            data:
            model:
            threshold:
        """
        if model == "recognition":
            # TODO: implement recognition vs sequential
            if len(data.shape) == 3:
                prediction = self.recognition_model.predict(
                    np.expand_dims(data, axis=0)
                )
            else:
                prediction = self.recognition_model.predict(data)
        else:
            prediction = self.sequential_model.predict(data)

        if threshold is None:
            return prediction, np.argmax(prediction).astype(int)
        prediction_idxs = list(range(len(prediction)))
        non_default_predictions = (
            prediction[:default_behavior] + prediction[default_behavior + 1 :]
        )
        non_default_prediction_idxs = (
            prediction_idxs[:default_behavior]
            + prediction_idxs[default_behavior + 1 :]
        )

        if np.max(non_default_predictions) > threshold:
            return prediction, np.argmax(non_default_prediction_idxs).astype(int)
        # TODO: fixme: not always behavior but also identification
        return prediction, default_behavior

    def predict_sequential(self, data):
        # TODO: implement recognition vs sequential
        """
        Args:
            data:
        """
        if len(data.shape) == 3:
            prediction = self.sequential_model.predict(np.expand_dims(data, axis=0))
            return np.argmax(prediction)
        # TODO: implement batches
        predictions = []
        for dat in data:
            prediction = self.sequential_model.predict(np.expand_dims(dat, axis=0))
            predictions.append(prediction)
        predictions = np.asarray(predictions)
        return np.argmax(predictions, axis=-1)

    def export_training_details(self):
        """TODO: Fill in description"""
        raise NotImplementedError

    def save_model(self, path):
        """TODO: Fill in description"""
        self.recognition_model.save(path + "_recognition")
        self.sequential_model.save(path + "_sequential")

    def load_model(self, recognition_path=None, sequential_path=None):
        """TODO: Fill in description"""
        if recognition_path is not None:
            self.recognition_model = load_model(recognition_path)
        if sequential_path is not None:
            self.sequential_model = load_model(sequential_path)

    def fix_recognition_layers(self, num=None):
        """
        Args:
            num:
        """
        if num is not None:
            for layer in self.recognition_model.layers[:num]:
                layer.trainable = False
        else:
            for layer in self.recognition_model.layers:
                layer.trainable = False

    def remove_classification_layers(self):
        """TODO: Fill in description"""
        self.recognition_model.layers.pop()
        self.recognition_model.layers.pop()

    def set_optimizer(self, name, lr=0.00035):
        """
        Args:
            name:
            lr:
        """
        self.optim = get_optimizer(name, lr)

    # TODO: fix hardcoded here
    def scheduler(self, epoch):
        """
        Args:
            epoch:
        """
        lr = self.scheduler_lr
        factor = self.scheduler_factor
        new_lr = lr / np.power(factor, epoch)
        print("new lr" + str(new_lr))
        print("lower lr" + str(self.scheduler_lower_lr))
        new_lr = np.max([new_lr, self.scheduler_lower_lr])
        print("new lr" + str(new_lr))
        print("epoch " + str(epoch))
        print("reducing to new learning rate" + str(new_lr))
        return new_lr

    def set_lr_scheduler(self):
        """TODO: Fill in description"""
        lr_callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        self.callbacks.append(lr_callback)

    def add_callbacks(self, callbacks):
        """
        Args:
            callbacks:
        """
        for callback in callbacks:
            self.callbacks.append(callback)

    def set_augmentation(self, augmentation):
        """
        Args:
            augmentation:
        """
        self.augmentation = augmentation

    def set_class_weight(self, class_weight):
        """
        Args:
            class_weight:
        """
        self.class_weight = class_weight

    def update_params(self, config):
        """
        Args:
            config:
        """
        pass


if __name__ == "__main__":
    pass

# TODO: finish IdNet here
# class IdNet(Model):
#     def __init__(self):
#         super(IdNet, self).__init__()
#
#     def detect_primate(_img, _model, classes, threshold):
#         prediction = _model.predict(np.expand_dims(_img, axis=0))
#         if prediction.max() > threshold:
#             return classes[np.argmax(prediction)], prediction.max()
#         else:
#             return "None detected", prediction.max()
