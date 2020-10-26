# SIPEC
# MARKUS MARKS
# MODEL CLASS
import numpy as np
import tensorflow as tf
from keras.engine.saving import load_model

from SwissKnife.architectures import (
    pretrained_recognition,
    recurrent_model_tcn,
    recurrent_model_lstm,
)
from SwissKnife.utils import get_optimizer

# TODO: import these DL utils into this class
from SwissKnife.utils import train_model


class Model:
    def __init__(self, config=None):
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
        self.recognition_model = load_model(path)

    def set_recognition_model(self, architecture, input_shape, num_classes):
        if architecture in [
            "densenet",
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
        if architecture == "tcn":
            self.sequential_model = recurrent_model_tcn(
                self.recognition_model, input_shape, classes=num_classes,
            )
        if architecture == "lstm":
            self.sequential_model = recurrent_model_lstm(
                self.recognition_model, input_shape, classes=num_classes,
            )
        # reset callbacks
        self.callbacks = []
        # reset augmentations
        self.augmentation = None

    def train_recognition_network(self, dataloader):
        self.recognition_model, self.recognition_model_history = train_model(
            self.recognition_model,
            self.optim,
            self.recognition_model_epochs,
            self.recognition_model_batch_size,
            (dataloader.x_train, dataloader.y_train),
            data_val=(dataloader.x_test, dataloader.y_test),
            callbacks=self.callbacks,
            loss=self.recognition_model_loss,
            # TODO: activate augmentation
            augmentation=self.augmentation,
            class_weights=self.class_weight,
        )

    def train_sequential_network(self, dataloader):
        self.sequential_model, self.sequential_model_history = train_model(
            self.sequential_model,
            self.optim,
            self.sequential_model_epochs,
            self.sequential_model_batch_size,
            (dataloader.x_train_recurrent, dataloader.y_train_recurrent),
            data_val=(dataloader.x_test_recurrent, dataloader.y_test_recurrent),
            callbacks=self.callbacks,
            loss=self.sequential_model_loss,
            # TODO: activate augmentation
            augmentation=self.augmentation,
            class_weights=self.class_weight,
        )

    def predict(self, data, model="recognition", threshold=None):
        if model == "recognition":
            # TODO: implement recognition vs sequential
            if len(data.shape) == 3:
                prediction = self.recognition_model.predict(
                    np.expand_dims(data, axis=0)
                )
                if threshold is None:
                    return np.argmax(prediction).astype(int)
                else:
                    if prediction.max() > threshold:
                        return np.argmax(prediction).astype(int)
                    else:
                        return "None detected"
            else:
                prediction = self.recognition_model.predict(data)
                return np.argmax(prediction, axis=-1).astype(int)
        else:
            prediction = self.sequential_model.predict(data)
            return np.argmax(prediction, axis=-1).astype(int)

    def predict_sequential(self, data):
        # TODO: implement recognition vs sequential
        if len(data.shape) == 3:
            prediction = self.sequential_model.predict(np.expand_dims(data, axis=0))
            return np.argmax(prediction)
        else:
            # TODO: implement batches
            predictions = []
            for dat in data:
                prediction = self.sequential_model.predict(np.expand_dims(dat, axis=0))
                predictions.append(prediction)
            predictions = np.asarray(predictions)
            return np.argmax(predictions, axis=-1)

    def export_training_details(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def fix_recognition_layers(self, num=None):
        if num is not None:
            for layer in self.recognition_model.layers[:num]:
                layer.trainable = False
        else:
            for layer in self.recognition_model.layers:
                layer.trainable = False

    def remove_classification_layers(self):
        self.recognition_model.layers.pop()
        self.recognition_model.layers.pop()

    def set_optimizer(self, name, lr=0.00035):
        self.optim = get_optimizer(name, lr)

    def scheduler(self, epoch):
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
        lr_callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        self.callbacks.append(lr_callback)

    def add_callbacks(self, callbacks):
        for callback in callbacks:
            self.callbacks.append(callback)

    def set_augmentation(self, augmentation):
        self.augmentation = augmentation

    def set_class_weight(self, class_weight):
        self.class_weight = class_weight


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
