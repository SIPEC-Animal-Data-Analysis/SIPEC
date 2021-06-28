# SIPEC
# MARKUS MARKS
# MODEL ARCHITECTURES
import tensorflow as tf
from tensorflow.keras import regularizers, Model
from tensorflow.keras.applications import (
    DenseNet121,
    DenseNet201,
    ResNet50,
    ResNet101,
    InceptionResNetV2,
    Xception,
    NASNetLarge,
    InceptionV3,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
)
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
    Activation,
    TimeDistributed,
    LSTM,
    Input,
    Bidirectional,
    MaxPooling2D,
    Conv1D,
    SpatialDropout1D,
    ZeroPadding2D,
    concatenate,
    GaussianNoise,
    Conv2DTranspose,
    UpSampling2D,
    Reshape,
    LeakyReLU,
)
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.applications.efficientnet import EfficientNetB1


def posenet(
    input_shape,
    num_classes,
    backbone="efficientnetb5",
    fix_backbone=False,
    gaussian_noise=0.05,
    features=256,
    bias=False,
):
    """Model that implements SIPEC:PoseNet architecture.

    This model uses an EfficientNet backbone and deconvolves generated features into landmarks in imagespace.
    It operates on single images and can be used in conjuntion with SIPEC:SegNet to perform top-down pose estimation.

    Parameters
    ----------
    input_shape : keras compatible input shape (W,H,Channels)
        keras compatible input shape (features,)
    num_classes : int
        Number of joints/landmarks to detect.
    backbone : str
        Backbone/feature detector to use, default is EfficientNet5. Choose smaller/bigger backbone depending on GPU memory.
    gaussian_noise : float
        Kernel size of gaussian noise layers to use.
    features : int
        Number of feature maps to generate at each level.
    bias : bool
        Use bias for deconvolutional layers.

    Returns
    -------
    keras.model
        SIPEC:PoseNet
    """
    if backbone == "efficientnetb5":
        recognition_model = EfficientNetB5(
            include_top=False,
            input_shape=input_shape,
            pooling=None,
            weights="imagenet",
        )
    elif backbone == "efficientnetb7":
        recognition_model = EfficientNetB7(
            include_top=False,
            input_shape=input_shape,
            pooling=None,
            weights="imagenet",
        )
    elif backbone == "efficientnetb1":
        recognition_model = EfficientNetB1(
            include_top=False,
            input_shape=input_shape,
            pooling=None,
            weights="imagenet",
        )
    else:
        raise NotImplementedError

    new_input = Input(
        batch_shape=(None, input_shape[0], input_shape[1], input_shape[2])
    )

    if fix_backbone:
        for layer in recognition_model.layers:
            layer.trainable = False

    x = Conv2D(3, kernel_size=(1, 1), strides=(1, 1))(new_input)
    x = recognition_model(x)

    for i in range(4):
        x = Conv2DTranspose(
            features, kernel_size=(2, 2), strides=(2, 2), padding="valid", use_bias=bias
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = GaussianNoise(gaussian_noise)(x)

    x = Conv2DTranspose(
        features, kernel_size=(2, 2), strides=(2, 2), padding="valid", use_bias=bias
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(
        num_classes, kernel_size=(1, 1), strides=(1, 1), padding="valid"
    )(x)

    x = Activation("sigmoid")(x)

    model = Model(inputs=new_input, outputs=x)
    model.summary()

    return model


def classification_scratch(input_shape):
    """
    Args:
        input_shape:
    """
    activation = "tanh"
    dropout = 0.3
    # conv model

    model = Sequential()

    model.add(
        Conv2D(
            64,
            kernel_size=(4, 4),
            strides=(4, 4),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(512, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    #     model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation("softmax"))

    return model


def classification_large(input_shape):
    """
    Args:
        input_shape:
    """
    activation = "tanh"
    dropout = 0.1
    # conv model

    model = Sequential()

    model.add(
        Conv2D(
            64,
            kernel_size=(4, 4),
            strides=(4, 4),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(512, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(1024, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    #     model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation("softmax"))

    return model


def classification_small(input_shape, num_classes):
    """
    Args:
        input_shape:
        num_classes:
    """
    dropout = 0.33
    # conv model

    model = Sequential()

    model.add(
        Conv2D(
            64,
            kernel_size=(4, 4),
            strides=(4, 4),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(512, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    #     model.add(Dropout(dropout))

    #     model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    return model


def dlc_model_sturman(input_shape, num_classes):
    """Model that implements behavioral classification based on Deeplabcut generated features as in Sturman et al.

    Reimplementation of the model used in the publication Sturman et al. that performs action recognition on top of pose estimation

    Parameters
    ----------
    input_shape : keras compatible input shape (W,H,Channels)
        keras compatible input shape (features,)
    num_classes : int
        Number of behaviors to classify.

    Returns
    -------
    keras.model
        Sturman et al. model
    """
    model = Sequential()

    model.add(
        Dense(
            256, input_shape=(input_shape[-1],), kernel_regularizer=regularizers.l2(0.0)
        )
    )
    model.add(Activation("relu"))
    model.add(Dropout(0.4))

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.0)))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    # TODO: parametrize # behaviors
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    return model


def dlc_model(input_shape, num_classes):
    """Model for classification on top of pose estimation.

    Classification model for behavior, operating on pose estimation. This model has more free parameters than Sturman et al.

    Parameters
    ----------
    input_shape : keras compatible input shape (W,H,Channels)
        keras compatible input shape (features,)
    num_classes : int
        Number of behaviors to classify.

    Returns
    -------
    keras.model
        behavior (from pose estimates) model
    """
    dropout = 0.3

    model = Sequential()

    model.add(Dense(256, input_shape=(input_shape[-1],)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    # TODO: parametrize # behaviors
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    return model


def recurrent_model_old(
    recognition_model, recurrent_input_shape, classes=4, recurrent_dropout=None
):
    """
    Args:
        recognition_model:
        recurrent_input_shape:
        classes:
        recurrent_dropout:
    """
    input_sequences = Input(shape=recurrent_input_shape)
    sequential_model_helper = TimeDistributed(recognition_model)(input_sequences)

    if recurrent_dropout:
        # TODO: adjust bidirectional
        k = LSTM(units=128, return_sequences=True, recurrent_dropout=recurrent_dropout)(
            sequential_model_helper
        )
        k = LSTM(units=64, return_sequences=True, recurrent_dropout=recurrent_dropout)(
            k
        )
        k = LSTM(units=32, return_sequences=False, recurrent_dropout=recurrent_dropout)(
            k
        )

    else:
        # As of TF 2, one can just use LSTM and there is no CuDNNLSTM
        k = Bidirectional(LSTM(units=128, return_sequences=True))(
            sequential_model_helper
        )
        k = Bidirectional(LSTM(units=64, return_sequences=True))(k)
        k = Bidirectional(LSTM(units=32, return_sequences=False))(k)

    dout = 0.3
    k = Dense(256)(k)
    k = Activation("relu")(k)
    k = Dropout(dout)(k)
    k = Dense(128)(k)
    k = Activation("relu")(k)
    k = Dropout(dout)(k)
    k = Dense(64)(k)
    k = Dropout(dout)(k)
    k = Activation("relu")(k)
    k = Dense(32)(k)
    k = Activation("relu")(k)
    # TODO: modelfy me!
    k = Dense(classes)(k)
    k = Activation("softmax")(k)

    sequential_model = Model(inputs=input_sequences, outputs=k)

    return sequential_model


def recurrent_model_tcn(
    recognition_model,
    recurrent_input_shape,
    classes=4,
):
    """Recurrent architecture for classification of temporal sequences of images based on temporal convolution architecture (TCN).
    This architecture is used for BehaviorNet in SIPEC.

    Parameters
    ----------
    recognition_model : keras.model
        Pretrained recognition model that extracts features for individual frames.
    recurrent_input_shape : np.ndarray - (Time, Width, Height, Channels)
        Shape of the images over time.
    classes : int
        Number of behaviors to recognise.

    Returns
    -------
    keras.model
        BehaviorNet
    """
    input_sequences = Input(shape=recurrent_input_shape)
    sequential_model_helper = TimeDistributed(recognition_model)(input_sequences)
    k = BatchNormalization()(sequential_model_helper)

    # TODO: config me!
    filters = 64
    kernel_size = 2
    # dout = 0.01
    act_fcn = "relu"
    k = Conv1D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=1,
        kernel_initializer="he_normal",
    )(k)
    k = BatchNormalization()(k)
    # k = Activation(LeakyReLU(alpha=0.3))(k)
    # k = Activation(Activation('relu'))(k)
    # k = wave_net_activation(k)
    k = Activation(act_fcn)(k)
    # k = SpatialDropout1D(rate=dout)(k)

    k = Conv1D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=2,
        kernel_initializer="he_normal",
    )(k)
    k = BatchNormalization()(k)
    # k = Activation(LeakyReLU(alpha=0.3))(k)
    # k = Activation(Activation('relu'))(k)
    # k = wave_net_activation(k)
    k = Activation(act_fcn)(k)
    # k = SpatialDropout1D(rate=dout)(k)

    k = Conv1D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=4,
        kernel_initializer="he_normal",
    )(k)
    k = BatchNormalization()(k)
    # k = Activation(LeakyReLU(alpha=0.3))(k)
    # k = Activation(Activation('relu'))(k)
    # k = wave_net_activation(k)
    k = Activation(act_fcn)(k)
    # k = SpatialDropout1D(rate=dout)(k)

    k = Conv1D(
        1,
        kernel_size=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer="he_normal",
    )(k)
    k = Activation(Activation("relu"))(k)
    k = Flatten()(k)

    k = Dense(64)(k)
    k = Activation(LeakyReLU(alpha=0.3))(k)
    # k = Dropout(dout)(k)
    k = Dense(32)(k)
    k = Activation(LeakyReLU(alpha=0.3))(k)
    # k = Dropout(dout)(k)
    k = Dense(16)(k)
    k = Activation(LeakyReLU(alpha=0.3))(k)

    k = Dense(classes)(k)
    k = Activation("softmax")(k)

    sequential_model = Model(inputs=input_sequences, outputs=k)

    return sequential_model


def recurrent_model_lstm(
    recognition_model, recurrent_input_shape, classes=4, recurrent_dropout=None
):
    """Recurrent architecture for classification of temporal sequences of images based on LSTMs or GRUs.
    This architecture is used for IdNet in SIPEC.

    Parameters
    ----------
    recognition_model : keras.model
        Pretrained recognition model that extracts features for individual frames.
    recurrent_input_shape : np.ndarray - (Time, Width, Height, Channels)
        Shape of the images over time.
    classes : int
        Number of behaviors to recognise.
    recurrent_dropout : float
        Recurrent dropout factor to use.

    Returns
    -------
    keras.model
        IdNet
    """
    input_sequences = Input(shape=recurrent_input_shape)
    sequential_model_helper = TimeDistributed(recognition_model)(input_sequences)
    k = BatchNormalization()(sequential_model_helper)

    dout = 0.2

    if recurrent_dropout:
        # TODO: adjust bidirectional
        k = LSTM(units=128, return_sequences=True, recurrent_dropout=recurrent_dropout)(
            k
        )
        k = LSTM(units=64, return_sequences=True, recurrent_dropout=recurrent_dropout)(
            k
        )
        k = LSTM(units=32, return_sequences=False, recurrent_dropout=recurrent_dropout)(
            k
        )

    else:
        # As of TF 2, one can just use LSTM and there is no CuDNNGRU
        k = Bidirectional(GRU(units=128, return_sequences=True))(k)
        k = Activation(LeakyReLU(alpha=0.3))(k)
        k = Bidirectional(GRU(units=64, return_sequences=True))(k)
        k = Activation(LeakyReLU(alpha=0.3))(k)
        k = Bidirectional(GRU(units=32, return_sequences=False))(k)
        k = Activation(LeakyReLU(alpha=0.3))(k)

    # k = Dense(256)(k)
    # k = Activation('relu')(k)
    # k = Dropout(dout)(k)

    # k = Dense(128)(k)
    # k = Activation("relu")(k)
    # k = Dropout(dout)(k)
    # k = Dense(64)(k)

    k = Dropout(dout)(k)
    k = Dense(classes)(k)
    k = Activation("softmax")(k)

    sequential_model = Model(inputs=input_sequences, outputs=k)

    return sequential_model


# TODO: adaptiv size
def pretrained_recognition(model_name, input_shape, num_classes, fix_layers=True):
    """This returns the model architecture for a model that operates on images and is pretrained with imagenet weights.
    This architecture is used for IdNet and BehaviorNet as backbone in SIPEC and is referred to as RecognitionNet.

    Parameters
    ----------
    model_name : keras.model
        Name of the pretrained recognition model to use (names include: "xception, "resnet", "densenet")
    input_shape : np.ndarray - (Time, Width, Height, Channels)
        Shape of the images over time.
    num_classes : int
        Number of behaviors to recognise.
    fix_layers : bool
        Recurrent dropout factor to use.

    Returns
    -------
    keras.model
        RecognitionNet
    """
    if model_name == "xception":
        recognition_model = Xception(
            include_top=False,
            input_shape=(75, 75, 3),
            # input_shape=(input_shape[0], input_shape[1], 3),
            pooling="avg",
            weights="imagenet",
        )
        # TODO: config me!
        # just if segmentation  mask is small (for 35)
        # for i in range(0, 17):
        #     recognition_model.layers.pop(0)

    elif model_name == "resnet":
        recognition_model = ResNet50(
            include_top=False,
            input_shape=(input_shape[0], input_shape[1], 3),
            pooling="avg",
            weights="imagenet",
        )
    elif model_name == "resnet150":
        recognition_model = ResNet152(
            include_top=False,
            input_shape=(input_shape[0], input_shape[1], 3),
            pooling="avg",
            weights="imagenet",
        )

    elif model_name == "inceptionv3":
        recognition_model = InceptionV3(
            include_top=False,
            input_shape=(max(input_shape[0], 75), max(input_shape[1], 75), 3),
            pooling="avg",
            weights="imagenet",
        )

    elif model_name == "classification_small":
        recognition_model = classification_small(input_shape, num_classes)

    elif model_name == "classification_large":
        recognition_model = classification_large(input_shape, num_classes)

    elif model_name == "densenet":
        recognition_model = DenseNet201(
            include_top=False,
            input_shape=(input_shape[0], input_shape[1], 3),
            pooling="avg",
            weights="imagenet",
        )
    elif model_name == "inceptionResnet":
        recognition_model = InceptionResNetV2(
            include_top=False,
            input_shape=(input_shape[0], input_shape[1], 3),
            pooling="avg",
            weights="imagenet",
        )
    else:
        raise NotImplementedError

    new_input = Input(
        batch_shape=(None, input_shape[0], input_shape[1], input_shape[2])
    )

    if model_name == "inceptionResnet":
        x = Conv2D(3, kernel_size=(1, 1), strides=(1, 1))(new_input)
        x = Activation("relu")(x)
        x = recognition_model(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = Dense(64)(x)
        x = Activation("relu")(x)
        x = Dense(num_classes)(x)
        x = Activation("softmax")(x)

    else:
        x = Conv2D(3, kernel_size=(1, 1), strides=(1, 1))(new_input)
        if model_name == "xception" or "inceptionv3":
            x = ZeroPadding2D(padding=(2, 2))(x)
        x = recognition_model(x)
        x = BatchNormalization()(x)
        if model_name == "xception":
            dout = 0.25
            x = Dropout(dout)(x)
        if model_name == "densenet":
            # dout = 0.25
            dout = 0.5
            x = Dropout(dout)(x)
            # x = Dense(256)(x)
            # x = Activation('relu')(x)
            # x = Dropout(dout)(x)
            # x = Dense(128)(x)
            # x = Activation('relu')(x)
            # x = Dropout(dout)(x)
            # x = Dense(64)(x)
            # x = Activation('relu')(x)
        x = Dense(num_classes)(x)
        x = Activation("softmax")(x)

    recognition_model = Model(inputs=new_input, outputs=x)
    recognition_model.summary()

    return recognition_model


# Model with hyperparameters from idtracker.ai


def idtracker_ai(input_shape, classes):
    """Implementation of the idtracker.ai identification module as described in the supplementary of Romero-Ferrero et al.

    Parameters
    ----------
    input_shape : keras compatible input shape (W,H,Channels)
        keras compatible input shape (features,)
    num_classes : int
        Number of behaviors to classify..

    Returns
    -------
    keras.model
        idtracker.ai identification module
    """

    activation = "tanh"
    dropout = 0.2
    # conv model

    model = Sequential()

    model.add(
        Conv2D(
            16,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(Activation("relu"))

    model.add(
        MaxPooling2D(
            strides=(2, 2),
        )
    )

    model.add(
        Conv2D(
            64,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(Activation("relu"))

    model.add(
        MaxPooling2D(
            strides=(2, 2),
        )
    )

    model.add(
        Conv2D(
            100,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(Activation("relu"))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation("relu"))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model


def SkipConNet(x_train, dropout):
    """
    Args:
        x_train:
        dropout:
    """
    inputs = Input(shape=(x_train.shape[1], 1))

    dout = dropout

    features = 128

    # a layer instance is callable on a tensor, and returns a tensor
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(
        inputs
    )
    x = GaussianNoise(0.5)(x)
    x = BatchNormalization()(x)
    x_1 = Activation("relu")(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x_2 = Activation("relu")(x)
    # x = Dropout(dout)(x_2)

    x = concatenate([x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x_3 = Activation("relu")(x)
    # x = Dropout(dout)(x_3)

    x = concatenate([x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x_4 = Activation("relu")(x)
    # x = Dropout(dout)(x_4)

    x = concatenate([x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x_5 = Activation("relu")(x)
    # x = Dropout(dout)(x_5)

    x = concatenate([x_5, x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    # x_6 = Activation('relu')(x)
    # x = Dropout(dout)(x_6)
    x_6 = Activation("relu")(x)

    x = concatenate([x_6, x_5, x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    # x_6 = Activation('relu')(x)
    # x = Dropout(dout)(x_6)
    x_7 = Activation("relu")(x)

    x = concatenate([x_7, x_6, x_5, x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    # x_6 = Activation('relu')(x)
    # x = Dropout(dout)(x_6)
    x_8 = Activation("relu")(x)

    x = concatenate([x_8, x_7, x_6, x_5, x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    # x_6 = Activation('relu')(x)
    # x = Dropout(dout)(x_6)
    x_9 = Activation("relu")(x)

    x = concatenate([x_9, x_7, x_6, x_5, x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    # x_6 = Activation('relu')(x)
    # x = Dropout(dout)(x_6)
    x = Activation("relu")(x)

    x = Flatten()(x)
    x = Dropout(dout)(x)

    x = Dense(1024, kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = Dropout(dout)(x)
    x = Dense(1024, kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = Dropout(dout)(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    predictions = Dense(3, activation="softmax")(x)
    model_mlp = Model(inputs=inputs, outputs=predictions)

    return model_mlp
