# SIPEC
# MARKUS MARKS
# MODEL ARCHITECTURES

from tensorflow.keras import regularizers, Model
from tensorflow.keras.applications import (
    DenseNet121,
    DenseNet201,
    ResNet50,
    ResNet101,
    InceptionResNetV2,
    Xception,
    # ResNet152,
    NASNetLarge,
    InceptionV3,
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
)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential


def posenet_mouse(input_shape, num_classes):
    """Mouse pose estimation architecture.

    Extended description of function.

    Parameters
    ----------
    arg1 : np.ndarray
        Input shape for mouse pose estimation network.
    arg2 : int
        Number of classes/landmarks.

    Returns
    -------
    keras.model
        model
    """
    recognition_model = Xception(
        include_top=False, input_shape=input_shape, pooling="avg", weights="imagenet",
    )

    new_input = Input(
        batch_shape=(None, input_shape[0], input_shape[1], input_shape[2])
    )

    #### mouse w resnet

    gaussian_noise = 0.01

    x = Conv2D(3, kernel_size=(1, 1), strides=(1, 1))(new_input)

    x = recognition_model(x)
    # x = Flatten()(x)
    x = Dense(1024)(x)
    x = Reshape((2, 2, 256))(x)
    u_7 = UpSampling2D(size=(64, 64))(x)

    x = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding="valid")(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    ###
    # x = Dropout(0.2)(x)
    u_6 = UpSampling2D(size=(32, 32))(x)
    x = GaussianNoise(gaussian_noise)(x)
    ###

    x = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding="valid")(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # x = Dropout(0.2)(x)

    ###
    u_5 = UpSampling2D(size=(16, 16))(x)
    x = GaussianNoise(gaussian_noise)(x)
    ###

    x = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding="valid")(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # x = Dropout(0.2)(x)

    ###
    u_4 = UpSampling2D(size=(8, 8))(x)
    x = GaussianNoise(gaussian_noise)(x)
    ###

    x = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding="valid")(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # x = Dropout(0.2)(x)

    ###
    u_3 = UpSampling2D(size=(4, 4))(x)
    x = GaussianNoise(gaussian_noise)(x)
    ###

    x = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding="valid")(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # x = Dropout(0.2)(x)

    ###
    u_2 = UpSampling2D(size=(2, 2))(x)
    x = GaussianNoise(gaussian_noise)(x)
    ###

    x = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), padding="valid")(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = concatenate([x, u_2, u_3, u_4, u_5, u_6, u_7], axis=-1)

    x = Conv2DTranspose(256, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(128, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(
        num_classes, kernel_size=(1, 1), strides=(1, 1), padding="valid"
    )(x)

    # x = Conv3DTranspose(3, kernel_size=(1,1,1), strides=(1,1,1))(x)

    x = Activation("sigmoid")(x)

    model = Model(inputs=new_input, outputs=x)
    model.summary()

    return model


def posenet_primate(input_shape, num_classes):  # recognition_model = DenseNet201(
    """Primate pose estimation architecture.
    Args:
        input_shape:Input shape for the network.
        num_classes:Number of classes for recognition or number of landmarks.
    """
    recognition_model = ResNet101(
        include_top=False, input_shape=input_shape, pooling="avg", weights="imagenet",
    )

    new_input = Input(
        batch_shape=(None, input_shape[0], input_shape[1], input_shape[2])
    )

    #### primate

    gaussian_noise = 0.3

    mask_size = 128

    x = Conv2D(3, kernel_size=(1, 1), strides=(1, 1))(new_input)

    x = recognition_model(x)
    x = Dropout(0.1)(x)
    # x = Reshape((2, 2, 512))(x)
    x = Dense(int(mask_size * 2 * 2 * 2))(x)
    x = Reshape((2, 2, int(mask_size * 2)))(x)
    u_7 = UpSampling2D(size=(mask_size, mask_size))(x)

    kernel_size = (2, 2)
    strides = (2, 2)

    filters = int(mask_size / 2)

    x = Conv2DTranspose(
        filters, kernel_size=kernel_size, strides=strides, padding="valid"
    )(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    ###
    # x = Dropout(0.2)(x)
    u_6 = UpSampling2D(size=(int(mask_size / 2), int(mask_size / 2)))(x)
    x = GaussianNoise(gaussian_noise)(x)
    ###

    x = Conv2DTranspose(
        filters, kernel_size=kernel_size, strides=strides, padding="valid"
    )(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # x = Dropout(0.2)(x)

    ###
    u_5 = UpSampling2D(size=(int(mask_size / 4), int(mask_size / 4)))(x)
    x = GaussianNoise(gaussian_noise)(x)
    ###

    x = Conv2DTranspose(
        filters, kernel_size=kernel_size, strides=strides, padding="valid"
    )(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # x = Dropout(0.2)(x)

    ###
    u_4 = UpSampling2D(size=(int(mask_size / 8), int(mask_size / 8)))(x)
    x = GaussianNoise(gaussian_noise)(x)
    ###

    x = Conv2DTranspose(
        filters, kernel_size=kernel_size, strides=strides, padding="valid"
    )(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # x = Dropout(0.2)(x)

    ###
    u_3 = UpSampling2D(size=(int(mask_size / 16), int(mask_size / 16)))(x)
    x = GaussianNoise(gaussian_noise)(x)
    ###

    x = Conv2DTranspose(
        filters, kernel_size=kernel_size, strides=strides, padding="valid"
    )(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # x = Dropout(0.2)(x)

    ###
    u_2 = UpSampling2D(size=(int(mask_size / 32), int(mask_size / 32)))(x)
    x = GaussianNoise(gaussian_noise)(x)
    ###

    x = Conv2DTranspose(
        filters, kernel_size=kernel_size, strides=strides, padding="valid"
    )(x)

    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # for primates
    # x = Dropout(0.2)(x)

    u_1 = UpSampling2D(size=(int(mask_size / 64), int(mask_size / 64)))(x)
    # x = GaussianNoise(0.5)(x)

    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    u_0 = UpSampling2D(size=(int(mask_size / 128), int(mask_size / 128)))(x)
    # x = GaussianNoise(0.5)(x)

    # x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding="valid")(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)

    # x = concatenate([x, u_0, u_1, u_2, u_3, u_4, u_5, u_6, u_7], axis=-1)

    # x = Conv2DTranspose(256, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization()(x)

    # x = Conv2DTranspose(128, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization()(x)
    #
    # x = Conv2DTranspose(64, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization()(x)
    #
    # x = Conv2DTranspose(32, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization()(x)

    x = Conv2DTranspose(
        num_classes, kernel_size=(1, 1), strides=(1, 1), padding="valid"
    )(x)

    # x = Conv3DTranspose(3, kernel_size=(1,1,1), strides=(1,1,1))(x)

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
    Args:
        input_shape:
        num_classes:Number of behaviors to classify.
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
    """
    Args:
        input_shape:
        num_classes:
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
    recognition_model, recurrent_input_shape, classes=4,
):
    """BehaviorNet architecture for behavioral classification based on temporal convolution architecture (TCN).

    Parameters
    ----------
    recognition_model : keras.model
        Pretrained recognition model that extracts features for individual frames.
    recurrent_input_shape : np.ndarray
        Number of classes/landmarks.
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
    """
    Args:
        recognition_model:
        recurrent_input_shape:
        classes:
        recurrent_dropout:
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


def pretrained_recognition(model_name, input_shape, num_classes, fix_layers=True):
    # TODO: adaptiv size
    """
    Args:
        model_name:
        input_shape:
        num_classes:
        fix_layers:
    """
    if model_name == "xception":
        recognition_model = Xception(
            include_top=False,
            # input_shape=(75, 75, 3),
            input_shape=(input_shape[0], input_shape[1], 3),
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
    """
    Args:
        input_shape:
        classes:
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

    model.add(MaxPooling2D(strides=(2, 2),))

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

    model.add(MaxPooling2D(strides=(2, 2),))

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
