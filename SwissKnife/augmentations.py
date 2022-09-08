"""
SIPEC
MARKUS MARKS
AUGMENTATIONS
"""

import imgaug as ia
from imgaug import augmenters as iaa


def primate_poseestimation():
    """TODO: Fill in description"""
    sometimes = lambda aug: iaa.Sometimes(0.4, aug)

    often = lambda aug: iaa.Sometimes(1.0, aug)
    augmentation_image = iaa.Sequential(
        [
            often(
                iaa.Affine(
                    scale=(
                        0.6,
                        1.4,
                    ),  # scale images to 80-120% of their size, individually per axis
                    #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-40, 40),  # rotate by -45 to +45 degrees
                )
            ),
            iaa.Fliplr(0.5, name="Flipper"),
            sometimes(iaa.CoarseDropout(p=0.2, size_percent=0.5, per_channel=False)),
            sometimes(iaa.GaussianBlur(sigma=(0, 1.0))),
            sometimes(iaa.CoarseDropout(p=0.2, size_percent=0.8, per_channel=False)),
            sometimes(iaa.CoarseDropout(p=0.05, size_percent=0.25, per_channel=False)),
        ],
        random_order=True,
    )

    return augmentation_image


def mouse_poseestimation():
    """TODO: Fill in description"""
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    often = lambda aug: iaa.Sometimes(0.95, aug)
    augmentation_image = iaa.Sequential(
        [
            often(
                iaa.Affine(
                    #                 scale={("x": (0.75, 1.25), "y": (0.75, 1.25))}, # scale images to 80-120% of their size, individually per axis
                    scale=(
                        0.9,
                        1.1,
                    ),  # scale images to 80-120% of their size, individually per axis
                    #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-180, 180),  # rotate by -45 to +45 degrees
                    # #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                )
            ),
            # iaa.Fliplr(0.5, name="Flipper"),
            # sometimes(iaa.CoarseDropout(p=0.2, size_percent=0.8, per_channel=False)),
            sometimes(iaa.CoarseDropout(p=0.2, size_percent=0.8, per_channel=False)),
            sometimes(iaa.CoarseDropout(p=0.1, size_percent=0.4, per_channel=False)),
            sometimes(iaa.GaussianBlur(sigma=(0, 1.0))),
        ],
        random_order=True,
    )

    return augmentation_image


def primate_identification(level=2):
    """Augmentation for primate identification.

    This functions returns an imgaug augmentation object according,
    where the strength can be controlled by some integers values called level.
    This augmentaiton object is directly usable with one of SIPEC's networks during training.

    Parameters
    ----------
    level : int
        Level of augmentation, higher value indicates stronger image manipulations.

    Returns
    -------
    imgaug.augmenters
        augmenter
    """
    if level == 0:
        sometimes = lambda aug: iaa.Sometimes(0.33, aug)
        augmentation = iaa.Sequential(
            [
                iaa.Fliplr(0.5, name="Flipper"),
                sometimes(
                    iaa.Affine(
                        #                 scale={("x": (0.75, 1.25), "y": (0.75, 1.25))}, # scale images to 80-120% of their size, individually per axis
                        scale=(
                            0.9,
                            1.1,
                        ),  # scale images to 80-120% of their size, individually per axis
                        #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-5, 5),  # rotate by -45 to +45 degrees
                        #                 shear=(-16, 16), # shear by -16 to +16 degrees
                        order=[
                            0,
                            1,
                        ],  # use nearest neighbour or bilinear interpolation (fast)
                        cval=0,  # if mode is constant, use a cval between 0 and 255
                        # mode=ia.ALL,  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.1, size_percent=0.1, per_channel=False)
                ),
            ],
            random_order=True,
        )
    if level == 1:
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        augmentation = iaa.Sequential(
            [
                iaa.Fliplr(0.5, name="Flipper"),
                sometimes(
                    iaa.Affine(
                        #                 scale={("x": (0.75, 1.25), "y": (0.75, 1.25))}, # scale images to 80-120% of their size, individually per axis
                        scale=(
                            0.85,
                            1.15,
                        ),  # scale images to 80-120% of their size, individually per axis
                        #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-10, 10),  # rotate by -45 to +45 degrees
                        #                 shear=(-16, 16), # shear by -16 to +16 degrees
                        order=[
                            0,
                            1,
                        ],  # use nearest neighbour or bilinear interpolation (fast)
                        cval=0,  # if mode is constant, use a cval between 0 and 255
                        # mode=ia.ALL,  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.1, size_percent=0.1, per_channel=False)
                ),
            ],
            random_order=True,
        )
    if level == 2:
        sometimes = lambda aug: iaa.Sometimes(0.75, aug)
        augmentation = iaa.Sequential(
            [
                iaa.Fliplr(0.5, name="Flipper"),
                sometimes(
                    iaa.Affine(
                        #                 scale={("x": (0.75, 1.25), "y": (0.75, 1.25))}, # scale images to 80-120% of their size, individually per axis
                        scale=(
                            0.8,
                            1.2,
                        ),  # scale images to 80-120% of their size, individually per axis
                        #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-30, 30),  # rotate by -45 to +45 degrees
                        #                 shear=(-16, 16), # shear by -16 to +16 degrees
                        order=[
                            0,
                            1,
                        ],  # use nearest neighbour or bilinear interpolation (fast)
                        cval=0,  # if mode is constant, use a cval between 0 and 255
                        mode=ia.ALL,
                        # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )
                ),
                iaa.GaussianBlur((0, 0.5), name="GaussianBlur"),
                #         iaa.Dropout(0.2, name="Dropout"),
                sometimes(
                    iaa.CoarseDropout(p=0.3, size_percent=0.1, per_channel=False)
                ),
            ],
            random_order=True,
        )
    if level == 3:
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        augmentation = iaa.Sequential(
            [
                iaa.Fliplr(0.5, name="Flipper"),
                sometimes(
                    iaa.Affine(
                        #                 scale={("x": (0.75, 1.25), "y": (0.75, 1.25))}, # scale images to 80-120% of their size, individually per axis
                        scale=(
                            0.7,
                            1.3,
                        ),  # scale images to 80-120% of their size, individually per axis
                        #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                    )
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.2, size_percent=0.05, per_channel=False)
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.25, size_percent=0.2, per_channel=False)
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.25, size_percent=0.8, per_channel=False)
                ),
                sometimes(iaa.GaussianBlur(sigma=(0, 1.0))),
            ],
            random_order=True,
        )

    if level == 9:
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        augmentation = iaa.Sequential(
            [
                iaa.Fliplr(0.5, name="Flipper"),
                sometimes(
                    iaa.Affine(
                        #                 scale={("x": (0.75, 1.25), "y": (0.75, 1.25))}, # scale images to 80-120% of their size, individually per axis
                        scale=(
                            0.7,
                            1.3,
                        ),  # scale images to 80-120% of their size, individually per axis
                        #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-40, 40),  # rotate by -45 to +45 degrees
                        shear=(-8, 8),  # shear by -16 to +16 degrees
                        # order=[
                        #     0,
                        #     1,
                        # ],  # use nearest neighbour or bilinear interpolation (fast)
                        # cval=0,  # if mode is constant, use a cval between 0 and 255
                        # mode=ia.ALL,  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.3, size_percent=0.1, per_channel=False)
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.1, size_percent=0.2, per_channel=False)
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.1, size_percent=0.8, per_channel=False)
                ),
                sometimes(iaa.GaussianBlur(sigma=(0, 0.75))),
            ],
            random_order=True,
        )

    return augmentation


def mouse_identification(level=2):
    """Augmentation for mouse identification.

    This functions returns an imgaug augmentation object according,
    where the strength can be controlled by some integers values called level.
    This augmentaiton object is directly usable with one of SIPEC's networks during training.

    Parameters
    ----------
    level : int
        Level of augmentation, higher value indicates stronger image manipulations.

    Returns
    -------
    imgaug.augmenters
        augmenter
    """
    if level == 2:
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        augmentation = iaa.Sequential(
            [
                iaa.Fliplr(0.5, name="Flipper"),
                sometimes(
                    iaa.Affine(
                        #                 scale={("x": (0.75, 1.25), "y": (0.75, 1.25))}, # scale images to 80-120% of their size, individually per axis
                        scale=(
                            0.9,
                            1.1,
                        ),  # scale images to 80-120% of their size, individually per axis
                        #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-90, 90),  # rotate by -45 to +45 degrees
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                        # order=[
                        #     0,
                        #     1,
                        # ],  # use nearest neighbour or bilinear interpolation (fast)
                        # cval=0,  # if mode is constant, use a cval between 0 and 255
                        # mode=ia.ALL,  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.1, size_percent=0.05, per_channel=False)
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.1, size_percent=0.2, per_channel=False)
                ),
                sometimes(
                    iaa.CoarseDropout(p=0.1, size_percent=0.8, per_channel=False)
                ),
                sometimes(iaa.GaussianBlur(sigma=(0, 1.0))),
            ],
            random_order=True,
        )

    return augmentation
