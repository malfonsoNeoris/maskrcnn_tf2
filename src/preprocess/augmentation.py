import albumentations as img_album


def get_training_augmentation(image_size, normalize=None):
    base_transform_list = [
        img_album.IAAFliplr(p=0.5),
        img_album.IAAFlipud(p=0.5),
        img_album.GaussianBlur(blur_limit=5, p=0.5),
        img_album.Rotate(limit=(10, 270)),
        img_album.MultiplicativeNoise(multiplier=(0.5, 1.2)),
        img_album.ChannelShuffle(p=0.5),
        # img_album.Cutout(num_holes=20, max_h_size=5, max_w_size=5, always_apply=False, p=0.2)
    ]
    weather_transform = [img_album.RandomSnow(p=0.3),
                         img_album.RandomRain(p=0.3),
                         img_album.RandomFog(p=0.2),
                         img_album.RandomSunFlare(p=0.2)
                         ]
    train_transform_hard = [

        img_album.OneOf(base_transform_list, p=0.5),
        img_album.OneOf(weather_transform, p=0.5),

        img_album.OneOf([img_album.HorizontalFlip(p=0.5),
                         img_album.VerticalFlip(p=0.5)]
                        ),

        img_album.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),

        img_album.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=0),
        # img_album.RandomCrop(height=image_size//10, width=image_size//10, p=0.4),

        img_album.IAAAdditiveGaussianNoise(p=0.2),
        img_album.IAAPerspective(p=0.5),

        img_album.OneOf(
            [
                img_album.CLAHE(p=0.5),
                img_album.RandomBrightness(p=0.5),
                img_album.RandomGamma(p=0.5),
            ],
            p=0.9,
        ),

        img_album.OneOf(
            [
                img_album.IAASharpen(p=0.5),
                img_album.Blur(blur_limit=3, p=0.5),
                img_album.MotionBlur(blur_limit=3, p=0.5),
            ],
            p=0.9,
        ),

        img_album.OneOf(
            [
                img_album.RandomContrast(p=0.5),
                img_album.HueSaturationValue(p=0.5),
            ],
            p=0.9,
        ),
        img_album.Lambda(mask=round_clip_0_1)
    ]
    train_transform = [

        img_album.OneOf(base_transform_list, p=0.5),
        img_album.OneOf(weather_transform, p=0.5),

        img_album.OneOf([img_album.HorizontalFlip(p=0.5),
                         img_album.VerticalFlip(p=0.5)]
                        ),

        img_album.Lambda(mask=round_clip_0_1)
    ]
    if normalize:
        # After normalization change array back to unit8 for further augmentation
        train_transform.insert(0, img_album.Lambda(image=denorm_image))
        train_transform.insert(0, img_album.Normalize(mean=normalize['mean'],
                                                      std=normalize['std'],
                                                      max_pixel_value=255.0,
                                                      always_apply=True,
                                                      p=1.0)

                               )

    return img_album.Compose(train_transform)


def get_validation_augmentation(image_size, normalize=None):
    """Add paddings to make image shape divisible by 32"""

    test_transform = [
        img_album.Resize(height=image_size, width=image_size)
    ]
    if normalize:
        test_transform.extend([img_album.Normalize(mean=normalize['mean'],
                                                   std=normalize['std'],
                                                   max_pixel_value=255.0,
                                                   always_apply=True)
                               ])

    return img_album.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data preprocessing function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        img_album.Lambda(image=preprocessing_fn),
    ]
    return img_album.Compose(_transform)


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def denorm_image(x, **kwargs):
    return (x * 255).astype('uint8')
