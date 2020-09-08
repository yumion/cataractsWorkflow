import albumentations as albu
import albumentations.augmentations.functional as F
import cv2
from PIL import Image, ImageOps
import numpy as np
import random


class get_augmentation_wrapper:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, height, width):
        self.transforms.append(
            albu.Resize(height, width)
        ),
        return albu.Compose(self.transforms)


class AutoContrast(albu.ImageOnlyTransform):
    """画像のコントラストを最大化（平坦化）する
    入力画像のヒストグラムを計算し、
    cutoff(0 <= cutoff < 50)で指定されたパーセント以上の明るさ/暗さのピクセルを除去し、
    最も暗いピクセルが黒（0）に、最も明るいピクセルが白（255）になるようマッピングし直す
    """

    def __init__(self, cutoff=0, always_apply=False, p=0.5):
        super(AutoContrast, self).__init__(always_apply, p)
        self.cutoff = cutoff

    def apply(self, img, **params):
        img_p = Image.fromarray(img)
        autocon = ImageOps.autocontrast(img_p, self.cutoff)
        return np.asarray(autocon)

    def get_transform_init_args_names(self):
        return {'cutoff': self.cutoff}


class HorizontalShift(albu.DualTransform):
    """x軸方向平行移動
    """

    def __init__(
        self,
        limit=0.0625,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(HorizontalShift, self).__init__(always_apply, p)
        self.limit = albu.to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, dx=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.shift_scale_rotate(img, 0, 1, dx, 0, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, dx=0, **params):
        return F.shift_scale_rotate(img, 0, 1, dx, 0, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def get_params(self):
        return {
            "dx": random.uniform(self.limit[0], self.limit[1]),
        }

    def get_transform_init_args(self):
        return {
            "limit": self.limit,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
            "value": self.value,
            "mask_value": self.mask_value,
        }


class VerticalShift(albu.DualTransform):
    """y軸方向平行移動
    """

    def __init__(
        self,
        limit=0.0625,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(VerticalShift, self).__init__(always_apply, p)
        self.limit = albu.to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.shift_scale_rotate(img, 0, 1, 0, dy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, dy=0, **params):
        return F.shift_scale_rotate(img, 0, 1, 0, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def get_params(self):
        return {
            "dy": random.uniform(self.limit[0], self.limit[1]),
        }

    def get_transform_init_args(self):
        return {
            "limit": self.limit,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
            "value": self.value,
            "mask_value": self.mask_value,
        }


class RandomShear(albu.DualTransform):
    """Randomly resize the input. Output image size is different from the input image size.
    Args:
        shear_x (float, tuple of floats): Shear along x axis. If single float shear_x is picked
            from (-shear_x, shear_x) interval. Default: 0.1.
        shear_y (float, tuple of floats): Shear along y axis. If single float shear_y is picked
            from (-shear_y, shear_y) interval. Default: 0.1.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        shear_x=0.1,
        shear_y=0.1,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.cv2.BORDER_REFLECT_101,
        value=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.shear_x = albu.to_tuple(shear_x)
        self.shear_y = albu.to_tuple(shear_y)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value

    def get_params(self):
        return {
            "shear_x": random.uniform(self.shear_x[0], self.shear_x[1]),
            "shear_y": random.uniform(self.shear_y[0], self.shear_y[1]),
        }

    def apply(self, img, shear_x=0, shear_y=0, **params):
        return shear(
            img, shear_x, shear_y,
            interpolation=self.interpolation,
            border_mode=self.border_mode,
            value=self.value,
        )

    def get_transform_init_args_names(self):
        return ("shear_x", "shear_y", "interpolation", "border_mode", "value")


def shear(
    img,
    shear_x=0,
    shear_y=0,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None
):
    matrix = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)

    warp_affine_fn = F._maybe_process_in_chunks(
        cv2.warpAffine,
        M=matrix,
        dsize=img.shape[:2][::-1],
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value
    )
    return warp_affine_fn(img)


class AugMix:
    def __init__(
        self,
        transforms=None,
        mean=(0, 0, 0),
        std=(1, 1, 1),
        width=3,
        depth=-1,
        alpha=1.,
        p=1.,
    ):
        self.transforms = transforms
        self.mean = mean
        self.std = std
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.p = p

        if self.transforms is None:
            self.transforms = [
                AutoContrast(cutoff=0, p=1),
                albu.Equalize(mode='pil', p=1),
                albu.Posterize(num_bits=(3, 4), p=1),
                albu.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1),  # rotate
                albu.Solarize(threshold=77, p=1),
                RandomShear(shear_x=0.09, shear_y=0, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1),
                RandomShear(shear_x=0, shear_y=0.09, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1),
                VerticalShift(shift_limit=0.09, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1),
                HorizontalShift(shift_limit=0.09, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1),
                # ImageNet-C
                albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-36, 0), val_shift_limit=0, p=1),  # saturation
                albu.RandomContrast(limit=(-0.36, 0), p=1),
                albu.RandomBrightness(limit=(-0.36, 0), p=1),
                albu.OneOf([  # sharpness
                    albu.IAASharpen(alpha=(0.1, 0.5), lightness=0, p=1),
                    albu.Blur(blur_limit=7, p=1),
                ], p=0.5),
            ]

    def __call__(self, image, mask, **kwargs):
        if np.random.uniform() < self.p:
            image, mask = dual_augment_and_mix(
                image, mask,
                self.transforms,
                self.mean, self.std,
                self.width,
                self.depth,
                self.alpha,
            )
        return dict(image=image, mask=mask, **kwargs)


def dual_augment_and_mix(image, mask, transforms, mean=(0, 0, 0), std=(1, 1, 1), width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: Raw input image(0-255) as uint8 `np.ndarray` of shape (h, w, c)
      mask: Onehot encoded mask as `np.ndarray` of shape (h, w, c)
      transforms: albumentations augmentations as `list` with parameterized intensity
      width: Width of augmentation chain
      depth: Depth of augmentation chain. -1 enables stochastic depth uniformly from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:
      image_mixed: Augmented and mixed image.
      mask_mixed: Augmented and mixed mask.
    """

    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    image_mix = np.zeros_like(image)
    mask_mix = np.zeros_like(mask, dtype=np.float)

    for i in range(width):
        image_aug, mask_aug = image.copy(), mask.copy()

        # decide augmentations of depth at random
        depth = np.random.randint(1, depth + 1 if depth > 0 else 4)
        op_depth = [np.random.choice(transforms) for _ in range(depth)]

        # apply augmentation
        augmentations = albu.Compose(op_depth, p=1)
        sample = augmentations(image=image_aug, mask=mask_aug)
        image_aug, mask_aug = sample['image'], sample['mask']

        # Preprocessing commutes since all coefficients are convex
        image_mix = image_mix + ws[i] * normalize(image_aug, mean, std)
        mask_mix = mask_mix + ws[i] * mask_aug

    # add augmentated image with normalization
    image_mixed = (1 - m) * normalize(image, mean, std) + m * image_mix
    mask_mixed = (1 - m) * mask + m * mask_mix
    # denormalization
    image_mixed = np.clip(image_mixed * 255, 0, 255).astype(np.uint8)
    mask_mixed = np.clip(mask_mixed, 0, 1)  # 色変換系で1以上になったピクセルを1にクリップ
    return image_mixed, mask_mixed


def normalize(image, mean, std):
    """Normalize input image channel-wise to zero mean and unit variance."""
    mean, std = np.asarray(mean), np.asarray(std)
    image = image.astype(np.float32) / 255.
    image = image.transpose(2, 0, 1)  # Switch to channel-first
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image.transpose(1, 2, 0)


class RandAugment:
    def __init__(self, N, M, transforms=None, max_magnitude=10):
        self.N = N  # [1, len(transforms)]
        self.M = M  # [0, 10]
        self.transforms = transforms
        self.max_magnitude = max_magnitude

        if self.transforms is None:
            self.transforms = [
                autocontrast,
                equalize,
                rotate,
                solarize,
                color,
                posterize,
                contrast,
                brightness,
                sharpness,
                blur,
                shear_x,
                shear_y,
                translate_x,
                translate_y,
            ]

    def __call__(self, image, **kwargs):
        # image[, mask] are required and operated
        sampled_ops = np.random.choice(self.transforms, self.N)

        for op in sampled_ops:
            sample = op(self.M)(image=image, **kwargs)

        return sample


def M2Level(M, minval, maxval, max_magnitude=10):
    return (float(M) / max_magnitude) * float(maxval - minval) + minval


def autocontrast(m):
    return AutoContrast(
        cutoff=0,
        p=1)


def equalize(m):
    return albu.Equalize(
        mode='pil',
        p=1)


def rotate(m, minval=0, maxval=30):
    """
    m: magnitude [0, 10]
    minval/maxval: apply augmentation's level(or power)
    """
    level = M2Level(m, minval, maxval)
    if np.random.rand() > 0.5:
        level = -level
    return albu.Rotate(
        limit=(level, level),
        interpolation=cv2.INTER_LANCZOS4,
        border_mode=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
        p=1)


def solarize(m, minval=0, maxval=256):
    level = M2Level(m, minval, maxval)
    return albu.Solarize(
        threshold=(level, level),
        p=1)


def color(m, minval=-128, maxval=128):
    level = M2Level(m, minval, maxval)
    return albu.HueSaturationValue(
        hue_shift_limit=0,
        sat_shift_limit=(level, level),
        val_shift_limit=0,
        p=1)


def posterize(m, minval=0, maxval=8):
    level = int(M2Level(m, minval, maxval))
    return albu.Posterize(
        num_bits=(level, level),
        p=1)


def contrast(m, minval=-1.0, maxval=1.0):
    level = M2Level(m, minval, maxval)
    return albu.RandomContrast(
        limit=(level, level),
        p=1)


def brightness(m, minval=-1.0, maxval=1.0):
    level = M2Level(m, minval, maxval)
    return albu.RandomBrightness(
        limit=(level, level),
        p=1)


def sharpness(m, minval=0., maxval=1.0):
    level = M2Level(m, minval, maxval)
    return albu.IAASharpen(
        alpha=(level, level),
        lightness=0,
        p=1)


def blur(m, minval=3, maxval=30):
    level = M2Level(m, minval, maxval)
    return albu.Blur(
        blur_limit=(level, level),
        p=1)


def shear_x(m, minval=0, maxval=0.33):
    level = M2Level(m, minval, maxval)
    if np.random.rand() > 0.5:
        level = -level
    return RandomShear(
        shear_x=(level, level),
        shear_y=0,
        border_mode=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
        p=1)


def shear_y(m, minval=0, maxval=0.33):
    level = M2Level(m, minval, maxval)
    if np.random.rand() > 0.5:
        level = -level
    return RandomShear(
        shear_x=0,
        shear_y=(level, level),
        border_mode=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
        p=1)


def translate_x(m, minval=0, maxval=0.33):
    level = M2Level(m, minval, maxval)
    if np.random.rand() > 0.5:
        level = -level
    return HorizontalShift(
        limit=(level, level),
        border_mode=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
        p=1)


def translate_y(m, minval=0, maxval=0.33):
    level = M2Level(m, minval, maxval)
    if np.random.rand() > 0.5:
        level = -level
    return VerticalShift(
        limit=(level, level),
        border_mode=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
        p=1)
