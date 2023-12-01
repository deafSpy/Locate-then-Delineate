# windowing, transforms
import numpy as np
import albumentations as albu


def window(img, level, width):
    windowed = np.copy(img)
    windowed[windowed < (level-width//2)] = (level-width//2)
    windowed[windowed > (level+width//2)] = (level+width//2)
    windowed = (windowed - (level-width//2)) / width
    windowed = windowed * 255
    return np.uint8(windowed)

def get_windowed_img(img):
    if(img.max() < 3000):
        return window(img, 580, 747)
    elif(img.max() < 10000):
        return window(img, 2254, 3146)
    else:
        return window(img, 7932, 11411)

def normalise(img):
    norm_img = img - img.min()
    norm_img = norm_img / norm_img.max()
    norm_img = norm_img * 255
    return np.uint8(norm_img)

def image_transforms(img_size):
    return albu.Compose([
        albu.OneOf([
            albu.RandomContrast(),
            albu.RandomGamma(),
            albu.RandomBrightness(),
            ], p=0.3),
        albu.OneOf([
            albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            albu.GridDistortion(),
            albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
        albu.ShiftScaleRotate()
    ])
