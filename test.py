import numpy as np
import cv2
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  #

    return np.clip(channels, 0, 1) * 255


def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


if __name__ == '__main__':
    x = np.random.rand(3, 32, 32) * 255
    x = x.astype(np.uint8)
    x= x.transpose((1,2,0))
    x_corrupted_d = defocus_blur(x, severity=5)
    x_corrupted_g = gaussian_noise(x, severity=5)
    x= x.transpose((2,0,1))
    print(x_corrupted_d.shape)
    print(x_corrupted_g.shape)
    print(x_corrupted_g == x_corrupted_d)
    # cv2.imshow('x_corrupted', x_corrupted)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
