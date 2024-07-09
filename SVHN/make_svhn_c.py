import numpy as np
import torch
import torchvision
from torchvision.datasets import SVHN
# from corruptions_32 import *
from corruption32 import *
import os
import scipy.io as sio
import torchvision.transforms as transforms
import random

corruption_dict = {'gaussian_noise': gaussian_noise, 'shot_noise': shot_noise, 'impulse_noise': impulse_noise,
                   'defocus_blur': defocus_blur, 'glass_blur': glass_blur,
                   'zoom_blur': zoom_blur, 'frost': frost, 'fog': fog, 'brightness': brightness,
                   'contrast': contrast, 'elastic_transform': elastic_transform, 'pixelate': pixelate,
                   'jpeg_compression': jpeg_compression, 'speckle_noise': speckle_noise, 'gaussian_blur': gaussian_blur,
                   'spatter': spatter, 'saturate': saturate,'snow': snow,
                   'motion_blur': motion_blur}

random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SVHN_C(SVHN):
    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(self, root, output_root, split='train', transform=None, target_transform=None, download=False,
                 domain='gaussian_noise', level=5, ckpt=None):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.domain = domain
        self.level = level
        self.ckpt = ckpt
        self.output_root = output_root
        self.output_dir = os.path.join(output_root, self.domain, str(self.level))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.method = corruption_dict[self.domain]
        self.method_name = self.method.__name__
        self.loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))
        self.data_cp = np.zeros_like(self.data)

    def __getitem__(self, index):
        if not os.path.exists('./temp_static'):
            os.makedirs('./temp_static')
        img_i, target_i = self.data[index], self.labels[index]
        if self.transform is not None:
            img_i = self.transform(img_i)
        img_corrupted = np.transpose(img_i, (1, 2, 0))
        img_corrupted = PILImage.fromarray(img_corrupted)  #
        img_corrupted = self.method(img_corrupted, self.level)
        img_corrupted = PILImage.fromarray(np.uint8(img_corrupted))  #
        img_corrupted = np.array(img_corrupted)  #
        img_corrupted = np.transpose(img_corrupted, (2, 0, 1))
        img_corrupted.dump('./temp_static/{}.npy'.format(index))
        return img_i, target_i  # we do not care about returning the data

    def save_corruption_imgs(self):
        for i in range(len(self)):
            self.data_cp[i] = np.load('./temp_static/{}.npy'.format(i), encoding='bytes', allow_pickle=True)
        self.loaded_mat['X'] = np.transpose(self.data_cp, (2, 3, 1, 0))
        sio.savemat(os.path.join(self.output_dir, self.filename), self.loaded_mat)
        # os.removedirs('./temp_static')


if __name__ == '__main__':
    for split in ['test', 'train', 'extra']:
        for level in range(5, 0, -1):
            for domain in corruption_dict.keys():
                # print(domain)
                dataset = SVHN_C(root='/data2/yongcan.yu/datasets/SVHN',
                                 output_root='/data2/yongcan.yu/datasets/SVHN-C',
                                 split=split, domain=domain, level=level)
                data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=64)
                for i, (img, target) in enumerate(data_loader):
                    if i % 5 == 0:
                        print('split: {}, domain: {}, level: {}, batch: {}/{}'.format(split, domain, level, i,
                                                                                      len(data_loader)))
                dataset.save_corruption_imgs()
                print('split: {}, domain: {}, level: {} done'.format(split, domain, level))
