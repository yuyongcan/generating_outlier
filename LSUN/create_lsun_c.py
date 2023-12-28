import numpy as np
import torch
import torchvision
from torchvision.datasets import SVHN
# from corruptions_32 import *
from corruptions_224 import *
import os
import scipy.io as sio
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import random
from typing import Callable, Optional
import shutil

# corruption_dict = {'gaussian_noise': gaussian_noise, 'shot_noise': shot_noise, 'impulse_noise': impulse_noise,
#                    'defocus_blur': defocus_blur, 'glass_blur': glass_blur,
#                    'zoom_blur': zoom_blur, 'frost': frost, 'fog': fog, 'brightness': brightness,
#                    'contrast': contrast, 'elastic_transform': elastic_transform, 'pixelate': pixelate,
#                    'jpeg_compression': jpeg_compression, 'speckle_noise': speckle_noise, 'gaussian_blur': gaussian_blur,
#                    'spatter': spatter, 'saturate': saturate,'snow': snow,
#                    'motion_blur': motion_blur}
corruption_dict = {'snow': snow,
                   'motion_blur': motion_blur}

random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LSUN_C(ImageFolder):
    base_folder = "LSUN_resize"

    def __init__(
            self, root: str, transform: Optional[Callable] = None, domain=None, output_dir=None, level=5, **kwargs
    ) -> None:
        self.root = os.path.expanduser(root)
        self.dataset_folder = os.path.join(self.root, self.base_folder)
        self.domain = domain
        self.level = level
        self.output_dir = os.path.join(output_dir, self.domain, str(self.level), self.base_folder)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        super().__init__(self.dataset_folder, transform=transform, **kwargs)
        self.method = corruption_dict[self.domain]
        self.method_name = self.method.__name__

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        img_corrupted = transforms.Resize((224, 224))(sample)
        img_corrupted = self.method(img_corrupted, self.level)
        img_corrupted = PILImage.fromarray(np.uint8(img_corrupted))  #
        img_corrupted = transforms.Resize((32, 32))(img_corrupted)
        self.output_path = os.path.join(self.output_dir, os.path.basename(path))
        img_corrupted.save(self.output_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


if __name__ == '__main__':
    root = '/data2/yongcan.yu/datasets'
    for level in range(5, 0, -1):
        for domain in corruption_dict.keys():
            dataset = LSUN_C(root, domain=domain, level=level, output_dir='/data2/yongcan.yu/datasets/LSUN_resize-C',
                             transform=transforms.ToTensor())
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=64)
            for i, (img, target) in enumerate(data_loader):
                if i % 5 == 0:
                    print('domain: {}, level: {}, batch: {}/{}'.format(domain, level, i,
                                                                       len(data_loader)))
            # data_dir = os.path.join(root, 'LSUN_resize-C', domain)
            # os.rename(os.path.join(data_dir, str(level)), os.path.join(data_dir, 'resize'))
            # os.makedirs(os.path.join(data_dir, str(level)))
            # shutil.move(os.path.join(data_dir, 'resize'), os.path.join(data_dir, str(level)))
