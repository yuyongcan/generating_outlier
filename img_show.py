import os.path

from SVHN import SVHN
from PIL import Image

corruption_dict = {'gaussian_noise', 'shot_noise', 'impulse_noise',
                   'defocus_blur', 'glass_blur',
                   'zoom_blur', 'frost', 'fog', 'brightness',
                   'contrast', 'elastic_transform', 'pixelate',
                   'jpeg_compression', 'speckle_noise', 'gaussian_blur',
                   'spatter', 'saturate','snow', 'motion_blur'}

save_dir = './imgs/SVHN'
dataset_origin = SVHN(root='/data2/yongcan.yu/datasets/SVHN', split='test', download=True)
# print(dataset_origin.data[0])
img_origin = Image.fromarray(dataset_origin.data[1].transpose(1, 2, 0))
img_origin.save(os.path.join(save_dir, 'test_origin.png'))
img_pre = img_origin
for domain in corruption_dict:
    dataset = SVHN(root=os.path.join('/data2/yongcan.yu/datasets/SVHN-C/', domain, '1'), split='test')
    img = Image.fromarray(dataset.data[25].transpose(1, 2, 0))
    img.save(os.path.join(save_dir, 'test_{}.png'.format(domain)))
    # print(img == img_pre)
    # print(dataset.data[0])
    # print(img)
    # img_pre = img
