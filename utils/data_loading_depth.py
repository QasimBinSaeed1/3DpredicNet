import logging
import numpy as np
import torch
from PIL import Image, ImageEnhance
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import random


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


class BasicDataset(Dataset):
    def __init__(self, dataset_name: str, scale: float = 1.0, mask_suffix: str = '', mode = 'train'):
        self.images_dir = Path(f"./data/{dataset_name}/{mode}/rgb/")
        self.mask_dir = Path(f"./data/{dataset_name}/{mode}/depth/")

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        print(self.images_dir)
        print(self.mask_dir)

        print(f"number of images: {len(list(self.images_dir.glob('*')))}")
        print(f"number of masks: {len(list(self.mask_dir.glob('*')))}")

        self.ids = [splitext(file)[0] for file in listdir(self.images_dir) if isfile(join(self.images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')


    def __len__(self):
        return len(self.ids)
    
    def augment(self, img, depth):
        # img and mask are PIL images

        # resize to 800 x 1200
        img = img.resize((800, 800), resample=Image.BICUBIC)
        depth = depth.resize((800, 800), resample=Image.NEAREST)


        # 1. horizontal flip
        if np.random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. vertical flip
        if np.random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 3. random brightness and contrast imageEnhance
        if np.random.random() < 0.1:
            brightness = np.random.uniform(0.5, 1.5)
            contrast = np.random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        # 4. random cropping and resizing
        if np.random.random() < 0.4:
            w, h = img.size
            newW, newH = int(w * 0.8), int(h * 0.8) # in this line we can change the size of the cropped image
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

            left = np.random.randint(0, w - newW)
            top = np.random.randint(0, h - newH)
            right = left + newW
            bottom = top + newH

            img = img.crop((left, top, right, bottom))
            depth = depth.crop((left, top, right, bottom))

            img = img.resize((w, h), resample=Image.BICUBIC)
            depth = depth.resize((w, h), resample=Image.NEAREST)
        
        # 5. random rotation
        if np.random.random() < 0.4:
            w, h = img.size
            angle = np.random.randint(-5, 5)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
            depth = depth.rotate(angle, resample=Image.NEAREST, expand=True)

            # resize the image to the original size
            img = img.resize((w, h), resample=Image.BICUBIC)
            depth = depth.resize((w, h), resample=Image.NEAREST)

        return img, depth  

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img

    # def __getitem__(self, idx):
    #     name = self.ids[idx]
    #     mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
    #     img_file = list(self.images_dir.glob(name + '.*'))

    #     assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
    #     assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
    #     mask = load_image(mask_file[0])
    #     img = load_image(img_file[0])

    #     assert img.size == mask.size, \
    #         f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        
    #     img, mask = self.augment(img, mask)
        
    #     depth = self.preprocess(mask, self.scale, is_mask=True)
    #     img = self.preprocess(img, self.scale, is_mask=False)

    #     # img and depth should be float32 tensors
    #     img = torch.as_tensor(img.copy(), dtype=torch.float32).contiguous()
    #     depth = torch.as_tensor(depth.copy(), dtype=torch.float32).contiguous()
        

    #     return {
    #         'image': img,
    #         'mask': depth,
    #     }

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0]).convert("RGB")
        img = load_image(img_file[0])

        # Check if the sizes of the image and mask match

        img, mask = self.augment(img, mask)

        depth = self.preprocess(mask, self.scale, is_mask=True)
        img = self.preprocess(img, self.scale, is_mask=False)

        if img.shape != depth.shape:
            # If the sizes don't match, log a warning message and skip this sample
            print(f'Warning: Image and mask {name} have mismatched sizes: {img.shape} and {depth.shape}')
            return None
        # img and depth should be float32 tensors
        img = torch.as_tensor(img.copy(), dtype=torch.float32).contiguous()
        depth = torch.as_tensor(depth.copy(), dtype=torch.float32).contiguous()
            
        return {
            'image': img,
            'mask': depth,
        }



class CarvanaDataset(BasicDataset):
    def __init__(self, dataset_name, mode = 'train', scale=1):
        super().__init__(dataset_name, scale, mask_suffix='', mode = mode)
