import glob

import numpy as np
from skimage.color import rgb2lab
from torch.utils.data import Dataset
import torch
import PIL.Image

data_path = "./data"
train_path = f"{data_path}/train/images"
test_path = f"{data_path}/test_color/images"


class ColorizationDataset(Dataset):
    def __init__(self, transform=None, use_test=False):
        self.transform = transform
        image_path = test_path if use_test else train_path
        self.data = sorted(glob.glob(f"{image_path}/*"))
        self.use_test = use_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.data[idx])

        if self.use_test:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
            image = np.array(image)
            image = rgb2lab(image, channel_axis=0).astype("float32")
            l = image[0] / 50.0 - 1.
            return l

        input_image = image.convert("L").convert("RGB")
        output = image

        if self.transform:
            input_image = self.transform(input_image)
            output = self.transform(output)

        input_image = np.array(input_image)
        output = np.array(output)
        axis = 0
        input_image = rgb2lab(input_image, channel_axis=axis).astype("float32")
        output = rgb2lab(output, channel_axis=axis).astype("float32")

        l = input_image[0] / 50.0 - 1.
        ab = output[1:] / 110.0

        return l, ab

