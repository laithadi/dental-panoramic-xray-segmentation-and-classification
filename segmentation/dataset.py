from PIL import Image
from torch.utils.data import Dataset
import os
from constants import TRAIN_XRAY_PATH, TRAIN_MASKS_PATH
import matplotlib.pyplot as plt

class TeethSegmentationData(Dataset):
    def __init__(self, data, transforms = None):
      self.transforms = transforms
      self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fname = self.data[index]

        img_fpath = os.path.join(TRAIN_XRAY_PATH, fname)
        img = Image.open(img_fpath).convert('L')

        mask_fpath = os.path.join(TRAIN_MASKS_PATH, fname)
        mask = Image.open(mask_fpath).convert('L')
        
        if self.transforms:
            img = self.transforms(img)
            mask = self.transforms(mask)

        return {'img':img, 'mask': mask, 'fname':fname}
    