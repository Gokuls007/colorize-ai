import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
from fastai.data.external import URLs, untar_data

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train', size=256, aug_intensity=0.5):
        self.paths = paths
        self.size = size
        self.aug_intensity = aug_intensity
        
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((size, size), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])
            # Additional intensive augmentations
            self.adv_transforms = transforms.Compose([
                transforms.RandomRotation(15 * aug_intensity),
                transforms.RandomAffine(degrees=0, translate=(0.1*aug_intensity, 0.1*aug_intensity)),
                transforms.ColorJitter(brightness=0.1*aug_intensity, contrast=0.1*aug_intensity)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((size, size), Image.BICUBIC),
            ])
            self.adv_transforms = None

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        if self.adv_transforms:
            img = self.adv_transforms(img)
            
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # [0, 100], [-128, 127], [-128, 127]
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # [0, 100] -> [-1, 1]
        ab = img_lab[[1, 2], ...] / 110. # [-110, 110] -> [-1, 1]
        
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)

class ColorizationDataLoader:
    def __init__(self, num_images=10000, batch_size=16, size=256, aug_intensity=0.5):
        dataset_path = untar_data(URLs.COCO_SAMPLE)
        paths = [os.path.join(dataset_path, 'train_sample', n) for n in os.listdir(os.path.join(dataset_path, 'train_sample'))]
        # Just use subset for training
        np.random.seed(42)
        paths = np.random.choice(paths, min(num_images, len(paths)), replace=False)
        
        # Split
        split_idx = int(len(paths) * 0.8)
        train_paths = paths[:split_idx]
        val_paths = paths[split_idx:]
        
        self.train_ds = ColorizationDataset(train_paths, split='train', size=size, aug_intensity=aug_intensity)
        self.val_ds = ColorizationDataset(val_paths, split='val', size=size)
        self.batch_size = batch_size

    def get_dataloaders(self):
        train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return train_dl, val_dl
