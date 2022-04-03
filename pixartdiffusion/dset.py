import torch
from torch.utils.data import Dataset
from glob import glob
import random

from pixartdiffusion.util import load_im, flip_hor, shift

# Performs a random flip and a random shift on a numpy array image
def transform(im):
    if random.random()<0.5:
        im = flip_hor(im)
    shifts = [(0,0),(0,1),(1,0),(-1,0),(0,-1)]
    return shift(im, *random.choice(shifts))

# Indexing gives transformed image tensors
# Initialisation: takes a path, reads all the images
# TODO: This is probably only appropriate for small datasets, as everything is read into memory
class PixDataset(Dataset):
    def __init__(self, data_path):
        """
        images: N x W x H x NUM_CHANNELS numpy array (or Tensor) of image data
        """

        self.ims = []
        
        print("Reading images... ", end="")
        for path in glob(data_path):
            im = load_im(path)
            self.ims.append(im)
        print("Done")
    
    def __len__(self):
        return len(self.ims)
    
    def __getitem__(self, idx):
        im = transform(self.ims[idx])
        im = torch.Tensor(im)
        return im
