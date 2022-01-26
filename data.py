#print("sdfds")
import sklearn as sl
import torch
from PIL import Image
import torch.nn
from pathlib import Path
#dataloader and transformation class

#plan
#create a class that would take in lists with masks and 
#return the the

#print('sfdsf')
#test what I wrote
class LungDataset(torch.utils.data.Dataset):
    def __init__(self, origin_mask_list = None, origins_folder = None, masks_folder = None, transforms = None):
        self.origin_mask_list = origin_mask_list
        self.origins_folder = origins_folder
        self.masks_folder = masks_folder
        self.transforms = transforms
        #print("works")

    def __getitem__(self, idx):
        #assume that we already have train, validation and test
        origin_name, mask_name = self.origin_mask_list[idx]
        path_to_origin = self.origins_folder / (origin_name + ".png")
        path_to_mask = self.masks_folder / (mask_name + ".png")
        origin = torch.Tensor(Image.open(path_to_origin).convert("P"))
        mask = torch.Tensor(Image.open(path_to_mask).convert("P"))


        return origin, mask
    #print("works")

#writing a function to split the dataset 

path_to_dataset = Path("Dataset", "dataset")
path_to_images = path_to_dataset / "images"
path_to_masks = path_to_dataset / "masks"
data = LungDataset()
#print("works")

