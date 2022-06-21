from glob import glob
from PIL import Image
import random
from torch.utils.data import Dataset

class FFHQ(Dataset):
    def __init__(self, data_root, img_transform, mask_transform, data='train'):
        super(FFHQ, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # get the list of image paths
        if data == 'train':
            self.paths = glob('{}/train/faces/*.*'.format(data_root),
                              recursive=True)
            self.mask_paths = glob('{}/train/mask_new2/*.*'.format(data_root))
        else:
            self.paths = glob('{}/val/faces/*.*'.format(data_root, data))
            self.mask_paths = glob('{}/val/mask_new2/*.*'.format(data_root))

        self.N_mask = len(self.mask_paths)

        print("[{}] # of dataset: {}".format(data, len(self.paths)))
        print("[{}] # of masks: {}".format(data, len(self.mask_paths)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index % len(self.paths)])
        img = self.img_transform(img.convert('RGB'))
        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        
        return img * mask, mask, img