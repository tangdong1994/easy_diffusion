import numpy as np
import torch
import cv2
import random
import lmdb, pickle
from cn_clip import clip as CLIP
import torch
from tqdm import tqdm

class Mess_Image:
    def __init__(self, image, label=None):
        # Dimensions of image for reconstruction - not really necessary 
        # for this dataset, but some datasets may include images of 
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)

def read_lmdb(dirname, filename):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        path   the path of file

        Returns:
        ----------
        images      images array, (N, H, W, C) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    env = lmdb.open(str(dirname / filename), readonly=True)
    num_images = int(filename.split("_")[1])

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember that it's a CIFAR_Image object 
            # that is stored as the value
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels
    
    
class PokenDataset:
    def __init__(self, dirname, filename,split="train", transform=None):
        images, labels = read_lmdb(dirname, filename)
        self.transform = transform
        if split.lower() == "train":
            split = 0
        elif split.lower() == "valid":
            split = 1
        elif split.lower() == "test":
            split = 2
        else:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="valid" or split="test"')
        splits = np.array([i > 1*len(images) for i in range(len(images))], dtype=int)

        mask = (splits == split)
        self.__images = np.array(images)[mask]
        self.__labels = np.array(labels)[mask]
        print("image:",len(self.__images))
        print(len(self.__labels))
            
    def __getitem__(self, index):
        # index = self.indices[index]  # linear, shuffled, or image_weights
        image = self.__images[index]
        label = self.__labels[index]
        # print("load img time ", end1-start)
        if self.transform:
            # flip up-down
            if random.random() < 0.5:
                image = np.flipud(image)
            # flip left-right
            if random.random() < 0.5:
                image = np.fliplr(image)
        
        # Convert
        image = image.transpose(2, 0, 1)  # to 3x416x416
        image = np.ascontiguousarray(image)
        image = image.astype('float32')
        # print("zengqiang ",time.time()-end1)
        return torch.from_numpy(image)/255.0, label
    
    def __len__(self):
        return len(self.__images)

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        return torch.stack(img, 0), None#torch.cat(label, 0)