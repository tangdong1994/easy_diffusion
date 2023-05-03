import numpy as np
import torch
import cv2
import random
import os, time
import albumentations
from PIL import Image
from torch.utils.data import Dataset
from food_detection.data_store.aipos import AiposDataStoreV2
from food_detection.utils.aipos.aipos_client import AiposClient, AiposJobClient

class AiposFoodDataset(Dataset):
    def __init__(self, path, source_id = ["330228","307028"], split="train", random_crop=False, size=512):
        self.size = size
        self.random_crop = random_crop
        aipos_client = AiposClient.for_prod()
        img_files = []
        for sub_id in source_id:
            data_store = AiposDataStoreV2(
                subscriber_id=sub_id,
                class_names=None,
                aipos_client=aipos_client,
                cache_dir=path
                )
            _, classification_dataset, _ = data_store.build_dataset_v3("2,3", None)
            img_files.extend([file.image_file for file in classification_dataset.annotations])
        self.__img_files = img_files
        if split.lower() == "train":
            split = 0
        elif split.lower() == "valid":
            split = 1
        elif split.lower() == "test":
            split = 2
        else:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="valid" or split="test"')
        splits = np.array([i > 0.8*len(self.__img_files) for i in range(len(self.__img_files))], dtype=int)

        mask = (splits == split)
        print(len(self.__img_files))
        self.__img_files = np.array(self.__img_files)[mask]
        self.img_files = self.__img_files
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs
            
    def __getitem__(self, index):
        image = self._process_image(self.img_files[index])
        return image
        
    def _process_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        # (512,512,3) ->(3,512,512)
        image = np.transpose(image, (2,0,1))
        return image
    
    def __len__(self):
        return len(self.__img_files)

