import os
import random
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset_UCF50(Dataset):
    """Dataset for the UCF-50 action recognition dataset

    Preparing data for training C3D model. 
    Read video from dataset file, then extract clips from the video.
    The extracted will be augmented before feeding into the model.

    Attribute:
        categories (list): An array of categories in UCF dataset. 
            The index of each element of the array is corresponding to 
            the label returned by the model.        
    """

    def __init__(self, dataset_file, categories_file,transform=None):
        """Init the Dataset class.

        Args:
            dataset_file (string): Location of the file generated using prepare_dataset.py.
                Path to the text file that specifies the location of video as 
                well as the additional information of each video. 
            
            category_file (string): Location of the file genereated using prepare_dataset.py.
                Path to the text file that contains the contains the categories of the 
                dataset.

            transform (torch transform object): Additional image/clip transformation that 
                will be passed to each clip during the training phrase.  
        """   
        self.transform = transform

        with open(categories_file) as f:
            self.categories = json.load(f)

        self.files, self.labels = self.__get_data(dataset_file=dataset_file)

    def __get_data(self, dataset_file):
        """Fetch clips and their corresponding label from the dataset

        Args:
            dataset_file: See __init__
        
        Returns:
            clips, labels: List of video clips and their corresponding labels, 
                which will be used for training
        """
        clips = []
        labels = []

        with open(dataset_file) as f:
            lines = f.readlines()
            for l in lines:
                clip, label = l.replace("\n", "").rsplit(" ", 1)
                clips.append(clip)
                labels.append(int(label))

        return clips, labels

    def __extract_clip(self, clip_info):
        """Read and extract clip from the clip information string

        Args:
            clip_info: The string specfiy the location of the video 
                that contain the clip (note that a video has many clips), 
                the index of the clip in the video, and how many frame in 
                each clip.
                The format of clip_info is `{file_path} {clip_id} {frame_per_clip}` 

        Return:
            The extracted clip
        """

        # Extract relavent information from clip_info
        video_path, clip_id, frame_per_clip = clip_info.split(" ")
        clip_id, frame_per_clip = int(clip_id), int(frame_per_clip)

        # Get clip from video. The first frame of the clip is computed using its index
        cap = cv2.VideoCapture(video_path)
        clip = []

        start_id = clip_id * frame_per_clip
        end_id = start_id + frame_per_clip

        i = 0
        while(cap.isOpened()):
            ret, frame = cap.read()

            if not ret:
                break 

            if i >= start_id and i < end_id:
                clip.append(frame)
            elif i > end_id:
                break

            i += 1

        return np.array(clip)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f, label = self.files[idx], self.labels[idx]
        clip = self.__extract_clip(f)

        if self.transform:
            clip = self.transform(clip)

        return {'clip': clip, 'label': label}


if __name__ == "__main__":
    from transform_ucf50 import Rescale, RandomCrop, RandomHorizontalFlip, ToTensor
    from torchvision import transforms

    dataset_file='../../dataset/dataset.txt'
    category_file='../../dataset/category.txt'

    dataset = Dataset_UCF50(dataset_file, category_file, transform=transforms.Compose([
        Rescale((128, 171)),
        RandomCrop((112, 112)),
        RandomHorizontalFlip(0.5),
        ToTensor(),
    ]))
    
    for i, data in enumerate(dataset):
        if i == 1:
            break

        print(data['clip'].shape, data['label'], dataset.categories[data['label']])
        clip = data['clip'].cpu().detach().numpy().astype(np.uint8)
        clip = np.transpose(clip, (1,2,3,0))

        for i in range(clip.shape[0]):
            cv2.imshow('test', clip[i,:,:,:])
            if cv2.waitKey(0) & 0xFF == ord('q'):
                b = True
                break
