import os
import random
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset_UCF50(Dataset):
    """
        Dataset for the UCF-50 action recognition dataset
    """
    def __init__(self, dataset_file, categories_file, frame_per_clip=16, frame_size=(128, 171), transform=None):
        self.transform = transform
        self.frame_per_clip = frame_per_clip

        with open(categories_file) as f:
            self.categories = json.load(f)

        self.files, self.classes = self.__get_clip_and_classes(dataset_file=dataset_file)

    def __get_clip_and_classes(self, dataset_file, frame_per_clip=None):
        """
            input:
                + frame_per_clip
                + dataset_file: get clip and classes from a pre-defined file

            output:
                + clips: 
                    * list of clips and files that contain that clip
                    * format: '{filename}_{clip_id}_{frame_per_clip}'.
                        * filename: original file
                        * clip_id: the index of clip in the file (for extracting)
                        * frame_per_clip: number of frames in a clip
                + clip_classes: the corresponding class of each clip
        """

        clips = []
        labels = []

        with open(dataset_file) as f:
            lines = f.readlines()
            for l in lines:
                clip, label = l.replace("\n", "").split(" ")
                clips.append(clip)
                labels.append(int(label))

        return clips, labels

    def  __extract_clip(self, clip_path):
        """
            input: 
                + clip_path: {file_path}_{clip_id}_{frame_per_clip}
                + file_path: the path of video file
                + clip_id: id of clip in the video file
                + frame_per_clip: number of frames in a clip
            output:
                + the extracted clip
        """

        # Extract relavent information from clip_path
        fragments = clip_path.split('_')
        video_path, clip_id, frame_per_clip = "_".join(fragments[0:-2]), int(fragments[-2]), int(fragments[-1])

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
        f, class_ = self.files[idx], self.classes[idx]
        clip = self.__extract_clip(f)

        if self.transform:
            clip = self.transform(clip)

        return {'clip': clip, 'label': class_}


if __name__ == "__main__":
    from transform_ucf50 import Rescale, RandomCrop, RandomHorizontalFlip, ToTensor
    from torchvision import transforms

    dataset_file='../../dataset/dataset_ucf50.txt'
    category_file='../../dataset/dataset_ucf50.json'

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
        print(clip.shape)

        for i in range(clip.shape[0]):
            cv2.imshow('test', clip[i,:,:,:])
            if cv2.waitKey(0) & 0xFF == ord('q'):
                b = True
                break
