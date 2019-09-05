import os
import json

import numpy as np
import cv2
import torch
from torchvision import transforms

from utils.ucf.transform_ucf50 import Rescale, CenterCrop, ToTest


location = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(location)
print("Device being used:", device)


if __name__ == "__main__":
    dataset_dir = '../dataset/UCF50'
    dataset_file_path = './dataset/dataset_ucf50.txt'
    dataset_categories_path = './dataset/categories.json'

    with open(dataset_categories_path, 'r') as f:
        categories = json.load(f)

    with open(dataset_file_path) as f:
        lines = f.readlines()

    new_lines = []

    for l in lines:
        path, cat = l.replace("\n", "").split(" ")
        label = categories[cat]
        new_lines.append(path + " " + str(label) + "\n")

    with open(dataset_file_path, "w") as f:
        f.writelines(new_lines)
    # print(os.listdir(dataset_dir))

    