import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from models.c3d import C3D
from utils.ucf.dataset_ucf50 import Dataset_UCF50
from utils.ucf.transform_ucf50 import Rescale, RandomCrop, RandomHorizontalFlip, ToTensor

DATESET_FILE_PATH = './dataset/dataset_ucf50.txt'
CATEGORY_FILE_PATH = './dataset/dataset_ucf50.json'
CHECKPOINT_PATH = './pretrained/checkpoints/ucf/'
MODEL_PATH = './pretrained/models/ucf/'
NUM_CLASSES = 50
BATCH_SZIE = 30
LEARNING_RATE = 0.003
NUM_EPOCHES = 15
SAVE_EPOCH = 5
SAVE_MODEL = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

model = C3D(NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

model.to(device)
criterion.to(device)


start_epoch = 0
max_epoch = NUM_EPOCHES

for epoch in range(start_epoch, max_epoch):
    dataset_transform = transforms.Compose([
            Rescale((128, 171)),
            RandomCrop((112, 112)),
            RandomHorizontalFlip(0.5),
            ToTensor(),
    ])

    train_loader = DataLoader(Dataset_UCF50(DATESET_FILE_PATH, CATEGORY_FILE_PATH, transform=dataset_transform), 
                                batch_size=BATCH_SZIE, shuffle=True, num_workers=4)

    model.train()
    
    total_loss = 0
    total_accuracy = 0
    total_batches = len(train_loader)
    for i, data in enumerate(train_loader):
        clip, label = data['clip'], data['label']

        start_time = time.time()
        optimizer.zero_grad()

        if torch.cuda.is_available():
            clip = clip.cuda()
            label = label.cuda()

        y_pred = model(clip)
        time_taken = time.time() - start_time

        loss = criterion(y_pred, label)

        loss.backward()
        optimizer.step()

        total_loss += loss

        print('[%d-%d] loss: %.3f, time: %.2f' % (i, total_batches, loss, time_taken))

    print("Epoch {}, avg loss {}".format(epoch, total_loss / total_batches))

    if epoch % SAVE_EPOCH == 0:
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'opt_dict': optimizer.state_dict()
        }, os.path.join(CHECKPOINT_PATH, 'models_{}_epoch-{}.pth.tar'.format('ucf50', str(epoch))))
    
    if epoch % SAVE_MODEL == 0:
        torch.save(model, os.path.join(MODEL_PATH, 'models_{}_epoch-{}.pth').format('ucf50', str(epoch)))


    scheduler.step(epoch)

torch.save(model, os.path.join(MODEL_PATH, 'models_{}_final.pth').format('ucf50'))