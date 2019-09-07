import os
import time
import json
import argparse
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
from utils.ucf.dataset import Dataset_UCF50
from utils.ucf.transform import Rescale, RandomCrop, RandomHorizontalFlip, ToTensor
import config


def main(checkpoint_path=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    conf = config.training

    # Load categories 
    with open(conf['category_filepath']) as f:
        category = json.load(f)

    print('{} categories'.format(len(category)))

    # Initialize training model
    print("Initializing training model")
    model = C3D(len(category))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=conf['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    start_epoch = 0

    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device) 
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            start_epoch = checkpoint['epoch']

            print("Checkpoint loaded, starting at {} epoch".format(start_epoch))
        except Exception as e:
            print(str(e))
            print("Could not load checkpoint, starting at the begining")

    model.to(device)
    criterion.to(device)

    print("Start training")
    for epoch in range(start_epoch, conf['max_epoches']):
        dataset_transform = transforms.Compose([
                Rescale((128, 171)),
                RandomCrop((112, 112)),
                RandomHorizontalFlip(0.5),
                ToTensor(),
        ])

        train_loader = DataLoader(
                                Dataset_UCF50(conf['dataset_filepath'], 
                                            conf['category_filepath'], 
                                            transform=dataset_transform,
                                            ), 
                                batch_size=conf['batch_size'], 
                                shuffle=True,
                                num_workers=32,
                                )

        model.train()

        total_loss = 0
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

            print('Epoch: %d, Iter [%d-%d] loss: %.3f, time: %.2f' % 
                (epoch, i, total_batches, loss, time_taken))

        
        print("Epoch {}, avg loss {}".format(epoch, total_loss / total_batches))

        if (epoch+1) % conf['epoch_checkpoint'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict()
            }, os.path.join(conf['checkpoint_path'], 
                            'models_{}_epoch-{}.pth.tar'.format('ucf101', str(epoch+1))))
        
        if (epoch+1) % conf['epoch_save'] == 0:
            torch.save(model, os.path.join(conf['model_path'], 'models_{}_epoch-{}.pth').format('ucf101', str(epoch+1)))


        scheduler.step(epoch)


    torch.save(model, os.path.join(conf['model_path'], 'models_{}_final.pth').format('ucf101'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train action recognition classifier')
    parser.add_argument('--checkpoint', help='Location of checkpoint file')

    args = parser.parse_args()
    
    main(args.checkpoint)
