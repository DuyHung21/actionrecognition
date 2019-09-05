import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class C3D(nn.Module):
    def __init__(self, num_classes):
        super(C3D, self).__init__()

        self.conv1a = nn.Conv3d(in_channels=3, out_channels=64,\
                            kernel_size=(3,3,3), stride=(1,1,1),\
                            padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), )

        self.conv2a = nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), )

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), )

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), )

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.pool1(self.conv1a(x))
        x = self.pool2(self.conv2a(x))
        x = self.pool3(self.conv3b(self.conv3a(x)))
        x = self.pool4(self.conv4b(self.conv4a(x)))
        x = self.pool5(self.conv5b(self.conv5a(x)))

        x = x.view(-1, 8192)

        fc6_features = self.fc6(x)
        fc7_features = self.fc7(fc6_features)

        logits = self.fc8(fc7_features)

        # Refer https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273/7 for more details
        # softmax = F.log_softmax(logits)
        
        return logits
