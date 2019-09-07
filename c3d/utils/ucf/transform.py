import numpy as np

import cv2
import torch
from torchvision import transforms, utils

class Rescale(object):
    """Rescale the clip in a sample to a given size

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, clip): 
        n, h, w = clip.shape[:3]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        # print(n, h, w)
        new_clip = []
        for i in range(n):
            new_clip.append(cv2.resize(clip[i], (new_w, new_h)))
        
        return np.array(new_clip, dtype=np.uint8)

class CenterCrop(object):
    """Center crop a clip to the desired shape

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip):
        n, h, w = clip.shape[:3]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        new_shape = list(clip.shape)
        new_shape[1] = new_h
        new_shape[2] = new_w

        new_clip = np.zeros(new_shape, dtype=np.uint8)

        for i in range(n):
            new_clip[i,:] = clip[i,top:top+new_h, left:left+new_w]
        
        return new_clip

        
class RandomCrop(object):
    """Crop randomly the clip in a sample

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip):
        n, h, w = clip.shape[:3]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_shape = list(clip.shape)
        new_shape[1] = new_h
        new_shape[2] = new_w
        new_clip = np.zeros(new_shape, dtype=np.uint8)

        for i in range(n):
            new_clip[i, :] = clip[i, top: top + new_h, left: left + new_w]

        return new_clip


class RandomHorizontalFlip(object):
    """Horizontal flip randomly the clip with probability p

    Args:
        p: Probability of flipping
    """

    def __init__(self, p):
        assert isinstance(p, (int, float))
        assert (p >= 0) and (p <= 1)
        self.p = p

    def __call__(self, clip):
        rand = np.random.random()
        if rand <= self.p:
            new_clip = np.zeros(clip.shape, dtype=np.uint8)

            for i in range(clip.shape[0]):
                new_clip[i, :] = cv2.flip(clip[i, :], 1)
            return new_clip
        else:
            return clip


class ToTensor(object):
    """Transform a clip with shape (n - h - w - c) to a 
        corresponding tensor with shape (c - n - h - w)
    """

    def __call__(self, clip):
        new_clip = np.transpose(clip, (3, 0, 1, 2))
        return torch.tensor(new_clip, dtype=torch.float)

    
class ToTest(object):
    """Transform a clip with shape (n - h - w - c) to
        a corresponding tensor with shape (1 - c - n - h - w)
    """

    def __call__(self, clip):
        new_clip = np.transpose(clip, (3, 0, 1, 2))
        new_clip = np.expand_dims(new_clip, axis=0)
        return torch.tensor(new_clip, dtype=torch.float)
