import cv2

import numpy as np
import torch
from torchvision.transforms import Normalize as Normalize_th


class CustomTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, name):
        return str(self) == name

    def __iter__(self):
        def iter_fn():
            for t in [self]:
                yield t
        return iter_fn()

    def __contains__(self, name):
        for t in self.__iter__():
            if isinstance(t, Compose):
                if name in t:
                    return True
            elif name == t:
                return True
        return False


class Compose(CustomTransform):
    """
    All transform in Compose should be able to accept two non None variable, img and boxes
    """
    def __init__(self, *transforms):
        self.transforms = [*transforms]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __iter__(self):
        return iter(self.transforms)

    def modules(self):
        yield self
        for t in self.transforms:
            if isinstance(t, Compose):
                for _t in t.modules():
                    yield _t
            else:
                yield t


class RandomHorizontalFlip(CustomTransform):
    '''
    Horizontally flip image randomly with given probability
    Args:
        p (float): probability of the image being flipped.
                   Defalut value = 0.5
    '''

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):

        flip_indices = [(0, 4), (1, 5)]
        # print(sample['segLabel'])
        # print(sample['segLabel'][::2])
        image, segLabel = sample['img'], sample['segLabel']
        if np.random.random() < self.p:
            image = image[:, ::-1]
            if segLabel is not None:
                for a, b in flip_indices:
                    segLabel[a], segLabel[b] = segLabel[b], segLabel[a]
            # print(segLabel[::2])
            segLabel[::2] = 1280. - segLabel[::2]
        # print(segLabel)
        _sample = sample.copy()
        _sample['img'] = image
        _sample['segLabel'] = segLabel

        return _sample

class Resize(CustomTransform):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size  #(W, H)

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)
        # INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood
        img = cv2.resize(img, (self.size), interpolation=cv2.INTER_CUBIC)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel

        return _sample

    def reset_size(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size



class RandomResize(Resize):
    """
    Resize to (w, h), where w randomly samples from (minW, maxW) and h randomly samples from (minH, maxH)
    """
    def __init__(self, minW, maxW, minH=None, maxH=None, batch=False):
        if minH is None or maxH is None:
            minH, maxH = minW, maxW
        super(RandomResize, self).__init__((minW, minH))
        self.minW = minW
        self.maxW = maxW
        self.minH = minH
        self.maxH = maxH
        self.batch = batch

    def random_set_size(self):
        w = np.random.randint(self.minW, self.maxW+1)
        h = np.random.randint(self.minH, self.maxH+1)
        self.reset_size((w, h))


class Rotation(CustomTransform):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        u = np.random.uniform()
        degree = (u-0.5) * self.theta
        R = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), degree, 1)
        img = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        # if segLabel is not None:
        #     segLabel = cv2.warpAffine(np.array(segLabel), R, (segLabel.shape[1], segLabel.shape[0]), flags=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample

    def reset_theta(self, theta):
        self.theta = theta
#

class Normalize(CustomTransform):
    def __init__(self, mean, std):
        self.transform = Normalize_th(mean, std)

    def __call__(self, sample):
        img = sample.get('img')

        img = self.transform(img)

        _sample = sample.copy()
        _sample['img'] = img
        return _sample


class ToTensor(CustomTransform):
    def __init__(self, dtype=torch.float):
        self.dtype=dtype

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(self.dtype) / 255.
        # img = torch.from_numpy(img).type(self.dtype)

        # if segLabel is not None:
        #     segLabel = torch.from_numpy(torch.tensor(segLabel)).type(torch.long)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample


