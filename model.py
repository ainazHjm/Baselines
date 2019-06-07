import numpy as np
import torch.nn as nn
import torch as th

class Area(nn.Module):
    def __init__(self, gt_path):
        super(Area, self).__init__()
        #from PIL import Image
        #Image.MAX_IMAGE_PIXELS = 100000000000
        #gt = np.array(Image.open(gt_path))
        #self.ratio = np.sum(gt == 1)/(np.sum(gt==0) + np.sum(gt==1))
        self.ratio = 0.003
        # print(self.ratio)
        # self.dist = th.distributions.bernoulli.Bernoulli(th.tensor(self.ratio))

    def sample(self):
        if np.random.rand() < self.ratio:
            return 9.21 # sigmoid of this value is close to 1.
        else:
            return -9.21
            
    def predict(self, x):
        (b, _, h, w) = x.shape
        samples = []
        for i in range(b*h*w):
           samples.append(self.sample())
        # import ipdb; ipdb.set_trace()
        return th.tensor(samples).view(b, 1, h, w)

class NoNghbr(nn.Module):
    def __init__(self, in_channels):
        super(NoNghbr, self).__init__()
        self.net = nn.Conv2d(in_channels, 1, (1,1), stride=1)
    
    def predict(self, x):
        return self.net(x)

class Ngbhr(nn.Module):
    def __init__(self, in_channels):
        super(Ngbhr, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, (3,3), stride=1),
            nn.ConvTranspose2d(8, 1, (3,3), stride=1)
        )
    
    def predict(self, x):
        return self.net(x)

class LinearLayer(nn.Module):
    def __init__(self, in_features):
        super(LinearLayer, self).__init__()
        self.net = nn.Linear(in_features, 1)
    def predict(self, x):
        return self.net(x)