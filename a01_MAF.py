import itertools

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torch.nn.parameter import Parameter

from nflib.flows import (
    AffineConstantFlow, ActNorm, AffineHalfFlow,
    SlowMAF, MAF, IAF, Invertible1x1Conv,
    NormalizingFlow, NormalizingFlowModel,
)
from nflib.spline_flows import NSF_AR, NSF_CL

# Lightweight datasets
import pickle
from sklearn import datasets

class DatasetSIGGRAPH:
    """
    haha, found from Eric https://blog.evjang.com/2018/01/nf2.html
    https://github.com/ericjang/normalizing-flows-tutorial/blob/master/siggraph.pkl
    """

    def __init__(self):
        with open('siggraph.pkl', 'rb') as f:
            XY = np.array(pickle.load(f), dtype=np.float32)
            XY -= np.mean(XY, axis=0)  # center
        self.XY = torch.from_numpy(XY)

    def sample(self, n):
        X = self.XY[np.random.randint(self.XY.shape[0], size=n)]
        return X


class DatasetMoons:
    """ two half-moons """

    def sample(self, n):
        moons = datasets.make_moons(n_samples=n, noise=0.05)[0].astype(np.float32)
        return torch.from_numpy(moons)


class DatasetMixture:
    """ 4 mixture of gaussians """

    def sample(self, n):
        assert n % 4 == 0
        r = np.r_[np.random.randn(n // 4, 2) * 0.5 + np.array([0, -2]),
                  np.random.randn(n // 4, 2) * 0.5 + np.array([0, 0]),
                  np.random.randn(n // 4, 2) * 0.5 + np.array([2, 2]),
                  np.random.randn(n // 4, 2) * 0.5 + np.array([-2, 2])]
        return torch.from_numpy(r.astype(np.float32))


# d = DatasetMoons()
d = DatasetMixture()
#d = DatasetSIGGRAPH()

# MAF (with MADE net, so we get very fast density estimation)
flows = [MAF(dim=2, parity=i%2) for i in range(4)]

## --------------- CONSTRUCT MODEL -----------------#

# construct the model
model = NormalizingFlowModel(prior, flows)