import math

import torch
import torch.nn.functional as F

class baseModel(torch.nn.Module):
    def __init__(self, params):
        super(baseModel, self).__init__()
        for k, v in params.items():
            setattr(self, k, v)

    ## override this method
    def embed(self, coords):
        pass

