from torch.nn import functional as F
from torch import nn as nn
class Norm(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self,x):
        return F.layer_norm(x,x.size()[1:])