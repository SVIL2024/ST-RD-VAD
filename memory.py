import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class Memory(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(Memory, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.memMatrix = nn.Parameter(torch.empty(mem_dim, fea_dim))  # M, C
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: query features with size [N, C]
        :return: query output retrieved from memory, with the same size as x.
        """
        att_weight = F.linear(input=x, weight=self.memMatrix)  # [N, M]
        att_weight = F.softmax(att_weight, dim=1)               # [N, M]

        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N, C]

        return dict(out=out, att_weight=att_weight, mem=self.memMatrix)