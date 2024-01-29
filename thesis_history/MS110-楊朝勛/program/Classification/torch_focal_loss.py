import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True, weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.weights = weights

    def forward(self, input, target):
        pt = F.softmax(input, dim=1)
        logpt = F.log_softmax(input, dim=1)
        weights = self.weights.index_select(0, target) # nice
        target = target.view(-1, 1) # 1維 -> 2維

        pt = pt.gather(1, target)
        logpt = logpt.gather(1, target)
        pt = pt.view(-1)
        logpt = logpt.view(-1)

        loss = -1 * weights * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()