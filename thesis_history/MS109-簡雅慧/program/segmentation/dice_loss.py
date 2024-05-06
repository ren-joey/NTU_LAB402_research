import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 1e-5
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        dice_ = (2 * self.inter.float() + eps) / self.union.float()
        iou_ = (self.inter.float() + eps) / (self.union.float()-self.inter.float())
        return dice_, iou_

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coe(output, target):
    """Dice coeff for batches"""
    # s = 0
    if output.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
        t = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
        t = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(output, target)):
        sx, tx = DiceCoeff().forward(c[0], c[1])
        s = s + sx
        t = t + tx

    return s / (i + 1),  t/ (i + 1)