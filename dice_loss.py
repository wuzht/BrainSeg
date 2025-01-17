import torch
from torch.autograd import Function, Variable

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.contiguous().view(-1),
                               target.contiguous().view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        dice = (2 * self.inter.float() + eps) / self.union.float()
        return dice

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target, device):
    """
    Dice coeff for batches
    input: B,H,W
    target: B,H,W    
    """
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    # iteration over B (batch_size)
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    # return the sum dicecoeff over B (batch_size)
    return s


class JaccardCoeff(Function):
    """Dice coeff for individual examples"""
    
    def forward(self, input, target):
        eps = 0.0001
        self.inter = torch.dot(input.contiguous().view(-1),\
                              target.contiguous().view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        jaccard = (self.inter.float() + eps) / (self.union.float() - self.inter.float())
        return jaccard