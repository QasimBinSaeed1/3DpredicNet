# this loss performs better
import torch
import torch.nn as nn
import torch.nn.functional as F
from piqa import SSIM
import kornia as K
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class SSIMLoss(SSIM):

    def __init__(self):
        super().__init__()
        self.js_loss = JsdCrossEntropy()

    def forward(self, inputs, targets):
        ssim_loss = 1 - super().forward(inputs, targets)
        g = torch.log(inputs + 1) - torch.log(targets + 1)
        n = inputs.shape[0] * inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
        SI_loss = torch.mean(g ** 2) - (torch.sum(g) ** 2) / n ** 2

        grads: torch.Tensor = K.filters.spatial_gradient(inputs, order=1)  # BxCx2xHxW
        grads_xx = grads[:, :, 0]
        grads_yx = grads[:, :, 1]

        grads: torch.Tensor = K.filters.spatial_gradient(targets, order=1)  # BxCx2xHxW
        grads_xy = grads[:, :, 0]
        grads_yy = grads[:, :, 1]
        GM_loss = torch.mean(torch.abs(grads_xx - grads_xy) + torch.abs(grads_yx - grads_yy))
        mse_loss = nn.MSELoss()


        return SI_loss + 0.5 * GM_loss + 0.9 * ssim_loss + mse_loss(inputs, targets) # + self.js_loss(inputs, targets)


class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss
    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    Hacked together by / Copyright 2020 Ross Wightman
    """
    def __init__(self, num_splits=1, alpha=12):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)

        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        loss += self.alpha * sum([F.kl_div(
            logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
        return 1.52311 - loss



if __name__ == "__main__":
    x = torch.rand(16, 3, 400, 400)
    y = torch.rand(16, 3, 400, 400)
    loss = SSIMLoss()
    print(loss(x, y))
    print(loss(x, x))