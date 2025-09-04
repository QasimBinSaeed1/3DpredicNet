import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, targets, returnIOU=False):
        # mIoU=0
        # iou=0
        epsilon = 1e-7

        if not (inputs.size() == targets.size()):
            inputs = torch.argmax(inputs, dim=1)

        tp = (targets * inputs).sum().to(torch.float32)
        # tn = ((1 - targets) * (1 - inputs)).sum().to(torch.float32)
        fp = ((1 - targets) * inputs).sum().to(torch.float32)
        fn = (targets * (1 - inputs)).sum().to(torch.float32)
        
        Liou = (1-tp/(tp+fp+fn+epsilon)).to(torch.float32)
        F1Score = (1-2*tp/(2*tp+fp+fn)).to(torch.float32)
        bce = nn.BCELoss()
        loss = 0.5 * Liou + 0.5 * bce(inputs.to(torch.float32), targets.to(torch.float32)) + F1Score

        if returnIOU:
            iou = self.compute_iou(inputs, targets)
            return loss, iou
        else:
            return loss


    def compute_iou(self, y_pred, y_true):
        # ytrue, ypred is a flatten vector
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        current = ConfusionMatrix(num_labels=1, task="binary").to(self.device)
        current = current(y_pred, y_true)
        # compute mean iou
        intersection = torch.diag(current)
        ground_truth_set = current.sum(axis=1)
        predicted_set = current.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        IoU = intersection / union.float()
        return torch.mean(IoU)


if __name__ == "__main__":
    x = torch.rand(16, 3, 400, 400)
    y = torch.rand(16, 3, 400, 400)
    iou = IoULoss()
    print(iou(x, y))