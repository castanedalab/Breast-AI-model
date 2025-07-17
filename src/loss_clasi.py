import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

import utils

def BCELogitsLoss(y_hat, y, weight = None):
    return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=weight)

def BCEDiceLoss(y_hat, y, weight = 0.1, device = 'cuda'):
    bce_loss = F.binary_cross_entropy_with_logits(y_hat, y)
    y_hat = torch.sigmoid(y_hat) 
    
    _, dice_loss = utils.dice_coeff_batch(y_hat, y, device)
    loss = bce_loss * weight + dice_loss * (1 - weight)

    return loss

def DiceLoss(y_hat, y):
    y_hat = torch.sigmoid(y_hat) 
    _, dice_loss = utils.dice_coeff_batch(y_hat, y)
    return dice_loss

def TverskyLoss(y_hat, y, alpha=0.7, beta=0.3):
    y_hat = torch.sigmoid(y_hat)
    _, tversky_loss = utils.tversky_coeff_batch(y_hat, y, alpha, beta)
    return tversky_loss

def FocalTverskyLoss(y_hat, y, alpha=0.7, beta=0.3, gamma=0.75):
    y_hat = torch.sigmoid(y_hat)
    tversky,_ = utils.tversky_coeff_batch(y_hat, y, alpha, beta)
    # focal_tversky_loss=K.pow((1 - tv), gamma)
    return (1 - tversky)**gamma

def FocalLoss(y_hat, y, alpha=1, gamma=2, logits=True, reduce=True):
    if logits:
        BCE_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none')
    else:
        BCE_loss = F.binary_cross_entropy(y_hat, y, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss

    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss

def BCELoss(y_hat, y):
    bce_loss = F.binary_cross_entropy(y_hat, y)
    loss = bce_loss 
    return loss

def CrossEntropyDiceLoss(y_hat, y, weight=0.1):
    ce_loss = F.cross_entropy(y_hat, y)
    y_hat = torch.softmax(y_hat, dim=1)
    #print(y_hat.shape)
    #print(y.shape)
    #y_one_hot = F.one_hot(y, num_classes=y_hat.shape[1]).permute(0, 3, 1, 2).float()
    # Convertir las etiquetas a formato one-hot encoded
    y_one_hot = F.one_hot(y, num_classes=y_hat.shape[1]).float()
    #print(f"Dimensiones de y_one_hot después de one-hot encoding: {y_one_hot.shape}")

    # Permutar si es necesario
    if y_one_hot.dim() == 5:  # Si es [batch_size, depth, height, width, num_classes]
        y_one_hot = y_one_hot.permute(0, 4, 1, 2, 3)  # Cambiar a [batch_size, num_classes, depth, height, width]
    elif y_one_hot.dim() == 4:  # Si es [batch_size, height, width, num_classes]
        y_one_hot = y_one_hot.permute(0, 3, 1, 2)  # Cambiar a [batch_size, num_classes, height, width]

    intersection = (y_hat * y_one_hot).sum(dim=(2, 3))
    dice_loss = 1 - (2. * intersection + 1) / (y_hat.sum(dim=(2, 3)) + y_one_hot.sum(dim=(2, 3)) + 1)
    dice_loss = dice_loss.mean()

    return ce_loss * weight + dice_loss * (1 - weight)

def CrossEntropyLoss(y_hat, y):
    ce_loss = F.cross_entropy(y_hat, y)
    return ce_loss


# def NLLLoss(y_hat, y):
#     # loss = torch.nn.NLLLoss()
#     # m = torch.nn.LogSoftmax(dim=1) # assuming tensor is of size N x C x height x width, where N is the batch size.
#     loss = F.nll_loss(F.log_softmax(y_hat), y)
#     return loss

# class WeightedFocalLoss(torch.nn.Module):
#     "Non weighted version of Focal Loss"
#     "https://amaarora.github.io/2020/06/29/FocalLoss.html"
#     def __init__(self, alpha=.25, gamma=2):
#         super(WeightedFocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()

def project_pooling_3d_tensor(input_tensor, kernel_size):
    """Applies max pooling on the 3D tensor with the specified kernel size."""
    project_pooling = nn.MaxPool3d(kernel_size=kernel_size, stride=1)
    return project_pooling(input_tensor)

def topology_pooling_2d_tensor(input_tensor, kernel, stride):
    """Applies max pooling on the 2D tensor with the specified kernel size and stride."""
    abstract_2d_pooling = nn.MaxPool2d(kernel_size=kernel, stride=stride)
    abstract_pooling = abstract_2d_pooling(input_tensor)
    return abstract_pooling

def topological_pooling(input_tensor, kernel, stride, dim):
    """Performs topological pooling on the input tensor."""
    if input_tensor.dim() == 5:  # 3D volumes
        projection_kernels = [(1, 1, input_tensor.size(4)), (input_tensor.size(2), 1, 1), (1, input_tensor.size(3), 1)]
        input_project_pooling_3d_tensor = project_pooling_3d_tensor(input_tensor, kernel_size=projection_kernels[dim])
        if dim == 0: squeeze_dim = 4
        else: squeeze_dim = 1
        input_project_pooling_3d_tensor = input_project_pooling_3d_tensor.squeeze(dim + squeeze_dim)
    elif input_tensor.dim() == 4:  # 2D images
        input_project_pooling_3d_tensor = input_tensor
    else:
        raise ValueError("'input_tensor' must be 4D or 5D tensors")
    input_2d_pooling = topology_pooling_2d_tensor(input_project_pooling_3d_tensor, kernel=kernel, stride=stride)
    return input_2d_pooling


def compute_per_channel_topology_component(input, target, start_channel, kernel_list, stride_list):
    """Computes the per-channel topology component of the input and target tensors."""
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    num_channels = input.size(1)
    num_dims = input.dim() - 2  # Calculate the number of dimensions: 3 for 3D, 2 for 2D
    difference_ks_list = []
    for kernel, stride in zip(kernel_list, stride_list):
        pooling_diff = []
        for dim in range(num_dims):  # Change the loop range to accommodate 2D and 3D tensors
            pred_pooling = topological_pooling(input, kernel=kernel, stride=stride, dim=dim)
            label_pooling = topological_pooling(target, kernel=kernel, stride=stride, dim=dim)
            channel_pooling_diff = []
            for channel in range(start_channel, num_channels):  # start from 1 to ignore the background channel.
                sum_pred_pooling = torch.sum(pred_pooling, dim=(-2, -1))[:, channel, ...]
                sum_label_pooling = torch.sum(label_pooling, dim=(-2, -1))[:, channel, ...]
                difference = torch.abs(sum_pred_pooling - sum_label_pooling)
                channel_pooling_diff.append(difference)
            pooling_diff.append(torch.mean(torch.stack(channel_pooling_diff)))
        difference_ks_list.append(torch.mean(torch.stack(pooling_diff)))
    return torch.mean(torch.stack(difference_ks_list))

class TopologicalPoolingLoss(nn.Module):
    def __init__(self, start_channel=1, kernel_list=None, stride_list=None):
        """Initializes the TopologicalPoolingLoss class."""
        super().__init__()
        self.start_channel = start_channel
        self.kernel_list = kernel_list or [4, 5, 8, 10, 20]
        self.stride_list = stride_list or self.kernel_list

    def forward(self, input, target):
        """Computes the topological pooling loss for the input and target tensors."""
        if input.dim() != target.dim():
            raise ValueError("'input' and 'target' have different number of dimensions")
        if input.dim() not in (4, 5):
            raise ValueError("'input' and 'target' must be 4D or 5D tensors")
        per_channel_topology_component = compute_per_channel_topology_component(input, target, self.start_channel, self.kernel_list, self.stride_list)
        return per_channel_topology_component