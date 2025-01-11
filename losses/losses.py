import torch
import torch.nn as nn
import torch.nn.functional as F

# BCE/Dice, Dice , GDL, abstract dice, per channel dice, masking loss wrapper from pytorch3dunet

def label_smooth(target, smooth_factor: float):
    """
    Applies label smoothing for binary targets:
        1 -> 1 - smooth_factor
        0 -> smooth_factor
    """
    # target is expected to be either 0 or 1
    # clamp to protect from any rounding or floating issues
    return target * (1 - smooth_factor) + (1 - target) * smooth_factor

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target, weight=None):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension
            target = torch.squeeze(target, dim=1)
        if weight is not None:
            return self.loss(input, target, weight)
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid', smooth_factor=0.0):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.smooth_factor = smooth_factor

        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # Optionally apply label smoothing for binary segmentation
        if self.smooth_factor > 0.0:
            target = label_smooth(target, self.smooth_factor)

        per_channel_dice = self.dice(input, target, weight=self.weight)

        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())

def normal_cosine_loss(pred, target):
    """
    pred:   [B, 3, D, H, W] predicted normals
    target: [B, 3, D, H, W] ground-truth normals
    Returns a scalar tensor = 1 - mean_cosine_similarity
    """
    # Compute cosine similarity over the channel dimension (dim=1),
    # ignoring batch (B) and spatial (D,H,W) dims.
    # eps avoids divide-by-zero if vectors are zero-length
    cos_sim = F.cosine_similarity(pred, target, dim=1, eps=1e-8)
    return 1.0 - cos_sim.mean()

def masked_cosine_loss(pred, target):
    """
    pred:   [B, 3, D, H, W] predicted normals
    target: [B, 3, D, H, W] ground-truth normals
    We derive a mask from target by checking which normals are nonzero.
    """

    mag = torch.norm(target, dim=1)         # [B, D, H, W]
    mask = (mag > 1e-6).float()             # [B, D, H, W]
    pred_norm = torch.norm(pred, dim=1, keepdim=True).clamp(min=1e-8)
    pred_unit = pred / pred_norm
    cos_sim = F.cosine_similarity(pred_unit, target, dim=1, eps=1e-8)  # [B, D, H, W]
    cos_sim_masked = cos_sim * mask
    valid_count = mask.sum() + 1e-8
    mean_cos_sim = cos_sim_masked.sum() / valid_count

    return 1.0 - mean_cos_sim

class BCEWithLogitsLossLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        """
        :param smoothing: float, how much to move 0 -> alpha, and 1 -> 1 - alpha
        :param reduction: 'mean' or 'sum' or 'none', same as BCEWithLogitsLoss
        """
        super().__init__()
        self.smoothing = smoothing
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets):
        """
        :param logits: model outputs [B, ..., D, H, W]
        :param targets: ground truth, in {0,1}, same shape as logits
        """
        with torch.no_grad():
            # Label smoothing:
            # For each target y in {0,1}, smoothed_y = y*(1 - 2*alpha) + alpha
            smoothed_targets = targets * (1.0 - 2.0 * self.smoothing) + self.smoothing

        loss = self.criterion(logits, smoothed_targets)
        return loss

class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = BCEWithLogitsLossLabelSmoothing(smoothing=0.1, reduction='mean')
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)
