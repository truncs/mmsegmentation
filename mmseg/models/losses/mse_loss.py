import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class MSELoss(nn.Module):
    """Mean Square Error Loss.

    Args:
        use_mask (bool, optional): Whether to use mask mean squared entropy
        loss.
            Defaults to False.
    """

    def __init__(self,
                 use_mask=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(MSELoss, self).__init__()
        self.use_mask = use_mask
        self.reduction = reduction

        if self.use_mask:
            loss_reduction = 'sum'
        else:
            loss_reduction = self.reduction

        self.loss_cls = nn.MSELoss(reduction=loss_reduction)

    def forward(self,
                input,
                target,
                mask,
                **kwargs):
        """Forward function."""
        if self.use_mask:
            input = mask * input
            target = mask * target

        loss = self.loss_cls(input, target)

        if self.use_mask and self.reduction == 'mean':
            loss = loss / mask.sum()

        return loss
