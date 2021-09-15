import torch
import torch.nn as nn

from ..builder import LOSSES


def get_loss_func(method: str):
    if method is 'l1':
        return nn.L1Loss()
    elif method is 'mse':
        return nn.MSELoss()
    else:
        raise ValueError("Unknown supervised loss function")

@LOSSES.register_module()
class SupervisedInverseDepthLoss(nn.Module):
    """Mean Square Error Loss.

    Args:
        use_mask (bool, optional): Whether to use mask mean squared entropy
        loss.
            Defaults to False.
    """

    def __init__(self, method='l1'):
        super(SupervisedInverseDepthLoss, self).__init__()
        self.loss_cls = get_loss_func(method)

    def forward(self, inv_depth_pred, depth_map, **kwargs):
        """Forward function."""

        mask = (depth_map.not_equal(0)).detach()

        masked_depth_map = torch.masked_select(depth_map, mask)
        masked_inv_depth_pred = torch.masked_select(inv_depth_pred, mask)

        masked_gt_depth_map = 1 / masked_depth_map

        loss = self.loss_cls(masked_inv_depth_pred, masked_gt_depth_map)

        return loss
