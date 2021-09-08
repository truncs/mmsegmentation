from mmcv.runner import force_fp32
from mmseg.ops import resize

from .uper_head import UPerHead
from ..builder import HEADS
from ..losses import accuracy


@HEADS.register_module()
class UPerHeadMSE(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHeadMSE, self).__init__(
            pool_scales=pool_scales, **kwargs)

    def forward_train(self, inputs, img_metas, gt_depth, mask, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        depth_target = self.forward(inputs)
        losses = self.losses(depth_target, gt_depth, mask)
        return losses

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, depth_target, depth_label, mask):
        """Compute segmentation loss."""
        loss = dict()
        depth_target = resize(
            input=depth_target,
            size=depth_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        depth_label = depth_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            depth_target,
            depth_label,
            mask=mask)
        loss['acc_seg'] = accuracy(depth_target, depth_label)
        return loss
