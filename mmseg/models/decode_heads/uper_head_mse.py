from mmcv.runner import force_fp32
from mmseg.ops import resize

from .uper_head import UPerHead
from ..builder import HEADS


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

    def forward_train(self, inputs, img_metas, gt_depth, train_cfg):
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
        pred = self.forward(inputs)
        losses = self.losses(pred, gt_depth)
        return losses

    @force_fp32(apply_to=('pred', ))
    def losses(self, pred, gt_depth):
        """Compute segmentation loss."""

        loss = dict()

        pred = pred.reshape(
            pred.shape[0],
            pred.shape[1],
            -1
        )
        pred = resize(
            input=pred.unsqueeze(1),
            size=gt_depth.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)

        pred = pred.squeeze()
        loss['loss_seg'] = self.loss_decode(
            pred,
            gt_depth)

        return loss
