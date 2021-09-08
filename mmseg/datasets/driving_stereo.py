# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .depth import DepthDataset


@DATASETS.register_module()
class DrivingStereo(DepthDataset):
    """Driving Stereo Dataset.

    """

    def __init__(self, **kwargs):
        super(DrivingStereo, self).__init__(
            **kwargs)
        assert osp.exists(self.img_dir)
