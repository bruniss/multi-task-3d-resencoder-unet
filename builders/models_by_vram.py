from builders.models import MultiTaskConfigurable3dUNet
from builders.blocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders, nnUNetStyleResNetBlockSE
from builders.utils import get_class, number_of_features_per_level, get_blocks

class MultiTaskConfigurable3dUNet_8gb(MultiTaskConfigurable3dUNet):
    def __init__(self, tasks, **kwargs):
        super().__init__(f_maps=[32, 64, 128, 256, 320],
                         basic_module=nnUNetStyleResNetBlockSE,
                         se_module='sse',
                         conv_kernel_size=3,
                         conv_padding=1,
                         conv_upscale=2,
                         dropout_prob=0.1,
                         layer_order='gcr',
                         num_groups=8,
                         pool_kernel_size=2,
                         upsample='default',
                         is3d=True,
                         **kwargs)

class MultiTaskConfigurable3dUNet_16gb(MultiTaskConfigurable3dUNet):
    def __init__(self, tasks, **kwargs):
        super().__init__(f_maps=[32, 64, 128, 256, 512, 768],
                         basic_module=nnUNetStyleResNetBlockSE,
                         se_module='sse',
                         conv_kernel_size=3,
                         conv_padding=1,
                         conv_upscale=2,
                         dropout_prob=0.1,
                         layer_order='gcr',
                         num_groups=8,
                         pool_kernel_size=2,
                         upsample='default',
                         is3d=True,
                         **kwargs)

