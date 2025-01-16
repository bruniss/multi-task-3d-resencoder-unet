# huge amounts copied from wolny pytorch3dunet. https://github.com/wolny/pytorch-3dunet/tree/master

import torch.nn as nn

from builders.blocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders, nnUNetStyleResNetBlockSE
from builders.utils import number_of_features_per_level


def get_activation_module(activation_str: str):
    """
    Simple helper that returns an nn.Module (e.g. nn.Sigmoid) or None
    if activation_str == 'none'
    """
    if activation_str.lower() == "none":
        return None
    elif activation_str.lower() == "sigmoid":
        return nn.Sigmoid()
    elif activation_str.lower() == "softmax":
        return nn.Softmax(dim=1)
    # Add others if you like
    else:
        return None  # default is None, or raise an error

class MultiTaskConfigurable3dUNet(nn.Module):

    def __init__(
            self,
            in_channels: int,
            tasks: dict,  # e.g. {"sheet": {"channels": 1, "activation": "sigmoid"}, ...}
            f_maps=[32, 64, 128, 256, 320, 528],
            basic_module=ResNetBlockSE,
            se_module='scse',
            conv_kernel_size=3,
            conv_padding=1,
            conv_upscale=2,
            dropout_prob=0.1,
            layer_order='gcr',
            num_groups=8,
            pool_kernel_size=2,
            upsample='default',
            is3d=True,
            **kwargs
    ):

        super().__init__()

        self.in_channels = in_channels
        self.tasks = tasks
        self.f_maps = f_maps
        self.num_levels = len(f_maps)
        self.basic_block= basic_module
        self.se_module= se_module
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_padding
        self.conv_upscale = conv_upscale
        self.dropout_prob = dropout_prob
        self.layer_order = layer_order
        self.num_groups = num_groups
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = pool_kernel_size
        self.upsample = upsample
        self.is3d = is3d
        self.extra_kwargs = kwargs

        if isinstance(f_maps, int):
                f_maps = number_of_features_per_level(f_maps, num_levels=self.num_levels)

        # create encoders
        self.encoders = create_encoders(
            in_channels=self.in_channels,
            f_maps=self.f_maps,
            basic_module=self.basic_block,
            se_module=self.se_module,
            conv_kernel_size=self.conv_kernel_size,
            conv_padding=self.conv_padding,
            conv_upscale=self.conv_upscale,
            dropout_prob=self.dropout_prob,
            layer_order=self.layer_order,
            num_groups=self.num_groups,
            pool_kernel_size=self.pool_kernel_size,
            is3d=self.is3d
        )

        # We'll store the final "deepest features" so we remember how many channels
        self.deepest_channels = f_maps[-1]

        # skip connections" for every level except the last
        # so that the "deepest" is separate from skip-level features.

        # for each task, create a separate decoder + final conv
        self.tasks = tasks
        self.task_decoders = nn.ModuleDict()
        self.task_final_convs = nn.ModuleDict()
        self.task_activations = nn.ModuleDict()

        for task_name, task_info in tasks.items():
            self.out_channels = task_info["channels"]
            activation_str = task_info.get("activation", "none")

            # create a new decoder path for each task
            self.task_decoders[task_name] = create_decoders(
                f_maps=self.f_maps,
                basic_module=self.basic_block,
                se_module=self.se_module,
                conv_kernel_size=self.conv_kernel_size,
                conv_padding=self.conv_padding,
                layer_order=self.layer_order,
                num_groups=self.num_groups,
                upsample=self.upsample,
                dropout_prob=self.dropout_prob,
                scale_factor=self.scale_factor,
                is3d=self.is3d
            )

            # Create the final 1x1 conv for this task
            self.task_final_convs[task_name] = nn.Conv3d(
                in_channels=self.f_maps[0],
                out_channels=self.out_channels,
                kernel_size=1
            )

            # activation
            self.task_activations[task_name] = get_activation_module(activation_str)

    def print_config(self):
        print("------------ model configuration ------------------")
        print(f"  in_channels: {self.in_channels}")
        print(f"  tasks: {self.tasks}")
        print(f"  f_maps: {self.f_maps}")
        print(f"  num_levels: {self.num_levels}")
        print(f"  basic_module: {self.basic_block}")
        print(f"  conv_kernel_size: {self.conv_kernel_size}")
        print(f"  conv_padding: {self.conv_padding}")
        print(f"  conv_upscale: {self.conv_upscale}")
        print(f"  dropout_prob: {self.dropout_prob}")
        print(f"  layer_order: {self.layer_order}")
        print(f"  num_groups: {self.num_groups}")
        print(f"  pool_kernel_size: {self.pool_kernel_size}")
        print(f"  upsample: {self.upsample}")
        print(f"  is3d: {self.is3d}")
        print(f"  extra_kwargs: {self.extra_kwargs}")
        print("--------------------------------------------------")

    def forward(self, x):
        # === Shared encoder ===
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        # Remove the deepest features from skip connections:
        # encoders_features[0] is the output of the last encoder block
        x = encoders_features[0]
        skip_features = encoders_features[1:]  # the rest are skip features

        # === Task-specific decoders + final convs ===
        results = {}
        for task_name, decoder in self.task_decoders.items():
            # Start from the deepest features
            task_x = x
            for dec, skip in zip(decoder, skip_features):
                task_x = dec(skip, task_x)

            # final 1×1 conv
            task_x = self.task_final_convs[task_name](task_x)

            # optional activation
            activation_fn = self.task_activations[task_name]
            if activation_fn is not None and not self.training:
                task_x = activation_fn(task_x)

            results[task_name] = task_x

        return results