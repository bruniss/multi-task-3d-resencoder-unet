import torch.nn as nn
from builders.vram_estimation import compute_3dunet_feature_map_shapes, estimate_vram
from builders.encoder import Encoder
from builders.decoder import Decoder

def get_activation_module(activation_str: str):
    """
    Returns an nn.Module (e.g. nn.Sigmoid) or None if 'none'.
    """
    act_str = activation_str.lower()
    if act_str == "none":
        return None
    elif act_str == "sigmoid":
        return nn.Sigmoid()
    elif act_str == "softmax":
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f"Unknown activation type: {activation_str}")

class NetworkFromConfig(nn.Module):
    def __init__(self, mgr):
        super().__init__()
        self.mgr = mgr
        self.tasks = mgr.tasks
        self.patch_size = mgr.train_patch_size
        self.batch_size = mgr.train_batch_size
        self.in_channels = mgr.in_channels
        self.vram_target = mgr.vram_max
        self.autoconfigure = mgr.autoconfigure

        # Read from mgr.model_config as a dict
        model_config = mgr.model_config

        # the defaults below are essentially straight copied from nnunetv2 base resnet encoder with regular conv decoder
        # you can swap these out , this is good for about 8gb -- nnunet does not really increase feature maps when scaling for
        # higher vram , but mostly prios larger patch size and adding additional stages with repeated similar feature maps
        self.model_name = model_config.get("model_name", "Model")
        self.use_timm = model_config.get("use_timm_encoder", False)
        self.basic_encoder_block = model_config.get("basic_encoder_block", "BasicBlockD")
        self.basic_decoder_block = model_config.get("basic_decoder_block", "ConvBlock")
        self.features_per_stage = model_config.get("features_per_stage", [32, 64, 128, 256, 320, 320])
        self.num_stages = model_config.get("num_stages", 6)
        self.n_blocks_per_stage = model_config.get("n_blocks_per_stage", [1, 3, 4, 6, 6, 6])
        self.n_conv_per_stage_decoder = model_config.get("n_conv_per_stage_decoder", [1, 1, 1, 1, 1])
        self.bottleneck_block = model_config.get("bottleneck_block", "BasicBlockD")
        self.op_dims = model_config.get("op_dims", 3)
        self.kernel_sizes = model_config.get("kernel_sizes", [3, 3, 3, 3, 3])

        # We may read these, but we override them below based on op_dims:
        self.conv_bias = model_config.get("conv_bias", False)
        self.norm_op_kwargs = model_config.get("norm_op_kwargs", {"affine": False, "eps": 1e-5})
        self.dropout_op_kwargs = model_config.get("dropout_op_kwargs", {"p": 0.0})
        self.nonlin = model_config.get("nonlin", "nn.LeakyReLU")
        self.nonlin_kwargs = model_config.get("nonlin_kwargs", {"inplace": True})
        self.strides = model_config.get("strides", [1, 2, 2, 2, 2, 2])
        self.return_skips = model_config.get("return_skips", True)
        self.do_stem = model_config.get("do_stem", True)
        self.stem_channels = model_config.get("stem_channels", None)
        self.bottleneck_channels = model_config.get("bottleneck_channels", None)
        self.stochastic_depth_p = model_config.get("stochastic_depth_p", 0.0)
        self.squeeze_excitation = model_config.get("squeeze_excitation", False)
        self.squeeze_excitation_reduction_ratio = 1.0 / 16.0 if self.squeeze_excitation else None

        if self.patch_size[0] < 128 and self.patch_size[1] < 128 and self.patch_size[2] < 128:
            print(f"With patch size of 128^3 we can't go above 6 layers without using anisotropic pooling. "
                  f"Recommend to either increase patch size or batch size.")

        self.stem_n_channels = self.features_per_stage[0]
        # Decide 2D vs 3D conv/pool based on op_dims
        if self.op_dims == 2:
            self.conv_op = nn.Conv2d
            self.pool_op = nn.AvgPool2d
            self.norm_op = nn.InstanceNorm2d
            self.dropout_op = nn.Dropout2d
        else:
            self.conv_op = nn.Conv3d
            self.pool_op = nn.AvgPool3d
            self.norm_op = nn.InstanceNorm3d
            self.dropout_op = nn.Dropout3d

        # convert activation strings to actual classes
        if self.nonlin == "nn.LeakyReLU":
            self.nonlin = nn.LeakyReLU
            self.nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        elif self.nonlin == "nn.ReLU":
            self.nonlin = nn.ReLU
            self.nonlin_kwargs = {"inplace": True}

        if self.bottleneck_block == "BottleneckBlockD":
            if self.bottleneck_channels is None:
                self.bottleneck_channels = [fps // 4 for fps in self.features_per_stage]
            elif isinstance(self.bottleneck_channels, int):
                self.bottleneck_channels = [self.bottleneck_channels] * len(self.features_per_stage)
            else:
                # if you did gave a number, this will do nothing; well assume you know what you are doing
                pass
        else:
            # If we’re not using BottleneckD, it’s either BasicBlock, etc.
            # Then bottleneck_channels typically don’t matter, so we can set them to None or ignore them
            self.bottleneck_channels = None

        # Shared encoder
        self.shared_encoder = Encoder(
            input_channels=self.in_channels,
            basic_block=self.basic_encoder_block,
            n_stages=self.num_stages,
            features_per_stage=self.features_per_stage,
            n_blocks_per_stage=self.n_blocks_per_stage,
            bottleneck_block=self.bottleneck_block,
            conv_op=self.conv_op,
            kernel_sizes=self.kernel_sizes,
            conv_bias=self.conv_bias,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            strides=self.strides,
            return_skips=self.return_skips,
            do_stem=self.do_stem,
            stem_channels=self.stem_n_channels,
            bottleneck_channels=self.bottleneck_channels,
            stochastic_depth_p=self.stochastic_depth_p,
            squeeze_excitation=self.squeeze_excitation,
            squeeze_excitation_reduction_ratio=self.squeeze_excitation_reduction_ratio
        )

        self.task_decoders = nn.ModuleDict()
        self.task_activations = nn.ModuleDict()

        for task_name, task_info in self.tasks.items():
            out_channels = task_info["channels"]  # number of channels for this task's output
            activation_str = task_info.get("activation", "none")

            # one decoder per task
            self.task_decoders[task_name] = Decoder(
                encoder=self.shared_encoder,
                basic_block=self.basic_decoder_block,
                num_classes=out_channels,
                n_conv_per_stage=self.n_conv_per_stage_decoder,
                deep_supervision=False
            )

            self.task_activations[task_name] = get_activation_module(activation_str)

        print(f"--- NetworkFromConfig initialized with the following settings ---")
        print(f"model_name: {self.model_name}")
        print(f"use_timm_encoder: {self.use_timm}")
        print(f"basic_encoder_block: {self.basic_encoder_block}")
        print(f"basic_decoder_block: {self.basic_decoder_block}")
        print(f"features_per_stage: {self.features_per_stage}")
        print(f"num_stages: {self.num_stages}")
        print(f"n_blocks_per_stage: {self.n_blocks_per_stage}")
        print(f"n_conv_per_stage_decoder: {self.n_conv_per_stage_decoder}")
        print(f"bottleneck_block: {self.bottleneck_block}")
        print(f"op_dims: {self.op_dims}")
        print(f"kernel_sizes: {self.kernel_sizes}")
        print(f"conv_bias: {self.conv_bias}")
        print(f"norm_op_kwargs: {self.norm_op_kwargs}")
        print(f"dropout_op_kwargs: {self.dropout_op_kwargs}")
        print(f"nonlin: {self.nonlin}")
        print(f"nonlin_kwargs: {self.nonlin_kwargs}")
        print(f"strides: {self.strides}")
        print(f"return_skips: {self.return_skips}")
        print(f"do_stem: {self.do_stem}")
        print(f"stem_channels: {self.stem_channels}")
        print(f"bottleneck_channels: {self.bottleneck_channels}")
        print(f"stochastic_depth_p: {self.stochastic_depth_p}")
        print(f"squeeze_excitation: {self.squeeze_excitation}")
        print(f"squeeze_excitation_reduction_ratio: {self.squeeze_excitation_reduction_ratio}")
        print(f"patch_size (from mgr): {self.patch_size}")
        print(f"batch_size (from mgr): {self.batch_size}")
        print(f"in_channels (from mgr): {self.in_channels}")
        print(f"vram_target (from mgr): {self.vram_target}")
        print(f"autoconfigure (from mgr): {self.autoconfigure}")
        print(f"tasks (from mgr): {self.tasks}")
        print("-------------------------------------------------------------")

    def forward(self, x):
        # Forward through the shared encoder
        skips = self.shared_encoder(x)

        results = {}
        for task_name, decoder in self.task_decoders.items():
            logits = decoder(skips)  # shape [B, out_channels, D, H, W]

            # apply the activation if not in training mode
            activation_fn = self.task_activations[task_name]
            if activation_fn is not None and not self.training:
                logits = activation_fn(logits)

            results[task_name] = logits
        return results
