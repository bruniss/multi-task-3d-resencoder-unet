import torch.nn as nn
from builders.utils import get_pool_and_conv_props, get_n_blocks_per_stage
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
        self.model_name = model_config.get("model_name", "Model")

        # the defaults below are essentially straight copied from nnunetv2 base resnet encoder with regular conv decoder
        # you can swap these out , this is good for about 8gb -- nnunet does not really increase feature maps when scaling for
        # higher vram , but mostly prios larger patch size and adding additional stages with repeated similar feature maps

        if mgr.autoconfigure:
            print("--- Autoconfiguring network from config ---")

            self.use_timm = False
            self.basic_encoder_block = "BasicBlockD"
            self.basic_decoder_block = "ConvBlock"
            self.bottleneck_block = "BasicBlockD"

            num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, final_patch_size, must_div = \
                get_pool_and_conv_props(
                    spacing=(1.0, 1.0, 1.0),
                    patch_size=mgr.train_patch_size,
                    min_feature_map_size=4,
                    max_numpool=999999
                )

            self.num_stages = len(pool_op_kernel_sizes)

            base_features = 32
            max_features = 512
            features = []
            for i in range(self.num_stages):
                feats = base_features * (2 ** i)
                features.append(min(feats, max_features))

            self.num_pool_per_axis = num_pool_per_axis
            self.pool_op_kernel_sizes = pool_op_kernel_sizes
            self.kernel_sizes = conv_kernel_sizes
            self.features_per_stage = features
            self.n_blocks_per_stage = get_n_blocks_per_stage(self.num_stages)
            self.n_conv_per_stage_decoder = [1] * (self.num_stages - 1)
            self.strides = pool_op_kernel_sizes

            print(f"------------------------------------------------------------")
            print(f"Final Autoconfigured Parameters:")
            print(f"num_stages: {self.num_stages}")
            print(f"features_per_stage: {self.features_per_stage}")
            print(f"n_blocks_per_stage: {self.n_blocks_per_stage}")
            print(f"n_conv_per_stage_decoder: {self.n_conv_per_stage_decoder}")
            print(f"strides: {self.strides}")
            print(f"final_patch_size: {final_patch_size}")
            print("-------------------------------------------------------------")

        else:
            print("--- Configuring network from config file ---")

            self.use_timm = model_config.get("use_timm_encoder", False)

            if "basic_encoder_block" not in model_config:
                raise ValueError(
                    "autoconfigure=False, but 'basic_encoder_block' was not provided in the config!"
                )
            else:
                self.basic_encoder_block = model_config["basic_encoder_block"]

            if "basic_decoder_block" not in model_config:
                raise ValueError(
                    "autoconfigure=False, but 'basic_decoder_block' was not provided in the config!"
                )
            else:
                self.basic_decoder_block = model_config["basic_decoder_block"]

            if "bottleneck_block" not in model_config:
                raise ValueError(
                    "autoconfigure=False, but 'bottleneck_block' was not provided in the config!"
                )
            else:
                self.bottleneck_block = model_config["bottleneck_block"]

            if "features_per_stage" not in model_config:
                raise ValueError(
                    "autoconfigure=False, but 'features_per_stage' was not provided in the config!"
                )
            else:
                self.features_per_stage = model_config["features_per_stage"]

            if "num_stages" not in model_config:
                raise ValueError(
                    "autoconfigure=False, but 'num_stages' was not provided in the config!"
                )
            else:
                self.num_stages = model_config["num_stages"]

            if "n_blocks_per_stage" not in model_config:
                raise ValueError(
                    "autoconfigure=False, but 'n_blocks_per_stage' was not provided in the config!"
                )
            else:
                self.n_blocks_per_stage = model_config["n_blocks_per_stage"]

            if "kernel_sizes" not in model_config:
                raise ValueError(
                    "autoconfigure=False, but 'kernel_sizes' was not provided in the config!"
                )
            else:
                self.kernel_sizes = model_config["kernel_sizes"]

            if "n_conv_per_stage_decoder" not in model_config:
                raise ValueError(
                    "autoconfigure=False, but 'n_conv_per_stage_decoder' was not provided in the config!"
                )
            else:
                self.n_conv_per_stage_decoder = model_config["n_conv_per_stage_decoder"]

            if "strides" not in model_config:
                raise ValueError(
                    "autoconfigure=False, but 'strides' was not provided in the config!"
                )
            else:
                self.strides = model_config["strides"]

            print("-------------------------------------------------------------")
            print(f"Final Manual Parameters:")
            print(f"use_timm_encoder: {self.use_timm}")
            print(f"basic_encoder_block: {self.basic_encoder_block}")
            print(f"basic_decoder_block: {self.basic_decoder_block}")
            print(f"bottleneck_block: {self.bottleneck_block}")
            print(f"features_per_stage: {self.features_per_stage}")
            print(f"num_stages: {self.num_stages}")
            print(f"n_blocks_per_stage: {self.n_blocks_per_stage}")
            print(f"kernel_sizes: {self.kernel_sizes}")
            print(f"n_conv_per_stage_decoder: {self.n_conv_per_stage_decoder}")
            print(f"strides: {self.strides}")
            print("-------------------------------------------------------------")

        # these are not currently configurable in the yaml, we read these, but we override them below based on patch dims
        # these are placeholders for when these become more configurable , but they are solid presets
        self.conv_op = model_config.get("conv_op", "nn.Conv3d")
        self.conv_op_kwargs = model_config.get("conv_op_kwargs", {"bias": False})
        self.pool_op = model_config.get("pool_op", "nn.AvgPool3d")
        self.dropout_op = model_config.get("dropout_op", "nn.Dropout3d")
        self.dropout_op_kwargs = model_config.get("dropout_op_kwargs", {"p": 0.0})
        self.norm_op = model_config.get("norm_op", "nn.InstanceNorm3d")
        self.norm_op_kwargs = model_config.get("norm_op_kwargs", {"affine": False, "eps": 1e-5})

        # these can be configured in the yaml but these are reasonable defaults. wouldn't change unless you have good reason to
        # other than se , they are mostly well tested in nnunetv2
        self.conv_bias = model_config.get("conv_bias", False)
        self.nonlin = model_config.get("nonlin", "nn.LeakyReLU")
        self.nonlin_kwargs = model_config.get("nonlin_kwargs", {"inplace": True})
        self.return_skips = model_config.get("return_skips", True)
        self.do_stem = model_config.get("do_stem", True)
        self.stem_channels = model_config.get("stem_channels", None)
        self.bottleneck_channels = model_config.get("bottleneck_channels", None)
        self.stochastic_depth_p = model_config.get("stochastic_depth_p", 0.0)
        self.squeeze_excitation = model_config.get("squeeze_excitation", False)
        self.squeeze_excitation_reduction_ratio = 1.0 / 16.0 if self.squeeze_excitation else None
        self.stem_n_channels = self.features_per_stage[0]

        if len(self.patch_size) == 2:
            self.op_dims = 2
        elif len(self.patch_size) == 3:
            self.op_dims = 3
        else:
            raise ValueError("Patch size must have either 2 or 3 dimensions!")

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

        # Suppose you have:
        #   spacing = (1.0, 1.0, 1.0) # for uniform 3D
        #   user_patch = self.train_patch_size
        #   min_feature_map_size = 4
        #   max_numpool = 999999

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
