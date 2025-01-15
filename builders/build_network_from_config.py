import yaml
import torch
from builders.utils import get_blocks, bcolors
from builders.basic_model import MultiTaskConfigurable3dUNet
from builders.models_by_vram import MultiTaskConfigurable3dUNet_8gb, MultiTaskConfigurable3dUNet_16gb


class BuildNetworkFromConfig:
    def __init__(
        self,
        tasks,
        patch_size,
        in_channels,
        batch_size,
        vram_target,
        autoconfigure,
        **model_kwargs
    ):
        self.tasks = tasks
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.vram_target = vram_target
        self.autoconfigure = autoconfigure

        # copy the model kwargs so i dont overwrite the original
        self.kwargs = model_kwargs.copy()

        z_dim = self.patch_size[0]
        n_downsamples = len(self.kwargs["f_maps"]) - 1  # e.g. for f_maps=[64,128,256,512,768], n_downsamples=4
        final_z_after_downsampling = z_dim / (2 ** n_downsamples)

        if final_z_after_downsampling < 1:
            print(f"{bcolors.WARNING} Your Z dimension will go below 1 after downsampling => switching pool_kernel_size to [1,2,2] {bcolors.ENDC}")
            self.kwargs["pool_kernel_size"] = [1, 2, 2]

        # Basic block checks
        if "basic_module" in self.kwargs:
            self.kwargs["basic_module"] = get_blocks(self.kwargs["basic_module"])
            block_class_name = self.kwargs["basic_module"].__name__
            # If it’s an SE block but no se_module specified, default to scse
            if 'SE' in block_class_name:
                if "se_module" not in self.kwargs:
                    self.kwargs["se_module"] = "scse"
            else:
                # If it’s not an SE block, remove se_module if present
                self.kwargs.pop("se_module", None)

    def estimate_vram_usage(self):
        device = "cuda"
        model = MultiTaskConfigurable3dUNet(
            tasks=self.tasks,
            in_channels=self.in_channels,
            **self.kwargs
        ).to(device)

        # create a dummy input with shape = (batch_size, in_channels, D, H, W)
        dummy_input = torch.randn(
            (self.batch_size, self.in_channels, *self.patch_size),
            dtype=torch.float32,
            device=device
        )

        # clear memory states
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # forward pass (without autocast => "worst-case" VRAM usage)
        output = model(dummy_input)

        # backward pass to estimate training usage
        if isinstance(output, dict):
            # multiple heads => sum them
            loss = sum([o.sum() for o in output.values()])
        else:
            loss = output.sum()

        loss.backward()

        # get peak memory usage (in bytes), convert to MB
        mem_bytes = torch.cuda.max_memory_allocated(device)
        mem_mb = mem_bytes / (1024 ** 2)

        return mem_mb


    def build(self):

        if self.vram_target < 8024 and self.autoconfigure == True:
            model = MultiTaskConfigurable3dUNet_8gb(tasks=self.tasks,
                                                    in_channels=self.in_channels,
                                                    **self.kwargs)

        if self.vram_target < 16000 and self.autoconfigure == True:
            model = MultiTaskConfigurable3dUNet_16gb(tasks=self.tasks,
                                                    in_channels=self.in_channels,
                                                    **self.kwargs)

        else:
            model = MultiTaskConfigurable3dUNet(
                tasks=self.tasks,
                in_channels=self.in_channels,
                **self.kwargs
            )

        return model
