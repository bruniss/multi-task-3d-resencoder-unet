from configuration.config_manager import ConfigManager
from builders.utils import get_blocks, bcolors
from builders.basic_model import MultiTaskConfigurable3dUNet



class BuildNetworkFromConfig:
    def __init__(self, mgr: ConfigManager):
        self.mgr = mgr
        self.tasks = mgr.tasks
        self.patch_size = mgr.train_patch_size
        self.batch_size = mgr.train_batch_size
        self.in_channels = mgr.in_channels
        self.vram_target = mgr.vram_max
        self.autoconfigure = mgr.autoconfigure

        # copy the model kwargs so i dont overwrite the original
        self.kwargs = mgr.model_kwargs.copy()

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


    def build(self):

        model = MultiTaskConfigurable3dUNet(
            tasks=self.tasks,
            in_channels=self.in_channels,
            **self.kwargs
        )

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")


        return model


if __name__ == "__main__":
    config_file = '../tasks/example.yaml'
    mgr = ConfigManager(config_file)
    model_kwargs = vars(mgr.model_config).copy()
    builder = BuildNetworkFromConfig(mgr)


    model = builder.build()
    print(model)