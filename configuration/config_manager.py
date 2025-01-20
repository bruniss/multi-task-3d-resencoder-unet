import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, config_file):
        self._config_path = Path(config_file)

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        self.tr_info = config["tr_setup"]
        self.tr_configs = config["tr_config"]
        self.model_config = config["model_config"]
        self.dataset_config = config["dataset_config"]
        self.inference_config = config["inference_config"]

        self.model_name = self.tr_info.get("model_name", "Model")
        self.vram_max = float(self.tr_info.get("vram_max", 16))
        self.autoconfigure = bool(self.tr_info.get("autoconfigure", True))
        self.tr_val_split = float(self.tr_info.get("tr_val_split", 0.95))
        self.dilate_label = bool(self.tr_info.get("dilate_label", False))

        ckpt_out_base = self.tr_info.get("ckpt_out_base", "./checkpoints/")
        self.ckpt_out_base = Path(ckpt_out_base)
        if not self.ckpt_out_base.exists():
            self.ckpt_out_base.mkdir(parents=True)
        ckpt_path = self.tr_info.get("checkpoint_path", None)
        self.checkpoint_path = Path(ckpt_path) if ckpt_path else None

        self.load_weights_only = bool(self.tr_info.get("load_weights_only", False))
        self.tensorboard_log_dir = self.tr_info.get("tensorboard_log_dir", "./tensorboard_logs/")

        # Training config
        self.optimizer = self.tr_configs.get("optimizer", "AdamW")
        self.initial_lr = float(self.tr_configs.get("initial_lr", 1e-3))
        self.weight_decay = float(self.tr_configs.get("weight_decay", 0))
        self.train_patch_size = tuple(self.tr_configs.get("patch_size", [192, 192, 192]))
        self.train_batch_size = int(self.tr_configs.get("batch_size", 2))
        self.gradient_accumulation = int(self.tr_configs.get("gradient_accumulation", 1))
        self.max_steps_per_epoch = int(self.tr_configs.get("max_steps_per_epoch", 500))
        self.max_val_steps_per_epoch = int(self.tr_configs.get("max_val_steps_per_epoch", 25))
        self.train_num_dataloader_workers = int(self.tr_configs.get("num_dataloader_workers", 4))
        self.max_epoch = int(self.tr_configs.get("max_epoch", 500))

        # Dataset config
        self.min_labeled_ratio = float(self.dataset_config.get("min_labeled_ratio", 0.1))
        self.min_bbox_percent = float(self.dataset_config.get("min_bbox_percent", 0.95))
        self.use_cache = bool(self.dataset_config.get("use_cache", True))
        self.cache_folder = Path(self.dataset_config.get("cache_folder", "patch_cache"))

        self.targets = self.dataset_config.get("targets", {})

        self.is_wk = self.dataset_config.get("is_wk", False)
        self.is_wk_zarr_link = self.dataset_config.get("is_wk_zarr_link", False)
        self.wk_url = self.dataset_config.get("wk_url", None)
        self.wk_token = self.dataset_config.get("wk_token", None)

        self.in_channels = 1
        self.spacing= [1, 1, 1]
        self.out_channels = ()
        for target_name, task_info in self.targets.items():
            if 'out_channels' not in task_info:
                raise ValueError(f"Target {target_name} is missing out_channels specification")
            self.out_channels += (task_info['out_channels'],)

        # Inference config
        self.infer_checkpoint_path = self.inference_config.get("checkpoint_path", None)
        self.infer_patch_size = tuple(self.inference_config.get("patch_size", self.train_patch_size))
        self.infer_batch_size = int(self.inference_config.get("batch_size", self.train_batch_size))
        self.infer_output_path = self.inference_config.get("output_path", "./outputs")
        self.infer_output_format = self.inference_config.get("output_format", "zarr")
        self.infer_type = self.inference_config.get("type", "np.uint8")
        self.infer_output_targets = self.inference_config.get("output_targets", ['all'])
        self.infer_overlap = float(self.inference_config.get("overlap", 0.25))

        self._print_summary()

    def save_config(self):
        """
        Dump the current config (including any updates in model_config["final_config"])
        to a YAML file in the same directory as the original config, but
        with "_final" appended before the extension.

        E.g. "my_config.yaml" -> "my_config_final.yaml"
        """
        # Reconstruct the full config dictionary
        combined_config = {
            "tr_setup": self.tr_info,
            "tr_config": self.tr_configs,
            "model_config": self.model_config,
            "dataset_config": self.dataset_config,
            "inference_config": self.inference_config,
        }

        # Figure out original path parts
        original_stem = self._config_path.stem  # e.g. "my_config"
        original_ext = self._config_path.suffix  # e.g. ".yaml"
        original_parent = self._config_path.parent

        # Create the new filename with "_final" inserted
        final_filename = f"{original_stem}_final{original_ext}"

        # Full path to the new file
        final_path = original_parent / final_filename

        # Write out the YAML
        with final_path.open("w") as f:
            yaml.safe_dump(combined_config, f, sort_keys=False)

        print(f"Configuration saved to: {final_path}")

    def _print_summary(self):
        print("____________________________________________")
        print("Training Setup (tr_info):")
        for k, v in self.tr_info.items():
            print(f"  {k}: {v}")

        print("\nTraining Config (tr_configs):")
        for k, v in self.tr_configs.items():
            print(f"  {k}: {v}")

        print("\nDataset Config (dataset_config):")
        for k, v in self.dataset_config.items():
            print(f"  {k}: {v}")

        print("\nInference Config (inference_config):")
        for k, v in self.inference_config.items():
            print(f"  {k}: {v}")
        print("____________________________________________")
