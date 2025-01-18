import logging
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
import yaml

#current_time = datetime.now()
#formatted_time = current_time.strftime("%Y-%m-%d %H:%M")
#logger = logging.basicConfig(filename=f'logs/{formatted_time}', level=logging.INFO)
import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Instead of SimpleNamespace, store them directly as dicts
        self.tr_info = config["tr_setup"]
        self.tr_configs = config["tr_config"]
        self.model_config = config["model_config"]
        self.dataset_config = config["dataset_config"]
        self.inference_config = config["inference_config"]

        # Now read your "training setup" keys as dictionary lookups:
        self.model_name = self.tr_info.get("model_name", "Model")
        self.vram_max = float(self.tr_info.get("vram_max", 16))
        self.autoconfigure = bool(self.tr_info.get("autoconfigure", True))
        self.tr_val_split = float(self.tr_info.get("tr_val_split", 0.95))
        self.dilate_label = bool(self.tr_info.get("dilate_label", False))

        ckpt_out_base = self.tr_info.get("ckpt_out_base", "./checkpoints/")
        self.ckpt_out_base = Path(ckpt_out_base)
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
        self.in_channels = int(self.dataset_config.get("in_channels", 1))
        self.tasks = self.dataset_config.get("targets", {})
        self.volume_paths = self.dataset_config.get("volume_paths", [])

        # For output channels, sum up the channels of each task:
        self.out_channels = ()
        for _, task_info in self.tasks.items():
            self.out_channels += (task_info["channels"],)
        self.num_tasks = len(self.tasks)

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

    def _print_summary(self):
        print("____________________________________________")
        print("Training Setup (tr_info):")
        for k, v in self.tr_info.items():
            print(f"  {k}: {v}")

        print("\nTraining Config (tr_configs):")
        for k, v in self.tr_configs.items():
            print(f"  {k}: {v}")

        print("\nModel Config (model_config):")
        for k, v in self.model_config.items():
            print(f"  {k}: {v}")

        print("\nDataset Config (dataset_config):")
        for k, v in self.dataset_config.items():
            print(f"  {k}: {v}")

        print("\nInference Config (inference_config):")
        for k, v in self.inference_config.items():
            print(f"  {k}: {v}")
        print("____________________________________________")



