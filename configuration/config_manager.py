import logging
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
import yaml

#current_time = datetime.now()
#formatted_time = current_time.strftime("%Y-%m-%d %H:%M")
#logger = logging.basicConfig(filename=f'logs/{formatted_time}', level=logging.INFO)

class ConfigManager:
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        self.tr_info = SimpleNamespace(**config["tr_setup"])
        self.tr_configs = SimpleNamespace(**config["tr_config"])
        self.model_config = SimpleNamespace(**config["model_config"])
        self.dataset_config = SimpleNamespace(**config["dataset_config"])
        self.inference_config = SimpleNamespace(**config["inference_config"])

        # training setup
        self.model_name = getattr(self.tr_info, "model_name", "Model")
        self.vram_max = float(getattr(self.tr_info, "vram_max", 16))
        self.autoconfigure = bool(getattr(self.tr_info, "autoconfigure", True))
        self.tr_val_split = float(getattr(self.tr_info, "tr_val_split", 0.95))
        self.dilate_label = bool(getattr(self.tr_info, "dilate_label", False))
        self.ckpt_out_base = Path(getattr(self.tr_info, "ckpt_out_base", "./checkpoints/"))
        self.checkpoint_path = getattr(self.tr_info, "checkpoint_path", None)
        self.load_weights_only = getattr(self.tr_info, "load_weights_only", False)
        self.tensorboard_log_dir = str(getattr(self.tr_info, "tensorboard_log_dir", "./tensorboard_logs/"))

        # parameters for training
        self.loss_only_on_label = bool(getattr(self.tr_configs, "loss_only_on_label", False))
        self.train_patch_size = tuple(getattr(self.tr_configs, "patch_size", [192, 192, 192]))
        self.train_batch_size = int(getattr(self.tr_configs, "batch_size", 2))
        self.gradient_accumulation = int(getattr(self.tr_configs, "gradient_accumulation", 1))
        self.optimizer = str(getattr(self.tr_configs, "optimizer", "AdamW"))
        self.ignore_label = getattr(self.tr_configs, "ignore_label", None)
        self.max_steps_per_epoch = int(getattr(self.tr_configs, "max_steps_per_epoch", 500))
        self.max_val_steps_per_epoch = int(getattr(self.tr_configs, "max_val_steps_per_epoch", 25))
        self.train_num_dataloader_workers = int(getattr(self.tr_configs, "num_dataloader_workers", 4))
        self.label_smoothing = float(getattr(self.tr_configs, "label_smoothing", 0.2))
        self.max_epoch = int(getattr(self.tr_configs, "max_epoch", 500))
        self.initial_lr = float(getattr(self.tr_configs, "initial_lr", 1e-3))
        self.weight_decay = float(getattr(self.tr_configs, "weight_decay", 0))
        self.tensorboard_log_dir = Path(self.tensorboard_log_dir) / self.model_name

        # model configuration -- no defaults here because it's handled by build_network_from_config dynamically
        self.model_kwargs = vars(self.model_config).copy()

        # dataset config
        self.min_labeled_ratio = float(getattr(self.dataset_config, "min_labeled_ratio", 0.1))
        self.min_bbox_percent = float(getattr(self.dataset_config, "min_bbox_percent", 0.95))
        self.use_cache = bool(getattr(self.dataset_config, "use_cache", True))
        self.cache_file = Path((getattr(self.dataset_config, "cache_file", 'valid_patches.json')))
        self.in_channels = int(getattr(self.dataset_config, "in_channels", 1))
        self.tasks = self.dataset_config.targets
        self.volume_paths = self.dataset_config.volume_paths
        self.out_channels = ()
        for task_name, task_info in self.tasks.items():
            self.out_channels += (task_info["channels"],)

        # inference config
        self.infer_input_path = str(getattr(self.inference_config, "input_path", None))
        self.infer_output_path = str(getattr(self.inference_config, "output_path", None))
        self.infer_input_format = str(getattr(self.inference_config, "input_format", "zarr"))
        self.infer_output_format = str(getattr(self.inference_config, "output_format", "zarr"))
        self.infer_load_all = bool(getattr(self.inference_config, "load_all", False))
        self.infer_output_dtype = str(getattr(self.inference_config, "output_type", "np.uint8"))
        self.infer_output_targets = list(getattr(self.inference_config, "output_targets", "all"))
        self.infer_overlap = float(getattr(self.inference_config, "overlap", 0.15))
        self.infer_blending = str(getattr(self.inference_config, "blending", "gaussian_importance"))
        self.infer_patch_size = tuple(getattr(self.inference_config, "patch_size", self.train_patch_size))
        self.infer_batch_size = int(getattr(self.inference_config, "batch_size", self.train_batch_size))
        self.infer_checkpoint_path = getattr(self.inference_config, "checkpoint_path", None)
        self.load_strict = bool(getattr(self.inference_config, "load_strict", True))
        self.infer_num_dataloader_workers = int(getattr(self.inference_config, "num_dataloader_workers", self.train_num_dataloader_workers))

        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
        else:
            self.checkpoint_path = None

        print("____________________________________________")
        print("             training setup: ")
        print("____________________________________________")
        for key, value in self.tr_info.__dict__.items():
            print(f"{key}: {value}")
        print("____________________________________________")
        print("             training config: ")
        print("____________________________________________")
        for key, value in self.tr_configs.__dict__.items():
            print(f"{key}: {value}")
        print("____________________________________________")
        print("             model config: ")
        print("____________________________________________")
        for key, value in self.model_config.__dict__.items():
            print(f"{key}: {value}")
        print("____________________________________________")
        print("             dataset config: ")
        print("____________________________________________")
        for key, value in self.dataset_config.__dict__.items():
            print(f"{key}: {value}")
        print("____________________________________________")

        print("inference config: ")
        for key, value in self.inference_config.__dict__.items():
            print(f"{key}: {value}")
        print("____________________________________________")

if __name__ == "__main__":
    config_path = Path("../tasks/example.yaml")

    config = ConfigManager(config_path)
    #
    # print("____________________________________________")
    # print("             training setup: ")
    # print("____________________________________________")
    # for key, value in config.tr_info.__dict__.items():
    #     print(f"{key}: {value}")
    # print("____________________________________________")
    # print("             training config: ")
    # print("____________________________________________")
    # for key, value in config.tr_configs.__dict__.items():
    #     print(f"{key}: {value}")
    # print("____________________________________________")
    # print("             model config: ")
    # print("____________________________________________")
    # for key, value in config.model_config.__dict__.items():
    #     print(f"{key}: {value}")
    # print("____________________________________________")
    # print("             dataset config: ")
    # print("____________________________________________")
    # for key, value in config.dataset_config.__dict__.items():
    #     print(f"{key}: {value}")
    # print("____________________________________________")
    #
    # print("inference config: ")
    # for key, value in config.inference_config.__dict__.items():
    #     print(f"{key}: {value}")
    # print("____________________________________________")




