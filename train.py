import os
from pathlib import Path
import json
from types import SimpleNamespace

from tqdm import tqdm
import numpy as np
from pytorch3dunet.unet3d.model import MultiTaskResidualUNetSE3D
from pytorch3dunet.augment.transforms import Standardize

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataloading.dataset import ZarrSegmentationDataset3D
from visualization.plotting import save_debug_gif,log_3d_slices_as_images, debug_dataloader_plot, export_data_dict_as_tif
from losses.losses import masked_cosine_loss, BCEWithLogitsLossLabelSmoothing, BCEDiceLoss


class BaseTrainer:
    """
            model_name: what you want the model to be called, and what the epochs are saved as
            patch_size: size of the patches to be cropped from the volume
            f_maps: number of feature maps in each level of the UNet
            n_levels: number of levels in the UNet
            ignore_label: value in the label volume that we will not compute losses against
            loss_only_on_label: if True, only compute loss against labeled regions
            label_smoothing: soften the loss on the label by a percentage
            min_labeled_ratio: minimum ratio of labeled pixels in a patch to be considered valid
            min_bbox_percent: minimum area a bounding box containing _all_ the labels must occupy, as a percentage of the patch size
            use_cache: if True, use a cache file to save the valid patches so you dont have to recompute
            cache_file: path to the cache file, if it exists, and also the path the new one will save to
            tasks: dictionary of tasks to be used in the model, each task must contain the number of channels and the activation type
            """

    def __init__(self,
                 config_file: str,
                 debug_dataloader: bool = False):
        with open(config_file, "r") as f:
            config = json.load(f)

        tr_params = SimpleNamespace(**config["tr_params"])
        model_config = SimpleNamespace(**config["model_config"])
        dataset_config = SimpleNamespace(**config["dataset_config"])

        # --- configs --- #
        self.model_name = getattr(tr_params, "model_name", "SheetNorm")
        self.patch_size = tuple(getattr(tr_params, "patch_size", [192, 192, 192]))
        self.batch_size = int(getattr(tr_params, "batch_size", 2))
        self.gradient_accumulation = int(getattr(tr_params, "gradient_accumulation", 1))
        self.optimizer = str(getattr(tr_params, "optimizer", "AdamW"))
        self.tr_val_split = float(getattr(tr_params, "tr_val_split", 0.95))
        self.f_maps = list(getattr(model_config, "f_maps", [32, 64, 128, 256]))
        self.num_levels = int(getattr(model_config, "n_levels", 6))
        self.ignore_label = getattr(tr_params, "ignore_label", None)
        self.dilate_label = bool(getattr(tr_params, "dilate_label", False))
        self.loss_only_on_label = bool(getattr(tr_params, "loss_only_on_label", False))
        self.label_smoothing = float(getattr(tr_params, "label_smoothing", 0.2))
        self.min_labeled_ratio = float(getattr(tr_params, "min_labeled_ratio", 0.1))
        self.min_bbox_percent = float(getattr(tr_params, "min_bbox_percent", 0.95))
        self.use_cache = bool(getattr(dataset_config, "use_cache", True))
        self.cache_file = Path((getattr(dataset_config, "cache_file",'valid_patches.json')))
        self.max_steps_per_epoch = int(getattr(tr_params, "max_steps_per_epoch", 500))
        self.max_val_steps_per_epoch = int(getattr(tr_params, "max_val_steps_per_epoch", 25))
        self.max_epoch = int(getattr(tr_params, "max_epoch", 500))
        self.initial_lr = float(getattr(tr_params, "initial_lr", 1e-3))
        self.weight_decay = float(getattr(tr_params, "weight_decay", 1e-4))
        self.ckpt_out_base = Path(getattr(tr_params, "ckpt_out_base", "./checkpoints/"))
        self.checkpoint_path = getattr(tr_params, "checkpoint_path", None)
        self.load_weights_only = getattr(tr_params, "load_weights_only", False)
        self.num_dataloader_workers = int(getattr(tr_params, "num_dataloader_workers", 4))
        self.tensorboard_log_dir = str(getattr(tr_params, "tensorboard_log_dir", "./tensorboard_logs/"))
        self.debug_dataloader = debug_dataloader
        self.loss_fns = {} # these get defined later

        self.normalization = Standardize(channelwise=False)

        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
        else:
            self.checkpoint_path = None

        os.makedirs(self.ckpt_out_base, exist_ok=True)
        #self.data_path = Path(getattr(dataset_config, "data_path", None))

        #targets = dataset_config.targets
        #self.tasks = {
        #    key: {
        #        "dataset_path": val["dataset_path"],
        #        "channels": val["channels"],
        #        "activation": val["activation"],
        #        "weight": val.get("weight", 1.0)
        #    }
        #    for key, val in targets.items()
        #}

        self.volume_paths = dataset_config.volume_paths
        self.tasks = dataset_config.targets


    def train(self):

        model = MultiTaskResidualUNetSE3D(
            in_channels=1,
            tasks=self.tasks,
            f_maps=self.f_maps,
            num_levels=self.num_levels
        )

        dataset = ZarrSegmentationDataset3D(
            volume_paths=self.volume_paths,
            tasks=self.tasks,
            patch_size=self.patch_size,
            min_labeled_ratio=self.min_labeled_ratio,
            min_bbox_percent = self.min_bbox_percent,
            normalization = self.normalization,
            dilate_label = False,
            transforms = None,
            use_cache = self.use_cache,
            cache_file=Path(self.cache_file)
        )

        if self.debug_dataloader:
            export_data_dict_as_tif(
                dataset=dataset,
                num_batches=10,
                out_dir="my_debug_dir"  # directory for saving .png files
            )
            print("Debug dataloader plots generated; exiting training early.")
            return  # <--- ends the 'train' function here

        device = torch.device('cuda')
        model = torch.compile(model)
        model = model.to(device)

        # --- losses ---- #

        self.loss_fns = {}
        for task_name, task_info in self.tasks.items():

            # i want to use cosine loss specifically for normals
            if task_name in ("normal", "normals"):
                self.loss_fns[task_name] = masked_cosine_loss

            else:
                #self.loss_fns[task_name] = BCEWithLogitsLossLabelSmoothing(smoothing=self.label_smoothing)
                self.loss_fns[task_name] = BCEDiceLoss(alpha=0.5, beta=0.5)
        if self.optimizer == "SGD":
            optimizer = SGD(
                model.parameters(),
                lr=self.initial_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = AdamW(
                model.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay
            )

        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=self.max_epoch,
                                      eta_min=0)


        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        train_val_split = self.tr_val_split
        split = int(np.floor(train_val_split * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]
        batch_size = self.batch_size

        # apply gradient accumulation
        grad_accumulate_n = self.gradient_accumulation

        scaler = torch.cuda.amp.GradScaler()

        # ---- dataloading ----- #

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(train_indices),
                                pin_memory=True,
                                num_workers=self.num_dataloader_workers)
        val_dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    sampler=SubsetRandomSampler(val_indices),
                                    pin_memory=True,
                                    num_workers=self.num_dataloader_workers)

        start_epoch = 0
        if self.checkpoint_path is not None and Path(self.checkpoint_path).exists():
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=device)

            # Always load model weights
            model.load_state_dict(checkpoint['model'])

            if not self.load_weights_only:
                # Only load optimizer, scheduler, epoch if we are NOT in "weights_only" mode
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch + 1}")
            else:
                # Start a 'new run' from epoch 0 or 1
                start_epoch = 0
                scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epoch, eta_min=0)
                print("Loaded model weights only; starting new training run from epoch 1.")

        writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        global_step = 0
        # ---- training! ----- #
        for epoch in range(start_epoch, self.max_epoch):
            model.train()

            train_running_losses = {t_name: 0.0 for t_name in self.tasks}
            pbar = tqdm(enumerate(dataloader), total=self.max_steps_per_epoch)
            steps = 0

            for i, data_dict in pbar:
                if i >= self.max_steps_per_epoch:
                    break

                global_step += 1

                inputs = data_dict["image"].to(device, dtype=torch.float32)
                targets_dict = {
                    k: v.to(device, dtype=torch.float32)
                    for k, v in data_dict.items()
                    if k != "image"
                }

                # forward
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    total_loss = 0.0
                    per_task_losses = {}

                    for t_name, t_gt in targets_dict.items():
                        t_pred = outputs[t_name]
                        t_loss_fn = self.loss_fns[t_name]
                        task_weight = self.tasks[t_name].get("weight", 1.0)
                        t_loss = t_loss_fn(t_pred, t_gt) * task_weight

                        total_loss += t_loss
                        train_running_losses[t_name] += t_loss.item()

                        # Also store the *current batch* loss for that task
                        per_task_losses[t_name] = t_loss.item()

                # backward
                scaler.scale(total_loss).backward()
                if (i + 1) % grad_accumulate_n == 0 or (i + 1) == len(dataloader):

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                    scaler.step(optimizer)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()

                steps += 1

                desc_parts = []
                for t_name in self.tasks:
                    avg_t_loss = train_running_losses[t_name] / steps
                    desc_parts.append(f"{t_name}: {avg_t_loss:.4f}")

                desc_str = f"Epoch {epoch + 1} => " + " | ".join(desc_parts)
                pbar.set_description(desc_str)

            pbar.close()

            for t_name in self.tasks:
                epoch_avg = train_running_losses[t_name] / steps
                writer.add_scalar(f"train/{t_name}_loss", epoch_avg, epoch)
            print(f"[Train] Epoch {epoch + 1} completed.")

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
            }, f"{self.ckpt_out_base}/{self.model_name}_{epoch + 1}.pth")

            # ---- validation ----- #
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_running_losses = {t_name: 0.0 for t_name in self.tasks}
                    val_steps = 0

                    pbar = tqdm(enumerate(val_dataloader), total=self.max_val_steps_per_epoch)
                    for i, data_dict in pbar:
                        if i >= self.max_val_steps_per_epoch:
                            break

                        inputs = data_dict["image"].to(device, dtype=torch.float32)
                        targets_dict = {
                            k: v.to(device, dtype=torch.float32)
                            for k, v in data_dict.items()
                            if k != "image"
                        }

                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            total_val_loss = 0.0
                            for t_name, t_gt in targets_dict.items():
                                t_pred = outputs[t_name]
                                t_loss_fn = self.loss_fns[t_name]
                                t_loss = t_loss_fn(t_pred, t_gt)

                                total_val_loss += t_loss
                                val_running_losses[t_name] += t_loss.item()

                            val_steps += 1

                            if i == 0:
                                b_idx = 0  # pick which sample in the batch to visualize
                                # Slicing shape: [1, C, D, H, W]
                                inputs_first = inputs[b_idx: b_idx + 1]

                                targets_dict_first = {}
                                for t_name, t_tensor in targets_dict.items():
                                    targets_dict_first[t_name] = t_tensor[b_idx: b_idx + 1]

                                outputs_dict_first = {}
                                for t_name, p_tensor in outputs.items():
                                    outputs_dict_first[t_name] = p_tensor[b_idx: b_idx + 1]

                                # create debug gif
                                save_debug_gif(
                                    input_volume=inputs_first,
                                    targets_dict=targets_dict_first,
                                    outputs_dict=outputs_dict_first,
                                    tasks_dict=self.tasks,
                                    # your dictionary, e.g. {"sheet": {"activation":"sigmoid"}, "normals": {"activation":"none"}}
                                    epoch=epoch,
                                    save_path=f"{self.model_name}_{epoch}_debug.gif",
                                    show_normal_magnitude=("normals" in targets_dict_first)  # show normal magnitude if your "normals" task exists
                                )

                                # Log 2D slices to TensorBoard --
                                # input volume
                                log_3d_slices_as_images(
                                    writer=writer,
                                    tag_prefix="val_input",
                                    volume_3d=inputs_first,
                                    global_step=epoch,
                                    max_slices=8  # Or whatever you prefer
                                )

                                # each task’s GT slices
                                for t_name, gt_vol in targets_dict_first.items():
                                    log_3d_slices_as_images(
                                        writer=writer,
                                        tag_prefix=f"val_{t_name}_gt",
                                        volume_3d=gt_vol,
                                        task_name=t_name,  # needed if you want to apply activation logic
                                        tasks_dict=self.tasks,
                                        global_step=epoch,
                                        max_slices=4
                                    )

                                # each task’s predicted slices
                                for t_name, pred_vol in outputs_dict_first.items():
                                    log_3d_slices_as_images(
                                        writer=writer,
                                        tag_prefix=f"val_{t_name}_pred",
                                        volume_3d=pred_vol,
                                        task_name=t_name,
                                        tasks_dict=self.tasks,
                                        global_step=epoch,
                                        max_slices=4
                                    )

                    desc_parts = []
                    for t_name in self.tasks:
                        avg_loss_for_t = val_running_losses[t_name] / val_steps
                        desc_parts.append(f"{t_name} {avg_loss_for_t:.4f}")
                    desc_str = "Val: " + " | ".join(desc_parts)
                    pbar.set_description(desc_str)

                pbar.close()

                # Final avg for each task
                for t_name in self.tasks:
                    val_avg = val_running_losses[t_name] / val_steps
                    print(f"Task '{t_name}', epoch {epoch + 1} avg val loss: {val_avg:.4f}")

            scheduler.step()

        print('Training Finished!')
        torch.save(model.state_dict(), f'{self.model_name}_final.pth')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train script for MultiTaskResidualUNetSE3D.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to your config file. Use the same one you used for training!")
    parser.add_argument("--debug_dataloader", action="store_true")
    args = parser.parse_args()

    trainer = BaseTrainer(args.config_path, args.debug_dataloader)
    trainer.train()





# During training, you'll get a dict with all outputs
# outputs = model(input_tensor)
# sheet_pred = outputs['sheet']          # Shape: [B, 1, D, H, W]
# normals_pred = outputs['normals']      # Shape: [B, 3, D, H, W]
# affinities_pred = outputs['affinities']  # Shape: [B, N_affinities, D, H, W]