import os
from pathlib import Path
import yaml

from tqdm import tqdm
import numpy as np
import logging
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataloading.dataset import ZarrSegmentationDataset3D
from visualization.plotting import save_debug_gif,log_3d_slices_as_images, debug_dataloader_plot, export_data_dict_as_tif
from builders.build_network_from_config import BuildNetworkFromConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss
from losses.losses import MaskedCosineLoss, BCEWithLogitsLossLabelSmoothing, BCEDiceLoss, BCEWithLogitsLossZSmooth
from configuration.config_manager import ConfigManager


class BaseTrainer:
    def __init__(self,
                 config_file: str,
                 verbose: bool = True,
                 debug_dataloader: bool = False):

        self.mgr = ConfigManager(config_file)
        self.debug_dataloader = debug_dataloader

    # --- build model --- #
    def _build_model(self):

        builder = BuildNetworkFromConfig(self.mgr)
        model = builder.build()
        model.print_config()

        return model

    # --- configure dataset --- #
    def _configure_dataset(self):
        dataset = ZarrSegmentationDataset3D(
            volume_paths=self.mgr.volume_paths,
            tasks=self.mgr.tasks,
            patch_size=self.mgr.train_patch_size,
            min_labeled_ratio=self.mgr.min_labeled_ratio,
            min_bbox_percent=self.mgr.min_bbox_percent,
            dilate_label=self.mgr.dilate_label,
            transforms=None,
            use_cache=self.mgr.use_cache,
            cache_file=Path(self.mgr.cache_file)
        )

        return dataset

    # --- losses ---- #
    def _build_loss(self):
        # if you override this you need to allow for a loss fn to apply to every single
        # possible target in the dictionary of targets . the easiest is probably
        # to add it in losses.loss, import it here, and then add it to the map
        LOSS_FN_MAP = {
            "BCEDiceLoss": BCEDiceLoss,
            "BCEWithLogitsLossLabelSmoothing": BCEWithLogitsLossLabelSmoothing,
            "BCEWithLogitsLossZSmooth": BCEWithLogitsLossZSmooth,
            "BCEWithLogitsLoss": BCEWithLogitsLoss,
            "BCELoss": BCELoss,
            "CrossEntropyLoss": CrossEntropyLoss,
            "MSELoss": MSELoss,
            "MaskedCosineLoss": MaskedCosineLoss
        }

        loss_fns = {}
        for task_name, task_info in self.mgr.tasks.items():
            loss_fn = task_info.get("loss_fn", "BCEDiceLoss")
            if loss_fn not in LOSS_FN_MAP:
                raise ValueError(f"Loss function {loss_fn} not found in LOSS_FN_MAP. Add it to the mapping and try again.")
            loss_kwargs = task_info.get("loss_kwargs", {})
            loss_fns[task_name] = LOSS_FN_MAP[loss_fn](**loss_kwargs)

        return loss_fns

    # --- optimizer ---- #
    def _get_optimizer(self, model):
        if self.mgr.optimizer == "SGD":
            optimizer = SGD(
                model.parameters(),
                lr=self.mgr.initial_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.mgr.weight_decay
            )
        else:
            optimizer = AdamW(
                model.parameters(),
                lr=self.mgr.initial_lr,
                weight_decay=self.mgr.weight_decay
            )
        return optimizer

    # --- scheduler --- #
    def _get_scheduler(self, optimizer):
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=self.mgr.max_epoch,
                                      eta_min=0)
        return scheduler

    # --- scaler --- #
    def _get_scaler(self):
        scaler = torch.amp.GradScaler("cuda")
        return scaler

    # --- dataloaders --- #
    def _configure_dataloaders(self, dataset):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        train_val_split = self.mgr.tr_val_split
        split = int(np.floor(train_val_split * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]
        batch_size = self.mgr.train_batch_size

        train_dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(train_indices),
                                pin_memory=True,
                                num_workers=self.mgr.train_num_dataloader_workers)
        val_dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    sampler=SubsetRandomSampler(val_indices),
                                    pin_memory=True,
                                    num_workers=self.mgr.train_num_dataloader_workers)

        return train_dataloader, val_dataloader

    def train(self):

        model = self._build_model()
        optimizer = self._get_optimizer(model)
        loss_fns = self._build_loss()
        dataset = self._configure_dataset()
        scheduler = self._get_scheduler(optimizer)
        scaler = self._get_scaler()

        device = torch.device('cuda')
        model = model.to(device)
        model = torch.compile(model)

        train_dataloader, val_dataloader = self._configure_dataloaders(dataset)

        if self.debug_dataloader:
            export_data_dict_as_tif(
                dataset=dataset,
                num_batches=25,
                out_dir="debug_dir"
            )
            print("Debug dataloader plots generated; exiting training early.")
            return

        start_epoch = 0

        if self.mgr.checkpoint_path is not None and Path(self.mgr.checkpoint_path).exists():
            print(f"Loading checkpoint from {self.mgr.checkpoint_path}")
            checkpoint = torch.load(self.mgr.checkpoint_path, map_location=device)

            # Always load model weights
            model.load_state_dict(checkpoint['model'])

            if not self.mgr.load_weights_only:
                # Only load optimizer, scheduler, epoch if we are NOT in "weights_only" mode
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch + 1}")
            else:
                # Start a 'new run' from epoch 0 or 1
                start_epoch = 0
                scheduler = self._get_scheduler(optimizer)
                print("Loaded model weights only; starting new training run from epoch 1.")

        writer = SummaryWriter(log_dir=self.mgr.tensorboard_log_dir)
        global_step = 0
        grad_accumulate_n = self.mgr.gradient_accumulation

        # ---- training! ----- #
        for epoch in range(start_epoch, self.mgr.max_epoch):
            model.train()

            train_running_losses = {t_name: 0.0 for t_name in self.mgr.tasks}
            pbar = tqdm(enumerate(train_dataloader), total=self.mgr.max_steps_per_epoch)
            steps = 0

            for i, data_dict in pbar:
                if i >= self.mgr.max_steps_per_epoch:
                    break

                if epoch == 0 and i == 0:
                    for item in data_dict:
                        print(f"Items from the first batch -- Double check that your shapes and values are expected:")
                        print(f"{item}: {data_dict[item].dtype}")
                        print(f"{item}: {data_dict[item].shape}")
                        print(f"{item}: min : {data_dict[item].min()} max : {data_dict[item].max()}")

                global_step += 1

                inputs = data_dict["image"].to(device, dtype=torch.float32)
                targets_dict = {
                    k: v.to(device, dtype=torch.float32)
                    for k, v in data_dict.items()
                    if k != "image"
                }

                # forward
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    total_loss = 0.0
                    per_task_losses = {}

                    for t_name, t_gt in targets_dict.items():
                        t_pred = outputs[t_name]
                        t_loss_fn = loss_fns[t_name]
                        task_weight = self.mgr.tasks[t_name].get("weight", 1.0)
                        t_loss = t_loss_fn(t_pred, t_gt) * task_weight

                        total_loss += t_loss
                        train_running_losses[t_name] += t_loss.item()

                        # Also store the *current batch* loss for that task
                        per_task_losses[t_name] = t_loss.item()

                # backward
                # loss \ accumulation steps to maintain same effective batch size
                total_loss = total_loss / grad_accumulate_n
                # backward
                scaler.scale(total_loss).backward()

                if (i + 1) % grad_accumulate_n == 0 or (i + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                steps += 1

                desc_parts = []
                for t_name in self.mgr.tasks:
                    avg_t_loss = train_running_losses[t_name] / steps
                    desc_parts.append(f"{t_name}: {avg_t_loss:.4f}")

                desc_str = f"Epoch {epoch + 1} => " + " | ".join(desc_parts)
                pbar.set_description(desc_str)

            pbar.close()

            for t_name in self.mgr.tasks:
                epoch_avg = train_running_losses[t_name] / steps
                writer.add_scalar(f"train/{t_name}_loss", epoch_avg, epoch)
            print(f"[Train] Epoch {epoch + 1} completed.")

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
            }, f"{self.mgr.ckpt_out_base}/{self.mgr.model_name}_{epoch + 1}.pth")

            # clean up old checkpoints -- currently just keeps 10 newest
            all_checkpoints = sorted(
                self.mgr.ckpt_out_base.glob(f"{self.mgr.model_name}_*.pth"),
                key=lambda x: x.stat().st_mtime
            )

            # if more than 10, remove the oldest
            while len(all_checkpoints) > 10:
                oldest = all_checkpoints.pop(0)
                oldest.unlink()  #

            # ---- validation ----- #
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_running_losses = {t_name: 0.0 for t_name in self.mgr.tasks}
                    val_steps = 0

                    pbar = tqdm(enumerate(val_dataloader), total=self.mgr.max_val_steps_per_epoch)
                    for i, data_dict in pbar:
                        if i >= self.mgr.max_val_steps_per_epoch:
                            break

                        inputs = data_dict["image"].to(device, dtype=torch.float32)
                        targets_dict = {
                            k: v.to(device, dtype=torch.float32)
                            for k, v in data_dict.items()
                            if k != "image"
                        }

                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs)
                            total_val_loss = 0.0
                            for t_name, t_gt in targets_dict.items():
                                t_pred = outputs[t_name]
                                t_loss_fn = loss_fns[t_name]
                                t_loss = t_loss_fn(t_pred, t_gt)

                                total_val_loss += t_loss
                                val_running_losses[t_name] += t_loss.item()

                            val_steps += 1

                            if i == 0:
                                b_idx = 0  # pick which sample in the batch to visualize
                                # Slicing shape: [1, c, z, y, x ]
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
                                    tasks_dict=self.mgr.tasks, # your dictionary, e.g. {"sheet": {"activation":"sigmoid"}, "normals": {"activation":"none"}}
                                    epoch=epoch,
                                    save_path=f"{self.mgr.model_name}_debug.gif"
                                )

                    desc_parts = []
                    for t_name in self.mgr.tasks:
                        avg_loss_for_t = val_running_losses[t_name] / val_steps
                        desc_parts.append(f"{t_name} {avg_loss_for_t:.4f}")
                    desc_str = "Val: " + " | ".join(desc_parts)
                    pbar.set_description(desc_str)

                pbar.close()

                # Final avg for each task
                for t_name in self.mgr.tasks:
                    val_avg = val_running_losses[t_name] / val_steps
                    print(f"Task '{t_name}', epoch {epoch + 1} avg val loss: {val_avg:.4f}")

            scheduler.step()

        print('Training Finished!')
        torch.save(model.state_dict(), f'{self.mgr.model_name}_final.pth')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train script for MultiTaskUnet.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to your config file. Use the same one you used for training!")
    parser.add_argument("--debug_dataloader", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    trainer = BaseTrainer(args.config_path, args.debug_dataloader, args.verbose)
    trainer.train()





# During training, you'll get a dict with all outputs
# outputs = model(input_tensor)
# sheet_pred = outputs['sheet']          # Shape: [B, 1, D, H, W]
# normals_pred = outputs['normals']      # Shape: [B, 3, D, H, W]
# affinities_pred = outputs['affinities']  # Shape: [B, N_affinities, D, H, W]