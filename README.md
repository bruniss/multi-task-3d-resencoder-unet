this repository contains training and inference scripts for a dynamically created 3d u-net with residual encoders/decoders with squeeze and excitation blocks. 

the model can learn both segmentation and regression tasks simultaneously, and can accept an arbitrary number of inputs and target labels. currently the only supported data format is zarr. 

there is only 1 model type currently implemented, but adding additional support is trivial and something i will do soon. currently the model only supports a shared encoder path with separate decoder heads/paths. 

___
## purpose
the primary focus of creating this was to enable efficient multi-task learning with 3d unets. i am not a particularly skilled or experience programmer nor am i a machine learning expert by any stretch of the imagination, no guarantees for performance!

### main features
**self configuring encoder/decoder paths**: provided a config with inputs and channels, the model will automatically create the required paths and select some defaults for loss

**automatic patch finding**: provided with a reference labeled zarr array, the dataset will find patches that meet the requirements for both volumetric size within the patch and label density within it. this allows for passing gigantic but relatively sparsely labeled datasets and only performing training on the areas which are densely labeled. this list is then saved to a json so you will not have to run this computation again, provided your patch size has not changed.

**surface normal rotations/flips**: includes augmentations to apply proper rotation and flipping for surface normals along with other targets

**a clear dataset and train path**: all data is stored in dictionaries where the key is the name of your provided target. this makes extending or modifying relatively easy, and even someone without much python experience should be able to follow what is happening in each stage.

___
## configuration
the primary method for configuring the model is through a json file provided as an argument. an example json is in the tasks folder. the json has the following parameters (do not use the type hinting, its only there to let you know what to provide):


### tr_params

- **model_name**: `SheetNorm`  - [str] the name the checkpoint will be saved as 
- **patch_size**: `[64, 192, 192]` - [tuple] patch size, in (z, y, x)
- **batch_size**: `4`  [int]
- **tr_val_split**: `0.80` - [float] the split the dataloader will apply to your patches
- **ignore_label**: `null` - [bool] NOT YET IMPLEMENTED
- **loss_only_on_label**: `false` - [bool] only implemented for cosine loss currently
- **dilate_label**: `false` - [bool] will dilate the label by the amount set in the dataset.py
- **label_smoothing**: `0.1` - [float] soften the loss on your labels, as a percentage
- **initial_lr**: `0.001` [float]
- **weight_decay**: `0.0001` [float]
- **max_steps_per_epoch**: `500` - [int] the number of batches sent through the model each epoch
- **max_val_steps_per_epoch**: `25` - [int] the same as above but for validation
- **max_epoch**: `500` - [int] when to stop training
- **gradient_accumulation**: `1` - [int] accumulate gradients to simulate higher batch sizes. 1 is same as none.
- **checkpoint_path**: `/mnt/raid_hdd/models/normals/checkpoints/SheetNorm_325.pth` - [str] the path to a pretrained checkpoint you can resume training from. 
- **load_weights_only**: `true` - [bool] will only load the model weights , if specified you will begin training at epoch 0 with reset LR
- **ckpt_out_base**: `/mnt/raid_hdd/models/normals/checkpoints/augs_and_s4` - [str] base folder for checkpoint saving
- **optimizer**: `AdamW` - [str] can also be SGD
- **num_dataloader_workers**: `4` [int] - number of workers for the pytorch dataloader
- **tensorboard_log_dir**: `/home/sean/Desktop/tensorboard` - [str] destination folder for tensorboard logs  

---

### inference_params
these will only be applied on inference. during training none of these are used. 
- **patch_size**: `[64, 192, 192]` - [tuple] patch size for inference, (z,y,x)
- **batch_size**: `4` 
- **checkpoint_path**: `/mnt/raid_hdd/models/normals/checkpoints/SheetNorm_301.pth` - [str] - checkpoint to use for inference
- **num_dataloader_workers**: `4` 
- **input_path**: `/mnt/raid_nvme/s4.zarr` - [str] path for inference zarr
- **input_format**: `zarr` - only zarr currently supported
- **output_dir**: `/mnt/raid_nvme/inference_out` [str] path to output zarrs 
- **output_format**: `zarr` 
- **output_type**: `np.uint8` - [str] - dtype of output zar
- **output_targets**: `["sheet", "normals"]` - **currently not needed, inference outputs all targets**
- **overlap**: `0.25` - [float] the overlap between patches, used to avoid edge artifacts

---

### model_config

- **f_maps**: `[32, 64, 128, 256, 320, 528]` - [tuple] the number of feature maps at each level *must contain an amount equal to your num_levels*. if you num_levels is 6, put 6 numbers in here
- **num_levels**: `6` - [int] the number of levels for downsampling/upsample in the model. each scales up or down by 2, so if your patch size is small enough that it will end up being smaller than the number of levels that can be divided by two, you will get an error 


---

### dataset_config

- **min_bbox_percent**: `0.95` - [float] - the amount of the patch that a bbox containing _all the labels_ must encompass, as a percentage
- **min_labeled_ratio**: `0.07` - [float] - the amount of pixels within the above bbox that must contain nonzero values
- **use_cache**: `true` - [bool] after computing patches (which can take a while), will save a json with the starting positions of all valid patches, so you don't have to find them again
- **cache_file**: `/home/sean/Documents/GitHub/VC-Surface-Models/models/64_192_192_patch_cache_ands4.json` - [str] a path to precomputed valid patches
- **normalization**: `ct_normalization` - *not yet implemented*, currently zscore standardization is applied to image only  

### volume_paths
You can provide an arbitrary number of inputs and targets for each input. the model will automatically create decoder paths for the provided target label paths. if you have a target label here, you must have a target defined below it.
  - **input**: `/mnt/raid_nvme/s1.zarr` - [str] this is the path to the _raw input data_ for the associated targets
  - **sheet**: `/mnt/raid_nvme/datasets/1-voxel-sheet_slices-closed.zarr/0.zarr` - [str] this is a path to a set of labels for the above raw input data
  - **normals**: `/home/sean/Documents/GitHub/VC-Surface-Models/models/normals.zarr` - [str] this is a path to another set of labels for the input above
  - **ref_label**: `sheet` - [str] this is the label we will use to find valid patches. in this setup, we will use the "sheet" volume referenced above

### targets
you must provide the number of channels for each target. weight and activation are optional. if none are provided each target will be weighed equally and have no activation applied to it
- **sheet**:
  - **channels**: `1` - [int] - number of input _and_ output channels of the target
  - **activation**: `none`[str] - activation type of the final layer - options are none, sigmoid, and softmax
  - **weight**: `1` - [int] the weight to be applied during loss calculation. this will be multiplied by the loss, so 1 is equal to no change. some experimentation here is needed, as different losses output different scales of loss, so you will need to adjust this for your task/loss  

- **normals**: 
  - **channels**: `3` - note how this has 3 channels, this means that the normals zarr above _must also have 3 channels_ , but the raw input volume does not 
  - **activation**: `none`
  - **weight**: `0.75` - the loss for this will be multiplied by .75, effectively reducing its loss contribution by 1/4. the cosine loss i use outputs numbers higher than the loss i use for the sheet, so i reduce this ones contribution to prevent this loss from dominating

